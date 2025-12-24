# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
cleaned main file
"""
import argparse
import copy
import datetime
import json
import os
import random
import shutil
import time
from collections import defaultdict
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, DistributedSampler

import rfdetr.util.misc as utils
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.engine import evaluate, train_one_epoch
from rfdetr.models import PostProcess, build_criterion_and_postprocessors, build_model
from rfdetr.training import (
    create_lr_scheduler,
    download_pretrain_weights,
    get_args_parser,
    load_pretrain_checkpoint,
    load_resume_checkpoint,
    populate_args,
)
from rfdetr.util.benchmark import benchmark
from rfdetr.util.drop_scheduler import drop_scheduler
from rfdetr.util.get_param_dicts import get_param_dict
from rfdetr.util.utils import BestMetricHolder, ModelEma

if str(os.environ.get("USE_FILE_SYSTEM_SHARING", "False")).lower() in ["true", "1"]:
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

logger = getLogger(__name__)


def main(**kwargs):
    """
    Main training entry point.

    Creates a Model instance and trains it with the provided configuration.
    """
    model = Model(**kwargs)
    callbacks = {}
    model.train(callbacks=callbacks, **kwargs)


def distill(**kwargs):
    """
    Distillation training entry point.

    Note: Distillation functionality may need to be implemented separately.
    For now, this falls back to regular training.
    """
    logger.warning("Distillation mode not yet fully implemented, falling back to regular training")
    main(**kwargs)


class Model:
    def __init__(self, **kwargs):
        args = populate_args(**kwargs)
        self.args = args
        self.resolution = args.resolution
        self.model = build_model(args)
        self.device = torch.device(args.device)
        if args.pretrain_weights is not None:
            logger.info(f"Preparing to load pretrain weights: {args.pretrain_weights}")

            # Step 1: Ensure checkpoint exists and is valid (download if needed)
            checkpoint_available = download_pretrain_weights(
                args.pretrain_weights,
                redownload=False,
                validate=True
            )

            if not checkpoint_available:
                # Check if it's a custom path (not in HOSTED_MODELS)
                from rfdetr.training.checkpoint import HOSTED_MODELS
                checkpoint_name = os.path.basename(args.pretrain_weights)
                if checkpoint_name not in HOSTED_MODELS and args.pretrain_weights not in HOSTED_MODELS:
                    raise FileNotFoundError(
                        f"Checkpoint not found: {args.pretrain_weights}\n"
                        f"This checkpoint is not available for automatic download.\n"
                        f"Please ensure the file exists at the specified path.\n"
                        f"Available hosted models: {list(HOSTED_MODELS.keys())}"
                    )
                else:
                    # Try re-downloading once more
                    logger.warning(
                        f"Initial checkpoint validation failed for {args.pretrain_weights}. "
                        f"Attempting to re-download..."
                    )
                    checkpoint_available = download_pretrain_weights(
                        args.pretrain_weights,
                        redownload=True,
                        validate=True
                    )
                    if not checkpoint_available:
                        raise RuntimeError(
                            f"Failed to download or validate checkpoint: {args.pretrain_weights}\n"
                            f"This may indicate:\n"
                            f"  - Network connectivity issues\n"
                            f"  - Corrupted download\n"
                            f"  - Server-side issues\n"
                            f"Please check your internet connection and try again, or manually download the checkpoint."
                        )

            # Step 2: Load checkpoint using the training module function
            checkpoint = load_pretrain_checkpoint(args.pretrain_weights, self.model, args, logger)

            # Extract class_names from checkpoint if available
            if 'args' in checkpoint and hasattr(checkpoint['args'], 'class_names'):
                self.args.class_names = checkpoint['args'].class_names
                self.class_names = checkpoint['args'].class_names

        if args.backbone_lora:
            print("Applying LORA to backbone")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                use_dora=True,
                target_modules=[
                    "q_proj", "v_proj", "k_proj",  # covers OWL-ViT
                    "qkv", # covers open_clip ie Siglip2
                    "query", "key", "value", "cls_token", "register_tokens", # covers Dinov2 with windowed attn
                ]
            )
            self.model.backbone[0].encoder = get_peft_model(self.model.backbone[0].encoder, lora_config)
        self.model = self.model.to(self.device)
        self.postprocess = PostProcess(num_select=args.num_select)
        self.stop_early = False

    def reinitialize_detection_head(self, num_classes):
        self.model.reinitialize_detection_head(num_classes)

    def request_early_stop(self):
        self.stop_early = True
        print("Early stopping requested, will complete current epoch and stop")

    def train(self, callbacks: defaultdict[str, list[Callable]], **kwargs):
        currently_supported_callbacks = ["on_fit_epoch_end", "on_train_batch_start", "on_train_end"]
        for key in callbacks.keys():
            if key not in currently_supported_callbacks:
                raise ValueError(
                    f"Callback {key} is not currently supported, please file an issue if you need it!\n"
                    f"Currently supported callbacks: {currently_supported_callbacks}"
                )
        args = populate_args(**kwargs)
        if args.class_names is not None:
            self.args.class_names = args.class_names
            self.args.num_classes = args.num_classes

        utils.init_distributed_mode(args)
        print(f"git:\n  {utils.get_sha()}\n")
        print(args)
        device = torch.device(args.device)

        # fix the seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        criterion, postprocess = build_criterion_and_postprocessors(args)
        model = self.model
        model.to(device)

        model_without_ddp = model
        if args.distributed:
            if args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        param_dicts = get_param_dict(args, model_without_ddp)

        param_dicts = [p for p in param_dicts if p['params'].requires_grad]

        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                    weight_decay=args.weight_decay)
        # Choose the learning rate scheduler based on the new argument

        dataset_train = build_dataset(image_set='train', args=args, resolution=args.resolution)
        dataset_val = build_dataset(image_set='val', args=args, resolution=args.resolution)
        dataset_test = build_dataset(image_set='test' if args.dataset_file == "roboflow" else "val", args=args, resolution=args.resolution)

        # Create learning rate scheduler
        lr_scheduler = create_lr_scheduler(optimizer, args, dataset_train)

        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
            sampler_test = DistributedSampler(dataset_test, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        effective_batch_size = args.batch_size * args.grad_accum_steps
        min_batches = kwargs.get('min_batches', 5)
        if len(dataset_train) < effective_batch_size * min_batches:
            logger.info(
                f"Training with uniform sampler because dataset is too small: {len(dataset_train)} < {effective_batch_size * min_batches}"
            )
            sampler = torch.utils.data.RandomSampler(
                dataset_train,
                replacement=True,
                num_samples=effective_batch_size * min_batches,
            )
            data_loader_train = DataLoader(
                dataset_train,
                batch_size=effective_batch_size,
                collate_fn=utils.collate_fn,
                num_workers=args.num_workers,
                sampler=sampler,
            )
        else:
            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, effective_batch_size, drop_last=True)
            data_loader_train = DataLoader(
                dataset_train,
                batch_sampler=batch_sampler_train,
                collate_fn=utils.collate_fn,
                num_workers=args.num_workers
            )

        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn,
                                    num_workers=args.num_workers)
        data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                    drop_last=False, collate_fn=utils.collate_fn,
                                    num_workers=args.num_workers)

        base_ds = get_coco_api_from_dataset(dataset_val)
        base_ds_test = get_coco_api_from_dataset(dataset_test)
        if args.use_ema:
            self.ema_m = ModelEma(model_without_ddp, decay=args.ema_decay, tau=args.ema_tau)
        else:
            self.ema_m = None


        output_dir = Path(args.output_dir)

        if  utils.is_main_process():
            print("Get benchmark")
            if args.do_benchmark:
                benchmark_model = copy.deepcopy(model_without_ddp)
                bm = benchmark(benchmark_model.float(), dataset_val, output_dir)
                print(json.dumps(bm, indent=2))
                del benchmark_model

        if args.resume:
            self.ema_m = load_resume_checkpoint(
                args.resume, model_without_ddp, self.ema_m, optimizer, lr_scheduler, args, logger
            )

        if args.eval:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocess, data_loader_val, base_ds, device, args)
            if args.output_dir:
                if not args.segmentation_head:
                    utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
                else:
                    utils.save_on_master(coco_evaluator.coco_eval["segm"].eval, output_dir / "eval.pth")
            return

        # for drop
        total_batch_size = effective_batch_size * utils.get_world_size()
        num_training_steps_per_epoch = (len(dataset_train) + total_batch_size - 1) // total_batch_size
        schedules = {}
        if args.dropout > 0:
            schedules['do'] = drop_scheduler(
                args.dropout, args.epochs, num_training_steps_per_epoch,
                args.cutoff_epoch, args.drop_mode, args.drop_schedule)
            print("Min DO = {:.7f}, Max DO = {:.7f}".format(min(schedules['do']), max(schedules['do'])))

        if args.drop_path > 0:
            schedules['dp'] = drop_scheduler(
                args.drop_path, args.epochs, num_training_steps_per_epoch,
                args.cutoff_epoch, args.drop_mode, args.drop_schedule)
            print("Min DP = {:.7f}, Max DP = {:.7f}".format(min(schedules['dp']), max(schedules['dp'])))
        print("Start training")
        start_time = time.time()
        best_map_holder = BestMetricHolder(use_ema=args.use_ema)
        best_map_5095 = 0
        best_map_50 = 0
        best_map_ema_5095 = 0
        best_map_ema_50 = 0
        for epoch in range(args.start_epoch, args.epochs):
            epoch_start_time = time.time()
            if args.distributed:
                sampler_train.set_epoch(epoch)

            model.train()
            criterion.train()
            train_stats = train_one_epoch(
                model, criterion, lr_scheduler, data_loader_train, optimizer, device, epoch,
                effective_batch_size, args.clip_max_norm, ema_m=self.ema_m, schedules=schedules,
                num_training_steps_per_epoch=num_training_steps_per_epoch,
                vit_encoder_num_layers=args.vit_encoder_num_layers, args=args, callbacks=callbacks)
            train_epoch_time = time.time() - epoch_start_time
            train_epoch_time_str = str(datetime.timedelta(seconds=int(train_epoch_time)))
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every `checkpoint_interval` epochs
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.checkpoint_interval == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    weights = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }
                    if args.use_ema:
                        weights.update({
                            'ema_model': self.ema_m.module.state_dict(),
                        })
                    if not args.dont_save_weights:
                        # create checkpoint dir
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                        utils.save_on_master(weights, checkpoint_path)

            with torch.inference_mode():
                test_stats, coco_evaluator = evaluate(
                    model, criterion, postprocess, data_loader_val, base_ds, device, args=args
                )
            if not args.segmentation_head:
                map_regular = test_stats["coco_eval_bbox"][0]
            else:
                map_regular = test_stats["coco_eval_masks"][0]
            _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
            if _isbest:
                best_map_5095 = max(best_map_5095, map_regular)
                if not args.segmentation_head:
                    map50 = test_stats["coco_eval_bbox"][1]
                else:
                    map50 = test_stats["coco_eval_masks"][1]
                best_map_50 = max(best_map_50, map50)
                checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
                if not args.dont_save_weights:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
            # Extract F1 score from results_json if available
            if 'results_json' in test_stats and 'f1' in test_stats['results_json']:
                log_stats['test_f1'] = test_stats['results_json']['f1']
            if args.use_ema:
                ema_test_stats, _ = evaluate(
                    self.ema_m.module, criterion, postprocess, data_loader_val, base_ds, device, args=args
                )
                log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats.items()})
                # Extract F1 score from EMA results_json if available
                if 'results_json' in ema_test_stats and 'f1' in ema_test_stats['results_json']:
                    log_stats['ema_test_f1'] = ema_test_stats['results_json']['f1']
                if not args.segmentation_head:
                    map_ema = ema_test_stats["coco_eval_bbox"][0]
                else:
                    map_ema = ema_test_stats["coco_eval_masks"][0]
                best_map_ema_5095 = max(best_map_ema_5095, map_ema)
                _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
                if _isbest:
                    if not args.segmentation_head:
                        map_ema_50 = ema_test_stats["coco_eval_bbox"][1]
                    else:
                        map_ema_50 = ema_test_stats["coco_eval_masks"][1]
                    best_map_ema_50 = max(best_map_ema_50, map_ema_50)
                    checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                    if not args.dont_save_weights:
                        utils.save_on_master({
                            'model': self.ema_m.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                        }, checkpoint_path)
            log_stats.update(best_map_holder.summary())

            # epoch parameters
            ep_paras = {
                    'epoch': epoch,
                    'n_parameters': n_parameters
                }
            log_stats.update(ep_paras)
            try:
                log_stats.update({'now_time': str(datetime.datetime.now())})
            except Exception:
                pass
            log_stats['train_epoch_time'] = train_epoch_time_str
            epoch_time = time.time() - epoch_start_time
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            log_stats['epoch_time'] = epoch_time_str
            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            if not args.segmentation_head:
                                torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                        output_dir / "eval" / name)
                            else:
                                torch.save(coco_evaluator.coco_eval["segm"].eval,
                                    output_dir / "eval" / name)


            for callback in callbacks["on_fit_epoch_end"]:
                callback(log_stats)

            if self.stop_early:
                print(f"Early stopping requested, stopping at epoch {epoch}")
                break

        best_is_ema = best_map_ema_5095 > best_map_5095

        if utils.is_main_process():
            if best_is_ema:
                shutil.copy2(output_dir / 'checkpoint_best_ema.pth', output_dir / 'checkpoint_best_total.pth')
            else:
                shutil.copy2(output_dir / 'checkpoint_best_regular.pth', output_dir / 'checkpoint_best_total.pth')

            utils.strip_checkpoint(output_dir / 'checkpoint_best_total.pth')

            best_map_5095 = max(best_map_5095, best_map_ema_5095)
            if best_is_ema:
                results = ema_test_stats["results_json"]
            else:
                results = test_stats["results_json"]

            class_map = results["class_map"]
            results["class_map"] = {"valid": class_map}
            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f)

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print(f'Training time {total_time_str}')
            print('Results saved to {}'.format(output_dir / "results.json"))


        if best_is_ema:
            self.model = self.ema_m.module
        self.model.eval()

        if args.run_test:
            best_state_dict = torch.load(output_dir / 'checkpoint_best_total.pth', map_location='cpu', weights_only=False)['model']
            model.load_state_dict(best_state_dict)
            model.eval()

            test_stats, _ = evaluate(
                model, criterion, postprocess, data_loader_test, base_ds_test, device, args=args
            )
            print(f"Test results: {test_stats}")
            with open(output_dir / "results.json") as f:
                results = json.load(f)
            test_metrics = test_stats["results_json"]["class_map"]
            results["class_map"]["test"] = test_metrics
            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f)

        for callback in callbacks["on_train_end"]:
            callback()

    def export(self, output_dir="output", infer_dir=None, simplify=False,  backbone_only=False, opset_version=17, verbose=True, force=False, shape=None, batch_size=1, **kwargs):
        """Export the trained model to ONNX format"""
        print("Exporting model to ONNX format")
        try:
            from rfdetr.deploy.export import export_onnx, make_infer_image, onnx_simplify
        except ImportError:
            print("It seems some dependencies for ONNX export are missing. Please run `pip install rfdetr[onnxexport]` and try again.")
            raise


        device = self.device
        model = deepcopy(self.model.to("cpu"))
        model.to(device)

        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)
        if shape is None:
            shape = (self.resolution, self.resolution)
        else:
            if shape[0] % 14 != 0 or shape[1] % 14 != 0:
                raise ValueError("Shape must be divisible by 14")

        input_tensors = make_infer_image(infer_dir, shape, batch_size, device).to(device)
        input_names = ['input']
        output_names = ['features'] if backbone_only else ['dets', 'labels']
        dynamic_axes = None
        self.model.eval()
        with torch.no_grad():
            if backbone_only:
                features = model(input_tensors)
                print(f"PyTorch inference output shape: {features.shape}")
            elif self.args.segmentation_head:
                outputs = model(input_tensors)
                dets = outputs['pred_boxes']
                labels = outputs['pred_logits']
                masks = outputs['pred_masks']
                print(f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}, Masks: {masks.shape}")
            else:
                outputs = model(input_tensors)
                dets = outputs['pred_boxes']
                labels = outputs['pred_logits']
                print(f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}")
        model.cpu()
        input_tensors = input_tensors.cpu()

        # Export to ONNX
        output_file = export_onnx(
            output_dir=output_dir,
            model=model,
            input_names=input_names,
            input_tensors=input_tensors,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            backbone_only=backbone_only,
            verbose=verbose,
            opset_version=opset_version
        )

        print(f"Successfully exported ONNX model to: {output_file}")

        if simplify:
            sim_output_file = onnx_simplify(
                onnx_dir=output_file,
                input_names=input_names,
                input_tensors=input_tensors,
                force=force
            )
            print(f"Successfully simplified ONNX model to: {sim_output_file}")

        print("ONNX export completed successfully")
        self.model = self.model.to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LWDETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = vars(args)  # Convert Namespace to dictionary

    if args.subcommand == 'distill':
        distill(**config)
    elif args.subcommand is None:
        main(**config)
    elif args.subcommand == 'export_model':
        filter_keys = [
            "num_classes",
            "grad_accum_steps",
            "lr",
            "lr_encoder",
            "weight_decay",
            "epochs",
            "lr_drop",
            "clip_max_norm",
            "lr_vit_layer_decay",
            "lr_component_decay",
            "dropout",
            "drop_path",
            "drop_mode",
            "drop_schedule",
            "cutoff_epoch",
            "pretrained_encoder",
            "pretrain_weights",
            "pretrain_exclude_keys",
            "pretrain_keys_modify_to_load",
            "freeze_florence",
            "freeze_aimv2",
            "decoder_norm",
            "set_cost_class",
            "set_cost_bbox",
            "set_cost_giou",
            "cls_loss_coef",
            "bbox_loss_coef",
            "giou_loss_coef",
            "focal_alpha",
            "aux_loss",
            "sum_group_losses",
            "use_varifocal_loss",
            "use_position_supervised_loss",
            "ia_bce_loss",
            "dataset_file",
            "coco_path",
            "dataset_dir",
            "square_resize_div_64",
            "output_dir",
            "checkpoint_interval",
            "seed",
            "resume",
            "start_epoch",
            "eval",
            "use_ema",
            "ema_decay",
            "ema_tau",
            "num_workers",
            "device",
            "world_size",
            "dist_url",
            "sync_bn",
            "fp16_eval",
            "infer_dir",
            "verbose",
            "opset_version",
            "dry_run",
            "shape",
        ]
        for key in filter_keys:
            config.pop(key, None)  # Use pop with None to avoid KeyError

        from deploy.export import main as export_main
        if args.batch_size != 1:
            config['batch_size'] = 1
            print(f"Only batch_size 1 is supported for onnx export, \
                 but got batchsize = {args.batch_size}. batch_size is forcibly set to 1.")
        export_main(**config)
