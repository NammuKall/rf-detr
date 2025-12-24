# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import os
from logging import getLogger

import torch

from rfdetr.util.checkpoint import download_file, download_resume_checkpoint, validate_checkpoint
from rfdetr.util.utils import clean_state_dict

logger = getLogger(__name__)

HOSTED_MODELS = {
    "rf-detr-base.pth": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
    "rf-detr-base-o365.pth": "https://storage.googleapis.com/rfdetr/top-secret-1234/lwdetr_dinov2_small_o365_checkpoint.pth",
    # below is a less converged model that may be better for finetuning but worse for inference
    "rf-detr-base-2.pth": "https://storage.googleapis.com/rfdetr/rf-detr-base-2.pth",
    "rf-detr-large.pth": "https://storage.googleapis.com/rfdetr/rf-detr-large.pth",
    "rf-detr-nano.pth": "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
    "rf-detr-small.pth": "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
    "rf-detr-medium.pth": "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
    "rf-detr-seg-preview.pt": "https://storage.googleapis.com/rfdetr/rf-detr-seg-preview.pt",
}


def download_pretrain_weights(pretrain_weights: str, redownload=False, validate=True) -> bool:
    """
    Download pretrained weights if needed and validate the checkpoint.

    Args:
        pretrain_weights: Path to checkpoint file (can be filename or full path)
        redownload: Force re-download even if file exists
        validate: Validate checkpoint structure after download

    Returns:
        True if checkpoint exists and is valid, False otherwise
    """
    if pretrain_weights is None:
        return False

    # Resolve path to absolute path for consistent handling
    if os.path.isabs(pretrain_weights):
        checkpoint_path = pretrain_weights
    else:
        # Relative path - resolve relative to current working directory
        checkpoint_path = os.path.abspath(pretrain_weights)

    # Check if checkpoint is in HOSTED_MODELS (downloadable)
    is_hosted = pretrain_weights in HOSTED_MODELS or os.path.basename(pretrain_weights) in HOSTED_MODELS

    # If not hosted and file doesn't exist, can't download
    if not is_hosted:
        if os.path.exists(checkpoint_path):
            # File exists locally, validate if requested
            if validate:
                is_valid, error_msg = validate_checkpoint(checkpoint_path, required_keys=['model'])
                if not is_valid:
                    logger.error(f"Checkpoint validation failed: {error_msg}")
                    return False
            return True
        else:
            logger.error(
                f"Checkpoint not found and not in HOSTED_MODELS: {pretrain_weights}\n"
                f"Available hosted models: {list(HOSTED_MODELS.keys())}"
            )
            return False

    # Determine the model name for HOSTED_MODELS lookup
    model_name = pretrain_weights if pretrain_weights in HOSTED_MODELS else os.path.basename(pretrain_weights)

    # Check if file already exists and is valid
    if os.path.exists(checkpoint_path) and not redownload:
        if validate:
            is_valid, error_msg = validate_checkpoint(checkpoint_path, required_keys=['model'])
            if is_valid:
                logger.info(f"Checkpoint already exists and is valid: {checkpoint_path}")
                return True
            else:
                logger.warning(
                    f"Existing checkpoint failed validation: {error_msg}\n"
                    f"Will re-download..."
                )
                # Remove corrupted file
                try:
                    os.remove(checkpoint_path)
                except Exception as e:
                    logger.warning(f"Failed to remove corrupted checkpoint: {e}")
        else:
            logger.info(f"Checkpoint already exists: {checkpoint_path}")
            return True

    # Download the checkpoint
    url = HOSTED_MODELS[model_name]
    logger.info(f"Downloading pretrained weights: {model_name} from {url}")

    # Ensure directory exists
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create checkpoint directory {checkpoint_dir}: {e}")
            return False

    # Download file
    download_success = download_file(url, checkpoint_path)

    if not download_success:
        logger.error(f"Failed to download checkpoint: {pretrain_weights}")
        return False

    # Validate downloaded checkpoint if requested
    if validate:
        is_valid, error_msg = validate_checkpoint(checkpoint_path, required_keys=['model'])
        if not is_valid:
            logger.error(
                f"Downloaded checkpoint failed validation: {error_msg}\n"
                f"This may indicate a corrupted download or server issue."
            )
            # Remove corrupted file
            try:
                os.remove(checkpoint_path)
            except Exception as e:
                logger.warning(f"Failed to remove corrupted checkpoint: {e}")
            return False
        else:
            logger.info(f"Successfully downloaded and validated checkpoint: {checkpoint_path}")

    return True


def load_pretrain_checkpoint(checkpoint_path: str, model, args, logger):
    """
    Load pretrained checkpoint into model.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load weights into
        args: Arguments namespace
        logger: Logger instance

    Returns:
        checkpoint dict if loaded successfully
    """
    logger.info(f"Loading pretrain weights from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        # This should rarely happen since we validated, but handle gracefully
        logger.error(
            f"Failed to load checkpoint despite validation: {e}\n"
            f"The checkpoint file may have become corrupted after validation.\n"
            f"Attempting to re-download..."
        )
        # Try one more re-download
        checkpoint_available = download_pretrain_weights(
            checkpoint_path,
            redownload=True,
            validate=True
        )
        if not checkpoint_available:
            raise RuntimeError(
                f"Failed to load checkpoint after re-download: {checkpoint_path}\n"
                f"Original error: {e}\n"
                f"Please check the checkpoint file manually or contact support."
            ) from e
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract class_names from checkpoint if available
    if 'args' in checkpoint and hasattr(checkpoint['args'], 'class_names'):
        args.class_names = checkpoint['args'].class_names

    # Issue 4 Fix: Handle missing 'args' key in checkpoint
    # Checkpoints without 'args' cannot be validated, which could be dangerous
    if 'args' not in checkpoint:
        warning_msg = (
            f"Checkpoint '{checkpoint_path}' does not contain 'args' key. "
            f"Config validation cannot be performed. "
            f"This checkpoint may be from an older version or may be incompatible. "
            f"Consider re-saving the checkpoint with current version."
        )

        if getattr(args, 'strict_checkpoint_validation', True):
            # Fail on missing args when strict validation is enabled (default)
            raise ValueError(
                warning_msg + "\n"
                "Set strict_checkpoint_validation=False to load anyway, "
                "but this may cause errors if configs are incompatible."
            )
        else:
            # Warn but continue when strict validation is disabled
            logger.warning(warning_msg)

    # Validate checkpoint config compatibility with current config
    # This accounts for transformations (e.g., num_classes increment) automatically
    # Uses actual model num_classes from state_dict (source of truth) rather than args
    # This fixes the root issue where checkpoint args.num_classes may not match the actual model
    if 'args' in checkpoint:
        from rfdetr.util.config import compare_configs
        try:
            # Get current model state_dict for accurate comparison (model is already built)
            current_model_state_dict = model.state_dict() if model is not None else None

            is_compatible, differences, warnings = compare_configs(
                checkpoint['args'],
                args,
                checkpoint_model_state_dict=checkpoint['model'],
                current_model_state_dict=current_model_state_dict,
                critical_only=True
            )

            # Log differences and warnings
            if differences:
                logger.warning(
                    "Config differences detected between checkpoint and current config:\n"
                    + "\n".join(f"  - {param}: checkpoint={vals['checkpoint']}, current={vals['current']}"
                              for param, vals in differences.items())
                )
            if warnings:
                for warning in warnings:
                    if 'CRITICAL' in warning:
                        logger.warning(warning)
                    else:
                        logger.info(warning)

            # Fail on critical mismatches if strict_checkpoint_validation is True
            if not is_compatible and getattr(args, 'strict_checkpoint_validation', True):
                critical_differences = {
                    param: vals for param, vals in differences.items()
                    if param in ['encoder', 'hidden_dim', 'sa_nheads', 'ca_nheads', 'dec_layers',
                                'dec_n_points', 'num_queries', 'group_detr', 'projector_scale',
                                'out_feature_indexes', 'num_classes_transformed']
                }
                error_msg = (
                    "CRITICAL: Checkpoint config is incompatible with current config.\n"
                    "Loading this checkpoint with mismatched architecture parameters will cause errors.\n\n"
                    "Critical mismatches:\n"
                    + "\n".join(f"  - {param}: checkpoint={vals['checkpoint']}, current={vals['current']}"
                              for param, vals in critical_differences.items())
                    + "\n\nTo proceed anyway, set strict_checkpoint_validation=False when creating the Model.\n"
                    "However, this may cause runtime errors or incorrect model behavior."
                )
                raise ValueError(error_msg)
        except ValueError:
            # Re-raise ValueError (our config mismatch error)
            raise
        except Exception as e:
            # Don't fail loading if validation fails unexpectedly - just log
            logger.warning(f"Config comparison failed (non-fatal): {e}")

    checkpoint_num_classes = checkpoint['model']['class_embed.bias'].shape[0]
    if checkpoint_num_classes != args.num_classes + 1:
        model.reinitialize_detection_head(checkpoint_num_classes)
    # add support to exclude_keys
    # e.g., when load object365 pretrain, do not load `class_embed.[weight, bias]`
    if args.pretrain_exclude_keys is not None:
        assert isinstance(args.pretrain_exclude_keys, list)
        for exclude_key in args.pretrain_exclude_keys:
            checkpoint['model'].pop(exclude_key)
    if args.pretrain_keys_modify_to_load is not None:
        from rfdetr.util.obj365_to_coco_model import get_coco_pretrain_from_obj365
        assert isinstance(args.pretrain_keys_modify_to_load, list)
        for modify_key_to_load in args.pretrain_keys_modify_to_load:
            try:
                checkpoint['model'][modify_key_to_load] = get_coco_pretrain_from_obj365(
                    model.state_dict()[modify_key_to_load],
                    checkpoint['model'][modify_key_to_load]
                )
            except Exception as e:
                logger.warning(f"Failed to load {modify_key_to_load}, deleting from checkpoint: {e}")
                checkpoint['model'].pop(modify_key_to_load)

    # we may want to resume training with a smaller number of groups for group detr
    num_desired_queries = args.num_queries * args.group_detr
    query_param_names = ["refpoint_embed.weight", "query_feat.weight"]
    for name, state in checkpoint['model'].items():
        if any(name.endswith(x) for x in query_param_names):
            checkpoint['model'][name] = state[:num_desired_queries]

    model.load_state_dict(checkpoint['model'], strict=False)
    return checkpoint


def load_resume_checkpoint(resume_path: str, model_without_ddp, ema_m, optimizer, lr_scheduler, args, logger):
    """
    Load resume checkpoint for continuing training.

    Args:
        resume_path: Path to resume checkpoint
        model_without_ddp: Model without DDP wrapper
        ema_m: EMA model (if used)
        optimizer: Optimizer instance
        lr_scheduler: Learning rate scheduler
        args: Arguments namespace
        logger: Logger instance
    """
    logger.info(f"Resuming training from checkpoint: {resume_path}")

    # Step 1: Ensure checkpoint exists (download if URL, validate if local)
    try:
        resume_checkpoint_path = download_resume_checkpoint(resume_path, validate=True)
    except (FileNotFoundError, RuntimeError) as e:
        raise RuntimeError(
            f"Failed to prepare resume checkpoint: {e}\n"
            f"Please check:\n"
            f"  - Checkpoint path/URL is correct\n"
            f"  - Network connectivity (if using URL)\n"
            f"  - File permissions (if using local path)"
        ) from e

    # Step 2: Load checkpoint (now guaranteed to exist and be valid)
    try:
        checkpoint = torch.load(resume_checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load resume checkpoint despite validation: {e}\n"
            f"Checkpoint path: {resume_checkpoint_path}\n"
            f"The checkpoint file may have become corrupted after validation.\n"
            f"If using a URL, try re-downloading by removing the cached file."
        ) from e

    # Step 3: Load model state
    logger.info("Loading model state from checkpoint...")
    model_without_ddp.load_state_dict(checkpoint['model'], strict=True)

    # Step 4: Load EMA model if applicable
    if args.use_ema:
        if 'ema_model' in checkpoint:
            logger.info("Loading EMA model state from checkpoint...")
            ema_m.module.load_state_dict(clean_state_dict(checkpoint['ema_model']))
        else:
            logger.warning("EMA model not found in checkpoint, reinitializing EMA...")
            del ema_m
            from rfdetr.util.utils import ModelEma
            ema_m = ModelEma(model_without_ddp, decay=args.ema_decay, tau=args.ema_tau)

    # Step 5: Load optimizer and scheduler state if available
    if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        logger.info("Loading optimizer and scheduler state from checkpoint...")
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming from epoch {args.start_epoch}")
    else:
        if not args.eval:
            logger.warning(
                "Checkpoint missing optimizer/scheduler/epoch information. "
                "Starting from epoch 0."
            )

    return ema_m

