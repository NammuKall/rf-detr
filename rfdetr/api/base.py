# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import json
import os
from collections import defaultdict
from logging import getLogger
from typing import Union

import numpy as np
import supervision as sv
import torch
import torchvision.transforms.functional as F
from PIL import Image

from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.main import Model
from rfdetr.training import download_pretrain_weights
from rfdetr.util.coco_classes import COCO_CLASSES
from rfdetr.util.metrics import MetricsPlotSink, MetricsTensorBoardSink, MetricsWandBSink

logger = getLogger(__name__)


def _validate_model_dump_result(result, config_name: str, config_type: str) -> dict:
    """
    Validate that model_dump() returns a dictionary and provide informative error messages.

    This function provides detailed error messages if model_dump() returns an unexpected type,
    which should never happen with Pydantic v2 but could indicate:
    - Pydantic version incompatibility
    - Config object corruption
    - Unexpected override of model_dump() method

    Args:
        result: The result from calling model_dump() on a Pydantic model
        config_name: A descriptive name for the config (e.g., "train_config", "model_config")
        config_type: The type/class name of the config object (e.g., "TrainConfig", "ModelConfig")

    Returns:
        dict: The validated dictionary result from model_dump()

    Raises:
        TypeError: If model_dump() returns a non-dict type, with detailed diagnostic information
    """
    if not isinstance(result, dict):
        # Get Pydantic version for diagnostics
        try:
            import pydantic

            pydantic_version = pydantic.__version__
        except (ImportError, AttributeError):
            pydantic_version = "unknown"

        # Log detailed diagnostic information before raising
        logger.error(
            f"CRITICAL: model_dump() returned unexpected type for {config_name}.\n"
            f"  Config type: {config_type}\n"
            f"  Expected type: dict\n"
            f"  Actual type: {type(result).__name__}\n"
            f"  Actual value: {repr(result)[:200]}...\n"
            f"  Pydantic version: {pydantic_version}\n"
            f"  This should never happen with Pydantic v2. Possible causes:\n"
            f"    1. Pydantic version incompatibility (expected v2.x)\n"
            f"    2. Config object corruption or unexpected state\n"
            f"    3. Custom model_dump() override returning wrong type\n"
            f"  Diagnostic steps:\n"
            f"    1. Verify Pydantic version: pip show pydantic\n"
            f"    2. Check config object: type({config_name})\n"
            f"    3. Inspect config object state: {config_name}.__dict__\n"
            f"    4. Try recreating config object from scratch"
        )

        raise TypeError(
            f"model_dump() returned unexpected type for {config_name} (type: {config_type}).\n"
            f"Expected dict, but got {type(result).__name__}.\n\n"
            f"This indicates a critical issue that should never occur with Pydantic v2:\n"
            f"  • Pydantic version: {pydantic_version}\n"
            f"  • Config type: {config_type}\n"
            f"  • Returned type: {type(result).__name__}\n"
            f"  • Returned value (first 200 chars): {repr(result)[:200]}\n\n"
            f"Possible causes and solutions:\n"
            f"  1. Pydantic version incompatibility:\n"
            f"     → Check version: pip show pydantic\n"
            f"     → Expected: Pydantic v2.x (model_dump() always returns dict)\n"
            f"     → Fix: pip install 'pydantic>=2.0'\n\n"
            f"  2. Config object corruption:\n"
            f"     → The config object may be in an invalid state\n"
            f"     → Try recreating the config object\n"
            f"     → Check if config was modified after creation\n\n"
            f"  3. Custom model_dump() override:\n"
            f"     → Check if {config_type} or its base classes override model_dump()\n"
            f"     → Ensure override returns dict\n\n"
            f"To verify everything is working properly after fixing:\n"
            f"  1. Create a fresh config object: config = {config_type}(...)\n"
            f"  2. Call model_dump(): result = config.model_dump()\n"
            f"  3. Verify type: assert isinstance(result, dict)\n"
            f"  4. Check result structure: print(result.keys())\n"
            f"  5. If issues persist, check Pydantic documentation for your version"
        )

    return result


class RFDETR:
    """
    The base RF-DETR class implements the core methods for training RF-DETR models,
    running inference on the models, optimising models, and uploading trained
    models for deployment.
    """

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    size = None

    def __init__(self, **kwargs):
        self.model_config = self.get_model_config(**kwargs)
        self.maybe_download_pretrain_weights()
        self.model = self.get_model(self.model_config)
        self.callbacks = defaultdict(list)

        self.model.inference_model = None
        self._is_optimized_for_inference = False
        self._has_warned_about_not_being_optimized_for_inference = False
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None
        self._optimized_resolution = None
        self._optimized_dtype = None

    def maybe_download_pretrain_weights(self):
        """
        Download pre-trained weights if they are not already downloaded.

        This method ensures the checkpoint exists and is valid before model initialization.
        Handles None cases gracefully and validates checkpoints before use.

        Behavior:
        - If pretrain_weights is None: Logs info and continues (model will train from scratch)
        - If pretrain_weights is provided: Downloads/validates checkpoint before model initialization
        - If download/validation fails: Logs warning but allows Model.__init__ to handle errors
        """
        if self.model_config.pretrain_weights is None:
            logger.info(
                "No pretrain_weights specified. Model will be initialized from scratch "
                "(backbone weights may still be loaded from pretrained encoder if specified)."
            )
            return

        # Validate that pretrain_weights is a string (not empty or wrong type)
        if not isinstance(self.model_config.pretrain_weights, str):
            logger.error(
                f"Invalid pretrain_weights type: {type(self.model_config.pretrain_weights).__name__}. "
                f"Expected str or None, got {type(self.model_config.pretrain_weights).__name__}."
            )
            raise TypeError(
                f"pretrain_weights must be a string or None, got {type(self.model_config.pretrain_weights).__name__}"
            )

        if not self.model_config.pretrain_weights.strip():
            logger.error("pretrain_weights is an empty string. Use None if no pretrained weights are needed.")
            raise ValueError(
                "pretrain_weights cannot be an empty string. Use None if no pretrained weights are needed."
            )

        pretrain_path = self.model_config.pretrain_weights.strip()
        logger.info(f"Preparing pretrain weights: {pretrain_path}")

        # Attempt to download and validate checkpoint
        try:
            success = download_pretrain_weights(pretrain_path, redownload=False, validate=True)

            if success:
                logger.info(
                    f"Pretrain weights ready: {pretrain_path}\n"
                    f"Checkpoint exists and is valid. Model initialization will proceed."
                )
            else:
                logger.warning(
                    f"Could not download or validate checkpoint: {pretrain_path}\n"
                    f"Model initialization will attempt to handle this. If initialization fails, "
                    f"please check:\n"
                    f"  - Network connectivity (if downloading from URL)\n"
                    f"  - Checkpoint path is correct\n"
                    f"  - File permissions (if using local path)\n"
                    f"  - Sufficient disk space\n"
                    f"  - Checkpoint is in HOSTED_MODELS or exists locally"
                )
        except Exception as e:
            logger.error(
                f"Unexpected error while preparing pretrain weights: {e}\n"
                f"Model initialization will continue, but may fail if checkpoint is required."
            )
            # Don't raise - let Model.__init__ handle the error
            # This allows for graceful degradation if checkpoint is optional

    def get_model_config(self, **kwargs):
        """
        Retrieve the configuration parameters used by the model.
        """
        return ModelConfig(**kwargs)

    def train(self, **kwargs):
        """
        Train an RF-DETR model.
        """
        config = self.get_train_config(**kwargs)
        self.train_from_config(config, **kwargs)

    def export(self, **kwargs):
        """
        Export your model to an ONNX file.

        See [the ONNX export documentation](https://rfdetr.roboflow.com/learn/train/#onnx-export) for more information.
        """
        self.model.export(**kwargs)

    def train_from_config(self, config: TrainConfig, **kwargs):
        if config.dataset_file == "roboflow":
            with open(os.path.join(config.dataset_dir, "train", "_annotations.coco.json")) as f:
                anns = json.load(f)
                num_classes = len(anns["categories"])
                class_names = [c["name"] for c in anns["categories"] if c["supercategory"] != "none"]
                self.model.class_names = class_names
        elif config.dataset_file == "simsurg":
            # SimSurg uses COCO format with annotations in a separate folder
            with open(os.path.join(config.dataset_dir, "annotations", "instances_train.json")) as f:
                anns = json.load(f)
                num_classes = len(anns["categories"])
                class_names = [c["name"] for c in anns["categories"] if c["supercategory"] != "none"]
                self.model.class_names = class_names
        elif config.dataset_file == "coco":
            class_names = COCO_CLASSES
            num_classes = 90
        else:
            raise ValueError(f"Invalid dataset file: {config.dataset_file}")

        if self.model_config.num_classes != num_classes:
            self.model.reinitialize_detection_head(num_classes)

        train_config = _validate_model_dump_result(
            config.model_dump(), config_name="train_config", config_type=config.__class__.__name__
        )

        model_config = _validate_model_dump_result(
            self.model_config.model_dump(), config_name="model_config", config_type=self.model_config.__class__.__name__
        )

        model_config.pop("num_classes")
        if "class_names" in model_config:
            model_config.pop("class_names")

        if "class_names" in train_config and train_config["class_names"] is None:
            train_config["class_names"] = class_names

        for k, _v in train_config.items():
            if k in model_config:
                model_config.pop(k)
            if k in kwargs:
                kwargs.pop(k)

        all_kwargs = {**model_config, **train_config, **kwargs, "num_classes": num_classes}

        metrics_plot_sink = MetricsPlotSink(output_dir=config.output_dir)
        self.callbacks["on_fit_epoch_end"].append(metrics_plot_sink.update)
        self.callbacks["on_train_end"].append(metrics_plot_sink.save)

        if config.tensorboard:
            metrics_tensor_board_sink = MetricsTensorBoardSink(output_dir=config.output_dir)
            self.callbacks["on_fit_epoch_end"].append(metrics_tensor_board_sink.update)
            self.callbacks["on_train_end"].append(metrics_tensor_board_sink.close)

        if config.wandb:
            wandb_config = _validate_model_dump_result(
                config.model_dump(), config_name="wandb_config", config_type=config.__class__.__name__
            )
            metrics_wandb_sink = MetricsWandBSink(
                output_dir=config.output_dir, project=config.project, run=config.run, config=wandb_config
            )
            self.callbacks["on_fit_epoch_end"].append(metrics_wandb_sink.update)
            self.callbacks["on_train_end"].append(metrics_wandb_sink.close)

        if config.early_stopping:
            from rfdetr.util.early_stopping import EarlyStoppingCallback

            early_stopping_callback = EarlyStoppingCallback(
                model=self.model,
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                use_ema=config.early_stopping_use_ema,
                segmentation_head=config.segmentation_head,
            )
            self.callbacks["on_fit_epoch_end"].append(early_stopping_callback.update)

        self.model.train(
            **all_kwargs,
            callbacks=self.callbacks,
        )

    def get_train_config(self, **kwargs):
        """
        Retrieve the configuration parameters that will be used for training.
        """
        return TrainConfig(**kwargs)

    def get_model(self, config: ModelConfig):
        """
        Retrieve a model instance based on the provided configuration.
        """
        config_dict = _validate_model_dump_result(
            config.model_dump(), config_name="config_dict", config_type=config.__class__.__name__
        )
        return Model(**config_dict)

    # Get class_names from the model
    @property
    def class_names(self):
        """
        Retrieve the class names supported by the loaded model.

        Returns:
            dict: A dictionary mapping class IDs to class names. The keys are integers starting from
        """
        if hasattr(self.model, "class_names") and self.model.class_names:
            return {i + 1: name for i, name in enumerate(self.model.class_names)}

        return COCO_CLASSES

    def optimize_for_inference(self, compile=True, batch_size=1, dtype=torch.float32):
        """Optimize model for inference by creating a separate inference model."""
        from copy import deepcopy

        self.remove_optimized_model()

        self.model.inference_model = deepcopy(self.model.model)
        self.model.inference_model.eval()
        self.model.inference_model.export()

        self._optimized_resolution = self.model.resolution
        self._is_optimized_for_inference = True

        self.model.inference_model = self.model.inference_model.to(dtype=dtype)
        self._optimized_dtype = dtype

        if compile:
            self.model.inference_model = torch.jit.trace(
                self.model.inference_model,
                torch.randn(
                    batch_size, 3, self.model.resolution, self.model.resolution, device=self.model.device, dtype=dtype
                ),
            )
            self._optimized_has_been_compiled = True
            self._optimized_batch_size = batch_size

    def remove_optimized_model(self):
        """Remove the optimized inference model."""
        self.model.inference_model = None
        self._is_optimized_for_inference = False
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None
        self._optimized_resolution = None
        self._optimized_half = False

    def predict(
        self,
        images: Union[
            str, Image.Image, np.ndarray, torch.Tensor, list[Union[str, np.ndarray, Image.Image, torch.Tensor]]
        ],
        threshold: float = 0.5,
        **kwargs,
    ) -> Union[sv.Detections, list[sv.Detections]]:
        """Performs object detection on the input images and returns bounding box predictions."""

        if not self._is_optimized_for_inference and not self._has_warned_about_not_being_optimized_for_inference:
            logger.warning(
                "Model is not optimized for inference. "
                "Latency may be higher than expected. "
                "You can optimize the model for inference by calling model.optimize_for_inference()."
            )
            self._has_warned_about_not_being_optimized_for_inference = True
            self.model.model.eval()

        if not isinstance(images, list):
            images = [images]

        orig_sizes = []
        processed_images = []

        for img in images:
            if isinstance(img, str):
                img = Image.open(img)

            if not isinstance(img, torch.Tensor):
                img = F.to_tensor(img)

            if (img > 1).any():
                raise ValueError(
                    "Image has pixel values above 1. Please ensure the image is normalized (scaled to [0, 1])."
                )
            if img.shape[0] != 3:
                raise ValueError(f"Invalid image shape. Expected 3 channels (RGB), but got {img.shape[0]} channels.")
            img_tensor = img

            h, w = img_tensor.shape[1:]
            orig_sizes.append((h, w))

            img_tensor = img_tensor.to(self.model.device)
            img_tensor = F.normalize(img_tensor, self.means, self.stds)
            img_tensor = F.resize(img_tensor, (self.model.resolution, self.model.resolution))

            processed_images.append(img_tensor)

        batch_tensor = torch.stack(processed_images)

        if self._is_optimized_for_inference:
            if self._optimized_resolution != batch_tensor.shape[2]:
                raise ValueError(
                    f"Resolution mismatch. "
                    f"Model was optimized for resolution {self._optimized_resolution}, "
                    f"but got {batch_tensor.shape[2]}. "
                    "You can explicitly remove the optimized model by calling model.remove_optimized_model()."
                )
            if self._optimized_has_been_compiled:
                if self._optimized_batch_size != batch_tensor.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch. "
                        f"Optimized model was compiled for batch size {self._optimized_batch_size}, "
                        f"but got {batch_tensor.shape[0]}. "
                        "You can explicitly remove the optimized model by calling model.remove_optimized_model(). "
                        "Alternatively, you can recompile the optimized model for a different batch size "
                        "by calling model.optimize_for_inference(batch_size=<new_batch_size>)."
                    )

        with torch.inference_mode():
            if self._is_optimized_for_inference:
                predictions = self.model.inference_model(batch_tensor.to(dtype=self._optimized_dtype))
            else:
                predictions = self.model.model(batch_tensor)
            if isinstance(predictions, tuple):
                predictions = {
                    "pred_logits": predictions[1],
                    "pred_boxes": predictions[0],
                }
                if len(predictions) == 3:
                    predictions["pred_masks"] = predictions[2]
            target_sizes = torch.tensor(orig_sizes, device=self.model.device)
            results = self.model.postprocess(predictions, target_sizes=target_sizes)

        detections_list = []
        for result in results:
            scores = result["scores"]
            labels = result["labels"]
            boxes = result["boxes"]

            keep = scores > threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            if "masks" in result:
                masks = result["masks"]
                masks = masks[keep]

                detections = sv.Detections(
                    xyxy=boxes.float().cpu().numpy(),
                    confidence=scores.float().cpu().numpy(),
                    class_id=labels.cpu().numpy(),
                    mask=masks.squeeze(1).cpu().numpy(),
                )
            else:
                detections = sv.Detections(
                    xyxy=boxes.float().cpu().numpy(),
                    confidence=scores.float().cpu().numpy(),
                    class_id=labels.cpu().numpy(),
                )

            detections_list.append(detections)

        return detections_list if len(detections_list) > 1 else detections_list[0]

    def deploy_to_roboflow(self, workspace: str, project_id: str, version: str, api_key: str = None, size: str = None):
        """Deploy the trained RF-DETR model to Roboflow."""
        import os
        import shutil

        from roboflow import Roboflow

        if api_key is None:
            api_key = os.getenv("ROBOFLOW_API_KEY")
            if api_key is None:
                raise ValueError("Set api_key=<KEY> in deploy_to_roboflow or export ROBOFLOW_API_KEY=<KEY>")

        rf = Roboflow(api_key=api_key)
        workspace = rf.workspace(workspace)

        if self.size is None and size is None:
            raise ValueError("Must set size for custom architectures")

        size = self.size or size
        tmp_out_dir = ".roboflow_temp_upload"
        os.makedirs(tmp_out_dir, exist_ok=True)
        outpath = os.path.join(tmp_out_dir, "weights.pt")
        torch.save({"model": self.model.model.state_dict(), "args": self.model.args}, outpath)
        project = workspace.project(project_id)
        version = project.version(version)
        version.deploy(model_type=size, model_path=tmp_out_dir, filename="weights.pt")
        shutil.rmtree(tmp_out_dir)
