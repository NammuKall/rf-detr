# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from typing import Union, List
from copy import deepcopy
from logging import getLogger

import numpy as np
import supervision as sv
import torch
import torchvision.transforms.functional as F
from PIL import Image

logger = getLogger(__name__)


def optimize_for_inference(self, compile=True, batch_size=1, dtype=torch.float32):
    """Optimize model for inference by creating a separate inference model."""
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
                batch_size, 3, self.model.resolution, self.model.resolution, 
                device=self.model.device,
                dtype=dtype
            )
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
    images: Union[str, Image.Image, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, Image.Image, torch.Tensor]]],
    threshold: float = 0.5,
    **kwargs,
) -> Union[sv.Detections, List[sv.Detections]]:
    """Performs object detection on the input images and returns bounding box
    predictions.

    This method accepts a single image or a list of images in various formats
    (file path, PIL Image, NumPy array, or torch.Tensor). The images should be in
    RGB channel order. If a torch.Tensor is provided, it must already be normalized
    to values in the [0, 1] range and have the shape (C, H, W).

    Args:
        images (Union[str, Image.Image, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, Image.Image, torch.Tensor]]]):
            A single image or a list of images to process. Images can be provided
            as file paths, PIL Images, NumPy arrays, or torch.Tensors.
        threshold (float, optional):
            The minimum confidence score needed to consider a detected bounding box valid.
        **kwargs:
            Additional keyword arguments.

    Returns:
        Union[sv.Detections, List[sv.Detections]]: A single or multiple Detections
            objects, each containing bounding box coordinates, confidence scores,
            and class IDs.
    """
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
                "Image has pixel values above 1. Please ensure the image is "
                "normalized (scaled to [0, 1])."
            )
        if img.shape[0] != 3:
            raise ValueError(
                f"Invalid image shape. Expected 3 channels (RGB), but got "
                f"{img.shape[0]} channels."
            )
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
            # this could happen if someone manually changes self.model.resolution after optimizing the model
            raise ValueError(f"Resolution mismatch. "
                             f"Model was optimized for resolution {self._optimized_resolution}, "
                             f"but got {batch_tensor.shape[2]}. "
                             "You can explicitly remove the optimized model by calling model.remove_optimized_model().")
        if self._optimized_has_been_compiled:
            if self._optimized_batch_size != batch_tensor.shape[0]:
                raise ValueError(f"Batch size mismatch. "
                                 f"Optimized model was compiled for batch size {self._optimized_batch_size}, "
                                 f"but got {batch_tensor.shape[0]}. "
                                 "You can explicitly remove the optimized model by calling model.remove_optimized_model(). "
                                 "Alternatively, you can recompile the optimized model for a different batch size "
                                 "by calling model.optimize_for_inference(batch_size=<new_batch_size>).")

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

