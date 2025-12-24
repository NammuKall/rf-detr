# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from typing import Literal, Optional

import torch
from pydantic import BaseModel

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class ModelConfig(BaseModel):
    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base"]
    out_feature_indexes: list[int]
    dec_layers: int
    two_stage: bool = True
    projector_scale: list[Literal["P3", "P4", "P5"]]
    hidden_dim: int
    patch_size: int
    num_windows: int
    sa_nheads: int
    ca_nheads: int
    dec_n_points: int
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    layer_norm: bool = True
    amp: bool = True
    num_classes: int = 90
    pretrain_weights: Optional[str] = None
    device: Literal["cpu", "cuda", "mps"] = DEVICE
    resolution: int
    group_detr: int = 13
    gradient_checkpointing: bool = False
    positional_encoding_size: int
    ia_bce_loss: bool = True
    cls_loss_coef: float = 1.0
    segmentation_head: bool = False
    mask_downsample_ratio: int = 4
    # NEW: Encoder layers to refine backbone features
    # Default to 0 (disabled) for compatibility with pretrained weights
    # Validators will override to 2/3 when use_improvements=True
    num_encoder_layers: int = 0
    enc_n_points: int = 4
    # NEW: Cross-scale fusion
    # Default to False (disabled) for compatibility with pretrained weights
    # Validators will override to True when use_improvements=True
    use_cross_scale_fusion: bool = False
