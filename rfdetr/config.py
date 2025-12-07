# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


from pydantic import BaseModel, model_validator
from typing import List, Optional, Literal, Type
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class ModelConfig(BaseModel):
    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base"]
    out_feature_indexes: List[int]
    dec_layers: int
    two_stage: bool = True
    projector_scale: List[Literal["P3", "P4", "P5"]]
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
    num_encoder_layers: int = 2
    enc_n_points: int = 4
    # NEW: Cross-scale fusion
    use_cross_scale_fusion: bool = True


class RFDETRBaseConfig(ModelConfig):
    """
    The configuration for an RF-DETR Base model.
    
    NOTE: The improved architecture (use_improvements=True) requires retrained weights.
    Set use_improvements=False to use pretrained weights with original architecture.
    """
    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base"] = "dinov2_windowed_small"
    # NEW: Flag to enable/disable improvements (for compatibility with pretrained weights)
    use_improvements: bool = False  # Set to True after retraining with improvements
    
    # Original dimensions (compatible with pretrained weights)
    hidden_dim: int = 256
    patch_size: int = 14
    num_windows: int = 4
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    
    # Improved dimensions (used when use_improvements=True)
    # These will override the above when improvements are enabled
    improved_hidden_dim: int = 320
    improved_sa_nheads: int = 10
    improved_ca_nheads: int = 20
    improved_dec_n_points: int = 4
    
    # Encoder and cross-scale fusion (only used when use_improvements=True)
    enc_n_points: int = 4
    num_encoder_layers: int = 2
    use_cross_scale_fusion: bool = True
    
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P4"]
    out_feature_indexes: List[int] = [2, 5, 8, 11]
    pretrain_weights: Optional[str] = "rf-detr-base.pth"
    resolution: int = 560
    positional_encoding_size: int = 37
    
    @model_validator(mode='after')
    def apply_improvements(self):
        """Apply improvements if enabled"""
        if self.use_improvements:
            self.hidden_dim = self.improved_hidden_dim
            self.sa_nheads = self.improved_sa_nheads
            self.ca_nheads = self.improved_ca_nheads
            self.dec_n_points = self.improved_dec_n_points
        else:
            # Disable encoder layers and cross-scale fusion when using original architecture
            self.num_encoder_layers = 0
            self.use_cross_scale_fusion = False
        return self

class RFDETRLargeConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Large model.
    
    NOTE: The improved architecture (use_improvements=True) requires retrained weights.
    Set use_improvements=False to use pretrained weights with original architecture.
    """
    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base"] = "dinov2_windowed_base"
    use_improvements: bool = False
    
    # Original dimensions (compatible with pretrained weights)
    hidden_dim: int = 384
    sa_nheads: int = 12
    ca_nheads: int = 24
    dec_n_points: int = 4
    
    # Improved dimensions (used when use_improvements=True)
    improved_hidden_dim: int = 512
    improved_sa_nheads: int = 16
    improved_ca_nheads: int = 32
    improved_dec_n_points: int = 6
    
    # Encoder and cross-scale fusion (only used when use_improvements=True)
    enc_n_points: int = 6
    num_encoder_layers: int = 3
    use_cross_scale_fusion: bool = True
    
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P3", "P5"]
    pretrain_weights: Optional[str] = "rf-detr-large.pth"
    
    @model_validator(mode='after')
    def apply_improvements(self):
        """Apply improvements if enabled"""
        if self.use_improvements:
            self.hidden_dim = self.improved_hidden_dim
            self.sa_nheads = self.improved_sa_nheads
            self.ca_nheads = self.improved_ca_nheads
            self.dec_n_points = self.improved_dec_n_points
        else:
            # Disable encoder layers and cross-scale fusion when using original architecture
            self.num_encoder_layers = 0
            self.use_cross_scale_fusion = False
        return self

class RFDETRNanoConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Nano model.
    """
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 2
    patch_size: int = 16
    resolution: int = 384
    positional_encoding_size: int = 24
    pretrain_weights: Optional[str] = "rf-detr-nano.pth"

class RFDETRSmallConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Small model.
    """
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 3
    patch_size: int = 16
    resolution: int = 512
    positional_encoding_size: int = 32
    pretrain_weights: Optional[str] = "rf-detr-small.pth"

class RFDETRMediumConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Medium model.
    
    NOTE: The improved architecture (use_improvements=True) requires retrained weights.
    Set use_improvements=False to use pretrained weights with original architecture.
    """
    use_improvements: bool = False
    
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 16
    resolution: int = 576
    positional_encoding_size: int = 36
    pretrain_weights: Optional[str] = "rf-detr-medium.pth"
    
    # Original dimensions (inherited from base, but Medium uses base dimensions)
    # Improved dimensions (used when use_improvements=True)
    improved_hidden_dim: int = 384
    improved_sa_nheads: int = 12
    improved_ca_nheads: int = 24
    improved_dec_n_points: int = 4
    
    # Encoder and cross-scale fusion (only used when use_improvements=True)
    enc_n_points: int = 4
    num_encoder_layers: int = 2
    use_cross_scale_fusion: bool = True
    
    @model_validator(mode='after')
    def apply_improvements(self):
        """Apply improvements if enabled"""
        if self.use_improvements:
            self.hidden_dim = self.improved_hidden_dim
            self.sa_nheads = self.improved_sa_nheads
            self.ca_nheads = self.improved_ca_nheads
            self.dec_n_points = self.improved_dec_n_points
        else:
            # Disable encoder layers and cross-scale fusion when using original architecture
            self.num_encoder_layers = 0
            self.use_cross_scale_fusion = False
        return self

class RFDETRSegPreviewConfig(RFDETRBaseConfig):
    segmentation_head: bool = True
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 12
    resolution: int = 432
    positional_encoding_size: int = 36
    num_queries: int = 200
    num_select: int = 200
    pretrain_weights: Optional[str] = "rf-detr-seg-preview.pt"
    num_classes: int = 90

class TrainConfig(BaseModel):
    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    batch_size: int = 4
    grad_accum_steps: int = 4
    epochs: int = 100
    ema_decay: float = 0.993
    ema_tau: int = 100
    lr_drop: int = 100
    checkpoint_interval: int = 10
    warmup_epochs: float = 0.0
    lr_vit_layer_decay: float = 0.8
    lr_component_decay: float = 0.7
    drop_path: float = 0.0
    group_detr: int = 13
    ia_bce_loss: bool = True
    cls_loss_coef: float = 1.0
    num_select: int = 300
    dataset_file: Literal["coco", "o365", "roboflow"] = "roboflow"
    square_resize_div_64: bool = True
    dataset_dir: str
    output_dir: str = "output"
    multi_scale: bool = True
    expanded_scales: bool = True
    do_random_resize_via_padding: bool = False
    use_ema: bool = True
    num_workers: int = 2
    weight_decay: float = 1e-4
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_use_ema: bool = False
    tensorboard: bool = True
    wandb: bool = False
    project: Optional[str] = None
    run: Optional[str] = None
    class_names: List[str] = None
    run_test: bool = True
    segmentation_head: bool = False


class SegmentationTrainConfig(TrainConfig):
    mask_point_sample_ratio: int = 16
    mask_ce_loss_coef: float = 5.0
    mask_dice_loss_coef: float = 5.0
    cls_loss_coef: float = 5.0
    segmentation_head: bool = True
