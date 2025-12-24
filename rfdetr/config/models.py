# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from typing import Literal, Optional

from pydantic import model_validator

from rfdetr.config.base import ModelConfig
from rfdetr.config.validators import (
    get_original_dimensions,
    get_required_fields,
    validate_config_values,
    validate_required_fields_before,
)


class RFDETRBaseConfig(ModelConfig):
    """
    The configuration for an RF-DETR Base model.

    NOTE: The improved architecture (use_improvements=True) requires retrained weights.
    Set use_improvements=False to use pretrained weights with original architecture.
    """

    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base"] = "dinov2_windowed_small"
    # NEW: Flag to enable/disable improvements (for compatibility with pretrained weights)
    use_improvements: bool = False  # Set to True after retraining with improvements

    # Original dimensions (compatible with pretrained weights) - DEFAULT VALUES
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
    num_encoder_layers: int = 0  # Default to 0 (disabled) for compatibility
    use_cross_scale_fusion: bool = False  # Default to False for compatibility

    num_queries: int = 300
    num_select: int = 300
    projector_scale: list[Literal["P3", "P4", "P5"]] = ["P4"]
    out_feature_indexes: list[int] = [2, 5, 8, 11]
    pretrain_weights: Optional[str] = "rf-detr-base.pth"
    resolution: int = 560
    positional_encoding_size: int = 37

    @classmethod
    def _get_required_fields(cls):
        return get_required_fields(cls)

    @model_validator(mode="before")
    @classmethod
    def validate_required_fields_before(cls, data):
        return validate_required_fields_before(cls, data)

    @model_validator(mode="before")
    @classmethod
    def apply_improvements_before(cls, data):
        """Apply improvements before model creation"""
        # Ensure data is a dict (handle None, empty dict, or other types)
        if not isinstance(data, dict):
            data = {} if data is None else dict(data)

        use_improvements = data.get("use_improvements", False)

        if use_improvements:
            # Apply improved dimensions - use improved_* values if provided, otherwise use defaults
            improved_hidden_dim = data.pop("improved_hidden_dim", None)
            improved_sa_nheads = data.pop("improved_sa_nheads", None)
            improved_ca_nheads = data.pop("improved_ca_nheads", None)
            improved_dec_n_points = data.pop("improved_dec_n_points", None)

            # Get class defaults for improved values
            if improved_hidden_dim is not None:
                data["hidden_dim"] = improved_hidden_dim
            elif "hidden_dim" not in data:
                # Use class default if not set
                data["hidden_dim"] = getattr(cls, "improved_hidden_dim", 320)

            if improved_sa_nheads is not None:
                data["sa_nheads"] = improved_sa_nheads
            elif "sa_nheads" not in data:
                data["sa_nheads"] = getattr(cls, "improved_sa_nheads", 10)

            if improved_ca_nheads is not None:
                data["ca_nheads"] = improved_ca_nheads
            elif "ca_nheads" not in data:
                data["ca_nheads"] = getattr(cls, "improved_ca_nheads", 20)

            if improved_dec_n_points is not None:
                data["dec_n_points"] = improved_dec_n_points
            elif "dec_n_points" not in data:
                data["dec_n_points"] = getattr(cls, "improved_dec_n_points", 4)

            data.setdefault("num_encoder_layers", 2)
            data.setdefault("use_cross_scale_fusion", True)
        else:
            # When use_improvements=False, ensure improved values are removed
            # but don't override existing values (let class defaults handle it)
            data.pop("improved_hidden_dim", None)
            data.pop("improved_sa_nheads", None)
            data.pop("improved_ca_nheads", None)
            data.pop("improved_dec_n_points", None)
            # Only set these if not already set (to allow class defaults to work)
            if "num_encoder_layers" not in data:
                data["num_encoder_layers"] = 0
            if "use_cross_scale_fusion" not in data:
                data["use_cross_scale_fusion"] = False

        return data

    def _get_original_dimensions(self):
        return get_original_dimensions(self)

    @model_validator(mode="after")
    def validate_config_values(self):
        return validate_config_values(self)


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
    num_encoder_layers: int = 0  # Default to 0 for compatibility
    use_cross_scale_fusion: bool = False  # Default to False for compatibility

    projector_scale: list[Literal["P3", "P4", "P5"]] = ["P3", "P5"]
    pretrain_weights: Optional[str] = "rf-detr-large.pth"

    @model_validator(mode="before")
    @classmethod
    def apply_improvements_before(cls, data):
        """Apply improvements before model creation"""
        # Ensure data is a dict (handle None, empty dict, or other types)
        if not isinstance(data, dict):
            data = {} if data is None else dict(data)

        use_improvements = data.get("use_improvements", False)

        if use_improvements:
            # Apply improved dimensions - use improved_* values if provided, otherwise use defaults
            improved_hidden_dim = data.pop("improved_hidden_dim", None)
            improved_sa_nheads = data.pop("improved_sa_nheads", None)
            improved_ca_nheads = data.pop("improved_ca_nheads", None)
            improved_dec_n_points = data.pop("improved_dec_n_points", None)

            # Get class defaults for improved values
            if improved_hidden_dim is not None:
                data["hidden_dim"] = improved_hidden_dim
            elif "hidden_dim" not in data:
                data["hidden_dim"] = getattr(cls, "improved_hidden_dim", 512)

            if improved_sa_nheads is not None:
                data["sa_nheads"] = improved_sa_nheads
            elif "sa_nheads" not in data:
                data["sa_nheads"] = getattr(cls, "improved_sa_nheads", 16)

            if improved_ca_nheads is not None:
                data["ca_nheads"] = improved_ca_nheads
            elif "ca_nheads" not in data:
                data["ca_nheads"] = getattr(cls, "improved_ca_nheads", 32)

            if improved_dec_n_points is not None:
                data["dec_n_points"] = improved_dec_n_points
            elif "dec_n_points" not in data:
                data["dec_n_points"] = getattr(cls, "improved_dec_n_points", 6)

            data.setdefault("num_encoder_layers", 3)
            data.setdefault("use_cross_scale_fusion", True)
        else:
            # When use_improvements=False, ensure improved values are removed
            # but don't override existing values (let class defaults handle it)
            data.pop("improved_hidden_dim", None)
            data.pop("improved_sa_nheads", None)
            data.pop("improved_ca_nheads", None)
            data.pop("improved_dec_n_points", None)
            # Only set these if not already set (to allow class defaults to work)
            if "num_encoder_layers" not in data:
                data["num_encoder_layers"] = 0
            if "use_cross_scale_fusion" not in data:
                data["use_cross_scale_fusion"] = False

        return data

    @model_validator(mode="after")
    def validate_config_values(self):
        """Comprehensive validation of config values after validator execution.

        Uses the inherited validator from RFDETRBaseConfig which dynamically
        determines original values from class defaults, making it work correctly
        for all config subclasses including LargeConfig.
        """
        # Call parent validator which uses dynamic class-based values
        return super().validate_config_values()


class RFDETRNanoConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Nano model.

    NOTE: This config does not support use_improvements=True.
    Only Base, Large, and Medium configs support improvements.
    """

    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 2
    patch_size: int = 16
    resolution: int = 384
    positional_encoding_size: int = 24
    pretrain_weights: Optional[str] = "rf-detr-nano.pth"

    @model_validator(mode="before")
    @classmethod
    def apply_improvements_before(cls, data):
        """Reject use_improvements for configs that don't support it"""
        if not isinstance(data, dict):
            data = {} if data is None else dict(data)

        if data.get("use_improvements", False):
            raise ValueError(
                f"{cls.__name__} does not support use_improvements=True. "
                f"Only Base, Large, and Medium configs support improvements. "
                f"Remove use_improvements parameter or use RFDETRBaseConfig/RFDETRLargeConfig/RFDETRMediumConfig instead."
            )

        # Remove use_improvements and improved_* fields if present (shouldn't be, but be safe)
        data.pop("use_improvements", None)
        data.pop("improved_hidden_dim", None)
        data.pop("improved_sa_nheads", None)
        data.pop("improved_ca_nheads", None)
        data.pop("improved_dec_n_points", None)

        return data


class RFDETRSmallConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Small model.

    NOTE: This config does not support use_improvements=True.
    Only Base, Large, and Medium configs support improvements.
    """

    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 3
    patch_size: int = 16
    resolution: int = 512
    positional_encoding_size: int = 32
    pretrain_weights: Optional[str] = "rf-detr-small.pth"

    @model_validator(mode="before")
    @classmethod
    def apply_improvements_before(cls, data):
        """Reject use_improvements for configs that don't support it"""
        if not isinstance(data, dict):
            data = {} if data is None else dict(data)

        if data.get("use_improvements", False):
            raise ValueError(
                f"{cls.__name__} does not support use_improvements=True. "
                f"Only Base, Large, and Medium configs support improvements. "
                f"Remove use_improvements parameter or use RFDETRBaseConfig/RFDETRLargeConfig/RFDETRMediumConfig instead."
            )

        # Remove use_improvements and improved_* fields if present (shouldn't be, but be safe)
        data.pop("use_improvements", None)
        data.pop("improved_hidden_dim", None)
        data.pop("improved_sa_nheads", None)
        data.pop("improved_ca_nheads", None)
        data.pop("improved_dec_n_points", None)

        return data


class RFDETRMediumConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Medium model.

    NOTE: The improved architecture (use_improvements=True) requires retrained weights.
    Set use_improvements=False to use pretrained weights with original architecture.
    """

    use_improvements: bool = False

    out_feature_indexes: list[int] = [3, 6, 9, 12]
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
    num_encoder_layers: int = 0  # Default to 0 for compatibility
    use_cross_scale_fusion: bool = False  # Default to False for compatibility

    @model_validator(mode="before")
    @classmethod
    def apply_improvements_before(cls, data):
        """Apply improvements before model creation"""
        # Ensure data is a dict (handle None, empty dict, or other types)
        if not isinstance(data, dict):
            data = {} if data is None else dict(data)

        use_improvements = data.get("use_improvements", False)

        if use_improvements:
            # Apply improved dimensions - use improved_* values if provided, otherwise use defaults
            improved_hidden_dim = data.pop("improved_hidden_dim", None)
            improved_sa_nheads = data.pop("improved_sa_nheads", None)
            improved_ca_nheads = data.pop("improved_ca_nheads", None)
            improved_dec_n_points = data.pop("improved_dec_n_points", None)

            # Get class defaults for improved values
            if improved_hidden_dim is not None:
                data["hidden_dim"] = improved_hidden_dim
            elif "hidden_dim" not in data:
                data["hidden_dim"] = getattr(cls, "improved_hidden_dim", 384)

            if improved_sa_nheads is not None:
                data["sa_nheads"] = improved_sa_nheads
            elif "sa_nheads" not in data:
                data["sa_nheads"] = getattr(cls, "improved_sa_nheads", 12)

            if improved_ca_nheads is not None:
                data["ca_nheads"] = improved_ca_nheads
            elif "ca_nheads" not in data:
                data["ca_nheads"] = getattr(cls, "improved_ca_nheads", 24)

            if improved_dec_n_points is not None:
                data["dec_n_points"] = improved_dec_n_points
            elif "dec_n_points" not in data:
                data["dec_n_points"] = getattr(cls, "improved_dec_n_points", 4)

            data.setdefault("num_encoder_layers", 2)
            data.setdefault("use_cross_scale_fusion", True)
        else:
            # When use_improvements=False, ensure improved values are removed
            # but don't override existing values (let class defaults handle it)
            data.pop("improved_hidden_dim", None)
            data.pop("improved_sa_nheads", None)
            data.pop("improved_ca_nheads", None)
            data.pop("improved_dec_n_points", None)
            # Only set these if not already set (to allow class defaults to work)
            if "num_encoder_layers" not in data:
                data["num_encoder_layers"] = 0
            if "use_cross_scale_fusion" not in data:
                data["use_cross_scale_fusion"] = False

        return data

    @model_validator(mode="after")
    def validate_config_values(self):
        """Comprehensive validation of config values after validator execution.

        Uses the inherited validator from RFDETRBaseConfig which dynamically
        determines original values from class defaults, making it work correctly
        for all config subclasses including MediumConfig.
        """
        # Call parent validator which uses dynamic class-based values
        return super().validate_config_values()


class RFDETRSegPreviewConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR SegPreview model.

    NOTE: This config does not support use_improvements=True.
    Only Base, Large, and Medium configs support improvements.
    """

    segmentation_head: bool = True
    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 12
    resolution: int = 432
    positional_encoding_size: int = 36
    num_queries: int = 200
    num_select: int = 200
    pretrain_weights: Optional[str] = "rf-detr-seg-preview.pt"
    num_classes: int = 90

    @model_validator(mode="before")
    @classmethod
    def apply_improvements_before(cls, data):
        """Reject use_improvements for configs that don't support it"""
        if not isinstance(data, dict):
            data = {} if data is None else dict(data)

        if data.get("use_improvements", False):
            raise ValueError(
                f"{cls.__name__} does not support use_improvements=True. "
                f"Only Base, Large, and Medium configs support improvements. "
                f"Remove use_improvements parameter or use RFDETRBaseConfig/RFDETRLargeConfig/RFDETRMediumConfig instead."
            )

        # Remove use_improvements and improved_* fields if present (shouldn't be, but be safe)
        data.pop("use_improvements", None)
        data.pop("improved_hidden_dim", None)
        data.pop("improved_sa_nheads", None)
        data.pop("improved_ca_nheads", None)
        data.pop("improved_dec_n_points", None)

        return data
