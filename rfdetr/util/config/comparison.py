"""
Utility functions for comparing checkpoint config with current config.

This module handles normalization and comparison of configs, accounting for
transformations that occur during model building.
"""

from logging import getLogger
from typing import Any, Optional

logger = getLogger(__name__)


def normalize_args_for_comparison(args: Any, model_state_dict: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """
    Normalize args to a comparable format, accounting for transformations.

    This function extracts args values and applies transformations that occur
    during model building, so comparisons are done on the actual model architecture
    values rather than the raw config values.

    Args:
        args: argparse.Namespace or dict containing model args
        model_state_dict: Optional model state_dict to get actual num_classes from model

    Returns:
        Dictionary of normalized parameter values for comparison
    """
    if hasattr(args, "__dict__"):
        args_dict = vars(args)
    elif isinstance(args, dict):
        args_dict = args
    else:
        return {}

    normalized = {}

    # Critical architecture parameters
    critical_params = [
        "encoder",
        "hidden_dim",
        "sa_nheads",
        "ca_nheads",
        "dec_layers",
        "dec_n_points",
        "num_queries",
        "group_detr",
        "projector_scale",
        "out_feature_indexes",
        "resolution",
        "patch_size",
        "num_windows",
        "num_encoder_layers",
        "enc_n_points",
        "use_cross_scale_fusion",
        "two_stage",
        "positional_encoding_size",
    ]

    for param in critical_params:
        if param in args_dict:
            normalized[param] = args_dict[param]

    # Handle num_classes transformation
    # CRITICAL: Use actual model num_classes if available (from class_embed.bias.shape[0])
    # This is the source of truth, as checkpoint args.num_classes may be incorrect
    if model_state_dict and "class_embed.bias" in model_state_dict:
        # Use actual model value (already transformed, includes background)
        actual_num_classes = model_state_dict["class_embed.bias"].shape[0]
        normalized["num_classes_transformed"] = actual_num_classes
        normalized["num_classes_original"] = actual_num_classes - 1
    elif "num_classes" in args_dict:
        # Fallback: transform args value (may be incorrect in some checkpoints)
        normalized["num_classes_transformed"] = args_dict["num_classes"] + 1
        normalized["num_classes_original"] = args_dict["num_classes"]

    # Handle use_cross_scale_fusion fallback
    # If missing, it defaults to False
    if "use_cross_scale_fusion" not in args_dict:
        normalized["use_cross_scale_fusion"] = False

    # Handle target_shape/resolution
    # Uses args.resolution or fallback to (640, 640)
    if "resolution" in args_dict:
        normalized["target_shape"] = (args_dict["resolution"], args_dict["resolution"])
    elif "shape" in args_dict:
        normalized["target_shape"] = args_dict["shape"]
    else:
        normalized["target_shape"] = (640, 640)

    return normalized


def compare_configs(
    checkpoint_args: Any,
    current_args: Any,
    checkpoint_model_state_dict: Optional[dict[str, Any]] = None,
    current_model_state_dict: Optional[dict[str, Any]] = None,
    critical_only: bool = True,
) -> tuple[bool, dict[str, Any], list[str]]:
    """
    Compare checkpoint config with current config.

    Args:
        checkpoint_args: Args from checkpoint (Namespace or dict)
        current_args: Current args (Namespace or dict)
        checkpoint_model_state_dict: Optional checkpoint model state_dict (for accurate num_classes)
        current_model_state_dict: Optional current model state_dict (for accurate num_classes)
        critical_only: If True, only compare critical architecture parameters

    Returns:
        Tuple of (is_compatible, differences_dict, warnings_list)
        - is_compatible: True if configs are compatible (mismatches are acceptable)
        - differences: Dict of parameter differences
        - warnings: List of warning messages
    """
    checkpoint_norm = normalize_args_for_comparison(checkpoint_args, checkpoint_model_state_dict)
    current_norm = normalize_args_for_comparison(current_args, current_model_state_dict)

    differences = {}
    warnings = []
    is_compatible = True

    # Critical parameters that must match for compatibility
    critical_params = [
        "encoder",
        "hidden_dim",
        "sa_nheads",
        "ca_nheads",
        "dec_layers",
        "dec_n_points",
        "num_queries",
        "group_detr",
        "projector_scale",
        "out_feature_indexes",
        "num_classes_transformed",
    ]

    # Parameters that can differ but should be noted
    flexible_params = [
        "resolution",
        "patch_size",
        "num_windows",
        "num_encoder_layers",
        "enc_n_points",
        "use_cross_scale_fusion",
        "two_stage",
    ]

    params_to_check = critical_params if critical_only else (critical_params + flexible_params)

    for param in params_to_check:
        checkpoint_val = checkpoint_norm.get(param)
        current_val = current_norm.get(param)

        if checkpoint_val is None and current_val is None:
            continue

        if checkpoint_val is None:
            warnings.append(f"Checkpoint missing parameter: {param}")
            continue

        if current_val is None:
            warnings.append(f"Current config missing parameter: {param}")
            continue

        # Handle list comparisons
        if isinstance(checkpoint_val, list) and isinstance(current_val, list):
            if checkpoint_val != current_val:
                differences[param] = {"checkpoint": checkpoint_val, "current": current_val}
                if param in critical_params:
                    is_compatible = False
                    warnings.append(f"CRITICAL mismatch: {param} - checkpoint={checkpoint_val}, current={current_val}")
        elif checkpoint_val != current_val:
            differences[param] = {"checkpoint": checkpoint_val, "current": current_val}
            if param in critical_params:
                is_compatible = False
                warnings.append(f"CRITICAL mismatch: {param} - checkpoint={checkpoint_val}, current={current_val}")
            elif param in flexible_params:
                warnings.append(
                    f"Difference in {param}: checkpoint={checkpoint_val}, current={current_val} (may be acceptable)"
                )

    return is_compatible, differences, warnings


def validate_checkpoint_config_compatibility(
    checkpoint_path: str, current_args: Any, warn_only: bool = True
) -> tuple[bool, list[str]]:
    """
    Validate that checkpoint config is compatible with current config.

    Args:
        checkpoint_path: Path to checkpoint file
        current_args: Current model args (Namespace or dict)
        warn_only: If True, only warn on mismatches; if False, raise errors

    Returns:
        Tuple of (is_compatible, warnings_list)
    """
    import torch

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load checkpoint for config validation: {e}")
        return False, [f"Failed to load checkpoint: {e}"]

    if "args" not in checkpoint:
        logger.warning("Checkpoint does not contain 'args' - cannot validate config compatibility")
        return True, ["Checkpoint missing 'args' - config validation skipped"]

    checkpoint_args = checkpoint["args"]

    is_compatible, differences, warnings = compare_configs(checkpoint_args, current_args, critical_only=True)

    if differences:
        logger.warning("Config differences found between checkpoint and current config:")
        for param, vals in differences.items():
            logger.warning(f"  {param}: checkpoint={vals['checkpoint']}, current={vals['current']}")

    if warnings:
        for warning in warnings:
            if warn_only:
                logger.warning(warning)
            else:
                logger.error(warning)

    if not is_compatible and not warn_only:
        error_msg = (
            f"Checkpoint config is incompatible with current config.\n"
            f"Critical mismatches found. Please use a compatible checkpoint or adjust your config.\n"
            f"Differences: {differences}"
        )
        raise ValueError(error_msg)

    return is_compatible, warnings
