# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

# Backward compatibility imports - re-export from new submodules
from rfdetr.util.checkpoint import (
    download_file,
    download_resume_checkpoint,
    validate_checkpoint,
    validate_checkpoint_keys,
    validate_checkpoint_structure,
)
from rfdetr.util.config import (
    compare_configs,
    normalize_args_for_comparison,
    validate_checkpoint_config_compatibility,
)
from rfdetr.util.metrics import (
    MetricsPlotSink,
    MetricsTensorBoardSink,
    MetricsWandBSink,
)

__all__ = [
    # Checkpoint utilities
    "validate_checkpoint",
    "validate_checkpoint_structure",
    "validate_checkpoint_keys",
    "download_file",
    "download_resume_checkpoint",
    # Metrics utilities
    "MetricsPlotSink",
    "MetricsTensorBoardSink",
    "MetricsWandBSink",
    # Config utilities
    "normalize_args_for_comparison",
    "compare_configs",
    "validate_checkpoint_config_compatibility",
]
