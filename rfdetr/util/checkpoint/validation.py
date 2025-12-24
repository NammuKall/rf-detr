# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import os
import torch
from logging import getLogger
from typing import Optional, Tuple, Dict, List

logger = getLogger(__name__)


def validate_checkpoint(
    checkpoint_path: str,
    required_keys: Optional[List[str]] = None,
    checkpoint_type: str = "auto",
    validate_structure: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a checkpoint file exists and is a valid PyTorch checkpoint.
    
    Enhanced validation that checks:
    - File existence and readability
    - File size (not corrupted)
    - PyTorch checkpoint format
    - Required keys presence
    - Checkpoint structure (model state_dict, optimizer, etc.)
    
    Args:
        checkpoint_path: Path to checkpoint file
        required_keys: List of required keys in checkpoint dict (e.g., ['model'])
        checkpoint_type: Type of checkpoint ("pretrain", "resume", "inference", "auto")
        validate_structure: Whether to perform detailed structure validation
        
    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        return False, f"Checkpoint file does not exist: {checkpoint_path}"
    
    # Check if file is readable
    if not os.access(checkpoint_path, os.R_OK):
        return False, f"Checkpoint file is not readable: {checkpoint_path}"
    
    # Check if file has reasonable size (at least 1KB)
    file_size = os.path.getsize(checkpoint_path)
    if file_size < 1024:
        return False, f"Checkpoint file is too small ({file_size} bytes), likely corrupted: {checkpoint_path}"
    
    # Try to load and validate checkpoint structure
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if validate_structure:
            is_valid, error_msg, _ = validate_checkpoint_structure(
                checkpoint, checkpoint_type=checkpoint_type, required_keys=required_keys
            )
            if not is_valid:
                return False, error_msg
        
        # Check required keys if specified
        if required_keys:
            is_valid, error_msg, _ = validate_checkpoint_keys(checkpoint, required_keys)
            if not is_valid:
                return False, error_msg
        
        return True, None
        
    except Exception as e:
        return False, f"Failed to load checkpoint: {str(e)}"


def validate_checkpoint_structure(
    checkpoint: Dict,
    checkpoint_type: str = "auto",
    required_keys: Optional[List[str]] = None
) -> Tuple[bool, Optional[str], Dict[str, bool]]:
    """
    Validate checkpoint structure based on checkpoint type.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        checkpoint_type: Type of checkpoint ("pretrain", "resume", "inference", "auto")
        required_keys: Optional list of required keys
        
    Returns:
        Tuple of (is_valid, error_message, key_presence_dict)
    """
    if not isinstance(checkpoint, dict):
        return False, f"Checkpoint must be a dictionary, got {type(checkpoint).__name__}", {}
    
    key_presence = {
        "model": "model" in checkpoint or "state_dict" in checkpoint,
        "optimizer": "optimizer" in checkpoint,
        "lr_scheduler": "lr_scheduler" in checkpoint,
        "epoch": "epoch" in checkpoint,
        "args": "args" in checkpoint,
    }
    
    # Auto-detect checkpoint type if needed
    if checkpoint_type == "auto":
        if key_presence["optimizer"] and key_presence["epoch"]:
            checkpoint_type = "resume"
        elif key_presence["model"] or key_presence["state_dict"]:
            checkpoint_type = "pretrain"
        else:
            checkpoint_type = "inference"
    
    # Validate based on type
    if checkpoint_type == "resume":
        if not key_presence["model"]:
            return False, "Resume checkpoint missing 'model' or 'state_dict' key", key_presence
    elif checkpoint_type == "pretrain":
        if not key_presence["model"]:
            return False, "Pretrain checkpoint missing 'model' or 'state_dict' key", key_presence
    
    return True, None, key_presence


def validate_checkpoint_keys(
    checkpoint: Dict,
    required_keys: List[str],
    optional_keys: Optional[List[str]] = None
) -> Tuple[bool, Optional[str], Dict[str, bool]]:
    """
    Validate that checkpoint contains required keys.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        required_keys: List of required keys
        optional_keys: Optional list of keys to check presence for
        
    Returns:
        Tuple of (is_valid, error_message, key_presence_dict)
    """
    key_presence = {}
    missing_keys = []
    
    for key in required_keys:
        present = key in checkpoint
        key_presence[key] = present
        if not present:
            missing_keys.append(key)
    
    if optional_keys:
        for key in optional_keys:
            key_presence[key] = key in checkpoint
    
    if missing_keys:
        return False, f"Checkpoint missing required keys: {', '.join(missing_keys)}", key_presence
    
    return True, None, key_presence

