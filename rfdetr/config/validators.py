# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from pydantic import model_validator
from typing import List

def get_required_fields(cls):
    """Get list of required fields from ModelConfig base class.
    
    Returns a list of field names that don't have defaults in ModelConfig.
    These fields must be provided when creating a config instance.
    """
    # Required fields from ModelConfig (those without defaults)
    model_config_required = [
        'encoder',
        'out_feature_indexes',
        'dec_layers',
        'projector_scale',
        'hidden_dim',
        'patch_size',
        'num_windows',
        'sa_nheads',
        'ca_nheads',
        'dec_n_points',
        'resolution',
        'positional_encoding_size',
    ]
    
    # Check which of these are actually required (don't have defaults in this class)
    required_fields = []
    for field_name in model_config_required:
        # Check if field has a default in this class or any parent class
        has_default = False
        for parent in cls.__mro__:
            if hasattr(parent, 'model_fields') and parent.model_fields:
                field_info = parent.model_fields.get(field_name)
                if field_info is not None:
                    # Check if field has a default value
                    if hasattr(field_info, 'default') and field_info.default is not None:
                        has_default = True
                        break
                    if hasattr(field_info, 'default_factory') and field_info.default_factory is not None:
                        has_default = True
                        break
            # Also check class attributes as fallback
            if hasattr(parent, field_name):
                default_value = getattr(parent, field_name, None)
                if default_value is not None:
                    has_default = True
                    break
        
        if not has_default:
            required_fields.append(field_name)
    
    return required_fields


def validate_required_fields_before(cls, data):
    """Validate that all required fields are present before model creation.
    
    This validator runs before Pydantic's built-in validation to provide
    clearer error messages for missing required fields.
    """
    # Ensure data is a dict (handle None, empty dict, or other types)
    if not isinstance(data, dict):
        data = {} if data is None else dict(data)
    
    # Get required fields for this config class
    required_fields = get_required_fields(cls)
    
    # Check for missing required fields
    missing_fields = []
    invalid_fields = []
    
    # Fields that cannot be empty lists (even if they have defaults)
    list_fields_that_cannot_be_empty = ['out_feature_indexes', 'projector_scale']
    
    for field_name in required_fields:
        if field_name not in data:
            # Check if field has a default in the class hierarchy
            has_default = False
            for parent in cls.__mro__:
                if hasattr(parent, 'model_fields') and parent.model_fields:
                    field_info = parent.model_fields.get(field_name)
                    if field_info is not None:
                        if hasattr(field_info, 'default') and field_info.default is not None:
                            has_default = True
                            break
                        if hasattr(field_info, 'default_factory') and field_info.default_factory is not None:
                            has_default = True
                            break
                elif hasattr(parent, field_name):
                    default_value = getattr(parent, field_name, None)
                    if default_value is not None:
                        has_default = True
                        break
            
            if not has_default:
                missing_fields.append(field_name)
        else:
            # Field is present, validate it's not None or empty
            value = data[field_name]
            if value is None:
                invalid_fields.append(f"{field_name} cannot be None")
            elif field_name in list_fields_that_cannot_be_empty and isinstance(value, list) and len(value) == 0:
                invalid_fields.append(f"{field_name} cannot be an empty list")
    
    # Also check for empty lists in fields that are provided, even if they're not required
    # (because empty lists are always invalid, regardless of whether field is required)
    for field_name in list_fields_that_cannot_be_empty:
        if field_name in data:
            value = data[field_name]
            if isinstance(value, list) and len(value) == 0:
                invalid_fields.append(f"{field_name} cannot be an empty list")
    
    # Collect all errors
    errors = []
    if missing_fields:
        errors.append(
            f"Missing required fields for {cls.__name__}: {', '.join(missing_fields)}. "
            f"These fields must be provided when creating a config instance. "
            f"Consider using a predefined config class (e.g., RFDETRBaseConfig) "
            f"or providing all required fields explicitly."
        )
    if invalid_fields:
        errors.extend(invalid_fields)
    
    if errors:
        error_msg = "Config validation failed (required fields):\n  " + "\n  ".join(errors)
        raise ValueError(error_msg)
    
    return data


def get_original_dimensions(self):
    """Get original dimensions for this config class (from class defaults).
    
    Uses Pydantic v2 model_fields to dynamically get class defaults,
    making this work correctly for all config subclasses.
    """
    cls = self.__class__
    # Access model_fields safely - in Pydantic v2, model_fields is a dict-like object
    def get_field_default(field_name, fallback):
        field_info = cls.model_fields.get(field_name) if hasattr(cls, 'model_fields') else None
        if field_info and hasattr(field_info, 'default') and field_info.default is not None:
            return field_info.default
        # Try to get from class attribute as fallback
        return getattr(cls, field_name, fallback)
    
    return {
        'hidden_dim': get_field_default('hidden_dim', 256),
        'sa_nheads': get_field_default('sa_nheads', 8),
        'ca_nheads': get_field_default('ca_nheads', 16),
        'dec_n_points': get_field_default('dec_n_points', 2),
    }


def validate_config_values(self):
    """Comprehensive validation of config values after validator execution.
    
    Validates:
    1. Required fields are present and valid
    2. Improved/original value consistency based on use_improvements flag
    3. Critical value ranges
    4. Field consistency (e.g., ca_nheads >= sa_nheads, hidden_dim divisibility)
    5. Encoder/cross-scale fusion settings consistency
    
    This validator uses dynamic class-based values instead of hardcoded ones,
    making it work correctly for all config subclasses.
    """
    errors = []
    
    # 0. Validate required fields are present and valid (double-check after Pydantic validation)
    required_fields = get_required_fields(self.__class__)
    for field_name in required_fields:
        if not hasattr(self, field_name):
            errors.append(f"Required field '{field_name}' is missing after validation. This should not happen.")
        else:
            value = getattr(self, field_name)
            if value is None:
                errors.append(f"Required field '{field_name}' cannot be None")
            elif field_name == 'out_feature_indexes' and isinstance(value, list) and len(value) == 0:
                errors.append(f"Required field '{field_name}' cannot be an empty list")
            elif field_name == 'projector_scale' and isinstance(value, list) and len(value) == 0:
                errors.append(f"Required field '{field_name}' cannot be an empty list")
            elif field_name == 'encoder' and value not in ["dinov2_windowed_small", "dinov2_windowed_base"]:
                errors.append(f"Required field '{field_name}' must be one of ['dinov2_windowed_small', 'dinov2_windowed_base'], got '{value}'")
            elif field_name in ['hidden_dim', 'patch_size', 'num_windows', 'sa_nheads', 'ca_nheads', 
                               'dec_n_points', 'dec_layers', 'resolution', 'positional_encoding_size']:
                if not isinstance(value, int):
                    errors.append(f"Required field '{field_name}' must be an integer, got {type(value).__name__}")
                elif value <= 0:
                    errors.append(f"Required field '{field_name}' must be > 0, got {value}")
            elif field_name == 'projector_scale':
                if not isinstance(value, list):
                    errors.append(f"Required field '{field_name}' must be a list, got {type(value).__name__}")
                else:
                    valid_scales = ["P3", "P4", "P5"]
                    invalid_scales = [s for s in value if s not in valid_scales]
                    if invalid_scales:
                        errors.append(f"Required field '{field_name}' contains invalid scales: {invalid_scales}. "
                                    f"Valid scales are: {valid_scales}")
            elif field_name == 'out_feature_indexes':
                if not isinstance(value, list):
                    errors.append(f"Required field '{field_name}' must be a list, got {type(value).__name__}")
                else:
                    invalid_indexes = [idx for idx in value if not isinstance(idx, int)]
                    if invalid_indexes:
                        errors.append(f"Required field '{field_name}' contains non-integer values: {invalid_indexes}")
    
    
    # Check if this config class supports improvements
    supports_improvements = hasattr(self, 'use_improvements') and hasattr(self, 'improved_hidden_dim')
    
    # 1. Validate use_improvements flag consistency (only if improvements are supported)
    if supports_improvements:
        if not self.use_improvements:
            # When improvements are disabled, validate original values are used
            if self.num_encoder_layers != 0:
                errors.append(f"num_encoder_layers should be 0 when use_improvements=False, got {self.num_encoder_layers}")
            if self.use_cross_scale_fusion:
                errors.append("use_cross_scale_fusion should be False when use_improvements=False")
            
            # Validate that we're using original (not improved) dimensions
            original_dims = get_original_dimensions(self)
            if hasattr(self, 'improved_hidden_dim') and self.hidden_dim == self.improved_hidden_dim:
                errors.append(f"hidden_dim={self.hidden_dim} matches improved_hidden_dim when use_improvements=False. "
                            f"Expected original hidden_dim={original_dims['hidden_dim']}. "
                            f"Did the validator fail to apply original values?")
        else:
            # When improvements are enabled, validate improved values were applied
            if self.num_encoder_layers == 0:
                errors.append(f"num_encoder_layers should be > 0 when use_improvements=True, got {self.num_encoder_layers}")
            if not self.use_cross_scale_fusion:
                errors.append("use_cross_scale_fusion should be True when use_improvements=True")
            
            # Validate that improved dimensions are being used (not original)
            if hasattr(self, 'improved_hidden_dim'):
                original_dims = get_original_dimensions(self)
                if self.hidden_dim == original_dims['hidden_dim']:
                    errors.append(f"hidden_dim={self.hidden_dim} matches original value when use_improvements=True. "
                                f"Expected improved_hidden_dim={self.improved_hidden_dim}. "
                                f"Did the validator fail to apply improved values?")
    else:
        # Configs without improvements support should have these disabled
        # Also check if use_improvements was somehow set (should be caught by before validator, but double-check)
        if hasattr(self, 'use_improvements') and self.use_improvements:
            errors.append(
                f"{self.__class__.__name__} does not support use_improvements=True. "
                f"Only Base, Large, and Medium configs support improvements. "
                f"This should have been caught by the before validator."
            )
        if self.num_encoder_layers != 0:
            errors.append(f"num_encoder_layers should be 0 for {self.__class__.__name__} (improvements not supported), got {self.num_encoder_layers}")
        if self.use_cross_scale_fusion:
            errors.append(f"use_cross_scale_fusion should be False for {self.__class__.__name__} (improvements not supported)")
    
    # 2. Validate critical value ranges
    if self.hidden_dim <= 0:
        errors.append(f"hidden_dim must be > 0, got {self.hidden_dim}")
    if self.sa_nheads <= 0:
        errors.append(f"sa_nheads must be > 0, got {self.sa_nheads}")
    if self.ca_nheads <= 0:
        errors.append(f"ca_nheads must be > 0, got {self.ca_nheads}")
    if self.dec_n_points <= 0:
        errors.append(f"dec_n_points must be > 0, got {self.dec_n_points}")
    if self.dec_layers <= 0:
        errors.append(f"dec_layers must be > 0, got {self.dec_layers}")
    if self.num_encoder_layers < 0:
        errors.append(f"num_encoder_layers must be >= 0, got {self.num_encoder_layers}")
    
    # 3. Validate field consistency
    # Only check consistency if values are valid (to avoid division by zero)
    if self.sa_nheads > 0 and self.ca_nheads > 0:
        # ca_nheads should typically be >= sa_nheads (cross-attention often needs more heads)
        if self.ca_nheads < self.sa_nheads:
            errors.append(f"ca_nheads ({self.ca_nheads}) < sa_nheads ({self.sa_nheads}). "
                         f"Cross-attention typically needs at least as many heads as self-attention.")
    
    # hidden_dim should be divisible by sa_nheads and ca_nheads for efficient attention
    # Only check divisibility if values are valid (to avoid division by zero)
    if self.hidden_dim > 0 and self.sa_nheads > 0:
        if self.hidden_dim % self.sa_nheads != 0:
            errors.append(f"hidden_dim ({self.hidden_dim}) is not divisible by sa_nheads ({self.sa_nheads}). "
                         f"This may cause dimension mismatches in attention layers.")
    if self.hidden_dim > 0 and self.ca_nheads > 0:
        if self.hidden_dim % self.ca_nheads != 0:
            errors.append(f"hidden_dim ({self.hidden_dim}) is not divisible by ca_nheads ({self.ca_nheads}). "
                         f"This may cause dimension mismatches in attention layers.")
    
    # 4. Validate encoder and cross-scale fusion consistency
    if self.num_encoder_layers > 0 and not self.use_cross_scale_fusion:
        # This is not necessarily an error, but worth warning
        # Cross-scale fusion is typically used with encoder layers
        pass  # Allow this combination, but could add warning if needed
    
    # 5. Validate resolution and positional encoding consistency
    if hasattr(self, 'resolution') and hasattr(self, 'positional_encoding_size'):
        # Positional encoding size should roughly match resolution / patch_size
        expected_pos_size = (self.resolution // self.patch_size) + 1  # Approximate
        if abs(self.positional_encoding_size - expected_pos_size) > 5:
            # Allow some flexibility, but warn if very different
            pass  # Could add warning if needed
    
    # Raise all errors at once for better debugging
    if errors:
        error_msg = "Config validation failed:\n  " + "\n  ".join(errors)
        raise ValueError(error_msg)
    
    return self

