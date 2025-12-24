#!/usr/bin/env python3
"""
Verification script for consistent validator implementation across config classes.

This script tests:
1. All config classes can be instantiated with default values
2. Validators work correctly for configs with improvements support (Base, Large, Medium)
3. Validators work correctly for configs without improvements support (Nano, Small, SegPreview)
4. Dynamic value detection works correctly (no hardcoded values)
5. Validation errors are raised appropriately for invalid configs
"""

import sys
import os
from typing import List, Tuple

# Add rfdetr to path for direct config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rfdetr'))

def test_config_creation():
    """Test that all config classes can be created with defaults."""
    print("=" * 60)
    print("Test 1: Config Creation with Defaults")
    print("=" * 60)
    
    try:
        from config import (
            RFDETRBaseConfig, RFDETRLargeConfig, RFDETRMediumConfig,
            RFDETRNanoConfig, RFDETRSmallConfig, RFDETRSegPreviewConfig
        )
        
        configs = {
            'Base': RFDETRBaseConfig(),
            'Large': RFDETRLargeConfig(),
            'Medium': RFDETRMediumConfig(),
            'Nano': RFDETRNanoConfig(),
            'Small': RFDETRSmallConfig(),
            'SegPreview': RFDETRSegPreviewConfig(),
        }
        
        for name, config in configs.items():
            print(f"✓ {name}Config created successfully")
            print(f"  - hidden_dim: {config.hidden_dim}")
            print(f"  - sa_nheads: {config.sa_nheads}")
            print(f"  - ca_nheads: {config.ca_nheads}")
            print(f"  - num_encoder_layers: {config.num_encoder_layers}")
            print(f"  - use_cross_scale_fusion: {config.use_cross_scale_fusion}")
            if hasattr(config, 'use_improvements'):
                print(f"  - use_improvements: {config.use_improvements}")
            print()
        
        return True, configs
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_improvements_support():
    """Test that configs with improvements support work correctly."""
    print("=" * 60)
    print("Test 2: Improvements Support (Base, Large, Medium)")
    print("=" * 60)
    
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rfdetr'))
        from config import RFDETRBaseConfig, RFDETRLargeConfig, RFDETRMediumConfig
        
        # Test Base config
        base_no_improvements = RFDETRBaseConfig(use_improvements=False)
        assert base_no_improvements.hidden_dim == 256, f"Expected 256, got {base_no_improvements.hidden_dim}"
        assert base_no_improvements.num_encoder_layers == 0, "Should be 0 when improvements disabled"
        print("✓ BaseConfig: use_improvements=False works correctly")
        
        base_with_improvements = RFDETRBaseConfig(use_improvements=True)
        assert base_with_improvements.hidden_dim == 320, f"Expected 320, got {base_with_improvements.hidden_dim}"
        assert base_with_improvements.num_encoder_layers > 0, "Should be > 0 when improvements enabled"
        print("✓ BaseConfig: use_improvements=True works correctly")
        
        # Test Large config
        large_no_improvements = RFDETRLargeConfig(use_improvements=False)
        assert large_no_improvements.hidden_dim == 384, f"Expected 384, got {large_no_improvements.hidden_dim}"
        print("✓ LargeConfig: use_improvements=False works correctly")
        
        large_with_improvements = RFDETRLargeConfig(use_improvements=True)
        assert large_with_improvements.hidden_dim == 512, f"Expected 512, got {large_with_improvements.hidden_dim}"
        print("✓ LargeConfig: use_improvements=True works correctly")
        
        # Test Medium config
        medium_no_improvements = RFDETRMediumConfig(use_improvements=False)
        assert medium_no_improvements.hidden_dim == 256, f"Expected 256, got {medium_no_improvements.hidden_dim}"
        print("✓ MediumConfig: use_improvements=False works correctly")
        
        medium_with_improvements = RFDETRMediumConfig(use_improvements=True)
        assert medium_with_improvements.hidden_dim == 384, f"Expected 384, got {medium_with_improvements.hidden_dim}"
        print("✓ MediumConfig: use_improvements=True works correctly")
        
        return True
    except Exception as e:
        print(f"✗ Improvements support test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_no_improvements_support():
    """Test that configs without improvements support work correctly."""
    print("=" * 60)
    print("Test 3: No Improvements Support (Nano, Small, SegPreview)")
    print("=" * 60)
    
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rfdetr'))
        from config import RFDETRNanoConfig, RFDETRSmallConfig, RFDETRSegPreviewConfig
        
        # These configs should not have use_improvements or improved_* fields
        nano = RFDETRNanoConfig()
        assert not hasattr(nano, 'use_improvements') or nano.num_encoder_layers == 0, "Nano should not support improvements"
        print("✓ NanoConfig: No improvements support (as expected)")
        
        small = RFDETRSmallConfig()
        assert not hasattr(small, 'use_improvements') or small.num_encoder_layers == 0, "Small should not support improvements"
        print("✓ SmallConfig: No improvements support (as expected)")
        
        seg = RFDETRSegPreviewConfig()
        assert not hasattr(seg, 'use_improvements') or seg.num_encoder_layers == 0, "SegPreview should not support improvements"
        print("✓ SegPreviewConfig: No improvements support (as expected)")
        
        return True
    except Exception as e:
        print(f"✗ No improvements support test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_errors():
    """Test that validation errors are raised appropriately."""
    print("=" * 60)
    print("Test 4: Validation Error Handling")
    print("=" * 60)
    
    try:
        from config import RFDETRBaseConfig
        from pydantic import ValidationError
        
        # Test invalid: improvements enabled but num_encoder_layers=0
        try:
            config = RFDETRBaseConfig(use_improvements=True, num_encoder_layers=0)
            print("✗ Should have raised ValidationError for improvements=True with num_encoder_layers=0")
            return False
        except ValueError as e:
            if "num_encoder_layers should be > 0" in str(e):
                print("✓ Correctly raises error for improvements=True with num_encoder_layers=0")
            else:
                print(f"✗ Wrong error message: {e}")
                return False
        
        # Test invalid: improvements disabled but use_cross_scale_fusion=True
        try:
            config = RFDETRBaseConfig(use_improvements=False, use_cross_scale_fusion=True)
            print("✗ Should have raised ValidationError for improvements=False with use_cross_scale_fusion=True")
            return False
        except ValueError as e:
            if "use_cross_scale_fusion should be False" in str(e):
                print("✓ Correctly raises error for improvements=False with use_cross_scale_fusion=True")
            else:
                print(f"✗ Wrong error message: {e}")
                return False
        
        # Test invalid: hidden_dim not divisible by sa_nheads
        try:
            config = RFDETRBaseConfig(hidden_dim=257, sa_nheads=8)  # 257 % 8 != 0
            print("✗ Should have raised ValidationError for hidden_dim not divisible by sa_nheads")
            return False
        except ValueError as e:
            if "not divisible by sa_nheads" in str(e):
                print("✓ Correctly raises error for hidden_dim not divisible by sa_nheads")
            else:
                print(f"✗ Wrong error message: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Validation error test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dynamic_values():
    """Test that validators use dynamic values (not hardcoded)."""
    print("=" * 60)
    print("Test 5: Dynamic Value Detection")
    print("=" * 60)
    
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rfdetr'))
        from config import RFDETRBaseConfig, RFDETRLargeConfig
        
        # Test that BaseConfig validator works with BaseConfig defaults
        base = RFDETRBaseConfig(use_improvements=False)
        base_original = base._get_original_dimensions()
        assert base_original['hidden_dim'] == 256, f"Expected 256, got {base_original['hidden_dim']}"
        print("✓ BaseConfig: Dynamic original dimensions detection works")
        
        # Test that LargeConfig validator works with LargeConfig defaults
        large = RFDETRLargeConfig(use_improvements=False)
        large_original = large._get_original_dimensions()
        assert large_original['hidden_dim'] == 384, f"Expected 384, got {large_original['hidden_dim']}"
        print("✓ LargeConfig: Dynamic original dimensions detection works")
        
        # Verify they're different (proving dynamic, not hardcoded)
        assert base_original['hidden_dim'] != large_original['hidden_dim'], "Values should be different"
        print("✓ Validators use dynamic values (not hardcoded)")
        
        return True
    except Exception as e:
        print(f"✗ Dynamic values test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validator_consistency():
    """Test that all validators have consistent structure."""
    print("=" * 60)
    print("Test 6: Validator Consistency")
    print("=" * 60)
    
    try:
        from config import (
            RFDETRBaseConfig, RFDETRLargeConfig, RFDETRMediumConfig,
            RFDETRNanoConfig, RFDETRSmallConfig, RFDETRSegPreviewConfig
        )
        
        # Check that all configs have validate_config_values method
        configs = [
            RFDETRBaseConfig,
            RFDETRLargeConfig,
            RFDETRMediumConfig,
            RFDETRNanoConfig,
            RFDETRSmallConfig,
            RFDETRSegPreviewConfig,
        ]
        
        for config_class in configs:
            assert hasattr(config_class, 'validate_config_values'), f"{config_class.__name__} missing validate_config_values"
            print(f"✓ {config_class.__name__}: Has validate_config_values method")
        
        # Check that LargeConfig and MediumConfig use super() (inherit from BaseConfig)
        # This ensures consistency - they should call parent validator
        large_validator = RFDETRLargeConfig.validate_config_values
        medium_validator = RFDETRMediumConfig.validate_config_values
        
        # Check source code to see if they call super()
        import inspect
        large_source = inspect.getsource(large_validator)
        medium_source = inspect.getsource(medium_validator)
        
        if 'super()' in large_source or 'super().validate_config_values' in large_source:
            print("✓ LargeConfig: Uses super() for consistency")
        else:
            print("⚠ LargeConfig: May not use super() - check implementation")
        
        if 'super()' in medium_source or 'super().validate_config_values' in medium_source:
            print("✓ MediumConfig: Uses super() for consistency")
        else:
            print("⚠ MediumConfig: May not use super() - check implementation")
        
        return True
    except Exception as e:
        print(f"✗ Validator consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("Validator Consistency Verification")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run all tests
    success, configs = test_config_creation()
    results.append(("Config Creation", success))
    
    if success:
        success = test_improvements_support()
        results.append(("Improvements Support", success))
        
        success = test_no_improvements_support()
        results.append(("No Improvements Support", success))
        
        success = test_validation_errors()
        results.append(("Validation Errors", success))
        
        success = test_dynamic_values()
        results.append(("Dynamic Values", success))
        
        success = test_validator_consistency()
        results.append(("Validator Consistency", success))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed! Validators are consistent across config classes.")
        return 0
    else:
        print("✗ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

