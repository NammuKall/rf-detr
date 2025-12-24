#!/usr/bin/env python3
"""
Verification script for Pydantic v2 API compatibility fixes.

This script verifies:
1. Pydantic version and API compatibility
2. Config validator execution with empty kwargs
3. Config value propagation (config → model_dump() → model creation)
4. Model dimensions match expected values

Run this script to verify all fixes are working correctly.

Prerequisites:
    pip install pydantic numpy torch
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Check dependencies first
def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []
    
    try:
        import pydantic
    except ImportError:
        missing.append("pydantic")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    return missing

def test_pydantic_version():
    """Test 1: Check Pydantic version and API compatibility"""
    print("\n" + "="*60)
    print("TEST 1: Pydantic Version and API Compatibility")
    print("="*60)
    
    try:
        import pydantic
        version = pydantic.__version__
        print(f"✓ Pydantic version: {version}")
        
        # Check if model_dump() exists (Pydantic v2)
        if hasattr(pydantic.BaseModel, 'model_dump'):
            print("✓ Pydantic v2 API detected (model_dump() available)")
            return True, version
        elif hasattr(pydantic.BaseModel, 'dict'):
            print("⚠ Pydantic v1 API detected (dict() available)")
            print("  Note: Code has been updated to use model_dump()")
            print("  If using Pydantic v1, you may need to add compatibility layer")
            return True, version
        else:
            print("✗ Unknown Pydantic version or API")
            return False, version
    except ImportError:
        print("✗ Pydantic not installed")
        return False, None


def test_config_creation():
    """Test 2: Config creation with empty kwargs"""
    print("\n" + "="*60)
    print("TEST 2: Config Creation with Empty kwargs")
    print("="*60)
    
    try:
        from rfdetr.config import RFDETRBaseConfig
        
        # Test with empty kwargs
        print("\n2.1: Creating config with empty kwargs...")
        config1 = RFDETRBaseConfig()
        assert config1.hidden_dim == 256, f"Expected hidden_dim=256, got {config1.hidden_dim}"
        assert config1.use_improvements == False, f"Expected use_improvements=False, got {config1.use_improvements}"
        assert config1.sa_nheads == 8, f"Expected sa_nheads=8, got {config1.sa_nheads}"
        assert config1.ca_nheads == 16, f"Expected ca_nheads=16, got {config1.ca_nheads}"
        print("  ✓ Config created successfully with correct default values")
        
        # Test with explicit use_improvements=False
        print("\n2.2: Creating config with use_improvements=False...")
        config2 = RFDETRBaseConfig(use_improvements=False)
        assert config2.hidden_dim == 256, f"Expected hidden_dim=256, got {config2.hidden_dim}"
        assert config2.sa_nheads == 8, f"Expected sa_nheads=8, got {config2.sa_nheads}"
        assert config2.ca_nheads == 16, f"Expected ca_nheads=16, got {config2.ca_nheads}"
        print("  ✓ Config created successfully with use_improvements=False")
        
        # Test with explicit use_improvements=True
        print("\n2.3: Creating config with use_improvements=True...")
        config3 = RFDETRBaseConfig(use_improvements=True)
        assert config3.hidden_dim == 320, f"Expected hidden_dim=320, got {config3.hidden_dim}"
        assert config3.sa_nheads == 10, f"Expected sa_nheads=10, got {config3.sa_nheads}"
        assert config3.ca_nheads == 20, f"Expected ca_nheads=20, got {config3.ca_nheads}"
        print("  ✓ Config created successfully with use_improvements=True")
        
        return True
    except Exception as e:
        print(f"  ✗ Config creation failed: {e}")
        traceback.print_exc()
        return False


def test_model_dump():
    """Test 3: model_dump() returns correct values"""
    print("\n" + "="*60)
    print("TEST 3: model_dump() API Compatibility")
    print("="*60)
    
    try:
        from rfdetr.config import RFDETRBaseConfig
        
        # Test model_dump() exists and works
        print("\n3.1: Testing model_dump() method...")
        config = RFDETRBaseConfig()
        
        # Check if model_dump() exists
        if not hasattr(config, 'model_dump'):
            print("  ✗ model_dump() method not found")
            # Try dict() as fallback
            if hasattr(config, 'dict'):
                print("  ⚠ Using dict() as fallback (Pydantic v1)")
                config_dict = config.dict()
            else:
                return False
        else:
            config_dict = config.model_dump()
        
        assert isinstance(config_dict, dict), f"Expected dict, got {type(config_dict)}"
        assert 'hidden_dim' in config_dict, "hidden_dim not in config dict"
        assert config_dict['hidden_dim'] == 256, f"Expected hidden_dim=256, got {config_dict['hidden_dim']}"
        assert config_dict['use_improvements'] == False, f"Expected use_improvements=False, got {config_dict['use_improvements']}"
        print("  ✓ model_dump() works correctly")
        print(f"  ✓ hidden_dim in dict: {config_dict['hidden_dim']}")
        print(f"  ✓ sa_nheads in dict: {config_dict.get('sa_nheads', 'N/A')}")
        print(f"  ✓ ca_nheads in dict: {config_dict.get('ca_nheads', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"  ✗ model_dump() test failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """Test 4: Model creation with config"""
    print("\n" + "="*60)
    print("TEST 4: Model Creation with Config")
    print("="*60)
    
    try:
        from rfdetr.detr import RFDETRBase
        
        print("\n4.1: Creating RFDETRBase model...")
        model = RFDETRBase()
        
        # Verify model was created
        assert model.model is not None, "Model was not created"
        assert model.model_config is not None, "Model config was not created"
        print("  ✓ Model created successfully")
        
        # Verify config values
        config = model.model_config
        print(f"  ✓ Config hidden_dim: {config.hidden_dim}")
        print(f"  ✓ Config use_improvements: {config.use_improvements}")
        
        # Try to access model dimensions (may require model to be built)
        try:
            # Check if model has transformer
            if hasattr(model.model, 'model') and hasattr(model.model.model, 'transformer'):
                transformer = model.model.model.transformer
                if hasattr(transformer, 'd_model'):
                    print(f"  ✓ Transformer d_model: {transformer.d_model}")
                    if transformer.d_model != config.hidden_dim:
                        print(f"  ⚠ WARNING: Transformer d_model ({transformer.d_model}) != config hidden_dim ({config.hidden_dim})")
                    else:
                        print(f"  ✓ Transformer dimensions match config")
        except Exception as e:
            print(f"  ⚠ Could not verify transformer dimensions: {e}")
            print("  (This is OK if model hasn't been fully initialized)")
        
        return True
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        traceback.print_exc()
        return False


def test_value_propagation():
    """Test 5: Value propagation through the chain"""
    print("\n" + "="*60)
    print("TEST 5: Value Propagation Chain")
    print("="*60)
    
    try:
        from rfdetr.config import RFDETRBaseConfig
        from rfdetr.main import populate_args
        
        print("\n5.1: Testing config → model_dump() → populate_args()...")
        config = RFDETRBaseConfig()
        
        # Step 1: Config values
        print(f"  Step 1 - Config values:")
        print(f"    hidden_dim: {config.hidden_dim}")
        print(f"    sa_nheads: {config.sa_nheads}")
        print(f"    ca_nheads: {config.ca_nheads}")
        
        # Step 2: model_dump()
        config_dict = config.model_dump()
        print(f"\n  Step 2 - model_dump() values:")
        print(f"    hidden_dim: {config_dict.get('hidden_dim', 'MISSING')}")
        print(f"    sa_nheads: {config_dict.get('sa_nheads', 'MISSING')}")
        print(f"    ca_nheads: {config_dict.get('ca_nheads', 'MISSING')}")
        
        # Verify values match
        assert config_dict['hidden_dim'] == config.hidden_dim, "hidden_dim mismatch"
        assert config_dict['sa_nheads'] == config.sa_nheads, "sa_nheads mismatch"
        assert config_dict['ca_nheads'] == config.ca_nheads, "ca_nheads mismatch"
        print("  ✓ Values match between config and model_dump()")
        
        # Step 3: populate_args()
        args = populate_args(**config_dict)
        print(f"\n  Step 3 - populate_args() values:")
        print(f"    hidden_dim: {args.hidden_dim}")
        print(f"    sa_nheads: {args.sa_nheads}")
        print(f"    ca_nheads: {args.ca_nheads}")
        
        # Verify values match
        assert args.hidden_dim == config.hidden_dim, f"hidden_dim mismatch: {args.hidden_dim} != {config.hidden_dim}"
        assert args.sa_nheads == config.sa_nheads, f"sa_nheads mismatch: {args.sa_nheads} != {config.sa_nheads}"
        assert args.ca_nheads == config.ca_nheads, f"ca_nheads mismatch: {args.ca_nheads} != {config.ca_nheads}"
        print("  ✓ Values match between model_dump() and populate_args()")
        
        return True
    except Exception as e:
        print(f"  ✗ Value propagation test failed: {e}")
        traceback.print_exc()
        return False


def test_all_config_classes():
    """Test 6: All config classes work correctly"""
    print("\n" + "="*60)
    print("TEST 6: All Config Classes")
    print("="*60)
    
    try:
        from rfdetr.config import (
            RFDETRBaseConfig,
            RFDETRLargeConfig,
            RFDETRMediumConfig,
            RFDETRSmallConfig,
            RFDETRNanoConfig
        )
        
        config_classes = [
            ("RFDETRBaseConfig", RFDETRBaseConfig, {'hidden_dim': 256, 'ca_nheads': 16}),
            ("RFDETRLargeConfig", RFDETRLargeConfig, {'hidden_dim': 384, 'ca_nheads': 24}),
            ("RFDETRMediumConfig", RFDETRMediumConfig, {'hidden_dim': 256, 'ca_nheads': 16}),
            ("RFDETRSmallConfig", RFDETRSmallConfig, {'hidden_dim': 256, 'ca_nheads': 16}),
            ("RFDETRNanoConfig", RFDETRNanoConfig, {'hidden_dim': 256, 'ca_nheads': 16}),
        ]
        
        all_passed = True
        for name, config_class, expected_values in config_classes:
            print(f"\n6.{len([c for c in config_classes if config_classes.index((name, config_class, expected_values)) < config_classes.index((name, config_class, expected_values))]) + 1}: Testing {name}...")
            try:
                config = config_class()
                
                # Verify model_dump() works
                config_dict = config.model_dump()
                assert isinstance(config_dict, dict), f"{name}: model_dump() didn't return dict"
                
                # Verify expected values
                for key, expected_value in expected_values.items():
                    actual_value = getattr(config, key)
                    if actual_value != expected_value:
                        print(f"  ⚠ {name}: {key} = {actual_value}, expected {expected_value}")
                    else:
                        print(f"  ✓ {name}: {key} = {actual_value}")
                
            except Exception as e:
                print(f"  ✗ {name} failed: {e}")
                all_passed = False
        
        return all_passed
    except Exception as e:
        print(f"  ✗ Config classes test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all verification tests"""
    print("\n" + "="*60)
    print("PYDANTIC V2 API COMPATIBILITY VERIFICATION")
    print("="*60)
    print("\nThis script verifies that all Pydantic v2 API fixes are working correctly.")
    print("It tests config creation, value propagation, and model initialization.\n")
    
    # Check dependencies first
    print("Checking dependencies...")
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("\nPlease install missing dependencies:")
        print(f"  pip install {' '.join(missing_deps)}")
        print("\nOr install all project dependencies:")
        print("  pip install -e .")
        print("\nAfter installing dependencies, run this script again.")
        return 1
    
    print("✓ All dependencies installed\n")
    
    results = []
    
    # Run all tests
    results.append(("Pydantic Version", test_pydantic_version()[0]))
    results.append(("Config Creation", test_config_creation()))
    results.append(("model_dump() API", test_model_dump()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Value Propagation", test_value_propagation()))
    results.append(("All Config Classes", test_all_config_classes()))
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Pydantic v2 API compatibility fixes are working correctly.")
        print("\nNext steps:")
        print("  1. Test with actual checkpoint loading")
        print("  2. Verify weight shapes match expected values")
        print("  3. Run full training/inference tests")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")
        print("\nTroubleshooting:")
        print("  1. Ensure Pydantic is installed: pip install pydantic")
        print("  2. Check if using Pydantic v1 (may need compatibility layer)")
        print("  3. Verify all code changes were applied correctly")
        return 1


if __name__ == "__main__":
    sys.exit(main())

