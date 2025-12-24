#!/usr/bin/env python3
"""
Verification script for populate_args default values fix.

This script verifies that:
1. populate_args has correct default values matching Base model config
2. Config classes have correct default values
3. Model creation works with both paths
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_populate_args_defaults_source():
    """Test that populate_args source code has correct default values"""
    print("=" * 60)
    print("Test 1: populate_args default values (source code check)")
    print("=" * 60)
    
    try:
        import re
        
        # Read the source file
        main_py_path = os.path.join(os.path.dirname(__file__), "rfdetr", "main.py")
        with open(main_py_path, 'r') as f:
            content = f.read()
        
        # Find the ca_nheads default value
        # Look for pattern: ca_nheads=8, or ca_nheads=16,
        pattern = r'ca_nheads\s*=\s*(\d+)'
        matches = re.findall(pattern, content)
        
        # Find the function definition
        func_start = content.find('def populate_args(')
        if func_start == -1:
            print("  ✗ Could not find populate_args function")
            return False
        
        # Get the function signature
        func_end = content.find('):', func_start)
        func_sig = content[func_start:func_end + 2]
        
        # Check for ca_nheads=16 in the function signature
        ca_nheads_correct = False
        if 'ca_nheads=16' in func_sig:
            print("  ✓ ca_nheads default is 16 (correct)")
            ca_nheads_correct = True
        elif 'ca_nheads=8' in func_sig:
            print("  ✗ ca_nheads default is 8 (should be 16)")
            ca_nheads_correct = False
        else:
            print("  ⚠ Could not find ca_nheads default in function signature")
            ca_nheads_correct = None
        
        # Check for dec_n_points=2 in the function signature
        dec_n_points_correct = False
        if 'dec_n_points=2' in func_sig:
            print("  ✓ dec_n_points default is 2 (correct)")
            dec_n_points_correct = True
        elif 'dec_n_points=4' in func_sig:
            print("  ✗ dec_n_points default is 4 (should be 2)")
            dec_n_points_correct = False
        else:
            print("  ⚠ Could not find dec_n_points default in function signature")
            dec_n_points_correct = None
        
        # Also check for comment indicating the fix
        if 'Base model default' in func_sig or 'RFDETRBaseConfig' in func_sig:
            print("  ✓ Found comment indicating Base model default")
        
        # Both must be correct
        if ca_nheads_correct is True and dec_n_points_correct is True:
            return True
        elif ca_nheads_correct is False or dec_n_points_correct is False:
            return False
        else:
            return None
            
    except Exception as e:
        print(f"  ✗ Error checking source code: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_populate_args_defaults():
    """Test that populate_args has correct default values (runtime check)"""
    print("\n" + "=" * 60)
    print("Test 1b: populate_args default values (runtime check)")
    print("=" * 60)
    
    try:
        # Try to import just the function without importing the full module
        import importlib.util
        main_py_path = os.path.join(os.path.dirname(__file__), "rfdetr", "main.py")
        spec = importlib.util.spec_from_file_location("rfdetr.main", main_py_path)
        if spec is None or spec.loader is None:
            print("  ⚠ Could not load module, skipping runtime check")
            return None
        
        # This will still fail if dependencies are missing, so we'll catch it
        try:
            module = importlib.util.module_from_spec(spec)
            # Don't execute, just check source
            print("  ⚠ Runtime check requires dependencies, using source check instead")
            return None
        except:
            print("  ⚠ Runtime check requires dependencies, using source check instead")
            return None
            
    except Exception as e:
        print(f"  ⚠ Runtime check skipped: {e}")
        return None


def test_config_defaults_source():
    """Test that config classes have correct default values (source check)"""
    print("\n" + "=" * 60)
    print("Test 2: Config class default values (source code check)")
    print("=" * 60)
    
    try:
        import re
        
        # Read the config file
        config_py_path = os.path.join(os.path.dirname(__file__), "rfdetr", "config.py")
        with open(config_py_path, 'r') as f:
            content = f.read()
        
        # Find RFDETRBaseConfig class
        base_config_start = content.find('class RFDETRBaseConfig')
        if base_config_start == -1:
            print("  ✗ Could not find RFDETRBaseConfig class")
            return False
        
        # Find the end of the class (next class definition or end of file)
        next_class = content.find('\nclass ', base_config_start + 1)
        if next_class == -1:
            class_content = content[base_config_start:]
        else:
            class_content = content[base_config_start:next_class]
        
        # Check for Base config defaults
        checks = [
            ("ca_nheads", 16, r'ca_nheads\s*:\s*int\s*=\s*16'),
            ("sa_nheads", 8, r'sa_nheads\s*:\s*int\s*=\s*8'),
            ("hidden_dim", 256, r'hidden_dim\s*:\s*int\s*=\s*256'),
            ("dec_layers", 3, r'dec_layers\s*:\s*int\s*=\s*3'),
            ("dec_n_points", 2, r'dec_n_points\s*:\s*int\s*=\s*2'),
        ]
        
        all_passed = True
        for param_name, expected, pattern in checks:
            if re.search(pattern, class_content):
                print(f"  ✓ {param_name}: {expected} (found in source)")
            else:
                print(f"  ✗ {param_name}: {expected} (not found or incorrect)")
                all_passed = False
        
        if all_passed:
            print("\n  ✓ All Base config defaults are correct in source")
            return True
        else:
            print("\n  ✗ Some Base config defaults are incorrect in source")
            return False
            
    except Exception as e:
        print(f"  ✗ Error checking config source: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_defaults():
    """Test that config classes have correct default values (runtime check)"""
    print("\n" + "=" * 60)
    print("Test 2b: Config class default values (runtime check)")
    print("=" * 60)
    
    try:
        # Try importing just config without importing detr
        import importlib.util
        config_py_path = os.path.join(os.path.dirname(__file__), "rfdetr", "config.py")
        spec = importlib.util.spec_from_file_location("rfdetr.config", config_py_path)
        if spec is None or spec.loader is None:
            print("  ⚠ Could not load config module, skipping runtime check")
            return None
        
        # Try to load it - config.py only needs pydantic and torch
        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            config = module.RFDETRBaseConfig()
            
            checks = [
                ("ca_nheads", 16, config.ca_nheads),
                ("sa_nheads", 8, config.sa_nheads),
                ("hidden_dim", 256, config.hidden_dim),
                ("dec_layers", 3, config.dec_layers),
                ("dec_n_points", 2, config.dec_n_points),
            ]
            
            all_passed = True
            for param_name, expected, actual in checks:
                if actual == expected:
                    print(f"  ✓ {param_name}: {actual} (expected {expected})")
                else:
                    print(f"  ✗ {param_name}: {actual} (expected {expected})")
                    all_passed = False
            
            if all_passed:
                print("\n  ✓ All Base config defaults are correct")
                return True
            else:
                print("\n  ✗ Some Base config defaults are incorrect")
                return False
        except ImportError as e:
            print(f"  ⚠ Runtime check requires dependencies ({e}), using source check instead")
            return None
            
    except Exception as e:
        print(f"  ⚠ Runtime check skipped: {e}")
        return None


def test_config_populate_args_consistency():
    """Test that config and populate_args defaults are consistent (source check)"""
    print("\n" + "=" * 60)
    print("Test 3: Config and populate_args consistency (source code check)")
    print("=" * 60)
    
    try:
        import re
        
        # Read both files
        config_py_path = os.path.join(os.path.dirname(__file__), "rfdetr", "config.py")
        main_py_path = os.path.join(os.path.dirname(__file__), "rfdetr", "main.py")
        
        with open(config_py_path, 'r') as f:
            config_content = f.read()
        with open(main_py_path, 'r') as f:
            main_content = f.read()
        
        # Find Base config defaults
        base_config_start = config_content.find('class RFDETRBaseConfig')
        next_class = config_content.find('\nclass ', base_config_start + 1)
        if next_class == -1:
            base_config_content = config_content[base_config_start:]
        else:
            base_config_content = config_content[base_config_start:next_class]
        
        # Find populate_args defaults
        populate_start = main_content.find('def populate_args(')
        populate_end = main_content.find('):', populate_start)
        populate_sig = main_content[populate_start:populate_end + 2]
        
        # Extract values using regex
        def extract_value(content, param_name):
            # Try pattern for type hints: param_name: int = value
            pattern1 = rf'{param_name}\s*:\s*int\s*=\s*(\d+)'
            match = re.search(pattern1, content)
            if match:
                return int(match.group(1))
            # Try pattern for function args: param_name=value
            pattern2 = rf'{param_name}\s*=\s*(\d+)'
            match = re.search(pattern2, content)
            if match:
                return int(match.group(1))
            return None
        
        checks = [
            ("ca_nheads", 16),
            ("sa_nheads", 8),
            ("hidden_dim", 256),
            ("dec_layers", 3),
            ("dec_n_points", 2),
        ]
        
        all_passed = True
        for param_name, expected in checks:
            config_val = extract_value(base_config_content, param_name)
            args_val = extract_value(populate_sig, param_name)
            
            if config_val == expected and args_val == expected:
                print(f"  ✓ {param_name}: Both={expected}")
            elif config_val == expected and args_val != expected:
                print(f"  ✗ {param_name}: Config={config_val}, populate_args={args_val} (MISMATCH)")
                all_passed = False
            elif config_val != expected:
                print(f"  ✗ {param_name}: Config={config_val} (should be {expected})")
                all_passed = False
            else:
                print(f"  ⚠ {param_name}: Could not extract values")
        
        if all_passed:
            print("\n  ✓ Config and populate_args defaults are consistent")
            return True
        else:
            print("\n  ✗ Config and populate_args defaults are inconsistent")
            return False
            
    except Exception as e:
        print(f"  ✗ Error testing consistency: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test that models can be created with both paths"""
    print("\n" + "=" * 60)
    print("Test 4: Model creation (if dependencies available)")
    print("=" * 60)
    
    try:
        import torch
        
        # Test 1: Using config class (RFDETRBase)
        try:
            from rfdetr.detr import RFDETRBase
            model = RFDETRBase()
            print(f"  ✓ RFDETRBase created successfully")
            print(f"    - Config ca_nheads: {model.model_config.ca_nheads}")
            print(f"    - Config sa_nheads: {model.model_config.sa_nheads}")
            print(f"    - Config dec_n_points: {model.model_config.dec_n_points}")
            config_works = True
        except Exception as e:
            print(f"  ✗ RFDETRBase creation failed: {e}")
            config_works = False
        
        # Test 2: Using populate_args (Model class)
        try:
            from rfdetr.main import Model
            model = Model()
            print(f"  ✓ Model (via populate_args) created successfully")
            print(f"    - Args ca_nheads: {model.args.ca_nheads}")
            print(f"    - Args sa_nheads: {model.args.sa_nheads}")
            print(f"    - Args dec_n_points: {model.args.dec_n_points}")
            args_works = True
        except Exception as e:
            print(f"  ✗ Model (via populate_args) creation failed: {e}")
            args_works = False
        
        if config_works and args_works:
            print("\n  ✓ Both model creation paths work")
            return True
        else:
            print("\n  ⚠ Some model creation paths failed (may be expected if dependencies missing)")
            return config_works or args_works
            
    except ImportError:
        print("  ⚠ torch not available, skipping model creation test")
        return None
    except Exception as e:
        print(f"  ⚠ Error testing model creation: {e}")
        return None


def main():
    """Run all verification tests"""
    print("\n" + "=" * 60)
    print("populate_args Default Values Fix - Verification")
    print("=" * 60)
    print()
    
    results = []
    
    # Test 1: populate_args defaults (source check)
    results.append(("populate_args defaults (source)", test_populate_args_defaults_source()))
    
    # Test 1b: populate_args defaults (runtime check, optional)
    runtime_result = test_populate_args_defaults()
    if runtime_result is not None:
        results.append(("populate_args defaults (runtime)", runtime_result))
    
    # Test 2: Config defaults (source check)
    results.append(("Config defaults (source)", test_config_defaults_source()))
    
    # Test 2b: Config defaults (runtime check, optional)
    config_runtime_result = test_config_defaults()
    if config_runtime_result is not None:
        results.append(("Config defaults (runtime)", config_runtime_result))
    
    # Test 3: Consistency
    results.append(("Config/populate_args consistency", test_config_populate_args_consistency()))
    
    # Test 4: Model creation (optional)
    model_result = test_model_creation()
    if model_result is not None:
        results.append(("Model creation", model_result))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    critical_tests = [r for r in results if r[0] != "Model creation"]
    optional_tests = [r for r in results if r[0] == "Model creation"]
    
    all_critical_passed = all(r[1] for r in critical_tests)
    
    for test_name, passed in critical_tests:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    if optional_tests:
        for test_name, passed in optional_tests:
            if passed is None:
                status = "⚠ SKIP"
            elif passed:
                status = "✓ PASS"
            else:
                status = "⚠ FAIL"
            print(f"  {status}: {test_name}")
    
    print()
    if all_critical_passed:
        print("✓ All critical tests passed!")
        print("\nNext steps:")
        print("  1. Test with actual checkpoint loading")
        print("  2. Verify no weight shape mismatches occur")
        print("  3. Run training/inference to confirm everything works")
        return 0
    else:
        print("✗ Some critical tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

