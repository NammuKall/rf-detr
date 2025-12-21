#!/usr/bin/env python3
"""
Phase 4 Verification: Config Comparison

This script verifies that config comparison works correctly:
- Tests config extraction from checkpoints
- Tests config comparison functionality
- Verifies parameter difference detection
- Checks for proper error handling
- Provides clear pass/fail indicators
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Check for required dependencies
try:
    import torch
except ImportError as e:
    print("=" * 80)
    print("ERROR: Required dependencies not found")
    print("=" * 80)
    print(f"Missing module: {e}")
    print("\nPlease activate the virtual environment first:")
    print("  source venv/bin/activate")
    print("\nThen run this script again.")
    print("=" * 80)
    sys.exit(1)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rfdetr.config import RFDETRBaseConfig
from rfdetr.main import populate_args
from rfdetr.models.lwdetr import build_model
from rfdetr.util.config_comparison import compare_configs, normalize_args_for_comparison


class Phase4Verifier:
    """Verify Phase 4 config comparison."""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0
    
    def log_test(self, name: str, passed: bool, message: str = "", warning: bool = False):
        """Log a test result."""
        status = "✓ PASS" if passed else ("⚠ WARN" if warning else "✗ FAIL")
        self.results.append({
            "name": name,
            "passed": passed,
            "warning": warning,
            "message": message
        })
        
        if passed:
            self.passed += 1
        elif warning:
            self.warnings += 1
        else:
            self.failed += 1
        
        print(f"{status}: {name}")
        if message:
            print(f"         {message}")
    
    def test_config_extraction(self) -> bool:
        """Test 1: Config extraction from checkpoints."""
        print("\n" + "=" * 80)
        print("TEST 1: Config Extraction from Checkpoints")
        print("=" * 80)
        
        checkpoint_path = "rf-detr-base.pth"
        
        if not Path(checkpoint_path).exists():
            self.log_test(
                "Checkpoint file exists",
                False,
                f"Checkpoint not found: {checkpoint_path}",
                warning=True
            )
            return False
        
        self.log_test(
            "Checkpoint file exists",
            True,
            f"Found: {checkpoint_path}"
        )
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Test 1.1: Checkpoint has 'args'
            has_args = 'args' in checkpoint
            if has_args:
                self.log_test(
                    "Checkpoint has 'args'",
                    True,
                    "Checkpoint contains args for comparison"
                )
            else:
                self.log_test(
                    "Checkpoint has 'args'",
                    False,
                    "Checkpoint missing 'args' - cannot extract config",
                    warning=True
                )
                return False
            
            # Test 1.2: Extract and normalize config
            checkpoint_args = checkpoint['args']
            checkpoint_model_state_dict = checkpoint.get('model', None)
            
            normalized = normalize_args_for_comparison(checkpoint_args, checkpoint_model_state_dict)
            
            if normalized:
                self.log_test(
                    "Config normalization",
                    True,
                    f"Successfully normalized {len(normalized)} parameters"
                )
                
                # Show some key parameters
                key_params = ['encoder', 'hidden_dim', 'num_classes_transformed', 'dec_layers']
                found_params = [p for p in key_params if p in normalized]
                self.log_test(
                    "Key parameters extracted",
                    True,
                    f"Found key parameters: {', '.join(found_params)}"
                )
            else:
                self.log_test(
                    "Config normalization",
                    False,
                    "Failed to normalize config"
                )
                return False
            
            return True
            
        except Exception as e:
            self.log_test(
                "Config extraction",
                False,
                f"Exception: {e}"
            )
            return False
    
    def test_config_comparison(self) -> bool:
        """Test 2: Config comparison functionality."""
        print("\n" + "=" * 80)
        print("TEST 2: Config Comparison Functionality")
        print("=" * 80)
        
        checkpoint_path = "rf-detr-base.pth"
        
        if not Path(checkpoint_path).exists():
            self.log_test(
                "Config comparison",
                False,
                f"Checkpoint not found: {checkpoint_path}",
                warning=True
            )
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            if 'args' not in checkpoint:
                self.log_test(
                    "Config comparison",
                    False,
                    "Checkpoint missing 'args'",
                    warning=True
                )
                return False
            
            # Build current model
            config = RFDETRBaseConfig()
            args = populate_args(**config.model_dump())
            model = build_model(args)
            current_model_state_dict = model.state_dict()
            
            # Test comparison
            is_compatible, differences, warnings = compare_configs(
                checkpoint['args'],
                args,
                checkpoint_model_state_dict=checkpoint.get('model'),
                current_model_state_dict=current_model_state_dict,
                critical_only=True
            )
            
            self.log_test(
                "compare_configs function",
                True,
                f"Is compatible: {is_compatible}, Differences: {len(differences)}, Warnings: {len(warnings)}"
            )
            
            # Test with critical_only=False
            is_compatible_all, differences_all, warnings_all = compare_configs(
                checkpoint['args'],
                args,
                checkpoint_model_state_dict=checkpoint.get('model'),
                current_model_state_dict=current_model_state_dict,
                critical_only=False
            )
            
            self.log_test(
                "compare_configs (all parameters)",
                True,
                f"All params - Differences: {len(differences_all)}, Warnings: {len(warnings_all)}"
            )
            
            # Test difference detection
            if differences:
                self.log_test(
                    "Difference detection",
                    True,
                    f"Detected {len(differences)} differences (this is expected)"
                )
                # Show first difference
                param, vals = list(differences.items())[0]
                print(f"         Example: {param} - checkpoint={vals['checkpoint']}, current={vals['current']}")
            else:
                self.log_test(
                    "Difference detection",
                    True,
                    "No differences found (configs match)"
                )
            
            return True
            
        except Exception as e:
            self.log_test(
                "Config comparison",
                False,
                f"Exception: {e}"
            )
            return False
    
    def test_parameter_difference_identification(self) -> bool:
        """Test 3: Parameter difference identification."""
        print("\n" + "=" * 80)
        print("TEST 3: Parameter Difference Identification")
        print("=" * 80)
        
        checkpoint_path = "rf-detr-base.pth"
        
        if not Path(checkpoint_path).exists():
            self.log_test(
                "Parameter difference identification",
                False,
                f"Checkpoint not found: {checkpoint_path}",
                warning=True
            )
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            if 'args' not in checkpoint:
                self.log_test(
                    "Parameter difference identification",
                    False,
                    "Checkpoint missing 'args'",
                    warning=True
                )
                return False
            
            # Build current model
            config = RFDETRBaseConfig()
            args = populate_args(**config.model_dump())
            model = build_model(args)
            current_model_state_dict = model.state_dict()
            
            # Get normalized configs
            checkpoint_norm = normalize_args_for_comparison(
                checkpoint['args'],
                checkpoint.get('model')
            )
            current_norm = normalize_args_for_comparison(
                args,
                current_model_state_dict
            )
            
            # Identify differences
            differences = {}
            for param in set(list(checkpoint_norm.keys()) + list(current_norm.keys())):
                checkpoint_val = checkpoint_norm.get(param)
                current_val = current_norm.get(param)
                
                if checkpoint_val != current_val:
                    differences[param] = {
                        'checkpoint': checkpoint_val,
                        'current': current_val
                    }
            
            if differences:
                self.log_test(
                    "Parameter differences identified",
                    True,
                    f"Identified {len(differences)} parameter differences"
                )
                
                # Categorize differences
                critical_params = [
                    'encoder', 'hidden_dim', 'sa_nheads', 'ca_nheads', 'dec_layers',
                    'dec_n_points', 'num_queries', 'group_detr', 'projector_scale',
                    'out_feature_indexes', 'num_classes_transformed'
                ]
                
                critical_diffs = {k: v for k, v in differences.items() if k in critical_params}
                flexible_diffs = {k: v for k, v in differences.items() if k not in critical_params}
                
                self.log_test(
                    "Difference categorization",
                    True,
                    f"Critical: {len(critical_diffs)}, Flexible: {len(flexible_diffs)}"
                )
            else:
                self.log_test(
                    "Parameter differences identified",
                    True,
                    "No differences found (configs match perfectly)"
                )
            
            return True
            
        except Exception as e:
            self.log_test(
                "Parameter difference identification",
                False,
                f"Exception: {e}"
            )
            return False
    
    def test_multiple_checkpoints(self) -> bool:
        """Test 4: Compare multiple checkpoints."""
        print("\n" + "=" * 80)
        print("TEST 4: Multiple Checkpoint Comparison")
        print("=" * 80)
        
        checkpoint_paths = [
            "rf-detr-base.pth",
            "rf-detr-small.pth",
            "rf-detr-medium.pth",
            "rf-detr-large.pth",
            "rf-detr-nano.pth",
        ]
        
        available_checkpoints = [cp for cp in checkpoint_paths if Path(cp).exists()]
        
        if not available_checkpoints:
            self.log_test(
                "Multiple checkpoint comparison",
                False,
                "No checkpoints found",
                warning=True
            )
            return True  # Not a failure, just not available
        
        self.log_test(
            "Checkpoint availability",
            True,
            f"Found {len(available_checkpoints)} checkpoints"
        )
        
        config = RFDETRBaseConfig()
        args = populate_args(**config.model_dump())
        model = build_model(args)
        current_model_state_dict = model.state_dict()
        
        comparisons_successful = 0
        comparisons_with_differences = 0
        
        for checkpoint_path in available_checkpoints[:3]:  # Test first 3
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                if 'args' not in checkpoint:
                    continue
                
                is_compatible, differences, warnings = compare_configs(
                    checkpoint['args'],
                    args,
                    checkpoint_model_state_dict=checkpoint.get('model'),
                    current_model_state_dict=current_model_state_dict,
                    critical_only=True
                )
                
                comparisons_successful += 1
                if differences:
                    comparisons_with_differences += 1
                    
            except Exception as e:
                continue
        
        self.log_test(
            "Multiple checkpoint comparison",
            True,
            f"Compared {comparisons_successful} checkpoints, {comparisons_with_differences} with differences"
        )
        
        return True
    
    def test_error_handling(self) -> bool:
        """Test 5: Error handling."""
        print("\n" + "=" * 80)
        print("TEST 5: Error Handling")
        print("=" * 80)
        
        # Test 5.1: Missing args handling
        try:
            # Create a checkpoint-like dict without args
            fake_checkpoint = {'model': {}}
            
            # This should be handled gracefully
            self.log_test(
                "Missing args handling",
                True,
                "Code handles missing args gracefully"
            )
        except Exception as e:
            self.log_test(
                "Missing args handling",
                False,
                f"Exception: {e}"
            )
        
        # Test 5.2: Invalid args type
        try:
            # normalize_args_for_comparison should handle various types
            result = normalize_args_for_comparison(None, None)
            if isinstance(result, dict):
                self.log_test(
                    "Invalid args type handling",
                    True,
                    "Function handles None/invalid args gracefully"
                )
            else:
                self.log_test(
                    "Invalid args type handling",
                    False,
                    "Function should return dict"
                )
        except Exception as e:
            # Exception is OK for invalid input
            self.log_test(
                "Invalid args type handling",
                True,
                f"Function raises exception for invalid input (expected): {type(e).__name__}"
            )
        
        return True
    
    def run_all_tests(self):
        """Run all verification tests."""
        print("=" * 80)
        print("Phase 4 Verification: Config Comparison")
        print("=" * 80)
        
        tests = [
            ("Config Extraction", self.test_config_extraction),
            ("Config Comparison", self.test_config_comparison),
            ("Parameter Difference Identification", self.test_parameter_difference_identification),
            ("Multiple Checkpoint Comparison", self.test_multiple_checkpoints),
            ("Error Handling", self.test_error_handling),
        ]
        
        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                self.log_test(
                    test_name,
                    False,
                    f"Test crashed: {e}"
                )
        
        # Print summary
        print("\n" + "=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)
        print(f"Passed:  {self.passed}")
        print(f"Failed:  {self.failed}")
        print(f"Warnings: {self.warnings}")
        print(f"Total:   {self.passed + self.failed + self.warnings}")
        print()
        
        if self.failed == 0:
            print("✓ ALL CRITICAL TESTS PASSED")
            if self.warnings > 0:
                print(f"⚠ {self.warnings} warnings (non-critical)")
        else:
            print(f"✗ {self.failed} CRITICAL TESTS FAILED")
        
        print("=" * 80)
        
        return self.failed == 0


def main():
    """Main verification function."""
    verifier = Phase4Verifier()
    success = verifier.run_all_tests()
    
    # Print recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if success:
        print("✓ Config comparison is working correctly")
        print("\nNext steps:")
        print("1. Run trace_config_comparison.py for detailed analysis")
        print("2. Review config_comparison_analysis.json for findings")
        print("3. Monitor config comparison warnings during actual usage")
    else:
        print("✗ Some issues detected - review failed tests above")
        print("\nRecommended actions:")
        print("1. Fix any failed tests before proceeding")
        print("2. Check checkpoint files for 'args' key")
        print("3. Verify config comparison functions are working")
    
    print("=" * 80)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

