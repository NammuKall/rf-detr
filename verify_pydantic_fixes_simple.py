#!/usr/bin/env python3
"""
Simple verification script for Pydantic v2 API compatibility fixes.

This script only tests the code changes without requiring all dependencies.
It verifies:
1. Code changes were applied correctly (dict() → model_dump())
2. Validator improvements were made

Run this script even if dependencies are not installed.
"""

import sys
import re
from pathlib import Path

def check_code_changes():
    """Check that code changes were applied correctly"""
    print("\n" + "="*60)
    print("CODE CHANGES VERIFICATION")
    print("="*60)
    
    project_root = Path(__file__).parent
    issues = []
    
    # Check rfdetr/detr.py
    print("\n1. Checking rfdetr/detr.py...")
    detr_file = project_root / "rfdetr" / "detr.py"
    if not detr_file.exists():
        print("  ✗ File not found: rfdetr/detr.py")
        return False
    
    with open(detr_file, 'r') as f:
        content = f.read()
    
    # Check for dict() calls (should be replaced)
    dict_calls = re.findall(r'\.dict\(\)', content)
    if dict_calls:
        print(f"  ✗ Found {len(dict_calls)} .dict() calls that should be replaced:")
        for match in dict_calls:
            # Find line numbers
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if '.dict()' in line:
                    print(f"    Line {i}: {line.strip()[:80]}")
        issues.append("dict() calls found")
    else:
        print("  ✓ No .dict() calls found (good - should use model_dump())")
    
    # Check for model_dump() calls (should be present)
    model_dump_calls = re.findall(r'\.model_dump\(\)', content)
    if model_dump_calls:
        print(f"  ✓ Found {len(model_dump_calls)} .model_dump() calls")
    else:
        print("  ⚠ No .model_dump() calls found")
        issues.append("model_dump() calls not found")
    
    # Check rfdetr/config.py
    print("\n2. Checking rfdetr/config.py...")
    config_file = project_root / "rfdetr" / "config.py"
    if not config_file.exists():
        print("  ✗ File not found: rfdetr/config.py")
        return False
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Check for improved validator pattern
    validator_pattern = r'if not isinstance\(data, dict\):'
    if re.search(validator_pattern, content):
        print("  ✓ Found improved validator pattern (handles empty kwargs)")
    else:
        print("  ⚠ Improved validator pattern not found")
        issues.append("Validator improvements not found")
    
    # Count validators
    validator_count = len(re.findall(r'@model_validator\(mode=\'before\'\)', content))
    print(f"  ✓ Found {validator_count} model validators")
    
    # Check for consistent improved_* field removal
    improved_removal = len(re.findall(r"data\.pop\('improved_.*', None\)", content))
    if improved_removal > 0:
        print(f"  ✓ Found {improved_removal} improved_* field removals (consistent cleanup)")
    else:
        print("  ⚠ No improved_* field removals found")
    
    return len(issues) == 0


def check_file_structure():
    """Check that verification files exist"""
    print("\n3. Checking verification files...")
    project_root = Path(__file__).parent
    
    files_to_check = [
        "verify_pydantic_fixes.py",
        "analysis/VERIFICATION_GUIDE.md",
        "analysis/FIX_SUMMARY.md"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} not found")
            all_exist = False
    
    return all_exist


def main():
    """Run simple verification"""
    print("\n" + "="*60)
    print("SIMPLE CODE VERIFICATION")
    print("="*60)
    print("\nThis script verifies code changes without requiring dependencies.")
    print("For full verification, install dependencies and run verify_pydantic_fixes.py\n")
    
    results = []
    
    results.append(("Code Changes", check_code_changes()))
    results.append(("Verification Files", check_file_structure()))
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ Code changes verified!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install pydantic numpy torch")
        print("  2. Run full verification: python3 verify_pydantic_fixes.py")
        print("  3. Test with actual model creation")
        return 0
    else:
        print(f"\n⚠️  {total - passed} check(s) failed.")
        print("Please review the output above and ensure all code changes were applied.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

