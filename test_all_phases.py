#!/usr/bin/env python3
"""
Run all phase tests and provide a summary.

This script runs all phase test scripts and provides a comprehensive summary.
"""
import sys
import subprocess
import os
from pathlib import Path

def run_test_script(script_name, phase_name):
    """Run a test script and return the result."""
    print(f"\n{'=' * 70}")
    print(f"Running {phase_name}")
    print(f"{'=' * 70}")
    
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"⚠️  Test script not found: {script_name}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"❌ Test timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def main():
    print("=" * 70)
    print("RF-DETR Checkpoint Download - All Phases Test Suite")
    print("=" * 70)
    
    phases = [
        ("test_phase2.py", "Phase 2: Model Initialization Fixes"),
        ("test_phase3.py", "Phase 3: Resume Checkpoint Download Support"),
        ("test_phase4.py", "Phase 4: RFDETR Class Improvements"),
        ("test_phase5.py", "Phase 5: Enhanced Validation Utility"),
    ]
    
    results = []
    
    for script_name, phase_name in phases:
        success = run_test_script(script_name, phase_name)
        results.append((phase_name, success))
    
    # Print summary
    print("\n" + "=" * 70)
    print("Test Suite Summary")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for phase_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {phase_name}")
    
    print(f"\nTotal: {passed}/{total} phases passed")
    
    if passed == total:
        print("\n🎉 All phases passed! Checkpoint download functionality is working correctly.")
        print("\nAll implemented features:")
        print("  ✅ Phase 2: Download before loading, validation, better errors")
        print("  ✅ Phase 3: URL-based resume checkpoints, caching")
        print("  ✅ Phase 4: None handling, input validation, better logging")
        print("  ✅ Phase 5: Enhanced structure validation, key verification")
        return 0
    else:
        print(f"\n⚠️  {total - passed} phase(s) failed. Please review the failures above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

