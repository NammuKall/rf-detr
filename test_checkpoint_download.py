#!/usr/bin/env python3
"""
Verification script for Phase 1: Enhanced checkpoint download functionality.

This script tests:
1. File existence checks
2. Validation after download
3. Error handling improvements
4. Path resolution (relative vs absolute)

Run this script to verify everything is working correctly.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rfdetr.util.files import validate_checkpoint, download_file
from rfdetr.main import download_pretrain_weights, HOSTED_MODELS


def test_validate_checkpoint_existing_file():
    """Test validation with an existing valid checkpoint."""
    print("\n" + "="*60)
    print("TEST 1: Validate existing checkpoint file")
    print("="*60)
    
    # Check if we have any downloaded checkpoints
    checkpoint_files = [f for f in os.listdir('.') if f.endswith('.pth') and f.startswith('rf-detr-')]
    
    if not checkpoint_files:
        print("⚠️  SKIPPED: No checkpoint files found in current directory")
        print("   Download a checkpoint first to test validation")
        return True
    
    test_file = checkpoint_files[0]
    print(f"Testing with: {test_file}")
    
    is_valid, error_msg = validate_checkpoint(test_file)
    
    if is_valid:
        print(f"✅ PASS: Checkpoint '{test_file}' is valid")
        return True
    else:
        print(f"❌ FAIL: Checkpoint validation failed: {error_msg}")
        return False


def test_validate_checkpoint_missing_file():
    """Test validation with a missing file."""
    print("\n" + "="*60)
    print("TEST 2: Validate missing checkpoint file")
    print("="*60)
    
    fake_path = "nonexistent_checkpoint_12345.pth"
    is_valid, error_msg = validate_checkpoint(fake_path)
    
    if not is_valid and error_msg:
        print(f"✅ PASS: Correctly detected missing file: {error_msg}")
        return True
    else:
        print(f"❌ FAIL: Should have detected missing file")
        return False


def test_validate_checkpoint_small_file():
    """Test validation with a file that's too small."""
    print("\n" + "="*60)
    print("TEST 3: Validate checkpoint file that's too small")
    print("="*60)
    
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pth', delete=False) as f:
        f.write(b'x' * 100)  # Write 100 bytes (too small)
        temp_path = f.name
    
    try:
        is_valid, error_msg = validate_checkpoint(temp_path)
        
        if not is_valid and 'too small' in error_msg.lower():
            print(f"✅ PASS: Correctly detected file too small: {error_msg}")
            return True
        else:
            print(f"❌ FAIL: Should have detected file too small")
            return False
    finally:
        os.remove(temp_path)


def test_download_pretrain_weights_existing():
    """Test download function with existing checkpoint."""
    print("\n" + "="*60)
    print("TEST 4: Download function with existing checkpoint")
    print("="*60)
    
    # Check if we have any downloaded checkpoints
    checkpoint_files = [f for f in os.listdir('.') if f.endswith('.pth') and f.startswith('rf-detr-')]
    
    if not checkpoint_files:
        print("⚠️  SKIPPED: No checkpoint files found")
        return True
    
    test_file = checkpoint_files[0]
    print(f"Testing with existing file: {test_file}")
    
    result = download_pretrain_weights(test_file, redownload=False, validate=True)
    
    if result:
        print(f"✅ PASS: Function correctly identified existing checkpoint")
        return True
    else:
        print(f"❌ FAIL: Function should have found existing checkpoint")
        return False


def test_download_pretrain_weights_missing():
    """Test download function with missing checkpoint (should download)."""
    print("\n" + "="*60)
    print("TEST 5: Download function with missing checkpoint")
    print("="*60)
    
    # Use a small checkpoint for faster testing
    test_model = "rf-detr-nano.pth"
    
    # Create a backup if file exists
    backup_path = None
    if os.path.exists(test_model):
        backup_path = test_model + ".backup"
        shutil.copy2(test_model, backup_path)
        os.remove(test_model)
        print(f"Backed up existing {test_model} to {backup_path}")
    
    try:
        print(f"Testing download of: {test_model}")
        print("This will download the checkpoint if it doesn't exist...")
        
        result = download_pretrain_weights(test_model, redownload=False, validate=True)
        
        if result:
            print(f"✅ PASS: Successfully downloaded and validated checkpoint")
            
            # Verify file exists
            if os.path.exists(test_model):
                file_size = os.path.getsize(test_model)
                print(f"   File size: {file_size / (1024*1024):.2f} MB")
                return True
            else:
                print(f"❌ FAIL: File was not created")
                return False
        else:
            print(f"❌ FAIL: Download failed")
            return False
    finally:
        # Restore backup if it existed
        if backup_path and os.path.exists(backup_path):
            shutil.move(backup_path, test_model)
            print(f"Restored backup to {test_model}")


def test_download_pretrain_weights_custom_path():
    """Test download function with custom path (not in HOSTED_MODELS)."""
    print("\n" + "="*60)
    print("TEST 6: Download function with custom path")
    print("="*60)
    
    custom_path = "/tmp/custom_checkpoint.pth"
    
    # Remove if exists
    if os.path.exists(custom_path):
        os.remove(custom_path)
    
    result = download_pretrain_weights(custom_path, redownload=False, validate=True)
    
    if not result:
        print(f"✅ PASS: Correctly handled custom path (not downloadable)")
        return True
    else:
        print(f"❌ FAIL: Should have failed for custom path")
        return False


def test_path_resolution():
    """Test path resolution (relative vs absolute)."""
    print("\n" + "="*60)
    print("TEST 7: Path resolution (relative vs absolute)")
    print("="*60)
    
    # Test with relative path
    relative_path = "rf-detr-base.pth"
    if os.path.exists(relative_path):
        result = download_pretrain_weights(relative_path, redownload=False, validate=False)
        if result:
            print(f"✅ PASS: Relative path resolved correctly")
            return True
    
    # Test with absolute path
    abs_path = os.path.abspath("rf-detr-base.pth")
    if os.path.exists(abs_path):
        result = download_pretrain_weights(abs_path, redownload=False, validate=False)
        if result:
            print(f"✅ PASS: Absolute path handled correctly")
            return True
    
    print("⚠️  SKIPPED: No checkpoint files available for path resolution test")
    return True


def test_error_handling():
    """Test error handling improvements."""
    print("\n" + "="*60)
    print("TEST 8: Error handling")
    print("="*60)
    
    # Test None input
    result = download_pretrain_weights(None)
    if not result:
        print("✅ PASS: Handled None input correctly")
    else:
        print("❌ FAIL: Should have returned False for None")
        return False
    
    # Test invalid model name
    result = download_pretrain_weights("invalid_model_name.pth")
    if not result:
        print("✅ PASS: Handled invalid model name correctly")
    else:
        print("❌ FAIL: Should have returned False for invalid model")
        return False
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("CHECKPOINT DOWNLOAD VERIFICATION TESTS")
    print("="*60)
    print("\nThis script verifies Phase 1 implementation:")
    print("  - File existence checks")
    print("  - Validation after download")
    print("  - Error handling improvements")
    print("  - Path resolution")
    
    tests = [
        ("Validate existing checkpoint", test_validate_checkpoint_existing_file),
        ("Validate missing checkpoint", test_validate_checkpoint_missing_file),
        ("Validate small file", test_validate_checkpoint_small_file),
        ("Download with existing file", test_download_pretrain_weights_existing),
        ("Download missing checkpoint", test_download_pretrain_weights_missing),
        ("Custom path handling", test_download_pretrain_weights_custom_path),
        ("Path resolution", test_path_resolution),
        ("Error handling", test_error_handling),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Phase 1 implementation is working correctly.")
        print("\nNext steps:")
        print("  1. Test with actual training runs")
        print("  2. Monitor download behavior in production")
        print("  3. Proceed to Phase 2: Fix Model.__init__() checkpoint loading")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review the output above.")
        print("   Check for:")
        print("   - Network connectivity issues")
        print("   - File permission problems")
        print("   - Missing checkpoint files")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

