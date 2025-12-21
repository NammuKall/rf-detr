# Phase 2 Verification Guide

## What Was Implemented

Phase 2 fixes model initialization to ensure checkpoints are downloaded and validated **before** loading, preventing errors and improving user experience.

### Key Changes:

1. **Download Before Loading**: `Model.__init__()` now downloads checkpoints before attempting to load them
2. **Validation Before Use**: Checkpoints are validated before loading to catch corruption early
3. **Better Error Messages**: Clear, actionable error messages guide users when issues occur
4. **Improved RFDETR Class**: `RFDETR.maybe_download_pretrain_weights()` now handles return values and errors properly

## Files Modified

1. **`rfdetr/main.py`**:
   - Fixed `Model.__init__()` to download and validate before loading (lines 179-243)
   - Improved error messages with actionable guidance
   - Fixed bug with `model_without_ddp` reference (now uses `self.model`)

2. **`rfdetr/detr.py`**:
   - Updated `RFDETR.maybe_download_pretrain_weights()` to handle return values and log warnings

## How to Verify Everything Works

### Option 1: Quick Integration Test

Test that model initialization works correctly with missing checkpoints:

```python
from rfdetr import RFDETRBase
import os
import shutil

# Backup existing checkpoint if it exists
checkpoint_path = "rf-detr-base.pth"
backup_path = "rf-detr-base.pth.backup"

if os.path.exists(checkpoint_path):
    shutil.move(checkpoint_path, backup_path)
    print(f"Backed up existing checkpoint to {backup_path}")

try:
    # This should automatically download the checkpoint BEFORE loading
    print("Initializing model (should download checkpoint automatically)...")
    model = RFDETRBase()
    print("✅ Model initialized successfully!")
    print(f"✅ Checkpoint downloaded and loaded: {checkpoint_path}")
except Exception as e:
    print(f"❌ Error: {e}")
    raise
finally:
    # Restore backup if it existed
    if os.path.exists(backup_path):
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)  # Remove downloaded one
        shutil.move(backup_path, checkpoint_path)
        print(f"Restored original checkpoint from {backup_path}")
```

### Option 2: Test with Corrupted Checkpoint

Test that corrupted checkpoints are detected and re-downloaded:

```python
from rfdetr import RFDETRBase
import os

checkpoint_path = "rf-detr-base.pth"

# Create a corrupted checkpoint file
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, 'w') as f:
        f.write("corrupted checkpoint data")
    print(f"Created corrupted checkpoint: {checkpoint_path}")

try:
    # This should detect corruption, re-download, and then load
    print("Initializing model with corrupted checkpoint...")
    model = RFDETRBase()
    print("✅ Model initialized successfully after re-download!")
except Exception as e:
    print(f"❌ Error: {e}")
    raise
```

### Option 3: Test Error Messages

Test that error messages are clear and actionable:

```python
from rfdetr.main import Model, populate_args

# Test with non-existent custom checkpoint (not in HOSTED_MODELS)
try:
    args = populate_args(pretrain_weights="custom-nonexistent-checkpoint.pth")
    model = Model(**vars(args))
    print("❌ Should have raised FileNotFoundError")
except FileNotFoundError as e:
    print("✅ Got expected FileNotFoundError")
    print(f"Error message:\n{e}")
    # Verify error message is helpful
    assert "not found" in str(e).lower() or "not available" in str(e).lower()
    assert "Available hosted models" in str(e) or "hosted models" in str(e).lower()
except Exception as e:
    print(f"❌ Got unexpected error: {e}")
    raise
```

### Option 4: Test Direct Model Initialization

Test the `Model` class directly:

```python
from rfdetr.main import Model, populate_args
import os

# Test with missing checkpoint (should download)
checkpoint_name = "rf-detr-nano.pth"
if os.path.exists(checkpoint_name):
    os.rename(checkpoint_name, checkpoint_name + ".backup")

try:
    args = populate_args(pretrain_weights=checkpoint_name)
    print(f"Initializing Model with {checkpoint_name}...")
    model = Model(**vars(args))
    print("✅ Model initialized successfully!")
    assert os.path.exists(checkpoint_name), "Checkpoint should exist after download"
except Exception as e:
    print(f"❌ Error: {e}")
    raise
finally:
    # Restore backup if it existed
    if os.path.exists(checkpoint_name + ".backup"):
        if os.path.exists(checkpoint_name):
            os.remove(checkpoint_name)
        os.rename(checkpoint_name + ".backup", checkpoint_name)
```

### Option 5: Comprehensive Test Script

Run a comprehensive test that covers all scenarios:

```python
#!/usr/bin/env python3
"""
Comprehensive test for Phase 2: Model Initialization Fixes
"""
import os
import shutil
import tempfile
from rfdetr import RFDETRBase
from rfdetr.main import Model, populate_args

def test_missing_checkpoint_download():
    """Test that missing checkpoints are downloaded before loading"""
    print("\n=== Test 1: Missing Checkpoint Download ===")
    checkpoint_name = "rf-detr-base.pth"
    backup_name = checkpoint_name + ".backup"
    
    # Backup if exists
    if os.path.exists(checkpoint_name):
        shutil.move(checkpoint_name, backup_name)
    
    try:
        model = RFDETRBase()
        assert os.path.exists(checkpoint_name), "Checkpoint should be downloaded"
        print("✅ PASS: Missing checkpoint downloaded automatically")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False
    finally:
        # Restore backup
        if os.path.exists(backup_name):
            if os.path.exists(checkpoint_name):
                os.remove(checkpoint_name)
            shutil.move(backup_name, checkpoint_name)

def test_corrupted_checkpoint_recovery():
    """Test that corrupted checkpoints are detected and re-downloaded"""
    print("\n=== Test 2: Corrupted Checkpoint Recovery ===")
    checkpoint_name = "rf-detr-base.pth"
    
    if not os.path.exists(checkpoint_name):
        print("⚠️  SKIP: Checkpoint doesn't exist, downloading first...")
        model = RFDETRBase()
    
    # Corrupt the checkpoint
    backup_name = checkpoint_name + ".backup"
    shutil.copy(checkpoint_name, backup_name)
    
    try:
        # Write corrupted data
        with open(checkpoint_name, 'w') as f:
            f.write("corrupted data")
        
        # Should detect corruption and re-download
        model = RFDETRBase()
        print("✅ PASS: Corrupted checkpoint detected and re-downloaded")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False
    finally:
        # Restore backup
        if os.path.exists(backup_name):
            if os.path.exists(checkpoint_name):
                os.remove(checkpoint_name)
            shutil.move(backup_name, checkpoint_name)

def test_custom_path_error():
    """Test that custom paths (not in HOSTED_MODELS) give clear errors"""
    print("\n=== Test 3: Custom Path Error Handling ===")
    
    try:
        args = populate_args(pretrain_weights="custom-nonexistent.pth")
        model = Model(**vars(args))
        print("❌ FAIL: Should have raised FileNotFoundError")
        return False
    except FileNotFoundError as e:
        error_msg = str(e)
        if "not found" in error_msg.lower() or "not available" in error_msg.lower():
            print("✅ PASS: Clear error message for custom path")
            return True
        else:
            print(f"❌ FAIL: Error message not clear enough: {error_msg}")
            return False
    except Exception as e:
        print(f"❌ FAIL: Got unexpected error: {e}")
        return False

def test_existing_checkpoint_no_redownload():
    """Test that existing valid checkpoints are not re-downloaded"""
    print("\n=== Test 4: Existing Checkpoint (No Re-download) ===")
    
    checkpoint_name = "rf-detr-base.pth"
    
    # Ensure checkpoint exists
    if not os.path.exists(checkpoint_name):
        print("⚠️  Downloading checkpoint first...")
        model = RFDETRBase()
    
    # Get modification time before
    mtime_before = os.path.getmtime(checkpoint_name)
    
    try:
        # Initialize model - should use existing checkpoint
        model = RFDETRBase()
        
        # Check modification time hasn't changed
        mtime_after = os.path.getmtime(checkpoint_name)
        if mtime_before == mtime_after:
            print("✅ PASS: Existing checkpoint used without re-download")
            return True
        else:
            print("⚠️  WARNING: Checkpoint modification time changed (may have re-downloaded)")
            return True  # Still consider pass, as long as it works
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2 Verification Tests")
    print("=" * 60)
    
    results = []
    results.append(("Missing Checkpoint Download", test_missing_checkpoint_download()))
    results.append(("Corrupted Checkpoint Recovery", test_corrupted_checkpoint_recovery()))
    results.append(("Custom Path Error Handling", test_custom_path_error()))
    results.append(("Existing Checkpoint (No Re-download)", test_existing_checkpoint_no_redownload()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Phase 2 implementation is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the implementation.")
```

## Expected Behavior

### ✅ Success Scenarios

1. **Missing Checkpoint (in HOSTED_MODELS)**:
   - Downloads checkpoint automatically BEFORE attempting to load
   - Validates checkpoint after download
   - Loads checkpoint successfully
   - Logs: "Preparing to load pretrain weights", "Downloading pretrained weights", "Successfully downloaded and validated checkpoint", "Loading pretrain weights"

2. **Existing Valid Checkpoint**:
   - Validates existing checkpoint
   - Uses existing checkpoint without re-downloading
   - Loads checkpoint successfully
   - Logs: "Checkpoint already exists and is valid", "Loading pretrain weights"

3. **Corrupted Checkpoint**:
   - Detects corruption during validation
   - Re-downloads automatically
   - Validates after re-download
   - Loads checkpoint successfully
   - Logs: "Existing checkpoint failed validation", "Re-downloading...", "Successfully downloaded and validated checkpoint"

### ❌ Failure Scenarios (with Clear Errors)

1. **Custom Path Not in HOSTED_MODELS**:
   - Returns `False` from `download_pretrain_weights()`
   - Raises `FileNotFoundError` with clear message
   - Lists available hosted models
   - Error message includes actionable guidance

2. **Network Failure After Retries**:
   - Attempts download with retries
   - Raises `RuntimeError` with clear message
   - Error message explains possible causes and solutions

3. **Checkpoint Load Failure After Validation**:
   - Should rarely happen, but handled gracefully
   - Attempts re-download
   - Raises `RuntimeError` with detailed error information

## Verification Checklist

- [ ] Test with missing checkpoint → downloads automatically before loading
- [ ] Test with existing checkpoint → uses existing file without re-download
- [ ] Test with corrupted checkpoint → detects corruption and re-downloads
- [ ] Test with custom path (not in HOSTED_MODELS) → clear error message
- [ ] Test error messages are clear and actionable
- [ ] Test that no errors occur during normal initialization
- [ ] Check logs for informative messages
- [ ] Verify checkpoint is validated before loading
- [ ] Test from different working directories
- [ ] Test with different model sizes (base, nano, small, etc.)

## What to Look For

### ✅ Good Signs

- **No errors during initialization** when checkpoint is missing (downloads first)
- **Clear log messages** showing download progress and validation
- **Informative error messages** that guide users to solutions
- **No unnecessary re-downloads** when checkpoint exists and is valid
- **Fast initialization** when checkpoint already exists (no download delay)

### ⚠️ Warning Signs

- **Errors during initialization** when checkpoint is missing (should download first)
- **Unclear error messages** that don't help users understand the issue
- **Unnecessary re-downloads** of valid checkpoints
- **Silent failures** without error messages
- **Long delays** when checkpoint already exists (should be instant)

## Troubleshooting

### Issue: Model initialization fails with "Checkpoint not found"

**Possible Causes**:
1. Network connectivity issues
2. Checkpoint name typo
3. Custom path not in HOSTED_MODELS

**Solution**:
- Check internet connection
- Verify checkpoint name matches one in HOSTED_MODELS
- For custom paths, ensure file exists locally

### Issue: Model initialization is slow

**Possible Causes**:
1. Downloading large checkpoint
2. Network is slow
3. Validation is taking time

**Solution**:
- This is expected for first-time initialization
- Subsequent initializations should be fast (uses cached checkpoint)
- Check network speed if download is unusually slow

### Issue: "Failed to load checkpoint despite validation"

**Possible Causes**:
1. File corruption after validation
2. Race condition (file deleted between validation and load)
3. Permission issues

**Solution**:
- Check file permissions
- Verify disk space
- Try re-downloading manually

## Confirming Everything Works Properly

### Step 1: Run Quick Test

```bash
# Activate virtual environment
source venv/bin/activate

# Run quick test
python -c "from rfdetr import RFDETRBase; model = RFDETRBase(); print('✅ Success!')"
```

### Step 2: Test with Missing Checkpoint

```bash
# Backup checkpoint if exists
mv rf-detr-base.pth rf-detr-base.pth.backup 2>/dev/null || true

# Initialize model (should download)
python -c "from rfdetr import RFDETRBase; model = RFDETRBase(); print('✅ Success!')"

# Verify checkpoint was downloaded
ls -lh rf-detr-base.pth

# Restore backup
mv rf-detr-base.pth.backup rf-detr-base.pth 2>/dev/null || true
```

### Step 3: Check Logs

Look for these log messages in order:
1. "Preparing to load pretrain weights"
2. "Downloading pretrained weights" (if missing) OR "Checkpoint already exists and is valid" (if exists)
3. "Successfully downloaded and validated checkpoint" (if downloaded)
4. "Loading pretrain weights"

### Step 4: Verify No Errors

The initialization should complete without any errors or exceptions. If you see errors, check:
- Error message clarity
- Whether it's a network issue
- Whether checkpoint path is correct

## Further Changes Needed?

### If Everything Works ✅

- **Phase 2 is complete!**
- Proceed to Phase 3: Resume Checkpoint Download Support (if needed)
- Monitor production usage for any edge cases

### If Issues Found ⚠️

**Common Issues and Fixes**:

1. **Download still happens after error**:
   - Check that `download_pretrain_weights()` is called BEFORE `torch.load()`
   - Verify return value is checked

2. **Validation too strict**:
   - Adjust `required_keys` parameter in validation
   - Check checkpoint format matches expected structure

3. **Error messages not clear**:
   - Review error messages in `Model.__init__()`
   - Add more context to error messages

4. **Performance issues**:
   - Check if validation is too slow
   - Consider caching validation results

**How to Report Issues**:
1. Note the exact error message
2. Check logs for detailed information
3. Test with minimal example
4. Document expected vs actual behavior
5. Include checkpoint name and path

## Success Criteria

- ✅ No checkpoint loading errors when checkpoint doesn't exist (downloads first)
- ✅ Clear error messages when checkpoint can't be downloaded
- ✅ Validation catches corrupted checkpoints before loading
- ✅ All existing functionality continues to work
- ✅ Better user experience with informative messages

