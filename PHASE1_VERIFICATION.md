# Phase 1 Verification Guide

## What Was Implemented

Phase 1 enhances the checkpoint download functionality with:

1. **File Existence Checks**: Checks if checkpoint exists before attempting download
2. **Validation After Download**: Validates checkpoint structure after download
3. **Improved Error Handling**: Better error messages and retry logic
4. **Path Resolution**: Handles both relative and absolute paths correctly

## Files Modified

1. **`rfdetr/util/files.py`**:
   - Added `validate_checkpoint()` function
   - Enhanced `download_file()` with retry logic and atomic writes

2. **`rfdetr/main.py`**:
   - Enhanced `download_pretrain_weights()` function
   - Added validation, better error handling, and path resolution

## How to Verify Everything Works

### Option 1: Run Automated Test Script

```bash
# Activate virtual environment
source venv/bin/activate

# Run verification script
python test_checkpoint_download.py
```

The script will run 8 tests covering:
- ✅ Validation of existing checkpoints
- ✅ Detection of missing files
- ✅ Detection of corrupted files
- ✅ Download functionality
- ✅ Path resolution
- ✅ Error handling

### Option 2: Manual Testing

#### Test 1: Validate Existing Checkpoint
```python
from rfdetr.util.files import validate_checkpoint

# Test with an existing checkpoint
is_valid, error = validate_checkpoint("rf-detr-base.pth")
print(f"Valid: {is_valid}, Error: {error}")
# Expected: Valid: True, Error: None
```

#### Test 2: Download Missing Checkpoint
```python
from rfdetr.main import download_pretrain_weights
import os

# Remove checkpoint if exists (backup first!)
if os.path.exists("rf-detr-nano.pth"):
    os.rename("rf-detr-nano.pth", "rf-detr-nano.pth.backup")

# Try to download
result = download_pretrain_weights("rf-detr-nano.pth", validate=True)
print(f"Download successful: {result}")
# Expected: Download successful: True

# Verify file exists and is valid
assert os.path.exists("rf-detr-nano.pth"), "File should exist"
```

#### Test 3: Test with Existing Checkpoint
```python
from rfdetr.main import download_pretrain_weights

# Should skip download if file exists and is valid
result = download_pretrain_weights("rf-detr-base.pth", redownload=False, validate=True)
print(f"Result: {result}")
# Expected: Result: True (without downloading)
```

#### Test 4: Test Error Handling
```python
from rfdetr.main import download_pretrain_weights

# Test None input
result = download_pretrain_weights(None)
assert result == False, "Should return False for None"

# Test invalid model name
result = download_pretrain_weights("invalid_model.pth")
assert result == False, "Should return False for invalid model"
```

### Option 3: Integration Test with Training

Test the enhanced download function in a real training scenario:

```python
from rfdetr import RFDETRBase
import os

# Remove checkpoint to test download
if os.path.exists("rf-detr-base.pth"):
    os.rename("rf-detr-base.pth", "rf-detr-base.pth.backup")

try:
    # This should automatically download the checkpoint
    model = RFDETRBase()
    print("✅ Model initialized successfully - checkpoint downloaded automatically")
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    # Restore backup
    if os.path.exists("rf-detr-base.pth.backup"):
        os.rename("rf-detr-base.pth.backup", "rf-detr-base.pth")
```

## Expected Behavior

### ✅ Success Scenarios

1. **Existing Valid Checkpoint**:
   - Function returns `True` immediately
   - No download occurs
   - Logs: "Checkpoint already exists and is valid"

2. **Missing Checkpoint (in HOSTED_MODELS)**:
   - Downloads checkpoint automatically
   - Validates after download
   - Returns `True` on success
   - Logs: "Successfully downloaded and validated checkpoint"

3. **Corrupted Checkpoint**:
   - Detects corruption during validation
   - Re-downloads automatically
   - Returns `True` after successful re-download

### ❌ Failure Scenarios

1. **Custom Path Not in HOSTED_MODELS**:
   - Returns `False`
   - Logs: "Checkpoint not found and not in HOSTED_MODELS"
   - Lists available hosted models

2. **Network Failure**:
   - Retries up to 3 times
   - Returns `False` if all retries fail
   - Logs: "Failed to download checkpoint after 3 attempts"

3. **Invalid Checkpoint Format**:
   - Returns `False` after validation
   - Logs: "Downloaded checkpoint failed validation"
   - Removes corrupted file

## Verification Checklist

- [ ] Run `test_checkpoint_download.py` - all tests pass
- [ ] Test with existing checkpoint - no unnecessary download
- [ ] Test with missing checkpoint - downloads automatically
- [ ] Test with corrupted checkpoint - re-downloads
- [ ] Test with custom path - fails gracefully
- [ ] Test with None/invalid input - handles errors
- [ ] Check logs for clear error messages
- [ ] Verify file permissions are correct
- [ ] Test from different working directories

## What to Look For

### ✅ Good Signs

- Clear, informative log messages
- No unnecessary downloads when checkpoint exists
- Automatic validation catches corrupted files
- Proper error messages guide users
- Files are created atomically (no partial downloads)

### ⚠️ Warning Signs

- Silent failures (no error messages)
- Unnecessary re-downloads of valid checkpoints
- Corrupted files not detected
- Unclear error messages
- Partial/corrupted downloads left on disk

## Troubleshooting

### Issue: Tests fail with "No checkpoint files found"
**Solution**: Download at least one checkpoint first:
```bash
python -c "from rfdetr.main import download_pretrain_weights; download_pretrain_weights('rf-detr-base.pth')"
```

### Issue: Permission errors
**Solution**: Check file permissions:
```bash
ls -la *.pth
chmod 644 *.pth  # If needed
```

### Issue: Network timeouts
**Solution**: Check internet connection and firewall settings. The function retries automatically.

### Issue: Validation fails on valid checkpoint
**Solution**: Check if checkpoint format matches expected structure:
```python
import torch
ckpt = torch.load("rf-detr-base.pth", map_location='cpu', weights_only=False)
print("Keys:", list(ckpt.keys()))
assert 'model' in ckpt, "Missing 'model' key"
```

## Next Steps

After verifying Phase 1 works correctly:

1. **Proceed to Phase 2**: Fix `Model.__init__()` to download before loading
2. **Monitor Production**: Watch for any edge cases in real usage
3. **Collect Feedback**: Note any user-reported issues
4. **Iterate**: Make improvements based on findings

## Further Changes Needed?

### If Everything Works ✅
- Proceed to Phase 2 implementation
- No changes needed to Phase 1

### If Issues Found ⚠️

**Common Issues and Fixes**:

1. **Validation too strict**: Adjust `required_keys` parameter
2. **Path resolution issues**: Check `os.path.abspath()` behavior
3. **Download failures**: Increase retry count or timeout
4. **Permission errors**: Add better permission checking

**How to Report Issues**:
1. Note the exact error message
2. Check logs for detailed error information
3. Test with minimal example
4. Document the expected vs actual behavior

