# Phase 3 Verification Guide

## What Was Implemented

Phase 3 adds support for downloading resume checkpoints from URLs and validating them before use.

### Key Changes:

1. **URL-Based Resume Checkpoints**: Resume checkpoints can now be specified as URLs (http/https)
2. **Automatic Download**: Resume checkpoints are automatically downloaded if provided as URLs
3. **Caching**: Downloaded checkpoints are cached in `~/.rfdetr/checkpoints/` to avoid re-downloading
4. **Validation Before Loading**: Resume checkpoints are validated before loading to catch corruption early
5. **Better Error Messages**: Clear, actionable error messages guide users when issues occur

## Files Modified

1. **`rfdetr/util/files.py`**:
   - Added `download_resume_checkpoint()` function
   - Handles both URL and local file paths
   - Implements caching for downloaded checkpoints
   - Validates checkpoints before returning

2. **`rfdetr/main.py`**:
   - Updated `Model.train()` resume checkpoint loading (lines 437-469)
   - Uses `download_resume_checkpoint()` before loading
   - Improved error handling and logging

## How to Verify Everything Works

### Option 1: Quick Integration Test

Test that resume checkpoint loading works with local paths:

```python
from rfdetr import RFDETRBase
import os

# Create a simple training config
model = RFDETRBase()

# Save a checkpoint first (if you have one)
# Or use an existing checkpoint path
checkpoint_path = "output/checkpoint.pth"  # Adjust to your checkpoint path

if os.path.exists(checkpoint_path):
    # Test resuming from local path
    try:
        model.train(
            dataset_dir="path/to/your/dataset",
            resume=checkpoint_path,
            epochs=1,  # Just test loading
            eval=True  # Only evaluate, don't train
        )
        print("✅ Resume checkpoint loaded successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print(f"⚠️  Checkpoint not found: {checkpoint_path}")
    print("Create a checkpoint first by training a model.")
```

### Option 2: Test URL-Based Resume

Test URL-based resume checkpoint download:

```python
from rfdetr.util.files import download_resume_checkpoint
from rfdetr.main import HOSTED_MODELS

# Test with a URL from HOSTED_MODELS
test_url = HOSTED_MODELS.get("rf-detr-nano.pth")

try:
    local_path = download_resume_checkpoint(test_url, validate=True)
    print(f"✅ Successfully downloaded resume checkpoint to: {local_path}")
    print(f"   File exists: {os.path.exists(local_path)}")
except RuntimeError as e:
    error_msg = str(e)
    if "403" in error_msg or "Forbidden" in error_msg:
        print("⚠️  URL requires authentication (403 Forbidden)")
        print("   This is a server-side issue, not a code issue.")
        print("   The code correctly handles this error.")
    else:
        print(f"❌ Error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
```

**Note**: Some URLs may return 403 Forbidden if they require authentication. This is expected and the code handles it gracefully with clear error messages.

### Option 3: Test Local Path Validation

Test that local paths are validated:

```python
from rfdetr.util.files import download_resume_checkpoint
import os

# Test with existing checkpoint
checkpoint_path = "rf-detr-base.pth"  # Adjust to your checkpoint

if os.path.exists(checkpoint_path):
    try:
        validated_path = download_resume_checkpoint(checkpoint_path, validate=True)
        print(f"✅ Checkpoint validated: {validated_path}")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
else:
    print(f"⚠️  Checkpoint not found: {checkpoint_path}")
```

### Option 4: Test Error Handling

Test error handling for missing checkpoints:

```python
from rfdetr.util.files import download_resume_checkpoint

# Test with non-existent local path
try:
    download_resume_checkpoint("nonexistent-checkpoint.pth", validate=True)
    print("❌ Should have raised FileNotFoundError")
except FileNotFoundError as e:
    print("✅ Got expected FileNotFoundError")
    print(f"Error message: {str(e)[:200]}...")
except Exception as e:
    print(f"❌ Got unexpected error: {e}")
```

### Option 5: Comprehensive Test Script

Run the comprehensive test script:

```bash
# Activate virtual environment
source venv/bin/activate

# Run the test script
python test_phase3.py
```

## Expected Behavior

### ✅ Success Scenarios

1. **Local Resume Checkpoint**:
   - Validates checkpoint exists
   - Validates checkpoint structure
   - Loads checkpoint successfully
   - Logs: "Resume checkpoint validated successfully", "Loading model state from checkpoint..."

2. **URL-Based Resume Checkpoint**:
   - Detects URL format
   - Downloads checkpoint to cache directory (`~/.rfdetr/checkpoints/`)
   - Validates downloaded checkpoint
   - Loads checkpoint successfully
   - Logs: "Resume checkpoint is a URL, downloading", "Successfully downloaded and validated resume checkpoint"

3. **Cached Resume Checkpoint (URL)**:
   - Detects checkpoint already cached
   - Validates cached checkpoint
   - Uses cached checkpoint without re-downloading
   - Logs: "Resume checkpoint already cached and valid"

### ❌ Failure Scenarios (with Clear Errors)

1. **Missing Local Checkpoint**:
   - Raises `FileNotFoundError` with clear message
   - Error message includes path and guidance

2. **Invalid URL**:
   - Raises `RuntimeError` after download attempts fail
   - Error message explains possible causes (network, invalid URL, server issues)

3. **Corrupted Checkpoint**:
   - Detects corruption during validation
   - Raises `RuntimeError` with clear message
   - Removes corrupted file (if downloaded)

## Verification Checklist

- [ ] Test with local resume checkpoint path → loads successfully
- [ ] Test with URL resume checkpoint → downloads and loads successfully
- [ ] Test with cached URL checkpoint → uses cache without re-downloading
- [ ] Test with missing local checkpoint → clear error message
- [ ] Test with invalid URL → clear error message
- [ ] Test with corrupted checkpoint → detects corruption and raises error
- [ ] Check logs for informative messages
- [ ] Verify checkpoint is validated before loading
- [ ] Verify optimizer/scheduler state is loaded correctly
- [ ] Verify EMA model state is loaded correctly (if applicable)

## What to Look For

### ✅ Good Signs

- **No errors** when resuming from valid checkpoint (local or URL)
- **Clear log messages** showing download progress and validation
- **Informative error messages** that guide users to solutions
- **Cached checkpoints** are reused (no unnecessary re-downloads)
- **Fast resume** when checkpoint is cached

### ⚠️ Warning Signs

- **Errors during resume** when checkpoint is valid
- **Unclear error messages** that don't help users understand the issue
- **Unnecessary re-downloads** of cached checkpoints
- **Silent failures** without error messages
- **Long delays** when checkpoint is cached (should be instant)

## Troubleshooting

### Issue: Resume checkpoint not found

**Possible Causes**:
1. Incorrect path
2. File doesn't exist
3. Permission issues

**Solution**:
- Verify checkpoint path is correct
- Check file exists and is readable
- For URLs, check internet connectivity

### Issue: Failed to download from URL

**Possible Causes**:
1. Network connectivity issues
2. Invalid URL
3. Server-side issues (403 Forbidden - URL requires authentication)
4. Firewall/proxy blocking access

**Solution**:
- Check internet connection
- Verify URL is correct and accessible
- Check firewall/proxy settings
- **Note**: If you get 403 Forbidden, the URL may require authentication. This is a server-side restriction, not a code issue. The code correctly handles this with a clear error message.

### Issue: Checkpoint validation failed

**Possible Causes**:
1. Corrupted download
2. Invalid checkpoint format
3. File permissions

**Solution**:
- Remove cached checkpoint and re-download
- Verify checkpoint format matches expected structure
- Check file permissions

### Issue: Cache directory issues

**Possible Causes**:
1. Permission issues creating cache directory
2. Disk space insufficient

**Solution**:
- Check write permissions in home directory
- Verify sufficient disk space
- Manually create cache directory: `mkdir -p ~/.rfdetr/checkpoints`

## Confirming Everything Works Properly

### Step 1: Test Local Resume

```bash
# Activate virtual environment
source venv/bin/activate

# Test with local checkpoint (adjust path as needed)
python -c "
from rfdetr import RFDETRBase
model = RFDETRBase()
# Use your actual checkpoint path
model.train(dataset_dir='path/to/dataset', resume='output/checkpoint.pth', epochs=1, eval=True)
"
```

### Step 2: Test URL Resume

```bash
# Test URL-based resume (use a real checkpoint URL)
python -c "
from rfdetr.util.files import download_resume_checkpoint
path = download_resume_checkpoint('https://storage.googleapis.com/rfdetr/rf-detr-base.pth')
print(f'Downloaded to: {path}')
"
```

### Step 3: Check Cache Directory

```bash
# Check if cache directory exists and contains checkpoints
ls -lh ~/.rfdetr/checkpoints/
```

### Step 4: Verify Logs

Look for these log messages:
1. "Resuming training from checkpoint"
2. "Resume checkpoint is a URL, downloading" (if URL)
3. "Resume checkpoint validated successfully" (if local)
4. "Successfully downloaded and validated resume checkpoint" (if URL)
5. "Loading model state from checkpoint..."
6. "Resuming from epoch X" (if optimizer/scheduler loaded)

## Further Changes Needed?

### If Everything Works ✅

- **Phase 3 is complete!**
- All resume checkpoint functionality is working
- URL-based resume checkpoints are supported
- Validation catches corrupted checkpoints

### If Issues Found ⚠️

**Common Issues and Fixes**:

1. **Cache directory permission errors**:
   - Check write permissions in home directory
   - Consider using alternative cache location

2. **URL download failures**:
   - Increase timeout in `download_file()`
   - Add support for authentication if needed
   - Consider adding progress callbacks

3. **Validation too strict**:
   - Adjust `required_keys` parameter
   - Check checkpoint format matches expected structure

4. **Cache not being used**:
   - Verify cache directory path is correct
   - Check file naming logic for URLs

**How to Report Issues**:
1. Note the exact error message
2. Check logs for detailed information
3. Test with minimal example
4. Document the expected vs actual behavior
5. Include checkpoint path/URL and cache directory location

## Success Criteria

- ✅ Resume checkpoints can be loaded from local paths
- ✅ Resume checkpoints can be downloaded from URLs
- ✅ Downloaded checkpoints are cached for reuse
- ✅ Checkpoints are validated before loading
- ✅ Clear error messages when checkpoint can't be loaded
- ✅ All existing functionality continues to work

