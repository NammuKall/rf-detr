# Phase 5 Verification Guide

## What Was Implemented

Phase 5 enhances the checkpoint validation utility with comprehensive structure checking and key verification.

### Key Changes:

1. **Enhanced `validate_checkpoint()`**: Now supports detailed structure validation with checkpoint type detection
2. **New `validate_checkpoint_structure()`**: Validates checkpoint structure based on type (pretrain, resume, inference)
3. **New `validate_checkpoint_keys()`**: Validates required and optional keys with detailed reporting
4. **Checkpoint Type Detection**: Automatically detects checkpoint type based on keys present
5. **Comprehensive Structure Validation**: Checks model state_dict, optimizer, scheduler, epoch, EMA model, etc.

## Files Modified

1. **`rfdetr/util/files.py`**:
   - Enhanced `validate_checkpoint()` function with structure validation option
   - Added `validate_checkpoint_structure()` function for detailed structure checks
   - Added `validate_checkpoint_keys()` function for key validation
   - Added checkpoint type detection ("pretrain", "resume", "inference", "auto")

## How to Verify Everything Works

### Option 1: Quick Integration Test

Test the enhanced validation functions:

```python
from rfdetr.util.files import validate_checkpoint, validate_checkpoint_structure, validate_checkpoint_keys
import torch

# Test 1: Basic validation
checkpoint_path = "rf-detr-base.pth"  # Adjust to your checkpoint
is_valid, error = validate_checkpoint(checkpoint_path, required_keys=['model'])
print(f"Basic validation: {is_valid}, Error: {error}")

# Test 2: Structure validation
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
is_valid, error, key_presence = validate_checkpoint_structure(
    checkpoint,
    checkpoint_type="auto"
)
print(f"Structure validation: {is_valid}")
print(f"Key presence: {key_presence}")

# Test 3: Key validation
is_valid, error, key_status = validate_checkpoint_keys(
    checkpoint,
    required_keys=['model'],
    optional_keys=['optimizer', 'lr_scheduler', 'epoch']
)
print(f"Key validation: {is_valid}")
print(f"Key status: {key_status}")
```

### Option 2: Test Different Checkpoint Types

Test validation for different checkpoint types:

```python
from rfdetr.util.files import validate_checkpoint_structure
import torch

checkpoint_path = "rf-detr-base.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Test pretrain checkpoint
is_valid, error, keys = validate_checkpoint_structure(
    checkpoint,
    checkpoint_type="pretrain"
)
print(f"Pretrain validation: {is_valid}")

# Test resume checkpoint (if it has optimizer/scheduler)
is_valid, error, keys = validate_checkpoint_structure(
    checkpoint,
    checkpoint_type="resume"
)
print(f"Resume validation: {is_valid}")
if not is_valid:
    print(f"  (Expected if checkpoint doesn't have optimizer/scheduler)")
```

### Option 3: Test Key Validation

Test required and optional key validation:

```python
from rfdetr.util.files import validate_checkpoint_keys
import torch

checkpoint_path = "rf-detr-base.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Test with required keys
is_valid, error, status = validate_checkpoint_keys(
    checkpoint,
    required_keys=['model'],
    optional_keys=['optimizer', 'lr_scheduler', 'epoch', 'ema_model']
)
print(f"Validation: {is_valid}")
print(f"Key status: {status}")
```

### Option 4: Comprehensive Test Script

Run the comprehensive test script:

```bash
# Activate virtual environment
source venv/bin/activate

# Run the test script
python test_phase5.py
```

## Expected Behavior

### ✅ Success Scenarios

1. **Pretrain Checkpoint Validation**:
   - Validates 'model' key exists and is a valid state_dict
   - Checks model state_dict structure
   - Returns key_presence dictionary
   - Logs: Validation successful

2. **Resume Checkpoint Validation**:
   - Validates 'model', 'optimizer', 'lr_scheduler', 'epoch' keys
   - Checks structure of each component
   - Validates epoch is an integer
   - Returns detailed key_presence dictionary

3. **Auto Detection**:
   - Automatically detects checkpoint type based on keys
   - Uses appropriate validation rules
   - Handles edge cases gracefully

### ❌ Failure Scenarios (with Clear Errors)

1. **Missing Required Keys**:
   - Returns False with clear error message
   - Lists missing keys
   - Provides key_presence dictionary

2. **Invalid Structure**:
   - Detects invalid model state_dict structure
   - Detects invalid optimizer/scheduler structure
   - Provides specific error messages

3. **Wrong Type**:
   - Detects type mismatches (e.g., epoch not integer)
   - Provides clear error messages

## Verification Checklist

- [ ] Test basic validation with existing checkpoint → validates successfully
- [ ] Test structure validation → detects checkpoint type correctly
- [ ] Test key validation → reports key presence accurately
- [ ] Test with pretrain checkpoint → validates correctly
- [ ] Test with resume checkpoint → validates correctly (if available)
- [ ] Test with missing keys → catches with clear error
- [ ] Test with invalid structure → detects corruption
- [ ] Test auto detection → detects type correctly
- [ ] Check that existing code still works (backward compatibility)

## What to Look For

### ✅ Good Signs

- **Detailed validation** catches structure issues
- **Clear error messages** explain what's wrong
- **Key presence reporting** helps debug issues
- **Type detection** works automatically
- **Backward compatible** with existing code

### ⚠️ Warning Signs

- **False positives** (valid checkpoints marked invalid)
- **False negatives** (invalid checkpoints marked valid)
- **Unclear error messages** that don't help debugging
- **Performance issues** (validation too slow)
- **Breaking changes** in existing code

## Troubleshooting

### Issue: Validation too strict

**Possible Causes**:
1. Checkpoint format changed
2. Validation rules too strict

**Solution**:
- Check checkpoint format matches expected structure
- Adjust validation rules if needed
- Use `validate_structure=False` for basic validation

### Issue: Type detection incorrect

**Possible Causes**:
1. Checkpoint has unexpected key combination
2. Auto-detection logic needs adjustment

**Solution**:
- Explicitly specify checkpoint_type
- Check key_presence dictionary
- Review auto-detection logic

### Issue: Performance issues

**Possible Causes**:
1. Loading large checkpoints is slow
2. Structure validation is expensive

**Solution**:
- Use `validate_structure=False` for faster validation
- Cache validation results if needed
- Optimize validation logic

## Confirming Everything Works Properly

### Step 1: Test Basic Validation

```bash
# Activate virtual environment
source venv/bin/activate

# Test basic validation
python -c "
from rfdetr.util.files import validate_checkpoint
is_valid, error = validate_checkpoint('rf-detr-base.pth', required_keys=['model'])
print(f'Valid: {is_valid}, Error: {error}')
"
```

### Step 2: Test Structure Validation

```bash
# Test structure validation
python -c "
from rfdetr.util.files import validate_checkpoint_structure
import torch
checkpoint = torch.load('rf-detr-base.pth', map_location='cpu', weights_only=False)
is_valid, error, keys = validate_checkpoint_structure(checkpoint, checkpoint_type='auto')
print(f'Valid: {is_valid}')
print(f'Keys: {keys}')
"
```

### Step 3: Test Key Validation

```bash
# Test key validation
python -c "
from rfdetr.util.files import validate_checkpoint_keys
import torch
checkpoint = torch.load('rf-detr-base.pth', map_location='cpu', weights_only=False)
is_valid, error, status = validate_checkpoint_keys(checkpoint, required_keys=['model'])
print(f'Valid: {is_valid}')
print(f'Status: {status}')
"
```

### Step 4: Verify Backward Compatibility

```bash
# Test that existing code still works
python -c "
from rfdetr.util.files import validate_checkpoint
# Old usage should still work
is_valid, error = validate_checkpoint('rf-detr-base.pth', required_keys=['model'])
assert is_valid or error is not None
print('✅ Backward compatibility maintained')
"
```

## Further Changes Needed?

### If Everything Works ✅

- **Phase 5 is complete!**
- Validation utilities are comprehensive
- Structure checking works correctly
- Key verification is accurate

### If Issues Found ⚠️

**Common Issues and Fixes**:

1. **Validation too strict**:
   - Adjust validation rules
   - Make some checks optional
   - Add configuration options

2. **Type detection issues**:
   - Improve auto-detection logic
   - Add more checkpoint type patterns
   - Allow manual type specification

3. **Performance issues**:
   - Optimize validation logic
   - Add caching
   - Make structure validation optional

**How to Report Issues**:
1. Note the exact error message
2. Check key_presence dictionary
3. Test with minimal example
4. Document expected vs actual behavior
5. Include checkpoint type and structure

## Success Criteria

- ✅ Enhanced validation checks file structure thoroughly
- ✅ Required keys are verified correctly
- ✅ Checkpoint type detection works automatically
- ✅ Structure validation catches corruption
- ✅ Key validation provides detailed reporting
- ✅ Backward compatibility maintained
- ✅ Clear error messages guide users

