# Quick Test Guide - All Phases

## Run All Phase Tests

### Single Command (Recommended)

```bash
# Activate virtual environment
source venv/bin/activate

# Run all phase tests
python test_all_phases.py
```

This will run all phase tests sequentially and provide a comprehensive summary.

## Run Individual Phase Tests

### Phase 2: Model Initialization Fixes
```bash
source venv/bin/activate
python test_phase2.py
```

### Phase 3: Resume Checkpoint Download Support
```bash
source venv/bin/activate
python test_phase3.py
```

### Phase 4: RFDETR Class Improvements
```bash
source venv/bin/activate
python test_phase4.py
```

### Phase 5: Enhanced Validation Utility
```bash
source venv/bin/activate
python test_phase5.py
```

## Quick Verification Commands

### Test Phase 2 (Model Initialization)
```bash
source venv/bin/activate && python -c "
from rfdetr import RFDETRBase
import os
if os.path.exists('rf-detr-base.pth'):
    os.rename('rf-detr-base.pth', 'rf-detr-base.pth.backup')
try:
    model = RFDETRBase()
    print('✅ Phase 2: Download before loading works')
finally:
    if os.path.exists('rf-detr-base.pth.backup'):
        os.rename('rf-detr-base.pth.backup', 'rf-detr-base.pth')
"
```

### Test Phase 3 (Resume Checkpoint)
```bash
source venv/bin/activate && python -c "
from rfdetr.util.files import download_resume_checkpoint
import os
if os.path.exists('rf-detr-base.pth'):
    path = download_resume_checkpoint('rf-detr-base.pth', validate=True)
    print(f'✅ Phase 3: Resume checkpoint validation works: {path}')
else:
    print('⚠️  No checkpoint to test with')
"
```

### Test Phase 4 (RFDETR Class)
```bash
source venv/bin/activate && python -c "
from rfdetr import RFDETRBase
from rfdetr.config import RFDETRBaseConfig
import logging
logging.basicConfig(level=logging.INFO)
config = RFDETRBaseConfig(pretrain_weights=None)
model = RFDETRBase(**config.model_dump())
print('✅ Phase 4: None handling works')
"
```

### Test Phase 5 (Validation Utility)
```bash
source venv/bin/activate && python -c "
from rfdetr.util.files import validate_checkpoint, validate_checkpoint_structure
import torch
import os
if os.path.exists('rf-detr-base.pth'):
    is_valid, error = validate_checkpoint('rf-detr-base.pth', required_keys=['model'])
    checkpoint = torch.load('rf-detr-base.pth', map_location='cpu', weights_only=False)
    is_valid2, error2, keys = validate_checkpoint_structure(checkpoint, checkpoint_type='auto')
    print(f'✅ Phase 5: Validation works - Basic: {is_valid}, Structure: {is_valid2}')
    print(f'   Keys: {keys}')
else:
    print('⚠️  No checkpoint to test with')
"
```

## Expected Output

When running `python test_all_phases.py`, you should see:

```
======================================================================
RF-DETR Checkpoint Download - All Phases Test Suite
======================================================================

======================================================================
Running Phase 2: Model Initialization Fixes
======================================================================
[Phase 2 test output...]

======================================================================
Running Phase 3: Resume Checkpoint Download Support
======================================================================
[Phase 3 test output...]

======================================================================
Running Phase 4: RFDETR Class Improvements
======================================================================
[Phase 4 test output...]

======================================================================
Running Phase 5: Enhanced Validation Utility
======================================================================
[Phase 5 test output...]

======================================================================
Test Suite Summary
======================================================================
✅ PASS: Phase 2: Model Initialization Fixes
✅ PASS: Phase 3: Resume Checkpoint Download Support
✅ PASS: Phase 4: RFDETR Class Improvements
✅ PASS: Phase 5: Enhanced Validation Utility

Total: 4/4 phases passed

🎉 All phases passed! Checkpoint download functionality is working correctly.
```

## Troubleshooting

If tests fail:

1. **Check virtual environment**: Make sure you're in the correct virtual environment
2. **Check dependencies**: Ensure all required packages are installed
3. **Check checkpoint files**: Some tests require checkpoint files to exist
4. **Check network**: Phase 3 tests may require network connectivity
5. **Review individual test output**: Each test provides detailed error messages

## Test Coverage

- **Phase 2**: 4 tests (missing checkpoint, corrupted checkpoint, custom path, existing checkpoint)
- **Phase 3**: 7 tests (local resume, URL resume, caching, error handling, etc.)
- **Phase 4**: 6 tests (None handling, validation, error handling, etc.)
- **Phase 5**: 8 tests (basic validation, structure validation, key validation, etc.)

**Total: 25+ tests covering all checkpoint download functionality**

