# Plan: Ensure Model Checkpoint Download Before Checkpoint Errors

## Current Problem Analysis

### Issues Identified:

1. **Race Condition in `main.py` Model.__init__()**:
   - Line 88: `torch.load()` is called BEFORE checking if file exists
   - Only downloads AFTER exception occurs (line 93)
   - This causes unnecessary error messages and poor UX

2. **Incomplete Download Logic**:
   - `download_pretrain_weights()` only downloads if `pretrain_weights in HOSTED_MODELS`
   - If user provides custom path (not in HOSTED_MODELS), no download attempt
   - No validation that file exists before loading

3. **Missing Download for Resume Checkpoints**:
   - Line 289: `args.resume` checkpoint loading has NO download logic
   - If resume checkpoint doesn't exist, will fail immediately
   - No way to download training checkpoints from remote

4. **No File Validation**:
   - No checksum verification after download
   - No validation that downloaded file is a valid PyTorch checkpoint
   - Corrupted downloads only detected during `torch.load()`

5. **Path Handling Issues**:
   - Downloads save to current working directory (relative paths)
   - If user provides absolute path that doesn't exist, download still saves to CWD
   - Path resolution inconsistencies

## Detailed Implementation Plan

### Phase 1: Enhance `download_pretrain_weights()` Function

**Location**: `rfdetr/main.py` (lines 67-76)

**Changes**:
1. Add file existence check BEFORE attempting download
2. Add file validation after download (basic PyTorch checkpoint structure check)
3. Improve error handling for network failures
4. Add optional checksum verification parameter
5. Handle both relative and absolute paths correctly
6. Return boolean indicating success/failure

**New Function Signature**:
```python
def download_pretrain_weights(pretrain_weights: str, redownload=False, validate=True) -> bool:
    """
    Download pretrained weights if needed.
    
    Args:
        pretrain_weights: Path to checkpoint file (can be filename or full path)
        redownload: Force re-download even if file exists
        validate: Validate checkpoint structure after download
        
    Returns:
        True if checkpoint exists and is valid, False otherwise
        
    Raises:
        FileNotFoundError: If checkpoint not in HOSTED_MODELS and doesn't exist locally
        ValueError: If downloaded file is corrupted/invalid
    """
```

### Phase 2: Fix Model.__init__() Checkpoint Loading

**Location**: `rfdetr/main.py` (lines 85-94)

**Changes**:
1. Call `download_pretrain_weights()` BEFORE `torch.load()`
2. Check return value to ensure download succeeded
3. Only attempt `torch.load()` if file exists and is valid
4. Improve error messages with actionable guidance
5. Handle case where checkpoint is not in HOSTED_MODELS (custom path)

**New Flow**:
```python
if args.pretrain_weights is not None:
    # Step 1: Ensure checkpoint exists (download if needed)
    checkpoint_available = download_pretrain_weights(args.pretrain_weights)
    if not checkpoint_available:
        raise FileNotFoundError(f"Checkpoint not found: {args.pretrain_weights}")
    
    # Step 2: Load checkpoint (now guaranteed to exist)
    try:
        checkpoint = torch.load(args.pretrain_weights, map_location='cpu', weights_only=False)
    except Exception as e:
        # File exists but is corrupted - re-download
        logger.warning(f"Checkpoint corrupted, re-downloading: {e}")
        download_pretrain_weights(args.pretrain_weights, redownload=True)
        checkpoint = torch.load(args.pretrain_weights, map_location='cpu', weights_only=False)
```

### Phase 3: Add Resume Checkpoint Download Support

**Location**: `rfdetr/main.py` (lines 288-300)

**Changes**:
1. Add `download_resume_checkpoint()` helper function
2. Check if resume checkpoint exists before loading
3. Support downloading from URL if resume path is a URL
4. Validate resume checkpoint structure before loading

**New Function**:
```python
def download_resume_checkpoint(resume_path: str) -> str:
    """
    Ensure resume checkpoint exists, download if it's a URL.
    
    Args:
        resume_path: Path to checkpoint or URL
        
    Returns:
        Local path to checkpoint file
        
    Raises:
        FileNotFoundError: If local path doesn't exist and not a URL
    """
```

### Phase 4: Improve RFDETR Class Download Logic

**Location**: `rfdetr/detr.py` (lines 66-70)

**Changes**:
1. Handle None case for `pretrain_weights` gracefully
2. Add validation after download
3. Improve error messages

**Updated Method**:
```python
def maybe_download_pretrain_weights(self):
    """Download pre-trained weights if they are not already downloaded."""
    if self.model_config.pretrain_weights is not None:
        success = download_pretrain_weights(self.model_config.pretrain_weights)
        if not success:
            logger.warning(f"Could not download/validate checkpoint: {self.model_config.pretrain_weights}")
```

### Phase 5: Add Checkpoint Validation Utility

**Location**: `rfdetr/util/files.py` (new function)

**Changes**:
1. Create `validate_checkpoint()` function
2. Check file exists and is readable
3. Verify it's a valid PyTorch checkpoint (has expected keys)
4. Return validation result with error details

**New Function**:
```python
def validate_checkpoint(checkpoint_path: str, required_keys=None) -> tuple[bool, Optional[str]]:
    """
    Validate that a checkpoint file exists and is valid.
    
    Args:
        checkpoint_path: Path to checkpoint file
        required_keys: List of required keys in checkpoint dict
        
    Returns:
        (is_valid, error_message)
    """
```

## Potential Issues & Mitigation Strategies

### Issue 1: Network Failures During Download
**Problem**: Download might fail mid-way, leaving corrupted file
**Mitigation**:
- Download to temporary file first, then rename on success
- Add retry logic with exponential backoff
- Verify file size matches expected size from Content-Length header

**What You Should Do**:
- Test with network interruptions
- Monitor download progress
- Consider adding resume capability for partial downloads

### Issue 2: Disk Space Insufficient
**Problem**: Large checkpoints (1.5GB+) might fail if disk is full
**Mitigation**:
- Check available disk space before starting download
- Provide clear error message with required space
- Suggest cleanup options

**What You Should Do**:
- Monitor disk usage in your environment
- Set up alerts for low disk space
- Consider using external storage for large checkpoints

### Issue 3: Custom Checkpoint Paths Not in HOSTED_MODELS
**Problem**: User provides custom path that doesn't exist locally
**Mitigation**:
- Check if path exists before attempting load
- Provide clear error message suggesting to download manually
- Consider supporting URL paths for custom checkpoints

**What You Should Do**:
- Document supported checkpoint sources
- Provide examples of custom checkpoint usage
- Consider adding support for custom download URLs

### Issue 4: Concurrent Downloads
**Problem**: Multiple processes might try to download same file simultaneously
**Mitigation**:
- Use file locking mechanism
- Check if another process is downloading (check for `.tmp` file)
- Wait for other process to finish instead of re-downloading

**What You Should Do**:
- Test with multiple training runs starting simultaneously
- Monitor for race conditions
- Consider using atomic file operations

### Issue 5: Path Resolution Confusion
**Problem**: Relative vs absolute paths, different working directories
**Mitigation**:
- Normalize all paths to absolute paths
- Store checkpoint location in config for consistency
- Document expected working directory behavior

**What You Should Do**:
- Test from different working directories
- Use absolute paths in production
- Document checkpoint location requirements

### Issue 6: Checkpoint Format Changes
**Problem**: Future checkpoint format changes might break validation
**Mitigation**:
- Make validation flexible (check for common keys, not all keys)
- Version checkpoint format
- Provide migration utilities

**What You Should Do**:
- Test with different checkpoint versions
- Keep validation logic updated with format changes
- Consider checkpoint versioning scheme

### Issue 7: Permission Issues
**Problem**: No write permission in current directory
**Mitigation**:
- Check write permissions before download
- Suggest alternative download location
- Use user's home directory as fallback

**What You Should Do**:
- Test with restricted permissions
- Ensure proper directory permissions
- Consider using XDG cache directory standard

## Testing Strategy

### Unit Tests Needed:
1. Test `download_pretrain_weights()` with existing file
2. Test `download_pretrain_weights()` with missing file
3. Test `download_pretrain_weights()` with corrupted file
4. Test `download_pretrain_weights()` with network failure
5. Test `validate_checkpoint()` with valid/invalid checkpoints
6. Test path resolution (relative vs absolute)

### Integration Tests Needed:
1. Test full training flow with missing checkpoint (should download)
2. Test resume with missing checkpoint (should fail gracefully)
3. Test concurrent downloads
4. Test with insufficient disk space

### Manual Testing Checklist:
- [ ] Start training with missing checkpoint → should download automatically
- [ ] Start training with existing checkpoint → should use existing file
- [ ] Start training with corrupted checkpoint → should re-download
- [ ] Resume training with missing checkpoint → should fail with clear error
- [ ] Test from different working directories
- [ ] Test with custom checkpoint path (not in HOSTED_MODELS)
- [ ] Test with network interruption during download
- [ ] Test with insufficient disk space

## Implementation Order

1. **First**: Add `validate_checkpoint()` utility function
2. **Second**: Enhance `download_pretrain_weights()` with validation
3. **Third**: Fix `Model.__init__()` to download before loading
4. **Fourth**: Add resume checkpoint download support
5. **Fifth**: Update `RFDETR.maybe_download_pretrain_weights()`
6. **Sixth**: Add comprehensive error handling and logging
7. **Seventh**: Add tests for all scenarios

## Success Criteria

- ✅ No checkpoint loading errors when checkpoint doesn't exist (downloads first)
- ✅ Clear error messages when checkpoint can't be downloaded
- ✅ Validation catches corrupted checkpoints before loading
- ✅ Resume checkpoint errors are handled gracefully
- ✅ All existing functionality continues to work
- ✅ Tests cover all edge cases

## Rollback Plan

If issues arise:
1. Revert changes to `download_pretrain_weights()` first (most critical)
2. Keep validation utilities (non-breaking)
3. Document any breaking changes for users
4. Provide migration guide if checkpoint location changes

