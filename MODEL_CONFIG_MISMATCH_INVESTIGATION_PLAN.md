# Model Config Mismatch Investigation Plan

## Problem Statement

The model configuration still doesn't match even when loading the initial model checkpoint. This suggests there may be a broken pathway in how:
1. The model architecture is built from config
2. The checkpoint weights are loaded
3. The checkpoint's saved config/args are compared (or not compared) with the current config

## Investigation Objectives

1. **Identify where config mismatches occur** - Find all points where model config is used vs checkpoint config
2. **Trace the checkpoint loading pathway** - Map the complete flow from checkpoint file to loaded weights
3. **Detect architectural incompatibilities** - Find mismatches in model architecture parameters
4. **Identify silent failures** - Find cases where mismatches are ignored but cause issues

## Investigation Steps

### Phase 1: Checkpoint Structure Analysis ✅ COMPLETE

**Goal**: Understand what information is stored in checkpoints

**Status**: ✅ **COMPLETE** - All checkpoints analyzed successfully

**Actions Completed**:
1. ✅ Loaded all checkpoints (`rf-detr-base.pth`, `rf-detr-small.pth`, `rf-detr-medium.pth`, `rf-detr-nano.pth`, `rf-detr-large.pth`)
2. ✅ Inspected checkpoint structure - all contain `'args'` key
3. ✅ Compared checkpoint structure across different model sizes
4. ✅ Documented checkpoint format and critical parameters

**Key Findings**:
- ✅ **All checkpoints contain `'args'` key** with saved config (5/5)
- ✅ **Consistent format**: All args are `argparse.Namespace` objects
- ✅ **Critical parameters present**: All architecture-affecting parameters stored
- ❌ **No `'config'` key**: Checkpoints use `'args'` instead
- ✅ **Model state_dict present**: All checkpoints contain weights

**Files Created**:
- `inspect_checkpoint_structure.py` - Inspection script
- `verify_phase1.py` - Verification script
- `checkpoint_structure_analysis.json` - Detailed results
- `PHASE1_CHECKPOINT_STRUCTURE_FINDINGS.md` - Findings report

**Next Step**: Proceed to **Phase 4: Config Comparison Analysis**

---

### Phase 2: Model Building Pathway Analysis ✅ COMPLETE

**Goal**: Trace how model architecture is built from config

**Status**: ✅ **COMPLETE** - Pathway successfully traced, no blocking issues found

**Actions Completed**:
1. ✅ Traced `Model.__init__()` flow:
   - `populate_args(**kwargs)` → converts config to args
   - `build_model(args)` → builds model architecture
   - Verified what happens BEFORE checkpoint loading

2. ✅ Identified critical config parameters (17 total):
   - `encoder`, `hidden_dim`, `sa_nheads`, `ca_nheads`, `dec_layers`
   - `dec_n_points`, `num_queries`, `group_detr`, `projector_scale`
   - `out_feature_indexes`, `num_classes`, `resolution`, `patch_size`
   - `num_windows`, `num_encoder_layers`, `use_cross_scale_fusion`, `enc_n_points`

3. ✅ Checked `build_model()` function:
   - Documented how args are used to build architecture
   - Verified default values don't override config unexpectedly
   - Identified transformations that occur during building

**Key Findings**:
- ✅ Model architecture is built BEFORE loading checkpoint
- ✅ Architecture depends on current config, not checkpoint config
- ✅ Config values properly override defaults
- ⚠️ **Transformations occur**: `num_classes` incremented by 1, fallback behaviors
- ✅ All critical parameters properly passed through

**Files Created**:
- `trace_model_building_pathway.py` - Pathway tracing script
- `verify_phase2.py` - Verification script
- `model_building_pathway_analysis.json` - Detailed results
- `PHASE2_MODEL_BUILDING_PATHWAY_FINDINGS.md` - Findings report

**Next Step**: Proceed to **Phase 4: Config Comparison Analysis** (accounting for transformations)

---

### Phase 3: Checkpoint Loading Pathway Analysis

**Goal**: Trace how checkpoint weights are loaded and matched

**Actions**:
1. Trace checkpoint loading in `Model.__init__()`:
   - `download_pretrain_weights()` → ensures checkpoint exists
   - `torch.load()` → loads checkpoint dictionary
   - `checkpoint['model']` → extracts state_dict
   - `self.model.load_state_dict(checkpoint['model'], strict=False)` → loads weights

2. Analyze `strict=False` behavior:
   - What happens with missing keys? (keys in model but not in checkpoint)
   - What happens with unexpected keys? (keys in checkpoint but not in model)
   - What happens with shape mismatches? (same key, different shape)
   - Are there any warnings logged?

3. Check for config comparison logic:
   - Is checkpoint `'args'` compared with current `args`?
   - Are there any validation checks before loading?
   - Are there any warnings about mismatches?

4. Check special handling:
   - `pretrain_exclude_keys` → keys removed before loading
   - `pretrain_keys_modify_to_load` → keys modified before loading
   - Query parameter resizing (`num_queries * group_detr`)
   - Class head reinitialization (`reinitialize_detection_head`)

**Expected Findings**:
- No config comparison happens before loading
- `strict=False` silently ignores mismatches
- Shape mismatches may cause silent failures or errors

---

### Phase 4: Config Comparison Analysis

**Goal**: Compare checkpoint config vs current config

**Actions**:
1. Extract checkpoint config (if available):
   - Load checkpoint and extract `checkpoint['args']` if present
   - Convert to comparable format (dict or Namespace)

2. Extract current config:
   - From `ModelConfig` object
   - From `args` object after `populate_args()`
   - List all architecture-affecting parameters

3. Compare critical parameters:
   - Create comparison function (not implemented yet, just plan)
   - Compare: encoder, hidden_dim, sa_nheads, ca_nheads, dec_layers, etc.
   - Identify which parameters differ

4. Test with different scenarios:
   - Load base checkpoint with base config → should match
   - Load base checkpoint with different config → should show mismatches
   - Load checkpoint without 'args' key → should detect missing config

**Expected Findings**:
- Checkpoint may not have `'args'` key
- If `'args'` exists, it may be in different format (Namespace vs dict)
- Many parameters may differ between checkpoint and current config

---

### Phase 5: State Dict Key Analysis

**Goal**: Analyze state_dict keys to infer architecture differences

**Actions**:
1. Extract state_dict keys from checkpoint:
   - List all keys in `checkpoint['model']`
   - Group by component (backbone, transformer, detection_head, etc.)

2. Extract state_dict keys from current model:
   - Build model with current config
   - Extract `model.state_dict().keys()`
   - Compare with checkpoint keys

3. Identify key differences:
   - Missing keys (in checkpoint but not in model)
   - Unexpected keys (in model but not in checkpoint)
   - Shape mismatches (same key, different shape)

4. Analyze shape mismatches:
   - Which parameters have shape mismatches?
   - What do these mismatches indicate about architecture differences?
   - Are mismatches in critical components (backbone, transformer, head)?

**Expected Findings**:
- Key differences reveal architecture mismatches
- Shape mismatches indicate dimension differences
- Some mismatches may be expected (num_classes, num_queries)

---

### Phase 6: Silent Failure Detection

**Goal**: Find cases where mismatches cause silent failures

**Actions**:
1. Test model loading with mismatched configs:
   - Load checkpoint with different `hidden_dim`
   - Load checkpoint with different `sa_nheads`
   - Load checkpoint with different `dec_layers`
   - Check if loading succeeds but model is broken

2. Test model inference after loading:
   - Does model run without errors?
   - Are outputs reasonable?
   - Are there NaN or inf values?

3. Check for warnings:
   - Are there any warnings logged during loading?
   - Are shape mismatches reported?
   - Are missing keys reported?

4. Test with `strict=True`:
   - What errors occur with `strict=True`?
   - Which keys cause failures?
   - What does this reveal about mismatches?

**Expected Findings**:
- Some mismatches may load successfully but break inference
- Warnings may not be logged for all mismatches
- `strict=True` may reveal critical mismatches

---

## Potential Issues to Look For

### Issue 1: Checkpoint Missing Config Information
**Symptoms**: Checkpoint doesn't contain `'args'` key, so config comparison is impossible
**Impact**: Cannot validate config compatibility
**Investigation**: Check if checkpoints are saved with args, check save logic

### Issue 2: Config Format Mismatch
**Symptoms**: Checkpoint has `'args'` but in different format (Namespace vs dict vs ModelConfig)
**Impact**: Cannot directly compare configs
**Investigation**: Check format conversion, normalization

### Issue 3: Architecture Parameter Mismatch
**Symptoms**: Checkpoint was saved with different architecture parameters (hidden_dim, heads, etc.)
**Impact**: Model architecture doesn't match checkpoint weights
**Investigation**: Compare checkpoint args with current config

### Issue 4: Silent Shape Mismatches
**Symptoms**: `strict=False` allows loading but weights have wrong shapes
**Impact**: Model may fail silently or produce incorrect results
**Investigation**: Check for shape mismatches, test inference

### Issue 5: Missing Validation Logic
**Symptoms**: No config comparison happens before loading
**Impact**: Incompatible checkpoints are loaded without warning
**Investigation**: Check if validation exists, add if missing

### Issue 6: Default Value Overrides
**Symptoms**: Default values in `populate_args()` or `build_model()` override config
**Impact**: Model built with wrong architecture despite correct config
**Investigation**: Trace default values, check override logic

### Issue 7: Config Transformation Issues
**Symptoms**: Config is transformed between ModelConfig → args → model building
**Impact**: Final architecture doesn't match intended config
**Investigation**: Trace config transformations, check for data loss

### Issue 8: Checkpoint Version Incompatibility
**Symptoms**: Checkpoint was saved with older code version
**Impact**: Architecture may have changed, checkpoint incompatible
**Investigation**: Check for version markers, compare code versions

---

## Tools and Scripts Needed

### 1. Checkpoint Inspector Script
**Purpose**: Load and inspect checkpoint structure
**What it should do**:
- Load checkpoint file
- Print all keys in checkpoint
- Extract and display `'args'` if present
- List all state_dict keys
- Show shapes of key parameters

### 2. Config Comparison Script
**Purpose**: Compare checkpoint config with current config
**What it should do**:
- Load checkpoint and extract config
- Load current ModelConfig
- Compare critical parameters
- Report differences

### 3. State Dict Analyzer Script
**Purpose**: Analyze state_dict keys and shapes
**What it should do**:
- Extract keys from checkpoint state_dict
- Extract keys from current model state_dict
- Compare keys (missing, unexpected)
- Compare shapes for matching keys
- Report mismatches

### 4. Model Loading Test Script
**Purpose**: Test model loading with different configs
**What it should do**:
- Load checkpoint with matching config → should work
- Load checkpoint with mismatched config → should detect issues
- Test with `strict=True` and `strict=False`
- Report warnings and errors

---

## What to Document

1. **Checkpoint Structure**: What keys exist, what format they're in
2. **Config Flow**: How config flows from ModelConfig → args → model
3. **Loading Flow**: Step-by-step checkpoint loading process
4. **Mismatch Points**: Where mismatches can occur
5. **Failure Modes**: How mismatches manifest (errors, warnings, silent failures)
6. **Comparison Results**: Actual differences found between checkpoint and current config

---

## Next Steps After Investigation

Once investigation is complete, we'll know:
1. **Where** config mismatches occur
2. **Why** they occur (missing validation, format issues, etc.)
3. **What** the impact is (silent failures, errors, etc.)
4. **How** to fix them (validation, normalization, warnings, etc.)

Then we can:
- Add config validation before loading
- Add config comparison logic
- Add warnings for mismatches
- Fix broken pathways
- Add tests to prevent regressions

---

## Potential Fixes (To Be Determined After Investigation)

### Fix Option 1: Add Config Validation
- Compare checkpoint config with current config before loading
- Warn or error on mismatches
- Allow override flag for intentional mismatches

### Fix Option 2: Normalize Config Formats
- Convert checkpoint args to ModelConfig format
- Ensure consistent format for comparison
- Handle missing config gracefully

### Fix Option 3: Add Shape Validation
- Check state_dict shapes match model architecture
- Warn on shape mismatches
- Provide detailed mismatch report

### Fix Option 4: Improve Error Messages
- Add clear error messages for config mismatches
- Suggest compatible config values
- Provide troubleshooting guidance

### Fix Option 5: Add Config Compatibility Check
- Define compatible config ranges
- Check if checkpoint config is compatible (even if not identical)
- Allow compatible mismatches, reject incompatible ones

---

## How to Approach Fixing Issues

### If Issue Found: Missing Config in Checkpoint
1. Check checkpoint save logic - ensure args are saved
2. If checkpoints don't have args, add backward compatibility
3. Infer config from state_dict if args missing
4. Add warning when config is missing

### If Issue Found: Config Format Mismatch
1. Normalize checkpoint args to ModelConfig format
2. Create conversion function
3. Handle both old and new formats
4. Add format detection logic

### If Issue Found: Architecture Mismatch
1. Add config comparison before loading
2. Warn on mismatches (don't fail unless critical)
3. List all mismatched parameters
4. Allow override flag for intentional mismatches

### If Issue Found: Silent Failures
1. Add shape validation
2. Test model after loading
3. Add warnings for potential issues
4. Consider failing on critical mismatches

### If Issue Found: Missing Validation
1. Add config validation function
2. Call before loading checkpoint
3. Report mismatches clearly
4. Provide actionable error messages

---

## Testing Strategy

After fixes are implemented:

1. **Test with matching configs**: Should load successfully
2. **Test with mismatched configs**: Should warn or error appropriately
3. **Test with missing config**: Should handle gracefully
4. **Test with old checkpoints**: Should maintain backward compatibility
5. **Test inference**: Should work correctly after loading
6. **Test edge cases**: Empty checkpoints, corrupted checkpoints, etc.

---

## Notes

- This investigation should be done WITHOUT changing code first
- Document all findings before proposing fixes
- Test with actual checkpoints (rf-detr-base.pth, etc.)
- Consider backward compatibility when fixing
- Provide clear error messages for users

