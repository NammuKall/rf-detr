# RF-DETR Weight Discrepancy Analysis

## Executive Summary

This document analyzes the discrepancies between pretrained weights and the current architecture after implementing improvements. The analysis identifies **why** weight loading fails and **what** specific mismatches exist.

---

## Pretrained Weights Architecture (Original)

Based on the error message and code analysis, the pretrained weights (`rf-detr-base.pth`) were trained with:

### Original Architecture Parameters:
- **hidden_dim**: 256
- **sa_nheads**: 8 (self-attention heads)
- **ca_nheads**: 16 (cross-attention heads)
- **dec_n_points**: 2 (sampling points per level in deformable attention)
- **num_encoder_layers**: 0 (no encoder layers)
- **use_cross_scale_fusion**: False (no cross-scale fusion)
- **dec_layers**: 3 (decoder layers)

### Weight Shapes in Pretrained Model:

**Transformer Decoder Layer 0:**
- `transformer.decoder.layers.0.self_attn.in_proj_weight`: `[768, 256]`
  - Formula: `(3 * hidden_dim * sa_nheads, hidden_dim)` = `(3 * 256 * 8, 256)` = `(6144, 256)` ❌
  - **Wait, this doesn't match!** Let me recalculate...
  - Actually: `in_proj_weight` in PyTorch MultiheadAttention is `(3 * embed_dim, embed_dim)` where `embed_dim = hidden_dim`
  - So: `(3 * 256, 256)` = `(768, 256)` ✓ **CORRECT**
  
- `transformer.decoder.layers.0.self_attn.in_proj_bias`: `[768]`
  - `3 * 256 = 768` ✓ **CORRECT**

- `transformer.decoder.layers.0.self_attn.out_proj.weight`: `[256, 256]`
  - `(hidden_dim, hidden_dim)` = `(256, 256)` ✓ **CORRECT**

- `transformer.decoder.layers.0.self_attn.out_proj.bias`: `[256]`
  - `hidden_dim = 256` ✓ **CORRECT**

- `transformer.decoder.layers.0.norm1.weight`: `[256]`
  - `hidden_dim = 256` ✓ **CORRECT**

- `transformer.decoder.layers.0.norm1.bias`: `[256]`
  - `hidden_dim = 256` ✓ **CORRECT**

- `transformer.decoder.layers.0.cross_attn.sampling_offsets.weight`: `[256, ?]`
  - Formula: `(d_model, n_heads * n_levels * n_points * 2)`
  - `(256, 16 * num_feature_levels * 2 * 2)` = `(256, 16 * 1 * 2 * 2)` = `(256, 64)` for Base (P4 only)
  
- `transformer.decoder.layers.0.cross_attn.attention_weights.weight`: `[256, ?]`
  - Formula: `(d_model, n_heads * n_levels * n_points)`
  - `(256, 16 * 1 * 2)` = `(256, 32)` for Base

---

## Current Architecture (After Improvements, use_improvements=False)

When `use_improvements=False` (default), the architecture should match the original:

### Current Architecture Parameters (Expected):
- **hidden_dim**: 256 ✓
- **sa_nheads**: 8 ✓
- **ca_nheads**: 16 ✓
- **dec_n_points**: 2 ✓
- **num_encoder_layers**: 0 ✓
- **use_cross_scale_fusion**: False ✓

### Current Model Weight Shapes (Expected):

**Transformer Decoder Layer 0:**
- `transformer.decoder.layers.0.self_attn.in_proj_weight`: Should be `[768, 256]` ✓
- `transformer.decoder.layers.0.self_attn.in_proj_bias`: Should be `[768]` ✓
- `transformer.decoder.layers.0.self_attn.out_proj.weight`: Should be `[256, 256]` ✓
- `transformer.decoder.layers.0.self_attn.out_proj.bias`: Should be `[256]` ✓
- `transformer.decoder.layers.0.norm1.weight`: Should be `[256]` ✓
- `transformer.decoder.layers.0.norm1.bias`: Should be `[256]` ✓

---

## Error Analysis

The error message shows:
```
size mismatch for transformer.decoder.layers.0.self_attn.in_proj_weight: 
  copying a param with shape torch.Size([768, 256]) from checkpoint, 
  the shape in current model is torch.Size([960, 320]).
```

### What This Tells Us:

1. **Checkpoint has**: `[768, 256]` = `(3 * 256, 256)` ✓ Original architecture
2. **Current model has**: `[960, 320]` = `(3 * 320, 320)` ❌ **Improved architecture!**

### Root Cause:

The error indicates that **even with `use_improvements=False`**, the model is still being built with:
- `hidden_dim = 320` (should be 256)
- `sa_nheads = 10` (should be 8) - Wait, let me recalculate...
  - `960 = 3 * hidden_dim * sa_nheads`? No, that's not right.
  - Actually: `in_proj_weight` shape is `(3 * embed_dim, embed_dim)` where `embed_dim = hidden_dim`
  - So: `960 = 3 * 320` ✓ and `320 = hidden_dim` ✓
  - Therefore: `hidden_dim = 320` (improved value, not original!)

### The Problem:

**The `@model_validator(mode='before')` is NOT working correctly**, or the config values are being set incorrectly before the validator runs.

---

## Detailed Discrepancy Analysis

### Issue 1: Config Validator Not Applying Correctly

**Location**: `rfdetr/config.py`

**Problem**: The `@model_validator(mode='before')` should set `hidden_dim=256` when `use_improvements=False`, but it's not working.

**Evidence**: Error shows model has `hidden_dim=320` even when improvements should be disabled.

**Possible Causes**:
1. Pydantic v2 `model_validator(mode='before')` might not be working as expected
2. The validator might be running but values are being overridden afterwards
3. Default values in class definition might be taking precedence
4. The validator might not be called at all

**Current Code**:
```python
@model_validator(mode='before')
@classmethod
def apply_improvements_before(cls, data):
    if isinstance(data, dict):
        use_improvements = data.get('use_improvements', False)
        if not use_improvements:
            data['hidden_dim'] = 256  # Should force to 256
            # ...
```

**Issue**: Even though we're setting `data['hidden_dim'] = 256`, the model still ends up with 320.

### Issue 2: Class Default Values

**Location**: `rfdetr/config.py:59`

**Problem**: The class has `hidden_dim: int = 256` as default, but somewhere this is being overridden.

**Investigation Needed**: Check if there's any code that sets `hidden_dim` after config creation.

### Issue 3: Model Building Process

**Location**: `rfdetr/models/lwdetr.py:build_model()`

**Problem**: The `build_model` function receives `args` which might have incorrect values.

**Flow**:
1. `RFDETRBase()` → `get_model_config()` → `RFDETRBaseConfig(**kwargs)`
2. Config should have `hidden_dim=256` (if validator works)
3. `config.dict()` → passed to `populate_args()` → `build_model(args)`
4. `build_model` uses `args.hidden_dim` to build transformer

**Potential Issue**: If `args.hidden_dim` is 320 instead of 256, the transformer will be built with wrong dimensions.

---

## Specific Weight Shape Mismatches

Based on the error, here are the exact mismatches:

### 1. Self-Attention Weights

| Weight Name | Checkpoint Shape | Current Model Shape | Expected (Original) |
|------------|------------------|---------------------|---------------------|
| `in_proj_weight` | `[768, 256]` | `[960, 320]` | `[768, 256]` |
| `in_proj_bias` | `[768]` | `[960]` | `[768]` |
| `out_proj.weight` | `[256, 256]` | `[320, 320]` | `[256, 256]` |
| `out_proj.bias` | `[256]` | `[320]` | `[256]` |

**Analysis**:
- Checkpoint uses `hidden_dim=256` ✓
- Current model uses `hidden_dim=320` ❌ (should be 256)
- `960 = 3 * 320` (for in_proj_weight: 3 * embed_dim)
- `768 = 3 * 256` (original)

### 2. Layer Normalization Weights

| Weight Name | Checkpoint Shape | Current Model Shape | Expected (Original) |
|------------|------------------|---------------------|---------------------|
| `norm1.weight` | `[256]` | `[320]` | `[256]` |
| `norm1.bias` | `[256]` | `[320]` | `[256]` |

**Analysis**: Same issue - model is using `hidden_dim=320` instead of `256`.

### 3. Cross-Attention Weights (Likely Affected)

| Weight Name | Checkpoint Shape | Current Model Shape | Expected (Original) |
|------------|------------------|---------------------|---------------------|
| `cross_attn.sampling_offsets.weight` | `[256, ?]` | `[320, ?]` | `[256, ?]` |
| `cross_attn.attention_weights.weight` | `[256, ?]` | `[320, ?]` | `[256, ?]` |
| `cross_attn.value_proj.weight` | `[256, 256]` | `[320, 320]` | `[256, 256]` |
| `cross_attn.output_proj.weight` | `[256, 256]` | `[320, 320]` | `[256, 256]` |

**Analysis**: All cross-attention weights depend on `hidden_dim`, so they'll all be wrong if `hidden_dim` is incorrect.

### 4. Feed-Forward Network Weights

| Weight Name | Checkpoint Shape | Current Model Shape | Expected (Original) |
|------------|------------------|---------------------|---------------------|
| `linear1.weight` | `[2048, 256]` | `[2048, 320]` | `[2048, 256]` |
| `linear1.bias` | `[2048]` | `[2048]` | `[2048]` |
| `linear2.weight` | `[256, 2048]` | `[320, 2048]` | `[256, 2048]` |
| `linear2.bias` | `[256]` | `[320]` | `[256]` |

**Analysis**: FFN weights also depend on `hidden_dim` for input/output dimensions.

### 5. Detection Head Weights

| Weight Name | Checkpoint Shape | Current Model Shape | Expected (Original) |
|------------|------------------|---------------------|---------------------|
| `class_embed.weight` | `[num_classes, 256]` | `[num_classes, 320]` | `[num_classes, 256]` |
| `class_embed.bias` | `[num_classes]` | `[num_classes]` | `[num_classes]` |
| `bbox_embed.layers.0.weight` | `[256, 256]` | `[320, 256]` | `[256, 256]` |
| `bbox_embed.layers.0.bias` | `[256]` | `[256]` | `[256]` |

**Analysis**: Detection heads also depend on `hidden_dim`.

---

## New Components (Not in Pretrained Weights)

### 1. Encoder Layers

**If `num_encoder_layers > 0`**, the model will have:
- `transformer.encoder.layers.*.self_attn.*` - Multi-scale deformable attention
- `transformer.encoder.layers.*.linear1.*` - FFN layers
- `transformer.encoder.layers.*.norm1.*`, `norm2.*` - Layer norms

**Status**: Should be **absent** when `use_improvements=False` (num_encoder_layers=0) ✓

### 2. Cross-Scale Fusion

**If `use_cross_scale_fusion=True`**, the model will have:
- `backbone.0.projector.cross_scale_fusion.fusion_weights`
- `backbone.0.projector.cross_scale_fusion.fusion_conv.*`

**Status**: Should be **absent** when `use_improvements=False` ✓

---

## Root Cause Analysis

### Primary Issue: Config Validator Not Working

The `@model_validator(mode='before')` is supposed to set `hidden_dim=256` when `use_improvements=False`, but the model is still being built with `hidden_dim=320`.

**Possible Reasons**:

1. **Pydantic Version Issue**: Different Pydantic versions handle validators differently
2. **Validator Execution Order**: The validator might run but values get overridden
3. **Default Value Precedence**: Class defaults might override validator settings
4. **Dict vs Object**: When `config.dict()` is called, it might use class defaults instead of validated values

### Secondary Issue: Value Propagation

Even if the validator works, the values need to propagate correctly:
1. Config → `config.dict()` → `populate_args(**kwargs)` → `args.hidden_dim`
2. If any step uses wrong values, the model will be wrong

---

## Code Flow Analysis

### Step-by-Step Flow:

1. **User calls**: `RFDETRBase()` (no kwargs)
2. **RFDETR.__init__**: Calls `get_model_config(**kwargs)` where `kwargs={}`
3. **RFDETRBase.get_model_config**: Returns `RFDETRBaseConfig(**kwargs)` where `kwargs={}`
4. **RFDETRBaseConfig.__init__**: Pydantic creates instance
   - Class defaults: `hidden_dim=256`, `use_improvements=False`
   - Validator runs: Should set `hidden_dim=256` (already correct)
5. **Model creation**: `self.model = self.get_model(self.model_config)`
6. **get_model**: Calls `Model(**config.dict())`
7. **Model.__init__**: Calls `populate_args(**kwargs)`
8. **populate_args**: Creates args object with `hidden_dim` from kwargs
9. **build_model**: Uses `args.hidden_dim` to build transformer

**Potential Issue Points**:
- Step 4: Validator might not be running or not working
- Step 6: `config.dict()` might return wrong values
- Step 7-8: `populate_args` might override values

---

## Investigation Checklist

To identify the exact issue, check:

- [ ] Does `RFDETRBaseConfig(use_improvements=False).hidden_dim` return 256 or 320?
- [ ] Does `RFDETRBaseConfig(use_improvements=False).dict()['hidden_dim']` return 256 or 320?
- [ ] What does `populate_args(**config.dict())` set for `hidden_dim`?
- [ ] Is the validator actually being called?
- [ ] Are there any other places that set `hidden_dim`?

---

## Expected vs Actual Behavior

### Expected (use_improvements=False):
```python
config = RFDETRBaseConfig(use_improvements=False)
assert config.hidden_dim == 256  # Should be True
assert config.sa_nheads == 8     # Should be True
assert config.ca_nheads == 16    # Should be True
assert config.dec_n_points == 2  # Should be True
```

### Actual (Based on Error):
```python
# Model is built with:
hidden_dim = 320  # ❌ Should be 256
sa_nheads = 10    # ❌ Should be 8 (inferred from 960 = 3*320, but wait...)
```

**Wait**: Let me recalculate `sa_nheads`:
- `in_proj_weight` shape: `[960, 320]`
- Formula: `(3 * embed_dim, embed_dim)` where `embed_dim = hidden_dim`
- So: `960 = 3 * 320` → `embed_dim = 320` → `hidden_dim = 320`
- `sa_nheads` doesn't affect `in_proj_weight` shape in MultiheadAttention
- The shape is always `(3 * embed_dim, embed_dim)` regardless of num_heads

So the issue is simply: **`hidden_dim` is 320 instead of 256**.

---

## Summary of Issues

### Critical Issues:

1. **Config validator not enforcing original dimensions**
   - **Symptom**: Model built with `hidden_dim=320` even when `use_improvements=False`
   - **Impact**: All transformer weights have wrong shapes
   - **Affected Components**: All decoder layers, detection heads

2. **Weight shape mismatches** (consequence of issue #1):
   - Self-attention: `[768,256]` vs `[960,320]`
   - Layer norms: `[256]` vs `[320]`
   - Cross-attention: `[256,?]` vs `[320,?]`
   - FFN: `[2048,256]` vs `[2048,320]`
   - Detection heads: `[num_classes,256]` vs `[num_classes,320]`

### Non-Issues (Working Correctly):

1. **Encoder layers**: Correctly absent when `num_encoder_layers=0` ✓
2. **Cross-scale fusion**: Correctly absent when `use_cross_scale_fusion=False` ✓

---

## Recommendations (Analysis Only - No Code Changes)

### Issue 1: Config Validator

**Problem**: The `@model_validator(mode='before')` is not working as expected.

**Possible Solutions** (to investigate, not implement):
1. Use `@field_validator` instead of `@model_validator`
2. Override `__init__` method directly (but Pydantic v2 might not support this well)
3. Use `model_config` with `validate_assignment=True`
4. Check Pydantic version compatibility

### Issue 2: Value Propagation

**Problem**: Values might be getting overridden somewhere in the pipeline.

**Investigation Points**:
1. Check `populate_args` function - does it respect config values?
2. Check if `config.dict()` returns correct values
3. Add debug prints to trace value flow

### Issue 3: Default Values

**Problem**: Class defaults might be conflicting with validator.

**Investigation**: Check if Pydantic is using class defaults before validator runs.

---

## Additional Findings

### Issue 4: populate_args Default Values

**Location**: `rfdetr/main.py:909-911`

**Problem**: `populate_args` has default values that might conflict:

```python
def populate_args(
    ...
    hidden_dim=256,  # ✓ Correct default
    sa_nheads=8,     # ✓ Correct default  
    ca_nheads=8,      # ❌ Should be 16 for Base model!
    ...
)
```

**Analysis**: 
- `ca_nheads=8` is wrong for Base model (should be 16)
- However, if `config.dict()` passes `ca_nheads=16`, it should override the default
- The main issue is still `hidden_dim` being 320 instead of 256

### Issue 5: Config Dict Conversion

**Location**: `rfdetr/detr.py:206`

**Problem**: `config.dict()` is called to convert config to dict:

```python
return Model(**config.dict())
```

**Investigation Needed**: 
- Does `config.dict()` return validated values or class defaults?
- Pydantic's `dict()` method should return the validated values, but need to verify

### Issue 6: Value Flow Trace

**Flow**:
1. `RFDETRBase()` → `get_model_config()` → `RFDETRBaseConfig(**kwargs)` where `kwargs={}`
2. Pydantic creates instance:
   - Class defaults: `hidden_dim=256`, `use_improvements=False`
   - Validator runs: Should set `hidden_dim=256` (already correct from defaults)
   - **BUT**: If validator doesn't run or fails, class defaults are used
3. `config.dict()` → Should return `{'hidden_dim': 256, ...}`
4. `Model(**config.dict())` → `populate_args(**{'hidden_dim': 256, ...})`
5. `populate_args` uses kwargs if provided, otherwise defaults
6. `build_model(args)` → Uses `args.hidden_dim`

**Potential Break Points**:
- Step 2: Validator might not be running
- Step 3: `config.dict()` might return wrong values
- Step 4: `populate_args` might not receive kwargs correctly

---

## Conclusion

The primary issue is that **the config validator is not correctly enforcing `hidden_dim=256` when `use_improvements=False`**. This causes the model to be built with improved dimensions (`hidden_dim=320`), which don't match the pretrained weights (`hidden_dim=256`).

**All weight mismatches stem from this single root cause**: incorrect `hidden_dim` value propagation.

### Root Cause Summary:

1. **Config Validator Issue**: The `@model_validator(mode='before')` is not working correctly
   - Even though it sets `data['hidden_dim'] = 256`, the model ends up with `hidden_dim=320`
   - Possible reasons: Pydantic version incompatibility, validator execution order, or value override

2. **Value Propagation Issue**: Values might be getting overridden somewhere in the pipeline
   - `config.dict()` might not return validated values
   - `populate_args` might have conflicting defaults (though `ca_nheads=8` vs `16` is separate)

3. **Class Defaults**: The class has correct defaults (`hidden_dim=256`), but something is overriding them

### The Fix Requires:

1. **Ensure validator works**: The validator must correctly set `hidden_dim=256` when `use_improvements=False`
2. **Verify value propagation**: `config.dict()` must return the validated values
3. **Check populate_args**: Ensure it respects kwargs over defaults
4. **Model building**: Ensure `build_model` uses correct `args.hidden_dim`

### Specific Weight Mismatches (All stem from hidden_dim=320 instead of 256):

| Component | Weight Name | Checkpoint | Current Model | Issue |
|-----------|-------------|------------|---------------|-------|
| Self-Attn | `in_proj_weight` | `[768, 256]` | `[960, 320]` | hidden_dim wrong |
| Self-Attn | `in_proj_bias` | `[768]` | `[960]` | hidden_dim wrong |
| Self-Attn | `out_proj.weight` | `[256, 256]` | `[320, 320]` | hidden_dim wrong |
| Self-Attn | `out_proj.bias` | `[256]` | `[320]` | hidden_dim wrong |
| Layer Norm | `norm1.weight` | `[256]` | `[320]` | hidden_dim wrong |
| Layer Norm | `norm1.bias` | `[256]` | `[320]` | hidden_dim wrong |
| Cross-Attn | `sampling_offsets.weight` | `[256, ?]` | `[320, ?]` | hidden_dim wrong |
| Cross-Attn | `attention_weights.weight` | `[256, ?]` | `[320, ?]` | hidden_dim wrong |
| Cross-Attn | `value_proj.weight` | `[256, 256]` | `[320, 320]` | hidden_dim wrong |
| Cross-Attn | `output_proj.weight` | `[256, 256]` | `[320, 320]` | hidden_dim wrong |
| FFN | `linear1.weight` | `[2048, 256]` | `[2048, 320]` | hidden_dim wrong |
| FFN | `linear2.weight` | `[256, 2048]` | `[320, 2048]` | hidden_dim wrong |
| Detection | `class_embed.weight` | `[91, 256]` | `[91, 320]` | hidden_dim wrong |
| Detection | `bbox_embed.layers.0.weight` | `[256, 256]` | `[320, 256]` | hidden_dim wrong |

**All mismatches are due to `hidden_dim` being 320 instead of 256.**

---

## Critical Discovery: Pydantic Validator Behavior

### The Real Problem

When `RFDETRBaseConfig()` is called with **no kwargs** (empty dict `{}`):

1. **Pydantic v2 behavior**: 
   - If `kwargs={}` (empty), Pydantic creates instance using **class field defaults directly**
   - The `@model_validator(mode='before')` receives `data={}` (empty dict)
   - Validator sets `data['hidden_dim'] = 256`
   - **BUT**: Pydantic then creates the instance using class defaults, which might override validator changes

2. **The Issue**:
   - Class has: `hidden_dim: int = 256` ✓ (correct default)
   - Validator sets: `data['hidden_dim'] = 256` ✓ (correct)
   - **BUT**: If validator runs AFTER class defaults are applied, or if there's a conflict, values might be wrong

3. **Evidence from Error**:
   - Model has `hidden_dim=320` (improved value)
   - This suggests class defaults OR validator is using improved values
   - **Wait**: Class defaults are `hidden_dim=256`, so where is 320 coming from?

### Hypothesis: Validator Not Running or Running Incorrectly

**Possible Scenarios**:

1. **Validator not called**: When `kwargs={}`, Pydantic might skip validator
2. **Validator runs but values overridden**: Class defaults applied after validator
3. **Wrong validator logic**: The `setdefault` vs direct assignment might have issues
4. **Pydantic version issue**: Different Pydantic versions handle validators differently

### The Smoking Gun

Looking at the validator code:
```python
if use_improvements:
    data.setdefault('hidden_dim', data.pop('improved_hidden_dim', 320))
```

**Problem**: When `use_improvements=False` and `data={}`, the validator sets:
```python
data['hidden_dim'] = 256  # Direct assignment
```

But if `data` is empty and Pydantic uses class defaults, the validator might not be effective.

### Additional Issue: populate_args Default

**Location**: `rfdetr/main.py:911`

```python
ca_nheads=8,  # Default is 8
```

**Problem**: Base model should have `ca_nheads=16`, but `populate_args` defaults to 8.

**Impact**: Even if config has `ca_nheads=16`, if it's not passed correctly, `populate_args` will use default 8.

**However**: This is a separate issue from the `hidden_dim` problem.

---

## Final Analysis

### Root Cause: Config Value Not Propagating Correctly

The error shows the model is built with `hidden_dim=320`, which means:

1. **Either**: The config validator is not working
2. **Or**: The config value is correct but gets overridden somewhere
3. **Or**: `config.dict()` is not returning the correct values

### Investigation Points:

1. **Check if validator runs**: Add print statements or check Pydantic logs
2. **Check config.dict() output**: Verify what values it returns
3. **Check populate_args**: See what values it receives
4. **Check build_model**: See what `args.hidden_dim` is when model is built

### Most Likely Issue:

**The `@model_validator(mode='before')` is not being called or not working correctly** when the config is created with no kwargs. Pydantic v2 might handle empty kwargs differently than expected.

### Why This Happens:

- When `RFDETRBase()` is called, it calls `RFDETRBaseConfig(**kwargs)` where `kwargs={}`
- Pydantic creates instance from class defaults
- Validator should run, but might not be effective with empty input
- Result: Model uses some default (possibly from somewhere else) that's 320

### Where 320 Might Come From:

1. **Improved defaults**: If validator logic is inverted or wrong
2. **Another config**: If there's inheritance or mixing of configs
3. **populate_args**: If it has a default of 320 (but it doesn't - it's 256)
4. **Build process**: If something overrides during model building

---

## Summary

**Primary Issue**: Config validator not correctly enforcing `hidden_dim=256` when `use_improvements=False`, causing model to be built with `hidden_dim=320`.

**All weight shape mismatches** are consequences of this single issue - every component that depends on `hidden_dim` will have wrong shapes.

**The fix requires**: Ensuring the validator works correctly and values propagate through the entire pipeline from config → dict → populate_args → build_model.

---

## Appendix: Code References

### Key Files Involved:

1. **`rfdetr/config.py`**: 
   - `RFDETRBaseConfig` class with validator (lines 47-112)
   - Validator sets `hidden_dim=256` when `use_improvements=False`

2. **`rfdetr/detr.py`**: 
   - `RFDETRBase.get_model()` calls `Model(**config.dict())` (line 206)
   - This converts config to dict and passes to Model

3. **`rfdetr/main.py`**: 
   - `populate_args()` has defaults including `hidden_dim=256` (line 909)
   - `build_model()` uses `args.hidden_dim` to build transformer

4. **`rfdetr/models/lwdetr.py`**: 
   - `build_model()` creates transformer with `args.hidden_dim`
   - `build_transformer()` uses `d_model=args.hidden_dim`

5. **`rfdetr/models/transformer.py`**: 
   - `Transformer.__init__()` uses `d_model` parameter
   - `TransformerDecoderLayer` uses `d_model` for all layers

### Weight Shape Formulas:

**Self-Attention (MultiheadAttention)**:
- `in_proj_weight`: `(3 * embed_dim, embed_dim)` where `embed_dim = hidden_dim`
- `out_proj.weight`: `(embed_dim, embed_dim)`
- `norm1.weight`: `(embed_dim,)`

**Cross-Attention (MSDeformAttn)**:
- `sampling_offsets.weight`: `(d_model, n_heads * n_levels * n_points * 2)`
- `attention_weights.weight`: `(d_model, n_heads * n_levels * n_points)`
- `value_proj.weight`: `(d_model, d_model)`
- `output_proj.weight`: `(d_model, d_model)`

**Feed-Forward Network**:
- `linear1.weight`: `(dim_feedforward, d_model)`
- `linear2.weight`: `(d_model, dim_feedforward)`

**Detection Heads**:
- `class_embed.weight`: `(num_classes, hidden_dim)`
- `bbox_embed.layers.0.weight`: `(hidden_dim, hidden_dim)`

All of these depend on `hidden_dim`, so if it's wrong, all weights will be wrong.

---

## End of Analysis

This document provides a comprehensive analysis of weight discrepancies between pretrained weights and the current architecture. The primary issue is that `hidden_dim` is 320 instead of 256, causing all transformer weights to have incorrect shapes.

