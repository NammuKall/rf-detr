# Using RF-DETR Improvements

## Important Note About Pretrained Weights

The architectural improvements (encoder layers, increased model width, etc.) **require retrained weights** because they change the model architecture. The existing pretrained weights (`rf-detr-base.pth`, `rf-detr-large.pth`, etc.) were trained with the original architecture.

## Default Behavior

**By default, improvements are DISABLED** (`use_improvements=False`) to ensure compatibility with existing pretrained weights.

```python
from rfdetr import RFDETRBase

# This uses the original architecture (compatible with pretrained weights)
model = RFDETRBase()  # use_improvements=False by default
```

## Enabling Improvements

To use the improved architecture, you have two options:

### Option 1: Enable Improvements (Requires Retraining)

```python
from rfdetr import RFDETRBase

# Enable improvements - requires retrained weights
model = RFDETRBase(use_improvements=True)

# Note: This will fail to load pretrained weights because architecture changed
# You'll need to train from scratch or fine-tune
```

### Option 2: Train with Improvements Enabled

```python
from rfdetr import RFDETRBase

# Create model with improvements
model = RFDETRBase(use_improvements=True)

# Train from scratch (or fine-tune from backbone)
model.train(
    dataset_dir="path/to/dataset",
    epochs=100,
    # ... other training parameters
)
```

## What Changes When Improvements Are Enabled?

When `use_improvements=True`:

1. **Model Width Increases:**
   - Base: hidden_dim 256→320, sa_nheads 8→10, ca_nheads 16→20, dec_n_points 2→4
   - Large: hidden_dim 384→512, sa_nheads 12→16, ca_nheads 24→32, dec_n_points 4→6
   - Medium: hidden_dim 256→384, sa_nheads 8→12, ca_nheads 16→24, dec_n_points 2→4

2. **Encoder Layers Added:**
   - Base: 2 encoder layers
   - Large: 3 encoder layers
   - Medium: 2 encoder layers

3. **Cross-Scale Fusion Enabled:**
   - Features at different scales can interact

4. **More Sampling Points:**
   - Better attention with more sampling points in deformable attention

## Expected Performance

With improvements enabled and retrained:
- **Accuracy**: +2.4-4.9 AP improvement on COCO
- **Latency**: +45-65% increase (can be offset with Flash Attention)

## Migration Guide

### If You Want to Use Pretrained Weights (Current Behavior)

**No changes needed!** The default behavior uses the original architecture:

```python
model = RFDETRBase()  # Works with pretrained weights
```

### If You Want to Use Improvements

1. **Train from scratch:**
   ```python
   model = RFDETRBase(use_improvements=True)
   model.train(...)
   ```

2. **Or fine-tune from backbone:**
   ```python
   model = RFDETRBase(use_improvements=True, pretrain_weights=None)
   # Load DINOv2 backbone weights
   model.train(...)
   ```

## Troubleshooting

### Error: "size mismatch" when loading weights

**Cause:** You're trying to load pretrained weights with improvements enabled.

**Solution:** 
- Set `use_improvements=False` to use pretrained weights, OR
- Train from scratch with `use_improvements=True`

### Error: "num_encoder_layers" not found

**Cause:** Old code trying to use new parameters.

**Solution:** Update your code to use the new config system.

## Summary

- **Default**: Original architecture (compatible with pretrained weights)
- **With Improvements**: Better accuracy but requires retraining
- **To Enable**: Set `use_improvements=True` and train from scratch

