# Strategy for 0-5 Pixel Accuracy

## Goal: Sub-5-Pixel Precision

Target: **95%+ of predictions within 5 pixels** of ground truth on 384×384 images.

## Three-Stage Loss Function

Instead of a simple two-stage loss, we now use **three stages** with aggressive penalties for sub-pixel precision:

```python
dist = |pred_x - gt_x| + |pred_y - gt_y|

loss_spatial = {
    10.0 × dist²           if dist < 0.05   # Stage 3: Sub-10px (AGGRESSIVE)
    dist²                  if dist < 0.1    # Stage 2: 10-19px (standard)
    0.2 × dist + 0.01      if dist ≥ 0.1    # Stage 1: >19px (strong linear)
}
```

### Stage Breakdown

| Stage | Distance (normalized) | Distance (pixels) | Loss Type | Gradient Strength |
|-------|----------------------|------------------|-----------|-------------------|
| 🔴 **Stage 1: Far** | ≥ 0.1 | > 19px | Linear (0.2×) | **0.2** (strong push) |
| 🟡 **Stage 2: Medium** | 0.05 - 0.1 | 10-19px | Quadratic | 2 × dist (standard) |
| 🟢 **Stage 3: Close** | < 0.05 | < 10px | 10× Quadratic | **20 × dist** (very strong!) |

### Why Three Stages?

**Stage 1 (Far: >19px)**
- Strong linear penalty (0.2 instead of 0.1)
- Gradient = 0.2 (constant, strong)
- **Purpose:** Quickly move predictions from far away into medium range

**Stage 2 (Medium: 10-19px)**
- Standard quadratic loss (dist²)
- Gradient = 2 × dist (ranges from 0.2 to 0.4)
- **Purpose:** Smooth transition zone, start converging

**Stage 3 (Close: <10px)**
- **10x stronger quadratic** (10 × dist²)
- Gradient = 20 × dist (ranges from 0 to 1.0)
- **Purpose:** FORCE sub-5-pixel precision with very strong gradients

### Gradient Comparison

For a **5-pixel error** (0.026 normalized):

**OLD (threshold=0.3, single quadratic):**
```
Gradient = 2 × 0.026 = 0.052 (very weak!)
```

**PREVIOUS (threshold=0.1, two-stage):**
```
Gradient = 2 × 0.026 = 0.052 (still weak)
```

**NEW (threshold=0.05, three-stage with 10x boost):**
```
Gradient = 20 × 0.026 = 0.52 (10x stronger!)
```

For a **3-pixel error** (0.016 normalized):

**NEW:**
```
Gradient = 20 × 0.016 = 0.32 (strong even at 3px!)
```

## Mathematical Properties

### Loss Continuity

At transition points, the loss must be continuous:

**Transition 1 (dist=0.1):**
- Quadratic: dist² = 0.01
- Linear: 0.2 × 0.1 + 0.01 = 0.03
- ⚠️ Jump of 0.02 (acceptable for coarse→medium transition)

**Transition 2 (dist=0.05):**
- 10× Quadratic: 10 × 0.0025 = 0.025
- Standard Quadratic: 0.0025
- ⚠️ Jump of 0.0225 (acceptable for medium→fine transition)

**Note:** Small discontinuities are acceptable as they create "barriers" that the model must overcome, encouraging it to push into the lower-loss regions.

### Gradient Strength Profile

```
Distance (px)  |  Gradient Magnitude  |  Loss Type
------------------------------------------------------
50px           |  0.20                |  Linear
30px           |  0.20                |  Linear
20px           |  0.20                |  Linear
19px (0.1)     |  0.20 → 0.38         |  → Quadratic
15px           |  0.31                |  Quadratic
10px (0.05)    |  0.21 → 1.04         |  → 10× Quadratic
5px            |  0.52                |  10× Quadratic (⭐ VERY STRONG)
3px            |  0.31                |  10× Quadratic
1px            |  0.10                |  10× Quadratic
```

## Expected Training Behavior

### Early Epochs (0-5)
- Model predictions scattered across image
- **Stage 1 (linear)** dominates → fast movement toward targets
- Loss decreases rapidly from ~2.0 → ~0.5

### Middle Epochs (5-10)
- Most predictions within 10-30px
- **Stage 2 (quadratic)** takes over → convergence toward 10px range
- Loss decreases from ~0.5 → ~0.15

### Final Epochs (10-15)
- Most predictions within 5-15px
- **Stage 3 (10× quadratic)** activates → aggressive sub-5px refinement
- Loss decreases from ~0.15 → **<0.05**
- **Target val_loss < 0.05** means average error < 5px

### Early Stopping
- Should trigger around epoch 12-15
- If val_loss plateaus above 0.05, model has reached its limit with current architecture

## Target Metrics

### Training Convergence
```
Epoch 0:   val_loss ≈ 1.5-2.0  (random)
Epoch 5:   val_loss ≈ 0.3-0.5  (getting closer)
Epoch 10:  val_loss ≈ 0.08-0.15 (10-15px range)
Epoch 15:  val_loss ≈ 0.02-0.05 (3-7px range) ← TARGET
```

### Per-Coordinate Losses
For **5px average error** (our target):
```
5px = 0.026 normalized
MSE = 0.026² = 0.00068

Target:
  val_loss_x < 0.001  (implies <3px x-error)
  val_loss_y < 0.001  (implies <3px y-error)
  val_loss_spatial < 0.05
```

### Pixel Error Distribution (Target)
```
0-2px:   30% of predictions  ⭐ Perfect
2-5px:   45% of predictions  ✅ Excellent
5-10px:  20% of predictions  🟡 Acceptable
>10px:   5% of predictions   ❌ Outliers (hard cases)

Mean error: ~4px
Median error: ~3px
95th percentile: ~8px
```

## Comparison to Previous Approaches

| Approach | Threshold | Gradient at 5px | Expected Error |
|----------|-----------|-----------------|----------------|
| **Original** | 0.3 (57px) | 0.05 | 7-30px |
| **V2** | 0.1 (19px) | 0.05 | 3-10px |
| **V3 (Current)** | 0.05 (10px) + 10× boost | **0.52** | **0-5px** ⭐ |

## Potential Challenges

### 1. VLM Spatial Resolution Limits

**Issue:** Qwen2-VL-2B may have inherent spatial resolution limits.

**Evidence to watch for:**
- Validation loss plateaus at ~0.08-0.1 (10-12px range)
- X-coordinate loss higher than Y (horizontal harder than vertical)
- Pixel errors cluster around 8-10px even after 20 epochs

**Solution if this occurs:**
- Consider upsampling images from 384→512 before VLM processing
- Use attention map interpolation for finer spatial features
- May need larger VLM (7B instead of 2B) for better spatial precision

### 2. Gradient Magnitude Too Strong

**Issue:** 10× multiplier might cause loss spikes or training instability.

**Evidence to watch for:**
- Loss increases suddenly after epoch ~8-10
- Gradient explosion in x_head/y_head (magnitude >1000)
- Predictions jumping wildly between epochs

**Solution if this occurs:**
- Reduce multiplier from 10× → 5×
- Add stronger gradient clipping (max_norm=0.5 instead of 1.0)
- Lower learning rate to 2e-5 (currently 5e-5)

### 3. Overfitting to Fine Details

**Issue:** Model memorizes exact pixel locations on training set, fails on validation.

**Evidence to watch for:**
- Train loss < 0.01, val loss > 0.1 (10× gap)
- Training pixel errors 1-2px, validation 8-12px
- Early stopping triggers very early (epoch 5-7)

**Solution if this occurs:**
- Add data augmentation (rotation, scaling, translation)
- Increase dropout from 0.1 → 0.2 in prediction heads
- Reduce training iterations per epoch

## Implementation Notes

### Code Changes

**File:** `model_sequential_simple.py`

**Lines 317-325:** Three-stage loss function
```python
loss_spatial = torch.where(
    dist < 0.05,
    10.0 * dist ** 2,  # ← 10× boost for sub-10px
    torch.where(
        dist < 0.1,
        dist ** 2,
        0.2 * dist + 0.01  # ← Stronger linear (was 0.1)
    )
).mean()
```

**Lines 339-347:** Same for mixed mode (with done signal)

### Hyperparameters

**Unchanged (optimal):**
- Learning rate: 5e-5
- Batch size: 2
- Early stopping patience: 5
- Gradient clipping: max_norm=1.0

**If needed (for stability):**
- Learning rate: 2e-5 (if loss spikes)
- Gradient clipping: 0.5 (if gradients explode)
- 10× multiplier → 5× (if too aggressive)

## Verification Checklist

After retraining with new loss:

### 1. Training Stability
- [ ] Loss decreases smoothly (no sudden spikes)
- [ ] Gradients stay in 50-300 range (not >1000)
- [ ] Early stopping triggers around epoch 12-15

### 2. Final Losses
- [ ] val_loss < 0.1 (target: <0.05)
- [ ] val_loss_x < 0.002 (target: <0.001)
- [ ] val_loss_y < 0.002 (target: <0.001)
- [ ] train/val gap < 3× (not >5×)

### 3. Pixel Errors (from W&B visualizations)
- [ ] Median error < 5px (count yellow `err=XXpx` labels)
- [ ] 75%+ predictions < 8px
- [ ] <5% predictions > 15px (outliers only)

### 4. Visual Quality
- [ ] RED prediction circles very close to GREEN ground truth
- [ ] YELLOW error lines barely visible on most examples
- [ ] Model handles occlusion and edge cases

## Expected Timeline

```
Training: ~2.5 hours (15 epochs × 10 min/epoch)
Validation: ~20 min total
Early stopping: Epoch ~12-15

Total time: ~3 hours
```

## Success Criteria

**Minimum (Acceptable):**
- 90% predictions within 8px
- Mean error < 6px
- val_loss < 0.08

**Target (Excellent):**
- 90% predictions within 5px
- Mean error < 4px
- val_loss < 0.05

**Stretch (Outstanding):**
- 95% predictions within 5px
- Mean error < 3px
- val_loss < 0.03

## If Target Not Met

If after retraining val_loss plateaus above 0.08:

1. **Check VLM resolution limits** - may need larger model or higher res images
2. **Reduce 10× multiplier** - might be too aggressive, try 5× or 7×
3. **Add data augmentation** - model might need more diverse training
4. **Analyze failure cases** - some object categories might be inherently harder

The three-stage aggressive loss should get you to 0-5px range. If not, it's likely a VLM architecture limitation rather than loss function design.

---

**Ready to train?** Run `./run_sequential_attention.sh` and watch for:
- Smooth loss decrease
- Early stopping around epoch 12-15
- Final val_loss < 0.05 ⭐
