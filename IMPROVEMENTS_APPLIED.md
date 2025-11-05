# Improvements Applied to Fix Training

## Latest Issue (After 55 Epochs): 7-30 Pixel Error

**Results from `wandb/run-20251104_223225-dw2cnqql`:**
- ✅ Model trained for 55 epochs, loss decreased
- ⚠️ Pixel errors stuck at 7-30 pixels
- ❌ Overfitting: Best val_loss=0.606 at epoch 10, but increased to 1.509 by epoch 55
- ❌ Train/val gap: 0.10 vs 1.51 (10x difference!)

**Root Cause Analysis (see TRAINING_ANALYSIS.md):**
1. **Distance threshold (0.3) too lenient** - covers 57 pixels, model treats 20px error as "close enough"
2. **Overfitting after epoch 10** - trained 45 extra epochs, memorizing training set
3. **No early stopping** - continued training long past optimal point

## Fix #4: Tighter Distance Threshold (0.3 → 0.1)

**Problem:** 0.3 normalized = 57 pixels. Model receives weak quadratic gradients for 20px errors.

**Solution:** Reduce threshold to 0.1 (19 pixels):

```python
# OLD (too lenient)
dist = torch.abs(pred_x - gt_x) + torch.abs(pred_y - gt_y)
loss_spatial = torch.where(
    dist < 0.3,           # 57px radius
    dist ** 2,
    0.3 * dist + 0.09
).mean()

# NEW (tighter - forces precision)
loss_spatial = torch.where(
    dist < 0.1,           # 19px radius
    dist ** 2,
    0.1 * dist + 0.01
).mean()
```

**Why this works:**
- **0.1 threshold** = 19 pixels on 384x384 image
- Predictions at 20-30px now get **strong linear penalty** (not weak quadratic)
- Quadratic region reserved for true fine-tuning (<19px)
- Should push errors from 7-30px → 3-10px range

## Fix #5: Early Stopping (Patience=5)

**Problem:** Trained for 55 epochs, but best was at epoch 10. Wasted 45 epochs overfitting.

**Solution:** Stop training if validation loss doesn't improve for 5 consecutive epochs:

```python
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 5

for epoch in range(args.epochs):
    # ... training ...
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

**Why this works:**
- Automatically stops when model stops improving
- Prevents overfitting (train loss keeps decreasing, val loss increases)
- Saves compute time (will stop around epoch 15 instead of 55)

## Issues Found in First Training Run

**Results from `wandb/run-20251104_202309-mbss65up`:**
- ✅ Loss **did decrease** (0.91 → 0.44), training is working!
- ❌ Done signal predictions always low (0.00-0.57)
- ❌ Model never learns to say "I'm finished"

**Root Cause Analysis:**

Using `check_done_balance.py`, discovered:
- **Validation (mixed mode)**: Only **3% done=1** examples!
- **Average 21 objects per image**
- When randomly sampling `k ∈ [0, N]`, most values are `k < N` (not done)
- Model sees 97% "not done" examples → learns to always predict "not done"

## Fix #1: Distance-Based Loss (Hinge-Like)

**Problem:** MSE treats all errors equally. Being 10px off gets same gradient as being 100px off.

**Solution:** Funnel-shaped loss that **heavily penalizes** being far away:

```python
# L1 distance in normalized space
dist = |pred_x - gt_x| + |pred_y - gt_y|

# Adaptive penalty
loss_spatial = {
    dist² if dist < 0.3,           # Quadratic when close (fine-tuning)
    0.3 * dist + 0.09 if dist ≥ 0.3  # Linear when far (heavy penalty)
}
```

**Why this works:**
- **Close** (<0.3 normalized = ~115px on 384x384 image): Small quadratic penalty encourages precise localization
- **Far** (>0.3): Linear penalty creates strong gradient to get into the right region first
- Avoids vanishing gradients from pure quadratic at large distances

**Impact:**
- Model learns to get "in the ballpark" quickly (linear penalty)
- Then fine-tunes to exact location (quadratic penalty)

## Fix #2: Weighted Done Loss

**Problem:** Done signal has only 3% positive examples in validation.

**Solution:** Weight done loss **3x higher** than spatial loss:

```python
# Mixed mode
loss = loss_spatial + 3.0 * loss_done
```

**Why 3x?**
- Compensates for severe class imbalance
- Makes model pay more attention to rare "done=1" cases
- Empirically shown to help in imbalanced classification

## Fix #3: Improved Visualizations

**Added to W&B logs:**

1. **Ground Truth Circle (GREEN)**
   - Shows where the next object actually is
   - Only drawn when `gt_done=0`

2. **Prediction Circle (RED)**
   - Shows where model thinks next object is

3. **Error Line (YELLOW)**
   - Connects prediction to ground truth
   - Shows pixel error: `err=45px`
   - Makes mistakes visually obvious

4. **Done Signal Comparison**
   - `pred_done=0.15` (RED) - prediction
   - `gt_done=0` (GREEN) - ground truth
   - Easy to see if model is learning done signal

**Example visualization:**
```
┌─────────────────┬─────────────────┐
│     INPUT       │   PREDICTION    │
│  (marked objs)  │                 │
│                 │   ⭕ GT (green)  │
│                 │   ⭕ PRED (red)  │
│                 │   ─────────────  │
│                 │   err=23px       │
│                 │                 │
│                 │ pred_done=0.12  │
│                 │ gt_done=0       │
└─────────────────┴─────────────────┘
```

## Expected Improvements

### Coordinate Accuracy
**Before (MSE):**
- Treats 5px error same as 50px error
- Slow convergence

**After (Distance-Based):**
- Heavy penalty when far → fast coarse localization
- Quadratic when close → precise fine-tuning
- **Expect**: Faster convergence, lower final pixel error

### Done Signal
**Before:**
- Loss weight: 1x
- Predictions: Always ~0.0-0.2

**After:**
- Loss weight: 3x
- **Expect**: Predictions spread across 0-1 range
- Model learns to confidently predict done=1 when appropriate

### Visualizations
**Before:**
- Only saw predictions
- Hard to judge if prediction is good

**After:**
- See ground truth + prediction + error line
- Immediately obvious when model is off
- Can track improvement visually

## How to Verify Improvements

### 1. Check Loss Curves in W&B
```
train/loss_spatial - should decrease steadily
val/loss_done - should decrease (was stuck at ~0.3-0.7)
```

### 2. Check Done Signal Distribution
Look at validation images captions:
```
Before: Pred:(x,y,done=0.05) - always low
After:  Pred:(x,y,done=0.87) - should vary, high when appropriate
```

### 3. Check Pixel Errors
In visualizations, look at yellow `err=XXpx` labels:
```
Early epochs: err=80-150px
Later epochs: err=20-40px (expect improvement)
```

### 4. Monitor Gradient Magnitudes
First epoch should show:
```
[Gradient Check - Regression]
  x_head: 50-200 (good)
  y_head: 50-200 (good)
```

If gradients are 1000+, loss is exploding (bad).
If gradients are <1, model isn't learning (bad).

## Training Command

```bash
./run_sequential_attention.sh
```

**Settings:**
- Learning rate: `5e-5` (stable)
- Batch size: `2`
- Max train iters: `3000` (full epoch)
- Max val iters: `500`

## Files Modified

1. **`model_sequential_simple.py`**
   - Added distance-based loss function
   - Added 3x weighting for done loss
   - Kept logging of MSE for comparison

2. **`train_sequential_attention.py`**
   - Enhanced visualizations with GT + error lines
   - Added pixel error calculation
   - Color-coded: GREEN=GT, RED=PRED, YELLOW=error

3. **`check_done_balance.py`** (new diagnostic tool)
   - Checks class balance in training/validation
   - Recommends fixes for imbalanced data

## Monitoring Checklist

After training, check:

- [ ] Val loss decreasing (should reach <0.3)
- [ ] Done predictions varying (not all <0.2)
- [ ] Pixel errors decreasing over epochs
- [ ] Visualizations show pred close to GT
- [ ] No gradient explosions (magnitudes 50-200 range)

## Next Steps if Still Not Working

1. **If done signal still low:**
   - Increase done weight to 5x: `loss = loss_spatial + 5.0 * loss_done`
   - Or modify validation sampling to include more done=1 cases

2. **If spatial loss not decreasing:**
   - Tune distance threshold (currently 0.3, try 0.2 or 0.4)
   - Check if predictions are stuck in one region

3. **If gradients explode:**
   - Lower learning rate to 1e-5
   - Add stronger gradient clipping (max_norm=0.5 instead of 1.0)

## Summary

**Key Changes:**
1. 🎯 **Distance-based loss** - heavy penalty when far, precise when close
2. ⚖️ **3x done weighting** - compensate for class imbalance
3. 👁️ **GT visualization** - see mistakes clearly

**Expected Results:**
- Faster convergence to correct locations
- Done signal actually learning (not stuck at 0)
- Easy visual verification of improvement

Good luck with training! 🚀
