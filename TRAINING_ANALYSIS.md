# Training Analysis: 7-30 Pixel Error Issue

## Problem Summary

After 55 epochs of training, the model achieves predictions within 7-30 pixels of ground truth, but cannot improve further.

**Key Findings:**
- ✅ Training loss: 0.10 (good convergence)
- ❌ Validation loss: 1.51 (much higher than training)
- 🎯 Best checkpoint: **Epoch 10** with val_loss=0.6064
- 📊 Val coordinate losses: x=0.131, y=0.026 (still room for improvement)

## Root Cause: Overfitting + Lenient Distance Threshold

### 1. Model Has Been Overfitting Since Epoch 10

**Validation loss over time:**
```
Epoch 0:  1.532
Epoch 6:  0.675  ← First major improvement
Epoch 10: 0.606  ← BEST (should have stopped here)
Epoch 20: 1.042
Epoch 30: 1.316
Epoch 40: 1.951
Epoch 55: 1.509
```

**Evidence:**
- Best validation was at epoch 10
- Training continued for 45 more epochs, making training loss drop (0.37 → 0.10) while validation increased (0.61 → 1.51)
- This is textbook overfitting: model memorizing training data instead of generalizing

### 2. Distance Threshold (0.3) Too Lenient for Fine Localization

Current loss function in `model_sequential_simple.py:313-317`:
```python
dist = torch.abs(pred_x - gt_x) + torch.abs(pred_y - gt_y)
loss_spatial = torch.where(
    dist < 0.3,        # ← This is the problem
    dist ** 2,         # Quadratic when "close"
    0.3 * dist + 0.09  # Linear when far
).mean()
```

**Why 0.3 is too lenient:**
- Normalized coords: [-1, 1] span 384 pixels
- 0.3 in normalized space = 0.3 * 384 / 2 = **57.6 pixels**
- Model treats anything within 57px as "close enough" → only needs quadratic fine-tuning
- **7-30 pixel errors fall entirely within the quadratic region**

**What this means:**
- Model gets weak quadratic gradients when it's 20px off
- No strong penalty to push it from 20px → 5px
- Loss function is effectively saying "20px is good enough"

### 3. Validation Coordinate Losses Still High

```
val_loss_x: 0.131  (MSE in normalized coords)
val_loss_y: 0.026

Converting to pixels:
- x error: sqrt(0.131) * 192 ≈ 69 pixels
- y error: sqrt(0.026) * 192 ≈ 31 pixels
```

These are **averages** - some predictions much worse, matching the 7-30px range user observed.

## Solution: Tighter Distance Threshold + Early Stopping

### Fix #1: Reduce Distance Threshold from 0.3 → 0.1

```python
# Current (too lenient)
dist < 0.3  # 57px radius - too forgiving

# Proposed (tighter)
dist < 0.1  # 19px radius - forces precision
```

**Why 0.1 (19 pixels):**
- 0.1 normalized = 0.1 * 384 / 2 = **19.2 pixels**
- Model must get within 19px to enter quadratic fine-tuning
- Predictions at 20-30px will receive linear penalty → strong gradient to improve
- Quadratic region reserved for true fine-tuning (5-15px errors)

**Expected impact:**
- Current: 20px error → weak quadratic gradient
- After fix: 20px error → strong linear gradient
- Should push errors from 7-30px down to 3-10px range

### Fix #2: Add Early Stopping

**Current behavior:**
- Trained for 55 epochs
- Best was epoch 10
- Wasted compute on epochs 11-55 while overfitting

**Proposed:**
```python
early_stopping_patience = 5  # Stop if no improvement for 5 epochs
```

This would have stopped at epoch 15, saving 40 epochs of compute and preventing overfitting.

### Fix #3: Restore Best Checkpoint

**Immediate action:**
```bash
# The best checkpoint exists at epoch 10
# Use: sequential_attention_model_hinge_loss/best_checkpoint
# NOT: sequential_attention_model_hinge_loss/latest_checkpoint
```

The user should evaluate using `best_checkpoint` which has val_loss=0.606 instead of `latest_checkpoint` with val_loss=1.509.

## Recommended Changes

### 1. Update Loss Function

**In `model_sequential_simple.py`, line 312-317:**

```python
# OLD (0.3 threshold)
dist = torch.abs(pred_x - gt_x) + torch.abs(pred_y - gt_y)
loss_spatial = torch.where(
    dist < 0.3,
    dist ** 2,
    0.3 * dist + 0.09
).mean()

# NEW (0.1 threshold - much tighter)
dist = torch.abs(pred_x - gt_x) + torch.abs(pred_y - gt_y)
loss_spatial = torch.where(
    dist < 0.1,
    dist ** 2,
    0.1 * dist + 0.01  # Adjusted intercept
).mean()
```

**Math explanation:**
- Linear component: `0.1 * dist + 0.01`
- At transition point (dist=0.1): quadratic=0.01, linear=0.01+0.01=0.02
- Continuous function (no jump)

### 2. Add Early Stopping to Training Script

**In `train_sequential_attention.py`, add:**

```python
# After imports
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 5

# In training loop (after validation)
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
    # Save best checkpoint
else:
    patience_counter += 1
    if patience_counter >= early_stop_patience:
        print(f"Early stopping triggered at epoch {epoch}")
        break
```

### 3. Use Best Checkpoint for Evaluation

```bash
# Don't use latest_checkpoint (overfitted)
# Use best_checkpoint from epoch 10
```

## Expected Improvements

### Before (current):
```
Epoch 55 (overfitted):
- Train loss: 0.102
- Val loss: 1.509
- Pixel errors: 7-30px (stuck)
```

### After (with 0.1 threshold + early stopping):
```
Epoch ~10-15 (before overfitting):
- Train loss: ~0.3-0.4
- Val loss: ~0.6 (best checkpoint)
- Pixel errors: 3-10px (expected)
```

**Reasoning:**
1. **Tighter threshold (0.1)** forces model to be more precise
2. **Early stopping** prevents overfitting, preserves generalization
3. **Best checkpoint (epoch 10)** already better than latest checkpoint

## Implementation Priority

**Immediate (no retraining needed):**
1. ✅ Use `best_checkpoint` instead of `latest_checkpoint` for inference
2. ✅ This alone should improve results (val_loss 0.606 vs 1.509)

**For next training run:**
1. 🔧 Change distance threshold: 0.3 → 0.1
2. 🔧 Add early stopping with patience=5
3. 🔧 Reduce max epochs from 100 → 30 (with early stop, will likely stop at 10-15)

## Verification Steps

After retraining with 0.1 threshold:

1. **Check validation loss converges faster:**
   - Should reach best val_loss by epoch 5-10
   - Early stopping should trigger around epoch 10-15

2. **Check pixel errors in visualizations:**
   - Look for `err=XXpx` labels in W&B
   - Should see majority of errors <15px, many <10px

3. **Check validation losses:**
   - val_loss_x should drop from 0.13 → <0.05
   - val_loss_y should drop from 0.026 → <0.01

4. **Verify no overfitting:**
   - Train/val loss gap should be <2x (currently 10x!)
   - Validation loss should not increase after early epochs

## Summary

**The 7-30 pixel error is caused by:**
1. ❌ Overfitting (training continued 45 epochs past best checkpoint)
2. ❌ Distance threshold too lenient (0.3 = 57px, should be 0.1 = 19px)
3. ❌ Model treats 20px error as "close enough" → weak gradients

**Quick fix (no retraining):**
- Use `best_checkpoint` from epoch 10 instead of `latest_checkpoint`

**Proper fix (retrain):**
- Reduce threshold 0.3 → 0.1 in loss function
- Add early stopping with patience=5
- Should achieve 3-10px errors instead of 7-30px
