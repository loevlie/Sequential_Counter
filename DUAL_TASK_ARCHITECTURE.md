# Dual-Task Training Architecture

## Overview

The model now uses **two separate optimizers** with **independent backward passes** for coordinate prediction and done signal classification. This is Option B from our design discussion.

## Why Dual-Task?

**Problem with single loss:**
- Coordinate prediction (regression) and done signal (classification) have different learning dynamics
- They converge at different rates
- Combined loss `loss = loss_spatial + 3.0 * loss_done` couples their gradients
- One task might overfit while the other is still learning

**Solution: Separate tasks completely:**
- Each task has its own optimizer
- Each task has its own backward pass
- Each task tracks its own best checkpoint
- Early stopping only when BOTH tasks stop improving

## Architecture

### Two Optimizers

**Optimizer 1: Coordinate Prediction**
```python
optimizer_coord = AdamW([
    {'params': model.model.parameters(), 'lr': 5e-5},  # VLM base
    {'params': model.x_head.parameters(), 'lr': 5e-5},  # X head
    {'params': model.y_head.parameters(), 'lr': 5e-5}   # Y head
])
```

**Optimizer 2: Done Signal**
```python
optimizer_done = AdamW([
    {'params': model.model.parameters(), 'lr': 5e-5},  # VLM base (shared!)
    {'params': model.done_head.parameters(), 'lr': 5e-5}  # Done head
])
```

**Note:** VLM parameters appear in BOTH optimizers, but that's okay! Each optimizer only steps when its task is trained.

### Training Loop (Per Iteration)

```python
for batch in train_loader:
    # ========== TASK 1: COORDINATE PREDICTION ==========
    images_coord, num_marked_coord, gt_x, gt_y = prepare_batch(batch, mode='regression')

    optimizer_coord.zero_grad()
    outputs_coord = model(images_coord, num_marked_coord, gt_x=gt_x, gt_y=gt_y, gt_done=None)
    loss_coord = outputs_coord['loss']  # Distance-based loss
    loss_coord.backward()
    clip_grad_norm_(coord_params, max_norm=1.0)
    optimizer_coord.step()

    # ========== TASK 2: DONE SIGNAL ==========
    images_done, num_marked_done, gt_done = prepare_batch(batch, mode='classification')

    optimizer_done.zero_grad()
    outputs_done = model(images_done, num_marked_done, gt_x=None, gt_y=None, gt_done=gt_done)
    loss_done = outputs_done['loss']  # Binary cross-entropy
    loss_done.backward()
    clip_grad_norm_(done_params, max_norm=1.0)
    optimizer_done.step()
```

**Key points:**
- **TWO forward passes** per iteration (one per task)
- **TWO backward passes** per iteration (independent gradients)
- VLM gets updated by BOTH tasks (accumulates gradients from both)
- Task-specific heads only updated by their respective task

### Validation

Both tasks evaluated separately each epoch:

```python
for batch in val_loader:
    # Evaluate coord task
    outputs_coord = model(batch, mode='regression')
    loss_coord = outputs_coord['loss']

    # Evaluate done task
    outputs_done = model(batch, mode='classification')
    loss_done = outputs_done['loss']

    # Track separately
    total_loss_coord += loss_coord
    total_loss_done += loss_done
```

### Checkpointing

**Three checkpoint types:**

1. **`best_coord_checkpoint/`** - Best coordinate prediction model
   - Saved when `val_loss_coord` improves
   - Contains full model (VLM + all heads)

2. **`best_done_checkpoint/`** - Best done signal model
   - Saved when `val_loss_done` improves
   - Contains full model (VLM + all heads)

3. **`latest_checkpoint/`** - Final epoch model
   - Always saved each epoch
   - For resuming training

### Early Stopping

**Dual-task early stopping:**
```python
if patience_coord >= 5 AND patience_done >= 5:
    print("Both tasks stopped improving - early stop!")
    break
```

**Why AND not OR:**
- One task might improve while the other plateaus
- We want to train until BOTH tasks have converged
- Example: Done signal might converge at epoch 8, but coords keep improving until epoch 15

## Training Metrics

### Logged to W&B

```
train/loss_coord  - Coordinate prediction loss (distance-based)
train/loss_x      - MSE for x coordinate (logging only)
train/loss_y      - MSE for y coordinate (logging only)
train/loss_done   - Done signal loss (binary cross-entropy)

val/loss_coord    - Validation coordinate loss
val/loss_x        - Validation x MSE
val/loss_y        - Validation y MSE
val/loss_done     - Validation done loss
```

### Console Output

```
Epoch 10:
  Train - coord: 0.0450, done: 0.1523, x: 0.0023, y: 0.0008
  Val   - coord: 0.0523, done: 0.1801, x: 0.0028, y: 0.0011
✅ New best COORD model! Val loss_coord: 0.0523
  Coord: no improvement for 0/5 epochs
  Done: no improvement for 3/5 epochs
```

## Comparison: Before vs After

### Before (Single Combined Loss)

```python
# Single optimizer
optimizer = AdamW(all_params, lr=5e-5)

# Mixed mode training
for batch in train_loader:
    images, gt_x, gt_y, gt_done = prepare_batch(batch, mode='mixed')
    outputs = model(images, gt_x=gt_x, gt_y=gt_y, gt_done=gt_done)
    loss = outputs['loss_spatial'] + 3.0 * outputs['loss_done']  # Combined!
    loss.backward()
    optimizer.step()

# Single early stopping
if val_loss stops improving:
    stop training
```

**Problems:**
- Coord and done losses coupled
- 3x weighting arbitrary
- One task can drag down the other
- Single convergence criterion

### After (Dual-Task)

```python
# Two optimizers
optimizer_coord = AdamW(coord_params, lr=5e-5)
optimizer_done = AdamW(done_params, lr=5e-5)

# Dual-task training
for batch in train_loader:
    # Task 1: Coordinates
    loss_coord.backward()
    optimizer_coord.step()

    # Task 2: Done
    loss_done.backward()
    optimizer_done.step()

# Dual early stopping
if val_loss_coord AND val_loss_done stop improving:
    stop training
```

**Benefits:**
- ✅ Independent learning dynamics
- ✅ No arbitrary weighting
- ✅ Each task optimizes at its own pace
- ✅ Separate convergence tracking
- ✅ Can analyze which task is harder

## Expected Behavior

### Coordinate Task
- **Converges**: 10-15 epochs
- **Final loss**: 0.02-0.05 (with aggressive 3-stage loss)
- **Pixel errors**: 0-5px target
- **Learning curve**: Smooth decrease

### Done Task
- **Converges**: 5-10 epochs (classification is faster)
- **Final loss**: 0.15-0.30 (binary cross-entropy)
- **Accuracy**: >95% done signal correctness
- **Learning curve**: May have plateaus due to class imbalance (97% not-done)

### VLM Base
- Gets gradients from **BOTH** tasks
- Updated twice per iteration
- Learns features useful for both counting and completion detection

## Inference

At inference time, **use separate checkpoints**:

```python
# For best coordinate prediction
model.load_pretrained('best_coord_checkpoint')
coords = model.predict_coordinates(image, num_marked)

# For best done signal
model.load_pretrained('best_done_checkpoint')
is_done = model.predict_done(image, num_marked)

# Or use latest for both
model.load_pretrained('latest_checkpoint')
coords, is_done = model.predict(image, num_marked)
```

**Recommendation:** Use `best_coord_checkpoint` since coordinate accuracy is the primary goal.

## Advantages of Dual-Task Setup

1. **Cleaner separation** - No magic weighting (3.0x) needed
2. **Independent convergence** - Each task finds its own optimum
3. **Better monitoring** - See which task is struggling
4. **Flexible stopping** - Can stop when both converge
5. **Easier debugging** - Isolate issues to specific task
6. **Research insight** - Learn which task is harder for the model

## Implementation Notes

### VLM Parameter Sharing

Q: "VLM params in both optimizers - won't this cause issues?"

A: No! Here's why:
```python
# Iteration N
optimizer_coord.zero_grad()  # Zeros VLM grads
loss_coord.backward()        # VLM gets coord gradients
optimizer_coord.step()       # VLM updated with coord gradients

optimizer_done.zero_grad()   # Zeros VLM grads again
loss_done.backward()         # VLM gets done gradients
optimizer_done.step()        # VLM updated with done gradients

# Net effect: VLM sees gradients from both tasks!
```

The VLM is updated twice per iteration, once per task. This is intentional and beneficial!

### Task-Specific Heads

- `x_head`, `y_head` only updated by coord optimizer
- `done_head` only updated by done optimizer
- Ensures heads specialize for their specific task

### Memory Usage

**Concern:** "Two forward passes doubles memory?"

**Reality:**
- Batch size = 2
- Two sequential forward passes with same batch
- Memory freed between passes
- Peak memory similar to before

**Speed:**
- 2× forward passes per iteration
- But simpler than mixed mode
- Still fast (~3 it/s expected)

## Files Changed

1. **`train_sequential_attention.py`**
   - Created `train_epoch_dual_task()` function
   - Updated `validate()` to compute both tasks
   - Two optimizers: `optimizer_coord`, `optimizer_done`
   - Separate early stopping trackers
   - Three checkpoint types

2. **`model_sequential_simple.py`**
   - No changes needed! Model supports both modes
   - Forward accepts `gt_x/gt_y` OR `gt_done` (not both)

3. **`DUAL_TASK_ARCHITECTURE.md`** (this file)
   - Complete documentation

## Training Command

Same as before:
```bash
./run_sequential_attention.sh
```

The script automatically uses dual-task training now!

## Monitoring in W&B

Look for these patterns:

**Healthy training:**
```
val/loss_coord: Decreasing smoothly
val/loss_done: Decreasing (may plateau earlier)
Best coord epoch: ~12-15
Best done epoch: ~8-10
```

**Overfitting:**
```
train/loss_coord: Still decreasing
val/loss_coord: Increasing (red flag!)
→ Early stopping should trigger
```

**Task imbalance:**
```
val/loss_coord: 0.05 (great!)
val/loss_done: 0.40 (struggling)
→ Done task needs more attention (but separate optimizer helps!)
```

## Summary

**Option B (Dual-Task) gives you:**
- ✅ Clean separation of concerns
- ✅ Independent optimization
- ✅ Separate best checkpoints
- ✅ Better interpretability
- ✅ Flexibility to tune tasks independently

**Compared to Option A (fully independent):**
- Still shares VLM forward pass (efficient)
- Both tasks train simultaneously
- VLM learns from both tasks (better features)

**Compared to mixed mode (previous):**
- No arbitrary loss weighting
- Each task converges at natural rate
- Easier to debug and analyze
- More principled approach

This is the recommended architecture for dual-task learning!
