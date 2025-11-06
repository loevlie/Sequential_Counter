# Nearest-Neighbor Loss Mode

## Overview

Added an **optional nearest-neighbor matching mode** that allows the model to predict ANY unmarked object, instead of forcing it to predict objects in a specific order.

## How It Works

### Standard Mode (Default)
```
Objects: [A, B, C, D, E]
Marked:  [A, B]          (k=2)
Target:   C              (must predict C exactly!)
Loss:    |pred - C|      (distance to C)
```

**Problem:** Model is penalized even if it correctly predicts D or E.

### Nearest-Neighbor Mode (NEW)
```
Objects: [A, B, C, D, E]
Marked:  [A, B]          (k=2)
Unmarked: [C, D, E]
Prediction: (x, y)
Target:    closest of [C, D, E] to (x, y)  ← Dynamic!
Loss:     |pred - target|  (distance to nearest unmarked)
```

**Benefit:** Model can pick any unmarked object and only needs to be close to ANY valid target.

## When to Use

### Use Nearest-Neighbor Mode When:
1. **Objects are densely clustered** - hard to predict exact order
2. **Order doesn't matter** - just want accurate localization
3. **Model struggles with ordering** - keeps picking "wrong" objects

### Use Standard Mode When:
1. **Objects are well-separated** - clear reading order
2. **Order matters** - want systematic counting
3. **Benchmark comparison** - standard mode is more constrained

## Implementation

### Flag
```bash
# Enable nearest-neighbor mode
./run_sequential_attention.sh --use_nearest_neighbor_loss

# Or in training script
python train_sequential_attention.py \
    --data_root /path/to/data \
    --use_nearest_neighbor_loss
```

### Code Changes

**Model (`model_sequential_simple.py`):**
- Added `use_nearest_neighbor_loss` parameter to `__init__`
- Added `find_nearest_unmarked()` method
- Modified forward pass to match predictions to nearest unmarked object when enabled
- Standard 3-stage aggressive loss still applied (0-5px target)

**Training (`train_sequential_attention.py`):**
- Added `--use_nearest_neighbor_loss` command-line flag
- Modified `prepare_batch()` to return all object coordinates
- Updated forward calls to pass `all_objects` parameter

## Mathematical Difference

### Standard Loss
```python
# Target is fixed (next in order)
target_x, target_y = gt_x, gt_y
dist = |pred_x - target_x| + |pred_y - target_y|
loss = three_stage_loss(dist)
```

### Nearest-Neighbor Loss
```python
# Target is dynamic (nearest unmarked)
unmarked = all_objects[k:]  # Objects after marked
distances = [|pred - obj| for obj in unmarked]
nearest_idx = argmin(distances)
target_x, target_y = unmarked[nearest_idx]
dist = |pred_x - target_x| + |pred_y - target_y|
loss = three_stage_loss(dist)
```

**Key:** The three-stage aggressive loss function is UNCHANGED. Only the target selection changes.

## Expected Behavior

### Training Dynamics

**Standard Mode:**
- Model learns to predict objects in reading order
- Penalized for "skipping" objects
- More constrained, potentially harder to learn

**Nearest-Neighbor Mode:**
- Model learns to predict ANY unmarked object accurately
- Not penalized for order mistakes
- More flexible, potentially easier to learn
- May achieve lower spatial loss, but less systematic

### Validation Metrics

**Both modes report same metrics:**
- `val_loss_x`, `val_loss_y` - coordinate accuracy
- `val_loss_done` - completion signal accuracy
- `val_loss_spatial` - combined spatial loss

**Expected difference:**
- Nearest-neighbor mode: Lower spatial loss (easier problem)
- Standard mode: Higher spatial loss (harder constraint)

**Important:** Metrics are NOT directly comparable between modes!

## Pros and Cons

### Nearest-Neighbor Mode

**Pros:**
- ✅ More flexible - model chooses easiest target
- ✅ Lower spatial loss - less constrained
- ✅ May work better for dense/cluttered scenes
- ✅ Matches how humans count (not always systematic)

**Cons:**
- ❌ Less systematic - might miss objects
- ❌ Harder to detect "done" signal (order unclear)
- ❌ Not comparable to standard benchmarks
- ❌ May learn to always pick closest object (lazy strategy)

### Standard Mode

**Pros:**
- ✅ Systematic counting - forced order
- ✅ Matches standard benchmarks
- ✅ Clearer "done" signal (when k==N)
- ✅ More human-interpretable predictions

**Cons:**
- ❌ Harder to learn - strict constraint
- ❌ Penalized even for "correct" predictions (wrong order)
- ❌ May struggle with dense clusters

## Reverting

To revert to standard mode:

### Option 1: Remove flag
```bash
# Simply don't use --use_nearest_neighbor_loss flag
./run_sequential_attention.sh
```

### Option 2: Undo code changes
```bash
git diff model_sequential_simple.py
git diff train_sequential_attention.py

# If needed
git checkout model_sequential_simple.py
git checkout train_sequential_attention.py
```

All nearest-neighbor code is gated behind the `use_nearest_neighbor_loss` flag, so standard mode is unaffected.

## Testing

### Quick Test (Single Epoch)
```bash
python train_sequential_attention.py \
    --data_root /path/to/FSC147 \
    --epochs 1 \
    --max_train_iters 50 \
    --max_val_iters 10 \
    --use_nearest_neighbor_loss \
    --output_dir test_nearest_neighbor
```

**Check:**
- Training runs without errors
- Loss decreases
- Predictions appear reasonable in W&B images

### Full Training
```bash
./run_sequential_attention.sh --use_nearest_neighbor_loss
```

**Compare:**
- Train 2 runs in parallel (one with flag, one without)
- Compare `val_loss_spatial` after 10 epochs
- Check which converges faster
- Inspect visualizations for systematic vs flexible counting

## Expected Results

### Nearest-Neighbor Mode
```
Epoch 10:
  val_loss_spatial: ~0.03-0.05  (lower - easier problem)
  val_loss_done: ~0.4-0.5  (similar or slightly worse)
  Pixel errors: 2-5px  (very accurate)
  Behavior: Picks nearest unmarked object (not systematic)
```

### Standard Mode
```
Epoch 10:
  val_loss_spatial: ~0.05-0.08  (higher - harder constraint)
  val_loss_done: ~0.3-0.4  (similar or slightly better)
  Pixel errors: 3-7px  (accurate but more constrained)
  Behavior: Predicts in reading order (systematic)
```

## Recommendation

### Try Standard Mode First
- More interpretable
- Matches benchmarks
- Forces systematic counting

### Use Nearest-Neighbor If:
- Standard mode spatial loss stuck >0.1
- Objects are very densely clustered
- Order truly doesn't matter for your application

**Note:** You can train both and compare! The code is designed to make switching trivial (just a command-line flag).

## Visualization Changes

### Standard Mode
```
Label: "GT" (Ground Truth)
Shows: The next object in reading order
Caption: "GT:(x,y,done=0)"
```

### Nearest-Neighbor Mode
```
Label: "NN" (Nearest Neighbor)
Shows: The actual nearest unmarked object to prediction
Caption: "NN:(x,y,done=0)"
```

**Important:** The green circle and yellow error line now show the **actual target used for loss computation**, not the original ordered target!

**Why this matters:**
- In nearest-neighbor mode, the model might predict object D when the "ordered" target is C
- If we showed C (ordered), the error line would be misleading
- Now we show D (nearest), so the error line accurately reflects the loss

**Example:**
```
Objects: A=(-0.5,-0.5), B=(0.5,-0.5), C=(-0.5,0.5), D=(0.5,0.5)
Marked: [A, B] (k=2)
Ordered target: C (next in order)
Prediction: (0.48, 0.52)
Nearest: D (dist=0.05)

Standard mode visualization:
  Green circle at C (-0.5, 0.5)
  Error = 1.4 (distance to C)
  Label: "GT"

Nearest-neighbor mode visualization:
  Green circle at D (0.5, 0.5)  ← Correct!
  Error = 0.05 (distance to D)   ← Matches loss!
  Label: "NN"
```

## Files Changed

1. ✅ `model_sequential_simple.py`
   - Added `use_nearest_neighbor_loss` param
   - Added `find_nearest_unmarked()` method
   - Modified forward pass (lines 347-370, 402-424)

2. ✅ `train_sequential_attention.py`
   - Added `--use_nearest_neighbor_loss` flag
   - Modified `prepare_batch()` to return all_objects
   - Updated forward calls to pass all_objects
   - **Fixed visualization** to show actual nearest-neighbor target (lines 327-345, 357-415)

3. ✅ `NEAREST_NEIGHBOR_MODE.md` (this file)
   - Documentation of new feature

**No other files modified** - easy to revert if needed!
