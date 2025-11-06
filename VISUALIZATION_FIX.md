# Critical Visualization Fix: Ground Truth Done Signal

## The Bug

**Problem:** Validation images showed incorrect `gt_done` values that didn't match the actual image.

**Example of bug:**
```
Image shows: 5/20 objects marked (clearly not done)
Caption says: "gt_done=0"
Wait, what? That's correct...

BUT the gt_done=0 came from a DIFFERENT image in the done batch
where 19/20 objects were marked!
```

## Root Cause

**Dual-task training creates TWO batches per iteration:**

```python
# Batch 1: Coordinate prediction (regression mode)
images_coord, num_marked_coord = prepare_batch(batch, mode='regression')
# Randomly picks k=5, shows 5/20 marked

# Batch 2: Done signal (classification mode)
images_done, num_marked_done, gt_done = prepare_batch(batch, mode='classification')
# Randomly picks k=19, gt_done=0 (not yet done)

# Visualization used:
# - images_coord (k=5 marked)
# - gt_done from done batch (k=19 context)
# MISMATCH! ❌
```

**Result:** Caption showed `gt_done=0` (from k=19 context) on an image with k=5 marked. Completely misleading!

## The Fix

**Compute `true_done` from the ACTUAL image being displayed:**

```python
# BEFORE (WRONG):
true_done = gt_done[i].item()  # From done batch - wrong context!

# AFTER (CORRECT):
total_objects = len(all_objects_coord[i])
marked_in_image = num_marked_coord[i]
true_done = 1.0 if marked_in_image >= total_objects else 0.0
```

## Updated Caption Format

**Before:**
```
seagulls | Marked:5 | Pred:(x,y,done=0.23) | GT:(x,y,done=0)
```
- Not clear how many total objects
- `done=0` misleading without context

**After:**
```
seagulls | 5/20 marked | Pred:(x,y,done=0.23) | GT:(x,y,done=0)
```
- ✅ Shows 5/20 - clearly NOT done
- ✅ `done=0` makes sense now
- ✅ Can see progress toward completion

## Examples

### Not Done (5/20 marked)
```
Caption: "grapes | 5/20 marked | Pred:(x,y,done=0.08) | GT:(x,y,done=0)"

Analysis:
- 5 out of 20 marked → 25% done
- gt_done=0 ✓ Correct! Not all marked yet
- pred_done=0.08 ✓ Good! Model correctly predicts not done
```

### Almost Done (19/20 marked)
```
Caption: "peaches | 19/20 marked | Pred:(x,y,done=0.85) | GT:(x,y,done=0)"

Analysis:
- 19 out of 20 marked → 95% done
- gt_done=0 ✓ Correct! One object remains
- pred_done=0.85 ✓ Model is uncertain (close to threshold)
```

### Completely Done (20/20 marked)
```
Caption: "seagulls | 20/20 marked | Pred:(x,y,done=0.98) | GT:(x,y,done=1)"

Analysis:
- 20 out of 20 marked → 100% done
- gt_done=1 ✓ Correct! All marked
- pred_done=0.98 ✓ Excellent! Model confidently predicts done
```

## Why This Matters

### For Training Monitoring

**Before fix:**
- Hard to tell if model is learning done signal correctly
- Visualization and caption didn't match
- Debugging was confusing

**After fix:**
- ✅ Can immediately see if model's done prediction makes sense
- ✅ Ratio (5/20) shows progress clearly
- ✅ Easy to spot errors (e.g., pred_done=0.9 when only 3/30 marked)

### For Done Signal Evaluation

The done signal is trained separately from coordinates. We need to verify:
1. Model predicts `done=0` when objects remain
2. Model predicts `done=1` when all marked
3. Model handles edge cases (19/20 marked)

**Without fix:** Can't trust the visualization!
**With fix:** Can clearly see model performance.

## Code Changes

**File:** `train_sequential_attention.py`

**Lines 356-363:** Compute true_done from coord image
```python
# CRITICAL FIX: Compute true_done based on COORD image (num_marked_coord)
# NOT from the separate done batch!
if i < len(all_objects_coord):
    total_objects = len(all_objects_coord[i])
    marked_in_image = num_marked_coord[i]
    true_done = 1.0 if marked_in_image >= total_objects else 0.0
else:
    true_done = 0.0
```

**Lines 446-450:** Show ratio in caption
```python
total_objs = len(all_objects_coord[i]) if i < len(all_objects_coord) else 0
caption = f"{cat} | {num_marked_coord[i]}/{total_objs} marked | ..."
```

## Verification

After this fix, check validation images for:

✅ **Consistency:**
- `3/20 marked` → `gt_done=0` ✓
- `20/20 marked` → `gt_done=1` ✓

✅ **Model accuracy:**
- Low marked ratio → pred_done should be low
- High marked ratio → pred_done should approach 1.0

✅ **Edge cases:**
- `19/20 marked` → pred_done anywhere from 0.3-0.9 is reasonable (ambiguous)
- `1/30 marked` → pred_done should be close to 0

## Impact on Training

**This fix is VISUALIZATION ONLY:**
- Training losses not affected (they use correct ground truth)
- Model learns correctly
- Only the W&B images were showing wrong context

**But it's critical for:**
- ✅ Monitoring training progress
- ✅ Debugging done signal issues
- ✅ Understanding model behavior
- ✅ Catching problems early

## Summary

**Bug:** Showed `gt_done` from classification batch on regression batch image → mismatch!

**Fix:** Compute `true_done` directly from the image being displayed → correct context!

**Result:** Now you can trust the visualizations and accurately monitor done signal learning!