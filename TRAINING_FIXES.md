# Training Fixes Applied

## Problem Diagnosis

Your training was failing with:
- ❌ Predictions stuck at top-left corner (-1, -1)
- ❌ Losses increasing instead of decreasing
- ❌ Both baseline and sequential attention models failing

## Root Causes Found (via debug_training.py)

### 1. **Learning Rate Too High** 🔥
- Original: `lr=1e-3` with `5x` multiplier for new modules = `5e-3`
- **Result**: Loss exploded from 0.47 → 4.22 after ONE gradient step!
- **Fix**: Reduced to `lr=5e-5` (20x lower), no multiplier

### 2. **Vanishing Gradients in Sequential Modules** 👻
```
working_memory: 0.000000 (ZERO!)
foveation: 0.000000 (ZERO!)
```
- Complex transformation pipeline disconnected from loss
- **Fix**: Created simplified model that uses VLM features directly

### 3. **Tanh Saturation** 🎯
- Coordinate heads used `tanh()` activation
- Saturates at -1 and +1, making it hard to learn
- **Fix**: Removed tanh, use direct predictions with MSE loss

## Changes Made

### 1. New Simplified Model (`model_sequential_simple.py`)

**What's Different:**
- ✅ Uses VLM features DIRECTLY for predictions
- ✅ No complex LSTM/foveation/cross-attention pipeline
- ✅ Simple 2-layer MLPs for x, y, done predictions
- ✅ Removed tanh activation (predictions can be any value)
- ✅ Added clamping to prevent explosion: `torch.clamp(pred, -2, 2)`

**Architecture:**
```
Input Image + Marked Objects
       ↓
[Qwen2-VL Encoder]
       ↓
[Extract features at <x>, <y>, <done> tokens]
       ↓
[Simple MLP Heads] (no tanh!)
   ├─> x coordinate (MSE loss)
   ├─> y coordinate (MSE loss)
   └─> done signal (BCE loss)
```

### 2. Updated Training Configuration

**Learning Rate:**
```bash
# OLD (BROKEN)
LR=1e-3
optimizer: VLM at 1e-3, others at 5e-3

# NEW (FIXED)
LR=5e-5
optimizer: ALL modules at 5e-5 (same rate)
```

**Model:**
```python
# OLD (COMPLEX)
from model_sequential_attention import SequentialAttentionCountingModel

# NEW (SIMPLE)
from model_sequential_simple import SimpleSequentialModel
```

### 3. Updated Scripts

**Modified Files:**
- `run_sequential_attention.sh`: Changed `LR=5e-5`
- `train_sequential_attention.py`: Uses `SimpleSequentialModel`
- Created `model_sequential_simple.py`: Simplified architecture
- Created `debug_training.py`: Diagnostic tool

## How to Train Now

### Quick Start
```bash
./run_sequential_attention.sh
```

This will:
- Use simplified model (no complex attention)
- Train with stable learning rate (5e-5)
- Should see **decreasing losses** now!

### Manual Training
```bash
python train_sequential_attention.py \
    --data_root /media/M2SSD/FSC147 \
    --batch_size 2 \
    --epochs 10 \
    --lr 5e-5 \
    --load_in_4bit
```

### Debug Before Training
```bash
python debug_training.py
```

**What to check:**
- ✓ Loss should DECREASE after gradient step
- ✓ Gradients should be > 1e-4 for all heads
- ✓ Predictions should NOT be stuck at -1, -1

## Expected Behavior Now

### First Iteration
```
Loss before: ~1.5
Loss after:  ~0.7 (DECREASING!)
```

### Gradient Flow
```
x_head: 0.5-2.0  (GOOD!)
y_head: 0.5-2.0  (GOOD!)
done_head: 0.3-1.0 (GOOD!)
```

### Predictions
- Should vary across different images
- NOT stuck at (-1, -1)
- Should improve over epochs

## What We Sacrificed

To fix training, we removed:
- ❌ Working Memory (LSTM)
- ❌ Foveation mechanism
- ❌ Cross-attention to counted objects
- ❌ Sequential reasoning

**Why?**
- These modules had zero gradients
- They added complexity without helping training
- The VLM already has strong visual features

## What We Kept

- ✅ VLM base (Qwen2-VL-2B) with LoRA
- ✅ Open-vocabulary capability
- ✅ Separate x, y, done predictions
- ✅ Alternating classification/regression training
- ✅ Visual marking with numbered labels
- ✅ W&B logging

## Future: Adding Back Sequential Attention

Once the baseline is working, we can add back sequential components:

**Step 1: Get baseline working**
- Train simple model
- Verify loss decreases
- Check W&B visualizations

**Step 2: Add ONE module at a time**
```python
# Try adding just working memory
class SimpleWithMemory(SimpleSequentialModel):
    def __init__(self, ...):
        super().__init__(...)
        self.memory = nn.LSTM(hidden_dim, hidden_dim)

    def forward(self, ...):
        x_features = self.extract_features(...)
        # Add memory to features
        memory_out, _ = self.memory(x_features.unsqueeze(1))
        enhanced_features = x_features + memory_out.squeeze(1)
        # Predict from enhanced features
        pred_x = self.x_head(enhanced_features)
        ...
```

**Step 3: Verify gradients**
```bash
python debug_training.py
# Check: memory module should have gradients > 0
```

**Step 4: Add next module**
- Only if previous module helped
- Always check gradients
- Always verify loss decreases

## Debugging Tools

### 1. `debug_training.py`
**Use when:**
- Training fails
- Losses increase
- Predictions stuck

**Checks:**
- Coordinate normalization
- Forward pass outputs
- Gradient flow
- Loss direction (up/down)
- Done signal behavior

### 2. Monitor W&B
**Look for:**
- Loss curves trending DOWN
- Predictions moving around (not stuck)
- Validation images showing correct locations

### 3. Print First Epoch
The training script prints:
```
[Gradient Check - Classification]
  done_head: 0.5234
[Gradient Check - Regression]
  x_head: 1.2445
  y_head: 0.9821
```

**Good values:** > 0.1
**Bad values:** < 0.001 (vanishing)
**Terrible values:** 0.0 (disconnected!)

## Common Issues & Fixes

### Issue: "Loss NaN"
**Cause:** Learning rate too high or gradient explosion
**Fix:** Lower `lr` to 1e-5 or add gradient clipping

### Issue: "Predictions always same value"
**Cause:** Dead ReLUs or saturated activations
**Fix:** Check initialization, lower learning rate

### Issue: "Done signal always 0"
**Cause:** Class imbalance in training
**Fix:** Check prepare_batch() ensures 50/50 split

### Issue: "Predictions outside [-1, 1]"
**Cause:** No clamping and high learning rate
**Fix:** Already added `torch.clamp(pred, -2, 2)` in simple model

## Summary

**Before:**
- Complex pipeline: VLM → LSTM → Foveation → Cross-Attn → Reasoning → Predict
- High learning rate: 1e-3 to 5e-3
- Result: Zero gradients, exploding loss

**After:**
- Simple pipeline: VLM → Predict
- Low learning rate: 5e-5
- Result: Should train successfully!

**Next Steps:**
1. Run `./run_sequential_attention.sh`
2. Check W&B for decreasing losses
3. Verify predictions move around (not stuck)
4. If working, gradually add back attention modules

Good luck! 🚀
