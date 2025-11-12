# Sequential Attention Mechanism for Object Counting

## Overview

This implementation introduces **sequential attention mechanisms** inspired by human serial counting behavior. Unlike parallel estimation where humans struggle with large numbers, this approach allows the model to process objects **one-by-one** with focused attention, maintaining working memory of what has been counted.

## Motivation

**Key Insight from Cognitive Science:**
> "Humans can precisely count large numbers when allowed to process items serially, but face severe capacity constraints with rapid estimation." - Edge.org

This implementation translates that insight into a neural architecture with:
1. **Serial Processing**: Attend to objects sequentially rather than all at once
2. **Foveation**: Focus on specific regions like human foveal vision
3. **Working Memory**: Track counting progress with recurrent state
4. **Extended Thinking**: Multi-step reasoning before each prediction

## Architecture

### 1. Vision-Language Model (VLM) Base
- **Model**: Qwen3-VL-4B-Thinking
- **Purpose**: Open-vocabulary visual understanding
- **Training**: LoRA adapters for efficient fine-tuning
- **Output**: Rich visual features that understand object categories

### 2. Working Memory Module

```python
class WorkingMemoryModule(nn.Module):
    """
    LSTM-based recurrent module that:
    - Tracks how many objects have been counted
    - Maintains sequential state across counting steps
    - Estimates running count in memory
    """
```

**Key Features:**
- 2-layer LSTM with 4096 hidden dimensions
- Maintains tuple (h, c) of hidden state across predictions
- Outputs running count estimate at each step

**Why it works:**
- Mimics human working memory during counting
- Prevents "forgetting" previous progress
- Allows model to maintain context over long sequences

### 3. Spatial Foveation Module

```python
class SpatialFoveationModule(nn.Module):
    """
    Implements foveal vision - focusing on specific regions sequentially.

    Creates Gaussian attention maps centered on different regions,
    simulating human eye movements during counting.
    """
```

**Implementation:**
- Predicts (x, y) center for next foveal glimpse
- Predicts scale (window size) of attention
- Creates 2D Gaussian attention map
- Takes `num_foveal_steps` sequential glimpses per prediction

**Mathematical Formula:**
```
attention(x, y) = exp(-((x - cx)² + (y - cy)²) / (2σ²))
```

**Why it works:**
- Humans don't process entire images uniformly
- Foveal vision concentrates on small regions with high acuity
- Sequential glimpses allow detailed inspection of different areas

### 4. Object Cross-Attention Module

```python
class ObjectCrossAttention(nn.Module):
    """
    Multi-head cross-attention to previously counted objects.

    Allows model to "remember" where it has already looked
    and avoid recounting the same objects.
    """
```

**Mechanism:**
- Query: Current processing state
- Keys/Values: Memory buffer of previously counted objects
- Multi-head attention (8 heads) for diverse attention patterns

**Why it works:**
- Prevents double-counting by attending to history
- Learns spatial relationships between counted objects
- Guides attention to unexplored regions

### 5. Sequential Reasoning Module

```python
class SequentialReasoningModule(nn.Module):
    """
    Multi-step reasoning for extended "thinking time".

    Processes features through multiple refinement steps
    before making a prediction - like chain-of-thought.
    """
```

**Implementation:**
- Stack of residual blocks
- Each block: Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm
- Configurable number of reasoning steps (default: 3)

**Why it works:**
- Complex counting requires deliberation, not instant judgment
- Residual connections allow iterative refinement
- Multiple steps = more "thinking time" for hard cases

## Training Strategy

### Alternating Mode Training

The model alternates between two training modes each batch:

**Classification Mode (50% of batches):**
- Goal: Learn to detect when counting is complete
- Data: 50% examples with all objects marked (done=1), 50% with one missing (done=0)
- Loss: Binary cross-entropy on `done` signal only

**Regression Mode (50% of batches):**
- Goal: Learn to predict next object location
- Data: Random k objects marked, predict location of object k+1
- Loss: MSE on (x, y) coordinates only

**Why alternating?**
- Prevents collapse where model always predicts done=0
- Ensures balanced training of both capabilities
- Mirrors human counting: "where next?" vs "am I done?"

### Memory Management

**Per-Batch Reset:**
- Working memory LSTM state reset for each new image
- Memory buffer cleared between independent counting tasks

**Sequential Updates:**
- After each prediction, features stored in memory buffer
- Buffer limited to `max_memory_objects=50` to prevent memory overflow

### Optimizer Configuration

```python
optimizer = torch.optim.AdamW([
    {'params': model.model.parameters(), 'lr': 1e-3},          # VLM (pre-trained)
    {'params': sequential_modules.parameters(), 'lr': 5e-3}    # New modules (5x higher)
])
```

**Rationale:**
- VLM already has visual understanding → conservative updates
- Sequential attention modules start from scratch → need faster learning

## Usage

### Training

```bash
./run_sequential_attention.sh
```

Or manually:

```bash
python train_sequential_attention.py \
    --data_root /path/to/FSC147 \
    --batch_size 2 \
    --epochs 10 \
    --num_foveal_steps 4 \
    --num_reasoning_steps 3 \
    --min_objects 5 \
    --max_objects 50 \
    --load_in_4bit \
    --wandb_project sequential-counting \
    --wandb_run_name my-experiment
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_foveal_steps` | 4 | Number of sequential foveal glimpses per prediction |
| `num_reasoning_steps` | 3 | Number of sequential reasoning refinement steps |
| `max_memory_objects` | 50 | Maximum objects to store in memory buffer |
| `min_objects` | 5 | Minimum objects per image (filter dataset) |
| `max_objects` | 50 | Maximum objects per image (filter dataset) |
| `lr` | 1e-3 | Learning rate for VLM |
| `lr * 5` | 5e-3 | Learning rate for sequential modules |

### Inference

```python
from model_sequential_attention import SequentialAttentionCountingModel
from PIL import Image

# Load model
model = SequentialAttentionCountingModel(
    model_name="Qwen/Qwen3-VL-4B-Thinking",
    load_in_4bit=True
)
model.load_pretrained("sequential_attention_model/best_checkpoint")
model.eval()

# Reset memory for new counting task
model.reset_memory(batch_size=1)

# Iteratively count objects
image = Image.open("example.jpg")
num_marked = 0
category = "apples"

while True:
    outputs = model.forward_with_attention(
        images=image,
        num_marked=num_marked,
        category=category,
        return_attention_maps=True
    )

    if outputs['done'] > 0.5:
        print(f"Counting complete! Count: {outputs['count_estimate']:.0f}")
        break

    pred_x = outputs['x'].item()
    pred_y = outputs['y'].item()
    print(f"Next object at: ({pred_x:.2f}, {pred_y:.2f})")

    # Mark this object and continue
    num_marked += 1
```

## Ablation Studies

To understand the contribution of each component, you can disable modules:

### Baseline (Original VLM without Sequential Attention)
```python
model = VLMCountingModelRegression(...)  # Original model
```

### No Foveation
```python
model = SequentialAttentionCountingModel(num_foveal_steps=0)
```

### No Working Memory
Comment out the working memory update in `forward_with_attention`:
```python
# memory_state, new_hidden, count_estimate = self.working_memory(...)
memory_state = combined_features  # Direct passthrough
```

### No Sequential Reasoning
```python
model = SequentialAttentionCountingModel(num_reasoning_steps=1)
```

### No Object Cross-Attention
Comment out the cross-attention in `forward_with_attention`:
```python
# object_context = self.object_attention(...)
object_context = torch.zeros_like(foveated_features)
```

## Expected Results

### Baseline vs Sequential Attention

| Metric | Baseline VLM | Sequential Attention | Improvement |
|--------|--------------|---------------------|-------------|
| MAE (5-20 objects) | ~2.5 | ~1.8 | 28% |
| MAE (20-50 objects) | ~8.3 | ~4.7 | 43% |
| Done Accuracy | 82% | 91% | +9pp |
| Location Error (pixels) | 35 | 22 | 37% |

*Note: Results are illustrative - actual performance depends on training.*

### Advantages

1. **Better on Large Counts**: Sequential processing scales better than parallel estimation
2. **Interpretable**: Attention maps show where model is looking
3. **Fewer Mistakes**: Cross-attention reduces double-counting
4. **Open Vocabulary**: VLM base handles any object category

### Limitations

1. **Slower Inference**: Sequential steps take more time than single forward pass
2. **Memory Requirements**: LSTM state and memory buffer add overhead
3. **Training Complexity**: More modules = more hyperparameters to tune

## Visualization

The training script logs predictions to W&B with:
- Input image with marked objects
- Predicted next location (red circle) or DONE signal
- Running count estimate
- Attention maps (if enabled)

## References

1. **Edge.org**: Discussion on human counting capacity constraints
2. **FeD Paper**: Direct regression heads for sensorimotor agents
3. **Foveated Vision**: Literature on human eye movements during visual search
4. **Working Memory**: Baddeley's model of phonological loop for counting

## Future Improvements

1. **Spatial Feature Extraction**: Better integration with VLM's vision encoder
2. **Learned Foveal Scheduling**: Let model learn optimal glimpse order
3. **Hierarchical Attention**: Coarse-to-fine region processing
4. **Self-Supervised Pre-training**: Pre-train attention modules on counting games
5. **Multi-Scale Foveation**: Different window sizes for different object densities

## Citation

If you use this implementation, please cite:

```bibtex
@software{sequential_attention_counting,
  title={Sequential Attention Mechanism for Vision-Language Object Counting},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/Sequential_Counter}
}
```

## Questions?

- Check training logs in W&B
- Review attention maps to debug focus issues
- Adjust `num_foveal_steps` if model misses objects
- Increase `num_reasoning_steps` for complex scenes
