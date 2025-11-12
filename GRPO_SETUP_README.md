# GRPO Fine-Tuning Setup for VLM Counting

## Status

✅ **Reward functions implemented** - smoothed GradCAM + count accuracy
✅ **Dataset preparation ready** - FSC147 loader with centroids
✅ **GradCAM extraction implemented** - hooks for attention maps
⚠️ **TRL version incompatibility** - Current TRL 0.9.6 doesn't include GRPO

## Issue

The current TRL version (0.9.6) doesn't have `GRPOConfig` or `GRPOTrainer` yet. GRPO was added in later versions (0.10.0+).

## Solutions

### Option 1: Upgrade TRL (Recommended for GRPO)

```bash
pip install --upgrade trl
# This will get you the latest version with GRPO support
```

After upgrading, the training script should work as-is:
```bash
./run_grpo_training.sh
```

### Option 2: Use Current TRL with PPO

TRL 0.9.6 has PPO (Proximal Policy Optimization) instead. The reward functions we built will work with PPO too, just need to change the trainer:

```python
from trl import PPOConfig, PPOTrainer

# Configure PPO training
ppo_config = PPOConfig(
    learning_rate=1e-6,
    batch_size=2,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
)

# Use PPOTrainer instead of GRPOTrainer
trainer = PPOTrainer(
    model=model,
    config=ppo_config,
    reward_fn=reward_fn,  # Same reward function!
    ...
)
```

### Option 3: Manual RL Training Loop

Implement a custom training loop using the reward functions directly without TRL.

## What's Ready to Use

### 1. Reward Functions (`src/grpo_train_counting.py`)

All these functions are ready and don't depend on TRL:

- `create_ideal_gradcam_smoothed()` - Creates smoothed target attention maps
- `extract_gradcam_from_vlm()` - Extracts model attention via hooks
- `compute_gradcam_similarity_reward()` - SSIM/MSE/correlation metrics
- `compute_count_accuracy_reward()` - Inverse distance reward
- `compute_combined_reward()` - Weighted combination

### 2. Dataset Loader

```python
from grpo_train_counting import load_fsc147_dataset

train_data = load_fsc147_dataset(
    data_root="/media/M2SSD/FSC147",
    split="train",
    max_samples=100
)

# Each item has:
# - image: PIL Image
# - centroids: List[(x, y)]
# - count: int
# - category: str
# - prompt: str
```

### 3. Reward Configuration

```python
from grpo_train_counting import CountingRewardConfig, CountingRewardFunction

config = CountingRewardConfig(
    gradcam_weight=0.6,        # 60% attention similarity
    count_weight=0.4,          # 40% count accuracy
    gaussian_sigma=20.0,       # Blob size in ideal map
    smoothing_sigma=5.0,       # Smoothing for nice gradients
    gradcam_metric='ssim'      # Structural similarity
)

reward_fn = CountingRewardFunction(model, processor, config)
```

## Test Reward Functions

You can test the reward functions without TRL:

```python
import sys
sys.path.append('src')
from grpo_train_counting import *
from PIL import Image

# Load test image
image = Image.open("/media/M2SSD/FSC147/images_384_VarV2/194.jpg")
gt_centroids = [(100, 100), (200, 150), (300, 200)]  # Example

# Create ideal attention map
ideal_map = create_ideal_gradcam_smoothed(
    centroids=gt_centroids,
    image_size=image.size,
    sigma=20.0,
    smoothing_sigma=5.0
)

# Visualize
import matplotlib.pyplot as plt
plt.imshow(ideal_map, cmap='hot')
plt.colorbar()
plt.title('Ideal Smoothed GradCAM')
plt.show()
```

## Notebook

See `notebooks/04_GRPO_Fine_Tuning_Setup.ipynb` for:
- Full reward function tests
- Visualization of ideal vs predicted attention
- Comparison of different scenarios (perfect, shifted, wrong count, random)

## Files Created

1. `src/grpo_train_counting.py` - Main training script (needs TRL upgrade)
2. `notebooks/04_GRPO_Fine_Tuning_Setup.ipynb` - Testing & visualization
3. `run_grpo_training.sh` - Easy launch script
4. This README

## Next Steps

1. **Upgrade TRL**: `pip install --upgrade trl` (get version 0.10.0+)
2. **Test reward functions**: Run notebook cells to verify everything works
3. **Run pilot training**: `./run_grpo_training.sh` with 100 samples
4. **Monitor rewards**: Check that both GradCAM and count rewards are reasonable
5. **Scale up**: Increase to full training set once pilot works

## Alternative: Custom RL Loop

If you want full control, you can use the reward functions in a custom training loop:

```python
# Pseudo-code for custom RL loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Generate responses
        outputs = model.generate(...)

        # Extract GradCAM
        gradcam = extract_gradcam_from_vlm(model, processor, image, prompt)

        # Create ideal GradCAM
        ideal = create_ideal_gradcam_smoothed(centroids, image_size)

        # Compute reward
        reward_dict = compute_combined_reward(
            gradcam, ideal, pred_count, gt_count
        )

        # Use reward for policy gradient update
        loss = compute_policy_gradient_loss(outputs, reward_dict['combined_reward'])
        loss.backward()
        optimizer.step()
```

## Contact

For issues or questions:
- Check TRL documentation: https://huggingface.co/docs/trl
- TRL GitHub: https://github.com/huggingface/trl
- GRPO paper: https://arxiv.org/abs/2402.03300
