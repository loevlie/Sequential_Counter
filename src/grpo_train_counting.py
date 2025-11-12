#!/usr/bin/env python3
"""
GRPO Fine-tuning Script for VLM Object Counting

This script fine-tunes a VLM (Qwen3-VL) on FSC147 counting task using GRPO
(Group Relative Policy Optimization) from HuggingFace TRL library.

Reward function uses:
1. Smoothed GradCAM attention map similarity (ideal vs predicted)
2. Count accuracy (predicted vs ground truth)
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from PIL import Image
import torch
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity
from tqdm import tqdm

# Try importing TRL components
try:
    from trl import GRPOConfig, GRPOTrainer
    TRL_AVAILABLE = True
except ImportError:
    print("WARNING: TRL library not available. Install with: pip install trl")
    TRL_AVAILABLE = False

# Try importing transformers components
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("WARNING: Transformers library not available. Install with: pip install transformers")
    TRANSFORMERS_AVAILABLE = False


# ============================================================================
# Reward Function Components
# ============================================================================

def create_ideal_gradcam_smoothed(
    centroids: List[Tuple[int, int]],
    image_size: Tuple[int, int],
    sigma: float = 20.0,
    normalize: bool = True,
    smoothing_sigma: float = 5.0
) -> np.ndarray:
    """
    Create smoothed ideal GradCAM attention map with Gaussian blobs at each centroid.

    Args:
        centroids: List of (x, y) coordinates for object centers
        image_size: (width, height) of the image
        sigma: Standard deviation of Gaussian blobs (controls spread around centroids)
        normalize: Whether to normalize to [0, 1] range
        smoothing_sigma: Additional Gaussian smoothing applied to final map

    Returns:
        ideal_map: 2D numpy array of shape (height, width) with attention values
    """
    width, height = image_size
    ideal_map = np.zeros((height, width), dtype=np.float32)

    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:height, 0:width]

    # Add Gaussian blob for each centroid
    for cx, cy in centroids:
        # Calculate distance from centroid
        distances_sq = (x_grid - cx)**2 + (y_grid - cy)**2

        # Create Gaussian blob
        gaussian_blob = np.exp(-distances_sq / (2 * sigma**2))

        # Add to ideal map (using max to handle overlapping objects)
        ideal_map = np.maximum(ideal_map, gaussian_blob)

    # Apply Gaussian smoothing for smoother appearance
    if smoothing_sigma > 0:
        ideal_map = gaussian_filter(ideal_map, sigma=smoothing_sigma)

    # Normalize to [0, 1] if requested
    if normalize and ideal_map.max() > 0:
        ideal_map = ideal_map / ideal_map.max()

    return ideal_map


def compute_gradcam_similarity_reward(
    predicted_gradcam: np.ndarray,
    ideal_gradcam: np.ndarray,
    metric: str = 'ssim'
) -> float:
    """
    Compute reward based on similarity between predicted and ideal GradCAM maps.

    Args:
        predicted_gradcam: Predicted attention map from model
        ideal_gradcam: Ideal smoothed attention map from ground truth
        metric: Similarity metric ('ssim', 'mse', 'correlation')

    Returns:
        reward: Similarity score in [0, 1] range (higher is better)
    """
    # Ensure same shape
    if predicted_gradcam.shape != ideal_gradcam.shape:
        from scipy.ndimage import zoom
        zoom_factors = (
            ideal_gradcam.shape[0] / predicted_gradcam.shape[0],
            ideal_gradcam.shape[1] / predicted_gradcam.shape[1]
        )
        predicted_gradcam = zoom(predicted_gradcam, zoom_factors, order=1)

    if metric == 'ssim':
        # Structural Similarity Index
        similarity = structural_similarity(
            ideal_gradcam,
            predicted_gradcam,
            data_range=1.0
        )
        # SSIM is in [-1, 1], convert to [0, 1]
        reward = (similarity + 1) / 2

    elif metric == 'mse':
        # Mean Squared Error (lower is better, so invert)
        mse = np.mean((predicted_gradcam - ideal_gradcam) ** 2)
        # Convert to reward (max possible MSE is 1.0 for normalized maps)
        reward = 1.0 / (1.0 + mse)

    elif metric == 'correlation':
        # Pearson correlation coefficient
        pred_flat = predicted_gradcam.flatten()
        ideal_flat = ideal_gradcam.flatten()
        correlation = np.corrcoef(pred_flat, ideal_flat)[0, 1]
        # Handle NaN (can occur if all values are identical)
        if np.isnan(correlation):
            correlation = 0.0
        # Correlation is in [-1, 1], convert to [0, 1]
        reward = (correlation + 1) / 2

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return float(reward)


def compute_count_accuracy_reward(
    predicted_count: int,
    ground_truth_count: int,
    offset: float = 10.0
) -> float:
    """
    Compute reward based on count accuracy.

    Args:
        predicted_count: Predicted object count
        ground_truth_count: Ground truth object count
        offset: Offset to avoid division by zero (higher = more forgiving)

    Returns:
        reward: Count accuracy reward (higher is better)
    """
    error = abs(predicted_count - ground_truth_count)
    reward = 1.0 / (error + offset)
    return float(reward)


def compute_combined_reward(
    predicted_gradcam: np.ndarray,
    ideal_gradcam: np.ndarray,
    predicted_count: int,
    ground_truth_count: int,
    gradcam_weight: float = 0.6,
    count_weight: float = 0.4,
    gradcam_metric: str = 'ssim',
    count_offset: float = 10.0
) -> Dict[str, float]:
    """
    Compute combined reward from GradCAM similarity and count accuracy.

    Args:
        predicted_gradcam: Predicted attention map
        ideal_gradcam: Ideal smoothed attention map from ground truth
        predicted_count: Predicted object count
        ground_truth_count: Ground truth object count
        gradcam_weight: Weight for GradCAM similarity component
        count_weight: Weight for count accuracy component
        gradcam_metric: Metric for GradCAM similarity
        count_offset: Offset for count accuracy reward

    Returns:
        reward_dict: Dictionary with individual and combined rewards
    """
    # Compute individual rewards
    gradcam_reward = compute_gradcam_similarity_reward(
        predicted_gradcam,
        ideal_gradcam,
        metric=gradcam_metric
    )

    count_reward = compute_count_accuracy_reward(
        predicted_count,
        ground_truth_count,
        offset=count_offset
    )

    # Combine with weights
    combined_reward = gradcam_weight * gradcam_reward + count_weight * count_reward

    return {
        'gradcam_reward': gradcam_reward,
        'count_reward': count_reward,
        'combined_reward': combined_reward,
        'gradcam_weight': gradcam_weight,
        'count_weight': count_weight
    }


# ============================================================================
# GradCAM Extraction
# ============================================================================

def extract_gradcam_from_vlm(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    target_layer_name: str = None
) -> np.ndarray:
    """
    Extract GradCAM attention map from VLM model.

    Args:
        model: VLM model (e.g., Qwen3-VL)
        processor: Model processor
        image: Input PIL image
        prompt: Text prompt for counting
        target_layer_name: Layer to extract gradients from (if None, use last vision layer)

    Returns:
        gradcam_map: 2D numpy array of shape (height, width) with attention values
    """
    model.eval()
    device = next(model.parameters()).device

    # Prepare inputs
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt"
    ).to(device)

    # Register hooks to capture activations and gradients
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    # Find target layer (default: last vision encoder layer)
    if target_layer_name is None:
        # For Qwen2-VL, typically use the last vision encoder layer
        if hasattr(model, 'visual'):
            target = model.visual.transformer.resblocks[-1]
        elif hasattr(model, 'vision_model'):
            target = model.vision_model.encoder.layers[-1]
        else:
            raise ValueError("Could not find vision encoder in model")
    else:
        target = dict(model.named_modules())[target_layer_name]

    # Register hooks
    forward_handle = target.register_forward_hook(forward_hook)
    backward_handle = target.register_full_backward_hook(backward_hook)

    try:
        # Forward pass
        with torch.enable_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            # Compute gradient by backpropagating from output
            # Use mean of logits as target
            target_value = logits.mean()
            target_value.backward()

        # Compute GradCAM
        if len(activations) == 0 or len(gradients) == 0:
            raise ValueError("Failed to capture activations or gradients")

        # Get activation and gradient
        activation = activations[0]
        gradient = gradients[0]

        # Global average pooling of gradients
        weights = torch.mean(gradient, dim=tuple(range(len(gradient.shape) - 1)), keepdim=True)

        # Weighted combination of activation maps
        gradcam = torch.sum(weights * activation, dim=-1)

        # Apply ReLU (only positive influence)
        gradcam = torch.relu(gradcam)

        # Convert to numpy and resize to image size
        gradcam_np = gradcam.cpu().numpy().squeeze()

        # Resize to match image dimensions
        from scipy.ndimage import zoom
        h, w = image.size[1], image.size[0]

        if gradcam_np.ndim > 2:
            # If there are extra dimensions, take mean
            while gradcam_np.ndim > 2:
                gradcam_np = gradcam_np.mean(axis=0)

        zoom_factors = (h / gradcam_np.shape[0], w / gradcam_np.shape[1])
        gradcam_resized = zoom(gradcam_np, zoom_factors, order=1)

        # Apply Gaussian smoothing for smoother appearance
        gradcam_resized = gaussian_filter(gradcam_resized, sigma=5.0)

        # Normalize
        if gradcam_resized.max() > 0:
            gradcam_resized = gradcam_resized / gradcam_resized.max()

        return gradcam_resized

    finally:
        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()


# ============================================================================
# GRPO Configuration and Reward Function
# ============================================================================

@dataclass
class CountingRewardConfig:
    """Configuration for counting task reward function."""
    gradcam_weight: float = 0.6
    count_weight: float = 0.4
    gradcam_metric: str = 'ssim'  # 'ssim', 'mse', or 'correlation'
    count_offset: float = 10.0
    gaussian_sigma: float = 20.0
    smoothing_sigma: float = 5.0


class CountingRewardFunction:
    """Reward function for GRPO fine-tuning on object counting task."""

    __name__ = "CountingRewardFunction"  # Required by GRPOTrainer

    def __init__(self, model, processor, config: CountingRewardConfig = None):
        self.model = model
        self.processor = processor
        self.config = config or CountingRewardConfig()

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        completion_ids: Any,
        **kwargs  # Handles image, count, centroids, image_size, category from inputs dict
    ) -> List[float]:
        """
        Compute rewards for generated completions.

        GRPO unpacks the inputs dict and passes all keys as kwargs.

        Args:
            inputs: Dictionary containing:
                - 'image': PIL Images
                - 'count': Ground truth counts
                - 'centroids': Object centroids
                - 'image_size': Image dimensions
            prompts: List of prompts (chat messages)
            completions: List of generated text completions
            completion_ids_list: Token IDs (not used)

        Returns:
            rewards: List of reward values (one per sample)
        """
        print(completions)
        rewards = []

        # Extract ground truth data from kwargs
        images = kwargs.get('image', [])
        gt_counts = kwargs.get('count', [])
        centroids_list = kwargs.get('centroids', [])
        image_sizes = kwargs.get('image_size', [])

        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            try:
                # Extract completion text from chat format
                # completion is like [{'role': 'assistant', 'content': '5'}]
                if isinstance(completion, list) and len(completion) > 0:
                    completion_text = completion[0].get('content', '')
                else:
                    completion_text = str(completion)

                # Extract predicted count from generated text
                predicted_count = self._extract_count_from_text(completion_text)

                # Get ground truth for this sample
                image = images[i] if isinstance(images, list) else images
                gt_count = gt_counts[i] if isinstance(gt_counts, list) else gt_counts
                centroids = centroids_list[i] if isinstance(centroids_list, list) else centroids_list
                image_size = image_sizes[i] if isinstance(image_sizes, list) else image_sizes

                # Extract prompt text from chat format
                prompt_text = prompt[0]['content'] if isinstance(prompt, list) else prompt

                # Extract GradCAM from model
                predicted_gradcam = extract_gradcam_from_vlm(
                    self.model,
                    self.processor,
                    image,
                    prompt_text
                )

                # Create ideal smoothed GradCAM
                ideal_gradcam = create_ideal_gradcam_smoothed(
                    centroids,
                    image_size,
                    sigma=self.config.gaussian_sigma,
                    smoothing_sigma=self.config.smoothing_sigma
                )

                # Compute combined reward
                reward_dict = compute_combined_reward(
                    predicted_gradcam,
                    ideal_gradcam,
                    predicted_count,
                    gt_count,
                    gradcam_weight=self.config.gradcam_weight,
                    count_weight=self.config.count_weight,
                    gradcam_metric=self.config.gradcam_metric,
                    count_offset=self.config.count_offset
                )

                rewards.append(reward_dict['combined_reward'])

            except Exception as e:
                print(f"Error computing reward: {e}")
                import traceback
                traceback.print_exc()
                # Give penalty for errors
                rewards.append(0.0)

        return rewards

    def _extract_count_from_text(self, text: str) -> int:
        """Extract count from generated text."""
        import re

        # Try to extract first integer from text
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[0])
        else:
            # If no number found, return 0 (will get low reward)
            return 0


# ============================================================================
# Dataset Preparation
# ============================================================================

def load_fsc147_dataset(
    data_root: str,
    split: str = 'train',
    max_samples: int = None
) -> List[Dict[str, Any]]:
    """
    Load FSC147 dataset for GRPO training.

    Args:
        data_root: Path to FSC147 dataset
        split: Dataset split ('train', 'val', 'test')
        max_samples: Maximum number of samples to load

    Returns:
        dataset: List of dictionaries with image, category, ground truth, and prompt
    """
    import random

    # Load dataset information
    split_file = os.path.join(data_root, "Train_Test_Val_FSC_147.json")
    with open(split_file, 'r') as f:
        splits = json.load(f)

    annotation_file = os.path.join(data_root, "annotation_FSC147_384.json")
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    classes_file = os.path.join(data_root, "ImageClasses_FSC147.txt")
    image_classes = {}
    with open(classes_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                image_classes[parts[0]] = parts[1]

    # Get images for split
    split_images = splits[split]
    if max_samples:
        random.seed(42)
        split_images = random.sample(split_images, min(max_samples, len(split_images)))

    # Prepare dataset
    dataset = []

    for img_name in tqdm(split_images, desc=f"Loading {split} split"):
        if img_name not in annotations:
            continue

        # Load image
        img_path = os.path.join(data_root, "images_384_VarV2", img_name)
        if not os.path.exists(img_path):
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            continue

        # Get annotations
        ann = annotations[img_name]
        centroids = [(int(p[0]), int(p[1])) for p in ann['points']]
        count = len(centroids)
        category = image_classes.get(img_name, 'objects')

        # Create prompt in chat message format (required by GRPO)
        prompt = [{
            "role": "user",
            "content": f"Count the number of {category} in this image. Provide only the number."
        }]

        dataset.append({
            'image': image,
            'category': category,
            'centroids': centroids,
            'count': count,
            'image_size': image.size,
            'prompt': prompt,
            'image_name': img_name
        })

    return dataset


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GRPO fine-tuning for VLM object counting"
    )

    # Data arguments
    parser.add_argument(
        "--data_root", type=str, default="/media/M2SSD/FSC147",
        help="Path to FSC147 dataset"
    )
    parser.add_argument(
        "--train_samples", type=int, default=100,
        help="Number of training samples (None = use all)"
    )
    parser.add_argument(
        "--val_samples", type=int, default=20,
        help="Number of validation samples"
    )

    # Model arguments
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./grpo_counting_checkpoints",
        help="Output directory for checkpoints"
    )

    # GRPO training arguments
    parser.add_argument(
        "--learning_rate", type=float, default=1e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Training batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--group_size", type=int, default=4,
        help="Number of responses per prompt for GRPO"
    )

    # Reward function arguments
    parser.add_argument(
        "--gradcam_weight", type=float, default=0.6,
        help="Weight for GradCAM similarity reward"
    )
    parser.add_argument(
        "--count_weight", type=float, default=0.4,
        help="Weight for count accuracy reward"
    )
    parser.add_argument(
        "--gaussian_sigma", type=float, default=20.0,
        help="Sigma for Gaussian blobs in ideal GradCAM"
    )
    parser.add_argument(
        "--smoothing_sigma", type=float, default=5.0,
        help="Sigma for Gaussian smoothing of GradCAM maps"
    )

    args = parser.parse_args()

    # Check dependencies
    if not TRL_AVAILABLE:
        print("ERROR: TRL library not available. Install with: pip install trl")
        return

    if not TRANSFORMERS_AVAILABLE:
        print("ERROR: Transformers library not available. Install with: pip install transformers")
        return

    print("="*80)
    print("GRPO Fine-tuning for VLM Object Counting")
    print("="*80)

    # Load dataset
    print(f"\nLoading FSC147 dataset from {args.data_root}...")
    train_dataset = load_fsc147_dataset(args.data_root, 'train', args.train_samples)
    val_dataset = load_fsc147_dataset(args.data_root, 'val', args.val_samples)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Load model and processor
    print(f"\nLoading model: {args.model_name}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    # Create reward function
    reward_config = CountingRewardConfig(
        gradcam_weight=args.gradcam_weight,
        count_weight=args.count_weight,
        gaussian_sigma=args.gaussian_sigma,
        smoothing_sigma=args.smoothing_sigma
    )
    reward_fn = CountingRewardFunction(model, processor, reward_config)

    # Configure GRPO training
    print("\nConfiguring GRPO trainer...")
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=50,
        save_total_limit=3,
        num_generations=args.batch_size,  # Must be divisible by batch_size
    )

    # Initialize GRPO trainer
    print("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
    )

    # Train
    print("\n" + "="*80)
    print("Starting GRPO training...")
    print("="*80 + "\n")

    trainer.train()

    # Save final model
    final_output_dir = os.path.join(args.output_dir, "final_model")
    print(f"\nSaving final model to {final_output_dir}...")
    model.save_pretrained(final_output_dir)
    processor.save_pretrained(final_output_dir)

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)


if __name__ == "__main__":
    main()
