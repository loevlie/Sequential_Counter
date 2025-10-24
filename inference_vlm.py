#!/usr/bin/env python3
"""
Simple inference script for VLM Sequential Counting

Usage:
    python inference_vlm.py --checkpoint_dir test_vlm_run/best_checkpoint_lora \
                            --image_path /path/to/image.jpg \
                            --output_dir inference_output
"""

import argparse
import torch
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from model_vlm import VLMCountingModel
from utils import VisualMarker


def predict_and_visualize(
    model: VLMCountingModel,
    image_path: str,
    output_dir: Path,
    max_objects: int = 500,
    temperature: float = 0.3,
    save_steps: bool = True
):
    """
    Run inference on a single image and create visualization.

    Args:
        model: Trained VLM model
        image_path: Path to input image
        output_dir: Directory to save outputs
        max_objects: Maximum objects to predict
        temperature: Sampling temperature
        save_steps: Save intermediate steps
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    W, H = image.size
    print(f"Image size: {W}x{H}")

    # Create marker
    marker = VisualMarker(strategy='heatmap', alpha=0.3)

    # Inference
    model.eval()
    image_np = np.array(image)
    predicted_points = []

    print("\nStarting sequential prediction...")
    print("-" * 60)

    with torch.no_grad():
        for step in range(max_objects):
            # Mark the image with predictions so far
            if len(predicted_points) > 0:
                marked_image_np = marker.mark_image(image_np, predicted_points)
                current_image = Image.fromarray(marked_image_np)
            else:
                current_image = image.copy()

            # Save step if requested
            if save_steps:
                step_path = output_dir / f"step_{step:03d}.jpg"
                current_image.save(step_path)

            # Get prediction
            output = model.forward(
                images=current_image,
                num_marked=len(predicted_points),
                max_new_tokens=128,
                temperature=temperature,
                top_p=0.9
            )

            # Check if done
            is_done = output['is_done'].item() > 0.5

            if is_done:
                print(f"\nModel signaled 'done' after predicting {len(predicted_points)} objects")
                break

            # Get predicted point
            x_norm = output['x'].item()
            y_norm = output['y'].item()

            x_pixel = int(((x_norm + 1) / 2) * W)
            y_pixel = int(((y_norm + 1) / 2) * H)

            # Clamp to image bounds
            x_pixel = max(0, min(W - 1, x_pixel))
            y_pixel = max(0, min(H - 1, y_pixel))

            predicted_points.append((x_pixel, y_pixel))
            print(f"Step {step + 1:3d}: Point #{len(predicted_points)} at ({x_pixel:4d}, {y_pixel:4d})")

    # Create final visualization
    print("\n" + "-" * 60)
    print(f"FINAL COUNT: {len(predicted_points)} objects")
    print("-" * 60)

    # Save final marked image with different visualization strategies
    for strategy in ['heatmap', 'numbers', 'dots']:
        marker_vis = VisualMarker(strategy=strategy, alpha=0.5)
        final_marked = marker_vis.mark_image(image_np.copy(), predicted_points)
        final_image = Image.fromarray(final_marked)
        final_path = output_dir / f"final_{strategy}.jpg"
        final_image.save(final_path)
        print(f"Saved {strategy} visualization: {final_path}")

    # Create matplotlib figure with comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')

    # Final marked image
    marker_final = VisualMarker(strategy='numbers', alpha=0.6)
    final_marked = marker_final.mark_image(image_np, predicted_points)
    axes[1].imshow(final_marked)
    axes[1].set_title(f"Predictions: {len(predicted_points)} objects", fontsize=14)
    axes[1].axis('off')

    plt.tight_layout()
    comparison_path = output_dir / "comparison.jpg"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison: {comparison_path}")

    # Save predictions as text
    predictions_path = output_dir / "predictions.txt"
    with open(predictions_path, 'w') as f:
        f.write(f"Total count: {len(predicted_points)}\n")
        f.write(f"Image size: {W}x{H}\n\n")
        f.write("Predicted points (x, y):\n")
        for i, (x, y) in enumerate(predicted_points, 1):
            f.write(f"{i:3d}. ({x:4d}, {y:4d})\n")
    print(f"Saved predictions: {predictions_path}")

    print(f"\nAll outputs saved to: {output_dir}")

    return predicted_points


def main():
    parser = argparse.ArgumentParser(description="VLM Sequential Counting Inference")

    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to checkpoint directory with LoRA adapters')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='inference_output',
                        help='Directory to save outputs')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-VL-4B-Thinking',
                        help='Base model name')
    parser.add_argument('--load_in_4bit', action='store_true',
                        help='Load model in 4-bit quantization')
    parser.add_argument('--max_objects', type=int, default=500,
                        help='Maximum objects to predict')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='Sampling temperature (lower = more deterministic)')
    parser.add_argument('--save_steps', action='store_true',
                        help='Save intermediate step images')

    args = parser.parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print("\nLoading VLM model...")
    model = VLMCountingModel(
        model_name=args.model_name,
        use_lora=True,
        load_in_4bit=args.load_in_4bit,
        device=device
    )

    print(f"Loading checkpoint from {args.checkpoint_dir}...")
    model.load_pretrained(args.checkpoint_dir)

    # Run inference
    output_dir = Path(args.output_dir)
    predict_and_visualize(
        model=model,
        image_path=args.image_path,
        output_dir=output_dir,
        max_objects=args.max_objects,
        temperature=args.temperature,
        save_steps=args.save_steps
    )


if __name__ == '__main__':
    main()
