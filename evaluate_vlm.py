#!/usr/bin/env python3
"""
Evaluate VLM Sequential Counting Model

This script:
1. Loads a trained VLM model (with LoRA adapters)
2. Takes an unmarked image
3. Sequentially predicts object locations
4. Marks each predicted location with a visual marker
5. Continues until model outputs "done"
6. Returns final count and saves visualizations
"""

import argparse
import torch
from pathlib import Path
from PIL import Image
import numpy as np
import json
from tqdm import tqdm

from model_vlm import VLMCountingModel
from utils import VisualMarker
from dataset import OmniCountDataset


def sequential_inference(
    model: VLMCountingModel,
    image: Image.Image,
    marker: VisualMarker,
    max_objects: int = 500,
    temperature: float = 0.3,
    top_p: float = 0.9,
    save_steps: bool = False,
    output_dir: Path = None
) -> dict:
    """
    Perform sequential inference on a single image.

    Args:
        model: Trained VLM model
        image: Input PIL Image (unmarked)
        marker: VisualMarker for visualization
        max_objects: Maximum objects to predict (safety limit)
        temperature: Sampling temperature (lower = more deterministic)
        top_p: Nucleus sampling parameter
        save_steps: Whether to save intermediate visualizations
        output_dir: Directory to save step images

    Returns:
        Dictionary with:
            - predicted_points: List of (x, y) pixel coordinates
            - count: Final count
            - steps: Number of prediction steps
            - finished_naturally: Whether model said "done" vs hit max_objects
    """
    model.eval()

    # Convert to numpy for marking
    current_image = image.copy()
    image_np = np.array(image)

    predicted_points = []
    W, H = image.size

    with torch.no_grad():
        for step in range(max_objects):
            # Mark the image with predictions so far
            if len(predicted_points) > 0:
                marked_image_np = marker.mark_image(image_np, predicted_points)
                current_image = Image.fromarray(marked_image_np)
            else:
                current_image = image.copy()

            # Save intermediate step if requested
            if save_steps and output_dir:
                step_path = output_dir / f"step_{step:03d}_marked_{len(predicted_points)}.jpg"
                current_image.save(step_path)

            # Get prediction from model
            output = model.forward(
                images=current_image,
                num_marked=len(predicted_points),
                max_new_tokens=128,
                temperature=temperature,
                top_p=top_p
            )

            # Check if done
            is_done = output['is_done'].item() > 0.5

            if is_done:
                print(f"Model signaled 'done' after {step} predictions")
                return {
                    'predicted_points': predicted_points,
                    'count': len(predicted_points),
                    'steps': step + 1,
                    'finished_naturally': True,
                    'final_image': current_image
                }

            # Get predicted point (denormalize from [-1, 1] to pixel coords)
            x_norm = output['x'].item()
            y_norm = output['y'].item()

            x_pixel = int(((x_norm + 1) / 2) * W)
            y_pixel = int(((y_norm + 1) / 2) * H)

            # Clamp to image bounds
            x_pixel = max(0, min(W - 1, x_pixel))
            y_pixel = max(0, min(H - 1, y_pixel))

            predicted_points.append((x_pixel, y_pixel))
            print(f"Step {step + 1}: Predicted point at ({x_pixel}, {y_pixel})")

    # Hit max_objects limit
    print(f"Reached maximum object limit ({max_objects})")
    return {
        'predicted_points': predicted_points,
        'count': len(predicted_points),
        'steps': max_objects,
        'finished_naturally': False,
        'final_image': current_image
    }


def evaluate_dataset(
    model: VLMCountingModel,
    dataset: OmniCountDataset,
    marker: VisualMarker,
    output_dir: Path,
    num_samples: int = 50,
    max_objects: int = 500,
    temperature: float = 0.3,
    save_visualizations: bool = True
):
    """
    Evaluate model on a dataset.

    Args:
        model: Trained VLM model
        dataset: Dataset to evaluate
        marker: VisualMarker for visualization
        output_dir: Directory to save results
        num_samples: Number of samples to evaluate
        max_objects: Maximum objects per image
        temperature: Sampling temperature
        save_visualizations: Whether to save marked images
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / 'visualizations'
    if save_visualizations:
        vis_dir.mkdir(exist_ok=True)

    results = []

    # Evaluate samples
    num_samples = min(num_samples, len(dataset))

    for idx in tqdm(range(num_samples), desc="Evaluating"):
        # Load sample
        image, gt_points, metadata = dataset[idx]

        # Convert tensor to PIL if needed
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            image = Image.fromarray(image_np)

        # Run sequential inference
        result = sequential_inference(
            model=model,
            image=image,
            marker=marker,
            max_objects=max_objects,
            temperature=temperature,
            save_steps=False,
            output_dir=None
        )

        # Calculate metrics
        gt_count = len(gt_points)
        pred_count = result['count']
        count_error = abs(pred_count - gt_count)
        count_mae = count_error

        # Calculate location accuracy (if counts match)
        location_accuracy = None
        if pred_count > 0 and gt_count > 0:
            # For each GT point, find nearest prediction
            # This is a simple metric - you might want something more sophisticated
            pass  # TODO: implement if needed

        sample_result = {
            'idx': idx,
            'category': metadata['category'],
            'gt_count': gt_count,
            'pred_count': pred_count,
            'count_error': count_error,
            'count_mae': count_mae,
            'finished_naturally': result['finished_naturally']
        }
        results.append(sample_result)

        # Save visualization
        if save_visualizations:
            # Create final marked image
            image_np = np.array(image)
            final_marked = marker.mark_image(image_np, result['predicted_points'])
            final_image = Image.fromarray(final_marked)

            vis_path = vis_dir / f"sample_{idx:04d}_gt{gt_count}_pred{pred_count}.jpg"
            final_image.save(vis_path)

        # Print progress
        if (idx + 1) % 10 == 0:
            avg_mae = np.mean([r['count_mae'] for r in results])
            print(f"\nProgress: {idx + 1}/{num_samples} | Avg MAE: {avg_mae:.2f}")

    # Calculate aggregate metrics
    count_maes = [r['count_mae'] for r in results]
    count_errors = [r['count_error'] for r in results]

    metrics = {
        'num_samples': num_samples,
        'count_mae': float(np.mean(count_maes)),
        'count_rmse': float(np.sqrt(np.mean([e**2 for e in count_errors]))),
        'median_error': float(np.median(count_errors)),
        'finished_naturally_pct': float(np.mean([r['finished_naturally'] for r in results]) * 100),
        'per_sample_results': results
    }

    # Save results
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Samples evaluated: {num_samples}")
    print(f"Count MAE: {metrics['count_mae']:.2f}")
    print(f"Count RMSE: {metrics['count_rmse']:.2f}")
    print(f"Median Error: {metrics['median_error']:.2f}")
    print(f"Finished naturally: {metrics['finished_naturally_pct']:.1f}%")
    print(f"\nResults saved to: {results_path}")
    if save_visualizations:
        print(f"Visualizations saved to: {vis_dir}")
    print("="*80)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM Sequential Counting Model")

    # Model arguments
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to checkpoint directory with LoRA adapters')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-VL-4B-Thinking',
                        help='Base model name')
    parser.add_argument('--load_in_4bit', action='store_true',
                        help='Load model in 4-bit quantization')

    # Dataset arguments
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to OmniCount-191 dataset')
    parser.add_argument('--categories', type=str, nargs='+', default=['Supermarket'],
                        help='Categories to evaluate')
    parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid', 'test'],
                        help='Dataset split to evaluate')

    # Evaluation arguments
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples to evaluate')
    parser.add_argument('--max_objects', type=int, default=500,
                        help='Maximum objects to predict per image')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='Sampling temperature (lower = more deterministic)')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Nucleus sampling parameter')
    parser.add_argument('--marking_alpha', type=float, default=0.3,
                        help='Visual marker overlay alpha')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--save_visualizations', action='store_true',
                        help='Save visualization images')

    args = parser.parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading VLM model...")
    model = VLMCountingModel(
        model_name=args.model_name,
        use_lora=True,
        load_in_4bit=args.load_in_4bit,
        device=device
    )

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_dir}...")
    model.load_pretrained(args.checkpoint_dir)
    model.eval()

    # Load dataset
    print("Loading dataset...")
    dataset = OmniCountDataset(
        dataset_root=args.data_root,
        categories=args.categories,
        split=args.split,
        spatial_order='reading_order',
        image_size=None  # Use original image sizes for evaluation
    )
    print(f"Loaded {len(dataset)} images")

    # Create marker
    marker = VisualMarker(strategy='heatmap', alpha=args.marking_alpha)

    # Run evaluation
    print("\nStarting evaluation...")
    metrics = evaluate_dataset(
        model=model,
        dataset=dataset,
        marker=marker,
        output_dir=output_dir,
        num_samples=args.num_samples,
        max_objects=args.max_objects,
        temperature=args.temperature,
        save_visualizations=args.save_visualizations
    )

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
