#!/usr/bin/env python3
"""
Evaluation script for VLM with regression head.

Sequential inference: starts with unmarked image, iteratively predicts
next location until model outputs done signal (-1, -1, -1).
"""

import argparse
import torch
from pathlib import Path
from PIL import Image
import numpy as np
import json
from tqdm import tqdm

from model_vlm_regression import VLMCountingModelRegression
from utils import VisualMarker
from dataset import OmniCountDataset


def sequential_inference(
    model: VLMCountingModelRegression,
    image: Image.Image,
    marker: VisualMarker,
    max_objects: int = 500
) -> dict:
    """
    Perform sequential inference on a single image.

    Returns:
        Dictionary with predicted_points, count, steps
    """
    model.eval()

    current_image = image.copy()
    image_np = np.array(image)
    predicted_points = []
    W, H = image.size

    with torch.no_grad():
        for step in range(max_objects):
            # Mark image with predictions so far
            if len(predicted_points) > 0:
                marked_image_np = marker.mark_image(image_np, predicted_points)
                current_image = Image.fromarray(marked_image_np)
            else:
                current_image = image.copy()

            # Get prediction
            outputs = model.forward_regression(
                images=current_image,
                num_marked=len(predicted_points)
            )

            # Extract predictions
            x_norm = outputs['x'].item()
            y_norm = outputs['y'].item()
            count_pred = outputs['count'].item()

            # Check if done (x=-1, y=-1, but count should be valid)
            if x_norm < -0.9 and y_norm < -0.9:
                print(f"Model signaled 'done' after {step} predictions")
                print(f"Model's predicted total count: {count_pred:.1f}")
                final_count = len(predicted_points)
                return {
                    'predicted_points': predicted_points,
                    'count': final_count,
                    'predicted_total': count_pred,
                    'steps': step + 1,
                    'finished_naturally': True
                }

            # Denormalize coordinates
            x_pixel = int(((x_norm + 1) / 2) * W)
            y_pixel = int(((y_norm + 1) / 2) * H)

            # Clamp to image bounds
            x_pixel = max(0, min(W - 1, x_pixel))
            y_pixel = max(0, min(H - 1, y_pixel))

            predicted_points.append((x_pixel, y_pixel))
            print(f"Step {step + 1}: Point at ({x_pixel}, {y_pixel}), predicted total: {count_pred:.1f}")

    # Hit max limit
    print(f"Reached maximum object limit ({max_objects})")
    return {
        'predicted_points': predicted_points,
        'count': len(predicted_points),
        'predicted_total': count_pred,
        'steps': max_objects,
        'finished_naturally': False
    }


def evaluate_dataset(
    model: VLMCountingModelRegression,
    dataset: OmniCountDataset,
    marker: VisualMarker,
    output_dir: Path,
    num_samples: int = 50,
    max_objects: int = 500,
    save_visualizations: bool = True
):
    """Evaluate model on dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / 'visualizations'
    if save_visualizations:
        vis_dir.mkdir(exist_ok=True)

    results = []
    num_samples = min(num_samples, len(dataset))

    for idx in tqdm(range(num_samples), desc="Evaluating"):
        # Load sample
        image, gt_points, metadata = dataset[idx]

        # Convert to PIL
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            image = Image.fromarray(image_np)

        # Run inference
        result = sequential_inference(
            model=model,
            image=image,
            marker=marker,
            max_objects=max_objects
        )

        # Calculate metrics
        gt_count = len(gt_points)
        pred_count = result['count']
        count_error = abs(pred_count - gt_count)

        sample_result = {
            'idx': idx,
            'category': metadata['category'],
            'gt_count': gt_count,
            'pred_count': pred_count,
            'count_error': count_error,
            'finished_naturally': result['finished_naturally']
        }
        results.append(sample_result)

        # Save visualization
        if save_visualizations:
            image_np = np.array(image)
            final_marked = marker.mark_image(image_np, result['predicted_points'])
            final_image = Image.fromarray(final_marked)

            vis_path = vis_dir / f"sample_{idx:04d}_gt{gt_count}_pred{pred_count}.jpg"
            final_image.save(vis_path)

        # Progress update
        if (idx + 1) % 10 == 0:
            avg_mae = np.mean([r['count_error'] for r in results])
            print(f"\nProgress: {idx + 1}/{num_samples} | Avg MAE: {avg_mae:.2f}")

    # Calculate aggregate metrics
    count_errors = [r['count_error'] for r in results]

    metrics = {
        'num_samples': num_samples,
        'count_mae': float(np.mean(count_errors)),
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
    parser = argparse.ArgumentParser(description="Evaluate VLM Regression Model")

    # Model args
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-VL-4B-Thinking')
    parser.add_argument('--load_in_4bit', action='store_true')

    # Dataset args
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--categories', type=str, nargs='+', default=['Supermarket'])
    parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid', 'test'])

    # Evaluation args
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--max_objects', type=int, default=500)
    parser.add_argument('--marking_alpha', type=float, default=0.3)
    parser.add_argument('--output_dir', type=str, default='eval_regression_results')
    parser.add_argument('--save_visualizations', action='store_true')

    args = parser.parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print("Loading VLM model...")
    model = VLMCountingModelRegression(
        model_name=args.model_name,
        use_lora=True,
        load_in_4bit=args.load_in_4bit,
        device=device
    )

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
        image_size=None
    )
    print(f"Loaded {len(dataset)} images")

    # Create marker
    marker = VisualMarker(strategy='heatmap', alpha=args.marking_alpha)

    # Run evaluation
    print("\nStarting evaluation...")
    output_dir = Path(args.output_dir)
    metrics = evaluate_dataset(
        model=model,
        dataset=dataset,
        marker=marker,
        output_dir=output_dir,
        num_samples=args.num_samples,
        max_objects=args.max_objects,
        save_visualizations=args.save_visualizations
    )

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
