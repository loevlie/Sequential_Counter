#!/usr/bin/env python3
"""
Evaluation script specifically for the new dense counting strategies:
- dense_with_validation
- dense_explicit_overlap

Compares against existing strategies: hybrid, dense_grid, adaptive_hierarchical
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import time
import traceback

# Import the enhanced VLM counter
from rl_vlm_enhanced import EnhancedVLMCounter


def load_fsc147_data(data_root: str, split: str, max_samples: int = None):
    """Load FSC147 dataset information."""

    # Load split information
    split_file = os.path.join(data_root, "Train_Test_Val_FSC_147.json")
    with open(split_file, 'r') as f:
        splits = json.load(f)

    # Load annotations
    annotation_file = os.path.join(data_root, "annotation_FSC147_384.json")
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Load image classes
    classes_file = os.path.join(data_root, "ImageClasses_FSC147.txt")
    image_classes = {}
    with open(classes_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                image_classes[parts[0]] = parts[1]

    # Get images for split
    split_images = splits[split]

    # Sample if requested
    if max_samples:
        import random
        random.seed(42)
        split_images = random.sample(split_images, min(max_samples, len(split_images)))

    return split_images, annotations, image_classes


def evaluate_strategies(data_root: str, split: str, max_samples: int = None,
                       strategies: list = None):
    """Evaluate counting strategies on FSC147 dataset."""

    if strategies is None:
        strategies = [
            "hybrid",
            "dense_grid",
            "adaptive_hierarchical",
            "dense_with_validation",
            "dense_explicit_overlap"
        ]

    print(f"Evaluating strategies: {strategies}")

    # Load dataset
    print(f"Loading FSC147 {split} split...")
    split_images, annotations, image_classes = load_fsc147_data(
        data_root, split, max_samples
    )

    print(f"Found {len(split_images)} images to evaluate")

    # Initialize counter
    print("Initializing VLM counter...")
    counter = EnhancedVLMCounter()

    # Results storage
    results = {
        'image_id': [],
        'category': [],
        'ground_truth': [],
        'strategies': {strategy: [] for strategy in strategies}
    }

    # Process each image
    print(f"\nEvaluating {len(split_images)} images...")

    for image_name in tqdm(split_images, desc="Processing images"):
        # Check if annotation exists
        if image_name not in annotations:
            print(f"\nSkipping {image_name} - no annotation found")
            continue

        ann = annotations[image_name]

        # Load image
        image_path = os.path.join(
            data_root, "images_384_VarV2",
            image_name.replace('.jpg', '') + '.jpg'
        )
        if not os.path.exists(image_path):
            image_path = os.path.join(data_root, "images_384_VarV2", image_name)
            if not os.path.exists(image_path):
                print(f"\nSkipping {image_name} - image not found")
                continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"\nError loading {image_path}: {e}")
            continue

        # Get ground truth and category
        gt_count = len(ann['points'])
        category = image_classes.get(image_name, 'objects')

        results['image_id'].append(image_name)
        results['category'].append(category)
        results['ground_truth'].append(gt_count)

        # Test each strategy
        for strategy in strategies:
            try:
                start_time = time.time()
                result = counter.count_objects(image, category, strategy=strategy)
                elapsed = time.time() - start_time

                pred_count = result['count']
                error = pred_count - gt_count
                abs_error = abs(error)
                squared_error = error ** 2

                results['strategies'][strategy].append({
                    'count': pred_count,
                    'error': error,
                    'abs_error': abs_error,
                    'squared_error': squared_error,
                    'time': elapsed,
                    'result_dict': result
                })

                print(f"\n  {image_name} ({category}): GT={gt_count}")
                print(f"    {strategy}: {pred_count} (error={error:+d}, time={elapsed:.1f}s)")

            except Exception as e:
                print(f"\nError with {strategy} on {image_name}: {e}")
                traceback.print_exc()

                results['strategies'][strategy].append({
                    'count': 0,
                    'error': -gt_count,
                    'abs_error': gt_count,
                    'squared_error': gt_count ** 2,
                    'time': 0,
                    'result_dict': None
                })

    return results


def create_summary_table(results: dict) -> pd.DataFrame:
    """Create summary statistics table."""

    summary_data = []

    for strategy_name, strategy_results in results['strategies'].items():
        if not strategy_results:
            continue

        # Calculate metrics
        mae = np.mean([r['abs_error'] for r in strategy_results])
        rmse = np.sqrt(np.mean([r['squared_error'] for r in strategy_results]))
        mean_error = np.mean([r['error'] for r in strategy_results])
        std_error = np.std([r['error'] for r in strategy_results])
        avg_time = np.mean([r['time'] for r in strategy_results])

        # Calculate accuracy within thresholds
        within_5 = sum(1 for r in strategy_results if r['abs_error'] <= 5)
        within_10 = sum(1 for r in strategy_results if r['abs_error'] <= 10)
        within_20pct = sum(
            1 for r, gt in zip(strategy_results, results['ground_truth'])
            if r['abs_error'] <= max(1, gt * 0.2)
        )

        total = len(strategy_results)

        summary_data.append({
            'Strategy': strategy_name,
            'MAE': f"{mae:.2f}",
            'RMSE': f"{rmse:.2f}",
            'Mean Error': f"{mean_error:.2f}",
            'Std Error': f"{std_error:.2f}",
            'Within 5': f"{within_5}/{total} ({within_5/total*100:.1f}%)",
            'Within 10': f"{within_10}/{total} ({within_10/total*100:.1f}%)",
            'Within 20%': f"{within_20pct}/{total} ({within_20pct/total*100:.1f}%)",
            'Avg Time (s)': f"{avg_time:.1f}"
        })

    # Sort by MAE
    summary_data.sort(key=lambda x: float(x['MAE']))

    return pd.DataFrame(summary_data)


def save_detailed_results(results: dict, output_path: str):
    """Save detailed results to JSON."""

    output = {
        'summary': {},
        'detailed': []
    }

    # Calculate summary for each strategy
    for strategy_name, strategy_results in results['strategies'].items():
        if strategy_results:
            output['summary'][strategy_name] = {
                'mae': float(np.mean([r['abs_error'] for r in strategy_results])),
                'rmse': float(np.sqrt(np.mean([r['squared_error'] for r in strategy_results]))),
                'mean_error': float(np.mean([r['error'] for r in strategy_results])),
                'std_error': float(np.std([r['error'] for r in strategy_results])),
                'avg_time': float(np.mean([r['time'] for r in strategy_results]))
            }

    # Store detailed results per image
    for i in range(len(results['ground_truth'])):
        img_result = {
            'image_id': results['image_id'][i],
            'category': results['category'][i],
            'ground_truth': results['ground_truth'][i],
            'predictions': {}
        }

        for strategy_name, strategy_results in results['strategies'].items():
            if i < len(strategy_results):
                img_result['predictions'][strategy_name] = strategy_results[i]

        output['detailed'].append(img_result)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate new dense counting strategies on FSC147"
    )
    parser.add_argument(
        "--data_root", type=str, default="/media/M2SSD/FSC147",
        help="Path to FSC147 dataset root"
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val", "test"],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--strategies", type=str, nargs='+',
        default=["hybrid", "dense_grid", "adaptive_hierarchical",
                 "dense_with_validation", "dense_explicit_overlap"],
        help="Strategies to evaluate"
    )
    parser.add_argument(
        "--output_dir", type=str, default="evaluation_results",
        help="Output directory for results"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run evaluation
    print("="*100)
    print("NEW STRATEGIES EVALUATION")
    print("="*100)

    results = evaluate_strategies(
        args.data_root,
        args.split,
        args.max_samples,
        args.strategies
    )

    # Create summary table
    summary_df = create_summary_table(results)

    # Print results
    print("\n" + "="*100)
    print("EVALUATION RESULTS")
    print("="*100)
    print(f"Dataset: FSC147 {args.split} split")
    print(f"Images evaluated: {len(results['ground_truth'])}")
    print("\n")
    print(summary_df.to_string(index=False))
    print("="*100)

    # Save results
    csv_path = os.path.join(
        args.output_dir, f"new_strategies_summary_{args.split}_{timestamp}.csv"
    )
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")

    json_path = os.path.join(
        args.output_dir, f"new_strategies_detailed_{args.split}_{timestamp}.json"
    )
    save_detailed_results(results, json_path)
    print(f"Detailed results saved to: {json_path}")

    # Create markdown report
    md_path = os.path.join(
        args.output_dir, f"new_strategies_report_{args.split}_{timestamp}.md"
    )
    with open(md_path, 'w') as f:
        f.write(f"# New Dense Strategies Evaluation Report\n\n")
        f.write(f"**Dataset**: FSC147 {args.split} split\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Images evaluated**: {len(results['ground_truth'])}\n")
        f.write(f"**Strategies**: {', '.join(args.strategies)}\n\n")
        f.write("## Summary Results\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n## Strategy Descriptions\n\n")
        f.write("### Baseline Strategies\n\n")
        f.write("- **hybrid**: Global count + 4 quadrant counts (original best)\n")
        f.write("- **dense_grid**: 3×3 grid with mathematical overlap correction\n")
        f.write("- **adaptive_hierarchical**: Recursive subdivision based on density\n\n")
        f.write("### New Strategies\n\n")
        f.write("- **dense_with_validation**: Dense grid (3×3 to 5×5) with global + sub-global validation and adaptive fusion\n")
        f.write("- **dense_explicit_overlap**: Dense 3×3 grid where VLM explicitly examines overlap regions to detect double-counting\n\n")
        f.write("## Metrics Explanation\n\n")
        f.write("- **MAE**: Mean Absolute Error (lower is better)\n")
        f.write("- **RMSE**: Root Mean Square Error (lower is better)\n")
        f.write("- **Mean Error**: Average signed error (shows bias - negative = undercount)\n")
        f.write("- **Std Error**: Standard deviation of errors (shows consistency)\n")
        f.write("- **Within X**: Count within X objects of ground truth\n")
        f.write("- **Within 20%**: Count within 20% of ground truth\n")
        f.write("- **Avg Time**: Average processing time per image\n")

    print(f"Report saved to: {md_path}")

    print("\n" + "="*100)
    print("Evaluation complete!")
    print("="*100)


if __name__ == "__main__":
    main()
