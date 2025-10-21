#!/usr/bin/env python3
"""
Analyze Hyperparameter Sweep Results

Reads all metrics.csv files from sweep outputs and finds best model.
"""

import pandas as pd
from pathlib import Path
import json
import argparse


def analyze_sweep(sweep_dir='sweep_outputs', output_file='sweep_results.csv'):
    """
    Analyze all sweep results.

    Returns DataFrame with all results sorted by best val loss.
    """
    sweep_dir = Path(sweep_dir)

    if not sweep_dir.exists():
        print(f"Error: {sweep_dir} does not exist")
        return None

    results = []

    # Find all config directories
    config_dirs = sorted(sweep_dir.glob('config_*'))

    print(f"Found {len(config_dirs)} config directories")

    for config_dir in config_dirs:
        metrics_file = config_dir / 'metrics.csv'
        args_file = config_dir / 'args.json'

        if not metrics_file.exists():
            print(f"Warning: {config_dir.name} missing metrics.csv")
            continue

        # Load metrics
        try:
            metrics_df = pd.read_csv(metrics_file)
        except Exception as e:
            print(f"Error reading {metrics_file}: {e}")
            continue

        # Load hyperparameters
        if args_file.exists():
            with open(args_file, 'r') as f:
                hyperparams = json.load(f)
        else:
            hyperparams = {}

        # Get best validation loss
        best_val_loss = metrics_df['val_total_loss'].min()
        best_epoch = metrics_df.loc[metrics_df['val_total_loss'].idxmin(), 'epoch']
        final_epoch = metrics_df['epoch'].max()

        # Get final metrics
        final_metrics = metrics_df.iloc[-1]

        result = {
            'config_dir': config_dir.name,
            'best_val_loss': best_val_loss,
            'best_epoch': int(best_epoch),
            'final_epoch': int(final_epoch),
            'final_train_loss': final_metrics['train_total_loss'],
            'final_val_loss': final_metrics['val_total_loss'],

            # Hyperparameters
            'lr': hyperparams.get('lr'),
            'hidden_dim': hyperparams.get('hidden_dim'),
            'batch_size': hyperparams.get('batch_size'),
            'coord_weight': hyperparams.get('coord_weight'),
            'done_weight': hyperparams.get('done_weight'),
            'marking_alpha': hyperparams.get('marking_alpha'),
            'spatial_order': hyperparams.get('spatial_order'),
        }

        results.append(result)

    if not results:
        print("No results found!")
        return None

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Sort by best val loss
    results_df = results_df.sort_values('best_val_loss')

    # Save
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    return results_df


def print_summary(results_df, top_n=10):
    """Print summary of top results."""
    print("\n" + "="*80)
    print(f"Top {top_n} Configurations by Validation Loss")
    print("="*80)

    for idx, row in results_df.head(top_n).iterrows():
        print(f"\nRank {idx + 1}: {row['config_dir']}")
        print(f"  Best Val Loss: {row['best_val_loss']:.4f} (epoch {row['best_epoch']})")
        print(f"  Hyperparameters:")
        print(f"    LR: {row['lr']:.2e}")
        print(f"    Hidden Dim: {row['hidden_dim']}")
        print(f"    Batch Size: {row['batch_size']}")
        print(f"    Coord Weight: {row['coord_weight']}")
        print(f"    Done Weight: {row['done_weight']}")
        print(f"    Marking Alpha: {row['marking_alpha']}")
        print(f"    Spatial Order: {row['spatial_order']}")

    print("\n" + "="*80)
    print("Best Model:")
    print("="*80)
    best = results_df.iloc[0]
    print(f"Config: {best['config_dir']}")
    print(f"Val Loss: {best['best_val_loss']:.4f}")
    print(f"Checkpoint: {Path(args.sweep_dir) / best['config_dir'] / 'checkpoint_best.pt'}")


def main():
    global args
    parser = argparse.ArgumentParser(description='Analyze sweep results')
    parser.add_argument('--sweep_dir', type=str, default='sweep_outputs',
                       help='Directory containing sweep outputs')
    parser.add_argument('--output', type=str, default='sweep_results.csv',
                       help='Output CSV file')
    parser.add_argument('--top_n', type=int, default=10,
                       help='Number of top configs to show')

    args = parser.parse_args()

    results_df = analyze_sweep(args.sweep_dir, args.output)

    if results_df is not None:
        print_summary(results_df, args.top_n)


if __name__ == "__main__":
    main()
