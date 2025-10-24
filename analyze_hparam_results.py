#!/usr/bin/env python3
"""
Analyze hyperparameter search results from W&B.

This script pulls results from W&B and creates visualizations showing
which hyperparameters lead to the best performance.

Usage:
    python analyze_hparam_results.py --project sequential-counting-hparam-search
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from pathlib import Path

def fetch_wandb_runs(project_name, entity=None):
    """Fetch all runs from a W&B project."""
    api = wandb.Api()

    if entity:
        project_path = f"{entity}/{project_name}"
    else:
        project_path = project_name

    runs = api.runs(project_path)

    summary_list = []
    config_list = []
    name_list = []

    for run in runs:
        # Skip failed runs
        if run.state != "finished":
            continue

        summary_list.append(run.summary._json_dict)
        config_list.append({k: v for k, v in run.config.items() if not k.startswith('_')})
        name_list.append(run.name)

    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({'name': name_list})

    all_df = pd.concat([name_df, config_df, summary_df], axis=1)

    return all_df


def plot_hparam_importance(df, output_dir):
    """Create plots showing hyperparameter importance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Metrics to analyze
    metrics = ['best_val_loss', 'val/loss', 'val/loss_spatial', 'val/loss_count']

    # Find which metric columns actually exist
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        print("Warning: No validation metrics found in results")
        return

    primary_metric = available_metrics[0]

    # 1. Learning rate vs performance
    if 'lr' in df.columns:
        plt.figure(figsize=(10, 6))
        for lora_r in df['lora_r'].unique():
            subset = df[df['lora_r'] == lora_r]
            plt.plot(subset['lr'], subset[primary_metric],
                    marker='o', label=f'LoRA r={lora_r}')
        plt.xlabel('Learning Rate')
        plt.ylabel('Validation Loss')
        plt.xscale('log')
        plt.title('Learning Rate vs Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'lr_vs_loss.png', dpi=150)
        plt.close()

    # 2. LoRA configuration heatmap
    if 'lora_r' in df.columns and 'lora_alpha' in df.columns:
        pivot = df.pivot_table(values=primary_metric,
                               index='lora_r',
                               columns='lora_alpha',
                               aggfunc='mean')

        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='viridis_r')
        plt.title('LoRA Configuration: Validation Loss')
        plt.xlabel('LoRA Alpha')
        plt.ylabel('LoRA Rank')
        plt.tight_layout()
        plt.savefig(output_dir / 'lora_heatmap.png', dpi=150)
        plt.close()

    # 3. MLP layers impact
    if 'mlp_layers' in df.columns:
        plt.figure(figsize=(10, 6))
        df_grouped = df.groupby('mlp_layers')[primary_metric].agg(['mean', 'std'])
        plt.errorbar(df_grouped.index, df_grouped['mean'],
                    yerr=df_grouped['std'], marker='o', capsize=5, capthick=2)
        plt.xlabel('Number of MLP Layers')
        plt.ylabel('Validation Loss')
        plt.title('MLP Depth vs Performance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'mlp_depth.png', dpi=150)
        plt.close()

    # 4. Batch size impact
    if 'batch_size' in df.columns:
        plt.figure(figsize=(10, 6))
        df_grouped = df.groupby('batch_size')[primary_metric].agg(['mean', 'std'])
        plt.errorbar(df_grouped.index, df_grouped['mean'],
                    yerr=df_grouped['std'], marker='o', capsize=5, capthick=2)
        plt.xlabel('Batch Size')
        plt.ylabel('Validation Loss')
        plt.title('Batch Size vs Performance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'batch_size.png', dpi=150)
        plt.close()

    # 5. Loss component breakdown for best models
    if all(m in df.columns for m in ['val/loss_x', 'val/loss_y', 'val/loss_count']):
        # Get top 10 models
        top_10 = df.nsmallest(10, primary_metric)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(top_10))
        width = 0.25

        ax.bar([i - width for i in x], top_10['val/loss_x'], width, label='X loss')
        ax.bar(x, top_10['val/loss_y'], width, label='Y loss')
        ax.bar([i + width for i in x], top_10['val/loss_count'], width, label='Count loss')

        ax.set_xlabel('Model Rank')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Component Breakdown for Top 10 Models')
        ax.set_xticks(x)
        ax.set_xticklabels([f'#{i+1}' for i in x])
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'top_models_breakdown.png', dpi=150)
        plt.close()

    print(f"\nPlots saved to {output_dir}")


def print_best_configs(df, n=5):
    """Print the best hyperparameter configurations."""
    # Find the validation loss column
    val_cols = [c for c in df.columns if 'val' in c.lower() and 'loss' in c.lower()]

    if not val_cols:
        print("No validation loss columns found")
        return

    primary_metric = val_cols[0]

    print(f"\n{'='*80}")
    print(f"TOP {n} HYPERPARAMETER CONFIGURATIONS")
    print(f"{'='*80}\n")

    # Relevant columns for display
    config_cols = ['lr', 'lora_r', 'lora_alpha', 'mlp_layers', 'batch_size']
    metric_cols = [c for c in df.columns if 'val' in c.lower() or 'best' in c.lower()]

    display_cols = ['name'] + [c for c in config_cols if c in df.columns] + \
                   [c for c in metric_cols if c in df.columns]

    top_n = df.nsmallest(n, primary_metric)[display_cols]

    for idx, (i, row) in enumerate(top_n.iterrows(), 1):
        print(f"Rank {idx}:")
        print(f"  Run: {row['name']}")
        for col in config_cols:
            if col in row:
                print(f"  {col}: {row[col]}")
        for col in metric_cols:
            if col in row and pd.notna(row[col]):
                print(f"  {col}: {row[col]:.4f}")
        print()

    # Print summary statistics
    print(f"\n{'='*80}")
    print("HYPERPARAMETER STATISTICS")
    print(f"{'='*80}\n")

    for col in config_cols:
        if col in df.columns:
            best_val = df.loc[df[primary_metric].idxmin(), col]
            print(f"{col}:")
            print(f"  Best value: {best_val}")
            print(f"  Mean: {df[col].mean():.4f}")
            print(f"  Std: {df[col].std():.4f}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hyperparameter search results from W&B"
    )
    parser.add_argument(
        '--project',
        type=str,
        default='sequential-counting-hparam-search',
        help='W&B project name'
    )
    parser.add_argument(
        '--entity',
        type=str,
        default=None,
        help='W&B entity (username or team name)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='hparam_analysis',
        help='Directory to save analysis plots'
    )
    parser.add_argument(
        '--export_csv',
        type=str,
        default=None,
        help='Export results to CSV file'
    )

    args = parser.parse_args()

    print(f"Fetching runs from W&B project: {args.project}")
    df = fetch_wandb_runs(args.project, args.entity)

    print(f"Found {len(df)} completed runs")

    if len(df) == 0:
        print("No completed runs found. Exiting.")
        return

    # Print best configurations
    print_best_configs(df, n=5)

    # Create visualizations
    print("\nCreating visualizations...")
    plot_hparam_importance(df, args.output_dir)

    # Export to CSV if requested
    if args.export_csv:
        df.to_csv(args.export_csv, index=False)
        print(f"\nResults exported to: {args.export_csv}")

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
