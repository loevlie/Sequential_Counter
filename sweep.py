#!/usr/bin/env python3
"""
Hyperparameter Sweep for Sequential Counting

Generates training configurations for grid search.
Designed to work with SLURM array jobs.
"""

import json
import itertools
from pathlib import Path
import argparse


def generate_sweep_configs(output_file='sweep_configs.json'):
    """
    Generate hyperparameter configurations for sweep.

    Returns list of config dicts.
    """

    # Hyperparameter grid
    grid = {
        'lr': [1e-4, 5e-5, 1e-5],
        'hidden_dim': [128, 256, 512],
        'batch_size': [8, 16],
        'coord_weight': [1.0, 2.0],
        'done_weight': [0.5, 1.0],
        'marking_alpha': [0.2, 0.3, 0.5],
        'spatial_order': ['reading_order', 'nearest_neighbor']
    }

    # Generate all combinations
    keys = list(grid.keys())
    values = list(grid.values())
    combinations = list(itertools.product(*values))

    configs = []
    for idx, combo in enumerate(combinations):
        config = dict(zip(keys, combo))
        config['config_id'] = idx
        configs.append(config)

    print(f"Generated {len(configs)} configurations")

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(configs, f, indent=2)

    print(f"Saved to: {output_file}")

    return configs


def get_config_by_id(config_id, config_file='sweep_configs.json'):
    """Get specific config by ID (for SLURM array job)."""
    with open(config_file, 'r') as f:
        configs = json.load(f)

    if config_id >= len(configs):
        raise ValueError(f"Config ID {config_id} out of range (max: {len(configs)-1})")

    return configs[config_id]


def generate_sweep_script(config, data_root, base_output_dir='sweep_outputs'):
    """Generate command line for a specific config."""

    output_dir = Path(base_output_dir) / f"config_{config['config_id']:04d}"

    cmd = [
        "python", "train.py",
        "--data_root", data_root,
        "--categories", "Supermarket", "Fruits", "Urban",
        "--min_objects", "5",
        "--max_objects", "30",
        "--image_size", "224",
        "--epochs", "50",
        "--early_stop_patience", "7",
        "--num_workers", "4",
        "--output_dir", str(output_dir),

        # Sweep params
        "--lr", str(config['lr']),
        "--hidden_dim", str(config['hidden_dim']),
        "--batch_size", str(config['batch_size']),
        "--coord_weight", str(config['coord_weight']),
        "--done_weight", str(config['done_weight']),
        "--marking_alpha", str(config['marking_alpha']),
        "--spatial_order", config['spatial_order']
    ]

    return " ".join(cmd)


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter sweep')
    parser.add_argument('--generate', action='store_true',
                       help='Generate sweep configs')
    parser.add_argument('--get_command', type=int, metavar='ID',
                       help='Get command for config ID')
    parser.add_argument('--data_root', type=str,
                       help='Path to OmniCount-191 dataset')
    parser.add_argument('--config_file', type=str, default='sweep_configs.json')
    parser.add_argument('--output_dir', type=str, default='sweep_outputs')

    args = parser.parse_args()

    if args.generate:
        # Generate configs
        generate_sweep_configs(args.config_file)

    elif args.get_command is not None:
        # Get command for specific config
        if not args.data_root:
            print("Error: --data_root required")
            return 1

        config = get_config_by_id(args.get_command, args.config_file)
        cmd = generate_sweep_script(config, args.data_root, args.output_dir)
        print(cmd)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
