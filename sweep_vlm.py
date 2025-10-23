#!/usr/bin/env python3
"""
Hyperparameter Sweep Configuration for VLM Training

Generates configurations for Qwen3-VL-4B-Thinking model hyperparameter sweep.
Optimized grid for 4B model (faster than 11B models).
"""

import json
import argparse
import itertools


def generate_configs():
    """Generate all hyperparameter configurations."""

    # VLM-specific hyperparameter grid
    grid = {
        'lr': [5e-6, 1e-5, 2e-5],  # Lower LRs for LoRA
        'lora_r': [8, 16, 32],  # LoRA rank
        'lora_alpha': [16, 32],  # LoRA alpha
        'batch_size': [2, 4],  # Smaller batches (VLM is large)
        'marking_alpha': [0.2, 0.3, 0.5],
        'spatial_order': ['reading_order', 'nearest_neighbor'],
        'max_new_tokens': [64, 128],  # Token generation limit
    }

    # Generate all combinations
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combinations = list(itertools.product(*values))

    configs = []
    for combo in combinations:
        config = dict(zip(keys, combo))
        configs.append(config)

    print(f"Generated {len(configs)} configurations")
    print(f"Grid: {grid}")

    return configs


def get_training_command(config_id, data_root, configs=None):
    """Get training command for a specific config ID."""

    if configs is None:
        # Load from file
        with open('sweep_vlm_configs.json', 'r') as f:
            configs = json.load(f)

    if config_id >= len(configs):
        raise ValueError(f"Config ID {config_id} out of range (max: {len(configs)-1})")

    config = configs[config_id]

    # Build command
    cmd_parts = [
        "python train_vlm.py",
        f"--data_root {data_root}",
        f"--lr {config['lr']}",
        f"--lora_r {config['lora_r']}",
        f"--lora_alpha {config['lora_alpha']}",
        f"--batch_size {config['batch_size']}",
        f"--marking_alpha {config['marking_alpha']}",
        f"--spatial_order {config['spatial_order']}",
        f"--max_new_tokens {config['max_new_tokens']}",
        "--epochs 50",
        "--early_stopping_patience 7",
        "--lr_scheduler_patience 3",
        "--lr_scheduler_factor 0.5",
        f"--output_dir sweep_vlm_outputs/config_{config_id:04d}",
        "--num_workers 4",
        "--load_in_4bit",
        f"--model_name Qwen/Qwen3-VL-4B-Thinking"
    ]

    return " ".join(cmd_parts)


def main():
    parser = argparse.ArgumentParser(description='VLM Hyperparameter sweep')
    parser.add_argument('--generate', action='store_true',
                       help='Generate sweep configurations')
    parser.add_argument('--get_command', type=int, metavar='CONFIG_ID',
                       help='Get training command for config ID')
    parser.add_argument('--data_root', type=str,
                       help='Path to OmniCount-191 dataset')

    args = parser.parse_args()

    if args.generate:
        # Generate and save configurations
        configs = generate_configs()

        with open('sweep_vlm_configs.json', 'w') as f:
            json.dump(configs, f, indent=2)

        print(f"\nSaved {len(configs)} configurations to sweep_vlm_configs.json")
        print("\nGrid breakdown:")
        print(f"  Learning rates: [5e-6, 1e-5, 2e-5]")
        print(f"  LoRA rank: [8, 16, 32]")
        print(f"  LoRA alpha: [16, 32]")
        print(f"  Batch sizes: [2, 4]")
        print(f"  Marking alpha: [0.2, 0.3, 0.5]")
        print(f"  Spatial order: [reading_order, nearest_neighbor]")
        print(f"  Max tokens: [64, 128]")
        print(f"\nTotal: 3×3×2×2×3×2×2 = {len(configs)} configurations")

    elif args.get_command is not None:
        if args.data_root is None:
            print("Error: --data_root required with --get_command")
            return

        cmd = get_training_command(args.get_command, args.data_root)
        print(cmd)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
