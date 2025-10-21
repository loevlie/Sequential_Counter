#!/usr/bin/env python3
"""
Quick test script to verify training works.

Runs 2 epochs on a small subset to ensure everything is set up correctly.
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("="*80)
    print("Testing Training Setup")
    print("="*80)
    print("\nThis will run 2 epochs on a small dataset to verify everything works.")
    print("Expected time: ~10-30 minutes depending on hardware\n")

    # Get data root from user
    data_root = input("Enter path to OmniCount-191 dataset\n(e.g., /path/to/OmniCount-191): ").strip()

    if not Path(data_root).exists():
        print(f"\nError: Dataset not found at {data_root}")
        sys.exit(1)

    # Test command
    cmd = [
        "python", "train.py",
        "--data_root", data_root,
        "--categories", "Supermarket",  # Just one category
        "--min_objects", "10",
        "--max_objects", "20",  # Narrow range
        "--image_size", "224",
        "--batch_size", "4",  # Small batch
        "--epochs", "2",  # Just 2 epochs
        "--lr", "1e-4",
        "--hidden_dim", "128",  # Smaller model
        "--early_stop_patience", "10",
        "--output_dir", "test_run",
        "--num_workers", "2"
    ]

    print("\nRunning command:")
    print(" ".join(cmd))
    print()

    try:
        subprocess.run(cmd, check=True)
        print("\n" + "="*80)
        print("✓ Test Training Successful!")
        print("="*80)
        print("\nYour setup is ready for full training.")
        print("Check test_run/ for outputs.")
        print("\nNext: Run full training or hyperparameter sweep.")
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "="*80)
        print("✗ Test Training Failed")
        print("="*80)
        print(f"\nError: {e}")
        print("\nCheck for missing dependencies or dataset issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
