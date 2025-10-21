#!/usr/bin/env python3
"""
Quick Test Script for VLM Training

Runs 2 epochs on a single category to verify:
1. Model loading works
2. LoRA setup is correct
3. Training loop runs without errors
4. Metrics are saved properly

Expected runtime: ~30-60 minutes (with 4-bit quantization)
"""

import subprocess
import sys
from pathlib import Path


def test_vlm_training():
    """Run quick VLM training test."""

    print("="*80)
    print("VLM TRAINING TEST - LLaVA-CoT")
    print("="*80)
    print()
    print("This will:")
    print("  - Load Llama-3.2-11B-Vision-Instruct with 4-bit quantization")
    print("  - Apply LoRA adapters (~16-32M trainable params)")
    print("  - Train for 2 epochs on Supermarket category")
    print("  - Test batch size: 2 (lower for memory constraints)")
    print()
    print("Expected time: ~30-60 minutes")
    print("GPU memory: ~16-24GB")
    print()

    # Get dataset path from user
    data_root = input("Enter path to OmniCount-191 dataset: ").strip()

    if not Path(data_root).exists():
        print(f"Error: Path does not exist: {data_root}")
        sys.exit(1)

    print(f"\nUsing dataset: {data_root}")
    print()

    # Training parameters
    cmd = [
        "python", "train_vlm.py",
        "--data_root", data_root,
        "--categories", "Supermarket",  # Single category
        "--batch_size", "2",  # Small batch for testing
        "--epochs", "2",  # Just 2 epochs
        "--lr", "1e-5",
        "--lora_r", "16",
        "--lora_alpha", "32",
        "--marking_alpha", "0.3",
        "--spatial_order", "reading_order",
        "--early_stopping_patience", "10",  # Disable early stop for test
        "--lr_scheduler_patience", "5",
        "--output_dir", "test_vlm_run",
        "--num_workers", "2",
        "--load_in_4bit"  # 4-bit quantization for efficiency
    ]

    print("Running command:")
    print(" ".join(cmd))
    print()
    print("="*80)
    print()

    # Run training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹ Training interrupted by user")
        sys.exit(1)

    # Check outputs
    print()
    print("="*80)
    print("VERIFICATION")
    print("="*80)

    output_dir = Path("test_vlm_run")

    if not output_dir.exists():
        print("❌ Output directory not created")
        sys.exit(1)

    # Check files
    expected_files = [
        "args.json",
        "metrics.csv",
        "checkpoint_latest.pt",
        "checkpoint_epoch_0.pt",
        "checkpoint_epoch_1.pt",
    ]

    for fname in expected_files:
        fpath = output_dir / fname
        if fpath.exists() or any(output_dir.glob(f"{fname.replace('.pt', '')}_lora")):
            print(f"✅ {fname}")
        else:
            print(f"❌ {fname} - NOT FOUND")

    # Check metrics
    metrics_file = output_dir / "metrics.csv"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
        print(f"\n✅ Metrics saved: {len(lines) - 1} epochs logged")

        # Print last epoch
        if len(lines) > 1:
            header = lines[0].strip()
            last_epoch = lines[-1].strip()
            print(f"\nLast epoch data:")
            print(f"  {header}")
            print(f"  {last_epoch}")

    print()
    print("="*80)
    print("✅ VLM TRAINING TEST COMPLETE!")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Check test_vlm_run/ for outputs")
    print("  2. Review metrics.csv for training progress")
    print("  3. If successful, run full sweep with train_vlm.py")
    print()


if __name__ == "__main__":
    test_vlm_training()
