#!/bin/bash
# Setup script for HPC environment with local cache directories

echo "=================================================="
echo "HPC Environment Setup for Qwen3-VL FSC147 Training"
echo "=================================================="

# Get the current directory
WORK_DIR="$(pwd)"
echo "Working directory: $WORK_DIR"

# Create local cache directories
echo ""
echo "Creating local cache directories..."
mkdir -p $WORK_DIR/.cache/huggingface
mkdir -p $WORK_DIR/.cache/torch
mkdir -p $WORK_DIR/.cache/pip
mkdir -p $WORK_DIR/.cache/wandb
mkdir -p $WORK_DIR/.cache/datasets

# Export environment variables for current session
export HF_HOME="$WORK_DIR/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$WORK_DIR/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$WORK_DIR/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$WORK_DIR/.cache/huggingface/datasets"
export TORCH_HOME="$WORK_DIR/.cache/torch"
export PIP_CACHE_DIR="$WORK_DIR/.cache/pip"
export WANDB_DIR="$WORK_DIR/.cache/wandb"
export WANDB_CACHE_DIR="$WORK_DIR/.cache/wandb"
export WANDB_DATA_DIR="$WORK_DIR/.cache/wandb"

# Also set XDG cache home to avoid any other tools using ~/.cache
export XDG_CACHE_HOME="$WORK_DIR/.cache"

# Create a env file that can be sourced
cat > $WORK_DIR/set_cache_env.sh << 'EOF'
#!/bin/bash
# Source this file to set cache directories: source ./set_cache_env.sh

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Hugging Face caches
export HF_HOME="$WORK_DIR/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$WORK_DIR/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$WORK_DIR/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$WORK_DIR/.cache/huggingface/datasets"

# PyTorch cache
export TORCH_HOME="$WORK_DIR/.cache/torch"

# Pip cache
export PIP_CACHE_DIR="$WORK_DIR/.cache/pip"

# Weights & Biases cache
export WANDB_DIR="$WORK_DIR/.cache/wandb"
export WANDB_CACHE_DIR="$WORK_DIR/.cache/wandb"
export WANDB_DATA_DIR="$WORK_DIR/.cache/wandb"

# XDG cache (catch-all for other tools)
export XDG_CACHE_HOME="$WORK_DIR/.cache"

echo "Cache directories set to: $WORK_DIR/.cache/"
echo "  HF_HOME=$HF_HOME"
echo "  TORCH_HOME=$TORCH_HOME"
echo "  WANDB_DIR=$WANDB_DIR"
EOF

chmod +x $WORK_DIR/set_cache_env.sh

# Create Python wrapper that sets environment variables
cat > $WORK_DIR/train_with_local_cache.py << 'EOF'
#!/usr/bin/env python3
"""
Wrapper script that ensures all caches are in local directory
"""
import os
import sys
from pathlib import Path

# Get the directory where this script is located
WORK_DIR = Path(__file__).parent.absolute()

# Set all cache environment variables
os.environ['HF_HOME'] = str(WORK_DIR / '.cache' / 'huggingface')
os.environ['HUGGINGFACE_HUB_CACHE'] = str(WORK_DIR / '.cache' / 'huggingface' / 'hub')
os.environ['TRANSFORMERS_CACHE'] = str(WORK_DIR / '.cache' / 'huggingface' / 'transformers')
os.environ['HF_DATASETS_CACHE'] = str(WORK_DIR / '.cache' / 'huggingface' / 'datasets')
os.environ['TORCH_HOME'] = str(WORK_DIR / '.cache' / 'torch')
os.environ['PIP_CACHE_DIR'] = str(WORK_DIR / '.cache' / 'pip')
os.environ['WANDB_DIR'] = str(WORK_DIR / '.cache' / 'wandb')
os.environ['WANDB_CACHE_DIR'] = str(WORK_DIR / '.cache' / 'wandb')
os.environ['WANDB_DATA_DIR'] = str(WORK_DIR / '.cache' / 'wandb')
os.environ['XDG_CACHE_HOME'] = str(WORK_DIR / '.cache')

# Create cache directories if they don't exist
for cache_dir in [
    WORK_DIR / '.cache' / 'huggingface' / 'hub',
    WORK_DIR / '.cache' / 'huggingface' / 'transformers',
    WORK_DIR / '.cache' / 'huggingface' / 'datasets',
    WORK_DIR / '.cache' / 'torch',
    WORK_DIR / '.cache' / 'pip',
    WORK_DIR / '.cache' / 'wandb'
]:
    cache_dir.mkdir(parents=True, exist_ok=True)

print(f"Cache directories configured in: {WORK_DIR / '.cache'}")

# Import and run the actual training script
sys.path.insert(0, str(WORK_DIR))
import train_fsc147_optimized
train_fsc147_optimized.main()
EOF

chmod +x $WORK_DIR/train_with_local_cache.py

# Create SLURM job script template for HPC
cat > $WORK_DIR/slurm_train.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=qwen_fsc147
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%j_train.out
#SBATCH --error=%j_train.err

# Load modules (adjust based on your HPC system)
# module load cuda/11.8
# module load python/3.9

# Source cache environment
source ./set_cache_env.sh

# Activate virtual environment if using one
# source venv/bin/activate

# Run training with local cache
python train_with_local_cache.py \
    --data_dir ./FSC147 \
    --epochs 10 \
    --lr 2e-6 \
    --gradient_accumulation 4 \
    --use_wandb
EOF

chmod +x $WORK_DIR/slurm_train.sh

echo ""
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "Cache configuration files created:"
echo "  1. set_cache_env.sh - Source this to set environment variables"
echo "  2. train_with_local_cache.py - Python wrapper with cache settings"
echo "  3. slurm_train.sh - SLURM job template (adjust as needed)"
echo ""
echo "To use:"
echo "  1. Source the environment: source ./set_cache_env.sh"
echo "  2. Run training: python train_with_local_cache.py [args]"
echo "  Or submit SLURM job: sbatch slurm_train.sh"
echo ""
echo "All caches will be stored in: $WORK_DIR/.cache/"
echo "=================================================="