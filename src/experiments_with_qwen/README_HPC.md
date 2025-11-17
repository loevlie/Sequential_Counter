# Qwen3-VL FSC147 Training - HPC Deployment Guide

## Quick Start

### 1. Clone and Setup Environment
```bash
# Clone the repository
git clone [your-repo-url]
cd Sequential_Counter/src/experiments_with_qwen

# Setup HPC environment with local caches
chmod +x setup_hpc_environment.sh
./setup_hpc_environment.sh

# Source the cache environment
source ./set_cache_env.sh
```

### 2. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Download FSC147 Dataset
```bash
# Download dataset to local directory
chmod +x download_fsc147.sh
./download_fsc147.sh ./FSC147

# This will download ~3GB of data and set up:
# FSC147/
# ├── annotation_FSC147_384.json
# ├── Train_Test_Val_FSC_147.json
# └── images_384_VarV2/
```

### 4. Run Training

#### Option A: Direct execution
```bash
# Make sure cache environment is sourced
source ./set_cache_env.sh

# Run training
python train_fsc147_optimized.py \
    --data_dir ./FSC147 \
    --model_name Qwen/Qwen3-VL-2B-Instruct \
    --epochs 10 \
    --lr 2e-6 \
    --gradient_accumulation 4 \
    --checkpoint_dir ./checkpoints \
    --use_wandb
```

#### Option B: Using the launch script
```bash
# Edit start_optimized_training.sh to set DATA_DIR="./FSC147"
./start_optimized_training.sh
```

#### Option C: SLURM submission
```bash
# Edit slurm_train.sh as needed for your cluster
sbatch slurm_train.sh
```

## Important Configuration

### Cache Management
**All caches are stored locally in `./.cache/` to avoid filling home directory:**
- Hugging Face models: `./.cache/huggingface/`
- PyTorch models: `./.cache/torch/`
- Weights & Biases: `./.cache/wandb/`

### A100 GPU Optimizations
For A100 GPUs, you can enable attention loss by disabling gradient checkpointing:

Edit `train_fsc147_optimized.py`:
```python
# Comment out or set to False:
# self.model.gradient_checkpointing_enable()
self.use_gradient_checkpointing = False
```

Or increase batch size:
```bash
python train_fsc147_optimized.py \
    --gradient_accumulation 8 \  # or higher
    --image_size 384 \  # larger images
    ...
```

## Files Overview

### Core Training Files
- `train_fsc147_optimized.py` - Main optimized training script with dual loss
- `start_optimized_training.sh` - Launch script with optimal parameters
- `requirements.txt` - Python dependencies

### Setup Files
- `setup_hpc_environment.sh` - Configures local cache directories
- `download_fsc147.sh` - Downloads FSC147 dataset
- `set_cache_env.sh` - Environment variables (auto-generated)

### Documentation
- `OPTIMIZED_TRAINING_SUMMARY.md` - Technical details about the implementation

## Training Parameters

### Recommended Settings for A100
```bash
--epochs 10-20
--lr 2e-6
--gradient_accumulation 8
--image_size 224  # or 384 for better accuracy
--checkpoint_dir ./checkpoints
```

### Memory Considerations
- **24GB GPU**: Use default settings
- **40GB+ GPU (A100)**:
  - Disable gradient checkpointing for attention loss
  - Increase batch size via gradient_accumulation
  - Use larger image sizes (384x384)

## Monitoring

### Weights & Biases
Training progress is logged to W&B:
```bash
# Login to wandb (first time only)
wandb login

# View runs at:
# https://wandb.ai/[your-username]/qwen-fsc147-optimized
```

### Local Checkpoints
Models are saved to `./checkpoints/` directory:
```
checkpoints/
├── checkpoint_epoch1.pt
├── checkpoint_epoch2.pt
└── ...
```

## Troubleshooting

### Out of Memory
- Reduce `gradient_accumulation` parameter
- Keep gradient checkpointing enabled
- Reduce image size to 224

### Slow Download
- The FSC147 dataset is ~3GB
- If Google Drive download fails, try the wget alternative in the script
- Consider downloading locally and transferring via scp

### Cache Issues
- Ensure `source ./set_cache_env.sh` is run before training
- Check disk space in working directory (need ~20GB free for models + data)

## Expected Results

With proper training:
- **MAE**: ~15-20 on FSC147 validation set
- **Training time**: ~2-4 hours per epoch on A100
- **Memory usage**: ~20GB VRAM with default settings

## Contact
For issues specific to this implementation, check the training summaries in:
- `OPTIMIZED_TRAINING_SUMMARY.md`
- W&B logs for detailed metrics