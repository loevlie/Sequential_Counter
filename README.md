# Sequential Counting with Visual Marking

Training code for sequential object counting inspired by how children learn to count.

## Model Architecture

- **Visual Encoder**: CLIP ViT-B/32 (frozen)
- **Point Predictor**: Cross-attention network (~890K parameters)
- **Training Strategy**: Random prefix with spatial ordering

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download OmniCount-191 from: https://github.com/mondalanindya/OmniCount

```bash
git clone https://github.com/mondalanindya/OmniCount.git
cd OmniCount
unzip OmniCount-191.zip
```

### 3. Test Training

```bash
python test_training.py
```

This runs 2 epochs on a small subset to verify everything works.

### 4. Full Training

```bash
python train.py \
  --data_root /path/to/OmniCount-191 \
  --categories Supermarket Fruits Urban \
  --batch_size 8 \
  --epochs 50 \
  --lr 1e-4 \
  --output_dir outputs/run_001
```

## Hyperparameter Sweep (HPC)

### On SLURM Cluster

1. **Generate sweep configs:**
```bash
python sweep.py --generate
```

This creates `sweep_configs.json` with 216 configurations.

2. **Edit SLURM script:**

Edit `run_sweep.sbatch`:
- Set `DATA_ROOT` to your OmniCount-191 path
- Adjust partition, resources based on your cluster
- Create virtual environment first

3. **Submit array job:**
```bash
# Create logs directory
mkdir -p logs

# Submit
sbatch run_sweep.sbatch
```

This runs 216 jobs (10 parallel) with different hyperparameter combinations.

4. **Monitor jobs:**
```bash
squeue -u $USER
```

5. **Analyze results:**
```bash
python analyze_sweep.py --sweep_dir sweep_outputs
```

This shows top-10 configurations and identifies the best model.

## Files

- `dataset.py` - OmniCount-191 dataset loader with spatial ordering
- `model.py` - Point prediction network
- `utils.py` - Visual marking utilities
- `train.py` - Training script with early stopping, LR scheduler, CSV logging
- `test_training.py` - Quick test to verify setup
- `sweep.py` - Hyperparameter sweep configuration generator
- `run_sweep.sbatch` - SLURM batch array script
- `analyze_sweep.py` - Analyze sweep results

## Output Structure

```
outputs/run_001/
├── checkpoint_best.pt         # Best model (lowest val loss)
├── checkpoint_latest.pt        # Latest checkpoint
├── checkpoint_epoch_N.pt       # Per-epoch checkpoints
├── metrics.csv                 # Training metrics (all epochs)
└── args.json                   # Hyperparameters used
```

## Hyperparameters

Key hyperparameters in sweep:
- `lr`: Learning rate [1e-5, 5e-5, 1e-4]
- `hidden_dim`: Hidden dimension [128, 256, 512]
- `batch_size`: Batch size [8, 16]
- `coord_weight`: Coordinate loss weight [1.0, 2.0]
- `done_weight`: Done signal loss weight [0.5, 1.0]
- `marking_alpha`: Visual marking transparency [0.2, 0.3, 0.5]
- `spatial_order`: Ordering strategy [reading_order, nearest_neighbor]

Total: 3×3×2×2×2×3×2 = **216 configurations**

## Training Features

✅ **Early Stopping**: Stops when val loss doesn't improve (patience=7)
✅ **LR Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
✅ **CSV Logging**: Saves metrics every epoch
✅ **Checkpointing**: Saves best, latest, and per-epoch checkpoints
✅ **Random Prefix**: Unbiased training strategy
✅ **Spatial Ordering**: Reading order or nearest neighbor

## Metrics Tracked

Saved to `metrics.csv` every epoch:
- `train_total_loss`, `train_coord_loss`, `train_done_loss`, `train_consistency_loss`
- `val_total_loss`, `val_coord_loss`, `val_done_loss`, `val_consistency_loss`
- `learning_rate`
- `timestamp`

## Citation

If you use this code, please cite:

```bibtex
@article{omnicount2025,
  title={OmniCount: Multi-label Object Counting with Semantic-Geometric Priors},
  author={Mondal, Anindya and Nag, Sauradip and Zhu, Xiatian and Dutta, Anjan},
  journal={AAAI},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details
