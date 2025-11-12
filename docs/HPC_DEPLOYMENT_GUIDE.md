# HPC Deployment Guide

## What You Have

A complete, production-ready training codebase for sequential object counting:

✅ **11 files, 1,666 lines of code**
✅ **Git repository ready to clone**
✅ **Hyperparameter sweep (216 configs)**
✅ **Full metrics tracking & early stopping**

## Repository Structure

```
sequential-counting/
├── README.md                   # Main documentation
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── dataset.py                  # OmniCount-191 loader with spatial ordering
├── model.py                    # Point prediction network
├── utils.py                    # Visual marking utilities
├── train.py                    # Training script (early stop, LR sched, CSV logging)
│
├── test_training.py            # Quick test (2 epochs)
├── sweep.py                    # Hyperparameter sweep generator
├── run_sweep.sbatch            # SLURM array job script
├── analyze_sweep.py            # Find best model from sweep
│
└── HPC_DEPLOYMENT_GUIDE.md     # This file
```

## Step-by-Step HPC Deployment

### 1. Clone on HPC

```bash
# SSH to your HPC
ssh your_hpc_cluster

# Clone the repo
cd ~
git clone /path/to/sequential-counting
cd sequential-counting
```

Or push to GitHub first and clone:

```bash
# On your local machine
cd /Users/dennisloevlie/Jivko_Research/sequential-counting
git remote add origin https://github.com/YOUR_USERNAME/sequential-counting.git
git push -u origin master

# Then on HPC
git clone https://github.com/YOUR_USERNAME/sequential-counting.git
```

### 2. Setup Environment

```bash
# Load Python module (adjust for your HPC)
module load python/3.10
module load cuda/11.8

# Create virtual environment
python -m venv ~/venvs/counting
source ~/venvs/counting/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Copy Dataset

```bash
# Copy OmniCount-191 to HPC (if not already there)
# Option A: scp from your machine
scp -r /Users/dennisloevlie/Jivko_Research/counting/OmniCount/OmniCount-191 \
      your_hpc:/scratch/your_username/data/

# Option B: download on HPC
cd /scratch/your_username/data
git clone https://github.com/mondalanindya/OmniCount.git
cd OmniCount
unzip OmniCount-191.zip
```

### 4. Test Training (IMPORTANT!)

```bash
# Quick test before launching sweep
python test_training.py
# Enter path: /scratch/your_username/data/OmniCount-191

# This runs 2 epochs (~10-30 min) to verify everything works
```

If test succeeds, you're ready for the full sweep!

### 5. Run Hyperparameter Sweep

```bash
# Generate 216 configurations
python sweep.py --generate
# Creates: sweep_configs.json

# Edit SLURM script
nano run_sweep.sbatch
# Change:
#   - DATA_ROOT="/scratch/your_username/data/OmniCount-191"
#   - Partition name (if needed)
#   - Resource requests (if needed)

# Create logs directory
mkdir -p logs

# Submit array job
sbatch run_sweep.sbatch
```

This submits 216 jobs, running 10 in parallel.

### 6. Monitor Progress

```bash
# Check queue
squeue -u $USER

# Watch a specific job log
tail -f logs/sweep_JOBID_0.out

# Count completed jobs
ls sweep_outputs/*/checkpoint_best.pt | wc -l
```

### 7. Analyze Results

After jobs complete:

```bash
python analyze_sweep.py --sweep_dir sweep_outputs

# Shows top-10 configs and best model
# Creates: sweep_results.csv
```

Output example:
```
Top 10 Configurations by Validation Loss
==================================================

Rank 1: config_0142
  Best Val Loss: 0.8234 (epoch 23)
  Hyperparameters:
    LR: 5.00e-05
    Hidden Dim: 256
    Batch Size: 16
    Coord Weight: 1.0
    Done Weight: 0.5
    Marking Alpha: 0.3
    Spatial Order: reading_order

Best Model:
Checkpoint: sweep_outputs/config_0142/checkpoint_best.pt
```

## Hyperparameter Grid

Current sweep covers **216 configurations**:

| Parameter | Values |
|-----------|--------|
| Learning Rate | 1e-5, 5e-5, 1e-4 (3) |
| Hidden Dim | 128, 256, 512 (3) |
| Batch Size | 8, 16 (2) |
| Coord Weight | 1.0, 2.0 (2) |
| Done Weight | 0.5, 1.0 (2) |
| Marking Alpha | 0.2, 0.3, 0.5 (3) |
| Spatial Order | reading_order, nearest_neighbor (2) |

**Total**: 3×3×2×2×2×3×2 = 216 configs

## Resource Requirements

Per job:
- **Time**: ~2-8 hours (with early stopping)
- **GPU**: 1 GPU (any modern GPU works)
- **RAM**: 32GB recommended
- **CPUs**: 4 cores
- **Storage**: ~500MB per config (checkpoints + metrics)

Total for sweep:
- **GPU hours**: ~432-1728 hours (10 parallel = ~43-173 wall hours)
- **Storage**: ~108GB for all configs

## Troubleshooting

### Test Training Fails

```bash
# Check Python version
python --version  # Should be 3.8+

# Check CUDA
nvidia-smi

# Check imports
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory

Reduce batch size in `run_sweep.sbatch`:
```bash
# In sweep.py, change:
'batch_size': [4, 8],  # Instead of [8, 16]
```

### Jobs Timing Out

Increase time limit:
```bash
#SBATCH --time=24:00:00  # 24 hours instead of 12
```

### Slow Data Loading

Set `num_workers=0` if filesystem is slow:
```python
# In train.py, line ~450
--num_workers 0
```

## After Finding Best Model

### Download Best Model

```bash
# On HPC, find best config (from analyze_sweep.py)
BEST_CONFIG="sweep_outputs/config_0142"

# Compress
tar -czf best_model.tar.gz $BEST_CONFIG/

# Download to local
scp your_hpc:~/sequential-counting/best_model.tar.gz .
```

### Evaluate on CountQA

(You'll implement this next step after training)

```python
# Load best model
checkpoint = torch.load('best_model/checkpoint_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Test on CountQA
# (evaluation code to be added)
```

## Tips

1. **Start small**: Run `test_training.py` first!
2. **Monitor early**: Check first few jobs to catch issues
3. **Save often**: Each epoch saves checkpoint (enabled by default)
4. **Use early stopping**: Saves time (enabled, patience=7)
5. **Check logs**: `logs/sweep_*.out` files show progress

## Next Steps

1. ✅ Clone repo to HPC
2. ✅ Setup environment
3. ✅ Test training
4. ✅ Run sweep
5. ⏸️ Analyze results
6. ⏸️ Evaluate best model on CountQA
7. ⏸️ Write paper!

---

**Questions?**
- Check README.md for detailed usage
- Check train.py for command-line arguments
- Check sweep.py to modify hyperparameter grid
