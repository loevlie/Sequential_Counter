# Hyperparameter Search for VLM Sequential Counting

This directory contains scripts for running a systematic hyperparameter search on SLURM.

## Quick Start

### 1. Submit the hyperparameter search job

```bash
sbatch run_hparam_search.sh
```

This will launch 72 parallel jobs (with max 10 running at once) that explore different hyperparameter combinations.

### 2. Monitor jobs

```bash
# Check job status
squeue -u dloevl01

# Monitor a specific job
tail -f slurm_logs/out/<job_id>_<array_id>.out

# Check W&B dashboard
# Navigate to: https://wandb.ai/<your-username>/sequential-counting-hparam-search
```

### 3. Analyze results

After jobs complete:

```bash
# Generate analysis plots and rankings
python analyze_hparam_results.py \
    --project sequential-counting-hparam-search \
    --output_dir hparam_analysis

# Export results to CSV
python analyze_hparam_results.py \
    --project sequential-counting-hparam-search \
    --export_csv hparam_results.csv
```

## Hyperparameter Grid

The search explores:

| Hyperparameter | Values | Rationale |
|----------------|--------|-----------|
| **Learning Rate** | 1e-3, 5e-4, 1e-4, 5e-5 | Critical for LoRA fine-tuning stability |
| **LoRA Rank** | 8, 16, 32 | Controls adapter capacity |
| **LoRA Alpha** | 16, 32 | Scaling factor for LoRA updates |
| **MLP Layers** | 2, 3, 4 | Depth of prediction head |
| **Batch Size** | 4 (fixed) | Use largest that fits in memory with 4-bit quantization |

**Total experiments:** 4 × 3 × 2 × 3 = **72 combinations**

## Customizing the Search

### Modify the hyperparameter grid

Edit `run_hparam_search.sh`:

```bash
# Add more learning rates
learning_rates=(1e-3 5e-4 1e-4 5e-5)

# Try larger LoRA ranks
lora_ranks=(8 16 32 64)

# Add weight decay
weight_decays=(0.0 0.01 0.1)
```

### Adjust SLURM settings

```bash
#SBATCH --array=0-71%10    # Current: 72 experiments, max 10 running at once
#SBATCH --time=24:00:00    # Increase if needed
#SBATCH --mem=64G          # Adjust based on your GPU
```

### Change the dataset or categories

In the experiment command:
```bash
--categories Supermarket Office Kitchen  # Multiple categories
--epochs 20  # More training epochs
```

## Results Structure

```
hparam_results/
├── lr0.001_r8_alpha16_mlp2/
│   ├── best_checkpoint/
│   └── latest_checkpoint/
├── lr0.0005_r16_alpha32_mlp3/
│   ├── best_checkpoint/
│   └── latest_checkpoint/
└── ...

hparam_analysis/
├── lr_vs_loss.png              # Learning rate impact
├── lora_heatmap.png            # LoRA config heatmap
├── mlp_depth.png               # MLP layers impact
├── batch_size.png              # Batch size impact
└── top_models_breakdown.png   # Loss component analysis
```

## Analysis Outputs

The analysis script generates:

1. **Learning Rate Plot**: Shows how learning rate affects validation loss
2. **LoRA Heatmap**: Visualizes best LoRA rank/alpha combinations
3. **MLP Depth**: Impact of prediction head complexity
4. **Batch Size**: Effect on training dynamics
5. **Top Models Breakdown**: Loss components for best models

It also prints:
- Top 5 hyperparameter configurations
- Summary statistics for each hyperparameter
- Best values for each hyperparameter

## Tips for Effective Search

### 1. Start with a coarse search

Run a small grid first to identify promising regions:
```bash
#SBATCH --array=0-11%4  # Just 12 experiments
```

### 2. Refine around best results

After initial search, zoom in on good regions:
```bash
# If lr=5e-4 and r=16 work well:
learning_rates=(3e-4 5e-4 7e-4)
lora_ranks=(12 16 20)
```

### 3. Use W&B parallel coordinates

```python
# In W&B UI:
# 1. Go to workspace
# 2. Create "Parallel Coordinates" plot
# 3. Add all hyperparameters and val/loss
# 4. Identify correlations visually
```

### 4. Check for overfitting

Compare train vs val loss:
```python
# In analysis script, add:
df['overfit_ratio'] = df['train/loss'] / df['val/loss']
```

## Troubleshooting

### Jobs fail immediately

```bash
# Check error logs
cat slurm_logs/err/<job_id>_<array_id>.err

# Common issues:
# - Wrong conda environment name
# - Data path doesn't exist
# - GPU not available
```

### Out of memory errors

Reduce batch size or use gradient accumulation:
```bash
--batch_size 1 \
--gradient_accumulation_steps 4  # Effective batch size = 4
```

### W&B login issues

```bash
# Login before running jobs
wandb login <your-api-key>

# Or disable W&B
export WANDB_MODE=disabled
```

## Next Steps After Search

1. **Select best model**:
   ```bash
   # Copy best checkpoint
   cp -r hparam_results/lr0.0005_r16_alpha32_mlp3_bs4/best_checkpoint ./best_model
   ```

2. **Evaluate on test set**:
   ```bash
   python evaluate_vlm_regression.py \
       --checkpoint_dir best_model \
       --data_root /media/M2SSD/OmniCount-191/OmniCount-191 \
       --categories Supermarket \
       --split test \
       --save_visualizations
   ```

3. **Train final model longer with best hyperparameters**:
   ```bash
   # Example: if best config was lr=5e-4, r=16, alpha=32, mlp=3
   python train_vlm_regression.py \
       --data_root /media/M2SSD/OmniCount-191/OmniCount-191 \
       --categories Supermarket \
       --lr 5e-4 \
       --lora_r 16 \
       --lora_alpha 32 \
       --mlp_layers 3 \
       --batch_size 4 \
       --load_in_4bit \
       --epochs 50 \
       --output_dir final_model \
       --wandb_project sequential-counting \
       --wandb_run_name final_model_50epochs
   ```

## Advanced: Bayesian Optimization

For even better hyperparameter search, consider using W&B Sweeps:

```bash
# Create sweep config
cat > sweep_config.yaml << EOF
program: train_vlm_regression.py
method: bayes
metric:
  name: val/loss
  goal: minimize
parameters:
  lr:
    min: 0.00001
    max: 0.001
    distribution: log_uniform
  lora_r:
    values: [8, 16, 32, 64]
  lora_alpha:
    values: [16, 32, 64]
  mlp_layers:
    values: [2, 3, 4]
  batch_size:
    values: [2, 4, 8]
EOF

# Initialize sweep
wandb sweep sweep_config.yaml

# Run agent on SLURM
# (modify run_hparam_search.sh to use: wandb agent <sweep-id>)
```

## Contact

For issues or questions about the hyperparameter search, check:
- W&B dashboard: https://wandb.ai
- SLURM documentation: `man sbatch`
- Project README: `README.md`
