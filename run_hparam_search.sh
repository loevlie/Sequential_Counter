#!/bin/bash
#SBATCH --job-name=vlm_counting_hparam_search
#SBATCH --array=0-71%10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/denny-loevlie/Jivko/Sequential_Counter/slurm_logs/out/%A_%a.out
#SBATCH --error=/home/denny-loevlie/Jivko/Sequential_Counter/slurm_logs/err/%A_%a.err

# =============================================================================
# VLM Sequential Counting Hyperparameter Search
# =============================================================================
# This script runs a systematic hyperparameter search for the VLM regression
# counting model. It explores:
# - Learning rates
# - LoRA ranks and alphas
# - MLP architectures
# - Batch sizes
# =============================================================================

# Setup environment
source ~/.bashrc
module load cuda/11.8  # Adjust based on your cluster

# Activate your conda environment
# Adjust the environment name based on your setup
conda activate Sequential_Counter || conda activate base

# Create output directories
mkdir -p /home/denny-loevlie/Jivko/Sequential_Counter/slurm_logs/out
mkdir -p /home/denny-loevlie/Jivko/Sequential_Counter/slurm_logs/err
mkdir -p /home/denny-loevlie/Jivko/Sequential_Counter/hparam_results

# Navigate to project directory
cd /home/denny-loevlie/Jivko/Sequential_Counter

# =============================================================================
# Hyperparameter Grid Definition
# =============================================================================
# We'll search over the most impactful hyperparameters for VLM fine-tuning

# Learning rates - MOST CRITICAL for LoRA fine-tuning stability
learning_rates=(1e-3 5e-4 1e-4 5e-5)

# LoRA rank - affects adapter capacity (higher = more expressive but slower)
lora_ranks=(8 16 32)

# LoRA alpha - scaling factor for LoRA updates (typically 1-2x rank)
lora_alphas=(16 32)

# MLP layers - depth of prediction head (2-4 layers usually sufficient)
mlp_layers=(2 3 4)

# Batch size - USE LARGEST THAT FITS IN MEMORY
# With 4-bit + LoRA, should be able to fit 4-8
batch_size=4  # Fixed, not searched

# =============================================================================
# Build experiment commands
# =============================================================================
experiments=()

for lr in "${learning_rates[@]}"; do
    for lora_r in "${lora_ranks[@]}"; do
        for lora_alpha in "${lora_alphas[@]}"; do
            for mlp in "${mlp_layers[@]}"; do
                # Create descriptive run name
                run_name="lr${lr}_r${lora_r}_alpha${lora_alpha}_mlp${mlp}"

                # Build command
                cmd="python train_vlm_regression.py \
                    --data_root /media/M2SSD/OmniCount-191/OmniCount-191 \
                    --categories Supermarket \
                    --batch_size ${batch_size} \
                    --epochs 10 \
                    --lr ${lr} \
                    --lora_r ${lora_r} \
                    --lora_alpha ${lora_alpha} \
                    --mlp_layers ${mlp} \
                    --load_in_4bit \
                    --output_dir hparam_results/${run_name} \
                    --wandb_project sequential-counting-hparam-search \
                    --wandb_run_name ${run_name}"

                experiments+=("$cmd")
            done
        done
    done
done

# =============================================================================
# Execute experiment for this array task
# =============================================================================
echo "==========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Total experiments: ${#experiments[@]}"
echo "==========================================="
echo ""

# Check if task ID is valid
if [ $SLURM_ARRAY_TASK_ID -ge ${#experiments[@]} ]; then
    echo "ERROR: Task ID $SLURM_ARRAY_TASK_ID exceeds number of experiments ${#experiments[@]}"
    exit 1
fi

# Print experiment configuration
echo "Experiment configuration:"
echo "${experiments[$SLURM_ARRAY_TASK_ID]}"
echo ""
echo "Starting training..."
echo "==========================================="
echo ""

# Run the experiment
eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

# Capture exit code
exit_code=$?

echo ""
echo "==========================================="
echo "Training completed with exit code: $exit_code"
echo "==========================================="

# Deactivate conda environment
conda deactivate

exit $exit_code
