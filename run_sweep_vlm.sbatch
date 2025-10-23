#!/bin/bash
#SBATCH --job-name=qwen_count_sweep
#SBATCH --output=logs/sweep_vlm_%A_%a.out
#SBATCH --error=logs/sweep_vlm_%A_%a.err
#SBATCH --array=0-215%8          # 216 configs (3*3*2*2*3*2*2), run 8 at a time
#SBATCH --time=16:00:00          # 16 hours per job (4B model is faster!)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                # Less memory needed for 4B model
#SBATCH --gres=gpu:1             # 1 GPU per job (24GB+ recommended)
#SBATCH --partition=gpu          # Adjust based on your cluster

# Set up environment
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Job Array Index: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Load modules (adjust for your HPC)
module purge
module load python/3.10
module load cuda/12.1  # Qwen3-VL works with CUDA 12+
module load cudnn/8.9

# Activate virtual environment (create it first if needed)
source ~/venvs/counting/bin/activate

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Create logs directory
mkdir -p logs

# Set data root (adjust for your HPC)
DATA_ROOT="/path/to/OmniCount-191"  # CHANGE THIS

# Get command for this config
CONFIG_ID=$SLURM_ARRAY_TASK_ID
CMD=$(python sweep_vlm.py --get_command $CONFIG_ID --data_root $DATA_ROOT)

echo "Running config $CONFIG_ID:"
echo "$CMD"
echo ""

# Run training
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed with exit code $?"
fi

echo "End time: $(date)"
