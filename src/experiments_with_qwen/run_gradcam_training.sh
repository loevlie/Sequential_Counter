#!/bin/bash
#SBATCH --job-name=qwen_gradcam
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks=4
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=cc1gpu004
#SBATCH --output=/cluster/tufts/sinapovlab/dloevl01/slurmlog/out/qwen_gradcam_%A.out
#SBATCH --error=/cluster/tufts/sinapovlab/dloevl01/slurmlog/err/qwen_gradcam_%A.err

# Source cache environment variables
source ./set_cache_env.sh

# Activate virtual environment
source venv/bin/activate

# Run training with GradCAM tracking
python train_with_gradcam_fixed.py \
  --data_dir ./FSC147 \
  --gradient_accumulation 12 \
  --epochs 10 \
  --num_track_samples 10 \
  --checkpoint_dir ./checkpoints_gradcam \
  --vis_dir ./gradcam_visualizations \
  --clear_cache_every 5 \
  --attention_weight 0.0

echo "Training complete!"
