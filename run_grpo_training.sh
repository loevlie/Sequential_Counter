#!/bin/bash

# GRPO Training Script for VLM Object Counting
# This script launches GRPO fine-tuning with recommended settings

# Default settings
DATA_ROOT="/media/M2SSD/FSC147"
MODEL_NAME="Qwen/Qwen3-VL-2B-Instruct"
OUTPUT_DIR="./grpo_counting_checkpoints"

# Training hyperparameters
LEARNING_RATE=1e-6
BATCH_SIZE=2  # GRPO requires at least 2 generations
NUM_EPOCHS=3
GROUP_SIZE=4

# Data settings (start small for testing)
TRAIN_SAMPLES=100
VAL_SAMPLES=20

# Reward function settings (disable GradCAM to save memory)
GRADCAM_WEIGHT=0.0  # Disabled to save GPU memory
COUNT_WEIGHT=1.0    # Use only count-based reward
GAUSSIAN_SIGMA=20.0
SMOOTHING_SIGMA=5.0

echo "=================================="
echo "GRPO Training for VLM Counting"
echo "=================================="
echo ""
echo "Model: $MODEL_NAME"
echo "Training samples: $TRAIN_SAMPLES"
echo "Validation samples: $VAL_SAMPLES"
echo "Learning rate: $LEARNING_RATE"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo ""
echo "Reward weights: GradCAM=$GRADCAM_WEIGHT, Count=$COUNT_WEIGHT"
echo "Using smoothed GradCAM (smoothing_sigma=$SMOOTHING_SIGMA)"
echo ""
echo "=================================="
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Load secrets from .secret file
if [ -f ".secret" ]; then
    source .secret
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    echo "✅ Loaded HuggingFace token from .secret file"
else
    echo "❌ Error: .secret file not found. Please create one with HF_TOKEN=your_token"
    exit 1
fi

cd src

python grpo_train_counting.py \
    --data_root "$DATA_ROOT" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --group_size $GROUP_SIZE \
    --train_samples $TRAIN_SAMPLES \
    --val_samples $VAL_SAMPLES \
    --gradcam_weight $GRADCAM_WEIGHT \
    --count_weight $COUNT_WEIGHT \
    --gaussian_sigma $GAUSSIAN_SIGMA \
    --smoothing_sigma $SMOOTHING_SIGMA \
    2>&1 | tee "../grpo_training_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=================================="
echo "Training complete!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "=================================="
