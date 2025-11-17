#!/bin/bash

# Script to start OPTIMIZED training of Qwen3-VL on FSC147
# This version includes all memory optimizations while keeping attention loss

# Source cache environment if it exists
if [ -f ./set_cache_env.sh ]; then
    source ./set_cache_env.sh
fi

# Training parameters
DATA_DIR="/media/M2SSD/FSC147"
MODEL_NAME="Qwen/Qwen3-VL-2B-Instruct"
IMAGE_SIZE=224
EPOCHS=5
LR=2e-6  # Slightly higher than stable version but still conservative
GRADIENT_ACCUMULATION=8  # High accumulation for effective larger batch
CHECKPOINT_DIR="./checkpoints_optimized"

echo "================================================"
echo "Starting OPTIMIZED training with full dual loss"
echo "================================================"
echo "Data directory: $DATA_DIR"
echo "Model: $MODEL_NAME"
echo "Image size: $IMAGE_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "================================================"
echo ""
echo "Key optimizations:"
echo "✓ Mixed precision (float16) training"
echo "✓ Gradient checkpointing enabled"
echo "✓ Efficient attention heatmap (14x14)"
echo "✓ Vision encoder frozen initially"
echo "✓ Proper gradient accumulation"
echo "✓ Memory-efficient collation"
echo "✓ Attention loss INCLUDED (0.1 weight)"
echo "✓ Automatic OOM recovery"
echo "================================================"

# Create checkpoint directory
mkdir -p $CHECKPOINT_DIR

# Kill any existing training processes
echo "Checking for existing training processes..."
pkill -f train_fsc147

# Start training with wandb logging
echo "Starting training..."
python train_fsc147_optimized.py \
    --data_dir $DATA_DIR \
    --model_name $MODEL_NAME \
    --image_size $IMAGE_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --gradient_accumulation $GRADIENT_ACCUMULATION \
    --checkpoint_dir $CHECKPOINT_DIR \
    --use_wandb

echo "Training completed!"