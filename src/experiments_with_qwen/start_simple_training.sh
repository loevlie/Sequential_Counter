#!/bin/bash

# Simple training script for Qwen3-VL on FSC147 with attention loss
# No gradient accumulation for simplicity

echo "======================================"
echo "Starting SIMPLE training with attention loss"
echo "======================================"

# Training parameters
DATA_DIR="/media/M2SSD/FSC147"
MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
IMAGE_SIZE=224
EPOCHS=5
LR=2e-6
BATCH_SIZE=1
CHECKPOINT_DIR="./checkpoints_simple"

echo "Data directory: $DATA_DIR"
echo "Model: $MODEL_NAME"
echo "Image size: $IMAGE_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Batch size: $BATCH_SIZE"
echo "======================================"

# Create checkpoint directory
mkdir -p $CHECKPOINT_DIR

# Run training
python train_fsc147_simple_attention.py \
    --data_dir $DATA_DIR \
    --model_name $MODEL_NAME \
    --image_size $IMAGE_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --checkpoint_dir $CHECKPOINT_DIR \
    --use_wandb

echo "Training completed!"