#!/bin/bash

# Script to start stable training of Qwen3-VL on FSC147

# Training parameters
DATA_DIR="/media/M2SSD/FSC147"
MODEL_NAME="Qwen/Qwen3-VL-2B-Instruct"
IMAGE_SIZE=224
EPOCHS=5
LR=1e-6  # Very low learning rate for stability
WARMUP_STEPS=100
GRADIENT_ACCUMULATION=4
MAX_GRAD_NORM=0.5
ATTENTION_WEIGHT=0.05
FREEZE_VISION_EPOCHS=1
CHECKPOINT_DIR="./checkpoints_stable"

echo "================================================"
echo "Starting STABLE training with gradient fixes"
echo "================================================"
echo "Data directory: $DATA_DIR"
echo "Model: $MODEL_NAME"
echo "Image size: $IMAGE_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR (with warmup)"
echo "Warmup steps: $WARMUP_STEPS"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "Max gradient norm: $MAX_GRAD_NORM"
echo "Attention weight: $ATTENTION_WEIGHT"
echo "Freeze vision for epochs: $FREEZE_VISION_EPOCHS"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "================================================"
echo ""
echo "Key stability improvements:"
echo "✓ Lower learning rate (1e-6)"
echo "✓ Learning rate warmup"
echo "✓ Frozen vision encoder initially"
echo "✓ Aggressive gradient clipping (0.5)"
echo "✓ NaN/Inf detection and recovery"
echo "✓ Loss clamping"
echo "✓ Lower attention weight initially"
echo "✓ Float32 precision for stability"
echo "================================================"

# Create checkpoint directory
mkdir -p $CHECKPOINT_DIR

# Start training with wandb logging
python train_fsc147_stable.py \
    --data_dir $DATA_DIR \
    --model_name $MODEL_NAME \
    --image_size $IMAGE_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --gradient_accumulation $GRADIENT_ACCUMULATION \
    --max_grad_norm $MAX_GRAD_NORM \
    --attention_weight $ATTENTION_WEIGHT \
    --freeze_vision_epochs $FREEZE_VISION_EPOCHS \
    --checkpoint_dir $CHECKPOINT_DIR \
    --use_wandb

echo "Training completed!"