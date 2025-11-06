#!/bin/bash
#
# Training script for Sequential Attention Counting Model
# Implements serial counting with foveation, working memory, and sequential reasoning
#

# FSC147 dataset path - UPDATE THIS
FSC147_PATH="/media/M2SSD/FSC147"

# Training configuration
BATCH_SIZE=6
EPOCHS=100
LR=5e-5  # Much lower to prevent loss explosion
MAX_TRAIN_ITERS=3000  # Fast feedback during training
MAX_VAL_ITERS=500

# Sequential attention hyperparameters
NUM_FOVEAL_STEPS=8      # Number of sequential foveal glimpses per prediction
NUM_REASONING_STEPS=5   # Number of sequential reasoning steps (thinking time)

# Object count range
MIN_OBJECTS=2
MAX_OBJECTS=200

# Output
OUTPUT_DIR="sequential_attention_model_hinge_loss_second_try_no_nn"
WANDB_PROJECT="sequential-counting"
WANDB_RUN_NAME="seqattn-fsc147-f${NUM_FOVEAL_STEPS}-r${NUM_REASONING_STEPS}"

echo "========================================="
echo "Sequential Attention Counting Training"
echo "========================================="
echo ""
echo "Configuration:"
echo "  Foveal steps: $NUM_FOVEAL_STEPS"
echo "  Reasoning steps: $NUM_REASONING_STEPS"
echo "  Object range: $MIN_OBJECTS-$MAX_OBJECTS"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo ""

python train_sequential_attention.py \
    --data_root "$FSC147_PATH" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --max_train_iters $MAX_TRAIN_ITERS \
    --max_val_iters $MAX_VAL_ITERS \
    --num_foveal_steps $NUM_FOVEAL_STEPS \
    --num_reasoning_steps $NUM_REASONING_STEPS \
    --min_objects $MIN_OBJECTS \
    --max_objects $MAX_OBJECTS \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --load_in_4bit \
    --spatial_order reading_order \
    --use_nearest_neighbor_loss \ 
    --marking_alpha 0.3

echo ""
echo "Training complete! Model saved to: $OUTPUT_DIR"
