# VLM Sequential Counting - Complete Workflow Guide

This guide walks you through training, saving, and evaluating a Vision-Language Model (VLM) for sequential object counting.

## Overview

The VLM approach uses **Qwen3-VL-4B-Thinking** with LoRA fine-tuning to:
1. Look at an image (possibly with some objects already marked)
2. Predict the location of the next unmarked object as `(x, y)` coordinates
3. Signal "done" when all objects have been counted
4. This process repeats sequentially until complete

## Step-by-Step Workflow

### 1. Train the Model

#### Quick Test Training (Recommended to start)
```bash
python test_vlm_training.py
```

This will:
- Prompt you for the dataset path (e.g., `/media/M2SSD/OmniCount-191/OmniCount-191`)
- Train for 2 epochs on Supermarket category
- Use batch size 2, optimized for the 4B model
- Save checkpoints to `test_vlm_run/`
- Expected time: ~20-40 minutes
- GPU memory: ~12-16GB

#### Full Training (Custom settings)
```bash
python train_vlm.py \
    --data_root /path/to/OmniCount-191 \
    --categories Supermarket Fruits Electronics \
    --batch_size 4 \
    --epochs 10 \
    --lr 1e-5 \
    --lora_r 16 \
    --lora_alpha 32 \
    --marking_alpha 0.3 \
    --spatial_order reading_order \
    --early_stopping_patience 10 \
    --output_dir my_vlm_model \
    --load_in_4bit
```

**Key Arguments:**
- `--categories`: Object categories to train on
- `--batch_size`: Larger = faster but more memory
- `--epochs`: Number of training epochs
- `--lr`: Learning rate (1e-5 is a good default for LoRA)
- `--lora_r`: LoRA rank (higher = more parameters, better fit but slower)
- `--lora_alpha`: LoRA alpha (usually 2x rank)
- `--marking_alpha`: Visual marker transparency
- `--spatial_order`: How to order points (reading_order, left_to_right, nearest_neighbor)
- `--load_in_4bit`: Use 4-bit quantization (recommended for memory efficiency)

#### Training Outputs

The training script saves:
- `{output_dir}/best_checkpoint_lora/`: Best model LoRA adapters (use this for inference!)
- `{output_dir}/best_checkpoint.pt`: Training state (optimizer, scheduler, etc.)
- `{output_dir}/latest_checkpoint_lora/`: Latest model LoRA adapters
- `{output_dir}/metrics.csv`: Training/validation metrics over time

### 2. Evaluate the Model

#### A. Full Dataset Evaluation

Evaluate on validation set with metrics and visualizations:

```bash
python evaluate_vlm.py \
    --checkpoint_dir test_vlm_run/best_checkpoint_lora \
    --data_root /path/to/OmniCount-191 \
    --categories Supermarket \
    --split val \
    --num_samples 50 \
    --temperature 0.3 \
    --output_dir evaluation_results \
    --save_visualizations \
    --load_in_4bit
```

**What it does:**
1. Loads the trained model from checkpoint
2. For each image:
   - Starts with unmarked image
   - Sequentially predicts object locations
   - Marks each predicted point on the image
   - Continues until model outputs "done" or hits max_objects limit
3. Calculates metrics (MAE, RMSE, etc.)
4. Saves visualizations showing predictions

**Outputs:**
- `evaluation_results/evaluation_results.json`: Detailed metrics
- `evaluation_results/visualizations/`: Marked images for each sample

**Key Arguments:**
- `--checkpoint_dir`: Path to saved LoRA adapters
- `--num_samples`: How many images to evaluate
- `--temperature`: Lower (0.1-0.3) = more deterministic, Higher (0.7-1.0) = more random
- `--save_visualizations`: Save marked images

#### B. Single Image Inference

Test on a single image with detailed visualization:

```bash
python inference_vlm.py \
    --checkpoint_dir test_vlm_run/best_checkpoint_lora \
    --image_path /path/to/test/image.jpg \
    --output_dir inference_output \
    --temperature 0.3 \
    --save_steps \
    --load_in_4bit
```

**What it does:**
1. Loads your image
2. Sequentially predicts object locations
3. Creates multiple visualizations (heatmap, numbers, dots)
4. Saves step-by-step images showing the prediction process
5. Saves final count and coordinates

**Outputs:**
- `inference_output/final_heatmap.jpg`: Heatmap visualization
- `inference_output/final_numbers.jpg`: Numbered objects
- `inference_output/final_dots.jpg`: Colored dots
- `inference_output/comparison.jpg`: Before/after comparison
- `inference_output/predictions.txt`: Text file with coordinates
- `inference_output/step_*.jpg`: Step-by-step images (if --save_steps)

**Key Arguments:**
- `--image_path`: Path to your test image
- `--save_steps`: Save intermediate prediction steps (useful for debugging)
- `--temperature`: Sampling temperature
- `--max_objects`: Safety limit (default 500)

### 3. Understanding the Sequential Process

The evaluation works like this:

```
Step 0: [Unmarked Image]
  ↓ Model predicts (x₁, y₁)

Step 1: [Image with 1 red halo at (x₁, y₁)]
  ↓ Model predicts (x₂, y₂)

Step 2: [Image with 2 red halos]
  ↓ Model predicts (x₃, y₃)

...

Step N: [Image with N red halos]
  ↓ Model outputs "done"

Final Count: N objects
```

### 4. Interpreting Results

#### Evaluation Metrics

From `evaluation_results.json`:

```json
{
  "num_samples": 50,
  "count_mae": 2.45,           // Mean Absolute Error (lower is better)
  "count_rmse": 3.12,          // Root Mean Square Error (lower is better)
  "median_error": 2.0,         // Median error (robust to outliers)
  "finished_naturally_pct": 85.0  // % that finished with "done" vs hitting limit
}
```

**Good Performance:**
- MAE < 3: Excellent counting accuracy
- MAE 3-5: Good accuracy
- MAE > 5: Needs improvement
- finished_naturally_pct > 80%: Model learned termination well

#### Inference Output

From `inference_output/predictions.txt`:

```
Total count: 24
Image size: 1920x1080

Predicted points (x, y):
  1. ( 145,  267)
  2. ( 423,  289)
  3. ( 701,  312)
  ...
```

### 5. Tips for Best Results

#### Training Tips
1. **Start small**: Test with 1-2 categories first
2. **Monitor validation loss**: Should decrease steadily
3. **Use early stopping**: Prevents overfitting
4. **Adjust learning rate**: If loss doesn't decrease, try lower LR (1e-6)

#### Inference Tips
1. **Temperature tuning**:
   - Start with 0.3 for deterministic predictions
   - Increase to 0.5-0.7 if predictions are too repetitive
   - Decrease to 0.1 for maximum accuracy

2. **If model predicts too many/few objects**:
   - Check training data quality
   - Try different temperature
   - May need more training epochs

3. **If predictions are off-target**:
   - Model may need more training
   - Check if visual markers are clear enough (adjust marking_alpha)
   - Ensure image sizes match training

### 6. Advanced: Hyperparameter Sweep

Run multiple training configurations:

```bash
python sweep_vlm.py
```

This tests different combinations of:
- Learning rates
- LoRA ranks
- Marking alphas
- Spatial orderings

Results saved to `vlm_sweep_results/`.

### 7. Common Issues & Solutions

#### Training Crashes (OOM)
- Reduce `--batch_size` to 1
- Use `--load_in_4bit`
- Reduce `--lora_r` to 8

#### File name too long errors
- These are automatically skipped now (fixed in dataset.py)
- Training will continue normally

#### Model always predicts "done" immediately
- Training may have failed
- Check training logs for loss values
- May need to train longer or adjust LR

#### Model never stops (predicts > 500 objects)
- Increase `--max_objects` limit
- Check if termination is learned (look at finished_naturally_pct)
- May need more training with proper "done" examples

### 8. File Structure

After training and evaluation:

```
Sequential_Counter/
├── test_vlm_run/                    # Training outputs
│   ├── best_checkpoint_lora/        # ← Use this for inference!
│   ├── best_checkpoint.pt
│   ├── latest_checkpoint_lora/
│   └── metrics.csv
├── evaluation_results/              # Evaluation outputs
│   ├── evaluation_results.json
│   └── visualizations/
│       ├── sample_0000_gt24_pred23.jpg
│       └── ...
└── inference_output/                # Single image inference
    ├── final_heatmap.jpg
    ├── final_numbers.jpg
    ├── comparison.jpg
    └── predictions.txt
```

## Quick Start Example

```bash
# 1. Train (takes ~30 mins)
python test_vlm_training.py
# Enter dataset path when prompted

# 2. Evaluate on validation set
python evaluate_vlm.py \
    --checkpoint_dir test_vlm_run/best_checkpoint_lora \
    --data_root /media/M2SSD/OmniCount-191/OmniCount-191 \
    --num_samples 20 \
    --save_visualizations \
    --load_in_4bit

# 3. Test on your own image
python inference_vlm.py \
    --checkpoint_dir test_vlm_run/best_checkpoint_lora \
    --image_path /path/to/my/image.jpg \
    --save_steps \
    --load_in_4bit
```

## Model Details

- **Base Model**: Qwen3-VL-4B-Thinking
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Trainable Parameters**: ~33M (0.74% of total)
- **Quantization**: 4-bit NF4 (optional, for memory efficiency)
- **Input Format**: Images with visual markers (red halos)
- **Output Format**: `(x, y)` pixel coordinates or `"done"`

## Next Steps

1. **Train your first model** with `test_vlm_training.py`
2. **Evaluate it** with `evaluate_vlm.py`
3. **Try different hyperparameters** if results aren't good enough
4. **Test on custom images** with `inference_vlm.py`
5. **Scale up** to more categories or longer training

Good luck! 🚀
