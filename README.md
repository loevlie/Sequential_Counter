# Sequential Counting with VLM Regression

Vision-Language Model (VLM) based approach for sequential object counting using direct coordinate regression with separate MLP heads.

## 🎯 Current Approach

**Sensorimotor Agent Architecture** (inspired by autonomous driving):
- **VLM**: Qwen3-VL-4B-Thinking with LoRA adapters (~33M trainable params)
- **Special Tokens**: `<x>` and `<y>` tokens in prompt
- **Separate MLP Heads**: Each coordinate has its own regression head
- **Direct Regression**: Predicts normalized coordinates in [-1, 1] with tanh activation
- **Sequential Marking**: Objects marked with numbered labels as they're counted

## 📁 Repository Structure

```
.
├── model_vlm_regression.py      # VLM + MLP regression model
├── train_vlm_regression.py      # Training script with W&B logging
├── dataset.py                   # OmniCount-191 dataset loader
├── utils.py                     # Visual marking utilities
├── requirements.txt             # Python dependencies
├── run_hparam_search.sh         # SLURM hyperparameter search
├── analyze_hparam_results.py    # Analyze sweep results
├── check_hparam_status.sh       # Check SLURM job status
└── docs/                        # Documentation
    ├── HPC_DEPLOYMENT_GUIDE.md
    ├── HPARAM_SEARCH_README.md
    ├── VLM_REGRESSION_APPROACH.md
    └── VLM_WORKFLOW.md
```

## 🚀 Quick Start

### Training Locally

**With FSC-147 dataset:**
```bash
python train_vlm_regression.py \
    --dataset fsc147 \
    --data_root /media/M2SSD/FSC147 \
    --batch_size 4 \
    --epochs 10 \
    --lr 1e-3
```

**With OmniCount-191 dataset:**
```bash
python train_vlm_regression.py \
    --dataset omnicount \
    --data_root /path/to/OmniCount-191 \
    --categories Supermarket \
    --batch_size 4 \
    --epochs 10 \
    --lr 1e-3
```

### Training on HPC with Hyperparameter Search

```bash
sbatch run_hparam_search.sh
```

Searches over:
- Learning rates: [1e-3, 5e-4, 1e-4, 5e-5]
- LoRA ranks: [8, 16, 32]
- LoRA alphas: [16, 32]
- MLP layers: [2, 3, 4]

Total: 72 experiments running in parallel.

## 🏗️ Model Architecture

```
Input Image + Text Prompt with <x> <y> tokens
              ↓
     Qwen3-VL-4B (LoRA)
              ↓
   Hidden States [batch, seq_len, 4096]
              ↓
   ┌──────────┴──────────┐
   ↓                     ↓
Extract <x>         Extract <y>
features            features
   ↓                     ↓
MLP Head (3 layers)  MLP Head (3 layers)
   ↓                     ↓
Tanh → x ∈ [-1,1]   Tanh → y ∈ [-1,1]
```

## 📝 Example Prompt

```
System: You are a vision assistant that locates eggs in images systematically.

User: [IMAGE]
Count the eggs in this image. 5 objects are already marked with numbered labels.

Task: Predict the (x, y) location of the next unmarked egg.

Rules:
1. Output x, y as normalized coordinates in [-1, 1]
2. If all eggs are marked, output x=-1, y=-1
3. Count systematically from top-to-bottom, left-to-right

Next location: <x> <y>
```

## 🔑 Key Features

### Dataset Integration
- Extracts specific object types from COCO annotations (e.g., "eggs", "apples", "bottles")
- Supports multiple OmniCount-191 categories (Supermarket, Fruits, etc.)
- Spatial ordering strategies: reading order, left-to-right, nearest neighbor

### Training Optimizations
- **Separate Learning Rates**: MLP heads use 5x higher LR than VLM
- **No DONE Training**: Only trains on spatial predictions to avoid collapse
- **Limited Iterations**: 200 train / 50 val iterations per epoch for fast feedback
- **Tanh Activation**: Constrains outputs to valid [-1, 1] range
- **Gradient Monitoring**: Checks gradient flow on first iteration

### Visualization
- Side-by-side input/output comparisons
- Numbered markers showing count order
- Full prompt text in W&B captions
- 16 validation images logged per epoch

## 📊 Monitoring with W&B

Tracks:
- Training/validation loss (total, x, y components)
- Learning rates for VLM and MLP heads
- Sample predictions with ground truth
- Full prompt text for debugging

## 🔧 Hyperparameters

**Default values:**
- `--lr`: 1e-3 (VLM), 5e-3 (MLP heads)
- `--batch_size`: 4
- `--lora_r`: 16
- `--lora_alpha`: 32
- `--mlp_layers`: 3
- `--epochs`: 10

## 📚 Documentation

- **[VLM Regression Approach](docs/VLM_REGRESSION_APPROACH.md)**: Detailed architecture explanation
- **[HPC Deployment Guide](docs/HPC_DEPLOYMENT_GUIDE.md)**: SLURM setup and best practices
- **[Hyperparameter Search](docs/HPARAM_SEARCH_README.md)**: Grid search configuration
- **[VLM Workflow](docs/VLM_WORKFLOW.md)**: Training and evaluation workflow

## 🛠️ Requirements

```
torch>=2.0.0,<2.5.0
transformers>=4.40.0
peft>=0.10.0
qwen-vl-utils>=0.0.2
accelerate>=0.27.0
bitsandbytes>=0.43.0
Pillow>=10.0.0
numpy>=1.24.0,<2.0.0
opencv-python>=4.8.0
tqdm>=4.66.0
wandb>=0.16.0
matplotlib>=3.7.0,<3.9.0
```

## 🎓 References

- **Sensorimotor Agent Architecture**: Inspired by autonomous driving VLM approaches
- **OmniCount-191**: Multi-category counting dataset
- **Qwen3-VL**: 4B parameter vision-language model
- **LoRA**: Low-rank adaptation for efficient fine-tuning

## 📝 License

See LICENSE file for details.
