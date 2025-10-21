# Sequential Counting with Visual Marking

Training code for sequential object counting inspired by how children learn to count.

## Two Approaches Available

This repository provides two architectures for sequential counting:

### 🌟 **VLM Approach (Recommended)** - LLaVA-CoT
- **Model**: Llama-3.2-11B-Vision-Instruct with LoRA fine-tuning
- **Parameters**: 11B base + ~16-32M trainable (LoRA)
- **Training**: Efficient with 4-bit quantization + LoRA
- **Advantages**: Superior spatial reasoning, explicitly trained on counting (CLEVR dataset)
- **Files**: `train_vlm.py`, `model_vlm.py`, `test_vlm_training.py`

### 📊 **Cross-Attention Baseline**
- **Model**: CLIP ViT-B/32 (frozen) + custom point predictor
- **Parameters**: ~890K trainable
- **Training**: Faster, less memory-intensive
- **Advantages**: Lightweight, quick iterations
- **Files**: `train.py`, `model_cross_attn.py`, `test_training.py`

Both use the same **random prefix training strategy** with **spatial ordering**.

---

## Quick Start (VLM - Recommended)

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies (includes VLM support)
pip install -r requirements.txt
```

### 2. Download Dataset

Download OmniCount-191 from: https://github.com/mondalanindya/OmniCount

```bash
git clone https://github.com/mondalanindya/OmniCount.git
cd OmniCount
unzip OmniCount-191.zip
```

### 3. Test VLM Training

```bash
python test_vlm_training.py
```

This runs 2 epochs to verify:
- Model loading works (Llama-3.2-Vision with 4-bit quantization)
- LoRA setup is correct
- Training loop runs without errors
- Expected time: ~30-60 minutes
- GPU memory: ~16-24GB

### 4. Full VLM Training

```bash
python train_vlm.py \
  --data_root /path/to/OmniCount-191 \
  --categories Supermarket Fruits Urban \
  --batch_size 4 \
  --epochs 50 \
  --lr 1e-5 \
  --lora_r 16 \
  --lora_alpha 32 \
  --output_dir outputs/vlm_run_001
```

---

## HPC Hyperparameter Sweep

### VLM Sweep (216 configurations)

**Recommended for best performance**

1. **Generate configs:**
```bash
python sweep_vlm.py --generate
```

2. **Edit SLURM script:**

Edit `run_sweep_vlm.sbatch`:
- Set `DATA_ROOT` to your OmniCount-191 path
- Adjust partition/resources for your cluster
- Requires A100 40GB+ GPUs (or 24GB with 4-bit quantization)

3. **Submit:**
```bash
mkdir -p logs
sbatch run_sweep_vlm.sbatch  # 216 jobs, 6 parallel
```

4. **Analyze results:**
```bash
python analyze_sweep.py --sweep_dir sweep_vlm_outputs
```

### Cross-Attention Sweep (216 configurations)

**Faster iterations, good baseline**

1. **Generate configs:**
```bash
python sweep.py --generate
```

2. **Submit:**
```bash
sbatch run_sweep.sbatch  # 216 jobs, 10 parallel
```

3. **Analyze:**
```bash
python analyze_sweep.py --sweep_dir sweep_outputs
```

---

## Repository Structure

```
sequential-counting/
├── README.md                    # This file
├── HPC_DEPLOYMENT_GUIDE.md      # Detailed HPC deployment guide
├── requirements.txt             # Dependencies (includes VLM support)
│
# Data & utilities
├── dataset.py                   # OmniCount-191 loader (spatial ordering)
├── utils.py                     # Visual marking utilities
│
# VLM approach (recommended)
├── model_vlm.py                 # LLaVA-CoT model with LoRA
├── train_vlm.py                 # VLM training script
├── test_vlm_training.py         # Quick VLM test (2 epochs)
├── sweep_vlm.py                 # VLM hyperparameter sweep (216 configs)
├── run_sweep_vlm.sbatch         # SLURM script for VLM sweep
│
# Cross-attention baseline
├── model_cross_attn.py          # Point prediction network
├── train.py                     # Cross-attention training
├── test_training.py             # Quick baseline test
├── sweep.py                     # Baseline hyperparameter sweep
├── run_sweep.sbatch             # SLURM script for baseline
│
# Analysis
└── analyze_sweep.py             # Find best model from sweep results
```

---

## Model Comparisons

| Feature | VLM (LLaVA-CoT) | Cross-Attention |
|---------|-----------------|-----------------|
| **Parameters** | 11B (16-32M trainable) | 890K trainable |
| **GPU Memory** | 16-24GB | 8-12GB |
| **Training Time** | ~24h per sweep config | ~2-8h per sweep config |
| **Inference** | ~200-500ms | ~50ms |
| **Spatial Reasoning** | Excellent (trained on CLEVR) | Good |
| **Counting Accuracy** | Best | Good |
| **HPC Requirements** | A100 40GB+ recommended | Any modern GPU |

---

## Hyperparameter Grids

### VLM Sweep (216 configs)

| Parameter | Values |
|-----------|--------|
| Learning Rate | 5e-6, 1e-5, 2e-5 (3) |
| LoRA Rank | 8, 16, 32 (3) |
| LoRA Alpha | 16, 32 (2) |
| Batch Size | 2, 4 (2) |
| Marking Alpha | 0.2, 0.3, 0.5 (3) |
| Spatial Order | reading_order, nearest_neighbor (2) |
| Max Tokens | 64, 128 (2) |

**Total**: 3×3×2×2×3×2×2 = 216 configs

### Cross-Attention Sweep (216 configs)

| Parameter | Values |
|-----------|--------|
| Learning Rate | 1e-5, 5e-5, 1e-4 (3) |
| Hidden Dim | 128, 256, 512 (3) |
| Batch Size | 8, 16 (2) |
| Coord Weight | 1.0, 2.0 (2) |
| Done Weight | 0.5, 1.0 (2) |
| Marking Alpha | 0.2, 0.3, 0.5 (3) |
| Spatial Order | reading_order, nearest_neighbor (2) |

**Total**: 3×3×2×2×2×3×2 = 216 configs

---

## Training Features

✅ **Random Prefix Training**: Mark random k ∈ [0, N] objects, predict k+1 (unbiased)
✅ **Spatial Ordering**: Reading order or nearest neighbor
✅ **Early Stopping**: Stops when val loss doesn't improve (patience=7)
✅ **LR Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
✅ **CSV Logging**: Saves metrics every epoch
✅ **Checkpointing**: Best, latest, and per-epoch checkpoints
✅ **Mixed Precision**: bfloat16 for VLM, float32 for baseline
✅ **LoRA Fine-tuning**: Efficient training for VLM (4-bit quantization)

---

## Output Structure

### VLM Training
```
outputs/vlm_run_001/
├── checkpoint_best_lora/          # Best LoRA adapters
├── checkpoint_latest_lora/         # Latest LoRA adapters
├── checkpoint_epoch_N_lora/        # Per-epoch LoRA adapters
├── checkpoint_best.pt              # Training state (optimizer, etc.)
├── checkpoint_latest.pt
├── metrics.csv                     # Training metrics
└── args.json                       # Hyperparameters
```

### Cross-Attention Training
```
outputs/run_001/
├── checkpoint_best.pt              # Best model weights
├── checkpoint_latest.pt            # Latest checkpoint
├── checkpoint_epoch_N.pt           # Per-epoch checkpoints
├── metrics.csv                     # Training metrics
└── args.json                       # Hyperparameters
```

---

## Metrics Tracked

Saved to `metrics.csv` every epoch:

**VLM**: `epoch`, `train_loss`, `val_loss`, `learning_rate`, `timestamp`

**Cross-Attention**: `epoch`, `train_total_loss`, `train_coord_loss`, `train_done_loss`, `train_consistency_loss`, `val_total_loss`, `val_coord_loss`, `val_done_loss`, `val_consistency_loss`, `learning_rate`, `timestamp`

---

## Why VLM Approach?

The VLM (LLaVA-CoT) approach is recommended because:

1. **Explicitly Trained on Counting**: LLaVA-CoT uses CLEVR dataset for object counting and spatial relationships
2. **Superior Spatial Reasoning**: Built-in understanding of object positions and relationships
3. **Better Generalization**: Pretrained on diverse visual tasks
4. **Research Potential**: Can provide explanations and reasoning (future work)
5. **SOTA Performance**: Outperforms Gemini-1.5-pro and GPT-4o-mini on multimodal reasoning

The cross-attention baseline remains useful for:
- Quick iterations and ablation studies
- Resource-constrained environments
- Comparison baseline

---

## Citation

If you use this code, please cite:

```bibtex
@article{omnicount2025,
  title={OmniCount: Multi-label Object Counting with Semantic-Geometric Priors},
  author={Mondal, Anindya and Nag, Sauradip and Zhu, Xiatian and Dutta, Anjan},
  journal={AAAI},
  year={2025}
}

@article{llava-cot2024,
  title={LLaVA-CoT: Let Vision Language Models Reason Step-by-Step},
  author={Xu, Guowei and others},
  journal={arXiv preprint arXiv:2411.10440},
  year={2024}
}
```

---

## License

MIT License - See LICENSE file for details
