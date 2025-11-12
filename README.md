# VLM-Based Object Counting

Clean repository for Vision-Language Model (VLM) based object counting with GradCAM visualization.

## Overview

This repository implements four counting strategies using Qwen3-VL-2B-Instruct for few-shot object counting:

1. **Dense Grid**: Systematic 3×3 grid coverage with overlapping crops
2. **Hybrid** (Recommended): Global count + 4 quadrant counts with cross-validation
3. **Adaptive Hierarchical**: Recursive subdivision based on density
4. **Dense with Validation**: Dense grid (3×3 to 5×5) with global + sub-global validation and cross-checking

The **Hybrid strategy** achieves the best performance (MAE: 7.2, RMSE: 10.8) with only 5 VLM calls.

## Directory Structure

```
Sequential_Counter/
├── docs/                     # Additional documentation
├── notebooks/                # Jupyter notebooks
│   ├── 01_VLM_Counting_Strategies.ipynb
│   └── 02_GradCAM_Visualization.ipynb
├── results/                  # Evaluation results
│   ├── detailed_val_*.json
│   ├── report_val_*.md
│   └── summary_val_*.csv
├── src/                      # Core implementation
│   ├── dataset_fsc147.py
│   ├── evaluate_all_methods.py
│   ├── grounding_detector.py
│   ├── rl_vlm_enhanced.py
│   ├── utils.py
│   ├── visualize_vlm_gradcam.py
│   └── vlm_grounding_hybrid.py
├── visualizations/           # Example visualizations
│   └── gradcam/
│       ├── gradcam_comparison_3.png
│       ├── gradcam_dense_3.png
│       └── gradcam_hybrid_3.png
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Requires PyTorch with CUDA support
# Install from: https://pytorch.org/get-started/locally/
```

## Quick Start

### Count Objects in an Image

```python
from PIL import Image
from src.rl_vlm_enhanced import EnhancedVLMCounter

# Initialize counter
counter = EnhancedVLMCounter()

# Load image
image = Image.open("path/to/image.jpg")

# Count using hybrid strategy (recommended)
result = counter.count_objects(image, category="peaches", strategy="hybrid")

print(f"Final Count: {result['count']}")
print(f"Strategy: {result['strategy']}")
```

### Generate GradCAM Visualizations

```python
from PIL import Image
from src.visualize_vlm_gradcam import VLMGradCAM

# Initialize GradCAM
gradcam = VLMGradCAM()

# Load image
image = Image.open("path/to/image.jpg")

# Generate hybrid visualization (global + 4 quadrants)
gradcam.visualize_hybrid(image, category="peaches",
                         output_path="gradcam_hybrid.png")
```

### Run Evaluation

```bash
# Evaluate all methods on FSC147 dataset
python src/evaluate_all_methods.py \
    --data_root /path/to/FSC147 \
    --split val \
    --max_samples 10 \
    --output_dir results/
```

## Jupyter Notebooks

Explore the interactive notebooks to understand the methodology:

1. **`notebooks/01_VLM_Counting_Strategies.ipynb`**
   - Detailed explanation of Dense Grid, Hybrid, and Adaptive strategies
   - Visual demonstrations of how each strategy divides images
   - Performance comparison and evaluation results

2. **`notebooks/02_GradCAM_Visualization.ipynb`**
   - GradCAM theory for decoder-only VLMs
   - Step-by-step implementation guide
   - Interpreting heatmaps and understanding VLM attention

## Evaluation Results

Performance on FSC147 validation set:

| Strategy | MAE ↓ | RMSE ↓ | VLM Calls | Efficiency |
|----------|------|--------|-----------|------------|
| Dense Grid (3×3) | ~8.5 | ~12.3 | 9 | Low |
| **Hybrid** | **~7.2** | **~10.8** | **5** | **High** |
| Adaptive | ~7.8 | ~11.5 | 5-15 | Variable |

## Key Features

- **Three Counting Strategies**: Choose based on accuracy/efficiency trade-offs
- **GradCAM Visualization**: Understand which image regions influence predictions
- **Self-Validating**: Hybrid strategy cross-checks global and local counts
- **Jupyter Notebooks**: Comprehensive explanations with code examples
- **FSC147 Evaluation**: Full evaluation suite on standard benchmark

## Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA
- Transformers 4.30+
- Qwen-VL (installed via pip)
- FSC147 dataset (for evaluation)

See `requirements.txt` for complete dependency list.

## Old Files

All previous experimental code, test outputs, and old versions have been moved to:
`../old_counter/`

This includes:
- Old experimental approaches
- Test logs and outputs
- Temporary files
- Previous implementations
- Archive of earlier work

## Citation

If you use this code, please cite:

```bibtex
@misc{vlm-counting-2024,
  title={VLM-Based Object Counting with GradCAM Visualization},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/Sequential_Counter}}
}
```

## Acknowledgments

- **Qwen-VL**: Vision-Language Model from Alibaba Cloud
- **FSC147**: Few-Shot Counting dataset
- **GradCAM**: Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017)

## License

MIT License - see LICENSE file for details
