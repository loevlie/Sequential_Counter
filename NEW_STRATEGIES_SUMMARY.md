# New Dense Counting Strategies - Development Summary

## Overview

This document summarizes the development and evaluation of two new dense counting strategies designed to improve object counting accuracy through better handling of overlapping grid regions.

## Motivation

The user wanted to explore combining dense grid counting with validation mechanisms to prevent double-counting in overlapping regions. The goal was to leverage the VLM's visual reasoning capabilities rather than relying solely on mathematical corrections.

## New Strategies Implemented

### 1. Dense with Validation (`dense_with_validation`)

**Location**: `src/rl_vlm_enhanced.py`, lines 633-739

**Approach**: Multi-phase counting with cross-validation and adaptive fusion

**Phases**:
1. **Global Validation**: Get overall count for the entire image
2. **Sub-global Validation**: 2×2 grid with 10% overlap for mid-level validation
3. **Dense Grid**: Adaptive 3×3 to 5×5 grid based on density estimate
   - Low density (<20): 3×3 grid
   - Medium density (20-50): 4×4 grid
   - High density (>50): 5×5 grid
4. **Aggregation**: Apply density map normalization to dense grid
5. **Cross-validation**: Adaptive weighted fusion based on agreement:
   - Good agreement (std < 15% of mean): Trust dense grid more (60% weight)
   - Moderate agreement (std < 30% of mean): Balanced (50% weight)
   - Poor agreement (std ≥ 30% of mean): Trust global/sub-global more (30% weight)

**Key Features**:
- Adaptive grid size based on estimated density
- Multi-scale validation (global, sub-global, dense)
- Confidence-aware fusion
- Mathematical density map correction for overlaps

**Test Result** (image 194.jpg, 67 peaches):
- Global: 67, Sub-global: 39, Dense: 95
- Final count: 65 (error: -2)

### 2. Dense Explicit Overlap (`dense_explicit_overlap`)

**Location**: `src/rl_vlm_enhanced.py`, lines 450-631

**Approach**: VLM-native overlap detection

**Phases**:
1. **Dense Grid Counting**: 3×3 grid with 25% overlap
   - Get raw counts from all 9 tiles
   - Calculate raw sum (includes double-counts)

2. **Explicit Overlap Detection**: Extract and analyze overlap regions
   - **Horizontal overlaps**: Vertical strips between adjacent columns (6 regions)
   - **Vertical overlaps**: Horizontal strips between adjacent rows (6 regions)
   - For each overlap region:
     ```
     Prompt: "This is a narrow [vertical/horizontal] strip from an overlap
     region between two tiles. Count how many {category} are FULLY or MOSTLY
     visible in this strip. Only count objects that would likely be counted
     in BOTH adjacent tiles."
     ```

3. **Double-Count Subtraction**: Subtract detected double-counts from raw sum
   ```
   Final Count = Raw Sum - Total Double-Counts
   ```

**Key Features**:
- Interpretable: Can inspect each overlap's contribution
- VLM-native: Uses visual reasoning instead of mathematical formulas
- Transparent: Each double-count detection is logged
- Direct: Explicitly addresses boundary double-counting

**Test Result** (image 194.jpg, 67 peaches):
- Raw dense sum: 91
- Horizontal overlaps: 6 regions × 2 objects = 12 double-counts
- Vertical overlaps: 6 regions × 2 objects = 12 double-counts
- Final count: 67 (error: 0) ✓

## Implementation Details

### Grid Overlap Calculation

Both strategies use `_create_overlapping_grid()` for generating overlapping crops:

```python
overlap = 0.25  # 25% overlap
grid_size = 3   # 3×3 grid

cell_width = int(width / (grid_size - overlap * (grid_size - 1)))
cell_height = int(height / (grid_size - overlap * (grid_size - 1)))

step_x = int(cell_width * (1 - overlap))
step_y = int(cell_height * (1 - overlap))

# Generate grid positions
for i in range(grid_size):
    for j in range(grid_size):
        x1 = min(j * step_x, width - cell_width)
        y1 = min(i * step_y, height - cell_height)
        x2 = min(x1 + cell_width, width)
        y2 = min(y1 + cell_height, height)
```

### Overlap Region Extraction (`dense_explicit_overlap`)

The key innovation is calculating exact coordinates of overlap strips:

```python
# Horizontal overlap (between columns j and j+1 in row i)
overlap_width = cell_width - step_x
x1 = j * step_x + cell_width - overlap_width  # Start of overlap
x2 = x1 + overlap_width                        # End of overlap
y1 = i * step_y
y2 = min(y1 + cell_height, height)

# Vertical overlap (between rows i and i+1 in column j)
overlap_height = cell_height - step_y
x1 = j * step_x
x2 = min(x1 + cell_width, width)
y1 = i * step_y + cell_height - overlap_height  # Start of overlap
y2 = y1 + overlap_height                        # End of overlap
```

## Evaluation

### Single Image Test (194.jpg - Peaches)

Ground truth: 67 peaches

| Strategy | Count | Error | Notes |
|----------|-------|-------|-------|
| **dense_explicit_overlap** | **67** | **0** | Perfect on this image |
| dense_with_validation | 65 | -2 | Very close |
| hybrid (baseline) | TBD | TBD | Evaluating... |
| dense_grid (baseline) | TBD | TBD | Evaluating... |
| adaptive_hierarchical | TBD | TBD | Evaluating... |

### Multi-Image Evaluation (In Progress)

Running comprehensive evaluation on 15 FSC147 validation images:

```bash
python evaluate_new_strategies.py \
    --data_root /media/M2SSD/FSC147 \
    --split val \
    --max_samples 15 \
    --output_dir evaluation_results
```

**Metrics to compare**:
- MAE (Mean Absolute Error) - lower is better
- RMSE (Root Mean Square Error) - lower is better
- Mean Error (shows bias - negative = undercounting)
- Std Error (shows consistency)
- Within 5/10/20% accuracy
- Average processing time per image

Results will be saved to:
- `evaluation_results/new_strategies_summary_val_*.csv`
- `evaluation_results/new_strategies_detailed_val_*.json`
- `evaluation_results/new_strategies_report_val_*.md`

## Key Differences Between Strategies

### Dense with Validation
✓ Multi-scale validation (global → sub-global → dense)
✓ Adaptive grid size based on density
✓ Confidence-aware fusion
✓ Mathematical density map correction
✗ More VLM calls (global + 4 sub-global + 9-25 dense = 14-30 calls)
✗ Less interpretable (complex weighted fusion)

### Dense Explicit Overlap
✓ Interpretable (can see each overlap's contribution)
✓ VLM-native reasoning
✓ Fewer VLM calls (9 dense + 12 overlaps = 21 calls)
✓ Direct approach to the double-counting problem
✗ Assumes objects in overlaps are double-counted (may miss edge cases)
✗ No global validation anchor

## Next Steps

1. **Complete multi-image evaluation** to get statistically significant results
2. **Analyze failure cases** to understand when each strategy works best
3. **Consider hybrid approaches**:
   - Could combine explicit overlap detection with global validation
   - Could use adaptive grid size with explicit overlap
4. **Optimize VLM call count** if needed for efficiency
5. **Test on different object types** (small vs large, sparse vs dense)

## Files Modified

1. `/home/denny-loevlie/Jivko/Sequential_Counter/src/rl_vlm_enhanced.py`
   - Added `_dense_with_validation_counting()` method
   - Added `_dense_explicit_overlap_counting()` method
   - Updated dispatcher and argparse to support new strategies

2. `/home/denny-loevlie/Jivko/Sequential_Counter/src/evaluate_new_strategies.py`
   - Created new evaluation script specifically for comparing strategies

3. `/home/denny-loevlie/Jivko/Sequential_Counter/README.md`
   - Updated to mention 4 strategies instead of 3

## Command Line Usage

### Count objects with new strategies:

```bash
# Dense with validation
python src/rl_vlm_enhanced.py \
    --image /path/to/image.jpg \
    --category "peaches" \
    --strategy dense_with_validation

# Dense explicit overlap
python src/rl_vlm_enhanced.py \
    --image /path/to/image.jpg \
    --category "peaches" \
    --strategy dense_explicit_overlap
```

### Evaluate strategies:

```bash
python src/evaluate_new_strategies.py \
    --data_root /media/M2SSD/FSC147 \
    --split val \
    --max_samples 20 \
    --strategies hybrid dense_grid dense_with_validation dense_explicit_overlap
```

## Conclusion

The `dense_explicit_overlap` strategy shows promise with perfect accuracy on the test image, using a more interpretable approach that asks the VLM to explicitly identify double-counted objects. The `dense_with_validation` strategy provides robust multi-scale validation but is more complex.

Multi-image evaluation results (currently running) will determine which strategy generalizes better across different object types, densities, and image conditions.
