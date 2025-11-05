#!/usr/bin/env python3
"""
Check if the done signal training is balanced.
"""

import torch
import numpy as np
from PIL import Image
from dataset_fsc147 import FSC147Dataset
from utils import VisualMarker

print("=" * 60)
print("Checking Done Signal Class Balance")
print("=" * 60)

# Load dataset
dataset = FSC147Dataset(
    dataset_root='/media/M2SSD/FSC147',
    split='train',
    spatial_order='reading_order',
    min_objects=5,
    max_objects=50
)

print(f"\nDataset: {len(dataset)} images")

# Simulate 100 batches to check done balance
marker = VisualMarker(strategy='numbers', alpha=0.3)

num_done_positive = 0
num_done_negative = 0
num_classification_batches = 0
num_regression_batches = 0

print("\nSimulating batch preparation...")
for i in range(100):
    # Get random image
    idx = np.random.randint(0, len(dataset))
    img, points_list, meta = dataset[idx]
    N = len(points_list)

    # Simulate alternating mode (like training)
    mode = 'classification' if i % 2 == 0 else 'regression'

    if mode == 'classification':
        # Check what k values are chosen
        if N > 0:
            k = N if np.random.random() < 0.5 else max(0, N - 1)
        else:
            k = 0

        done = 1.0 if k >= N else 0.0

        num_classification_batches += 1
        if done == 1.0:
            num_done_positive += 1
        else:
            num_done_negative += 1
    else:
        num_regression_batches += 1

print(f"\nResults after 100 iterations:")
print(f"  Classification batches: {num_classification_batches}")
print(f"  Regression batches: {num_regression_batches}")
print(f"\nDone signal distribution in classification mode:")
print(f"  done=1 (finished): {num_done_positive}")
print(f"  done=0 (not done): {num_done_negative}")

if num_done_positive + num_done_negative > 0:
    total = num_done_positive + num_done_negative
    pos_pct = 100 * num_done_positive / total
    neg_pct = 100 * num_done_negative / total
    print(f"\nPercentages:")
    print(f"  done=1: {pos_pct:.1f}%")
    print(f"  done=0: {neg_pct:.1f}%")

    if pos_pct < 40 or pos_pct > 60:
        print(f"\n⚠️  WARNING: Imbalanced! Should be ~50/50")
        print(f"   This explains why done signal doesn't learn properly")
    else:
        print(f"\n✓ Class balance looks good (~50/50)")

# Now check mixed mode (validation)
print("\n" + "=" * 60)
print("Checking MIXED mode (used in validation)")
print("=" * 60)

done_distribution = []
for i in range(100):
    idx = np.random.randint(0, len(dataset))
    img, points_list, meta = dataset[idx]
    N = len(points_list)

    if N > 0:
        k = np.random.randint(0, N + 1)  # Mixed mode
    else:
        k = 0

    done = 1.0 if k >= N else 0.0
    done_distribution.append(done)

done_array = np.array(done_distribution)
num_done = (done_array == 1.0).sum()
num_not_done = (done_array == 0.0).sum()

print(f"\nMixed mode (validation) distribution:")
print(f"  done=1: {num_done} ({100*num_done/len(done_distribution):.1f}%)")
print(f"  done=0: {num_not_done} ({100*num_not_done/len(done_distribution):.1f}%)")

if num_done < 10:
    print(f"\n⚠️  WARNING: Very few done=1 examples in validation!")
    print(f"   Model sees mostly done=0, so it learns to always predict 0")

# Check average number of objects
print("\n" + "=" * 60)
print("Checking Object Count Distribution")
print("=" * 60)

object_counts = []
for i in range(len(dataset)):
    _, points_list, _ = dataset[i]
    object_counts.append(len(points_list))

print(f"\nObject counts in dataset:")
print(f"  Min: {np.min(object_counts)}")
print(f"  Max: {np.max(object_counts)}")
print(f"  Mean: {np.mean(object_counts):.1f}")
print(f"  Median: {np.median(object_counts):.0f}")

print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

avg_objects = np.mean(object_counts)
if avg_objects > 20:
    print("\n1. With high object counts (~{:.0f}), the model sees".format(avg_objects))
    print("   k={N} (done) rarely compared to k<N (not done)")
    print("   ")
    print("   FIX: Increase probability of done=1 in classification mode:")
    print("   Change: k = N if np.random.random() < 0.5")
    print("   To:     k = N if np.random.random() < 0.7  # 70% done")

print("\n2. Add loss weighting to emphasize done signal:")
print("   In model, change:")
print("   loss = loss_x + loss_y + loss_done")
print("   To:")
print("   loss = loss_x + loss_y + 2.0 * loss_done  # 2x weight")

print("\n3. Monitor done predictions during training:")
print("   Track average pred_done separately for done vs not-done cases")

print("\n" + "=" * 60)
