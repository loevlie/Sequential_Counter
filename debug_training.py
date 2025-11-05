#!/usr/bin/env python3
"""
Debug script to identify why training fails.
Tests the full pipeline with a single batch.
"""

import torch
import numpy as np
from PIL import Image
from model_sequential_attention import SequentialAttentionCountingModel
from dataset_fsc147 import FSC147Dataset
from utils import VisualMarker

print("=" * 60)
print("Training Debug Script")
print("=" * 60)

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")

# Load a tiny dataset
print("\nLoading FSC147 dataset...")
try:
    dataset = FSC147Dataset(
        dataset_root='/media/M2SSD/FSC147',
        split='train',
        spatial_order='reading_order',
        min_objects=5,
        max_objects=20
    )
    print(f"Dataset loaded: {len(dataset)} images")
except Exception as e:
    print(f"ERROR: Could not load dataset: {e}")
    print("Please update the dataset path in this script")
    exit(1)

# Get one sample
print("\nLoading sample image...")
img, points_list, meta = dataset[0]
print(f"  Object type: {meta['object_type']}")
print(f"  Num objects: {len(points_list)}")
print(f"  Image size: {img.size}")
print(f"  First 3 points: {points_list[:3]}")

# Create marker
marker = VisualMarker(strategy='numbers', alpha=0.3)

# Test coordinate normalization
print("\n" + "=" * 60)
print("Testing Coordinate Normalization")
print("=" * 60)

W, H = img.size
for i, (x, y) in enumerate(points_list[:3]):
    x_norm = (x / W) * 2 - 1
    y_norm = (y / H) * 2 - 1
    print(f"Point {i}: pixel=({x:.1f}, {y:.1f}) -> normalized=({x_norm:.3f}, {y_norm:.3f})")

    # Check if in valid range
    if x_norm < -1 or x_norm > 1 or y_norm < -1 or y_norm > 1:
        print(f"  WARNING: Normalized coordinates out of [-1, 1] range!")

# Load model
print("\n" + "=" * 60)
print("Loading Model")
print("=" * 60)

model = SequentialAttentionCountingModel(
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    load_in_4bit=True,
    device=device,
    num_foveal_steps=2,  # Reduce for faster testing
    num_reasoning_steps=2
)
print("Model loaded successfully")

# Prepare a simple batch (k=0, predict first point)
print("\n" + "=" * 60)
print("Testing Forward Pass (k=0, predict first point)")
print("=" * 60)

k = 0  # No objects marked yet
N = len(points_list)
category = meta['object_type']

# Ground truth for first point
gt_x_pixel, gt_y_pixel = points_list[0]
gt_x_norm = (gt_x_pixel / W) * 2 - 1
gt_y_norm = (gt_y_pixel / H) * 2 - 1

print(f"Ground truth:")
print(f"  Pixel: ({gt_x_pixel:.1f}, {gt_y_pixel:.1f})")
print(f"  Normalized: ({gt_x_norm:.3f}, {gt_y_norm:.3f})")
print(f"  Image size: {W}x{H}")

# Create tensors
gt_x = torch.tensor([gt_x_norm], dtype=torch.bfloat16, device=device)
gt_y = torch.tensor([gt_y_norm], dtype=torch.bfloat16, device=device)
gt_done = torch.tensor([0.0], dtype=torch.bfloat16, device=device)

# Reset memory
model.reset_memory(batch_size=1)
model.train()

print("\nRunning forward pass...")
outputs = model.forward_with_attention(
    images=img,
    num_marked=k,
    category=category,
    gt_x=gt_x,
    gt_y=gt_y,
    gt_done=gt_done
)

print("\nOutput:")
print(f"  pred_x: {outputs['x'].item():.3f} (target: {gt_x_norm:.3f})")
print(f"  pred_y: {outputs['y'].item():.3f} (target: {gt_y_norm:.3f})")
print(f"  pred_done: {outputs['done'].item():.3f} (target: 0.0)")
print(f"  count_estimate: {outputs['count_estimate'].item():.1f}")
print(f"\nLosses:")
print(f"  loss_x: {outputs['loss_x'].item():.4f}")
print(f"  loss_y: {outputs['loss_y'].item():.4f}")
print(f"  loss_done: {outputs['loss_done'].item():.4f}")
print(f"  total_loss: {outputs['loss'].item():.4f}")

# Check if prediction is stuck at -1, -1
if outputs['x'].item() < -0.9 and outputs['y'].item() < -0.9:
    print("\n⚠️  WARNING: Prediction stuck at top-left corner (-1, -1)!")
    print("This indicates the model is not learning properly.")

# Test backward pass
print("\n" + "=" * 60)
print("Testing Backward Pass")
print("=" * 60)

optimizer = torch.optim.AdamW([
    {'params': model.model.parameters(), 'lr': 1e-4},
    {'params': model.x_head.parameters(), 'lr': 1e-3},
    {'params': model.y_head.parameters(), 'lr': 1e-3},
    {'params': model.done_head.parameters(), 'lr': 1e-3},
    {'params': model.working_memory.parameters(), 'lr': 1e-3},
    {'params': model.foveation.parameters(), 'lr': 1e-3},
    {'params': model.object_attention.parameters(), 'lr': 1e-3},
    {'params': model.sequential_reasoning.parameters(), 'lr': 1e-3}
])

optimizer.zero_grad()
loss = outputs['loss']
print(f"Loss before backward: {loss.item():.4f}")

print("Running backward pass...")
loss.backward()

# Check gradients
print("\nGradient magnitudes:")
x_head_grad = sum(p.grad.abs().mean().item() for p in model.x_head.parameters() if p.grad is not None)
y_head_grad = sum(p.grad.abs().mean().item() for p in model.y_head.parameters() if p.grad is not None)
done_head_grad = sum(p.grad.abs().mean().item() for p in model.done_head.parameters() if p.grad is not None)
working_memory_grad = sum(p.grad.abs().mean().item() for p in model.working_memory.parameters() if p.grad is not None)
foveation_grad = sum(p.grad.abs().mean().item() for p in model.foveation.parameters() if p.grad is not None)

print(f"  x_head: {x_head_grad:.6f}")
print(f"  y_head: {y_head_grad:.6f}")
print(f"  done_head: {done_head_grad:.6f}")
print(f"  working_memory: {working_memory_grad:.6f}")
print(f"  foveation: {foveation_grad:.6f}")

if x_head_grad < 1e-6 or y_head_grad < 1e-6:
    print("\n⚠️  WARNING: Very small gradients detected!")
    print("This indicates vanishing gradients or disconnected computation graph.")

# Check VLM gradients
vlm_grad_count = 0
vlm_grad_sum = 0
for p in model.model.parameters():
    if p.grad is not None:
        vlm_grad_count += 1
        vlm_grad_sum += p.grad.abs().mean().item()

if vlm_grad_count > 0:
    print(f"  VLM (avg): {vlm_grad_sum / vlm_grad_count:.6f}")
else:
    print("  ⚠️  WARNING: No gradients in VLM!")

print("\nApplying gradients...")
optimizer.step()

# Run another forward pass to see if loss decreased
print("\n" + "=" * 60)
print("Testing After One Gradient Step")
print("=" * 60)

model.reset_memory(batch_size=1)
with torch.no_grad():
    outputs2 = model.forward_with_attention(
        images=img,
        num_marked=k,
        category=category,
        gt_x=gt_x,
        gt_y=gt_y,
        gt_done=gt_done
    )

print(f"\nBefore gradient step:")
print(f"  pred_x: {outputs['x'].item():.3f}, pred_y: {outputs['y'].item():.3f}")
print(f"  loss: {outputs['loss'].item():.4f}")

print(f"\nAfter gradient step:")
print(f"  pred_x: {outputs2['x'].item():.3f}, pred_y: {outputs2['y'].item():.3f}")
print(f"  loss: {outputs2['loss'].item():.4f}")

loss_change = outputs2['loss'].item() - outputs['loss'].item()
if loss_change < 0:
    print(f"\n✓ Loss DECREASED by {abs(loss_change):.4f} (good!)")
elif loss_change > 0:
    print(f"\n⚠️  Loss INCREASED by {loss_change:.4f} (bad!)")
    print("This indicates training instability or learning rate too high")
else:
    print(f"\n⚠️  Loss unchanged (concerning)")

# Test with k=N-1 (almost done)
print("\n" + "=" * 60)
print("Testing Classification Mode (k=N-1, should predict done)")
print("=" * 60)

# Mark all but last object
k = N - 1
img_marked = np.array(img)
marked_pts = points_list[:k]
img_marked = marker.mark_image(img_marked, marked_pts)
img_marked_pil = Image.fromarray(img_marked)

# Target: last point
gt_x_pixel, gt_y_pixel = points_list[k]
gt_x_norm = (gt_x_pixel / W) * 2 - 1
gt_y_norm = (gt_y_pixel / H) * 2 - 1

gt_x = torch.tensor([gt_x_norm], dtype=torch.bfloat16, device=device)
gt_y = torch.tensor([gt_y_norm], dtype=torch.bfloat16, device=device)
gt_done = torch.tensor([0.0], dtype=torch.bfloat16, device=device)  # Not done yet

model.reset_memory(batch_size=1)
with torch.no_grad():
    outputs_almost_done = model.forward_with_attention(
        images=img_marked_pil,
        num_marked=k,
        category=category,
        gt_x=gt_x,
        gt_y=gt_y,
        gt_done=gt_done
    )

print(f"\nWith {k}/{N} objects marked:")
print(f"  pred_done: {outputs_almost_done['done'].item():.3f} (should be low, close to 0)")
print(f"  count_estimate: {outputs_almost_done['count_estimate'].item():.1f} (should be ~{k})")

# Test with k=N (all marked, should be done)
print("\n" + "=" * 60)
print("Testing Done Signal (k=N, all objects marked)")
print("=" * 60)

k = N
img_marked = np.array(img)
marked_pts = points_list[:k]
img_marked = marker.mark_image(img_marked, marked_pts)
img_marked_pil = Image.fromarray(img_marked)

gt_done = torch.tensor([1.0], dtype=torch.bfloat16, device=device)  # Done!

model.reset_memory(batch_size=1)
with torch.no_grad():
    outputs_done = model.forward_with_attention(
        images=img_marked_pil,
        num_marked=k,
        category=category,
        gt_done=gt_done
    )

print(f"\nWith {k}/{N} objects marked (all done):")
print(f"  pred_done: {outputs_done['done'].item():.3f} (should be high, close to 1)")
print(f"  count_estimate: {outputs_done['count_estimate'].item():.1f} (should be ~{N})")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

issues = []

if outputs['x'].item() < -0.9 and outputs['y'].item() < -0.9:
    issues.append("❌ Predictions stuck at top-left corner")
else:
    print("✓ Predictions not stuck at corner")

if x_head_grad < 1e-6 or y_head_grad < 1e-6:
    issues.append("❌ Vanishing gradients in prediction heads")
else:
    print("✓ Gradients flowing to prediction heads")

if loss_change > 0:
    issues.append("❌ Loss increased after gradient step")
else:
    print("✓ Loss decreased or stable after gradient step")

if outputs_done['done'].item() < 0.5:
    issues.append("❌ Done signal not working (should be high when all marked)")
else:
    print("✓ Done signal working")

if issues:
    print("\nISSUES FOUND:")
    for issue in issues:
        print(f"  {issue}")
    print("\nRECOMMENDATIONS:")
    print("1. Remove tanh activation from coordinate heads")
    print("2. Simplify model architecture (remove some sequential modules)")
    print("3. Lower learning rates")
    print("4. Check that VLM features contain spatial information")
else:
    print("\n✓ All checks passed!")
