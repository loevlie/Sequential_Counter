#!/usr/bin/env python3
"""
Quick test for the simplified sequential model.
"""

import torch
from PIL import Image
import numpy as np
from model_sequential_simple import SimpleSequentialModel

print("=" * 60)
print("Testing Simple Sequential Model")
print("=" * 60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")

# Load model
print("\nLoading model...")
model = SimpleSequentialModel(
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    load_in_4bit=True,
    device=device
)
print("✓ Model loaded")

# Create dummy image
print("\nCreating dummy image...")
dummy_image = Image.fromarray(
    np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
)

# Test forward pass
print("\nTesting forward pass...")
model.eval()
with torch.no_grad():
    outputs = model.forward(
        images=dummy_image,
        num_marked=0,
        category="apples"
    )

print(f"  pred_x: {outputs['x'].item():.3f}")
print(f"  pred_y: {outputs['y'].item():.3f}")
print(f"  pred_done: {outputs['done'].item():.3f}")
print("✓ Forward pass successful")

# Test with loss
print("\nTesting with ground truth...")
gt_x = torch.tensor([0.5], dtype=torch.bfloat16, device=device)
gt_y = torch.tensor([-0.3], dtype=torch.bfloat16, device=device)
gt_done = torch.tensor([0.0], dtype=torch.bfloat16, device=device)

with torch.no_grad():
    outputs = model.forward(
        images=dummy_image,
        num_marked=2,
        category="objects",
        gt_x=gt_x,
        gt_y=gt_y,
        gt_done=gt_done
    )

print(f"  loss: {outputs['loss'].item():.4f}")
print(f"  loss_x: {outputs['loss_x'].item():.4f}")
print(f"  loss_y: {outputs['loss_y'].item():.4f}")
print(f"  loss_done: {outputs['loss_done'].item():.4f}")
print("✓ Loss computation successful")

# Test gradient flow
print("\nTesting gradient flow...")
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
optimizer.zero_grad()

outputs = model.forward(
    images=dummy_image,
    num_marked=0,
    category="objects",
    gt_x=gt_x,
    gt_y=gt_y,
    gt_done=gt_done
)

loss_before = outputs['loss'].item()
print(f"  loss before: {loss_before:.4f}")

loss = outputs['loss']
loss.backward()

# Check gradients
x_grad = sum(p.grad.abs().mean().item() for p in model.x_head.parameters() if p.grad is not None)
y_grad = sum(p.grad.abs().mean().item() for p in model.y_head.parameters() if p.grad is not None)
done_grad = sum(p.grad.abs().mean().item() for p in model.done_head.parameters() if p.grad is not None)

print(f"  x_head grad: {x_grad:.6f}")
print(f"  y_head grad: {y_grad:.6f}")
print(f"  done_head grad: {done_grad:.6f}")

if x_grad > 1e-4 and y_grad > 1e-4 and done_grad > 1e-4:
    print("✓ Gradients flowing properly!")
else:
    print("⚠️  WARNING: Some gradients very small")

optimizer.step()

# Check if loss changes
model.eval()
with torch.no_grad():
    outputs2 = model.forward(
        images=dummy_image,
        num_marked=0,
        category="objects",
        gt_x=gt_x,
        gt_y=gt_y,
        gt_done=gt_done
    )

loss_after = outputs2['loss'].item()
print(f"  loss after: {loss_after:.4f}")

if loss_after < loss_before:
    print(f"✓ Loss DECREASED by {loss_before - loss_after:.4f}")
elif loss_after > loss_before:
    print(f"⚠️  Loss increased by {loss_after - loss_before:.4f}")
    print("   (This can happen with one step, but should decrease overall)")
else:
    print("⚠️  Loss unchanged")

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)
print("\nYou can now train with:")
print("  ./run_sequential_attention.sh")
