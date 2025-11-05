#!/usr/bin/env python3
"""
Quick test script for Sequential Attention Counting Model.

Tests that all components are properly initialized and can forward pass.
"""

import torch
from PIL import Image
import numpy as np

print("=" * 60)
print("Testing Sequential Attention Counting Model")
print("=" * 60)

# Test 1: Import model
print("\n[Test 1] Importing model...")
try:
    from model_sequential_attention import SequentialAttentionCountingModel
    print("✓ Model imported successfully")
except Exception as e:
    print(f"✗ Failed to import model: {e}")
    exit(1)

# Test 2: Initialize model components
print("\n[Test 2] Testing individual components...")
try:
    from model_sequential_attention import (
        WorkingMemoryModule,
        SpatialFoveationModule,
        ObjectCrossAttention,
        SequentialReasoningModule
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_dim = 4096
    batch_size = 2

    # Working Memory
    print("  Testing WorkingMemoryModule...")
    working_memory = WorkingMemoryModule(hidden_dim).to(device).to(torch.float32)
    test_features = torch.randn(batch_size, hidden_dim, device=device, dtype=torch.float32)
    memory_state, hidden_state, count_est = working_memory(test_features)
    assert memory_state.shape == (batch_size, hidden_dim), f"Wrong shape: {memory_state.shape}"
    assert count_est.shape == (batch_size, 1), f"Wrong count shape: {count_est.shape}"
    print("    ✓ WorkingMemoryModule passed")

    # Foveation
    print("  Testing SpatialFoveationModule...")
    foveation = SpatialFoveationModule(hidden_dim, num_foveal_steps=4).to(device).to(torch.float32)
    spatial_features = torch.randn(batch_size, hidden_dim, 8, 8, device=device, dtype=torch.float32)
    foveated, attention_maps = foveation(test_features, memory_state, spatial_features)
    assert foveated.shape == (batch_size, hidden_dim), f"Wrong shape: {foveated.shape}"
    assert len(attention_maps) == 4, f"Wrong number of attention maps: {len(attention_maps)}"
    print("    ✓ SpatialFoveationModule passed")

    # Object Attention
    print("  Testing ObjectCrossAttention...")
    object_attn = ObjectCrossAttention(hidden_dim, num_heads=8).to(device).to(torch.float32)
    memory_buffer = torch.randn(batch_size, 5, hidden_dim, device=device, dtype=torch.float32)
    attended = object_attn(test_features, memory_buffer)
    assert attended.shape == (batch_size, hidden_dim), f"Wrong shape: {attended.shape}"
    print("    ✓ ObjectCrossAttention passed")

    # Sequential Reasoning
    print("  Testing SequentialReasoningModule...")
    reasoning = SequentialReasoningModule(hidden_dim, num_reasoning_steps=3).to(device).to(torch.float32)
    reasoned = reasoning(test_features)
    assert reasoned.shape == (batch_size, hidden_dim), f"Wrong shape: {reasoned.shape}"
    print("    ✓ SequentialReasoningModule passed")

    print("✓ All components passed")

except Exception as e:
    print(f"✗ Component test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Initialize full model (without loading weights)
print("\n[Test 3] Initializing full model (this may take a minute)...")
try:
    model = SequentialAttentionCountingModel(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        load_in_4bit=True,
        device=device,
        num_foveal_steps=2,  # Reduce for faster testing
        num_reasoning_steps=2
    )
    print("✓ Model initialized successfully")

except Exception as e:
    print(f"✗ Failed to initialize model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Create dummy input
print("\n[Test 4] Testing forward pass with dummy input...")
try:
    # Create a dummy image
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    )

    # Reset memory
    model.reset_memory(batch_size=1)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model.forward_with_attention(
            images=dummy_image,
            num_marked=0,
            category="apples",
            return_attention_maps=True
        )

    # Check outputs
    assert 'x' in outputs, "Missing 'x' in outputs"
    assert 'y' in outputs, "Missing 'y' in outputs"
    assert 'done' in outputs, "Missing 'done' in outputs"
    assert 'count_estimate' in outputs, "Missing 'count_estimate' in outputs"
    assert 'attention_maps' in outputs, "Missing 'attention_maps' in outputs"

    print(f"  Output keys: {list(outputs.keys())}")
    print(f"  x: {outputs['x'].item():.3f}")
    print(f"  y: {outputs['y'].item():.3f}")
    print(f"  done: {outputs['done'].item():.3f}")
    print(f"  count_estimate: {outputs['count_estimate'].item():.1f}")
    print(f"  attention_maps: {len(outputs['attention_maps'])} maps")
    print("✓ Forward pass successful")

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Test with multiple images in batch
print("\n[Test 5] Testing batch processing...")
try:
    dummy_images = [
        Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))
        for _ in range(2)
    ]

    model.reset_memory(batch_size=2)

    with torch.no_grad():
        outputs = model.forward_with_attention(
            images=dummy_images,
            num_marked=[0, 3],
            category="bottles"
        )

    assert outputs['x'].shape[0] == 2, f"Wrong batch size: {outputs['x'].shape}"
    print(f"  Batch outputs shape: {outputs['x'].shape}")
    print("✓ Batch processing successful")

except Exception as e:
    print(f"✗ Batch processing failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 6: Test loss computation
print("\n[Test 6] Testing loss computation...")
try:
    model.reset_memory(batch_size=1)

    # Create ground truth
    gt_x = torch.tensor([0.5], dtype=torch.bfloat16, device=device)
    gt_y = torch.tensor([-0.3], dtype=torch.bfloat16, device=device)
    gt_done = torch.tensor([0.0], dtype=torch.bfloat16, device=device)

    with torch.no_grad():
        outputs = model.forward_with_attention(
            images=dummy_image,
            num_marked=2,
            category="objects",
            gt_x=gt_x,
            gt_y=gt_y,
            gt_done=gt_done
        )

    assert 'loss' in outputs, "Missing 'loss' in outputs"
    assert 'loss_x' in outputs, "Missing 'loss_x' in outputs"
    assert 'loss_y' in outputs, "Missing 'loss_y' in outputs"
    assert 'loss_done' in outputs, "Missing 'loss_done' in outputs"

    print(f"  loss: {outputs['loss'].item():.4f}")
    print(f"  loss_x: {outputs['loss_x'].item():.4f}")
    print(f"  loss_y: {outputs['loss_y'].item():.4f}")
    print(f"  loss_done: {outputs['loss_done'].item():.4f}")
    print("✓ Loss computation successful")

except Exception as e:
    print(f"✗ Loss computation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 7: Test memory persistence
print("\n[Test 7] Testing memory persistence across steps...")
try:
    model.reset_memory(batch_size=1)

    # First prediction
    with torch.no_grad():
        outputs1 = model.forward_with_attention(
            images=dummy_image,
            num_marked=0,
            category="objects"
        )

    # Check memory buffer size increased
    assert model.memory_buffer.shape[1] == 0, "Memory should be empty after first step with 0 marked"

    # Second prediction (with 1 marked)
    with torch.no_grad():
        outputs2 = model.forward_with_attention(
            images=dummy_image,
            num_marked=1,
            category="objects"
        )

    # Memory should now have 1 entry
    assert model.memory_buffer.shape[1] == 1, f"Memory should have 1 entry, has {model.memory_buffer.shape[1]}"

    print(f"  Memory buffer size after step 1: {model.memory_buffer.shape[1]}")
    print("✓ Memory persistence working correctly")

except Exception as e:
    print(f"✗ Memory persistence test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nYou can now train the model using:")
print("  ./run_sequential_attention.sh")
print("\nOr manually:")
print("  python train_sequential_attention.py --data_root /path/to/FSC147")
