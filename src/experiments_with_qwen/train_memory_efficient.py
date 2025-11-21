#!/usr/bin/env python3
"""
Memory-efficient training script for Qwen3-VL on FSC147.
Prevents OOM errors through aggressive memory management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
import numpy as np
from PIL import Image
import json
import os
from tqdm import tqdm
import gc
import re
import math
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import Dict, List, Tuple
import argparse

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed, logging disabled")


def extract_count_from_text(text: str) -> int:
    """Extract count from generated text."""
    match = re.search(r'COUNT:\s*(\d+)', text)
    if match:
        return int(match.group(1))
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[0])
    return -1


def custom_collate_fn(batch):
    """Custom collate function to handle PIL Images"""
    if len(batch) == 1:
        item = batch[0]
        return {
            'image': [item['image']],
            'count': torch.tensor([item['count']]),
            'target_heatmap': item['target_heatmap'].unsqueeze(0),
            'image_id': [item['image_id']]
        }
    return {
        'image': [item['image'] for item in batch],
        'count': torch.tensor([item['count'] for item in batch]),
        'target_heatmap': torch.stack([item['target_heatmap'] for item in batch]),
        'image_id': [item['image_id'] for item in batch]
    }


def create_gaussian_heatmap(points: np.ndarray, image_shape: Tuple[int, int],
                           sigma: float = 10.0, baseline: float = 0.1) -> np.ndarray:
    """Create target heatmap with Gaussian peaks at object centers."""
    height, width = image_shape
    heatmap = np.full((height, width), baseline, dtype=np.float32)

    if len(points) == 0:
        return heatmap

    y, x = np.ogrid[:height, :width]
    for px, py in points:
        gaussian = np.exp(-((x - px)**2 + (y - py)**2) / (2 * sigma**2))
        heatmap = np.maximum(heatmap, gaussian)

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


class FSC147Dataset(Dataset):
    """Dataset class for FSC147"""
    def __init__(self, root_dir: str, split: str = 'train', image_size: int = 224):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size

        with open(os.path.join(root_dir, 'annotation_FSC147_384.json'), 'r') as f:
            self.annotations = json.load(f)

        with open(os.path.join(root_dir, 'Train_Test_Val_FSC_147.json'), 'r') as f:
            splits = json.load(f)

        self.image_ids = splits[split]
        self.data = {img_id: self.annotations[img_id] for img_id in self.image_ids}
        print(f"Loaded {len(self.data)} images for {split} split")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        ann = self.data[img_id]

        img_path = os.path.join(self.root_dir, 'images_384_VarV2', img_id)
        image = Image.open(img_path).convert('RGB')

        original_size = image.size
        image = image.resize((self.image_size, self.image_size))

        points = np.array(ann['points'])
        if len(points) > 0:
            scale_x = self.image_size / original_size[0]
            scale_y = self.image_size / original_size[1]
            points[:, 0] *= scale_x
            points[:, 1] *= scale_y

        count = len(points)

        # Create target heatmap at lower resolution
        heatmap_size = 14
        target_heatmap = create_gaussian_heatmap(
            points * (heatmap_size / self.image_size),
            (heatmap_size, heatmap_size),
            sigma=1.5,
            baseline=0.1
        )

        return {
            'image': image,
            'count': count,
            'target_heatmap': torch.FloatTensor(target_heatmap),
            'image_id': img_id
        }


class MemoryEfficientQwenTrainer:
    """Memory-efficient trainer with aggressive memory management"""

    def __init__(self, model_name="Qwen/Qwen3-VL-2B-Instruct", device=None,
                 freeze_vision=True, use_bfloat16=False):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model: {model_name}")

        # Choose dtype based on availability and preference
        if use_bfloat16 and torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
            print("Using bfloat16 for better stability than float16")
        else:
            model_dtype = torch.float32
            print("Using float32 for maximum stability")

        self.model_dtype = model_dtype

        # Load model with chosen dtype
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map=self.device,
            torch_dtype=model_dtype
        )

        self.processor = AutoProcessor.from_pretrained(model_name)

        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled for memory efficiency")

        # Freeze vision encoder initially for stability
        if freeze_vision:
            print("Freezing vision encoder for stability")
            for param in self.model.visual.parameters():
                param.requires_grad = False

        # Set to eager attention
        self.model.set_attn_implementation("eager")

        # Set model to training mode
        self.model.train()

        print(f"Model loaded successfully with dtype {model_dtype}")

    def unfreeze_vision(self):
        """Unfreeze vision encoder after initial training"""
        print("Unfreezing vision encoder")
        for param in self.model.visual.parameters():
            param.requires_grad = True

    def prepare_inputs(self, image: Image.Image, count: int):
        """Prepare inputs with proper label masking"""
        question = "Count the objects in this image. Respond with COUNT: followed by a number."
        answer = f"COUNT: {count}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
            padding=True
        )

        inputs.pop("token_type_ids", None)

        # Move to device and ensure correct dtype
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model_dtype)

        # Create labels with proper masking
        labels = inputs["input_ids"].clone()

        # Find where "COUNT:" appears
        decoded = self.processor.decode(inputs["input_ids"][0], skip_special_tokens=False)
        count_pos = decoded.find("COUNT:")

        if count_pos > 0:
            ratio = count_pos / len(decoded)
            mask_until = int(ratio * labels.shape[1] * 0.9)
            labels[:, :mask_until] = -100
        else:
            mask_length = int(labels.shape[1] * 0.75)
            labels[:, :mask_length] = -100

        # Ensure we have tokens to predict
        if (labels != -100).sum() < 5:
            labels = inputs["input_ids"].clone()
            labels[:, :labels.shape[1]//2] = -100

        inputs["labels"] = labels

        return inputs


def train_epoch_memory_efficient(trainer: MemoryEfficientQwenTrainer,
                                dataloader: DataLoader,
                                optimizer: optim.Optimizer,
                                scheduler=None,
                                gradient_accumulation_steps: int = 4,
                                max_grad_norm: float = 0.5,
                                epoch: int = 0,
                                clear_cache_every: int = 10) -> Dict:
    """Memory-efficient training epoch with aggressive cache management"""
    trainer.model.train()

    total_loss = 0
    total_count_loss = 0
    num_valid_batches = 0
    num_oom_batches = 0
    accumulated_loss = 0

    pbar = tqdm(dataloader, desc="Training")

    # CRITICAL: Clear cache and reset optimizer at start
    torch.cuda.empty_cache()
    gc.collect()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        try:
            # CRITICAL: Clear cache periodically BEFORE processing
            if batch_idx > 0 and batch_idx % clear_cache_every == 0:
                torch.cuda.empty_cache()
                gc.collect()
                # Also synchronize to ensure all operations complete
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            image = batch['image'][0]
            count = batch['count'][0].item()

            # Prepare inputs
            inputs = trainer.prepare_inputs(image, count)

            # Forward pass with appropriate precision
            if trainer.model_dtype == torch.bfloat16:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = trainer.model(**inputs, return_dict=True)
            elif trainer.model_dtype == torch.float16:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = trainer.model(**inputs, return_dict=True)
            else:
                outputs = trainer.model(**inputs, return_dict=True)

            # Get loss
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
                loss = torch.clamp(loss, min=0.0, max=100.0)
            else:
                print(f"No loss at batch {batch_idx}")
                continue

            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf loss at batch {batch_idx}")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue

            # Scale for gradient accumulation
            loss = loss / gradient_accumulation_steps
            accumulated_loss += loss.item()

            # Backward pass
            loss.backward()

            # CRITICAL: Delete intermediate variables
            del outputs, loss, inputs

            # Optimizer step every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=max_grad_norm)

                # Check for NaN gradients
                has_nan_grad = False
                for param in trainer.model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            break

                if not has_nan_grad:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                    # Track metrics
                    total_loss += accumulated_loss
                    total_count_loss += accumulated_loss  # Since we don't have attention loss
                    num_valid_batches += 1

                # CRITICAL: Zero gradients to free memory
                optimizer.zero_grad()
                accumulated_loss = 0

                # Force garbage collection after optimizer step
                gc.collect()

            # Update progress bar
            if num_valid_batches > 0:
                avg_loss = total_loss / num_valid_batches
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'valid': num_valid_batches,
                    'oom': num_oom_batches,
                    'mem': f'{torch.cuda.memory_allocated()/1e9:.1f}GB' if torch.cuda.is_available() else 'N/A'
                })

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                num_oom_batches += 1
                print(f"OOM at batch {batch_idx}, clearing cache...")

                # Aggressive memory clearing
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                gc.collect()

                # Synchronize to ensure all operations complete
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # Skip this batch
                continue
            else:
                print(f"Runtime error at batch {batch_idx}: {e}")
                raise e
        except Exception as e:
            print(f"Unexpected error at batch {batch_idx}: {e}")
            optimizer.zero_grad()
            continue

    # Final optimizer step if needed
    if (batch_idx + 1) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Valid batches: {num_valid_batches}/{len(dataloader)}")
    print(f"  OOM batches: {num_oom_batches}")
    if torch.cuda.is_available():
        print(f"  Final memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.max_memory_allocated()/1e9:.2f}GB max")

    return {
        'loss': total_loss / max(num_valid_batches, 1),
        'count_loss': total_count_loss / max(num_valid_batches, 1),
        'oom_batches': num_oom_batches,
        'valid_batches': num_valid_batches
    }


def validate_memory_efficient(trainer: MemoryEfficientQwenTrainer, dataloader: DataLoader) -> Dict:
    """Memory-efficient validation"""
    trainer.model.eval()

    total_mae = 0
    total_mse = 0
    num_samples = 0

    # Clear cache before validation
    torch.cuda.empty_cache()
    gc.collect()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            try:
                # Clear cache periodically
                if batch_idx > 0 and batch_idx % 20 == 0:
                    torch.cuda.empty_cache()

                image = batch['image'][0]
                true_count = batch['count'][0].item()

                question = "Count the objects in this image. Respond with COUNT: followed by a number."
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": question}
                        ]
                    }
                ]

                text = trainer.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = trainer.processor(
                    text=text,
                    images=[image],
                    return_tensors="pt"
                )

                inputs = {k: v.to(trainer.device) for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(trainer.model_dtype)

                # Generate with low temperature
                generated_ids = trainer.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=0.1
                )

                generated_text = trainer.processor.decode(generated_ids[0], skip_special_tokens=True)
                predicted_count = extract_count_from_text(generated_text)

                if predicted_count == -1:
                    predicted_count = 0

                error = abs(predicted_count - true_count)
                total_mae += error
                total_mse += error ** 2
                num_samples += 1

                # Clean up
                del inputs, generated_ids

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM during validation at sample {batch_idx}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
            except Exception as e:
                print(f"Validation error: {e}")
                continue

    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()

    mae = total_mae / max(num_samples, 1)
    rmse = np.sqrt(total_mse / max(num_samples, 1))

    return {'mae': mae, 'rmse': rmse, 'num_samples': num_samples}


def main():
    parser = argparse.ArgumentParser(description='Memory-efficient Qwen3-VL training')
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/FSC147')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-VL-2B-Instruct')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--gradient_accumulation', type=int, default=8)  # Increased for memory
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--freeze_vision_epochs', type=int, default=2)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_memory')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--use_bfloat16', action='store_true')
    parser.add_argument('--clear_cache_every', type=int, default=10)
    parser.add_argument('--debug_subset', type=int, default=0)

    args = parser.parse_args()

    print("=" * 50)
    print("Memory-Efficient Training Configuration")
    print("=" * 50)
    print(f"Model: {args.model_name}")
    print(f"Learning rate: {args.lr}")
    print(f"Gradient accumulation: {args.gradient_accumulation}")
    print(f"Clear cache every: {args.clear_cache_every} batches")
    print(f"Use bfloat16: {args.use_bfloat16}")
    print("=" * 50)

    # Initialize wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="qwen-fsc147-memory",
            name=f"memory_lr{args.lr}_ga{args.gradient_accumulation}",
            config=vars(args)
        )

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize trainer
    trainer = MemoryEfficientQwenTrainer(
        args.model_name,
        freeze_vision=True,
        use_bfloat16=args.use_bfloat16
    )

    # Create datasets
    train_dataset = FSC147Dataset(args.data_dir, split='train', image_size=args.image_size)
    val_dataset = FSC147Dataset(args.data_dir, split='val', image_size=args.image_size)

    # Use debug subset if specified
    if args.debug_subset > 0:
        print(f"Using debug subset of {args.debug_subset} samples")
        train_dataset.image_ids = train_dataset.image_ids[:args.debug_subset]
        val_dataset.image_ids = val_dataset.image_ids[:min(50, args.debug_subset//4)]

    # Create dataloaders with num_workers=0 to avoid memory issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=0,  # Important for memory
        pin_memory=False  # Important for memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0,
        pin_memory=False
    )

    # Optimizer
    optimizer = optim.AdamW(
        [p for p in trainer.model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
        eps=1e-6
    )

    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_steps)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=total_steps - args.warmup_steps, T_mult=1)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_steps])

    # Training loop
    best_mae = float('inf')

    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        if torch.cuda.is_available():
            print(f"Starting memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        print(f"{'='*50}")

        # Unfreeze vision encoder after initial epochs
        if epoch == args.freeze_vision_epochs:
            trainer.unfreeze_vision()
            # Recreate optimizer with all parameters
            optimizer = optim.AdamW(
                trainer.model.parameters(),
                lr=args.lr * 0.5,
                weight_decay=0.01,
                eps=1e-6
            )
            print(f"Vision encoder unfrozen, LR reduced to {args.lr * 0.5}")

        # Train
        train_metrics = train_epoch_memory_efficient(
            trainer,
            train_loader,
            optimizer,
            scheduler,
            gradient_accumulation_steps=args.gradient_accumulation,
            max_grad_norm=args.max_grad_norm,
            epoch=epoch,
            clear_cache_every=args.clear_cache_every
        )

        print(f"\nTrain Loss: {train_metrics['loss']:.4f} | "
              f"Valid Batches: {train_metrics['valid_batches']} | "
              f"OOM Batches: {train_metrics['oom_batches']}")

        # Validate
        val_metrics = validate_memory_efficient(trainer, val_loader)
        print(f"Validation MAE: {val_metrics['mae']:.2f} | RMSE: {val_metrics['rmse']:.2f}")

        # Log to wandb
        if args.use_wandb and WANDB_AVAILABLE:
            log_dict = {
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/oom_batches': train_metrics['oom_batches'],
                'train/valid_batches': train_metrics['valid_batches'],
                'val/mae': val_metrics['mae'],
                'val/rmse': val_metrics['rmse'],
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            if torch.cuda.is_available():
                log_dict['memory/allocated_gb'] = torch.cuda.memory_allocated() / 1e9
                log_dict['memory/max_allocated_gb'] = torch.cuda.max_memory_allocated() / 1e9
            wandb.log(log_dict)

        # Save checkpoint if best
        if val_metrics['mae'] < best_mae:
            best_mae = val_metrics['mae']
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')

            # Save with memory efficiency in mind
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'best_mae': best_mae
            }, checkpoint_path)
            print(f"Saved best model with MAE: {best_mae:.2f}")

            # Don't save optimizer state to save memory
            # Can reinitialize optimizer when resuming

        # Aggressive cleanup after each epoch
        torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()

    print(f"\nTraining complete! Best MAE: {best_mae:.2f}")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()