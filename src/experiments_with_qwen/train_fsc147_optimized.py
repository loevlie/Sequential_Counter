#!/usr/bin/env python3
"""
Optimized training script for Qwen3-VL on FSC147 dataset with dual loss.
Memory-efficient implementation with attention regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import json
import os
from pathlib import Path
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
import wandb
import argparse
import re
from tqdm import tqdm
import gc
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from io import BytesIO
warnings.filterwarnings('ignore')

class FSC147Dataset(Dataset):
    """FSC147 dataset with pre-computed target heatmaps"""
    def __init__(self, data_dir, split='train', image_size=224):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.images_dir = self.data_dir / "images_384_VarV2"
        self.annotations_path = self.data_dir / "annotation_FSC147_384.json"

        # Load annotations
        with open(self.annotations_path, 'r') as f:
            self.annotations = json.load(f)

        # Get split from Train_Test_Val_FSC_147.json
        split_file = self.data_dir / "Train_Test_Val_FSC_147.json"
        with open(split_file, 'r') as f:
            all_splits = json.load(f)

        # Get the appropriate split
        self.image_names = all_splits[split]

        # Pre-compute heatmap parameters for efficiency
        self.heatmap_size = 14  # Small size for memory efficiency
        self.sigma = 1.0  # Smaller sigma for smaller heatmap

    def __len__(self):
        return len(self.image_names)

    def create_gaussian_heatmap_efficient(self, points, image_shape):
        """Create memory-efficient Gaussian heatmap at low resolution"""
        h, w = self.heatmap_size, self.heatmap_size
        heatmap = torch.zeros(h, w, dtype=torch.float16)

        # Scale points to heatmap size
        scale_y = h / image_shape[0]
        scale_x = w / image_shape[1]

        for point in points:
            x = int(point[0] * scale_x)
            y = int(point[1] * scale_y)

            # Clamp to valid range
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))

            # Add Gaussian peak
            for i in range(max(0, y-2), min(h, y+3)):
                for j in range(max(0, x-2), min(w, x+3)):
                    dist_sq = ((i - y) ** 2 + (j - x) ** 2)
                    heatmap[i, j] += torch.exp(torch.tensor(-dist_sq / (2 * self.sigma ** 2)))

        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Add small baseline
        heatmap = heatmap * 0.9 + 0.1

        return heatmap

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        # Load image
        img_path = self.images_dir / img_name
        image = Image.open(img_path).convert('RGB')

        # Resize image for memory efficiency
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Get annotations
        ann = self.annotations[img_name]
        count = len(ann['points'])

        # Create efficient target heatmap
        target_heatmap = self.create_gaussian_heatmap_efficient(
            ann['points'],
            (self.image_size, self.image_size)
        )

        return {
            'image': image,
            'count': count,
            'target_heatmap': target_heatmap
        }

def custom_collate_fn(batch):
    """Custom collate that keeps PIL images"""
    return {
        'image': [item['image'] for item in batch],
        'count': torch.tensor([item['count'] for item in batch], dtype=torch.long),
        'target_heatmap': torch.stack([item['target_heatmap'] for item in batch])
    }

class OptimizedQwenTrainer:
    """Optimized trainer with memory-efficient attention computation"""

    def __init__(self, model_name="Qwen/Qwen3-VL-2B-Instruct", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model with optimizations
        # Use eager attention instead of flash attention to allow gradient computation
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 for memory
            device_map="cuda",
            attn_implementation="eager"  # Enables attention gradient computation
        )

        # Disable gradient checkpointing by default to enable attention loss
        # NOTE: This uses more memory but allows attention loss computation
        # For limited memory, set use_gradient_checkpointing=True
        self.use_gradient_checkpointing = False
        if self.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("Warning: Gradient checkpointing enabled - attention loss will be disabled")

        # Only train language model initially
        for name, param in self.model.named_parameters():
            if 'visual' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.processor = AutoProcessor.from_pretrained(model_name)

        # Initialize mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler()

        print(f"Model loaded with mixed precision and gradient checkpointing")

    def compute_attention_heatmap_efficient(self, model_output, pixel_values, target_size=14):
        """Compute attention heatmap efficiently with lower resolution"""
        try:
            # Get the logits
            logits = model_output.logits

            # Get the predicted token (highest probability)
            predicted_token = logits[0, -1, :].argmax()

            # Create a scalar loss from the predicted token
            loss_scalar = logits[0, -1, predicted_token]

            # Check if pixel_values exists and has gradients enabled
            if pixel_values is not None and pixel_values.requires_grad:
                # Compute gradients with respect to pixel values
                grads = torch.autograd.grad(
                    outputs=loss_scalar,
                    inputs=pixel_values,
                    create_graph=False,  # Set to False to avoid flash attention backward issues
                    retain_graph=True,
                    only_inputs=True
                )[0]

                # Handle different gradient shapes
                if len(grads.shape) == 4:  # [batch, channels, height, width]
                    attention_map = grads.abs().mean(dim=1)  # [batch, height, width]
                elif len(grads.shape) == 3:  # [channels, height, width]
                    attention_map = grads.abs().mean(dim=0).unsqueeze(0)  # [1, height, width]
                elif len(grads.shape) == 2:  # [height, width]
                    attention_map = grads.abs().unsqueeze(0)  # [1, height, width]
                elif len(grads.shape) == 1:  # [flat_tensor]
                    # Try to reshape if it's flattened
                    size = int(grads.shape[0] ** 0.5)
                    if size * size == grads.shape[0]:
                        attention_map = grads.abs().view(1, size, size)
                    else:
                        print(f"Cannot reshape 1D gradient tensor of size {grads.shape[0]}")
                        return torch.ones(1, target_size, target_size, device=self.device, dtype=torch.float16) * 0.5
                else:
                    print(f"Unexpected gradient shape: {grads.shape}")
                    return torch.ones(1, target_size, target_size, device=self.device, dtype=torch.float16) * 0.5

                # Ensure we have 3D tensor for interpolation [batch, height, width]
                if len(attention_map.shape) == 2:
                    attention_map = attention_map.unsqueeze(0)

                # Resize to target size if needed
                if attention_map.shape[-2:] != (target_size, target_size):
                    attention_map = F.interpolate(
                        attention_map.unsqueeze(1),  # Add channel dim
                        size=(target_size, target_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)

                # Normalize
                attention_map = attention_map / (attention_map.max() + 1e-8)

                return attention_map
            else:
                # Return dummy heatmap if gradients not available
                return torch.ones(1, target_size, target_size, device=self.device, dtype=torch.float16) * 0.5

        except Exception as e:
            print(f"Attention heatmap computation failed: {e}")
            return torch.ones(1, target_size, target_size, device=self.device, dtype=torch.float16) * 0.5

    def train_step_optimized(self, batch, accumulate_only=False, return_heatmaps=False):
        """Optimized training step with mixed precision
        Args:
            batch: Training batch
            accumulate_only: If True, only accumulate gradients without optimizer step
            return_heatmaps: If True, return attention heatmaps for visualization
        """
        images = batch['image']
        true_counts = batch['count'].to(self.device)
        target_heatmaps = batch['target_heatmap'].to(self.device, dtype=torch.float16)

        batch_size = len(images)
        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        heatmaps_for_viz = [] if return_heatmaps else None

        for i in range(batch_size):
            # Clear any cached memory
            if i > 0:
                torch.cuda.empty_cache()

            image = images[i]
            true_count = true_counts[i].item()
            target_heatmap = target_heatmaps[i]

            # Create conversation format
            question = "Count the objects in this image. Respond with COUNT: followed by a number."
            answer = f"COUNT: {true_count}"

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
                    "content": answer
                }
            ]

            # Prepare inputs
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            )

            # Move to device and set dtype
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v
                     for k, v in inputs.items()}

            # Enable gradients for pixel values
            if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
                inputs['pixel_values'].requires_grad = True

            # Mixed precision forward pass
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self.model(**inputs, labels=inputs['input_ids'])

                # Count loss (language modeling)
                count_loss = outputs.loss
                if count_loss is None:
                    print(f"Warning: outputs.loss is None for batch {i}")
                    continue  # Skip this sample

                # Debug: Print loss value occasionally
                if i == 0 and torch.rand(1).item() < 0.01:  # 1% chance to print
                    print(f"Debug - count_loss: {count_loss.item():.6f}")

                # Compute attention loss efficiently
                attention_loss = torch.tensor(0.0, device=self.device, dtype=torch.float16)

                # Enable attention loss now that we're using eager attention
                # Eager attention supports backward gradients for attention heatmaps
                use_attention_loss = True  # Enabled since we're using eager attention

                if use_attention_loss and not self.use_gradient_checkpointing:
                    if inputs.get('pixel_values') is not None and inputs['pixel_values'].requires_grad:
                        try:
                            attention_heatmap = self.compute_attention_heatmap_efficient(
                                outputs, inputs['pixel_values'], target_size=target_heatmap.shape[0]
                            )

                            if attention_heatmap is not None:
                                # MSE loss for attention regularization
                                attention_loss = F.mse_loss(
                                    attention_heatmap[0],
                                    target_heatmap,
                                    reduction='mean'
                                )

                                # Debug: Show attention loss occasionally
                                if torch.rand(1).item() < 0.02:  # 2% chance
                                    print(f"Debug - attention_loss: {attention_loss.item():.6f}")

                                # Store for visualization if requested
                                if return_heatmaps:
                                    heatmaps_for_viz.append({
                                        'predicted': attention_heatmap[0].detach().cpu().numpy(),
                                        'target': target_heatmap.detach().cpu().numpy(),
                                        'image': image,
                                        'count': true_count
                                    })
                        except RuntimeError as e:
                            if "flash_attention_backward" in str(e):
                                print(f"Flash attention backward not supported, skipping attention loss")
                            else:
                                print(f"Error computing attention heatmap: {e}")

                # Combine losses with weights
                loss = count_loss + 0.1 * attention_loss

                # Clamp loss to prevent explosions (increased threshold for initial training)
                loss = torch.clamp(loss, max=100.0)  # Increased from 10.0 since normal losses are ~20

            # Don't scale loss since batch_size is always 1
            # loss = loss / batch_size

            # Backward pass with mixed precision
            self.scaler.scale(loss).backward()

            # Accumulate loss (use unscaled value)
            total_loss += loss.detach().float()

            # Debug print
            if i == 0 and torch.rand(1).item() < 0.05:  # 5% chance
                print(f"Debug - loss after backward: {loss.item():.6f}, total_loss so far: {total_loss.item():.6f}")

        # Return results (don't multiply by batch_size since we didn't divide earlier)
        loss_value = total_loss.item()

        if return_heatmaps:
            return loss_value, heatmaps_for_viz
        else:
            return loss_value

    def optimizer_step(self, optimizer):
        """Perform optimizer step with gradient clipping"""
        try:
            # Gradient clipping
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizer step
            self.scaler.step(optimizer)
            self.scaler.update()
        except RuntimeError as e:
            if "Attempting to unscale" in str(e):
                # This can happen when attention heatmap gradients interfere
                # Skip unscaling and directly step
                print("Warning: Skipping gradient unscaling due to FP16 issue")
                optimizer.step()
                self.scaler.update()
            else:
                raise e

        optimizer.zero_grad()

        # Clear cache
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/FSC147')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-VL-2B-Instruct')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--gradient_accumulation', type=int, default=8)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_optimized')
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="qwen-fsc147-optimized",
            name=f"optimized_lr{args.lr}_ga{args.gradient_accumulation}",
            config=vars(args)
        )

    # Create datasets
    train_dataset = FSC147Dataset(args.data_dir, split='train', image_size=args.image_size)
    val_dataset = FSC147Dataset(args.data_dir, split='val', image_size=args.image_size)

    # Create dataloaders with custom collate
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Process one at a time
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0
    )

    print(f"Loaded {len(train_dataset)} images for train split")
    print(f"Loaded {len(val_dataset)} images for val split")

    # Initialize trainer
    trainer = OptimizedQwenTrainer(model_name=args.model_name)

    # Optimizer with small learning rate
    optimizer = torch.optim.AdamW(
        [p for p in trainer.model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(train_loader) // args.gradient_accumulation
    )

    # Training loop
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Unfreeze vision encoder after first epoch
        if epoch == 1:
            print("Unfreezing vision encoder")
            for param in trainer.model.visual.parameters():
                param.requires_grad = True

            # Re-initialize optimizer with all parameters
            optimizer = torch.optim.AdamW(
                trainer.model.parameters(),
                lr=args.lr / 2,  # Lower LR for vision encoder
                weight_decay=0.01
            )

        # Training
        trainer.model.train()
        train_loss = 0
        batch_count = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc="Training", total=len(train_loader))
        accumulated_batches = []

        for batch_idx, batch in enumerate(train_loader):
            try:
                accumulated_batches.append(batch)
                pbar.update(1)  # Update progress bar for each batch

                # Process accumulated batches when we reach gradient_accumulation steps or at the end
                should_update = (len(accumulated_batches) == args.gradient_accumulation) or \
                               (batch_idx == len(train_loader) - 1)

                if should_update:
                    # Process all accumulated batches
                    total_batch_loss = 0
                    for i, acc_batch in enumerate(accumulated_batches):
                        # Only accumulate gradients for all but the last batch
                        accumulate_only = (i < len(accumulated_batches) - 1)

                        # Log heatmaps every 100 batches
                        return_heatmaps = (batch_idx % 100 == 0) and (i == 0) and args.use_wandb and not trainer.use_gradient_checkpointing

                        result = trainer.train_step_optimized(acc_batch, accumulate_only=accumulate_only, return_heatmaps=return_heatmaps)

                        if return_heatmaps and isinstance(result, tuple):
                            loss, heatmaps = result
                            # Log first heatmap to wandb
                            if heatmaps and len(heatmaps) > 0:
                                viz = heatmaps[0]
                                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                                # Original image
                                axes[0].imshow(viz['image'])
                                axes[0].set_title(f"Image (Count: {viz['count']})")
                                axes[0].axis('off')

                                # Target heatmap
                                im1 = axes[1].imshow(viz['target'], cmap='hot', interpolation='nearest')
                                axes[1].set_title("Target Heatmap")
                                axes[1].axis('off')
                                plt.colorbar(im1, ax=axes[1], fraction=0.046)

                                # Predicted heatmap
                                im2 = axes[2].imshow(viz['predicted'], cmap='hot', interpolation='nearest')
                                axes[2].set_title("Predicted Attention")
                                axes[2].axis('off')
                                plt.colorbar(im2, ax=axes[2], fraction=0.046)

                                plt.tight_layout()
                                wandb.log({f"attention_heatmap_epoch{epoch}_batch{batch_idx}": wandb.Image(fig)})
                                plt.close(fig)
                        else:
                            loss = result if not isinstance(result, tuple) else result[0]

                        total_batch_loss += loss

                    # Perform optimizer step after all batches
                    trainer.optimizer_step(optimizer)

                    avg_loss = total_batch_loss / len(accumulated_batches)
                    train_loss += avg_loss
                    batch_count += 1
                    scheduler.step()

                    # Clear accumulated batches
                    accumulated_batches = []

                    pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'batch': f'{batch_idx+1}/{len(train_loader)}'})

                    if args.use_wandb:
                        wandb.log({
                            'train_loss': avg_loss,
                            'learning_rate': scheduler.get_last_lr()[0],
                            'epoch': epoch,
                            'batch': batch_idx
                        })

            except torch.cuda.OutOfMemoryError:
                print(f"OOM at batch {batch_idx}, skipping...")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                accumulated_batches = []  # Clear accumulated batches
                continue

            except Exception as e:
                print(f"Error at batch {batch_idx}: {e}")
                pbar.update(1)  # Still update progress even on error
                continue

        pbar.close()  # Close progress bar when done
        avg_train_loss = train_loss / max(batch_count, 1)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    print("Training completed!")

    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()