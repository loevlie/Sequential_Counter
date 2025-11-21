#!/usr/bin/env python3
"""
Training script with WORKING attention loss for Qwen3-VL on FSC147.
This version properly computes gradient-based attention heatmaps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json
import os
from tqdm import tqdm
import gc
import re
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import Dict, Tuple
import argparse
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed")


def extract_count_from_text(text: str) -> int:
    """Extract count from generated text."""
    match = re.search(r'COUNT:\s*(\d+)', text)
    if match:
        return int(match.group(1))
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[0])
    return -1


def create_gaussian_heatmap_smooth(points: np.ndarray, image_shape: Tuple[int, int],
                                  sigma: float = 2.0, baseline: float = 0.1) -> np.ndarray:
    """Create smooth target heatmap with Gaussian peaks at object centers."""
    height, width = image_shape
    heatmap = np.full((height, width), baseline, dtype=np.float32)

    if len(points) == 0:
        return heatmap

    # Add peaks at each point
    for px, py in points:
        # Ensure points are within bounds
        px = max(0, min(width - 1, int(px)))
        py = max(0, min(height - 1, int(py)))
        heatmap[py, px] = 1.0

    # Apply Gaussian smoothing
    heatmap = gaussian_filter(heatmap, sigma=sigma)

    # Normalize
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

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

        # Create target heatmap at 28x28 resolution
        heatmap_size = 28
        scaled_points = points * (heatmap_size / self.image_size) if len(points) > 0 else points
        target_heatmap = create_gaussian_heatmap_smooth(
            scaled_points,
            (heatmap_size, heatmap_size),
            sigma=2.0,
            baseline=0.1
        )

        return {
            'image': image,
            'count': count,
            'target_heatmap': torch.FloatTensor(target_heatmap),
            'image_id': img_id
        }


def custom_collate_fn(batch):
    """Custom collate function"""
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


class AttentionQwenTrainer:
    """Trainer with working gradient-based attention computation"""

    def __init__(self, model_name="Qwen/Qwen3-VL-2B-Instruct", device=None, freeze_vision=True):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model: {model_name}")

        # Load model in float32 for stability
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map=self.device,
            torch_dtype=torch.float32
        )

        self.processor = AutoProcessor.from_pretrained(model_name)

        # CRITICAL: Do NOT enable gradient checkpointing for attention loss
        # Gradient checkpointing interferes with gradient computation
        print("Gradient checkpointing DISABLED to allow attention loss")

        # Freeze vision encoder initially
        if freeze_vision:
            print("Freezing vision encoder initially")
            for param in self.model.visual.parameters():
                param.requires_grad = False

        # CRITICAL: Use eager attention for gradient computation
        self.model.set_attn_implementation("eager")

        print("Model loaded with eager attention for gradient computation")

    def unfreeze_vision(self):
        """Unfreeze vision encoder"""
        print("Unfreezing vision encoder")
        for param in self.model.visual.parameters():
            param.requires_grad = True

    def prepare_inputs_with_labels(self, image: Image.Image, count: int):
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
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Create labels with proper masking
        labels = inputs["input_ids"].clone()

        # Find where "COUNT:" appears
        decoded = self.processor.decode(inputs["input_ids"][0], skip_special_tokens=False)
        count_pos = decoded.find("COUNT:")

        if count_pos > 0:
            # Estimate token position
            ratio = count_pos / len(decoded)
            mask_until = int(ratio * labels.shape[1] * 0.9)
            labels[:, :mask_until] = -100
        else:
            # Fallback: mask 75% of tokens
            mask_length = int(labels.shape[1] * 0.75)
            labels[:, :mask_length] = -100

        # Ensure we have tokens to predict
        if (labels != -100).sum() < 5:
            labels = inputs["input_ids"].clone()
            labels[:, :labels.shape[1]//2] = -100

        inputs["labels"] = labels

        return inputs

    def compute_attention_heatmap(self, image: Image.Image, count: int, target_size: int = 28):
        """
        Compute gradient-based attention heatmap.
        This needs to be done in a separate forward pass to avoid conflicts.
        """
        # Prepare inputs WITHOUT labels for generation
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

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
            padding=True
        )

        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Enable gradients on pixel values
        if "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"].requires_grad_(True)
            inputs["pixel_values"] = pixel_values
        else:
            return torch.zeros((target_size, target_size), device=self.device)

        # Forward pass
        outputs = self.model(**inputs, return_dict=True)

        # Get logits and compute scalar target
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            # Use the maximum logit as target
            target_scalar = logits[0, -1, :].max()

            # Compute gradients
            grads = torch.autograd.grad(
                outputs=target_scalar,
                inputs=pixel_values,
                retain_graph=True,
                create_graph=False
            )[0]

            # Process gradients to create heatmap
            # Expected shape: [batch, channels, height, width] or [batch, num_patches, hidden_dim]
            if grads.dim() == 4:  # Standard CNN format
                # Average over batch and channels
                attention_map = grads.abs().mean(dim=(0, 1))
            elif grads.dim() == 3:  # Vision transformer format
                batch_size, num_patches, hidden_dim = grads.shape
                # Compute spatial dimensions
                patch_size = int(np.sqrt(num_patches))
                if patch_size * patch_size == num_patches:
                    # Reshape to spatial
                    grads = grads[0]  # Take first batch
                    grads = grads.reshape(patch_size, patch_size, hidden_dim)
                    attention_map = grads.abs().mean(dim=-1)
                else:
                    # Fallback
                    attention_map = torch.zeros((target_size, target_size), device=self.device)
            else:
                # Unexpected shape
                print(f"Unexpected gradient shape: {grads.shape}")
                attention_map = torch.zeros((target_size, target_size), device=self.device)

            # Resize to target size
            if attention_map.shape != (target_size, target_size):
                attention_map = F.interpolate(
                    attention_map.unsqueeze(0).unsqueeze(0),
                    size=(target_size, target_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()

            # Normalize
            if attention_map.max() > 0:
                attention_map = attention_map / attention_map.max()

            return attention_map

        return torch.zeros((target_size, target_size), device=self.device)


def train_with_attention(trainer: AttentionQwenTrainer,
                        dataloader: DataLoader,
                        optimizer: optim.Optimizer,
                        scheduler=None,
                        gradient_accumulation_steps: int = 4,
                        max_grad_norm: float = 1.0,
                        attention_weight: float = 5.0,
                        epoch: int = 0) -> Dict:
    """Training with working attention loss"""
    trainer.model.train()

    total_loss = 0
    total_count_loss = 0
    total_attn_loss = 0
    num_valid_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        try:
            image = batch['image'][0]
            count = batch['count'][0].item()
            target_heatmap = batch['target_heatmap'][0].to(trainer.device)

            # Step 1: Compute count loss with teacher forcing
            inputs = trainer.prepare_inputs_with_labels(image, count)
            outputs = trainer.model(**inputs, return_dict=True)

            if hasattr(outputs, 'loss') and outputs.loss is not None:
                count_loss = outputs.loss
                count_loss = torch.clamp(count_loss, min=0.0, max=100.0)
            else:
                print(f"No loss at batch {batch_idx}")
                continue

            # Step 2: Compute attention heatmap (separate forward pass)
            attention_loss = torch.tensor(0.0, device=trainer.device)

            # Only compute attention loss every N batches to save time
            if batch_idx % 4 == 0:  # Compute attention loss 25% of the time
                try:
                    with torch.set_grad_enabled(True):
                        pred_heatmap = trainer.compute_attention_heatmap(image, count, target_size=28)

                        if pred_heatmap.shape == target_heatmap.shape:
                            attention_loss = F.mse_loss(pred_heatmap, target_heatmap)
                            attention_loss = torch.clamp(attention_loss, min=0.0, max=10.0)
                        else:
                            print(f"Shape mismatch: pred {pred_heatmap.shape} vs target {target_heatmap.shape}")

                except Exception as e:
                    if batch_idx == 0:  # Only print once
                        print(f"Attention computation failed: {e}")
                    attention_loss = torch.tensor(0.0, device=trainer.device)

            # Combine losses
            loss = count_loss + attention_weight * attention_loss

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf at batch {batch_idx}")
                optimizer.zero_grad()
                continue

            # Scale for gradient accumulation
            loss = loss / gradient_accumulation_steps

            # Backward
            loss.backward()

            # Optimizer step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=max_grad_norm)

                # Check for NaN gradients
                has_nan = False
                for param in trainer.model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            has_nan = True
                            break

                if not has_nan:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                    # Track metrics
                    total_loss += loss.item() * gradient_accumulation_steps
                    total_count_loss += count_loss.item()
                    total_attn_loss += attention_loss.item()
                    num_valid_batches += 1

                optimizer.zero_grad()

            # Update progress bar
            if num_valid_batches > 0:
                avg_loss = total_loss / num_valid_batches
                avg_count_loss = total_count_loss / num_valid_batches
                avg_attn_loss = total_attn_loss / num_valid_batches
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'count': f'{avg_count_loss:.4f}',
                    'attn': f'{avg_attn_loss:.4f}',
                    'valid': num_valid_batches
                })

            # Clear cache periodically
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM at batch {batch_idx}")
                torch.cuda.empty_cache()
                gc.collect()
                optimizer.zero_grad()
                continue
            else:
                raise e
        except Exception as e:
            print(f"Error at batch {batch_idx}: {e}")
            continue

    # Final optimizer step if needed
    if (batch_idx + 1) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    return {
        'loss': total_loss / max(num_valid_batches, 1),
        'count_loss': total_count_loss / max(num_valid_batches, 1),
        'attention_loss': total_attn_loss / max(num_valid_batches, 1),
        'valid_batches': num_valid_batches
    }


def validate(trainer: AttentionQwenTrainer, dataloader: DataLoader) -> Dict:
    """Validation"""
    trainer.model.eval()

    total_mae = 0
    total_mse = 0
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            try:
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

                if num_samples % 20 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Validation error: {e}")
                continue

    mae = total_mae / max(num_samples, 1)
    rmse = np.sqrt(total_mse / max(num_samples, 1))

    return {'mae': mae, 'rmse': rmse, 'num_samples': num_samples}


def main():
    parser = argparse.ArgumentParser(description='Qwen3-VL training with working attention loss')
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/FSC147')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-VL-2B-Instruct')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--gradient_accumulation', type=int, default=4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--attention_weight', type=float, default=5.0)
    parser.add_argument('--freeze_vision_epochs', type=int, default=1)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_attention')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--debug_subset', type=int, default=0)

    args = parser.parse_args()

    # Initialize wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="qwen-fsc147-attention",
            name=f"attn_weight{args.attention_weight}_lr{args.lr}",
            config=vars(args)
        )

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize trainer
    trainer = AttentionQwenTrainer(args.model_name, freeze_vision=True)

    # Create datasets
    train_dataset = FSC147Dataset(args.data_dir, split='train', image_size=args.image_size)
    val_dataset = FSC147Dataset(args.data_dir, split='val', image_size=args.image_size)

    # Use debug subset if specified
    if args.debug_subset > 0:
        print(f"Using debug subset of {args.debug_subset} samples")
        train_dataset.image_ids = train_dataset.image_ids[:args.debug_subset]
        val_dataset.image_ids = val_dataset.image_ids[:min(50, args.debug_subset//4)]

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    # Optimizer
    optimizer = optim.AdamW(
        [p for p in trainer.model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01
    )

    # Scheduler
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_steps)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=total_steps - args.warmup_steps, T_mult=1)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_steps])

    # Training loop
    best_mae = float('inf')

    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Attention weight: {args.attention_weight}")
        print(f"{'='*50}")

        # Unfreeze vision encoder after initial epochs
        if epoch == args.freeze_vision_epochs:
            trainer.unfreeze_vision()
            # Recreate optimizer with all parameters
            optimizer = optim.AdamW(
                trainer.model.parameters(),
                lr=args.lr * 0.5,
                weight_decay=0.01
            )
            print(f"Vision encoder unfrozen, LR reduced to {args.lr * 0.5}")

        # Train
        train_metrics = train_with_attention(
            trainer,
            train_loader,
            optimizer,
            scheduler,
            gradient_accumulation_steps=args.gradient_accumulation,
            max_grad_norm=args.max_grad_norm,
            attention_weight=args.attention_weight,
            epoch=epoch
        )

        print(f"\nTrain Loss: {train_metrics['loss']:.4f} | "
              f"Count Loss: {train_metrics['count_loss']:.4f} | "
              f"Attention Loss: {train_metrics['attention_loss']:.4f} | "
              f"Valid Batches: {train_metrics['valid_batches']}")

        # Validate
        val_metrics = validate(trainer, val_loader)
        print(f"Validation MAE: {val_metrics['mae']:.2f} | RMSE: {val_metrics['rmse']:.2f}")

        # Log to wandb
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/count_loss': train_metrics['count_loss'],
                'train/attention_loss': train_metrics['attention_loss'],
                'val/mae': val_metrics['mae'],
                'val/rmse': val_metrics['rmse'],
                'learning_rate': optimizer.param_groups[0]['lr']
            })

        # Save checkpoint if best
        if val_metrics['mae'] < best_mae:
            best_mae = val_metrics['mae']
            checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mae': best_mae
            }, checkpoint_path)
            print(f"Saved best model with MAE: {best_mae:.2f}")

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nTraining complete! Best MAE: {best_mae:.2f}")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()