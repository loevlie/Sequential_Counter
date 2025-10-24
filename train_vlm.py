#!/usr/bin/env python3
"""
Training Script for Sequential Counting with VLM (Qwen3-VL)

Uses Qwen3-VL-4B-Thinking with LoRA fine-tuning for efficient training.
Implements random prefix training strategy with visual marking.

Features:
- LoRA fine-tuning (efficient ~10-50M trainable parameters)
- CSV logging every epoch
- Early stopping
- LR scheduler (ReduceLROnPlateau)
- Mixed precision training (bfloat16)
- Saves all metrics and checkpoints
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import json
import csv
import argparse
from datetime import datetime
from PIL import Image

from dataset import OmniCountDataset
from model_vlm import VLMCountingModel, create_target_text
from utils import VisualMarker


class EarlyStopping:
    """Early stopping to stop training when val loss doesn't improve."""

    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class RandomPrefixTrainer:
    """Trainer using random prefix strategy with VLM."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        output_dir,
        lr=1e-5,
        epochs=50,
        device='cuda',
        early_stopping_patience=7,
        lr_scheduler_patience=3,
        lr_scheduler_factor=0.5,
        marking_alpha=0.3,
        max_new_tokens=128
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.epochs = epochs
        self.max_new_tokens = max_new_tokens

        # Optimizer (only LoRA parameters are trainable)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience
        )

        # Early stopping
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)

        # Visual marker
        self.marker = VisualMarker(
            strategy='heatmap',
            alpha=marking_alpha
        )

        # Metrics tracking
        self.metrics_file = self.output_dir / 'metrics.csv'
        self._init_csv()

        self.best_val_loss = float('inf')
        self.epoch = 0

    def _init_csv(self):
        """Initialize CSV file for metrics."""
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch',
                'train_loss',
                'val_loss',
                'learning_rate',
                'timestamp'
            ])

    def _log_metrics(self, train_loss, val_loss):
        """Log metrics to CSV."""
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.epoch,
                f'{train_loss:.6f}',
                f'{val_loss:.6f}',
                f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                datetime.now().isoformat()
            ])

    def prepare_training_batch(self, images, gt_points_list):
        """
        Prepare batch with random prefix strategy.

        For each image:
        1. Choose random k ∈ [0, N]
        2. Mark first k points
        3. Create prompt asking for point k+1 (or "done" if k=N)
        4. Create target text

        Returns:
            marked_images: List of PIL Images with visual markers
            prompts: List of prompt strings
            targets: List of target strings
        """
        marked_images = []
        prompts = []
        targets = []

        for image, gt_points in zip(images, gt_points_list):
            N = len(gt_points)
            k = random.randint(0, N)  # Random prefix

            # Convert tensor image to PIL if needed
            if isinstance(image, torch.Tensor):
                image_np = image.permute(1, 2, 0).cpu().numpy()
                image_np = (image_np * 255).astype(np.uint8)
                image_pil = Image.fromarray(image_np)
            else:
                image_pil = image

            # Mark first k points
            if k > 0:
                marked_pts = gt_points[:k].cpu().numpy() if isinstance(gt_points, torch.Tensor) else gt_points[:k]
                image_np = np.array(image_pil)
                marked_img_np = self.marker.mark_image(image_np, marked_pts)
                marked_img = Image.fromarray(marked_img_np)
            else:
                marked_img = image_pil

            # Create prompt
            prompt = self.model.create_prompt(num_marked=k)

            # Create target
            if k < N:
                # Predict next point
                target_pt = gt_points[k]
                H, W = image_pil.height, image_pil.width

                # Convert pixel to normalized [-1, 1]
                x_norm = (target_pt[0] / W) * 2 - 1
                y_norm = (target_pt[1] / H) * 2 - 1

                target = create_target_text(x_norm, y_norm, False, (W, H))
            else:
                # All done
                target = create_target_text(-1.0, -1.0, True, (image_pil.width, image_pil.height))

            marked_images.append(marked_img)
            prompts.append(prompt)
            targets.append(target)

        return marked_images, prompts, targets

    def train_epoch(self):
        """Train for one epoch."""
        self.model.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch} [Train]')

        for batch in pbar:
            images, gt_points_list, metadata = batch

            # Prepare batch with random prefix
            marked_images, prompts, targets = self.prepare_training_batch(
                images, gt_points_list
            )

            # Forward pass with teacher forcing
            loss = self.model.forward_with_target(
                marked_images,
                prompts,
                targets
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self):
        """Validate on validation set."""
        self.model.model.eval()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f'Epoch {self.epoch} [Val]')

        with torch.no_grad():
            for batch in pbar:
                images, gt_points_list, metadata = batch

                # Use same random prefix strategy for validation
                marked_images, prompts, targets = self.prepare_training_batch(
                    images, gt_points_list
                )

                # Forward pass
                loss = self.model.forward_with_target(
                    marked_images,
                    prompts,
                    targets
                )

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, checkpoint_name='checkpoint_latest.pt'):
        """Save checkpoint."""
        checkpoint_path = self.output_dir / checkpoint_name

        # Save LoRA adapters
        lora_dir = self.output_dir / checkpoint_name.replace('.pt', '_lora')
        self.model.save_pretrained(str(lora_dir))

        # Save training state
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, checkpoint_path)

        print(f"Checkpoint saved: {checkpoint_path} and {lora_dir}")

    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.epochs} epochs...")
        print(f"Output directory: {self.output_dir}")

        for epoch in range(self.epochs):
            self.epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Log metrics
            self._log_metrics(train_loss, val_loss)

            print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # LR scheduler
            self.scheduler.step(val_loss)

            # Save latest
            self.save_checkpoint('checkpoint_latest.pt')

            # Save best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('checkpoint_best.pt')
                print(f"✅ New best model! Val loss: {val_loss:.4f}")

            # Per-epoch checkpoint
            self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

            # Early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"\n⏹ Early stopping triggered at epoch {epoch}")
                break

        print("\n🎉 Training complete!")
        print(f"Best val loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train VLM for sequential counting')

    # Data
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to OmniCount-191 dataset')
    parser.add_argument('--categories', nargs='+', default=None,
                       help='Categories to use (default: all)')
    parser.add_argument('--spatial_order', type=str, default='reading_order',
                       choices=['reading_order', 'nearest_neighbor'],
                       help='Spatial ordering strategy')

    # Model
    parser.add_argument('--model_name', type=str,
                       default='Qwen/Qwen3-VL-4B-Thinking',
                       help='Base VLM model name')
    parser.add_argument('--lora_r', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha')
    parser.add_argument('--load_in_4bit', action='store_true', default=True,
                       help='Load model in 4-bit quantization')

    # Training
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--marking_alpha', type=float, default=0.3,
                       help='Visual marking transparency')
    parser.add_argument('--max_new_tokens', type=int, default=128,
                       help='Max tokens for generation')

    # Optimization
    parser.add_argument('--early_stopping_patience', type=int, default=7,
                       help='Early stopping patience')
    parser.add_argument('--lr_scheduler_patience', type=int, default=3,
                       help='LR scheduler patience')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5,
                       help='LR scheduler reduction factor')

    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')

    args = parser.parse_args()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Save args
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load datasets
    print("Loading datasets...")
    train_dataset = OmniCountDataset(
        dataset_root=args.data_root,
        split='train',
        categories=args.categories,
        spatial_order=args.spatial_order
    )

    val_dataset = OmniCountDataset(
        dataset_root=args.data_root,
        split='valid',  # Note: dataset uses 'valid' not 'val'
        categories=args.categories,
        spatial_order=args.spatial_order
    )

    print(f"Train: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images")

    # Custom collate function for variable-length data
    def collate_fn(batch):
        """Collate batch of (image, points, metadata)."""
        images = [item[0] for item in batch]  # PIL Images
        points = [torch.tensor(item[1], dtype=torch.float32) for item in batch]  # Lists of points
        metadata = [item[2] for item in batch]  # Metadata dicts
        return images, points, metadata

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # Create model
    print("Loading VLM model...")
    model = VLMCountingModel(
        model_name=args.model_name,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        load_in_4bit=args.load_in_4bit,
        device=device
    )

    # Create trainer
    trainer = RandomPrefixTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        lr=args.lr,
        epochs=args.epochs,
        device=device,
        early_stopping_patience=args.early_stopping_patience,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        marking_alpha=args.marking_alpha,
        max_new_tokens=args.max_new_tokens
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
