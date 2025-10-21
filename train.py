"""
Training Script for Sequential Counting with Random Prefix

Features:
- CSV logging every epoch
- Early stopping
- LR scheduler (ReduceLROnPlateau)
- Hyperparameter support via command line
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

from dataset import OmniCountDataset
from model_cross_attn import PointPredictionHead, PointPredictionLoss
from utils import VisualMarker
from transformers import CLIPVisionModel, CLIPProcessor


class EarlyStopping:
    """Early stopping to stop training when val loss doesn't improve."""

    def __init__(self, patience=5, min_delta=0.0001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class RandomPrefixTrainer:
    """Trainer with full metrics tracking and hyperparameter support."""

    def __init__(self,
                 vlm_encoder,
                 point_predictor,
                 train_loader,
                 val_loader,
                 processor,
                 args):
        """
        Args:
            vlm_encoder: Pretrained vision encoder
            point_predictor: PointPredictionHead
            train_loader, val_loader: DataLoaders
            processor: Image processor
            args: Argparse args with hyperparameters
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vlm_encoder = vlm_encoder.to(self.device)
        self.point_predictor = point_predictor.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.processor = processor
        self.args = args

        # Freeze VLM
        for param in self.vlm_encoder.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.point_predictor.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # LR Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-7
        )

        # Loss function
        self.loss_fn = PointPredictionLoss(
            coord_weight=args.coord_weight,
            done_weight=args.done_weight,
            consistency_weight=args.consistency_weight
        )

        # Visual marker
        self.marker = VisualMarker(
            strategy=args.marking_strategy,
            alpha=args.marking_alpha
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=args.early_stop_patience,
            min_delta=args.early_stop_delta
        )

        # Metrics tracking
        self.metrics_file = args.output_dir / 'metrics.csv'
        self._init_metrics_csv()

    def _init_metrics_csv(self):
        """Initialize CSV file for metrics."""
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch',
                'train_total_loss',
                'train_coord_loss',
                'train_done_loss',
                'train_consistency_loss',
                'val_total_loss',
                'val_coord_loss',
                'val_done_loss',
                'val_consistency_loss',
                'learning_rate',
                'timestamp'
            ])

    def prepare_training_batch(self, images, gt_points_list):
        """Prepare batch with random prefix strategy."""
        batch_size = len(images)
        max_objects = max(len(points) for points in gt_points_list)

        marked_images = []
        marked_positions = torch.zeros(batch_size, max_objects, 2)
        num_marked_list = []
        targets = torch.zeros(batch_size, 2)
        is_done_list = []

        for i, (image, gt_points) in enumerate(zip(images, gt_points_list)):
            N = len(gt_points)
            W, H = image.size

            # Random prefix
            k = random.randint(0, N)

            # Mark first k points
            marked_pts = gt_points[:k] if k > 0 else []

            # Apply visual marking
            marked_img_np = self.marker.mark_image(np.array(image), marked_pts)
            marked_images.append(marked_img_np)

            # Store positions (normalized)
            if k > 0:
                for j, (x, y) in enumerate(marked_pts):
                    marked_positions[i, j, 0] = x / W
                    marked_positions[i, j, 1] = y / H

            num_marked_list.append(k)

            # Target
            if k < N:
                target_pt = gt_points[k]
                targets[i, 0] = (target_pt[0] / W) * 2 - 1  # [-1, 1]
                targets[i, 1] = (target_pt[1] / H) * 2 - 1
                is_done_list.append(False)
            else:
                targets[i, 0] = -1.0
                targets[i, 1] = -1.0
                is_done_list.append(True)

        # Process images
        from PIL import Image
        marked_images_pil = [Image.fromarray(img) for img in marked_images]
        pixel_values = self.processor(images=marked_images_pil, return_tensors="pt")['pixel_values']

        return {
            'pixel_values': pixel_values.to(self.device),
            'marked_positions': marked_positions.to(self.device),
            'num_marked': torch.tensor(num_marked_list).to(self.device),
            'targets': targets.to(self.device),
            'is_done': torch.tensor(is_done_list).to(self.device)
        }

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.point_predictor.train()
        epoch_losses = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for images, gt_points_list, metadata in pbar:
            batch = self.prepare_training_batch(images, gt_points_list)

            # Extract features
            with torch.no_grad():
                visual_outputs = self.vlm_encoder(pixel_values=batch['pixel_values'])
                visual_features = visual_outputs.last_hidden_state

            # Predict
            predictions = self.point_predictor(
                visual_features,
                batch['marked_positions'],
                batch['num_marked']
            )

            # Loss
            losses = self.loss_fn(predictions, batch['targets'], batch['is_done'])

            # Backward
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.point_predictor.parameters(), 1.0)
            self.optimizer.step()

            # Track
            epoch_losses.append({
                'total': losses['total_loss'].item(),
                'coord': losses['coord_loss'].item(),
                'done': losses['done_loss'].item(),
                'consistency': losses['consistency_loss'].item()
            })

            pbar.set_postfix(loss=losses['total_loss'].item())

        # Average
        avg_losses = {
            key: sum(d[key] for d in epoch_losses) / len(epoch_losses)
            for key in epoch_losses[0].keys()
        }

        return avg_losses

    @torch.no_grad()
    def validate(self):
        """Validate."""
        self.point_predictor.eval()
        val_losses = []

        for images, gt_points_list, metadata in tqdm(self.val_loader, desc="Validating"):
            batch = self.prepare_training_batch(images, gt_points_list)

            visual_outputs = self.vlm_encoder(pixel_values=batch['pixel_values'])
            visual_features = visual_outputs.last_hidden_state

            predictions = self.point_predictor(
                visual_features,
                batch['marked_positions'],
                batch['num_marked']
            )

            losses = self.loss_fn(predictions, batch['targets'], batch['is_done'])

            val_losses.append({
                'total': losses['total_loss'].item(),
                'coord': losses['coord_loss'].item(),
                'done': losses['done_loss'].item(),
                'consistency': losses['consistency_loss'].item()
            })

        avg_losses = {
            key: sum(d[key] for d in val_losses) / len(val_losses)
            for key in val_losses[0].keys()
        }

        return avg_losses

    def save_checkpoint(self, epoch, train_losses, val_losses, is_best=False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.point_predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'args': vars(self.args)
        }

        # Save latest
        torch.save(checkpoint, self.args.output_dir / 'checkpoint_latest.pt')

        # Save best
        if is_best:
            torch.save(checkpoint, self.args.output_dir / 'checkpoint_best.pt')

        # Save numbered
        torch.save(checkpoint, self.args.output_dir / f'checkpoint_epoch_{epoch}.pt')

    def log_metrics(self, epoch, train_losses, val_losses, lr):
        """Log metrics to CSV."""
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_losses['total'],
                train_losses['coord'],
                train_losses['done'],
                train_losses['consistency'],
                val_losses['total'],
                val_losses['coord'],
                val_losses['done'],
                val_losses['consistency'],
                lr,
                datetime.now().isoformat()
            ])

    def train(self):
        """Full training loop."""
        print(f"\n{'='*80}")
        print(f"Starting Training")
        print(f"{'='*80}")
        print(f"Output dir: {self.args.output_dir}")
        print(f"Device: {self.device}")
        print(f"Hyperparameters:")
        for key, value in vars(self.args).items():
            print(f"  {key}: {value}")
        print(f"{'='*80}\n")

        best_val_loss = float('inf')

        for epoch in range(1, self.args.epochs + 1):
            print(f"\n=== Epoch {epoch}/{self.args.epochs} ===")

            # Train
            train_losses = self.train_epoch(epoch)

            # Validate
            val_losses = self.validate()

            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log
            print(f"Train - Total: {train_losses['total']:.4f}, "
                  f"Coord: {train_losses['coord']:.4f}, "
                  f"Done: {train_losses['done']:.4f}")
            print(f"Val   - Total: {val_losses['total']:.4f}, "
                  f"Coord: {val_losses['coord']:.4f}, "
                  f"Done: {val_losses['done']:.4f}")
            print(f"LR: {current_lr:.2e}")

            # Save metrics
            self.log_metrics(epoch, train_losses, val_losses, current_lr)

            # Save checkpoint
            is_best = val_losses['total'] < best_val_loss
            if is_best:
                best_val_loss = val_losses['total']
                print(f"✓ New best model! (val_loss={best_val_loss:.4f})")

            self.save_checkpoint(epoch, train_losses, val_losses, is_best)

            # LR scheduler step
            self.scheduler.step(val_losses['total'])

            # Early stopping
            self.early_stopping(val_losses['total'])
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Metrics saved to: {self.metrics_file}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Train sequential counting model')

    # Data
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to OmniCount-191 dataset')
    parser.add_argument('--categories', type=str, nargs='+',
                       default=['Supermarket', 'Fruits', 'Urban'],
                       help='Categories to use')
    parser.add_argument('--min_objects', type=int, default=5)
    parser.add_argument('--max_objects', type=int, default=30)
    parser.add_argument('--image_size', type=int, default=224,
                       help='Resize images to this size')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    # Loss weights
    parser.add_argument('--coord_weight', type=float, default=1.0)
    parser.add_argument('--done_weight', type=float, default=0.5)
    parser.add_argument('--consistency_weight', type=float, default=0.3)

    # Marking
    parser.add_argument('--marking_strategy', type=str, default='heatmap',
                       choices=['heatmap', 'numbers', 'dots'])
    parser.add_argument('--marking_alpha', type=float, default=0.3)

    # Spatial order
    parser.add_argument('--spatial_order', type=str, default='reading_order',
                       choices=['reading_order', 'left_to_right', 'nearest_neighbor'])

    # Early stopping
    parser.add_argument('--early_stop_patience', type=int, default=7)
    parser.add_argument('--early_stop_delta', type=float, default=0.0001)

    # Output
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory to save outputs')
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    # Create output directory
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(args.output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load datasets
    print("Loading datasets...")
    train_dataset = OmniCountDataset(
        dataset_root=args.data_root,
        categories=args.categories,
        split='train',
        spatial_order=args.spatial_order,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        image_size=(args.image_size, args.image_size)
    )

    val_dataset = OmniCountDataset(
        dataset_root=args.data_root,
        categories=args.categories,
        split='valid',
        spatial_order=args.spatial_order,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        image_size=(args.image_size, args.image_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )

    print(f"✓ Train: {len(train_dataset)} examples")
    print(f"✓ Val: {len(val_dataset)} examples")

    # Load VLM
    print("Loading CLIP ViT-B/32...")
    vlm_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Create model
    point_predictor = PointPredictionHead(
        visual_feature_dim=768,
        hidden_dim=args.hidden_dim,
        max_objects=args.max_objects,
        dropout=args.dropout
    )

    # Train
    trainer = RandomPrefixTrainer(
        vlm_encoder, point_predictor,
        train_loader, val_loader,
        processor, args
    )

    trainer.train()


if __name__ == "__main__":
    main()
