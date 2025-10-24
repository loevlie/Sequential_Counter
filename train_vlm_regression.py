#!/usr/bin/env python3
"""
Training script for VLM with MLP regression head.

Uses direct coordinate + count regression instead of text generation.
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import wandb

from model_vlm_regression import VLMCountingModelRegression
from dataset import OmniCountDataset
from utils import VisualMarker


def normalize_coordinates(points, image_size):
    """
    Normalize pixel coordinates to [-1, 1].

    Args:
        points: (x, y) in pixels
        image_size: (width, height)

    Returns:
        (x_norm, y_norm) in [-1, 1]
    """
    x, y = points
    W, H = image_size

    x_norm = (x / W) * 2 - 1
    y_norm = (y / H) * 2 - 1

    return x_norm, y_norm


def prepare_batch(batch, marker, device):
    """
    Prepare training batch with visual marking and targets.

    For each image, we randomly select how many points are already marked (k),
    then predict the (k+1)-th point and total count.
    """
    # batch is a list of tuples: [(image, points, metadata), ...]
    batch_marked_images = []
    batch_num_marked = []
    batch_gt_x = []
    batch_gt_y = []
    batch_gt_count = []

    for img, points_list, meta in batch:
        # Convert tensor image to PIL if needed
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
        else:
            img_pil = img

        # Total count
        N = len(points_list)

        # Randomly choose how many to mark (0 to N-1)
        # If k < N: predict (k+1)-th point and count N
        # If k == N: predict done (-1, -1, -1)
        if N > 0:
            k = np.random.randint(0, N + 1)
        else:
            k = 0

        # Mark first k points
        if k > 0:
            marked_pts = points_list[:k]
            img_np = np.array(img_pil)
            img_marked = marker.mark_image(img_np, marked_pts)
            img_pil = Image.fromarray(img_marked)

        batch_marked_images.append(img_pil)
        batch_num_marked.append(k)

        # Ground truth
        if k < N:
            # Predict next point
            next_pt = points_list[k]
            x_norm, y_norm = normalize_coordinates(next_pt, img_pil.size)
            count = float(N)  # Total count
        else:
            # Done signal: x=-1, y=-1, but count is still the total
            x_norm, y_norm = -1.0, -1.0
            count = float(N)  # Still output correct total count!

        batch_gt_x.append(x_norm)
        batch_gt_y.append(y_norm)
        batch_gt_count.append(count)

    # Convert to tensors
    gt_x = torch.tensor(batch_gt_x, dtype=torch.float32, device=device)
    gt_y = torch.tensor(batch_gt_y, dtype=torch.float32, device=device)
    gt_count = torch.tensor(batch_gt_count, dtype=torch.float32, device=device)

    return batch_marked_images, batch_num_marked, gt_x, gt_y, gt_count


def train_epoch(model, train_loader, optimizer, marker, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_loss_x = 0
    total_loss_y = 0
    total_loss_count = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, batch in enumerate(pbar):
        # Prepare batch
        images, num_marked, gt_x, gt_y, gt_count = prepare_batch(
            batch, marker, device
        )

        # Forward pass
        optimizer.zero_grad()

        outputs = model.forward_regression(
            images=images,
            num_marked=num_marked,
            gt_x=gt_x,
            gt_y=gt_y,
            gt_count=gt_count
        )

        loss = outputs['loss']

        # Compute individual losses for logging
        loss_x = nn.functional.l1_loss(outputs['x'], gt_x)
        loss_y = nn.functional.l1_loss(outputs['y'], gt_y)
        loss_count = nn.functional.l1_loss(outputs['count'], gt_count)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_loss_x += loss_x.item()
        total_loss_y += loss_y.item()
        total_loss_count += loss_count.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    avg_loss_x = total_loss_x / num_batches
    avg_loss_y = total_loss_y / num_batches
    avg_loss_count = total_loss_count / num_batches

    return {
        'loss': avg_loss,
        'loss_x': avg_loss_x,
        'loss_y': avg_loss_y,
        'loss_count': avg_loss_count
    }


def validate(model, val_loader, marker, device, epoch, log_images=True, num_images_to_log=4):
    """Validate on validation set."""
    model.eval()
    total_loss = 0
    total_loss_x = 0
    total_loss_y = 0
    total_loss_count = 0
    num_batches = 0

    # For image logging
    images_logged = 0
    wandb_images = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[Validation]")

        for batch_idx, batch in enumerate(pbar):
            # Prepare batch
            images, num_marked, gt_x, gt_y, gt_count = prepare_batch(
                batch, marker, device
            )

            # Forward pass
            outputs = model.forward_regression(
                images=images,
                num_marked=num_marked,
                gt_x=gt_x,
                gt_y=gt_y,
                gt_count=gt_count
            )

            loss = outputs['loss']

            # Compute individual losses for logging
            loss_x = nn.functional.l1_loss(outputs['x'], gt_x)
            loss_y = nn.functional.l1_loss(outputs['y'], gt_y)
            loss_count = nn.functional.l1_loss(outputs['count'], gt_count)

            total_loss += loss.item()
            total_loss_x += loss_x.item()
            total_loss_y += loss_y.item()
            total_loss_count += loss_count.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Log images from first few batches
            if log_images and images_logged < num_images_to_log:
                for i in range(len(images)):
                    if images_logged >= num_images_to_log:
                        break

                    # Get the input image (already marked with num_marked[i] points)
                    input_img = images[i]

                    # Get predictions
                    pred_x = outputs['x'][i].item()
                    pred_y = outputs['y'][i].item()
                    pred_count = outputs['count'][i].item()

                    # Get ground truth
                    true_x = gt_x[i].item()
                    true_y = gt_y[i].item()
                    true_count = gt_count[i].item()

                    # Convert to PIL if needed
                    if isinstance(input_img, torch.Tensor):
                        input_img_np = input_img.permute(1, 2, 0).cpu().numpy()
                        input_img_np = (input_img_np * 255).astype(np.uint8)
                        input_img = Image.fromarray(input_img_np)

                    # Create output image with predicted point marked
                    input_img_np = np.array(input_img)

                    # Only mark if not done signal
                    if pred_x > -0.9 and pred_y > -0.9:
                        W, H = input_img.size
                        # Denormalize predicted coordinates
                        pred_x_pixel = int(((pred_x + 1) / 2) * W)
                        pred_y_pixel = int(((pred_y + 1) / 2) * H)

                        # Mark the predicted point
                        output_img_np = marker.mark_image(input_img_np, [(pred_x_pixel, pred_y_pixel)])
                        output_img = Image.fromarray(output_img_np)
                    else:
                        output_img = input_img.copy()

                    # Create W&B image with annotations
                    wandb_images.append(wandb.Image(
                        input_img,
                        caption=f"Input (marked: {num_marked[i]})"
                    ))

                    wandb_images.append(wandb.Image(
                        output_img,
                        caption=f"Output | Pred: ({pred_x:.2f}, {pred_y:.2f}), Count: {pred_count:.1f} | GT: ({true_x:.2f}, {true_y:.2f}), Count: {true_count:.1f}"
                    ))

                    images_logged += 1

    avg_loss = total_loss / num_batches
    avg_loss_x = total_loss_x / num_batches
    avg_loss_y = total_loss_y / num_batches
    avg_loss_count = total_loss_count / num_batches

    return {
        'loss': avg_loss,
        'loss_x': avg_loss_x,
        'loss_y': avg_loss_y,
        'loss_count': avg_loss_count,
        'images': wandb_images
    }


def main():
    parser = argparse.ArgumentParser(description="Train VLM with Regression Head")

    # Model args
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-VL-4B-Thinking')
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--mlp_layers', type=int, default=3)

    # Data args
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--categories', type=str, nargs='+', default=['Supermarket'])
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--spatial_order', type=str, default='reading_order')
    parser.add_argument('--marking_alpha', type=float, default=0.3)

    # Training args
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--output_dir', type=str, default='vlm_regression_model')

    # W&B args
    parser.add_argument('--wandb_project', type=str, default='sequential-counting')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)

    args = parser.parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        config={
            "model_name": args.model_name,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "mlp_layers": args.mlp_layers,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "categories": args.categories,
            "spatial_order": args.spatial_order,
            "marking_alpha": args.marking_alpha,
            "load_in_4bit": args.load_in_4bit
        }
    )

    # Load datasets
    print("Loading datasets...")
    train_dataset = OmniCountDataset(
        dataset_root=args.data_root,
        categories=args.categories,
        split='train',
        spatial_order=args.spatial_order,
        image_size=None
    )

    val_dataset = OmniCountDataset(
        dataset_root=args.data_root,
        categories=args.categories,
        split='valid',
        spatial_order=args.spatial_order,
        image_size=None
    )

    print(f"Train: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda x: x  # Return list of samples
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: x
    )

    # Load model
    print("Loading VLM model...")
    model = VLMCountingModelRegression(
        model_name=args.model_name,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        load_in_4bit=args.load_in_4bit,
        device=device,
        mlp_layers=args.mlp_layers
    )

    # Create marker
    marker = VisualMarker(strategy='heatmap', alpha=args.marking_alpha)

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.model.parameters()) + list(model.prediction_head.parameters()),
        lr=args.lr
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Output directory: {output_dir}")

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, marker, device, epoch
        )

        # Log training metrics
        wandb.log({
            "epoch": epoch,
            "train/loss": train_metrics['loss'],
            "train/loss_x": train_metrics['loss_x'],
            "train/loss_y": train_metrics['loss_y'],
            "train/loss_count": train_metrics['loss_count'],
            "train/loss_spatial": train_metrics['loss_x'] + train_metrics['loss_y']  # Combined x,y distance
        }, step=epoch)

        # Validate
        val_metrics = validate(model, val_loader, marker, device, epoch, log_images=True, num_images_to_log=4)

        # Log validation metrics
        wandb.log({
            "val/loss": val_metrics['loss'],
            "val/loss_x": val_metrics['loss_x'],
            "val/loss_y": val_metrics['loss_y'],
            "val/loss_count": val_metrics['loss_count'],
            "val/loss_spatial": val_metrics['loss_x'] + val_metrics['loss_y']  # Combined x,y distance
        }, step=epoch)

        # Log validation images
        if val_metrics['images']:
            wandb.log({
                "val/predictions": val_metrics['images']
            }, step=epoch)

        print(f"\nEpoch {epoch}:")
        print(f"  Train - loss: {train_metrics['loss']:.4f}, x: {train_metrics['loss_x']:.4f}, y: {train_metrics['loss_y']:.4f}, count: {train_metrics['loss_count']:.4f}")
        print(f"  Val   - loss: {val_metrics['loss']:.4f}, x: {val_metrics['loss_x']:.4f}, y: {val_metrics['loss_y']:.4f}, count: {val_metrics['loss_count']:.4f}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            model.save_pretrained(str(output_dir / 'best_checkpoint'))
            print(f"✅ New best model! Val loss: {val_metrics['loss']:.4f}")
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_epoch"] = epoch

        # Save latest
        model.save_pretrained(str(output_dir / 'latest_checkpoint'))

    print("\n Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")

    wandb.finish()


if __name__ == '__main__':
    main()
