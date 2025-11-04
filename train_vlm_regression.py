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
from dataset_fsc147 import FSC147Dataset
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


def prepare_batch(batch, marker, device, mode='mixed'):
    """
    Prepare training batch with visual marking and targets.

    Args:
        batch: List of (image, points, metadata) tuples
        marker: Visual marker for marking points
        device: torch device
        mode: Training mode
            - 'classification': Train done classifier only (50/50 done vs not-done)
            - 'regression': Train coordinate regression only (never done)
            - 'mixed': Original behavior (for validation)

    Returns:
        For classification mode: images, num_marked, categories, None, None, gt_done
        For regression mode: images, num_marked, categories, gt_x, gt_y, None
        For mixed mode: images, num_marked, categories, gt_x, gt_y, gt_done
    """
    batch_marked_images = []
    batch_num_marked = []
    batch_gt_x = []
    batch_gt_y = []
    batch_gt_done = []
    batch_categories = []

    for img, points_list, meta in batch:
        # Convert tensor image to PIL if needed
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
        else:
            img_pil = img

        N = len(points_list)
        object_type = meta.get('object_type', meta.get('category', 'objects')) if isinstance(meta, dict) else 'objects'

        # Choose k based on mode
        if mode == 'classification':
            # Binary classification: 50% done (k=N), 50% not done (k=N-1)
            if N > 0:
                k = N if np.random.random() < 0.5 else max(0, N - 1)
            else:
                k = 0
        elif mode == 'regression':
            # Regression: never done, sample from [0, N-1]
            if N > 1:
                k = np.random.randint(0, N)  # Never equals N
            else:
                k = 0
        else:  # mixed mode (for validation)
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
        batch_categories.append(object_type)

        # Ground truth based on mode
        if mode == 'classification':
            # Only done label needed
            done = 1.0 if k >= N else 0.0
            batch_gt_done.append(done)
        elif mode == 'regression':
            # Only coordinate labels needed
            next_pt = points_list[k]
            x_norm, y_norm = normalize_coordinates(next_pt, img_pil.size)
            batch_gt_x.append(x_norm)
            batch_gt_y.append(y_norm)
        else:  # mixed
            if k < N:
                next_pt = points_list[k]
                x_norm, y_norm = normalize_coordinates(next_pt, img_pil.size)
                done = 0.0
            else:
                x_norm, y_norm = 0.0, 0.0
                done = 1.0
            batch_gt_x.append(x_norm)
            batch_gt_y.append(y_norm)
            batch_gt_done.append(done)

    # Convert to tensors based on mode
    if mode == 'classification':
        gt_done = torch.tensor(batch_gt_done, dtype=torch.bfloat16, device=device)
        return batch_marked_images, batch_num_marked, batch_categories, None, None, gt_done
    elif mode == 'regression':
        gt_x = torch.tensor(batch_gt_x, dtype=torch.bfloat16, device=device)
        gt_y = torch.tensor(batch_gt_y, dtype=torch.bfloat16, device=device)
        return batch_marked_images, batch_num_marked, batch_categories, gt_x, gt_y, None
    else:  # mixed
        gt_x = torch.tensor(batch_gt_x, dtype=torch.bfloat16, device=device)
        gt_y = torch.tensor(batch_gt_y, dtype=torch.bfloat16, device=device)
        gt_done = torch.tensor(batch_gt_done, dtype=torch.bfloat16, device=device)
        return batch_marked_images, batch_num_marked, batch_categories, gt_x, gt_y, gt_done


def train_epoch(model, train_loader, optimizer, marker, device, epoch, max_iters=None):
    """
    Train for one epoch with alternating classification and regression modes.

    50% of iterations train classification head (done signal)
    50% of iterations train regression heads (x, y coordinates)

    Args:
        max_iters: Maximum iterations per epoch (None = full dataset)
    """
    model.train()
    total_loss = 0
    total_loss_x = 0
    total_loss_y = 0
    total_loss_done = 0
    num_batches_cls = 0
    num_batches_reg = 0

    # Track class balance for classification mode (debug for first epoch)
    num_done_positive = 0
    num_done_negative = 0

    total_iters = len(train_loader) if max_iters is None else min(max_iters, len(train_loader))
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", total=total_iters)

    for batch_idx, batch in enumerate(pbar):
        if max_iters is not None and batch_idx >= max_iters:
            break

        # Alternate between classification and regression modes
        mode = 'classification' if batch_idx % 2 == 0 else 'regression'

        # Prepare batch with appropriate mode
        images, num_marked, categories, gt_x, gt_y, gt_done = prepare_batch(
            batch, marker, device, mode=mode
        )

        # Forward pass
        optimizer.zero_grad()

        # Use first category for the batch
        category = categories[0] if categories else "objects"

        outputs = model.forward_regression(
            images=images,
            num_marked=num_marked,
            category=category,
            gt_x=gt_x,
            gt_y=gt_y,
            gt_done=gt_done
        )

        loss = outputs['loss']

        # Backward pass
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Debug: Check gradients and predictions for both modes (first iteration of each)
        if epoch == 0:
            if batch_idx == 0:  # First classification iteration
                done_head_grad = sum(p.grad.abs().sum().item() for p in model.done_head.parameters() if p.grad is not None)
                pred_done_vals = outputs['done'].detach().cpu().float().numpy()
                gt_done_vals = gt_done.detach().cpu().float().numpy()
                print(f"\n[Gradient Check - Classification] done_head: {done_head_grad:.4f}")
                print(f"[Debug] pred_done range: [{pred_done_vals.min():.4f}, {pred_done_vals.max():.4f}], mean: {pred_done_vals.mean():.4f}")
                print(f"[Debug] gt_done: {gt_done_vals}")
                print(f"[Debug] loss_done: {loss.item():.4f}")
            elif batch_idx == 1:  # First regression iteration
                x_head_grad = sum(p.grad.abs().sum().item() for p in model.x_head.parameters() if p.grad is not None)
                y_head_grad = sum(p.grad.abs().sum().item() for p in model.y_head.parameters() if p.grad is not None)
                print(f"[Gradient Check - Regression] x_head: {x_head_grad:.4f}, y_head: {y_head_grad:.4f}")

        optimizer.step()

        # Track metrics - extract .item() immediately to detach from graph
        loss_val = loss.item()
        total_loss += loss_val

        if mode == 'classification':
            loss_done_val = outputs['loss_done'].item()
            total_loss_done += loss_done_val
            num_batches_cls += 1

            # Track class balance (for first epoch)
            if epoch == 0:
                num_done_positive += (gt_done == 1.0).sum().item()
                num_done_negative += (gt_done == 0.0).sum().item()

            pbar.set_postfix({
                'mode': 'cls',
                'loss': f'{loss_val:.4f}',
                'loss_done': f'{loss_done_val:.3f}'
            })
        else:  # regression
            loss_x_val = outputs['loss_x'].item()
            loss_y_val = outputs['loss_y'].item()
            total_loss_x += loss_x_val
            total_loss_y += loss_y_val
            num_batches_reg += 1
            pbar.set_postfix({
                'mode': 'reg',
                'loss': f'{loss_val:.4f}',
                'loss_x': f'{loss_x_val:.3f}',
                'loss_y': f'{loss_y_val:.3f}'
            })

        # Clear all references to prevent memory leaks
        del outputs, loss, images, num_marked, categories
        if gt_x is not None:
            del gt_x, gt_y
        if gt_done is not None:
            del gt_done

        # Only call empty_cache periodically (it's expensive)
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    # Compute averages
    num_batches = num_batches_cls + num_batches_reg
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_loss_x = total_loss_x / num_batches_reg if num_batches_reg > 0 else 0
    avg_loss_y = total_loss_y / num_batches_reg if num_batches_reg > 0 else 0
    avg_loss_done = total_loss_done / num_batches_cls if num_batches_cls > 0 else 0

    print(f"\n[Epoch {epoch} Summary] Total batches: {num_batches} (cls: {num_batches_cls}, reg: {num_batches_reg})")

    # Report class balance for first epoch
    if epoch == 0:
        total_cls_samples = num_done_positive + num_done_negative
        if total_cls_samples > 0:
            pos_pct = 100 * num_done_positive / total_cls_samples
            neg_pct = 100 * num_done_negative / total_cls_samples
            print(f"[Class Balance] Done=1: {num_done_positive} ({pos_pct:.1f}%), Done=0: {num_done_negative} ({neg_pct:.1f}%)")

    return {
        'loss': avg_loss,
        'loss_x': avg_loss_x,
        'loss_y': avg_loss_y,
        'loss_done': avg_loss_done
    }


def validate(model, val_loader, marker, device, epoch, log_images=True, num_images_to_log=16, max_iters=None):
    """
    Validate on validation set.

    Args:
        max_iters: Maximum iterations for validation (None = full dataset)
    """
    model.eval()
    total_loss = 0
    total_loss_x = 0
    total_loss_y = 0
    total_loss_done = 0
    num_batches = 0

    # For image logging
    images_logged = 0
    wandb_images = []

    total_iters = len(val_loader) if max_iters is None else min(max_iters, len(val_loader))

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[Validation]", total=total_iters)

        for batch_idx, batch in enumerate(pbar):
            if max_iters is not None and batch_idx >= max_iters:
                break
            # Prepare batch (use mixed mode for validation)
            images, num_marked, categories, gt_x, gt_y, gt_done = prepare_batch(
                batch, marker, device, mode='mixed'
            )

            # Forward pass
            category = categories[0] if categories else "objects"

            outputs = model.forward_regression(
                images=images,
                num_marked=num_marked,
                category=category,
                gt_x=gt_x,
                gt_y=gt_y,
                gt_done=gt_done
            )

            # Extract values immediately to detach from graph
            loss_val = outputs['loss'].item()
            loss_x_val = outputs['loss_x'].item()
            loss_y_val = outputs['loss_y'].item()
            loss_done_val = outputs['loss_done'].item()

            total_loss += loss_val
            total_loss_x += loss_x_val
            total_loss_y += loss_y_val
            total_loss_done += loss_done_val
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss_val:.4f}',
                'loss_x': f'{loss_x_val:.3f}',
                'loss_y': f'{loss_y_val:.3f}',
                'loss_done': f'{loss_done_val:.3f}'
            })

            # Log images from first few batches
            if log_images and images_logged < num_images_to_log:
                for i in range(len(images)):
                    if images_logged >= num_images_to_log:
                        break

                    # Get the input image (already marked with num_marked[i] points)
                    input_img = images[i]
                    cat = categories[i] if i < len(categories) else "objects"

                    # Get predictions
                    pred_x = outputs['x'][i].item()
                    pred_y = outputs['y'][i].item()
                    pred_done = outputs['done'][i].item()

                    # Get ground truth
                    true_x = gt_x[i].item()
                    true_y = gt_y[i].item()
                    true_done = gt_done[i].item()

                    # Convert to PIL if needed
                    if isinstance(input_img, torch.Tensor):
                        input_img_np = input_img.permute(1, 2, 0).cpu().numpy()
                        input_img_np = (input_img_np * 255).astype(np.uint8)
                        input_img = Image.fromarray(input_img_np)

                    # Create output image with predicted point marked
                    input_img_np = np.array(input_img)
                    W, H = input_img.size

                    # Create prompt text for display (show what model actually sees)
                    if num_marked[i] == 0:
                        marked_desc = "No objects marked yet"
                    elif num_marked[i] == 1:
                        marked_desc = "1 object marked"
                    else:
                        marked_desc = f"{num_marked[i]} objects marked"

                    # Shorten for display but keep object type visible
                    prompt_text = f"Find next {cat} | {marked_desc}"

                    # Mark predicted point based on done signal
                    import cv2
                    output_img_np = input_img_np.copy()

                    if pred_done > 0.5:
                        # Model predicts DONE
                        cv2.putText(output_img_np, "DONE", (W//2 - 50, H//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        cv2.putText(output_img_np, f"done={pred_done:.2f}", (W//2 - 60, H//2 + 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        # Model predicts next point
                        # Denormalize predicted coordinates
                        pred_x_pixel = int(((pred_x + 1) / 2) * W)
                        pred_y_pixel = int(((pred_y + 1) / 2) * H)

                        # Draw red circle for prediction
                        cv2.circle(output_img_np, (pred_x_pixel, pred_y_pixel), 8, (255, 0, 0), 2)
                        # Add "NEXT" label near prediction
                        cv2.putText(output_img_np, "NEXT", (pred_x_pixel - 20, pred_y_pixel - 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        cv2.putText(output_img_np, f"done={pred_done:.2f}", (10, H - 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Create side-by-side comparison
                    side_by_side = np.concatenate([input_img_np, output_img_np], axis=1)

                    # Add prompt at the top
                    cv2.putText(side_by_side, prompt_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Add labels at bottom
                    cv2.putText(side_by_side, "INPUT", (10, H - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(side_by_side, "PREDICTION", (W + 10, H - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Get the actual full prompt that was sent to the model
                    actual_prompt = model.create_prompt(num_marked[i], cat)
                    # Extract just the text content from the messages
                    full_text = ""
                    for msg in actual_prompt:
                        if msg["role"] == "user":
                            for content in msg["content"]:
                                if content["type"] == "text":
                                    full_text = content["text"]

                    # Create W&B image with side-by-side
                    wandb_images.append(wandb.Image(
                        side_by_side,
                        caption=f"{cat} | Marked:{num_marked[i]} | Pred:({pred_x:.2f},{pred_y:.2f},done={pred_done:.2f}) | GT:({true_x:.2f},{true_y:.2f},done={true_done:.0f})\n\nFull prompt: {full_text[:200]}..."
                    ))

                    images_logged += 1

            # Clear references to prevent memory leaks
            del outputs, images, gt_x, gt_y, gt_done, num_marked, categories

            # Periodically clear CUDA cache
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

    avg_loss = total_loss / num_batches
    avg_loss_x = total_loss_x / num_batches
    avg_loss_y = total_loss_y / num_batches
    avg_loss_done = total_loss_done / num_batches

    # Clear any remaining CUDA cache
    torch.cuda.empty_cache()

    return {
        'loss': avg_loss,
        'loss_x': avg_loss_x,
        'loss_y': avg_loss_y,
        'loss_done': avg_loss_done,
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
    parser.add_argument('--dataset', type=str, default='omnicount', choices=['omnicount', 'fsc147'],
                       help='Dataset to use: omnicount or fsc147')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--categories', type=str, nargs='+', default=['Supermarket'],
                       help='Categories for OmniCount dataset (ignored for FSC147)')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--spatial_order', type=str, default='reading_order')
    parser.add_argument('--marking_alpha', type=float, default=0.3)

    # Training args
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
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
    print(f"Loading {args.dataset.upper()} dataset...")

    if args.dataset == 'fsc147':
        train_dataset = FSC147Dataset(
            dataset_root=args.data_root,
            split='train',
            spatial_order=args.spatial_order,
            image_size=None
        )
        val_dataset = FSC147Dataset(
            dataset_root=args.data_root,
            split='val',
            spatial_order=args.spatial_order,
            image_size=None
        )
    else:  # omnicount
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

    # Create marker (use 'numbers' strategy for clearer visualization)
    marker = VisualMarker(strategy='numbers', alpha=args.marking_alpha)

    # Optimizer - use higher learning rate for MLP heads to help them break out of local minima
    # VLM (with LoRA) uses base lr, MLP heads use 5x higher lr
    optimizer = torch.optim.AdamW([
        {'params': model.model.parameters(), 'lr': args.lr},
        {'params': model.x_head.parameters(), 'lr': args.lr * 5},
        {'params': model.y_head.parameters(), 'lr': args.lr * 5},
        {'params': model.done_head.parameters(), 'lr': args.lr * 5}
    ])

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Output directory: {output_dir}")
    print(f"Learning rates: VLM={args.lr}, MLP heads={args.lr * 5}")

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
            "train/loss_done": train_metrics['loss_done']
        }, step=epoch)

        # Validate (log 16 images every epoch to see progress faster)
        val_metrics = validate(model, val_loader, marker, device, epoch, log_images=True, num_images_to_log=16)

        # Log validation metrics
        wandb.log({
            "val/loss": val_metrics['loss'],
            "val/loss_x": val_metrics['loss_x'],
            "val/loss_y": val_metrics['loss_y'],
            "val/loss_done": val_metrics['loss_done']
        }, step=epoch)

        # Log validation images
        if val_metrics['images']:
            wandb.log({
                "val/predictions": val_metrics['images']
            }, step=epoch)

        print(f"\nEpoch {epoch}:")
        print(f"  Train - loss: {train_metrics['loss']:.4f}, x: {train_metrics['loss_x']:.4f}, y: {train_metrics['loss_y']:.4f}, done: {train_metrics['loss_done']:.4f}")
        print(f"  Val   - loss: {val_metrics['loss']:.4f}, x: {val_metrics['loss_x']:.4f}, y: {val_metrics['loss_y']:.4f}, done: {val_metrics['loss_done']:.4f}")

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
