#!/usr/bin/env python3
"""
Training script for Sequential Attention Counting Model

Uses the FSC147 dataset with sequential attention mechanisms inspired by
human serial counting behavior.
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

from model_sequential_simple import SimpleSequentialModel
from dataset_fsc147 import FSC147Dataset
from utils import VisualMarker


def normalize_coordinates(points, image_size):
    """Normalize pixel coordinates to [-1, 1]."""
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
        mode: 'classification', 'regression', or 'mixed'

    Returns:
        Tuple of (images, num_marked, categories, gt_x, gt_y, gt_done)
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
            if N > 0:
                k = N if np.random.random() < 0.5 else max(0, N - 1)
            else:
                k = 0
        elif mode == 'regression':
            if N > 1:
                k = np.random.randint(0, N)
            else:
                k = 0
        else:  # mixed mode
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

        # Ground truth
        if mode == 'classification':
            done = 1.0 if k >= N else 0.0
            batch_gt_done.append(done)
        elif mode == 'regression':
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

    # Convert to tensors
    if mode == 'classification':
        gt_done = torch.tensor(batch_gt_done, dtype=torch.bfloat16, device=device)
        return batch_marked_images, batch_num_marked, batch_categories, None, None, gt_done
    elif mode == 'regression':
        gt_x = torch.tensor(batch_gt_x, dtype=torch.bfloat16, device=device)
        gt_y = torch.tensor(batch_gt_y, dtype=torch.bfloat16, device=device)
        return batch_marked_images, batch_num_marked, batch_categories, gt_x, gt_y, None
    else:
        gt_x = torch.tensor(batch_gt_x, dtype=torch.bfloat16, device=device)
        gt_y = torch.tensor(batch_gt_y, dtype=torch.bfloat16, device=device)
        gt_done = torch.tensor(batch_gt_done, dtype=torch.bfloat16, device=device)
        return batch_marked_images, batch_num_marked, batch_categories, gt_x, gt_y, gt_done


def train_epoch(model, train_loader, optimizer, marker, device, epoch, max_iters=None):
    """
    Train for one epoch with alternating modes.
    """
    model.train()
    total_loss = 0
    total_loss_x = 0
    total_loss_y = 0
    total_loss_done = 0
    num_batches_cls = 0
    num_batches_reg = 0

    total_iters = len(train_loader) if max_iters is None else min(max_iters, len(train_loader))
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", total=total_iters)

    for batch_idx, batch in enumerate(pbar):
        if max_iters is not None and batch_idx >= max_iters:
            break

        # Alternate between classification and regression
        mode = 'classification' if batch_idx % 2 == 0 else 'regression'

        # Prepare batch
        images, num_marked, categories, gt_x, gt_y, gt_done = prepare_batch(
            batch, marker, device, mode=mode
        )

        # Forward pass
        optimizer.zero_grad()

        category = categories[0] if categories else "objects"

        outputs = model.forward(
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Debug gradient flow for first epoch
        if epoch == 0:
            if batch_idx == 0:
                done_head_grad = sum(p.grad.abs().sum().item() for p in model.done_head.parameters() if p.grad is not None)
                print(f"\n[Gradient Check - Classification]")
                print(f"  done_head: {done_head_grad:.4f}")
            elif batch_idx == 1:
                x_head_grad = sum(p.grad.abs().sum().item() for p in model.x_head.parameters() if p.grad is not None)
                y_head_grad = sum(p.grad.abs().sum().item() for p in model.y_head.parameters() if p.grad is not None)
                print(f"[Gradient Check - Regression]")
                print(f"  x_head: {x_head_grad:.4f}")
                print(f"  y_head: {y_head_grad:.4f}")

        optimizer.step()

        # Track metrics
        loss_val = loss.item()
        total_loss += loss_val

        if mode == 'classification':
            loss_done_val = outputs['loss_done'].item()
            total_loss_done += loss_done_val
            num_batches_cls += 1
            pbar.set_postfix({
                'mode': 'cls',
                'loss': f'{loss_val:.4f}',
                'loss_done': f'{loss_done_val:.3f}'
            })
        else:
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

        # Clear references
        del outputs, loss, images, num_marked, categories
        if gt_x is not None:
            del gt_x, gt_y
        if gt_done is not None:
            del gt_done

        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    # Compute averages
    num_batches = num_batches_cls + num_batches_reg
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_loss_x = total_loss_x / num_batches_reg if num_batches_reg > 0 else 0
    avg_loss_y = total_loss_y / num_batches_reg if num_batches_reg > 0 else 0
    avg_loss_done = total_loss_done / num_batches_cls if num_batches_cls > 0 else 0

    return {
        'loss': avg_loss,
        'loss_x': avg_loss_x,
        'loss_y': avg_loss_y,
        'loss_done': avg_loss_done
    }


def validate(model, val_loader, marker, device, epoch, log_images=True, num_images_to_log=16, max_iters=None):
    """Validate on validation set."""
    model.eval()
    total_loss = 0
    total_loss_x = 0
    total_loss_y = 0
    total_loss_done = 0
    num_batches = 0

    images_logged = 0
    wandb_images = []

    total_iters = len(val_loader) if max_iters is None else min(max_iters, len(val_loader))

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[Validation]", total=total_iters)

        for batch_idx, batch in enumerate(pbar):
            if max_iters is not None and batch_idx >= max_iters:
                break

            # Prepare batch (mixed mode)
            images, num_marked, categories, gt_x, gt_y, gt_done = prepare_batch(
                batch, marker, device, mode='mixed'
            )

            category = categories[0] if categories else "objects"

            # Forward pass
            outputs = model.forward(
                images=images,
                num_marked=num_marked,
                category=category,
                gt_x=gt_x,
                gt_y=gt_y,
                gt_done=gt_done
            )

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

            # Log images
            if log_images and images_logged < num_images_to_log:
                import cv2
                for i in range(len(images)):
                    if images_logged >= num_images_to_log:
                        break

                    input_img = images[i]
                    cat = categories[i] if i < len(categories) else "objects"

                    pred_x = outputs['x'][i].item()
                    pred_y = outputs['y'][i].item()
                    pred_done = outputs['done'][i].item()

                    true_x = gt_x[i].item()
                    true_y = gt_y[i].item()
                    true_done = gt_done[i].item()

                    if isinstance(input_img, torch.Tensor):
                        input_img_np = input_img.permute(1, 2, 0).cpu().numpy()
                        input_img_np = (input_img_np * 255).astype(np.uint8)
                        input_img = Image.fromarray(input_img_np)

                    input_img_np = np.array(input_img)
                    W, H = input_img.size

                    output_img_np = input_img_np.copy()

                    # Draw GROUND TRUTH in GREEN
                    if true_done < 0.5:  # Not done, so there IS a ground truth location
                        gt_x_pixel = int(((true_x + 1) / 2) * W)
                        gt_y_pixel = int(((true_y + 1) / 2) * H)
                        cv2.circle(output_img_np, (gt_x_pixel, gt_y_pixel), 10, (0, 255, 0), 2)
                        cv2.putText(output_img_np, "GT", (gt_x_pixel - 15, gt_y_pixel - 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Draw PREDICTION in RED
                    if pred_done > 0.5:
                        cv2.putText(output_img_np, "PRED: DONE", (W//2 - 80, H//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                    else:
                        pred_x_pixel = int(((pred_x + 1) / 2) * W)
                        pred_y_pixel = int(((pred_y + 1) / 2) * H)
                        cv2.circle(output_img_np, (pred_x_pixel, pred_y_pixel), 8, (255, 0, 0), 2)
                        cv2.putText(output_img_np, "PRED", (pred_x_pixel - 20, pred_y_pixel - 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        # Draw line connecting prediction to ground truth if both exist
                        if true_done < 0.5:
                            gt_x_pixel = int(((true_x + 1) / 2) * W)
                            gt_y_pixel = int(((true_y + 1) / 2) * H)
                            cv2.line(output_img_np, (pred_x_pixel, pred_y_pixel),
                                   (gt_x_pixel, gt_y_pixel), (255, 255, 0), 1)
                            # Calculate pixel distance
                            dist_pixels = np.sqrt((pred_x_pixel - gt_x_pixel)**2 + (pred_y_pixel - gt_y_pixel)**2)
                            cv2.putText(output_img_np, f"err={dist_pixels:.0f}px",
                                       ((pred_x_pixel + gt_x_pixel)//2, (pred_y_pixel + gt_y_pixel)//2 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                    # Add done signals
                    cv2.putText(output_img_np, f"pred_done={pred_done:.2f}", (10, H - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(output_img_np, f"gt_done={true_done:.0f}", (10, H - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    side_by_side = np.concatenate([input_img_np, output_img_np], axis=1)

                    prompt_text = f"{cat} | Marked: {num_marked[i]}"
                    cv2.putText(side_by_side, prompt_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    cv2.putText(side_by_side, "INPUT", (10, H - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(side_by_side, "PREDICTION (w/ Sequential Attention)", (W + 10, H - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    wandb_images.append(wandb.Image(
                        side_by_side,
                        caption=f"{cat} | Marked:{num_marked[i]} | Pred:({pred_x:.2f},{pred_y:.2f},done={pred_done:.2f}) | GT:({true_x:.2f},{true_y:.2f},done={true_done:.0f})"
                    ))

                    images_logged += 1

            del outputs, images, gt_x, gt_y, gt_done, num_marked, categories

            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

    avg_loss = total_loss / num_batches
    avg_loss_x = total_loss_x / num_batches
    avg_loss_y = total_loss_y / num_batches
    avg_loss_done = total_loss_done / num_batches

    torch.cuda.empty_cache()

    return {
        'loss': avg_loss,
        'loss_x': avg_loss_x,
        'loss_y': avg_loss_y,
        'loss_done': avg_loss_done,
        'images': wandb_images
    }


def main():
    parser = argparse.ArgumentParser(description="Train Sequential Attention Counting Model")

    # Model args
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2-VL-2B-Instruct')
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--num_foveal_steps', type=int, default=4,
                       help='Number of sequential foveal glimpses')
    parser.add_argument('--num_reasoning_steps', type=int, default=3,
                       help='Number of sequential reasoning steps (thinking time)')

    # Data args
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--spatial_order', type=str, default='reading_order')
    parser.add_argument('--marking_alpha', type=float, default=0.3)
    parser.add_argument('--min_objects', type=int, default=5,
                       help='Minimum objects per image')
    parser.add_argument('--max_objects', type=int, default=50,
                       help='Maximum objects per image')

    # Training args
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_train_iters', type=int, default=200,
                       help='Max iterations per training epoch (for fast feedback)')
    parser.add_argument('--max_val_iters', type=int, default=50,
                       help='Max iterations per validation epoch')
    parser.add_argument('--output_dir', type=str, default='sequential_attention_model')

    # W&B args
    parser.add_argument('--wandb_project', type=str, default='sequential-counting')
    parser.add_argument('--wandb_run_name', type=str, default='sequential-attention-fsc147')
    parser.add_argument('--wandb_entity', type=str, default=None)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        config=vars(args)
    )

    # Load FSC147 dataset
    print("Loading FSC147 dataset...")
    train_dataset = FSC147Dataset(
        dataset_root=args.data_root,
        split='train',
        spatial_order=args.spatial_order,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        image_size=None
    )
    val_dataset = FSC147Dataset(
        dataset_root=args.data_root,
        split='val',
        spatial_order=args.spatial_order,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
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
        collate_fn=lambda x: x
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: x
    )

    # Load model
    print("Loading Simple Sequential Counting Model...")
    model = SimpleSequentialModel(
        model_name=args.model_name,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        load_in_4bit=args.load_in_4bit,
        device=device
    )

    # Create marker
    marker = VisualMarker(strategy='numbers', alpha=args.marking_alpha)

    # Optimizer - simple version
    optimizer = torch.optim.AdamW([
        {'params': model.model.parameters(), 'lr': args.lr},
        {'params': model.x_head.parameters(), 'lr': args.lr},
        {'params': model.y_head.parameters(), 'lr': args.lr},
        {'params': model.done_head.parameters(), 'lr': args.lr}
    ])

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Output directory: {output_dir}")
    print(f"Learning rate: {args.lr} (same for all modules)")
    print("Using SIMPLIFIED model (direct VLM features, no complex transformations)")

    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 5
    print(f"Early stopping enabled with patience={early_stop_patience}")

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, marker, device, epoch,
            max_iters=args.max_train_iters
        )

        # Log training metrics
        wandb.log({
            "epoch": epoch,
            "train/loss": train_metrics['loss'],
            "train/loss_x": train_metrics['loss_x'],
            "train/loss_y": train_metrics['loss_y'],
            "train/loss_done": train_metrics['loss_done']
        }, step=epoch)

        # Validate
        val_metrics = validate(
            model, val_loader, marker, device, epoch,
            log_images=True, num_images_to_log=16,
            max_iters=args.max_val_iters
        )

        # Log validation metrics
        wandb.log({
            "val/loss": val_metrics['loss'],
            "val/loss_x": val_metrics['loss_x'],
            "val/loss_y": val_metrics['loss_y'],
            "val/loss_done": val_metrics['loss_done']
        }, step=epoch)

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
            patience_counter = 0  # Reset counter on improvement
            model.save_pretrained(str(output_dir / 'best_checkpoint'))
            print(f"✅ New best model! Val loss: {val_metrics['loss']:.4f}")
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_epoch"] = epoch
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")
            if patience_counter >= early_stop_patience:
                print(f"\n🛑 Early stopping triggered at epoch {epoch}")
                print(f"Best validation loss: {best_val_loss:.4f}")
                wandb.run.summary["stopped_early"] = True
                wandb.run.summary["stopped_at_epoch"] = epoch
                break

        # Save latest
        model.save_pretrained(str(output_dir / 'latest_checkpoint'))

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")

    wandb.finish()


if __name__ == '__main__':
    main()
