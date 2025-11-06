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
    batch_all_objects = []  # NEW: Store all object coordinates for nearest-neighbor loss

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

        # NEW: Store all object coordinates (normalized) for nearest-neighbor loss
        all_obj_coords = []
        for pt in points_list:
            x_norm, y_norm = normalize_coordinates(pt, img_pil.size)
            all_obj_coords.append([x_norm, y_norm])
        all_obj_tensor = torch.tensor(all_obj_coords, dtype=torch.bfloat16, device=device) if all_obj_coords else torch.empty((0, 2), dtype=torch.bfloat16, device=device)
        batch_all_objects.append(all_obj_tensor)

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
        return batch_marked_images, batch_num_marked, batch_categories, None, None, gt_done, batch_all_objects
    elif mode == 'regression':
        gt_x = torch.tensor(batch_gt_x, dtype=torch.bfloat16, device=device)
        gt_y = torch.tensor(batch_gt_y, dtype=torch.bfloat16, device=device)
        return batch_marked_images, batch_num_marked, batch_categories, gt_x, gt_y, None, batch_all_objects
    else:
        gt_x = torch.tensor(batch_gt_x, dtype=torch.bfloat16, device=device)
        gt_y = torch.tensor(batch_gt_y, dtype=torch.bfloat16, device=device)
        gt_done = torch.tensor(batch_gt_done, dtype=torch.bfloat16, device=device)
        return batch_marked_images, batch_num_marked, batch_categories, gt_x, gt_y, gt_done, batch_all_objects


def train_epoch_dual_task(model, train_loader, optimizer_coord, optimizer_done, marker, device, epoch, max_iters=None):
    """
    Train for one epoch with DUAL-TASK setup:
    - Task 1: Coordinate prediction (separate backward pass)
    - Task 2: Done signal (separate backward pass)
    Both tasks trained every iteration.
    """
    model.train()
    total_loss_coord = 0
    total_loss_done = 0
    total_loss_x = 0
    total_loss_y = 0
    num_batches = 0

    total_iters = len(train_loader) if max_iters is None else min(max_iters, len(train_loader))
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", total=total_iters)

    for batch_idx, batch in enumerate(pbar):
        if max_iters is not None and batch_idx >= max_iters:
            break

        # DUAL-TASK: Train both coordinate and done tasks each iteration
        # We do this by running TWO forward/backward passes with different ground truths

        category = batch[0][2].get('object_type', 'objects') if len(batch) > 0 and isinstance(batch[0][2], dict) else 'objects'

        # ========== TASK 1: COORDINATE PREDICTION ==========
        images_coord, num_marked_coord, categories_coord, gt_x, gt_y, _, all_objects_coord = prepare_batch(
            batch, marker, device, mode='regression'
        )

        optimizer_coord.zero_grad()

        outputs_coord = model.forward(
            images=images_coord,
            num_marked=num_marked_coord,
            category=category,
            gt_x=gt_x,
            gt_y=gt_y,
            gt_done=None,  # No done signal for coord task
            all_objects=all_objects_coord
        )

        loss_coord = outputs_coord['loss']
        loss_coord.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.model.parameters()) + list(model.x_head.parameters()) + list(model.y_head.parameters()),
            max_norm=1.0
        )
        optimizer_coord.step()

        # ========== TASK 2: DONE SIGNAL ==========
        images_done, num_marked_done, categories_done, _, _, gt_done, all_objects_done = prepare_batch(
            batch, marker, device, mode='classification'
        )

        optimizer_done.zero_grad()

        outputs_done = model.forward(
            images=images_done,
            num_marked=num_marked_done,
            category=category,
            gt_x=None,  # No coordinates for done task
            gt_y=None,
            gt_done=gt_done,
            all_objects=all_objects_done
        )

        loss_done = outputs_done['loss']
        loss_done.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.model.parameters()) + list(model.done_head.parameters()),
            max_norm=1.0
        )
        optimizer_done.step()

        # Debug gradient flow for first epoch
        if epoch == 0 and batch_idx == 0:
            x_head_grad = sum(p.grad.abs().sum().item() for p in model.x_head.parameters() if p.grad is not None)
            y_head_grad = sum(p.grad.abs().sum().item() for p in model.y_head.parameters() if p.grad is not None)
            done_head_grad = sum(p.grad.abs().sum().item() for p in model.done_head.parameters() if p.grad is not None)
            print(f"\n[Gradient Check - Dual Task]")
            print(f"  x_head: {x_head_grad:.4f}")
            print(f"  y_head: {y_head_grad:.4f}")
            print(f"  done_head: {done_head_grad:.4f}")

        # Track metrics
        loss_coord_val = loss_coord.item()
        loss_done_val = loss_done.item()
        loss_x_val = outputs_coord['loss_x'].item()
        loss_y_val = outputs_coord['loss_y'].item()

        total_loss_coord += loss_coord_val
        total_loss_done += loss_done_val
        total_loss_x += loss_x_val
        total_loss_y += loss_y_val
        num_batches += 1

        pbar.set_postfix({
            'coord': f'{loss_coord_val:.4f}',
            'done': f'{loss_done_val:.4f}',
            'x': f'{loss_x_val:.3f}',
            'y': f'{loss_y_val:.3f}'
        })

        # Clear references
        del outputs_coord, outputs_done, loss_coord, loss_done
        del images_coord, images_done, gt_x, gt_y, gt_done
        del num_marked_coord, num_marked_done

        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    # Compute averages
    avg_loss_coord = total_loss_coord / num_batches if num_batches > 0 else 0
    avg_loss_done = total_loss_done / num_batches if num_batches > 0 else 0
    avg_loss_x = total_loss_x / num_batches if num_batches > 0 else 0
    avg_loss_y = total_loss_y / num_batches if num_batches > 0 else 0

    return {
        'loss_coord': avg_loss_coord,
        'loss_done': avg_loss_done,
        'loss_x': avg_loss_x,
        'loss_y': avg_loss_y
    }


def validate(model, val_loader, marker, device, epoch, log_images=True, num_images_to_log=16, max_iters=None):
    """
    Validate on validation set with DUAL-TASK evaluation.
    Both coord and done tasks evaluated separately.
    """
    model.eval()
    total_loss_coord = 0
    total_loss_done = 0
    total_loss_x = 0
    total_loss_y = 0
    num_batches = 0

    images_logged = 0
    wandb_images = []

    total_iters = len(val_loader) if max_iters is None else min(max_iters, len(val_loader))

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[Validation]", total=total_iters)

        for batch_idx, batch in enumerate(pbar):
            if max_iters is not None and batch_idx >= max_iters:
                break

            category = batch[0][2].get('object_type', 'objects') if len(batch) > 0 and isinstance(batch[0][2], dict) else 'objects'

            # Evaluate TASK 1: Coordinate prediction
            images_coord, num_marked_coord, categories_coord, gt_x, gt_y, _, all_objects_coord = prepare_batch(
                batch, marker, device, mode='regression'
            )

            outputs_coord = model.forward(
                images=images_coord,
                num_marked=num_marked_coord,
                category=category,
                gt_x=gt_x,
                gt_y=gt_y,
                gt_done=None,
                all_objects=all_objects_coord
            )

            # Evaluate TASK 2: Done signal
            images_done, num_marked_done, categories_done, _, _, gt_done, all_objects_done = prepare_batch(
                batch, marker, device, mode='classification'
            )

            outputs_done = model.forward(
                images=images_done,
                num_marked=num_marked_done,
                category=category,
                gt_x=None,
                gt_y=None,
                gt_done=gt_done,
                all_objects=all_objects_done
            )

            loss_coord_val = outputs_coord['loss'].item()
            loss_x_val = outputs_coord['loss_x'].item()
            loss_y_val = outputs_coord['loss_y'].item()
            loss_done_val = outputs_done['loss'].item()

            total_loss_coord += loss_coord_val
            total_loss_x += loss_x_val
            total_loss_y += loss_y_val
            total_loss_done += loss_done_val
            num_batches += 1

            pbar.set_postfix({
                'coord': f'{loss_coord_val:.4f}',
                'done': f'{loss_done_val:.4f}',
                'x': f'{loss_x_val:.3f}',
                'y': f'{loss_y_val:.3f}'
            })

            # Log images (using coord task outputs for visualization)
            if log_images and images_logged < num_images_to_log:
                import cv2
                for i in range(len(images_coord)):
                    if images_logged >= num_images_to_log:
                        break

                    input_img = images_coord[i]
                    cat = categories_coord[i] if i < len(categories_coord) else "objects"

                    pred_x = outputs_coord['x'][i].item()
                    pred_y = outputs_coord['y'][i].item()
                    pred_done = outputs_done['done'][i].item() if i < len(outputs_done['done']) else 0.0

                    true_x = gt_x[i].item()
                    true_y = gt_y[i].item()

                    # CRITICAL FIX: Compute true_done based on COORD image (num_marked_coord)
                    # NOT from the separate done batch!
                    if i < len(all_objects_coord):
                        total_objects = len(all_objects_coord[i])
                        marked_in_image = num_marked_coord[i]
                        true_done = 1.0 if marked_in_image >= total_objects else 0.0
                    else:
                        true_done = 0.0

                    # NEW: If using nearest-neighbor, compute actual target used for loss
                    actual_target_x = true_x
                    actual_target_y = true_y
                    if model.use_nearest_neighbor_loss and all_objects_coord is not None and true_done < 0.5:
                        # Recompute which object was actually matched
                        if isinstance(all_objects_coord, list):
                            objs = all_objects_coord[i]
                        else:
                            objs = all_objects_coord

                        unmarked = objs[num_marked_coord[i]:]
                        if len(unmarked) > 0:
                            # Find nearest to prediction
                            pred_x_tensor = torch.tensor(pred_x, dtype=torch.bfloat16, device=objs.device)
                            pred_y_tensor = torch.tensor(pred_y, dtype=torch.bfloat16, device=objs.device)
                            distances = torch.abs(unmarked[:, 0] - pred_x_tensor) + torch.abs(unmarked[:, 1] - pred_y_tensor)
                            nearest_idx = torch.argmin(distances)
                            actual_target_x = unmarked[nearest_idx, 0].item()
                            actual_target_y = unmarked[nearest_idx, 1].item()

                    if isinstance(input_img, torch.Tensor):
                        input_img_np = input_img.permute(1, 2, 0).cpu().numpy()
                        input_img_np = (input_img_np * 255).astype(np.uint8)
                        input_img = Image.fromarray(input_img_np)

                    input_img_np = np.array(input_img)
                    W, H = input_img.size

                    output_img_np = input_img_np.copy()

                    # Draw TARGET in GREEN (actual target used for loss)
                    if true_done < 0.5:  # Not done, so there IS a target location
                        target_x_pixel = int(((actual_target_x + 1) / 2) * W)
                        target_y_pixel = int(((actual_target_y + 1) / 2) * H)
                        cv2.circle(output_img_np, (target_x_pixel, target_y_pixel), 10, (0, 255, 0), 2)
                        label = "NN" if model.use_nearest_neighbor_loss else "GT"
                        cv2.putText(output_img_np, label, (target_x_pixel - 15, target_y_pixel - 15),
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

                        # Draw line connecting prediction to target if both exist
                        if true_done < 0.5:
                            target_x_pixel = int(((actual_target_x + 1) / 2) * W)
                            target_y_pixel = int(((actual_target_y + 1) / 2) * H)
                            cv2.line(output_img_np, (pred_x_pixel, pred_y_pixel),
                                   (target_x_pixel, target_y_pixel), (255, 255, 0), 1)
                            # Calculate pixel distance
                            dist_pixels = np.sqrt((pred_x_pixel - target_x_pixel)**2 + (pred_y_pixel - target_y_pixel)**2)
                            cv2.putText(output_img_np, f"err={dist_pixels:.0f}px",
                                       ((pred_x_pixel + target_x_pixel)//2, (pred_y_pixel + target_y_pixel)//2 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                    # Add done signals
                    cv2.putText(output_img_np, f"pred_done={pred_done:.2f}", (10, H - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(output_img_np, f"gt_done={true_done:.0f}", (10, H - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    side_by_side = np.concatenate([input_img_np, output_img_np], axis=1)

                    prompt_text = f"{cat} | Marked: {num_marked_coord[i]}"
                    cv2.putText(side_by_side, prompt_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    cv2.putText(side_by_side, "INPUT", (10, H - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(side_by_side, "PREDICTION (w/ Sequential Attention)", (W + 10, H - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Caption shows actual target used (nearest-neighbor if enabled)
                    # Also show total objects to make done signal clear
                    total_objs = len(all_objects_coord[i]) if i < len(all_objects_coord) else 0
                    if model.use_nearest_neighbor_loss and true_done < 0.5:
                        caption = f"{cat} | {num_marked_coord[i]}/{total_objs} marked | Pred:({pred_x:.2f},{pred_y:.2f},done={pred_done:.2f}) | NN:({actual_target_x:.2f},{actual_target_y:.2f},done={true_done:.0f})"
                    else:
                        caption = f"{cat} | {num_marked_coord[i]}/{total_objs} marked | Pred:({pred_x:.2f},{pred_y:.2f},done={pred_done:.2f}) | GT:({true_x:.2f},{true_y:.2f},done={true_done:.0f})"

                    wandb_images.append(wandb.Image(
                        side_by_side,
                        caption=caption
                    ))

                    images_logged += 1

            del outputs_coord, outputs_done, images_coord, images_done, gt_x, gt_y, gt_done
            del num_marked_coord, num_marked_done, categories_coord

            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

    avg_loss_coord = total_loss_coord / num_batches if num_batches > 0 else 0
    avg_loss_x = total_loss_x / num_batches if num_batches > 0 else 0
    avg_loss_y = total_loss_y / num_batches if num_batches > 0 else 0
    avg_loss_done = total_loss_done / num_batches if num_batches > 0 else 0

    torch.cuda.empty_cache()

    return {
        'loss_coord': avg_loss_coord,
        'loss_done': avg_loss_done,
        'loss_x': avg_loss_x,
        'loss_y': avg_loss_y,
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
    parser.add_argument('--use_nearest_neighbor_loss', action='store_true',
                       help='Match predictions to nearest unmarked object instead of ordered')

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
        device=device,
        use_nearest_neighbor_loss=args.use_nearest_neighbor_loss
    )

    if args.use_nearest_neighbor_loss:
        print("✨ Using NEAREST-NEIGHBOR loss: model can pick any unmarked object")

    # Create marker
    marker = VisualMarker(strategy='numbers', alpha=args.marking_alpha)

    # DUAL-TASK SETUP: Separate optimizers for coordinate and done tasks
    print("\n🔧 Setting up DUAL-TASK training:")
    print("  Task 1: Coordinate prediction (x, y)")
    print("  Task 2: Done signal (binary classification)")

    # Optimizer 1: Coordinate prediction (VLM + x_head + y_head)
    optimizer_coord = torch.optim.AdamW([
        {'params': model.model.parameters(), 'lr': args.lr},
        {'params': model.x_head.parameters(), 'lr': args.lr},
        {'params': model.y_head.parameters(), 'lr': args.lr}
    ])

    # Optimizer 2: Done signal (VLM + done_head)
    optimizer_done = torch.optim.AdamW([
        {'params': model.model.parameters(), 'lr': args.lr},
        {'params': model.done_head.parameters(), 'lr': args.lr}
    ])

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Output directory: {output_dir}")
    print(f"Learning rate: {args.lr}")
    print("Using SIMPLIFIED model (direct VLM features, no complex transformations)")

    # Separate tracking for each task
    best_val_loss_coord = float('inf')
    best_val_loss_done = float('inf')
    patience_counter_coord = 0
    patience_counter_done = 0
    early_stop_patience = 5
    print(f"Early stopping enabled with patience={early_stop_patience} (tracked separately per task)")

    for epoch in range(args.epochs):
        # Train with dual-task setup
        train_metrics = train_epoch_dual_task(
            model, train_loader, optimizer_coord, optimizer_done, marker, device, epoch,
            max_iters=args.max_train_iters
        )

        # Log training metrics
        wandb.log({
            "epoch": epoch,
            "train/loss_coord": train_metrics['loss_coord'],
            "train/loss_done": train_metrics['loss_done'],
            "train/loss_x": train_metrics['loss_x'],
            "train/loss_y": train_metrics['loss_y']
        }, step=epoch)

        # Validate
        val_metrics = validate(
            model, val_loader, marker, device, epoch,
            log_images=True, num_images_to_log=16,
            max_iters=args.max_val_iters
        )

        # Log validation metrics
        wandb.log({
            "val/loss_coord": val_metrics['loss_coord'],
            "val/loss_done": val_metrics['loss_done'],
            "val/loss_x": val_metrics['loss_x'],
            "val/loss_y": val_metrics['loss_y']
        }, step=epoch)

        if val_metrics['images']:
            wandb.log({
                "val/predictions": val_metrics['images']
            }, step=epoch)

        print(f"\nEpoch {epoch}:")
        print(f"  Train - coord: {train_metrics['loss_coord']:.4f}, done: {train_metrics['loss_done']:.4f}, x: {train_metrics['loss_x']:.4f}, y: {train_metrics['loss_y']:.4f}")
        print(f"  Val   - coord: {val_metrics['loss_coord']:.4f}, done: {val_metrics['loss_done']:.4f}, x: {val_metrics['loss_x']:.4f}, y: {val_metrics['loss_y']:.4f}")

        # Track best models separately for each task
        improved_coord = False
        improved_done = False

        # TASK 1: Coordinate prediction
        if val_metrics['loss_coord'] < best_val_loss_coord:
            best_val_loss_coord = val_metrics['loss_coord']
            patience_counter_coord = 0
            improved_coord = True
            model.save_pretrained(str(output_dir / 'best_coord_checkpoint'))
            print(f"✅ New best COORD model! Val loss_coord: {val_metrics['loss_coord']:.4f}")
            wandb.run.summary["best_val_loss_coord"] = best_val_loss_coord
            wandb.run.summary["best_coord_epoch"] = epoch
        else:
            patience_counter_coord += 1

        # TASK 2: Done signal
        if val_metrics['loss_done'] < best_val_loss_done:
            best_val_loss_done = val_metrics['loss_done']
            patience_counter_done = 0
            improved_done = True
            model.save_pretrained(str(output_dir / 'best_done_checkpoint'))
            print(f"✅ New best DONE model! Val loss_done: {val_metrics['loss_done']:.4f}")
            wandb.run.summary["best_val_loss_done"] = best_val_loss_done
            wandb.run.summary["best_done_epoch"] = epoch
        else:
            patience_counter_done += 1

        # Early stopping: Stop if BOTH tasks have stopped improving
        if patience_counter_coord >= early_stop_patience and patience_counter_done >= early_stop_patience:
            print(f"\n🛑 Early stopping triggered at epoch {epoch}")
            print(f"  Coord task: no improvement for {patience_counter_coord} epochs (best: {best_val_loss_coord:.4f})")
            print(f"  Done task: no improvement for {patience_counter_done} epochs (best: {best_val_loss_done:.4f})")
            wandb.run.summary["stopped_early"] = True
            wandb.run.summary["stopped_at_epoch"] = epoch
            break
        elif patience_counter_coord < early_stop_patience or patience_counter_done < early_stop_patience:
            if not improved_coord and not improved_done:
                print(f"  Coord: no improvement for {patience_counter_coord}/{early_stop_patience} epochs")
                print(f"  Done: no improvement for {patience_counter_done}/{early_stop_patience} epochs")

        # Save latest
        model.save_pretrained(str(output_dir / 'latest_checkpoint'))

    print("\nTraining complete!")
    print(f"Best coord loss: {best_val_loss_coord:.4f}")
    print(f"Best done loss: {best_val_loss_done:.4f}")
    print(f"Models saved to: {output_dir}")
    print(f"  - best_coord_checkpoint/ (best coordinate prediction)")
    print(f"  - best_done_checkpoint/ (best done signal)")
    print(f"  - latest_checkpoint/ (final epoch)")

    wandb.finish()


if __name__ == '__main__':
    main()
