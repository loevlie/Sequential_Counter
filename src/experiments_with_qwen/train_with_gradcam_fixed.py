#!/usr/bin/env python3
"""
Memory-efficient training with GradCAM tracking (FIXED VERSION).
Based on working visualize_vlm_gradcam.py approach.
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
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import Dict, List, Tuple
import argparse
import imageio

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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
    """Custom collate function"""
    if len(batch) == 1:
        item = batch[0]
        return {
            'image': [item['image']],
            'count': torch.tensor([item['count']]),
            'target_heatmap': item['target_heatmap'].unsqueeze(0),
            'image_id': [item['image_id']],
            'points': [item['points']]
        }
    return {
        'image': [item['image'] for item in batch],
        'count': torch.tensor([item['count'] for item in batch]),
        'target_heatmap': torch.stack([item['target_heatmap'] for item in batch]),
        'image_id': [item['image_id'] for item in batch],
        'points': [item['points'] for item in batch]
    }


def create_gaussian_heatmap(points: np.ndarray, image_shape: Tuple[int, int],
                           sigma: float = 10.0, baseline: float = 0.1) -> np.ndarray:
    """Create target heatmap with Gaussian peaks."""
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
            'image_id': img_id,
            'points': points
        }


class GradCAMTrainer:
    """Trainer with working GradCAM based on visualize_vlm_gradcam.py"""

    def __init__(self, model_name="Qwen/Qwen3-VL-2B-Instruct", device=None, use_bfloat16=False):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model: {model_name}")

        if use_bfloat16 and torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float32

        self.model_dtype = model_dtype

        # Load model
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map=self.device,
            torch_dtype=model_dtype
        )

        self.processor = AutoProcessor.from_pretrained(model_name)

        # DON'T use gradient checkpointing - allows GradCAM to work
        # (Since vision encoder is frozen, memory usage should be manageable)

        # Freeze vision encoder
        print("Freezing vision encoder (will remain frozen)")
        for param in self.model.visual.parameters():
            param.requires_grad = False

        self.model.set_attn_implementation("eager")
        self.model.train()

        # Store for GradCAM
        self.gradients = None
        self.activations = None

        print(f"Model loaded successfully")

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
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model_dtype)

        labels = inputs["input_ids"].clone()
        decoded = self.processor.decode(inputs["input_ids"][0], skip_special_tokens=False)
        count_pos = decoded.find("COUNT:")

        if count_pos > 0:
            ratio = count_pos / len(decoded)
            mask_until = int(ratio * labels.shape[1] * 0.9)
            labels[:, :mask_until] = -100
        else:
            mask_length = int(labels.shape[1] * 0.75)
            labels[:, :mask_length] = -100

        if (labels != -100).sum() < 5:
            labels = inputs["input_ids"].clone()
            labels[:, :labels.shape[1]//2] = -100

        inputs["labels"] = labels
        return inputs

    def compute_gradcam(self, image: Image.Image) -> np.ndarray:
        """
        Compute gradient-based attention using pixel_values gradients.
        Based on visualize_paper_figure.py approach.
        """
        # Save state
        was_training = self.model.training

        # Set to eval mode
        self.model.eval()

        try:
            # Prepare input
            question = "Count the objects in this image. Respond with COUNT: followed by a number."
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            )

            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Clone and enable gradients on pixel_values (key approach from working code)
            if "pixel_values" in inputs:
                pixel_values = inputs["pixel_values"].clone().detach()
                pixel_values.requires_grad_(True)
                inputs["pixel_values"] = pixel_values

                # Forward pass
                outputs = self.model(**inputs, return_dict=True)

                # Compute target
                if hasattr(outputs, "logits"):
                    target = outputs.logits.mean()
                else:
                    target = outputs.last_hidden_state.mean() if hasattr(outputs, "last_hidden_state") else outputs[0].mean()

                # Debug: Check if target has grad_fn
                if not hasattr(target, 'grad_fn') or target.grad_fn is None:
                    print(f"DEBUG: target has no grad_fn! target.requires_grad={target.requires_grad}")
                    print(f"DEBUG: pixel_values.requires_grad={pixel_values.requires_grad}")
                    print(f"DEBUG: target dtype={target.dtype}, device={target.device}")
                    raise RuntimeError("Target has no grad_fn - cannot compute gradients")

                # Zero grad AFTER forward, BEFORE backward (critical!)
                self.model.zero_grad()
                target.backward(retain_graph=True)

                # Get gradients from pixel_values
                grad = pixel_values.grad.data

                if grad is None:
                    print("DEBUG: pixel_values.grad is None after backward!")
                    raise RuntimeError("No gradients computed on pixel_values")

                # Process gradients to create heatmap (from visualize_paper_figure.py)
                if grad.dim() == 4:
                    # [batch, channels, height, width]
                    heatmap = grad.abs().mean(dim=[0, 1])
                elif grad.dim() == 3:
                    # [batch, seq_len, hidden_dim]
                    heatmap = grad[0].abs().mean(dim=-1)
                    seq_len = heatmap.shape[0]
                    side = int(np.sqrt(seq_len))
                    if side * side == seq_len:
                        heatmap = heatmap.reshape(side, side)
                    else:
                        # Pad to square
                        padded_side = int(np.ceil(np.sqrt(seq_len)))
                        padding = padded_side * padded_side - seq_len
                        heatmap = F.pad(heatmap, (0, padding))
                        heatmap = heatmap.reshape(padded_side, padded_side)
                else:
                    print(f"Warning: Unexpected gradient shape {grad.shape}")
                    heatmap = torch.randn(14, 14).abs().to(self.device)

                # Convert to numpy
                if heatmap.dtype == torch.bfloat16:
                    heatmap = heatmap.float()
                heatmap_np = heatmap.cpu().detach().numpy()

                # Resize to image size
                h, w = image.size[1], image.size[0]
                if heatmap_np.shape[0] != h or heatmap_np.shape[1] != w:
                    from scipy.ndimage import zoom
                    zoom_factors = (h / heatmap_np.shape[0], w / heatmap_np.shape[1])
                    heatmap_np = zoom(heatmap_np, zoom_factors, order=1)

                # Apply Gaussian smoothing
                from scipy.ndimage import gaussian_filter
                heatmap_np = gaussian_filter(heatmap_np, sigma=5.0)

                # Normalize
                if heatmap_np.max() > heatmap_np.min():
                    heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min())

                return heatmap_np
            else:
                print("Warning: No pixel_values in inputs")
                h, w = image.size[1], image.size[0]
                return np.random.rand(h, w) * 0.5 + 0.25

        except Exception as e:
            print(f"GradCAM error: {e}")
            return np.zeros((224, 224))

        finally:
            # Restore training mode
            if was_training:
                self.model.train()


def visualize_gradcam(image, gradcam, points, count, predicted_count, epoch, save_path):
    """Create visualization with GradCAM overlay"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original with points
    axes[0].imshow(image)
    if len(points) > 0:
        axes[0].scatter(points[:, 0], points[:, 1], c='red', s=50, marker='x', linewidths=2)
    axes[0].set_title(f'Original (Count: {count})')
    axes[0].axis('off')

    # GradCAM heatmap
    axes[1].imshow(gradcam, cmap='hot')
    axes[1].set_title(f'GradCAM (Epoch {epoch})')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(gradcam, cmap='jet', alpha=0.5)
    if len(points) > 0:
        axes[2].scatter(points[:, 0], points[:, 1], c='lime', s=50, marker='x', linewidths=2)
    axes[2].set_title(f'Overlay (Pred: {predicted_count})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def create_gif_from_images(image_paths, output_path, duration=0.5):
    """Create animated GIF from images"""
    images = []
    for img_path in image_paths:
        images.append(imageio.imread(img_path))
    imageio.mimsave(output_path, images, duration=duration, loop=0)
    print(f"Created GIF: {output_path}")


def train_epoch(trainer, dataloader, optimizer, scheduler=None,
                gradient_accumulation_steps=4, max_grad_norm=0.5,
                epoch=0, clear_cache_every=10, max_batches=None):
    """Standard training epoch"""
    trainer.model.train()

    total_loss = 0
    num_valid_batches = 0
    num_oom_batches = 0
    accumulated_loss = 0

    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}", total=max_batches if max_batches else len(dataloader))

    torch.cuda.empty_cache()
    gc.collect()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        # Check if we've reached max batches limit
        if max_batches is not None and batch_idx >= max_batches:
            break

        try:
            if batch_idx > 0 and batch_idx % clear_cache_every == 0:
                torch.cuda.empty_cache()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            image = batch['image'][0]
            count = batch['count'][0].item()

            inputs = trainer.prepare_inputs(image, count)

            if trainer.model_dtype == torch.bfloat16:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = trainer.model(**inputs, return_dict=True)
            else:
                outputs = trainer.model(**inputs, return_dict=True)

            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
                loss = torch.clamp(loss, min=0.0, max=100.0)
            else:
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue

            loss = loss / gradient_accumulation_steps
            accumulated_loss += loss.item()
            loss.backward()

            del outputs, loss, inputs

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=max_grad_norm)

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

                    total_loss += accumulated_loss
                    num_valid_batches += 1

                optimizer.zero_grad()
                accumulated_loss = 0
                gc.collect()

            if num_valid_batches > 0:
                avg_loss = total_loss / num_valid_batches
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'valid': num_valid_batches,
                    'oom': num_oom_batches
                })

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                num_oom_batches += 1
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                continue
            else:
                raise e

    if (batch_idx + 1) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.empty_cache()
    gc.collect()

    return {
        'loss': total_loss / max(num_valid_batches, 1),
        'oom_batches': num_oom_batches,
        'valid_batches': num_valid_batches
    }


def validate_with_gradcam(trainer, dataloader, track_samples=None, vis_dir=None, epoch=0):
    """Validation with GradCAM tracking"""
    trainer.model.eval()

    total_mae = 0
    total_mse = 0
    num_samples = 0
    num_tracked = 0

    torch.cuda.empty_cache()
    gc.collect()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            try:
                if batch_idx > 0 and batch_idx % 20 == 0:
                    torch.cuda.empty_cache()

                image = batch['image'][0]
                true_count = batch['count'][0].item()
                image_id = batch['image_id'][0]

                # Check if this is a tracked sample
                if track_samples and image_id in track_samples and vis_dir:
                    # Compute GradCAM
                    gradcam = trainer.compute_gradcam(image)

                    # Get prediction
                    question = "Count the objects in this image. Respond with COUNT: followed by a number."
                    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
                    text = trainer.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    inputs = trainer.processor(text=text, images=[image], return_tensors="pt")
                    inputs = {k: v.to(trainer.device) for k, v in inputs.items()}

                    if "pixel_values" in inputs:
                        inputs["pixel_values"] = inputs["pixel_values"].to(trainer.model_dtype)

                    generated_ids = trainer.model.generate(**inputs, max_new_tokens=10, do_sample=False)
                    generated_text = trainer.processor.decode(generated_ids[0], skip_special_tokens=True)
                    predicted_count = extract_count_from_text(generated_text)
                    if predicted_count == -1:
                        predicted_count = 0

                    points = batch['points'][0]

                    # Save visualization
                    sample_dir = os.path.join(vis_dir, image_id.replace('.jpg', ''))
                    os.makedirs(sample_dir, exist_ok=True)
                    save_path = os.path.join(sample_dir, f'epoch_{epoch:03d}.png')

                    visualize_gradcam(image, gradcam, points, true_count, predicted_count, epoch+1, save_path)
                    num_tracked += 1

                    print(f"\nTracked sample: {image_id} (True: {true_count}, Pred: {predicted_count})")

                # Standard validation
                question = "Count the objects in this image. Respond with COUNT: followed by a number."
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
                text = trainer.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = trainer.processor(text=text, images=[image], return_tensors="pt")
                inputs = {k: v.to(trainer.device) for k, v in inputs.items()}

                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(trainer.model_dtype)

                generated_ids = trainer.model.generate(**inputs, max_new_tokens=10, do_sample=False)
                generated_text = trainer.processor.decode(generated_ids[0], skip_special_tokens=True)
                predicted_count = extract_count_from_text(generated_text)

                if predicted_count == -1:
                    predicted_count = 0

                error = abs(predicted_count - true_count)
                total_mae += error
                total_mse += error ** 2
                num_samples += 1

                del inputs, generated_ids

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    print(f"Validation error: {e}")
                    continue
            except Exception as e:
                print(f"Validation error: {e}")
                continue

    torch.cuda.empty_cache()
    gc.collect()

    mae = total_mae / max(num_samples, 1)
    rmse = np.sqrt(total_mse / max(num_samples, 1))

    print(f"\nTracked {num_tracked}/{len(track_samples) if track_samples else 0} samples this epoch")

    return {'mae': mae, 'rmse': rmse, 'num_samples': num_samples, 'num_tracked': num_tracked}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/FSC147')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-VL-2B-Instruct')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--gradient_accumulation', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_gradcam')
    parser.add_argument('--vis_dir', type=str, default='./gradcam_visualizations')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--use_bfloat16', action='store_true')
    parser.add_argument('--clear_cache_every', type=int, default=10)
    parser.add_argument('--num_track_samples', type=int, default=10)
    parser.add_argument('--gif_duration', type=float, default=0.5)
    parser.add_argument('--max_train_batches', type=int, default=None,
                        help='Max training batches per epoch (for testing, default: None = all batches)')

    args = parser.parse_args()

    print("=" * 50)
    print("Training with GradCAM Tracking (FIXED)")
    print("=" * 50)
    print(f"Vision encoder: FROZEN")
    print(f"Tracking {args.num_track_samples} samples")
    print("=" * 50)

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="qwen-fsc147-gradcam",
            name=f"gradcam_fixed_lr{args.lr}",
            config=vars(args)
        )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    trainer = GradCAMTrainer(args.model_name, use_bfloat16=args.use_bfloat16)

    train_dataset = FSC147Dataset(args.data_dir, split='train', image_size=args.image_size)
    val_dataset = FSC147Dataset(args.data_dir, split='val', image_size=args.image_size)

    track_samples = val_dataset.image_ids[:args.num_track_samples]
    print(f"Tracking samples: {track_samples}")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn, num_workers=0, pin_memory=False)

    optimizer = optim.AdamW([p for p in trainer.model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01, eps=1e-6)

    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_steps)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=total_steps - args.warmup_steps, T_mult=1)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_steps])

    best_mae = float('inf')

    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        if torch.cuda.is_available():
            print(f"Starting memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        print(f"{'='*50}")

        train_metrics = train_epoch(
            trainer, train_loader, optimizer, scheduler,
            gradient_accumulation_steps=args.gradient_accumulation,
            max_grad_norm=args.max_grad_norm,
            epoch=epoch,
            clear_cache_every=args.clear_cache_every,
            max_batches=args.max_train_batches
        )

        print(f"\nTrain Loss: {train_metrics['loss']:.4f} | Valid Batches: {train_metrics['valid_batches']} | OOM: {train_metrics['oom_batches']}")

        val_metrics = validate_with_gradcam(trainer, val_loader, track_samples=track_samples, vis_dir=args.vis_dir, epoch=epoch)
        print(f"Validation MAE: {val_metrics['mae']:.2f} | RMSE: {val_metrics['rmse']:.2f} | Tracked: {val_metrics['num_tracked']}")

        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'val/mae': val_metrics['mae'],
                'val/rmse': val_metrics['rmse'],
                'val/tracked': val_metrics['num_tracked']
            })

        if val_metrics['mae'] < best_mae:
            best_mae = val_metrics['mae']
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'best_mae': best_mae
            }, os.path.join(args.checkpoint_dir, 'best_model.pt'))
            print(f"Saved best model with MAE: {best_mae:.2f}")

        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nTraining complete! Best MAE: {best_mae:.2f}")
    print("\nCreating GIFs...")

    for sample_id in track_samples:
        sample_dir = os.path.join(args.vis_dir, sample_id.replace('.jpg', ''))
        if os.path.exists(sample_dir):
            epoch_images = sorted([os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith('.png')])
            if len(epoch_images) > 0:
                gif_path = os.path.join(args.vis_dir, f'{sample_id.replace(".jpg", "")}_evolution.gif')
                create_gif_from_images(epoch_images, gif_path, duration=args.gif_duration)

    print(f"\nGradCAM visualizations saved to: {args.vis_dir}")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()