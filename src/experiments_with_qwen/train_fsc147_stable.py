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

# Optional import for wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed, logging disabled")


def extract_count_from_text(text: str) -> int:
    """Extract count from generated text."""
    # Try to find COUNT: pattern
    match = re.search(r'COUNT:\s*(\d+)', text)
    if match:
        return int(match.group(1))
    # Fallback: find any number
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


class StableQwenTrainer:
    """Stable trainer for Qwen3-VL with improved gradient handling"""

    def __init__(self, model_name="Qwen/Qwen3-VL-2B-Instruct", device=None, freeze_vision=True):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model: {model_name}")
        # Load model with memory-efficient settings
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map=self.device,
            dtype=torch.float16 if self.device == "cuda" else torch.float32  # Use float16 to save memory
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()

        # Freeze vision encoder initially for stability
        if freeze_vision:
            print("Freezing vision encoder for stability")
            for param in self.model.visual.parameters():
                param.requires_grad = False

        # Set to eager attention
        self.model.set_attn_implementation("eager")

        print("Model loaded with gradient checkpointing enabled")

    def unfreeze_vision(self):
        """Unfreeze vision encoder after initial training"""
        print("Unfreezing vision encoder")
        for param in self.model.visual.parameters():
            param.requires_grad = True

    def prepare_inputs(self, image: Image.Image, count: int):
        """Prepare inputs with proper format"""
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

        # Create labels (mask everything except the answer)
        labels = inputs["input_ids"].clone()
        # Mask first 80% of tokens (approximately the question part)
        mask_length = int(labels.shape[1] * 0.8)
        labels[:, :mask_length] = -100

        inputs["labels"] = labels
        return inputs

    def compute_stable_gradient_heatmap(self, inputs, clip_value=1.0):
        """Compute gradient heatmap with clipping for stability"""
        pixel_values = inputs["pixel_values"].clone().detach().requires_grad_(True)
        grad_inputs = inputs.copy()
        grad_inputs["pixel_values"] = pixel_values

        # Forward pass
        outputs = self.model(**grad_inputs, return_dict=True)

        if hasattr(outputs, 'loss') and outputs.loss is not None:
            # Clip loss before backward to prevent explosion
            loss = torch.clamp(outputs.loss, max=10.0)
            loss.backward(retain_graph=True)

            if pixel_values.grad is not None:
                # Clip gradients
                grad = pixel_values.grad.abs()
                grad = torch.clamp(grad, max=clip_value)

                # Process gradient
                grad = grad.mean(dim=(0, 1))  # Average over batch and channels
                num_elements = grad.numel()
                side = int(np.sqrt(num_elements))
                if side * side < num_elements:
                    side += 1

                padding = side * side - num_elements
                if padding > 0:
                    grad = torch.nn.functional.pad(grad.flatten(), (0, padding))
                else:
                    grad = grad.flatten()[:side*side]

                heatmap = grad.reshape(side, side)

                # Resize to target size
                heatmap = torch.nn.functional.interpolate(
                    heatmap.unsqueeze(0).unsqueeze(0),
                    size=(14, 14),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()

                # Normalize
                if heatmap.max() > 0:
                    heatmap = heatmap / heatmap.max()

                return heatmap

        return torch.zeros((14, 14), device=self.device)


def train_epoch_stable(trainer: StableQwenTrainer,
                       dataloader: DataLoader,
                       optimizer: optim.Optimizer,
                       scheduler=None,
                       gradient_accumulation_steps: int = 4,
                       max_grad_norm: float = 0.5,
                       attention_weight: float = 0.05) -> Dict:
    """Stable training epoch with improved gradient handling"""
    trainer.model.train()

    total_loss = 0
    total_count_loss = 0
    total_attn_loss = 0
    num_valid_batches = 0
    num_nan_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        try:
            image = batch['image'][0]
            count = batch['count'][0].item()
            target_heatmap = batch['target_heatmap'][0].to(trainer.device)

            # Prepare inputs with teacher forcing
            inputs = trainer.prepare_inputs(image, count)

            # Forward pass with mixed precision disabled for stability
            with torch.cuda.amp.autocast(enabled=False):
                outputs = trainer.model(**inputs, return_dict=True)

            # Count loss
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                count_loss = outputs.loss
                # Clamp loss to prevent explosion
                count_loss = torch.clamp(count_loss, max=10.0)
            else:
                count_loss = torch.tensor(0.0, device=trainer.device)

            # Skip attention loss to save memory - can be re-enabled later
            # For now, focus on getting count loss to work without OOM
            attn_loss = torch.tensor(0.0, device=trainer.device)

            # Uncomment below to enable attention loss once memory is resolved
            # if count_loss.item() < 5.0:
            #     try:
            #         grad_heatmap = trainer.compute_stable_gradient_heatmap(inputs, clip_value=1.0)
            #         attn_loss = torch.nn.functional.mse_loss(grad_heatmap, target_heatmap) * attention_weight
            #         attn_loss = torch.clamp(attn_loss, max=1.0)
            #     except Exception as e:
            #         attn_loss = torch.tensor(0.0, device=trainer.device)
            # else:
            #     attn_loss = torch.tensor(0.0, device=trainer.device)

            # Total loss
            loss = count_loss + attn_loss

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                num_nan_batches += 1
                print(f"Warning: NaN/Inf loss at batch {batch_idx}, skipping...")
                optimizer.zero_grad()  # Clear any accumulated gradients
                continue

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Gradient clipping before accumulation
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

                if has_nan_grad:
                    print(f"Warning: NaN/Inf gradients at batch {batch_idx}, skipping update...")
                    optimizer.zero_grad()
                    num_nan_batches += 1
                else:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()

                    # Track metrics
                    total_loss += loss.item() * gradient_accumulation_steps
                    total_count_loss += count_loss.item()
                    total_attn_loss += attn_loss.item()
                    num_valid_batches += 1

            # Update progress bar
            if num_valid_batches > 0:
                avg_loss = total_loss / num_valid_batches
                avg_count_loss = total_count_loss / num_valid_batches
                avg_attn_loss = total_attn_loss / num_valid_batches
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'count_l': f'{avg_count_loss:.4f}',
                    'attn_l': f'{avg_attn_loss:.4f}',
                    'nan': num_nan_batches
                })

            # Clear cache periodically
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM at batch {batch_idx}, clearing cache...")
                torch.cuda.empty_cache()
                gc.collect()
                optimizer.zero_grad()
                continue
            else:
                raise e

    # Final optimizer step if needed
    if (batch_idx + 1) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    return {
        'loss': total_loss / max(num_valid_batches, 1),
        'count_loss': total_count_loss / max(num_valid_batches, 1),
        'attention_loss': total_attn_loss / max(num_valid_batches, 1),
        'nan_batches': num_nan_batches
    }


def validate_stable(trainer: StableQwenTrainer, dataloader: DataLoader) -> Dict:
    """Stable validation"""
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

                # Generate with low temperature for consistency
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

                # Clear cache periodically
                if num_samples % 20 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Validation error: {e}")
                continue

    mae = total_mae / max(num_samples, 1)
    rmse = np.sqrt(total_mse / max(num_samples, 1))

    return {'mae': mae, 'rmse': rmse, 'num_samples': num_samples}


def main():
    parser = argparse.ArgumentParser(description='Stable Qwen3-VL training on FSC147')
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/FSC147')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-VL-2B-Instruct')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-6)  # Lower LR for stability
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--gradient_accumulation', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--attention_weight', type=float, default=0.05)  # Lower weight initially
    parser.add_argument('--freeze_vision_epochs', type=int, default=1)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_stable')
    parser.add_argument('--use_wandb', action='store_true')

    args = parser.parse_args()

    # Initialize wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="qwen-fsc147-stable",
            name=f"stable_lr{args.lr}_ga{args.gradient_accumulation}",
            config=vars(args)
        )

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize model with frozen vision encoder
    trainer = StableQwenTrainer(args.model_name, freeze_vision=True)

    # Create datasets (smaller subset for testing)
    train_dataset = FSC147Dataset(args.data_dir, split='train', image_size=args.image_size)
    val_dataset = FSC147Dataset(args.data_dir, split='val', image_size=args.image_size)

    # Use smaller subset for initial testing
    train_dataset.image_ids = train_dataset.image_ids[:200]
    val_dataset.image_ids = val_dataset.image_ids[:50]

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    # Optimizer with very low learning rate
    optimizer = optim.AdamW(trainer.model.parameters(), lr=args.lr, weight_decay=0.01)

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_steps)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=total_steps - args.warmup_steps, T_mult=1)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_steps])

    # Training loop
    best_mae = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Unfreeze vision encoder after initial epochs
        if epoch == args.freeze_vision_epochs:
            trainer.unfreeze_vision()
            # Reduce learning rate when unfreezing
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

        # Train
        train_metrics = train_epoch_stable(
            trainer,
            train_loader,
            optimizer,
            scheduler,
            gradient_accumulation_steps=args.gradient_accumulation,
            max_grad_norm=args.max_grad_norm,
            attention_weight=args.attention_weight * (epoch + 1) / args.epochs  # Gradually increase
        )

        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Count Loss: {train_metrics['count_loss']:.4f} | "
              f"Attention Loss: {train_metrics['attention_loss']:.4f} | "
              f"NaN Batches: {train_metrics['nan_batches']}")

        # Validate
        val_metrics = validate_stable(trainer, val_loader)
        print(f"Validation MAE: {val_metrics['mae']:.2f} | RMSE: {val_metrics['rmse']:.2f}")

        # Log to wandb
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/count_loss': train_metrics['count_loss'],
                'train/attention_loss': train_metrics['attention_loss'],
                'train/nan_batches': train_metrics['nan_batches'],
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
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mae': best_mae
            }, checkpoint_path)
            print(f"Saved best model with MAE: {best_mae:.2f}")

        # Clear cache after each epoch
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nTraining complete! Best MAE: {best_mae:.2f}")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()