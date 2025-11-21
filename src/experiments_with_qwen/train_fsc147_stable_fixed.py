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


class FixedStableQwenTrainer:
    """Fixed trainer with robust error handling and proper dtype management"""

    def __init__(self, model_name="Qwen/Qwen3-VL-2B-Instruct", device=None,
                 freeze_vision=True, use_gradient_checkpointing=False, dtype="float32"):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gradient_checkpointing = use_gradient_checkpointing

        print(f"Loading model: {model_name}")
        print(f"Using dtype: {dtype}")

        # CRITICAL FIX: Allow choice of dtype but default to float32 for stability
        if dtype == "float32":
            model_dtype = torch.float32
        elif dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
            print("Using bfloat16 (more stable than float16)")
        else:
            model_dtype = torch.float16
            print("Warning: Using float16 may cause instability")

        # Load model with chosen dtype
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map=self.device,
            torch_dtype=model_dtype
        )

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model_dtype = model_dtype

        # Only enable gradient checkpointing if explicitly requested
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        else:
            print("Gradient checkpointing disabled for stability")

        # Freeze vision encoder initially for stability
        if freeze_vision:
            print("Freezing vision encoder for stability")
            for param in self.model.visual.parameters():
                param.requires_grad = False

        # Set to eager attention
        self.model.set_attn_implementation("eager")

        print(f"Model loaded successfully with dtype {model_dtype}")

    def unfreeze_vision(self):
        """Unfreeze vision encoder after initial training"""
        print("Unfreezing vision encoder")
        for param in self.model.visual.parameters():
            param.requires_grad = True

    def validate_labels(self, labels, input_ids):
        """Validate that labels are properly masked"""
        non_masked = (labels != -100).sum()
        total = labels.numel()

        if non_masked == 0:
            print("ERROR: All tokens are masked! No loss will be computed.")
            return False
        elif non_masked < 5:
            print(f"Warning: Only {non_masked}/{total} tokens are unmasked.")
            return False

        return True

    def prepare_inputs(self, image: Image.Image, count: int):
        """Prepare inputs with robust label masking"""
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

        # Ensure pixel values are in the correct dtype
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model_dtype)

        # CRITICAL FIX: Robust label masking
        labels = inputs["input_ids"].clone()

        # Method 1: Try to find "COUNT:" in the tokenized sequence
        decoded_tokens = [self.processor.decode([token_id]) for token_id in inputs["input_ids"][0]]

        # Find where "COUNT" appears
        count_position = -1
        for i, token_text in enumerate(decoded_tokens):
            if "COUNT" in token_text or "Count" in token_text or "count" in token_text:
                count_position = i
                break

        # If we found COUNT, mask everything before it
        if count_position > 0:
            labels[:, :count_position] = -100
        else:
            # Fallback: Use a conservative masking approach
            # Mask 70% of tokens (leaving 30% for the answer)
            mask_length = int(labels.shape[1] * 0.7)
            labels[:, :mask_length] = -100

        # CRITICAL: Ensure we have tokens to predict
        if not self.validate_labels(labels, inputs["input_ids"]):
            # Emergency fallback: only mask first half
            print("Applying emergency fallback masking")
            labels = inputs["input_ids"].clone()
            labels[:, :labels.shape[1]//2] = -100

        inputs["labels"] = labels

        # Debug logging (occasionally)
        if torch.rand(1).item() < 0.005:  # 0.5% chance
            non_masked = (labels != -100).sum().item()
            print(f"Debug - Label stats: Total tokens: {labels.shape[1]}, Non-masked: {non_masked}")

        return inputs

    def compute_stable_gradient_heatmap(self, inputs, clip_value=1.0):
        """Compute gradient heatmap with clipping for stability"""
        # Skip for now to focus on getting basic training working
        return torch.zeros((14, 14), device=self.device, dtype=self.model_dtype)


def train_epoch_fixed(trainer: FixedStableQwenTrainer,
                     dataloader: DataLoader,
                     optimizer: optim.Optimizer,
                     scheduler=None,
                     gradient_accumulation_steps: int = 4,
                     max_grad_norm: float = 0.5,
                     attention_weight: float = 0.05,
                     epoch: int = 0) -> Dict:
    """Fixed training epoch with improved error handling"""
    trainer.model.train()

    total_loss = 0
    total_count_loss = 0
    total_attn_loss = 0
    num_valid_batches = 0
    num_nan_batches = 0
    num_zero_loss_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        try:
            image = batch['image'][0]
            count = batch['count'][0].item()
            target_heatmap = batch['target_heatmap'][0].to(trainer.device)

            # Prepare inputs with teacher forcing
            inputs = trainer.prepare_inputs(image, count)

            # Forward pass - use appropriate precision
            if trainer.model_dtype == torch.float32:
                # No autocast for float32
                outputs = trainer.model(**inputs, return_dict=True)
            elif trainer.model_dtype == torch.bfloat16:
                # Use bfloat16 autocast
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = trainer.model(**inputs, return_dict=True)
            else:
                # Use float16 autocast
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = trainer.model(**inputs, return_dict=True)

            # Count loss
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                count_loss = outputs.loss

                # CRITICAL: Check if loss is valid
                if count_loss.item() == 0:
                    num_zero_loss_batches += 1
                    if num_zero_loss_batches % 10 == 0:
                        print(f"Warning: Zero loss encountered {num_zero_loss_batches} times")

                # Clamp loss to prevent explosion
                count_loss = torch.clamp(count_loss, min=0.0, max=100.0)
            else:
                print(f"Warning: No loss returned for batch {batch_idx}")
                continue

            # For now, skip attention loss to focus on getting basic training working
            attn_loss = torch.tensor(0.0, device=trainer.device)

            # Total loss
            loss = count_loss + attn_loss

            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                num_nan_batches += 1
                print(f"Warning: NaN/Inf loss at batch {batch_idx}, loss value: {loss.item()}")

                # Clear gradients and skip
                optimizer.zero_grad()

                # Log some debug info
                if num_nan_batches <= 3:  # Only for first few NaNs
                    print(f"  Count: {count}, Loss components: count={count_loss.item()}")
                    print(f"  Input IDs shape: {inputs['input_ids'].shape}")
                    print(f"  Labels shape: {inputs['labels'].shape}")
                    non_masked = (inputs['labels'] != -100).sum().item()
                    print(f"  Non-masked tokens: {non_masked}")

                continue

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation and optimization
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=max_grad_norm)

                # Check for NaN gradients
                has_nan_grad = False
                for name, param in trainer.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            print(f"NaN/Inf gradient in {name}")
                            break

                if has_nan_grad:
                    print(f"Warning: NaN/Inf gradients at batch {batch_idx}, skipping update")
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
                    'nan': num_nan_batches,
                    'valid': num_valid_batches
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
                print(f"Runtime error at batch {batch_idx}: {e}")
                raise e
        except Exception as e:
            print(f"Unexpected error at batch {batch_idx}: {e}")
            continue

    # Final optimizer step if needed
    remaining_grads = (batch_idx + 1) % gradient_accumulation_steps
    if remaining_grads != 0:
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    # Print summary
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Valid batches: {num_valid_batches}/{len(dataloader)}")
    print(f"  NaN/Inf batches: {num_nan_batches}")
    print(f"  Zero loss batches: {num_zero_loss_batches}")

    return {
        'loss': total_loss / max(num_valid_batches, 1),
        'count_loss': total_count_loss / max(num_valid_batches, 1),
        'attention_loss': total_attn_loss / max(num_valid_batches, 1),
        'nan_batches': num_nan_batches,
        'valid_batches': num_valid_batches
    }


def validate_fixed(trainer: FixedStableQwenTrainer, dataloader: DataLoader) -> Dict:
    """Fixed validation"""
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

                # Ensure correct dtype
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(trainer.model_dtype)

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
    parser = argparse.ArgumentParser(description='Fixed Stable Qwen3-VL training on FSC147')
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/FSC147')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-VL-2B-Instruct')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-7)  # Even lower LR for stability
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--gradient_accumulation', type=int, default=4)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--attention_weight', type=float, default=0.0)  # Start with 0
    parser.add_argument('--freeze_vision_epochs', type=int, default=2)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_fixed')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--use_gradient_checkpointing', action='store_true')
    parser.add_argument('--debug_subset', type=int, default=0, help='Use subset of data for debugging')

    args = parser.parse_args()

    # Initialize wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="qwen-fsc147-fixed",
            name=f"fixed_{args.dtype}_lr{args.lr}_ga{args.gradient_accumulation}",
            config=vars(args)
        )

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize model with chosen dtype
    trainer = FixedStableQwenTrainer(
        args.model_name,
        freeze_vision=True,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        dtype=args.dtype
    )

    # Create datasets
    train_dataset = FSC147Dataset(args.data_dir, split='train', image_size=args.image_size)
    val_dataset = FSC147Dataset(args.data_dir, split='val', image_size=args.image_size)

    # Use subset for debugging if specified
    if args.debug_subset > 0:
        print(f"Using debug subset of {args.debug_subset} samples")
        train_dataset.image_ids = train_dataset.image_ids[:args.debug_subset]
        val_dataset.image_ids = val_dataset.image_ids[:min(50, args.debug_subset//4)]

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    # Optimizer with very low learning rate
    optimizer = optim.AdamW(
        [p for p in trainer.model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
        eps=1e-6  # Slightly higher epsilon for stability
    )

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_steps)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=total_steps - args.warmup_steps, T_mult=1)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_steps])

    # Training loop
    best_mae = float('inf')

    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*50}")

        # Unfreeze vision encoder after initial epochs
        if epoch == args.freeze_vision_epochs:
            trainer.unfreeze_vision()
            # Recreate optimizer with all parameters
            optimizer = optim.AdamW(
                trainer.model.parameters(),
                lr=args.lr * 0.5,  # Lower LR when unfreezing
                weight_decay=0.01,
                eps=1e-6
            )
            print(f"Vision encoder unfrozen, LR reduced to {args.lr * 0.5}")

        # Train
        train_metrics = train_epoch_fixed(
            trainer,
            train_loader,
            optimizer,
            scheduler,
            gradient_accumulation_steps=args.gradient_accumulation,
            max_grad_norm=args.max_grad_norm,
            attention_weight=args.attention_weight * min(1.0, (epoch + 1) / args.epochs),
            epoch=epoch
        )

        print(f"\nTrain Loss: {train_metrics['loss']:.4f} | "
              f"Count Loss: {train_metrics['count_loss']:.4f} | "
              f"Valid Batches: {train_metrics['valid_batches']}")

        # Validate every epoch
        val_metrics = validate_fixed(trainer, val_loader)
        print(f"Validation MAE: {val_metrics['mae']:.2f} | RMSE: {val_metrics['rmse']:.2f}")

        # Log to wandb
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/count_loss': train_metrics['count_loss'],
                'train/attention_loss': train_metrics['attention_loss'],
                'train/nan_batches': train_metrics['nan_batches'],
                'train/valid_batches': train_metrics['valid_batches'],
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

        # Save periodic checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'mae': val_metrics['mae']
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch+1}")

        # Clear cache after each epoch
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nTraining complete! Best MAE: {best_mae:.2f}")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()