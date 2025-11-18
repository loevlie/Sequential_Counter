#!/usr/bin/env python3
"""
Simplified FSC147 training with Qwen3-VL - WITH attention loss, NO gradient accumulation
This version focuses on getting the dual loss working correctly.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from PIL import Image
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import argparse
import os
import wandb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


class FSC147Dataset(Dataset):
    def __init__(self, data_dir, split='train', image_size=224):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.images_dir = self.data_dir / 'images_384_VarV2'

        # Load annotations
        with open(self.data_dir / 'annotation_FSC147_384.json', 'r') as f:
            self.annotations = json.load(f)

        # Load split
        with open(self.data_dir / 'Train_Test_Val_FSC_147.json', 'r') as f:
            splits = json.load(f)

        self.image_names = splits[split]
        print(f"Loaded {len(self.image_names)} images for {split} split")

    def create_gaussian_heatmap(self, points, img_size):
        """Create Gaussian heatmap from point annotations"""
        heatmap = np.zeros(img_size, dtype=np.float32)

        if len(points) == 0:
            return heatmap

        # Scale points to image size
        scale_x = img_size[1] / 384
        scale_y = img_size[0] / 384

        for point in points:
            x = int(point[0] * scale_x)
            y = int(point[1] * scale_y)

            # Ensure point is within bounds
            x = max(0, min(x, img_size[1] - 1))
            y = max(0, min(y, img_size[0] - 1))

            heatmap[y, x] = 1.0

        # Apply Gaussian blur
        sigma = max(img_size) / 40  # Adaptive sigma
        heatmap = gaussian_filter(heatmap, sigma=sigma)

        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        # Load image
        img_path = self.images_dir / img_name
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Get annotations
        ann = self.annotations[img_name]
        count = len(ann['points'])

        # Create target heatmap
        target_heatmap = self.create_gaussian_heatmap(
            ann['points'],
            (self.image_size, self.image_size)
        )

        return {
            'image': image,
            'count': count,
            'target_heatmap': torch.FloatTensor(target_heatmap),
            'image_name': img_name
        }

    def __len__(self):
        return len(self.image_names)


class SimplifiedTrainer:
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct", device="cuda"):
        self.device = device

        # Load model with eager attention for gradient computation
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="eager",  # Required for attention gradients
            device_map=self.device
        )

        # Disable gradient checkpointing to allow attention loss
        # self.model.gradient_checkpointing_enable()  # Don't enable this!

        # Initially freeze vision encoder
        for name, param in self.model.named_parameters():
            if 'visual' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.processor = AutoProcessor.from_pretrained(model_name)
        print("Model loaded with eager attention for gradient computation")

    def compute_attention_heatmap(self, model_output, pixel_values, target_size):
        """Compute attention-based heatmap using gradients"""
        # Get the predicted logits
        logits = model_output.logits

        # Use the max logit as the target for gradient computation
        max_logit = logits[0, -1, :].max()

        # Compute gradients with respect to pixel values
        grads = torch.autograd.grad(
            outputs=max_logit,
            inputs=pixel_values,
            retain_graph=True,
            create_graph=False
        )[0]

        # Average gradients across channels to get spatial importance
        # Shape: [batch, channels, height, width] -> [batch, height, width]
        attention_map = grads.abs().mean(dim=1)

        # Resize to target size
        if attention_map.shape[-1] != target_size:
            attention_map = F.interpolate(
                attention_map.unsqueeze(1),
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

        # Normalize
        attention_map = attention_map - attention_map.min()
        if attention_map.max() > 0:
            attention_map = attention_map / attention_map.max()

        return attention_map

    def training_step(self, batch):
        """Single training step with dual loss"""
        images = batch['image']
        true_counts = batch['count'].to(self.device)
        target_heatmaps = batch['target_heatmap'].to(self.device)

        batch_size = len(images)
        total_loss = 0
        count_losses = []
        attention_losses = []

        for i in range(batch_size):
            image = images[i]
            true_count = true_counts[i].item()
            target_heatmap = target_heatmaps[i]

            # Create conversation
            question = "Count the objects in this image. Respond with COUNT: followed by a number."
            answer = f"COUNT: {true_count}"

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
                    "content": answer
                }
            ]

            # Prepare inputs
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            )

            # Move to device
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v
                     for k, v in inputs.items()}

            # Enable gradients for pixel values
            if 'pixel_values' in inputs:
                pixel_values = inputs['pixel_values'].to(torch.float16).requires_grad_(True)
                inputs['pixel_values'] = pixel_values

            # Forward pass
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = self.model(**inputs, labels=inputs['input_ids'])

            # Count loss
            count_loss = outputs.loss
            count_losses.append(count_loss.item())

            # Attention loss
            try:
                attention_heatmap = self.compute_attention_heatmap(
                    outputs, pixel_values, target_heatmap.shape[0]
                )

                attention_loss = F.mse_loss(attention_heatmap[0], target_heatmap)
                attention_losses.append(attention_loss.item())

                # Combine losses
                loss = count_loss + 5.0 * attention_loss  # Weight attention loss
            except Exception as e:
                print(f"Attention computation failed: {e}")
                loss = count_loss
                attention_losses.append(0.0)

            total_loss += loss

        # Return average losses
        avg_loss = total_loss / batch_size
        avg_count_loss = np.mean(count_losses)
        avg_attention_loss = np.mean(attention_losses)

        # Backward pass
        avg_loss.backward()

        return avg_loss.item(), avg_count_loss, avg_attention_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/media/M2SSD/FSC147')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2-VL-2B-Instruct')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_simple')
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="qwen-fsc147-simple",
            name=f"simple_attn_lr{args.lr}",
            config=vars(args)
        )

    # Create datasets
    train_dataset = FSC147Dataset(args.data_dir, split='train', image_size=args.image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )

    # Initialize trainer
    trainer = SimplifiedTrainer(args.model_name)

    # Optimizer
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=args.lr, weight_decay=0.01)

    # Training loop
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        trainer.model.train()
        train_losses = []
        count_losses = []
        attention_losses = []

        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()

            try:
                loss, count_loss, attention_loss = trainer.training_step(batch)

                train_losses.append(loss)
                count_losses.append(count_loss)
                attention_losses.append(attention_loss)

                optimizer.step()

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'count': f'{count_loss:.4f}',
                    'attn': f'{attention_loss:.4f}'
                })

                # Log to wandb
                if args.use_wandb and batch_idx % 10 == 0:
                    wandb.log({
                        'train_loss': loss,
                        'count_loss': count_loss,
                        'attention_loss': attention_loss,
                        'epoch': epoch,
                        'batch': batch_idx
                    })

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        # Save checkpoint
        avg_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1} - Avg loss: {avg_loss:.4f}, Count: {np.mean(count_losses):.4f}, Attn: {np.mean(attention_losses):.4f}")

        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()