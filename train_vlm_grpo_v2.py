"""
Train VLM for Sequential Counting using GRPO (Group Relative Policy Optimization)
Based on HuggingFace Cookbook example for VLM GRPO training

This implementation follows the official cookbook approach with proper:
- Dataset formatting (prompt + image)
- Reward function signatures
- Model and processor setup
"""

import argparse
import torch
import re
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets import Dataset
import wandb

from trl import GRPOConfig, GRPOTrainer
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model

from dataset import OmniCountDataset
from dataset_fsc147 import FSC147Dataset


def extract_count_from_response(response: str) -> int:
    """
    Extract the count from model's response.

    The model should output a number, possibly with explanation.
    We try to extract any integer from the response.
    """
    import re

    # Try to find numbers in the response
    numbers = re.findall(r'\b\d+\b', response)

    if numbers:
        # Return the first number found (usually the count)
        return int(numbers[0])

    # If we can't parse, return 0
    return 0


def counting_accuracy_reward(completions, **kwargs):
    """
    Reward function that checks if the completion matches the ground truth count.

    This follows the HuggingFace cookbook format where the reward function
    receives completions and kwargs containing additional data.

    Args:
        completions: List of generated text responses
        **kwargs: Additional arguments including 'solution' with ground truth counts

    Returns:
        List of reward values (floats)
    """
    # Extract ground truth counts from kwargs
    # In the cookbook, they access solution directly from kwargs
    true_counts = kwargs.get('true_count', kwargs.get('solution', []))

    rewards = []

    for completion, true_count in zip(completions, true_counts):
        # Extract predicted count from completion
        predicted_count = extract_count_from_response(completion)

        # Convert true_count to int if it's a tensor
        if isinstance(true_count, torch.Tensor):
            true_count = true_count.item()
        elif isinstance(true_count, str):
            true_count = int(true_count) if true_count.isdigit() else 0

        # Calculate reward based on distance
        distance = abs(predicted_count - true_count)
        if distance == 0:
            reward = 5.0  # Perfect match bonus
        elif distance <= 2:
            reward = 2.0  # Close match
        elif distance <= 5:
            reward = 1.0 / distance
        else:
            reward = 0.1 / distance

        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """
    Optional: Reward function that checks if the completion has a specific format.
    For now, we just give a small bonus for numeric-only responses.
    """
    rewards = []
    for completion in completions:
        # Check if response starts with a number
        if completion.strip() and completion.strip()[0].isdigit():
            rewards.append(1.0)
        else:
            rewards.append(0.5)

    return rewards


def create_grpo_dataset(raw_dataset, processor, max_samples=None):
    """
    Convert our counting dataset into format required by GRPOTrainer.

    Following the cookbook example, we need:
    - 'prompt': The formatted prompt text
    - 'image': The PIL image
    - 'solution' or custom fields for reward function
    """
    # Limit samples if specified
    if max_samples is not None:
        num_samples = min(len(raw_dataset), max_samples)
    else:
        num_samples = len(raw_dataset)

    data = []

    print(f"Preparing GRPO dataset ({num_samples} samples)...")
    for idx in tqdm(range(num_samples)):
        img, points_list, meta = raw_dataset[idx]

        # Convert tensor image to PIL if needed
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
        else:
            img_pil = img

        # Ground truth count
        true_count = len(points_list)

        # Get object type
        object_type = meta.get('object_type', meta.get('category', 'objects')) if isinstance(meta, dict) else 'objects'

        # Create conversation following cookbook format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Count the number of {object_type} in this image. Provide only the count as a number."}
                ]
            }
        ]

        # Apply chat template (without tokenization)
        prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False  # Important: don't tokenize here
        )

        # Prepare the sample following cookbook format
        sample = {
            "prompt": prompt,  # Text prompt
            "image": img_pil,  # PIL image
            "true_count": true_count,  # For reward function
            "solution": str(true_count)  # Alternative field name
        }

        data.append(sample)

    # Convert to HuggingFace Dataset
    return Dataset.from_list(data)


def main():
    parser = argparse.ArgumentParser(description="Train VLM with GRPO for Sequential Counting")

    # Model args
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2-VL-2B-Instruct')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)

    # Data args
    parser.add_argument('--dataset', type=str, default='fsc147', choices=['omnicount', 'fsc147'])
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--categories', type=str, nargs='+', default=['Supermarket'])
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_val_samples', type=int, default=None)

    # GRPO training args
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--output_dir', type=str, default='vlm_grpo_model')
    parser.add_argument('--num_generations', type=int, default=2)
    parser.add_argument('--max_completion_length', type=int, default=128)
    parser.add_argument('--max_prompt_length', type=int, default=2048)

    # W&B args
    parser.add_argument('--wandb_project', type=str, default='sequential-counting')
    parser.add_argument('--wandb_run_name', type=str, default=None)

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
        config=vars(args)
    )

    # Load datasets
    print(f"Loading {args.dataset.upper()} dataset...")

    if args.dataset == 'fsc147':
        train_dataset = FSC147Dataset(
            dataset_root=args.data_root,
            split='train',
            spatial_order='reading_order',
            image_size=None
        )
        val_dataset = FSC147Dataset(
            dataset_root=args.data_root,
            split='val',
            spatial_order='reading_order',
            image_size=None
        )
    else:  # omnicount
        train_dataset = OmniCountDataset(
            dataset_root=args.data_root,
            categories=args.categories,
            split='train',
            spatial_order='reading_order',
            image_size=None
        )
        val_dataset = OmniCountDataset(
            dataset_root=args.data_root,
            categories=args.categories,
            split='valid',
            spatial_order='reading_order',
            image_size=None
        )

    print(f"Train: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images")

    # Load model and processor
    print("Loading VLM model and processor...")

    # Load processor with padding_side="left" as per cookbook
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        use_fast=True,
        padding_side="left"  # Important for GRPO
    )

    # Load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Apply LoRA
    print("Applying LoRA adapters...")
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Convert datasets to GRPO format
    print("Converting datasets to GRPO format...")
    grpo_train_dataset = create_grpo_dataset(
        train_dataset, processor, max_samples=args.max_train_samples
    )
    grpo_val_dataset = create_grpo_dataset(
        val_dataset, processor, max_samples=args.max_val_samples
    )

    print(f"GRPO Train dataset: {len(grpo_train_dataset)} samples")
    print(f"GRPO Val dataset: {len(grpo_val_dataset)} samples")

    # Configure GRPO training following cookbook
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        remove_unused_columns=False,  # Important: to access custom fields in reward function
        bf16=True,

        # GRPO specific parameters
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,

        # Logging and saving
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to=["wandb"],

        # Other settings
        warmup_steps=50,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
    )

    # Create GRPO Trainer following cookbook format
    print("Setting up GRPO Trainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=grpo_train_dataset,
        eval_dataset=grpo_val_dataset,
        reward_funcs=[counting_accuracy_reward, format_reward],  # Pass reward functions as list
    )

    # Training
    print(f"\nStarting GRPO training for {args.epochs} epochs...")
    print(f"Each prompt will generate {args.num_generations} completions for comparison")

    trainer.train()

    # Save final model
    print("\nTraining complete!")
    print(f"Saving final model to: {output_dir}")
    trainer.save_model(str(output_dir / 'final_model'))

    # Push to hub if desired
    # trainer.push_to_hub()

    wandb.finish()


if __name__ == '__main__':
    main()