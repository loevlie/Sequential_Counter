"""
Train VLM for Sequential Counting using GRPO (Group Relative Policy Optimization)

Uses reward based on counting accuracy:
- Reward = 1/distance if distance > 0
- Reward = 5 if distance = 0 (perfect count)

This encourages the model to predict accurate counts through RL.
"""

import argparse
import torch
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


def compute_reward(predicted_count: int, true_count: int) -> float:
    """
    Compute reward based on counting accuracy.

    Args:
        predicted_count: Model's predicted count
        true_count: Ground truth count

    Returns:
        Reward value:
        - 5.0 if perfect (distance = 0)
        - 1/distance otherwise
    """
    distance = abs(predicted_count - true_count)
    if distance == 0:
        return 5.0
    else:
        return 1.0 / distance


def extract_count_from_response(response: str) -> int:
    """
    Extract the count from model's response.

    The model should output a number, possibly with explanation.
    Examples:
    - "42"
    - "There are 15 objects in the image."
    - "Count: 23"

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


def prepare_prompt(image: Image.Image, category: str = "objects") -> str:
    """
    Create prompt for direct counting task.

    Args:
        image: Image to count objects in
        category: Object category to count

    Returns:
        Formatted prompt string
    """
    prompt = f"""Count the number of {category} in this image.

Output only the count as a single number.

Example responses:
- 42
- 7
- 153

Count:"""

    return prompt



def create_grpo_dataset(raw_dataset, processor, max_samples=None):
    """
    Create a dataset for GRPO training with minimal memory usage.

    Args:
        raw_dataset: The raw dataset
        processor: The processor for tokenization
        max_samples: Maximum number of samples to use (for memory constraints)
    """
    # Limit samples if needed for memory
    if max_samples is not None:
        num_samples = min(len(raw_dataset), max_samples)
    else:
        num_samples = len(raw_dataset)

    # Process samples in smaller batches to avoid memory issues
    batch_size = 100  # Process 100 samples at a time
    all_data = []

    print(f"Preparing GRPO dataset ({num_samples} samples)...")
    for start_idx in tqdm(range(0, num_samples, batch_size)):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_data = []

        for idx in range(start_idx, end_idx):
            img, points_list, meta = raw_dataset[idx]

            # Convert tensor image to PIL if needed
            if isinstance(img, torch.Tensor):
                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
            else:
                img_pil = img

            # Ground truth count
            N = len(points_list)

            # Get object type
            object_type = meta.get('object_type', meta.get('category', 'objects')) if isinstance(meta, dict) else 'objects'

            # Create simple counting prompt
            prompt = prepare_prompt(img_pil, object_type)

            # Format for Qwen2VL - just store the text and PIL image
            # We'll tokenize on the fly during training
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Apply chat template to get the text
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # GRPO expects 'prompt' to be text and 'image' to be PIL
            # It will handle tokenization and processing during generation
            sample = {
                "prompt": text,  # Raw text with chat template applied
                "true_count": N,
                "category": object_type,
                "image": img_pil  # PIL image for processing during generation
            }

            batch_data.append(sample)

        all_data.extend(batch_data)

    # Convert to HuggingFace Dataset
    return Dataset.from_list(all_data)


def build_reward_fn(prompts, completions, **kwargs):
    """
    Compute rewards for GRPO training.

    This is called directly by GRPO with the prompts and completions.
    We need to extract the true counts from kwargs and compute rewards.

    Args:
        prompts: List of prompt texts
        completions: List of generated completions
        **kwargs: Additional arguments including the batch data

    Returns:
        List of reward values (floats)
    """
    # Try to get true counts from kwargs
    true_counts = kwargs.get('true_count', [])

    # If not found, try to extract from the batch
    if not true_counts:
        batch = kwargs.get('batch', {})
        true_counts = batch.get('true_count', [])

    rewards = []

    # GRPO generates multiple completions per prompt
    # completions will be a flat list: [prompt1_comp1, prompt1_comp2, ..., prompt1_compN, prompt2_comp1, ...]
    # We need to figure out which true_count corresponds to each completion

    # Get number of generations per prompt (from trainer config)
    num_generations = 2  # This should match grpo_config.num_generations

    for idx, completion in enumerate(completions):
        # Figure out which prompt this completion belongs to
        prompt_idx = idx // num_generations

        # Get the true count for this prompt
        if isinstance(true_counts, (list, torch.Tensor)):
            if prompt_idx < len(true_counts):
                true_count = true_counts[prompt_idx]
                if isinstance(true_count, torch.Tensor):
                    true_count = true_count.item()
            else:
                true_count = 0
        else:
            true_count = true_counts if isinstance(true_counts, int) else 0

        # Extract predicted count from completion
        predicted_count = extract_count_from_response(completion)

        # Compute reward
        reward = compute_reward(predicted_count, true_count)
        rewards.append(reward)

    return rewards


def main():
    parser = argparse.ArgumentParser(description="Train VLM with GRPO for Sequential Counting")

    # Model args
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2-VL-2B-Instruct')
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)

    # Data args
    parser.add_argument('--dataset', type=str, default='omnicount', choices=['omnicount', 'fsc147'])
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--categories', type=str, nargs='+', default=['Supermarket'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--max_train_samples', type=int, default=None, help='Limit training samples for testing')
    parser.add_argument('--max_val_samples', type=int, default=None, help='Limit validation samples for testing')

    # GRPO training args
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--output_dir', type=str, default='vlm_grpo_model')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.7)

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
    print("Loading VLM model...")
    processor = AutoProcessor.from_pretrained(args.model_name)

    # Patch the processor to ensure correct dtype and disable truncation
    original_process = processor.__call__

    def process_with_fixes(*args, **kwargs):
        """Wrapper to ensure pixel_values are in bfloat16 and disable truncation."""
        # Force disable truncation to avoid token mismatch
        kwargs['truncation'] = False
        # Also remove max_length if present
        kwargs.pop('max_length', None)

        result = original_process(*args, **kwargs)

        # Convert pixel_values to bfloat16
        if 'pixel_values' in result and result['pixel_values'] is not None:
            result['pixel_values'] = result['pixel_values'].to(torch.bfloat16)

        return result

    processor.__call__ = process_with_fixes

    # Also patch the tokenizer to disable truncation
    if hasattr(processor, 'tokenizer'):
        original_tokenizer_call = processor.tokenizer.__call__

        def tokenizer_no_truncation(*args, **kwargs):
            kwargs['truncation'] = False
            kwargs.pop('max_length', None)
            return original_tokenizer_call(*args, **kwargs)

        # processor.tokenizer.__call__ = tokenizer_no_truncation

    # Load model with optional quantization
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,  # Ensure consistent dtype
            device_map="auto"
        )

        # For 4-bit quantization, ensure vision encoder is also in bfloat16
        # Try multiple possible vision encoder attribute names
        for attr_name in ['visual', 'vision_model', 'vision_tower']:
            if hasattr(model, attr_name):
                vision_model = getattr(model, attr_name)
                if vision_model is not None:
                    try:
                        vision_model = vision_model.to(torch.bfloat16)
                        setattr(model, attr_name, vision_model)
                        print(f"Cast {attr_name} to bfloat16")
                    except Exception as e:
                        print(f"Could not cast {attr_name} to bfloat16: {e}")
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    # Apply LoRA
    print("Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Ensure model is in the correct dtype for generation
    # This helps avoid mixed precision issues during GRPO generation
    if not args.load_in_4bit:
        model = model.to(torch.bfloat16)

    # Convert datasets to GRPO format
    print("Converting datasets to GRPO format...")
    grpo_train_dataset = create_grpo_dataset(train_dataset, processor, max_samples=args.max_train_samples)
    grpo_val_dataset = create_grpo_dataset(val_dataset, processor, max_samples=args.max_val_samples)

    print(f"GRPO Train dataset: {len(grpo_train_dataset)} samples")
    print(f"GRPO Val dataset: {len(grpo_val_dataset)} samples")

    # GRPO configuration
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        num_generations=2,  # Generate 2 completions per prompt for comparison
        gradient_accumulation_steps=1,
        warmup_steps=100,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,  # Disable fp16 to avoid mixed precision issues
        report_to="wandb",
        remove_unused_columns=False,
        temperature=args.temperature,
        # Generation config
        # max_new_tokens=args.max_new_tokens,
        # truncation_side="left",  # Truncate from left if needed
        # padding_side="left",  # Pad from left for batch generation
    )

    # GRPO Trainer
    print("Setting up GRPO Trainer...")

    # Setup generation config for GRPO
    from transformers import GenerationConfig
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens*2,
        temperature=args.temperature,
        do_sample=True,
        top_p=0.9,
        top_k=50,
    )

    # Set generation config on the model
    model.generation_config = generation_config

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=processor,
        train_dataset=grpo_train_dataset,
        eval_dataset=grpo_val_dataset,
        reward_funcs=build_reward_fn,
    )

    # Training
    print(f"\nStarting GRPO training for {args.epochs} epochs...")
    print(f"Each prompt will generate {grpo_config.num_generations} completions for comparison")

    trainer.train()

    # Save final model
    print("\nTraining complete!")
    print(f"Saving final model to: {output_dir}")
    trainer.save_model(str(output_dir / 'final_model'))

    wandb.finish()


if __name__ == '__main__':
    main()
