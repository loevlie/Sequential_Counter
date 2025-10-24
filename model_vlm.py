#!/usr/bin/env python3
"""
Vision-Language Model for Sequential Counting (Qwen3-VL)

Uses Qwen3-VL-4B-Thinking with LoRA fine-tuning for efficient training.
Designed for point prediction in sequential object counting tasks.

Advantages of Qwen3-VL-4B-Thinking:
- Advanced spatial perception with 2D grounding
- 15-20% better accuracy on spatial reasoning tasks
- Smaller (4B vs 11B) - faster training, less memory
- Step-by-step thinking/reasoning capabilities
- Native support for object positioning and coordinates
"""

import torch
import torch.nn as nn
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
import re
from typing import Dict, List, Optional, Tuple, Union


class VLMCountingModel(nn.Module):
    """
    Qwen3-VL-4B-Thinking model wrapper for sequential object counting.

    Outputs point predictions in format: (x, y) or "done"
    Uses LoRA for efficient fine-tuning.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Thinking",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_4bit: bool = True,
        device: str = "cuda"
    ):
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.use_lora = use_lora

        # Load processor
        print(f"Loading processor from {model_name}...")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Quantization config for efficient training
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )

        # Load base model
        print(f"Loading base model {model_name}...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        # Apply LoRA
        if use_lora:
            print("Applying LoRA adapters...")
            self.model = prepare_model_for_kbit_training(self.model)

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )

            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        self.model.train()

    def create_prompt(self, num_marked: int) -> List[Dict]:
        """
        Create prompt for the VLM to predict next object location.

        Uses Qwen3-VL chat format with thinking mode.

        Args:
            num_marked: Number of objects already marked in the image

        Returns:
            Messages list for Qwen3-VL processor
        """
        if num_marked == 0:
            marked_text = "No objects are marked yet."
        elif num_marked == 1:
            marked_text = "1 object is already marked with a red halo."
        else:
            marked_text = f"{num_marked} objects are already marked with red halos."

        # Return messages format for processor
        messages = [
            {
                "role": "system",
                "content": "You are a vision assistant that counts objects systematically."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"""This image shows objects to be counted. {marked_text}

Task: Identify the next unmarked object and output its pixel coordinates.

Rules:
1. If there are unmarked objects remaining, output ONLY: (x, y) where x and y are pixel coordinates
2. If ALL objects are marked, output ONLY: done
3. Count objects systematically from top-to-bottom, left-to-right (reading order)
4. Output format must be exactly: (x, y) or done - nothing else

Think step by step, then provide your answer."""}
                ]
            }
        ]
        return messages

    def parse_output(self, text: str, image_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """
        Parse VLM output text to extract coordinates or done signal.

        Args:
            text: Generated text from VLM
            image_size: (width, height) of the image

        Returns:
            Dictionary with:
                - x: torch.Tensor (normalized to [-1, 1])
                - y: torch.Tensor (normalized to [-1, 1])
                - is_done: torch.Tensor (1.0 if done, 0.0 otherwise)
        """
        text = text.strip().lower()

        # Check for done signal
        if "done" in text or "all objects" in text or "complete" in text or "no more" in text:
            return {
                'x': torch.tensor(-1.0),
                'y': torch.tensor(-1.0),
                'is_done': torch.tensor(1.0)
            }

        # Extract coordinates - try multiple patterns
        patterns = [
            r'\((\d+),\s*(\d+)\)',  # (x, y)
            r'\((\d+)\s+(\d+)\)',    # (x y)
            r'(\d+),\s*(\d+)',       # x, y
            r'x[:\s=]*(\d+).*y[:\s=]*(\d+)',  # x: 123 y: 456
            r'coordinate[s]?[:\s]+(\d+)[,\s]+(\d+)',  # coordinates: x, y
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    x_pixel = int(match.group(1))
                    y_pixel = int(match.group(2))

                    # Normalize to [-1, 1]
                    W, H = image_size
                    x_norm = (x_pixel / W) * 2 - 1
                    y_norm = (y_pixel / H) * 2 - 1

                    # Clamp to valid range
                    x_norm = max(-1.0, min(1.0, x_norm))
                    y_norm = max(-1.0, min(1.0, y_norm))

                    return {
                        'x': torch.tensor(x_norm),
                        'y': torch.tensor(y_norm),
                        'is_done': torch.tensor(0.0)
                    }
                except (ValueError, IndexError):
                    continue

        # If parsing fails, return invalid signal (treat as done to avoid errors)
        print(f"Warning: Failed to parse output: '{text}'. Treating as done.")
        return {
            'x': torch.tensor(-1.0),
            'y': torch.tensor(-1.0),
            'is_done': torch.tensor(1.0)
        }

    def forward(
        self,
        images: Union[Image.Image, List[Image.Image]],
        num_marked: Union[int, List[int]],
        max_new_tokens: int = 128,
        temperature: float = 0.6,
        top_p: float = 0.9
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: generate next point prediction.

        Args:
            images: PIL Image or list of PIL Images (marked with halos)
            num_marked: Number of objects already marked (or list for batch)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Dictionary with x, y, is_done tensors
        """
        # Handle single image
        if isinstance(images, Image.Image):
            images = [images]
            num_marked = [num_marked]

        batch_size = len(images)

        # Create conversation format for Qwen3-VL
        conversations = []
        for img, n_marked in zip(images, num_marked):
            prompt_text = self.create_prompt(n_marked)

            # Qwen3-VL format: list of messages with image and text
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            conversations.append(conversation)

        # Process inputs
        texts = [self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                 for conv in conversations]

        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Generate
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )

        # Decode outputs
        generated_texts = self.processor.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Parse predictions
        predictions = []
        for i, text in enumerate(generated_texts):
            pred = self.parse_output(text, images[i].size)
            predictions.append(pred)

        # Stack into batch tensors
        result = {
            'x': torch.stack([p['x'] for p in predictions]).to(self.device),
            'y': torch.stack([p['y'] for p in predictions]).to(self.device),
            'is_done': torch.stack([p['is_done'] for p in predictions]).to(self.device)
        }

        return result

    def forward_with_target(
        self,
        images: Union[Image.Image, List[Image.Image]],
        prompts: Union[List[Dict], List[List[Dict]]],
        target_texts: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Forward pass with target text for training (teacher forcing).

        Args:
            images: PIL Image or list of PIL Images
            prompts: Messages list or list of messages lists
            target_texts: Target answer strings (e.g., "(245, 367)" or "done")

        Returns:
            Loss tensor
        """
        # Handle single image
        if isinstance(images, Image.Image):
            images = [images]
            prompts = [prompts]
            target_texts = [target_texts]

        # Build conversation with assistant response for each sample
        conversations = []
        for messages, target in zip(prompts, target_texts):
            # Add assistant response to messages
            conv = messages + [{"role": "assistant", "content": target}]
            conversations.append(conv)

        # Apply chat template to get text with proper image tokens
        texts = [
            self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            for conv in conversations
        ]

        # Process inputs
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Create labels for loss computation
        # Labels should be input_ids, but with -100 for tokens we don't want to compute loss on
        labels = inputs["input_ids"].clone()

        # For each sample, we need to mask the prompt tokens (only compute loss on assistant response)
        # We'll tokenize just the prompts to find where the assistant response starts
        for i, (messages, target) in enumerate(zip(prompts, target_texts)):
            # Get prompt without assistant response
            prompt_only = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_tokens = self.processor(
                text=prompt_only,
                images=images[i],
                return_tensors="pt"
            )["input_ids"].to(self.device)

            # Mask prompt tokens (set to -100 so they're ignored in loss)
            prompt_len = prompt_tokens.shape[1]
            labels[i, :prompt_len] = -100

        # Add labels to inputs
        inputs["labels"] = labels

        # Forward pass
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model(**inputs)
            loss = outputs.loss

        return loss

    def save_pretrained(self, output_dir: str):
        """Save LoRA adapters and processor."""
        if self.use_lora:
            self.model.save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

    def load_pretrained(self, checkpoint_dir: str):
        """Load LoRA adapters."""
        if self.use_lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model,
                checkpoint_dir,
                is_trainable=True
            )
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                checkpoint_dir,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        print(f"Model loaded from {checkpoint_dir}")


def create_target_text(x_norm: float, y_norm: float, is_done: bool, image_size: Tuple[int, int]) -> str:
    """
    Create target text from normalized coordinates.

    Args:
        x_norm: x coordinate in [-1, 1]
        y_norm: y coordinate in [-1, 1]
        is_done: Whether counting is done
        image_size: (width, height)

    Returns:
        Target string like "(245, 367)" or "done"
    """
    if is_done or (x_norm == -1.0 and y_norm == -1.0):
        return "done<|im_end|>"

    # Convert normalized to pixel coordinates
    W, H = image_size
    x_pixel = int((x_norm + 1) / 2 * W)
    y_pixel = int((y_norm + 1) / 2 * H)

    return f"({x_pixel}, {y_pixel})<|im_end|>"


if __name__ == "__main__":
    # Test loading
    print("Testing Qwen3-VL-4B-Thinking model loading...")

    model = VLMCountingModel(
        model_name="Qwen/Qwen3-VL-4B-Thinking",
        use_lora=True,
        lora_r=16,
        load_in_4bit=True
    )

    print("\nModel loaded successfully!")
    print(f"Device: {model.device}")

    # Test with dummy image
    from PIL import Image
    import numpy as np

    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    print("\nTesting inference...")
    with torch.no_grad():
        output = model(dummy_img, num_marked=3)

    print(f"Output: {output}")
    print("✅ Model test complete!")
