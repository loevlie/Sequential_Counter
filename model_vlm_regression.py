#!/usr/bin/env python3
"""
Vision-Language Model with MLP Regression Head for Sequential Counting

Based on FeD paper approach: Uses VLM features + MLP head to directly predict
coordinates and count, rather than generating text tokens.

Key improvements over text generation:
- Direct regression loss on coordinates (L1/MSE)
- Much faster inference (no token-by-token generation)
- Better spatial accuracy (gradient directly optimizes coordinate error)
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
from typing import Dict, List, Optional, Tuple, Union


class PredictionHead(nn.Module):
    """
    MLP head that predicts next coordinates and total count.

    Outputs:
        - x, y: Next object location (normalized [-1, 1])
        - count: Total count prediction
        - All outputs are -1 if done
    """

    def __init__(self, hidden_dim: int = 4096, num_layers: int = 3):
        super().__init__()

        layers = []
        current_dim = hidden_dim

        # Build MLP with decreasing dimensions
        for i in range(num_layers - 1):
            next_dim = current_dim // 2
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = next_dim

        # Final layer outputs 3 values: x, y, count
        layers.append(nn.Linear(current_dim, 3))

        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch_size, hidden_dim] - features from prediction token

        Returns:
            Dictionary with x, y, count predictions
        """
        output = self.mlp(features)  # [batch, 3]

        return {
            'x': output[:, 0],        # x coordinate
            'y': output[:, 1],        # y coordinate
            'count': output[:, 2]     # total count
        }


class VLMCountingModelRegression(nn.Module):
    """
    VLM-based counting model with MLP regression head.

    Uses special <pred> token whose hidden features are fed to MLP
    to directly predict (x, y, count).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Thinking",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_4bit: bool = True,
        device: str = "cuda",
        mlp_layers: int = 3
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

        # Add special prediction token to vocabulary
        special_tokens = {"additional_special_tokens": ["<pred>"]}
        self.processor.tokenizer.add_special_tokens(special_tokens)
        self.pred_token_id = self.processor.tokenizer.convert_tokens_to_ids("<pred>")

        # Quantization config
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

        # Resize embeddings for new token
        self.model.resize_token_embeddings(len(self.processor.tokenizer))

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

        # Get hidden dimension from text config
        # Qwen3VL has nested config structure
        if hasattr(self.model.config, 'text_config'):
            hidden_dim = self.model.config.text_config.hidden_size
        elif hasattr(self.model.config, 'hidden_size'):
            hidden_dim = self.model.config.hidden_size
        elif hasattr(self.model.config, 'hidden_dim'):
            hidden_dim = self.model.config.hidden_dim
        else:
            # Default for Qwen3-VL-4B
            hidden_dim = 4096
            print(f"Warning: Could not find hidden_size in config, using default {hidden_dim}")

        # Initialize prediction head
        self.prediction_head = PredictionHead(
            hidden_dim=hidden_dim,
            num_layers=mlp_layers
        ).to(device)

    def create_prompt(self, num_marked: int) -> List[Dict]:
        """
        Create prompt with image and prediction token.

        The <pred> token's hidden features will be used for prediction.
        """
        if num_marked == 0:
            marked_text = "No objects are marked yet."
        elif num_marked == 1:
            marked_text = "1 object is already marked with a red halo."
        else:
            marked_text = f"{num_marked} objects are already marked with red halos."

        # Construct messages with prediction token
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

Task: Predict the next unmarked object location (x, y) and total count.

Rules:
1. Output x, y as normalized coordinates in [-1, 1]
2. Output total count of all objects in the image
3. If all objects are marked, output x=-1, y=-1, but still output the correct total count
4. Count systematically from top-to-bottom, left-to-right

<pred>"""}
                ]
            }
        ]

        return messages

    def forward_regression(
        self,
        images: Union[Image.Image, List[Image.Image]],
        num_marked: Union[int, List[int]],
        gt_x: Optional[torch.Tensor] = None,
        gt_y: Optional[torch.Tensor] = None,
        gt_count: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with MLP prediction head.

        Args:
            images: PIL Image or list
            num_marked: Number of already marked objects
            gt_x, gt_y, gt_count: Ground truth for training (optional)

        Returns:
            Dictionary with predictions and loss (if ground truth provided)
        """
        # Handle single image
        if isinstance(images, Image.Image):
            images = [images]
            num_marked = [num_marked]

        batch_size = len(images)

        # Create prompts
        prompts = [self.create_prompt(n) for n in num_marked]

        # Apply chat template
        texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in prompts
        ]

        # Process inputs
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Forward through VLM to get hidden states
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        # Get last hidden layer
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]

        # Find <pred> token positions
        input_ids = inputs["input_ids"]
        pred_token_mask = (input_ids == self.pred_token_id)

        # Extract features at <pred> token positions
        pred_features = []
        for i in range(batch_size):
            # Find position of <pred> token
            pred_positions = torch.where(pred_token_mask[i])[0]
            if len(pred_positions) == 0:
                # Fallback to last token if <pred> not found
                pred_pos = -1
            else:
                pred_pos = pred_positions[-1]  # Use last occurrence

            pred_features.append(hidden_states[i, pred_pos, :])

        pred_features = torch.stack(pred_features)  # [batch, hidden_dim]

        # Get predictions from MLP head
        predictions = self.prediction_head(pred_features)

        # Compute loss if ground truth provided
        loss = None
        if gt_x is not None and gt_y is not None and gt_count is not None:
            # L1 loss for coordinates and count
            loss_x = nn.functional.l1_loss(predictions['x'], gt_x)
            loss_y = nn.functional.l1_loss(predictions['y'], gt_y)
            loss_count = nn.functional.l1_loss(predictions['count'], gt_count)

            # Combined loss
            loss = loss_x + loss_y + loss_count
            predictions['loss'] = loss

        return predictions

    def save_pretrained(self, output_dir: str):
        """Save model, processor, and prediction head."""
        if self.use_lora:
            self.model.save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)

        # Save prediction head
        torch.save(
            self.prediction_head.state_dict(),
            f"{output_dir}/prediction_head.pt"
        )
        print(f"Model and prediction head saved to {output_dir}")

    def load_pretrained(self, checkpoint_dir: str):
        """Load LoRA adapters and prediction head."""
        if self.use_lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model,
                checkpoint_dir,
                is_trainable=True
            )

        # Load prediction head
        pred_head_path = f"{checkpoint_dir}/prediction_head.pt"
        self.prediction_head.load_state_dict(
            torch.load(pred_head_path, map_location=self.device)
        )
        print(f"Model loaded from {checkpoint_dir}")
