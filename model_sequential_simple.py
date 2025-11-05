#!/usr/bin/env python3
"""
SIMPLIFIED Sequential Attention Model

This is a minimal version that focuses on what works:
- Uses VLM features directly (no complex transformations)
- Simple working memory (just track count)
- Minimal sequential attention
- Direct predictions from VLM features

Based on debugging, the complex pipeline was causing:
1. Zero gradients to working memory and foveation
2. Loss explosions
3. Predictions stuck at corners
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union


class SimplePredictionHead(nn.Module):
    """Simple 2-layer MLP for predictions - NO TANH to avoid saturation."""

    def __init__(self, hidden_dim: int = 1536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
            # NO activation - let the loss handle the range
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features).squeeze(-1)


class SimpleDoneHead(nn.Module):
    """Simple binary classification head."""

    def __init__(self, hidden_dim: int = 1536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, features: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        logits = self.mlp(features).squeeze(-1)
        if return_logits:
            return logits
        return torch.sigmoid(logits)


class SimpleSequentialModel(nn.Module):
    """
    Simplified sequential counting model.

    Key simplifications:
    - Use VLM features DIRECTLY for predictions
    - Add simple count tracking (no complex LSTM)
    - Minimal transformations to prevent gradient issues
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
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
        self.load_in_4bit = load_in_4bit

        # Load processor
        print(f"Loading processor from {model_name}...")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Add special tokens
        special_tokens = {"additional_special_tokens": ["<x>", "<y>", "<done>"]}
        self.processor.tokenizer.add_special_tokens(special_tokens)
        self.x_token_id = self.processor.tokenizer.convert_tokens_to_ids("<x>")
        self.y_token_id = self.processor.tokenizer.convert_tokens_to_ids("<y>")
        self.done_token_id = self.processor.tokenizer.convert_tokens_to_ids("<done>")

        # Quantization
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )

        # Load VLM
        print(f"Loading base model {model_name}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        # Resize embeddings
        self.model.resize_token_embeddings(len(self.processor.tokenizer))

        # Apply LoRA
        if use_lora:
            print("Applying LoRA adapters...")
            if self.load_in_4bit:
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

        # Get hidden dimension
        if hasattr(self.model.config, 'text_config'):
            hidden_dim = self.model.config.text_config.hidden_size
        elif hasattr(self.model.config, 'hidden_size'):
            hidden_dim = self.model.config.hidden_size
        else:
            hidden_dim = 1536

        self.hidden_dim = hidden_dim
        print(f"Using hidden_dim={hidden_dim}")

        # Simple prediction heads
        print("Initializing prediction heads...")
        self.x_head = SimplePredictionHead(hidden_dim).to(device).to(torch.bfloat16)
        self.y_head = SimplePredictionHead(hidden_dim).to(device).to(torch.bfloat16)
        self.done_head = SimpleDoneHead(hidden_dim).to(device).to(torch.bfloat16)

    def create_prompt(self, num_marked: int, category: str = "objects") -> List[Dict]:
        """Create prompt for the model."""
        if num_marked == 0:
            marked_text = "No objects are marked yet. Start counting systematically."
        elif num_marked == 1:
            marked_text = "1 object is already marked. Continue counting."
        else:
            marked_text = f"{num_marked} objects are already marked. Continue counting."

        messages = [
            {
                "role": "system",
                "content": f"You are a vision assistant that counts {category} systematically."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"""Count the {category} in this image. {marked_text}

Task: Predict the (x, y) location of the next unmarked {category.rstrip('s')}, and whether counting is complete.

Rules:
1. Output x, y as normalized coordinates in [-1, 1]
2. Count from top-to-bottom, left-to-right
3. Output done=1 if all {category} are counted, otherwise done=0

Next prediction: <x> <y> <done>"""}
                ]
            }
        ]

        return messages

    def forward(
        self,
        images: Union[Image.Image, List[Image.Image]],
        num_marked: Union[int, List[int]],
        category: str = "objects",
        gt_x: Optional[torch.Tensor] = None,
        gt_y: Optional[torch.Tensor] = None,
        gt_done: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass - SIMPLIFIED version.

        Just extract VLM features and predict directly.
        No complex transformations.
        """
        # Handle single image
        if isinstance(images, Image.Image):
            images = [images]
            num_marked = [num_marked]

        batch_size = len(images)

        # Create prompts
        prompts = [self.create_prompt(n, category) for n in num_marked]

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

        # Forward through VLM
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        # Get hidden states
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]

        # Extract features at token positions
        input_ids = inputs["input_ids"]
        x_token_mask = (input_ids == self.x_token_id)
        y_token_mask = (input_ids == self.y_token_id)
        done_token_mask = (input_ids == self.done_token_id)

        x_features = []
        y_features = []
        done_features = []

        for i in range(batch_size):
            x_pos = torch.where(x_token_mask[i])[0][-1]
            y_pos = torch.where(y_token_mask[i])[0][-1]
            done_pos = torch.where(done_token_mask[i])[0][-1]

            x_features.append(hidden_states[i, x_pos, :])
            y_features.append(hidden_states[i, y_pos, :])
            done_features.append(hidden_states[i, done_pos, :])

        x_features = torch.stack(x_features)
        y_features = torch.stack(y_features)
        done_features = torch.stack(done_features)

        # Ensure features match head dtype (bfloat16)
        x_features = x_features.to(torch.bfloat16)
        y_features = y_features.to(torch.bfloat16)
        done_features = done_features.to(torch.bfloat16)

        # Direct predictions from VLM features (no complex transformations!)
        pred_x = self.x_head(x_features)
        pred_y = self.y_head(y_features)
        pred_done_logits = self.done_head(done_features, return_logits=True)
        pred_done_probs = torch.sigmoid(pred_done_logits)

        # Clip predictions to prevent explosions
        pred_x = torch.clamp(pred_x, -2.0, 2.0)
        pred_y = torch.clamp(pred_y, -2.0, 2.0)

        predictions = {
            'x': pred_x,
            'y': pred_y,
            'done': pred_done_probs
        }

        # Compute loss
        loss = None
        if gt_done is not None and gt_x is None and gt_y is None:
            # Classification mode
            loss_done = F.binary_cross_entropy_with_logits(pred_done_logits, gt_done)
            loss = loss_done
            predictions['loss'] = loss
            predictions['loss_done'] = loss_done
        elif gt_x is not None and gt_y is not None and gt_done is None:
            # Regression mode - AGGRESSIVE MULTI-STAGE LOSS for 0-5px accuracy
            # L1 distance in normalized space
            dist = torch.abs(pred_x - gt_x) + torch.abs(pred_y - gt_y)

            # Three-stage penalty for sub-5-pixel precision:
            # Stage 1: dist >= 0.1 (>19px) - Strong linear penalty to get closer
            # Stage 2: 0.05 <= dist < 0.1 (10-19px) - Quadratic penalty
            # Stage 3: dist < 0.05 (<10px) - Extra strong quadratic for sub-5px precision
            #
            # Target: 0.05 normalized = ~9.6 pixels, forcing most predictions < 5px
            loss_spatial = torch.where(
                dist < 0.05,
                10.0 * dist ** 2,  # 10x stronger quadratic when very close (sub-5px)
                torch.where(
                    dist < 0.1,
                    dist ** 2,  # Standard quadratic for medium distance (5-19px)
                    0.2 * dist + 0.01  # Stronger linear when far (>19px)
                )
            ).mean()

            # Also keep per-coordinate MSE for logging
            loss_x = F.mse_loss(pred_x, gt_x)
            loss_y = F.mse_loss(pred_y, gt_y)

            loss = loss_spatial
            predictions['loss'] = loss
            predictions['loss_x'] = loss_x  # For logging
            predictions['loss_y'] = loss_y  # For logging
            predictions['loss_spatial'] = loss_spatial
        elif gt_x is not None and gt_y is not None and gt_done is not None:
            # Mixed mode - AGGRESSIVE MULTI-STAGE LOSS + weighted done
            dist = torch.abs(pred_x - gt_x) + torch.abs(pred_y - gt_y)
            loss_spatial = torch.where(
                dist < 0.05,
                10.0 * dist ** 2,  # 10x stronger for sub-5px precision
                torch.where(
                    dist < 0.1,
                    dist ** 2,
                    0.2 * dist + 0.01
                )
            ).mean()

            loss_x = F.mse_loss(pred_x, gt_x)
            loss_y = F.mse_loss(pred_y, gt_y)
            loss_done = F.binary_cross_entropy_with_logits(pred_done_logits, gt_done)

            loss = loss_spatial + 3.0 * loss_done  # 3x weight on done
            predictions['loss'] = loss
            predictions['loss_x'] = loss_x
            predictions['loss_y'] = loss_y
            predictions['loss_done'] = loss_done
            predictions['loss_spatial'] = loss_spatial

        return predictions

    def save_pretrained(self, output_dir: str):
        """Save model."""
        if self.use_lora:
            self.model.save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)

        torch.save(self.x_head.state_dict(), f"{output_dir}/x_head.pt")
        torch.save(self.y_head.state_dict(), f"{output_dir}/y_head.pt")
        torch.save(self.done_head.state_dict(), f"{output_dir}/done_head.pt")

        print(f"Model saved to {output_dir}")

    def load_pretrained(self, checkpoint_dir: str):
        """Load model."""
        if self.use_lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model,
                checkpoint_dir,
                is_trainable=True
            )

        self.x_head.load_state_dict(torch.load(f"{checkpoint_dir}/x_head.pt", map_location=self.device))
        self.y_head.load_state_dict(torch.load(f"{checkpoint_dir}/y_head.pt", map_location=self.device))
        self.done_head.load_state_dict(torch.load(f"{checkpoint_dir}/done_head.pt", map_location=self.device))

        print(f"Model loaded from {checkpoint_dir}")
