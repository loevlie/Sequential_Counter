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
    MLP head that predicts scalar values from VLM hidden features.

    Following the autonomous driving sensorimotor agent approach:
    - Separate <x> and <y> tokens in prompt
    - Each token's hidden features fed to its own MLP head
    - Each MLP outputs a single scalar value

    This design allows the model to learn separate feature representations
    for x and y coordinates, similar to how autonomous driving models
    predict multiple waypoint coordinates.
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

        # Final layer outputs 1 scalar value
        layers.append(nn.Linear(current_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Add tanh to constrain outputs to [-1, 1]
        self.activation = nn.Tanh()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, hidden_dim] - features from a special token

        Returns:
            Scalar predictions in [-1, 1] range [batch_size]
        """
        output = self.mlp(features)  # [batch, 1]
        output = self.activation(output)  # Apply tanh to constrain to [-1, 1]
        return output.squeeze(-1)  # [batch]


class DoneClassificationHead(nn.Module):
    """
    MLP head for binary classification of "done" signal.

    Predicts whether all objects have been counted (1) or not (0).
    Uses sigmoid activation for binary classification.
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

        # Final layer outputs 1 value (binary classification logits)
        layers.append(nn.Linear(current_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Args:
            features: [batch_size, hidden_dim] - features from a special token
            return_logits: If True, return raw logits; if False, return probabilities

        Returns:
            If return_logits=True: logits [batch_size]
            If return_logits=False: Done probability in [0, 1] range [batch_size]
        """
        logits = self.mlp(features).squeeze(-1)  # [batch]
        if return_logits:
            return logits
        else:
            return torch.sigmoid(logits)  # Apply sigmoid to get probability


class VLMCountingModelRegression(nn.Module):
    """
    VLM-based counting model with MLP regression head.

    Uses special <x>, <y>, <done> tokens:
    - <x> and <y> for coordinate regression (RMSE loss)
    - <done> for binary classification (BCE loss)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Thinking",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        device: str = "cuda",
        mlp_layers: int = 3,
        loss_weight_spatial: float = 1.0,
        loss_weight_count: float = 0.1
    ):
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.use_lora = use_lora
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.loss_weight_spatial = loss_weight_spatial
        self.loss_weight_count = loss_weight_count

        # Load processor
        print(f"Loading processor from {model_name}...")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Add special prediction tokens to vocabulary (<x>, <y>, and <done>)
        # Following autonomous driving sensorimotor agent approach
        special_tokens = {"additional_special_tokens": ["<x>", "<y>", "<done>"]}
        self.processor.tokenizer.add_special_tokens(special_tokens)
        self.x_token_id = self.processor.tokenizer.convert_tokens_to_ids("<x>")
        self.y_token_id = self.processor.tokenizer.convert_tokens_to_ids("<y>")
        self.done_token_id = self.processor.tokenizer.convert_tokens_to_ids("<done>")

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

            # Only prepare for kbit training if using quantization
            if self.load_in_4bit or self.load_in_8bit:
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

        # Initialize separate prediction heads for x, y, and done (use bfloat16 to match VLM)
        # Following sensorimotor agent: each output gets its own MLP
        self.x_head = PredictionHead(
            hidden_dim=hidden_dim,
            num_layers=mlp_layers
        ).to(device).to(torch.bfloat16)

        self.y_head = PredictionHead(
            hidden_dim=hidden_dim,
            num_layers=mlp_layers
        ).to(device).to(torch.bfloat16)

        self.done_head = DoneClassificationHead(
            hidden_dim=hidden_dim,
            num_layers=mlp_layers
        ).to(device).to(torch.bfloat16)

    def create_prompt(self, num_marked: int, category: str = "objects") -> List[Dict]:
        """
        Create prompt with image and <x>, <y>, <done> prediction tokens.

        Following sensorimotor agent approach: The prompt ends with special tokens
        whose hidden features are extracted and fed to separate MLP heads:
        - <x> and <y>: for coordinate regression (RMSE loss)
        - <done>: for binary classification (BCE loss, 1 if all objects counted)
        """
        if num_marked == 0:
            marked_text = "No objects are marked yet."
        elif num_marked == 1:
            marked_text = "1 object is already marked with a numbered label."
        else:
            marked_text = f"{num_marked} objects are already marked with numbered labels."

        # Construct messages with <x>, <y>, and <done> prediction tokens
        messages = [
            {
                "role": "system",
                "content": f"You are a vision assistant that locates {category} in images systematically."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"""Count the {category} in this image. {marked_text}

Task: Predict the (x, y) location of the next unmarked {category.rstrip('s')}, and whether counting is complete.

Rules:
1. Output x, y as normalized coordinates in [-1, 1]
2. Count systematically from top-to-bottom, left-to-right
3. Output done=1 if all {category} are counted, otherwise done=0

Next prediction: <x> <y> <done>"""}
                ]
            }
        ]

        return messages

    def forward_regression(
        self,
        images: Union[Image.Image, List[Image.Image]],
        num_marked: Union[int, List[int]],
        category: str = "objects",
        gt_x: Optional[torch.Tensor] = None,
        gt_y: Optional[torch.Tensor] = None,
        gt_done: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with MLP prediction heads.

        Args:
            images: PIL Image or list
            num_marked: Number of already marked objects
            category: Object category name (e.g., "products", "bottles")
            gt_x, gt_y: Ground truth coordinates for training (optional)
            gt_done: Ground truth done signal (1 if all counted, 0 otherwise) (optional)

        Returns:
            Dictionary with predictions and loss (if ground truth provided)
        """
        # Handle single image
        if isinstance(images, Image.Image):
            images = [images]
            num_marked = [num_marked]

        batch_size = len(images)

        # Create prompts with category
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

        # Forward through VLM to get hidden states
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        # Get last hidden layer
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]

        # Find <x>, <y>, and <done> token positions
        input_ids = inputs["input_ids"]
        x_token_mask = (input_ids == self.x_token_id)
        y_token_mask = (input_ids == self.y_token_id)
        done_token_mask = (input_ids == self.done_token_id)

        # Extract features at <x>, <y>, and <done> token positions
        x_features = []
        y_features = []
        done_features = []

        for i in range(batch_size):
            # Find position of <x> token
            x_positions = torch.where(x_token_mask[i])[0]
            if len(x_positions) == 0:
                # Debug: print the input_ids to see what tokens are present
                print(f"ERROR: <x> token not found in batch item {i}")
                print(f"  Expected token ID: {self.x_token_id}")
                print(f"  Input IDs: {input_ids[i]}")
                print(f"  Decoded text: {self.processor.tokenizer.decode(input_ids[i])}")
                raise ValueError(f"<x> token not found in batch item {i}")
            x_pos = x_positions[-1]  # Use last occurrence
            x_features.append(hidden_states[i, x_pos, :])

            # Find position of <y> token
            y_positions = torch.where(y_token_mask[i])[0]
            if len(y_positions) == 0:
                print(f"ERROR: <y> token not found in batch item {i}")
                print(f"  Expected token ID: {self.y_token_id}")
                print(f"  Input IDs: {input_ids[i]}")
                print(f"  Decoded text: {self.processor.tokenizer.decode(input_ids[i])}")
                raise ValueError(f"<y> token not found in batch item {i}")
            y_pos = y_positions[-1]  # Use last occurrence
            y_features.append(hidden_states[i, y_pos, :])

            # Find position of <done> token
            done_positions = torch.where(done_token_mask[i])[0]
            if len(done_positions) == 0:
                print(f"ERROR: <done> token not found in batch item {i}")
                print(f"  Expected token ID: {self.done_token_id}")
                print(f"  Input IDs: {input_ids[i]}")
                print(f"  Decoded text: {self.processor.tokenizer.decode(input_ids[i])}")
                raise ValueError(f"<done> token not found in batch item {i}")
            done_pos = done_positions[-1]  # Use last occurrence
            done_features.append(hidden_states[i, done_pos, :])

        x_features = torch.stack(x_features)  # [batch, hidden_dim]
        y_features = torch.stack(y_features)  # [batch, hidden_dim]
        done_features = torch.stack(done_features)  # [batch, hidden_dim]

        # Get predictions from separate MLP heads
        pred_x = self.x_head(x_features)  # [batch] - coordinates in [-1, 1]
        pred_y = self.y_head(y_features)  # [batch] - coordinates in [-1, 1]

        # For done head: get logits for loss, probabilities for inference
        pred_done_logits = self.done_head(done_features, return_logits=True)  # [batch] - logits
        pred_done_probs = torch.sigmoid(pred_done_logits)  # [batch] - probability in [0, 1]

        predictions = {
            'x': pred_x,
            'y': pred_y,
            'done': pred_done_probs  # Return probabilities for inference/logging
        }

        # Compute loss if ground truth provided
        # Support mode-specific training:
        # - Classification mode: only gt_done provided, compute only done loss
        # - Regression mode: only gt_x, gt_y provided, compute only regression loss
        # - Mixed mode: all provided, compute all losses
        loss = None
        if gt_done is not None and gt_x is None and gt_y is None:
            # Classification mode only - use BCE with logits for numerical stability
            loss_done = nn.functional.binary_cross_entropy_with_logits(pred_done_logits, gt_done)
            loss = loss_done
            predictions['loss'] = loss
            predictions['loss_done'] = loss_done
        elif gt_x is not None and gt_y is not None and gt_done is None:
            # Regression mode only
            loss_x = nn.functional.mse_loss(pred_x, gt_x)
            loss_y = nn.functional.mse_loss(pred_y, gt_y)
            loss = loss_x + loss_y
            predictions['loss'] = loss
            predictions['loss_x'] = loss_x
            predictions['loss_y'] = loss_y
        elif gt_x is not None and gt_y is not None and gt_done is not None:
            # Mixed mode (for validation) - use BCE with logits for numerical stability
            loss_x = nn.functional.mse_loss(pred_x, gt_x)
            loss_y = nn.functional.mse_loss(pred_y, gt_y)
            loss_done = nn.functional.binary_cross_entropy_with_logits(pred_done_logits, gt_done)
            loss = loss_x + loss_y + loss_done
            predictions['loss'] = loss
            predictions['loss_x'] = loss_x
            predictions['loss_y'] = loss_y
            predictions['loss_done'] = loss_done

        return predictions

    def save_pretrained(self, output_dir: str):
        """Save model, processor, and prediction heads."""
        if self.use_lora:
            self.model.save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)

        # Save x, y, and done prediction heads
        torch.save(
            self.x_head.state_dict(),
            f"{output_dir}/x_head.pt"
        )
        torch.save(
            self.y_head.state_dict(),
            f"{output_dir}/y_head.pt"
        )
        torch.save(
            self.done_head.state_dict(),
            f"{output_dir}/done_head.pt"
        )
        print(f"Model and prediction heads saved to {output_dir}")

    def load_pretrained(self, checkpoint_dir: str):
        """Load LoRA adapters and prediction heads."""
        if self.use_lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model,
                checkpoint_dir,
                is_trainable=True
            )

        # Load x, y, and done prediction heads
        x_head_path = f"{checkpoint_dir}/x_head.pt"
        y_head_path = f"{checkpoint_dir}/y_head.pt"
        done_head_path = f"{checkpoint_dir}/done_head.pt"
        self.x_head.load_state_dict(
            torch.load(x_head_path, map_location=self.device)
        )
        self.y_head.load_state_dict(
            torch.load(y_head_path, map_location=self.device)
        )
        self.done_head.load_state_dict(
            torch.load(done_head_path, map_location=self.device)
        )
        print(f"Model loaded from {checkpoint_dir}")
