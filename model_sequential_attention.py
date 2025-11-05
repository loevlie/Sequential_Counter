#!/usr/bin/env python3
"""
Sequential Attention Mechanism for Object Counting

Inspired by human serial counting:
- Explicit serial counting mode with one-by-one attention
- Foveation mechanism that focuses on different regions sequentially
- Recurrent working memory that maintains a running count
- Extended "thinking time" for complex counting tasks

Key components:
1. Working Memory Module: LSTM that tracks counting progress and maintains state
2. Foveation Attention: Spatial attention that creates focused "foveal" regions
3. Object-Level Cross-Attention: Attends to previously counted objects
4. Sequential Reasoning: Multi-step processing before each prediction
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
import math


class SpatialFoveationModule(nn.Module):
    """
    Implements spatial foveation - focusing attention on specific regions.

    Creates gaussian-like attention maps centered on different regions,
    simulating human foveal vision that focuses on one area at a time.
    """

    def __init__(self, hidden_dim: int = 4096, num_foveal_steps: int = 4):
        """
        Args:
            hidden_dim: Dimension of hidden features
            num_foveal_steps: Number of sequential foveal glimpses
        """
        super().__init__()
        self.num_foveal_steps = num_foveal_steps

        # Learn where to look next based on current state
        self.fovea_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2)  # Predicts (x, y) center for next fovea
        )

        # Learn the size/spread of foveal window
        self.fovea_scale_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Scale between 0 and 1
        )

    def create_gaussian_attention(self, center_x: torch.Tensor, center_y: torch.Tensor,
                                 scale: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        """
        Create 2D Gaussian attention map centered at (center_x, center_y).

        Args:
            center_x, center_y: Center coordinates in [-1, 1]
            scale: Standard deviation of Gaussian (controls foveal window size)
            grid_size: (height, width) of spatial grid

        Returns:
            Attention map of shape [batch, 1, height, width]
        """
        batch_size = center_x.shape[0]
        H, W = grid_size
        device = center_x.device

        # Create coordinate grids in [-1, 1]
        y_coords = torch.linspace(-1, 1, H, device=device).view(1, H, 1).expand(batch_size, H, W)
        x_coords = torch.linspace(-1, 1, W, device=device).view(1, 1, W).expand(batch_size, H, W)

        # Compute Gaussian attention
        center_x = center_x.view(batch_size, 1, 1)
        center_y = center_y.view(batch_size, 1, 1)
        scale = scale.view(batch_size, 1, 1)

        # Gaussian formula: exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2))
        dist_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2
        attention = torch.exp(-dist_sq / (2 * (scale + 0.1)**2))  # Add epsilon to avoid division by zero

        # Normalize to sum to 1
        attention = attention / (attention.sum(dim=[1, 2], keepdim=True) + 1e-8)

        return attention.unsqueeze(1)  # [batch, 1, H, W]

    def forward(self, features: torch.Tensor, memory_state: torch.Tensor,
                spatial_features: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Apply sequential foveation to spatial features.

        Args:
            features: Global features [batch, hidden_dim]
            memory_state: Working memory state [batch, hidden_dim]
            spatial_features: Spatial feature map [batch, hidden_dim, H, W]

        Returns:
            - Aggregated foveated features [batch, hidden_dim]
            - List of attention maps for visualization
        """
        batch_size = features.shape[0]
        _, _, H, W = spatial_features.shape

        # Match dtype of the module parameters
        module_dtype = next(self.fovea_predictor.parameters()).dtype
        memory_state = memory_state.to(module_dtype)
        spatial_features = spatial_features.to(module_dtype)

        current_state = memory_state
        foveated_features = []
        attention_maps = []

        for step in range(self.num_foveal_steps):
            # Ensure current_state is correct dtype
            current_state = current_state.to(module_dtype)

            # Predict where to look next
            fovea_center = self.fovea_predictor(current_state)  # [batch, 2]
            fovea_center = torch.tanh(fovea_center)  # Constrain to [-1, 1]
            center_x, center_y = fovea_center[:, 0], fovea_center[:, 1]

            # Predict foveal window size
            fovea_scale = self.fovea_scale_predictor(current_state).squeeze(-1)  # [batch]
            fovea_scale = fovea_scale * 0.3 + 0.1  # Scale to reasonable range [0.1, 0.4]

            # Create Gaussian attention map
            attention_map = self.create_gaussian_attention(
                center_x, center_y, fovea_scale, (H, W)
            )  # [batch, 1, H, W]
            attention_maps.append(attention_map)

            # Apply attention to spatial features
            attended_features = (spatial_features * attention_map).sum(dim=[2, 3])  # [batch, hidden_dim]
            foveated_features.append(attended_features)

            # Update state for next step
            current_state = current_state + attended_features

        # Aggregate all foveated features
        aggregated = torch.stack(foveated_features, dim=1).mean(dim=1)  # [batch, hidden_dim]

        return aggregated, attention_maps


class WorkingMemoryModule(nn.Module):
    """
    Maintains working memory of counting progress.

    Implements a recurrent module that:
    - Tracks how many objects have been counted
    - Remembers locations of previously counted objects
    - Maintains a running "count state" in memory
    """

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()
        self.hidden_dim = hidden_dim

        # LSTM for maintaining sequential counting state
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Running count estimator
        self.count_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU()  # Count must be positive
        )

    def forward(self, current_features: torch.Tensor,
                prev_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple, torch.Tensor]:
        """
        Update working memory with new observation.

        Args:
            current_features: Features from current counting step [batch, hidden_dim]
            prev_hidden: Previous LSTM hidden state (h, c) or None for initialization

        Returns:
            - Updated memory state [batch, hidden_dim]
            - New hidden state tuple (h, c)
            - Estimated count so far [batch, 1]
        """
        batch_size = current_features.shape[0]

        # Initialize hidden state if needed
        # Match the LSTM's dtype (check first layer)
        lstm_dtype = next(self.lstm.parameters()).dtype

        if prev_hidden is None:
            h0 = torch.zeros(2, batch_size, self.hidden_dim,
                           device=current_features.device, dtype=lstm_dtype)
            c0 = torch.zeros(2, batch_size, self.hidden_dim,
                           device=current_features.device, dtype=lstm_dtype)
            prev_hidden = (h0, c0)

        # Ensure input matches LSTM dtype
        current_features = current_features.to(lstm_dtype)

        # Process through LSTM
        # LSTM expects [batch, seq_len, features], we have single step so seq_len=1
        lstm_input = current_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        lstm_out, new_hidden = self.lstm(lstm_input, prev_hidden)

        # Extract memory state
        memory_state = lstm_out.squeeze(1)  # [batch, hidden_dim]

        # Estimate running count
        count_estimate = self.count_estimator(memory_state)  # [batch, 1]

        return memory_state, new_hidden, count_estimate


class ObjectCrossAttention(nn.Module):
    """
    Cross-attention mechanism for attending to previously counted objects.

    Allows the model to "remember" where it has already looked and
    avoid recounting the same objects.
    """

    def __init__(self, hidden_dim: int = 4096, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, query: torch.Tensor, memory_of_counted: torch.Tensor) -> torch.Tensor:
        """
        Attend to previously counted objects.

        Args:
            query: Current state [batch, hidden_dim]
            memory_of_counted: Memory of counted objects [batch, num_counted, hidden_dim]

        Returns:
            Attended features [batch, hidden_dim]
        """
        batch_size = query.shape[0]

        # Match dtype of module parameters
        module_dtype = next(self.query_proj.parameters()).dtype
        query = query.to(module_dtype)
        memory_of_counted = memory_of_counted.to(module_dtype)

        # Handle case where nothing has been counted yet
        if memory_of_counted.shape[1] == 0:
            return torch.zeros_like(query)

        # Project to Q, K, V
        Q = self.query_proj(query).unsqueeze(1)  # [batch, 1, hidden_dim]
        K = self.key_proj(memory_of_counted)  # [batch, num_counted, hidden_dim]
        V = self.value_proj(memory_of_counted)  # [batch, num_counted, hidden_dim]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, 1, head_dim]
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, num_counted, head_dim]
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, num_counted, head_dim]

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [batch, num_heads, 1, num_counted]
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # [batch, num_heads, 1, head_dim]

        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(batch_size, 1, -1)  # [batch, 1, hidden_dim]
        attended = attended.squeeze(1)  # [batch, hidden_dim]

        # Output projection
        output = self.out_proj(attended)

        return output


class SequentialReasoningModule(nn.Module):
    """
    Multi-step reasoning module for extended "thinking time".

    Implements chain-of-thought-like reasoning by processing features
    through multiple refinement steps before making a prediction.
    """

    def __init__(self, hidden_dim: int = 4096, num_reasoning_steps: int = 3):
        super().__init__()
        self.num_reasoning_steps = num_reasoning_steps

        # Reasoning steps are implemented as residual blocks
        self.reasoning_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_reasoning_steps)
        ])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply sequential reasoning steps.

        Args:
            features: Input features [batch, hidden_dim]

        Returns:
            Refined features after reasoning [batch, hidden_dim]
        """
        # Match dtype of module parameters
        module_dtype = next(self.reasoning_layers[0].parameters()).dtype
        features = features.to(module_dtype)

        current = features

        for layer in self.reasoning_layers:
            # Residual connection
            refined = layer(current)
            current = current + refined

        return current


class SequentialAttentionCountingModel(nn.Module):
    """
    Full sequential attention counting model.

    Combines:
    - VLM for visual encoding
    - Working memory for tracking counting progress
    - Foveation for focused spatial attention
    - Cross-attention to previously counted objects
    - Sequential reasoning for extended thinking time
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        device: str = "cuda",
        num_foveal_steps: int = 4,
        num_reasoning_steps: int = 3,
        max_memory_objects: int = 50
    ):
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.use_lora = use_lora
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.max_memory_objects = max_memory_objects

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

        # Quantization config
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )

        # Load base VLM
        print(f"Loading base model {model_name}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        # Resize embeddings for new tokens
        self.model.resize_token_embeddings(len(self.processor.tokenizer))

        # Apply LoRA
        if use_lora:
            print("Applying LoRA adapters...")
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

        # Get hidden dimension
        if hasattr(self.model.config, 'text_config'):
            hidden_dim = self.model.config.text_config.hidden_size
        elif hasattr(self.model.config, 'hidden_size'):
            hidden_dim = self.model.config.hidden_size
        else:
            hidden_dim = 4096
            print(f"Warning: Could not find hidden_size in config, using default {hidden_dim}")

        self.hidden_dim = hidden_dim
        print(f"Using hidden_dim={hidden_dim}")

        # Initialize sequential attention components
        print("Initializing sequential attention modules...")
        self.working_memory = WorkingMemoryModule(hidden_dim).to(device).to(torch.bfloat16)
        self.foveation = SpatialFoveationModule(hidden_dim, num_foveal_steps).to(device).to(torch.bfloat16)
        self.object_attention = ObjectCrossAttention(hidden_dim).to(device).to(torch.bfloat16)
        self.sequential_reasoning = SequentialReasoningModule(hidden_dim, num_reasoning_steps).to(device).to(torch.bfloat16)

        # Prediction heads (same as original model)
        self.x_head = self._build_prediction_head(hidden_dim).to(device).to(torch.bfloat16)
        self.y_head = self._build_prediction_head(hidden_dim).to(device).to(torch.bfloat16)
        self.done_head = self._build_classification_head(hidden_dim).to(device).to(torch.bfloat16)

        # Memory buffer for storing features of counted objects
        self.memory_buffer = None
        self.memory_hidden_state = None

    def _build_prediction_head(self, hidden_dim: int) -> nn.Module:
        """Build MLP head for coordinate prediction."""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1),
            nn.Tanh()  # Constrain to [-1, 1]
        )

    def _build_classification_head(self, hidden_dim: int) -> nn.Module:
        """Build MLP head for done classification."""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1)
        )

    def reset_memory(self, batch_size: int):
        """Reset working memory for new counting task."""
        self.memory_buffer = torch.zeros(
            batch_size, 0, self.hidden_dim,  # Will be expanded as objects are counted
            device=self.device, dtype=torch.bfloat16
        )
        self.memory_hidden_state = None

    def create_prompt(self, num_marked: int, category: str = "objects") -> List[Dict]:
        """Create prompt for the model."""
        if num_marked == 0:
            marked_text = "No objects are marked yet. Start counting systematically."
        elif num_marked == 1:
            marked_text = "1 object is already marked. Continue counting the next one."
        else:
            marked_text = f"{num_marked} objects are already marked. Continue counting the next one."

        messages = [
            {
                "role": "system",
                "content": f"You are a vision assistant that counts {category} one-by-one, attending to each object sequentially."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"""Count the {category} in this image systematically, one at a time. {marked_text}

Task: Predict the (x, y) location of the next unmarked {category.rstrip('s')}, and whether counting is complete.

Rules:
1. Focus attention on one object at a time
2. Output x, y as normalized coordinates in [-1, 1]
3. Count systematically from top-to-bottom, left-to-right
4. Output done=1 if all {category} are counted, otherwise done=0

Next prediction: <x> <y> <done>"""}
                ]
            }
        ]

        return messages

    def forward_with_attention(
        self,
        images: Union[Image.Image, List[Image.Image]],
        num_marked: Union[int, List[int]],
        category: str = "objects",
        gt_x: Optional[torch.Tensor] = None,
        gt_y: Optional[torch.Tensor] = None,
        gt_done: Optional[torch.Tensor] = None,
        return_attention_maps: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with sequential attention mechanism.

        This method implements the full sequential counting pipeline:
        1. Extract features from VLM
        2. Update working memory with current observation
        3. Apply foveation to focus on specific regions
        4. Cross-attend to previously counted objects
        5. Sequential reasoning for extended thinking
        6. Predict next object location or done signal
        """
        # Handle single image
        if isinstance(images, Image.Image):
            images = [images]
            num_marked = [num_marked]

        batch_size = len(images)

        # Reset memory if this is a new batch
        if self.memory_buffer is None or self.memory_buffer.shape[0] != batch_size:
            self.reset_memory(batch_size)

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

        # Forward through VLM to get hidden states
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        # Get last hidden layer
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]

        # Extract features at special token positions
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

        x_features = torch.stack(x_features)  # [batch, hidden_dim]
        y_features = torch.stack(y_features)
        done_features = torch.stack(done_features)

        # Combine features for processing
        combined_features = (x_features + y_features + done_features) / 3.0

        # === SEQUENTIAL ATTENTION PIPELINE ===

        # 1. Update working memory
        memory_state, new_hidden, count_estimate = self.working_memory(
            combined_features, self.memory_hidden_state
        )
        self.memory_hidden_state = new_hidden

        # 2. Create spatial feature map (simplified: reshape hidden states)
        # In a full implementation, this would extract spatial features from vision encoder
        # For now, we'll use a learned projection
        seq_len = hidden_states.shape[1]
        grid_size = int(math.sqrt(seq_len))
        if grid_size * grid_size == seq_len:
            spatial_features = hidden_states.permute(0, 2, 1).view(
                batch_size, -1, grid_size, grid_size
            )
        else:
            # Fallback: create dummy spatial features
            spatial_features = combined_features.view(batch_size, -1, 1, 1).expand(
                batch_size, -1, 8, 8
            )

        # 3. Apply foveation
        foveated_features, attention_maps = self.foveation(
            combined_features, memory_state, spatial_features
        )

        # 4. Cross-attend to previously counted objects
        object_context = self.object_attention(foveated_features, self.memory_buffer)

        # 5. Combine features - include memory_state so working_memory gets gradients!
        integrated_features = combined_features + memory_state + foveated_features + object_context

        # 6. Sequential reasoning (extended thinking time)
        reasoned_features = self.sequential_reasoning(integrated_features)

        # Update memory buffer with current features
        if num_marked[0] > 0:  # Only store if we've counted something
            current_memory = reasoned_features.unsqueeze(1)  # [batch, 1, hidden_dim]
            self.memory_buffer = torch.cat([self.memory_buffer, current_memory], dim=1)

            # Limit memory size
            if self.memory_buffer.shape[1] > self.max_memory_objects:
                self.memory_buffer = self.memory_buffer[:, -self.max_memory_objects:, :]

        # 7. Make predictions
        pred_x = self.x_head(reasoned_features).squeeze(-1)  # [batch]
        pred_y = self.y_head(reasoned_features).squeeze(-1)  # [batch]
        pred_done_logits = self.done_head(reasoned_features).squeeze(-1)  # [batch]
        pred_done_probs = torch.sigmoid(pred_done_logits)

        predictions = {
            'x': pred_x,
            'y': pred_y,
            'done': pred_done_probs,
            'count_estimate': count_estimate.squeeze(-1)
        }

        if return_attention_maps:
            predictions['attention_maps'] = attention_maps

        # Compute loss if ground truth provided
        loss = None
        if gt_done is not None and gt_x is None and gt_y is None:
            # Classification mode
            loss_done = F.binary_cross_entropy_with_logits(pred_done_logits, gt_done)
            loss = loss_done
            predictions['loss'] = loss
            predictions['loss_done'] = loss_done
        elif gt_x is not None and gt_y is not None and gt_done is None:
            # Regression mode
            loss_x = F.mse_loss(pred_x, gt_x)
            loss_y = F.mse_loss(pred_y, gt_y)
            loss = loss_x + loss_y
            predictions['loss'] = loss
            predictions['loss_x'] = loss_x
            predictions['loss_y'] = loss_y
        elif gt_x is not None and gt_y is not None and gt_done is not None:
            # Mixed mode
            loss_x = F.mse_loss(pred_x, gt_x)
            loss_y = F.mse_loss(pred_y, gt_y)
            loss_done = F.binary_cross_entropy_with_logits(pred_done_logits, gt_done)
            loss = loss_x + loss_y + loss_done
            predictions['loss'] = loss
            predictions['loss_x'] = loss_x
            predictions['loss_y'] = loss_y
            predictions['loss_done'] = loss_done

        return predictions

    def save_pretrained(self, output_dir: str):
        """Save model components."""
        if self.use_lora:
            self.model.save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)

        # Save all components
        torch.save(self.x_head.state_dict(), f"{output_dir}/x_head.pt")
        torch.save(self.y_head.state_dict(), f"{output_dir}/y_head.pt")
        torch.save(self.done_head.state_dict(), f"{output_dir}/done_head.pt")
        torch.save(self.working_memory.state_dict(), f"{output_dir}/working_memory.pt")
        torch.save(self.foveation.state_dict(), f"{output_dir}/foveation.pt")
        torch.save(self.object_attention.state_dict(), f"{output_dir}/object_attention.pt")
        torch.save(self.sequential_reasoning.state_dict(), f"{output_dir}/sequential_reasoning.pt")

        print(f"Sequential attention model saved to {output_dir}")

    def load_pretrained(self, checkpoint_dir: str):
        """Load model components."""
        if self.use_lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model,
                checkpoint_dir,
                is_trainable=True
            )

        # Load all components
        self.x_head.load_state_dict(torch.load(f"{checkpoint_dir}/x_head.pt", map_location=self.device))
        self.y_head.load_state_dict(torch.load(f"{checkpoint_dir}/y_head.pt", map_location=self.device))
        self.done_head.load_state_dict(torch.load(f"{checkpoint_dir}/done_head.pt", map_location=self.device))
        self.working_memory.load_state_dict(torch.load(f"{checkpoint_dir}/working_memory.pt", map_location=self.device))
        self.foveation.load_state_dict(torch.load(f"{checkpoint_dir}/foveation.pt", map_location=self.device))
        self.object_attention.load_state_dict(torch.load(f"{checkpoint_dir}/object_attention.pt", map_location=self.device))
        self.sequential_reasoning.load_state_dict(torch.load(f"{checkpoint_dir}/sequential_reasoning.pt", map_location=self.device))

        print(f"Sequential attention model loaded from {checkpoint_dir}")
