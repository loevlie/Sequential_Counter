#!/usr/bin/env python3
"""
Point Prediction Network

Predicts the next object location given:
- Visual features from VLM encoder
- Previously marked object positions

Outputs:
- x, y: Next point coordinates [0, 1]
- is_done: Boolean indicating all objects are marked
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointPredictionHead(nn.Module):
    """
    Point prediction network for sequential counting.

    Architecture:
        1. Encode previous marked positions
        2. Cross-attend visual features with position history
        3. Predict next point (x, y) and done signal
    """

    def __init__(self,
                 visual_feature_dim: int = 768,  # CLIP ViT-B/32
                 hidden_dim: int = 256,
                 max_objects: int = 150,
                 dropout: float = 0.1):
        """
        Args:
            visual_feature_dim: Dimension of visual features from VLM encoder
            hidden_dim: Hidden dimension for internal layers
            max_objects: Maximum number of objects (for padding)
            dropout: Dropout rate
        """
        super().__init__()

        self.visual_feature_dim = visual_feature_dim
        self.hidden_dim = hidden_dim
        self.max_objects = max_objects

        # Position encoder: (x, y) -> hidden_dim
        self.position_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # Project visual features to hidden_dim
        self.visual_projection = nn.Linear(visual_feature_dim, hidden_dim)

        # Cross-attention: visual features attend to marked positions
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Self-attention on visual features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Prediction head for (x, y, is_done)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # (x, y, is_done_logit)
        )

    def forward(self, visual_features, marked_positions, num_marked=None):
        """
        Forward pass.

        Args:
            visual_features: [batch, num_patches, visual_feature_dim]
                           Features from VLM vision encoder
            marked_positions: [batch, max_objects, 2]
                            Previously marked positions (normalized to [0,1])
                            Padded with zeros for unused positions
            num_marked: [batch] Number of actually marked positions (not padding)
                       If None, assumes all positions are valid

        Returns:
            dict with:
                - x: [batch] predicted x-coordinate (0-1, or -1 if done)
                - y: [batch] predicted y-coordinate (0-1, or -1 if done)
                - is_done: [batch] probability that all objects are marked
                - is_done_logit: [batch] raw logit for is_done (for loss computation)
        """
        batch_size = visual_features.shape[0]

        # Project visual features
        visual_proj = self.visual_projection(visual_features)
        # [batch, num_patches, hidden_dim]

        # Self-attention on visual features
        visual_attended = self.self_attention(visual_proj, visual_proj, visual_proj)[0]
        visual_attended = self.ln1(visual_attended + visual_proj)
        # [batch, num_patches, hidden_dim]

        # Encode marked positions
        pos_embeddings = self.position_encoder(marked_positions)
        # [batch, max_objects, hidden_dim]

        # Create attention mask (mask out padding)
        if num_marked is not None:
            # Create mask: True where position is padding (should be ignored)
            max_len = marked_positions.shape[1]
            mask = torch.arange(max_len, device=marked_positions.device)[None, :] >= num_marked[:, None]
            # [batch, max_objects]
        else:
            # Simple mask: ignore zero positions
            mask = (marked_positions.abs().sum(dim=-1) == 0)
            # [batch, max_objects]

        # Cross-attend: visual features attend to marked positions
        # If all positions are masked for a sample, attention will handle it
        attended_features = self.cross_attention(
            query=visual_attended,
            key=pos_embeddings,
            value=pos_embeddings,
            key_padding_mask=mask,  # True = ignore this position
            need_weights=False
        )[0]

        attended_features = self.ln2(attended_features + visual_attended)

        # Pool visual features (mean pooling)
        pooled_features = attended_features.mean(dim=1)
        # [batch, hidden_dim]

        # Predict (x, y, is_done)
        prediction = self.predictor(pooled_features)
        # [batch, 3]

        # Split outputs
        x_raw = prediction[:, 0]
        y_raw = prediction[:, 1]
        is_done_logit = prediction[:, 2]

        # Apply activations
        # For x, y: sigmoid to [0, 1], but can output -1 if done
        # Strategy: predict in [-1, 1] range, then shift/scale
        x = torch.tanh(x_raw)  # [-1, 1]
        y = torch.tanh(y_raw)  # [-1, 1]

        # is_done: sigmoid to [0, 1]
        is_done = torch.sigmoid(is_done_logit)

        return {
            'x': x,  # [-1, 1] range
            'y': y,  # [-1, 1] range
            'is_done': is_done,  # [0, 1] probability
            'is_done_logit': is_done_logit  # raw logit for BCE loss
        }

    def predict_point(self, visual_features, marked_positions, threshold=0.5):
        """
        Inference mode: returns clean prediction.

        Args:
            visual_features: [1, num_patches, visual_feature_dim]
            marked_positions: [1, num_marked, 2]
            threshold: Threshold for is_done (default 0.5)

        Returns:
            tuple: (x, y, is_done)
                - x, y: float in [0, 1] or -1 if done
                - is_done: bool
        """
        with torch.no_grad():
            output = self.forward(visual_features, marked_positions)

            is_done = output['is_done'].item() > threshold

            if is_done:
                return -1.0, -1.0, True
            else:
                # Convert from [-1, 1] to [0, 1]
                x = (output['x'].item() + 1) / 2
                y = (output['y'].item() + 1) / 2
                return x, y, False


# ============================================================================
# Loss Functions
# ============================================================================

class PointPredictionLoss(nn.Module):
    """
    Combined loss for point prediction.

    Loss components:
        1. MSE for (x, y) coordinates when not done
        2. BCE for is_done signal
        3. Penalty for predicting valid coordinates when done
    """

    def __init__(self,
                 coord_weight: float = 1.0,
                 done_weight: float = 0.5,
                 consistency_weight: float = 0.3):
        """
        Args:
            coord_weight: Weight for coordinate MSE loss
            done_weight: Weight for is_done BCE loss
            consistency_weight: Weight for consistency loss (coordinates should be -1 when done)
        """
        super().__init__()
        self.coord_weight = coord_weight
        self.done_weight = done_weight
        self.consistency_weight = consistency_weight

    def forward(self, predictions, targets, is_done_target):
        """
        Args:
            predictions: dict from PointPredictionHead.forward()
            targets: [batch, 2] target (x, y) coordinates
                    Should be [-1, -1] when is_done_target is True
            is_done_target: [batch] boolean tensor indicating if done

        Returns:
            dict with:
                - total_loss: combined loss
                - coord_loss: MSE for coordinates
                - done_loss: BCE for is_done
                - consistency_loss: penalty for bad coordinates when done
        """
