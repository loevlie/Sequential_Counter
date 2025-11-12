#!/usr/bin/env python3
"""
VLM GradCAM Visualization for Counting
Uses gradient-based attention to visualize what image regions the VLM focuses on
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import Optional, Tuple, Dict
import argparse
import os
from scipy.ndimage import gaussian_filter
from dataset_fsc147 import FSC147Dataset

class VLMGradCAM:
    """GradCAM visualization for Vision-Language Models."""

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
        """Initialize the VLM model for GradCAM."""
        print(f"Loading VLM model: {model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()

        # Store gradients and activations
        self.gradients = None
        self.activations = None

        print("VLM model loaded!")

    def _register_hooks(self):
        """Register hooks to capture gradients and activations from vision encoder."""

        def forward_hook(module, input, output):
            """Capture activations."""
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            """Capture gradients."""
            self.gradients = grad_output[0]

        # Find the vision encoder's last layer
        # For Qwen3-VL, we need to hook into the vision model
        target_layer = None

        # Try to find vision transformer layers
        if hasattr(self.model, 'visual'):
            # Look for the last transformer block
            if hasattr(self.model.visual, 'blocks'):
                target_layer = self.model.visual.blocks[-1]
        elif hasattr(self.model, 'vision_model'):
            if hasattr(self.model.vision_model, 'encoder'):
                if hasattr(self.model.vision_model.encoder, 'layers'):
                    target_layer = self.model.vision_model.encoder.layers[-1]

        # Fallback: try to find any vision-related module
        if target_layer is None:
            for name, module in self.model.named_modules():
                if 'vision' in name.lower() or 'visual' in name.lower():
                    if 'layer' in name.lower() or 'block' in name.lower():
                        target_layer = module

        if target_layer is not None:
            print(f"Registered hooks on: {target_layer.__class__.__name__}")
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_full_backward_hook(backward_hook)
            return True
        else:
            print("Warning: Could not find vision encoder layer for hooks")
            return False

    def generate_gradcam(
        self,
        image: Image.Image,
        category: str,
        crop_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[np.ndarray]:
        """
        Generate GradCAM heatmap for the image.

        Args:
            image: Input PIL image
            category: Object category being counted
            crop_bbox: Optional crop region (x1, y1, x2, y2)

        Returns:
            GradCAM heatmap as numpy array
        """
        # Crop if needed
        if crop_bbox:
            x1, y1, x2, y2 = crop_bbox
            img_to_process = image.crop(crop_bbox)
            image_size = (x2 - x1, y2 - y1)
        else:
            img_to_process = image
            image_size = image.size

        # Create counting prompt
        prompt = f"Count the number of {category} in this image. Provide only the number."

        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_to_process},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_prompt],
            images=[img_to_process],
            return_tensors="pt",
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in inputs.items()}

        # Enable gradients
        for param in self.model.parameters():
            param.requires_grad = True

        # Register hooks
        hooks_registered = self._register_hooks()
        if not hooks_registered:
            print("Warning: Hooks not registered, trying alternative approach")
            return self._generate_attention_rollout(inputs, image_size)

        # Forward pass
        self.model.zero_grad()

        with torch.enable_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            logits = outputs.logits

            # Get the prediction for the last token
            # This is where the model outputs the count
            last_token_logits = logits[0, -1, :]

            # Use max logit as the signal to backprop
            score = last_token_logits.max()

            # Backward pass
            score.backward()

        # Generate GradCAM
        if self.gradients is not None and self.activations is not None:
            print(f"Debug - Gradients shape: {self.gradients.shape}")
            print(f"Debug - Activations shape: {self.activations.shape}")

            # Handle different tensor shapes
            grads = self.gradients
            acts = self.activations

            # Get batch dimension
            if grads.dim() == 4:
                # [batch, seq, hidden] or [batch, heads, seq, hidden]
                grads = grads[0]  # Remove batch
            if acts.dim() == 4:
                acts = acts[0]

            # Pool gradients over sequence dimension
            if grads.dim() == 3:
                # [heads, seq, hidden] or [seq, heads, hidden]
                pooled_gradients = grads.mean(dim=(0, 1))  # Average over first two dims
            elif grads.dim() == 2:
                # [seq, hidden]
                pooled_gradients = grads.mean(dim=0)
            else:
                pooled_gradients = grads.mean(dim=tuple(range(grads.dim()-1)))

            # Get activations in right shape
            if acts.dim() == 3:
                # [heads, seq, hidden] or [seq, heads, hidden]
                # Try to identify which dimension is sequence
                acts = acts.mean(dim=0) if acts.shape[0] < acts.shape[1] else acts.mean(dim=1)
            elif acts.dim() == 2:
                # [seq, hidden] - already good
                pass
            else:
                # Flatten to [seq, hidden]
                acts = acts.reshape(-1, acts.shape[-1])

            # Weight by gradients
            weighted_acts = acts * pooled_gradients.unsqueeze(0)

            # Create heatmap by averaging over feature dimension
            heatmap = weighted_acts.mean(dim=-1)
            heatmap = F.relu(heatmap)  # ReLU to keep only positive contributions
            heatmap = heatmap.detach().cpu().numpy()

            # Try to reshape to 2D
            n_tokens = len(heatmap)
            print(f"Debug - Heatmap has {n_tokens} tokens")

            # Try common grid sizes (perfect squares)
            reshaped = False
            for size in [14, 16, 24, 27, 28, 32, 36, 48]:
                if n_tokens == size * size:
                    heatmap = heatmap.reshape(size, size)
                    print(f"Debug - Reshaped to {size}x{size}")
                    reshaped = True
                    break

            # If not a perfect square, try common aspect ratios
            if not reshaped:
                import math
                sqrt_n = int(math.sqrt(n_tokens))
                # Try nearby rectangular grids
                for h in range(max(1, sqrt_n - 5), sqrt_n + 10):
                    if n_tokens % h == 0:
                        w = n_tokens // h
                        heatmap = heatmap.reshape(h, w)
                        print(f"Debug - Reshaped to {h}x{w}")
                        reshaped = True
                        break

            # If still 1D, keep it 1D and we'll handle it in plotting
            if not reshaped:
                print(f"Warning - Could not reshape {n_tokens} tokens to 2D grid")
                # Keep as 1D - will be handled in _plot_gradcam_overlay

            # Normalize
            if heatmap.max() > 0:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

            return heatmap
        else:
            print("Warning: Could not capture gradients/activations")
            return None

    def _generate_attention_rollout(
        self,
        inputs: Dict,
        image_size: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        Fallback: Use attention rollout to visualize attention.
        """
        print("Using attention rollout as fallback...")

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                return_dict=True
            )

        if hasattr(outputs, 'attentions') and outputs.attentions:
            # Get all attention layers
            attentions = outputs.attentions

            # Average across heads and layers
            avg_attention = None
            for attn in attentions:
                # Average over heads
                attn_heads_fused = attn.mean(dim=1)  # [batch, seq_len, seq_len]

                if avg_attention is None:
                    avg_attention = attn_heads_fused
                else:
                    # Matmul to combine attention across layers (rollout)
                    avg_attention = torch.matmul(avg_attention, attn_heads_fused)

            # Extract attention to CLS token or average
            # Take mean over all query positions
            final_attention = avg_attention.mean(dim=1)  # [batch, seq_len]
            final_attention = final_attention[0].cpu().numpy()

            # Try to extract image tokens (first N tokens)
            # Try 196 (14x14), 256 (16x16), 576 (24x24)
            for patch_count in [196, 256, 576, 729, 1024]:
                if len(final_attention) >= patch_count:
                    img_attention = final_attention[:patch_count]

                    # Try to reshape to square
                    size = int(np.sqrt(patch_count))
                    if size * size == patch_count:
                        heatmap = img_attention.reshape(size, size)
                        # Normalize
                        if heatmap.max() > 0:
                            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                        return heatmap

            print(f"Warning: Could not reshape {len(final_attention)} tokens to 2D grid")

        return None

    def visualize_counting_attention(
        self,
        image: Image.Image,
        category: str,
        strategy: str = "comparison",
        output_path: str = "gradcam_viz.png"
    ):
        """Visualize GradCAM for different counting strategies."""

        if strategy == "comparison":
            # Compare global vs center crop
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Global view
            print("Generating GradCAM for global view...")
            global_heatmap = self.generate_gradcam(image, category)
            self._plot_gradcam_overlay(axes[0], image, global_heatmap, "Global View")

            # Center crop
            width, height = image.size
            crop_size_w, crop_size_h = width // 2, height // 2
            x1 = (width - crop_size_w) // 2
            y1 = (height - crop_size_h) // 2
            x2, y2 = x1 + crop_size_w, y1 + crop_size_h

            print("Generating GradCAM for center crop...")
            crop_heatmap = self.generate_gradcam(image, category, (x1, y1, x2, y2))
            self._plot_gradcam_overlay(axes[1], image, crop_heatmap, "Center Crop", (x1, y1, x2, y2))

            # Original image
            axes[2].imshow(image)
            axes[2].set_title("Original Image")
            axes[2].axis('off')

            plt.suptitle(f"VLM GradCAM Visualization - {category.title()}")
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"\nVisualization saved to: {output_path}")

        elif strategy == "dense":
            # Dense grid visualization
            self._visualize_dense_grid(image, category, output_path)

        elif strategy == "hybrid":
            # Hybrid visualization
            self._visualize_hybrid(image, category, output_path)

    def _plot_gradcam_overlay(
        self,
        ax,
        original_image: Image.Image,
        heatmap: Optional[np.ndarray],
        title: str,
        crop_bbox: Optional[Tuple[int, int, int, int]] = None
    ):
        """Plot GradCAM heatmap overlayed on image."""

        # Show original image or crop
        if crop_bbox:
            x1, y1, x2, y2 = crop_bbox
            img_to_show = original_image.crop(crop_bbox)
        else:
            img_to_show = original_image

        ax.imshow(img_to_show)

        # Overlay heatmap if available
        if heatmap is not None:
            # Convert to float32 for scipy operations
            heatmap = heatmap.astype(np.float32)

            # Resize heatmap to image size
            from scipy.ndimage import zoom
            h, w = img_to_show.size[1], img_to_show.size[0]

            # Handle both 1D and 2D heatmaps
            if heatmap.ndim == 1:
                # 1D heatmap - reshape to approximate square grid
                import math
                n = len(heatmap)
                side = int(math.sqrt(n))
                # Truncate to perfect square or pad
                if side * side == n:
                    heatmap = heatmap.reshape(side, side)
                else:
                    # Pad to next perfect square
                    next_square = (side + 1) ** 2
                    padded = np.zeros(next_square)
                    padded[:n] = heatmap
                    heatmap = padded.reshape(side + 1, side + 1)
                print(f"Debug - Reshaped 1D heatmap to {heatmap.shape}")

            # Apply smoothing
            heatmap_smooth = gaussian_filter(heatmap, sigma=1.0)

            # Resize to image dimensions
            zoom_factors = (h / heatmap_smooth.shape[0], w / heatmap_smooth.shape[1])
            heatmap_resized = zoom(heatmap_smooth, zoom_factors, order=1)

            # Overlay as heatmap
            im = ax.imshow(heatmap_resized, cmap='jet', alpha=0.5, interpolation='bilinear')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, 'GradCAM Not Available',
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_title(title)
        ax.axis('off')

    def _visualize_dense_grid(self, image: Image.Image, category: str, output_path: str):
        """Visualize GradCAM for dense grid strategy."""
        print("\nGenerating dense grid GradCAM visualization...")

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        width, height = image.size

        # 3x3 grid
        crop_w, crop_h = width // 3, height // 3

        for row in range(3):
            for col in range(3):
                ax = axes[row, col]

                x1 = col * crop_w
                y1 = row * crop_h
                x2 = min(x1 + crop_w, width)
                y2 = min(y1 + crop_h, height)

                print(f"Processing grid cell ({row},{col})...")
                heatmap = self.generate_gradcam(image, category, (x1, y1, x2, y2))
                self._plot_gradcam_overlay(ax, image, heatmap, f"Cell ({row},{col})", (x1, y1, x2, y2))

        plt.suptitle(f"Dense Grid GradCAM - {category.title()}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nVisualization saved to: {output_path}")

    def _visualize_hybrid(self, image: Image.Image, category: str, output_path: str):
        """Visualize GradCAM for hybrid strategy."""
        print("\nGenerating hybrid GradCAM visualization...")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        width, height = image.size

        # Global
        print("Processing global view...")
        global_heatmap = self.generate_gradcam(image, category)
        self._plot_gradcam_overlay(axes[0, 0], image, global_heatmap, "Global View")

        # 4 quadrants
        mid_x, mid_y = width // 2, height // 2
        quadrants = [
            (0, 0, mid_x, mid_y, "Top-Left"),
            (mid_x, 0, width, mid_y, "Top-Right"),
            (0, mid_y, mid_x, height, "Bottom-Left"),
            (mid_x, mid_y, width, height, "Bottom-Right")
        ]

        for idx, (x1, y1, x2, y2, label) in enumerate(quadrants):
            row = (idx + 1) // 3
            col = (idx + 1) % 3

            print(f"Processing {label}...")
            heatmap = self.generate_gradcam(image, category, (x1, y1, x2, y2))
            self._plot_gradcam_overlay(axes[row, col], image, heatmap, label, (x1, y1, x2, y2))

        # Original
        axes[1, 2].imshow(image)
        axes[1, 2].set_title("Original Image")
        axes[1, 2].axis('off')

        plt.suptitle(f"Hybrid GradCAM - {category.title()}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nVisualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize VLM attention using GradCAM")
    parser.add_argument("--data_root", type=str, required=True, help="Path to FSC147 dataset")
    parser.add_argument("--sample_idx", type=int, default=3, help="Sample index to visualize")
    parser.add_argument("--strategy", type=str, default="comparison",
                       choices=["comparison", "dense", "hybrid"],
                       help="Visualization strategy")
    parser.add_argument("--output_dir", type=str, default="gradcam_visualizations",
                       help="Output directory for visualizations")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print("Loading FSC147 dataset...")
    dataset = FSC147Dataset(args.data_root, split='val')

    # Get sample
    image, sorted_points, metadata = dataset[args.sample_idx]
    category = metadata['object_type']
    gt_count = metadata['num_objects']

    print(f"\nSample {args.sample_idx}:")
    print(f"  Category: {category}")
    print(f"  Ground Truth Count: {gt_count}")
    print(f"  Image Size: {image.size}")

    # Initialize visualizer
    visualizer = VLMGradCAM()

    # Generate visualization
    output_path = os.path.join(args.output_dir, f'gradcam_{args.strategy}_{args.sample_idx}.png')
    visualizer.visualize_counting_attention(image, category, args.strategy, output_path)

    print(f"\n{'='*80}")
    print("GradCAM visualization complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
