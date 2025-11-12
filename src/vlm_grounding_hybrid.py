#!/usr/bin/env python3
"""
Hybrid VLM + GroundingDINO counting approach with attention visualization.
Combines the strengths of both methods:
- VLM for understanding context and global counting
- GroundingDINO for precise object localization
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from groundingdino.util.inference import load_model, load_image, predict, annotate
from torchvision.ops import nms
# Set matplotlib backend to non-interactive to avoid Qt issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Dict, Tuple, Optional
import argparse
import tempfile
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter


@dataclass
class DetectionResult:
    """Container for detection results."""
    boxes: np.ndarray  # [N, 4] xyxy format
    scores: np.ndarray  # [N]
    labels: List[str]  # [N]
    count: int


class HybridVLMGroundingCounter:
    """
    Hybrid counting approach that combines VLM and GroundingDINO.

    Strategy:
    1. Use VLM to get initial count and identify dense regions
    2. Use GroundingDINO to detect objects with locations
    3. Use VLM to verify ambiguous detections
    4. Combine results intelligently
    """

    def __init__(self, device="cuda"):
        self.device = device

        # Load VLM model
        print("Loading VLM model...")
        self.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

        # Load GroundingDINO model
        print("Loading GroundingDINO model...")
        self.grounding_model = load_model(
            "/home/denny-loevlie/.local/lib/python3.10/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "weights/groundingdino_swint_ogc.pth"
        )

    def count_with_visualization(
        self,
        image: Image.Image,
        category: str,
        visualize_attention: bool = True
    ) -> Dict:
        """
        Count objects using hybrid approach with visualization.
        """
        width, height = image.size

        # Step 1: Get VLM global count and density map
        print(f"\n1. Getting VLM global assessment...")
        vlm_global = self._vlm_global_assessment(image, category)

        # Step 2: Get GroundingDINO detections
        print(f"2. Running GroundingDINO detection...")
        grounding_results = self._grounding_detect(image, category)

        # Step 3: Identify problematic regions (where counts differ significantly)
        print(f"3. Identifying regions needing refinement...")
        problem_regions = self._identify_problem_regions(
            image, vlm_global, grounding_results
        )

        # Step 4: Use VLM to refine counts in problem regions
        print(f"4. Refining counts in problem regions...")
        refined_counts = self._refine_with_vlm(image, category, problem_regions)

        # Step 5: Combine all results
        print(f"5. Combining results...")
        final_result = self._combine_results(
            vlm_global, grounding_results, refined_counts
        )

        # Step 6: Visualize if requested
        visualizations = {}
        if visualize_attention:
            print(f"6. Creating visualizations...")
            visualizations = self._create_visualizations(
                image, vlm_global, grounding_results, problem_regions, final_result
            )

        return {
            'final_count': final_result['count'],
            'vlm_count': vlm_global['count'],
            'grounding_count': grounding_results.count,
            'confidence': final_result['confidence'],
            'visualizations': visualizations,
            'details': {
                'problem_regions': len(problem_regions),
                'refined_regions': len(refined_counts),
                'detection_boxes': len(grounding_results.boxes)
            }
        }

    def _vlm_global_assessment(self, image: Image.Image, category: str) -> Dict:
        """Get VLM's global count and attention map."""
        # Global count
        prompt = f"Count ALL the {category} in this image. Provide ONLY the number."
        count = self._vlm_count(image, prompt)

        # Get attention maps (if possible with current model)
        attention_maps = self._extract_attention_maps(image, prompt)

        # Density assessment
        density_prompt = f"Describe where the {category} are most densely packed in the image. Are they clustered or spread out?"
        density_response = self._vlm_query(image, density_prompt)

        return {
            'count': count,
            'attention_maps': attention_maps,
            'density_info': density_response
        }

    def _grounding_detect(self, image: Image.Image, category: str) -> DetectionResult:
        """Run GroundingDINO detection."""
        # Save image temporarily for GroundingDINO
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Load and predict
            image_source, image_tensor = load_image(tmp_path)

            boxes, logits, phrases = predict(
                model=self.grounding_model,
                image=image_tensor,
                caption=category,
                box_threshold=0.3,
                text_threshold=0.25,
                device=self.device
            )

            # Convert to image coordinates
            h, w = image.size
            boxes = boxes * torch.tensor([w, h, w, h])

            # Apply NMS
            nms_idx = nms(boxes, logits, 0.5)
            boxes = boxes[nms_idx].cpu().numpy()
            logits = logits[nms_idx].cpu().numpy()
            phrases = [phrases[i] for i in nms_idx.tolist()]

            return DetectionResult(
                boxes=boxes,
                scores=logits,
                labels=phrases,
                count=len(boxes)
            )

        finally:
            os.unlink(tmp_path)

    def _identify_problem_regions(
        self,
        image: Image.Image,
        vlm_global: Dict,
        grounding_results: DetectionResult
    ) -> List[Tuple[int, int, int, int]]:
        """
        Identify regions where VLM and GroundingDINO disagree.
        These are typically:
        1. Dense clusters where GroundingDINO misses objects
        2. Ambiguous regions where GroundingDINO over-detects
        """
        problem_regions = []
        width, height = image.size

        # If counts differ significantly, analyze the image in grids
        count_diff = abs(vlm_global['count'] - grounding_results.count)

        if count_diff > 5:  # Significant disagreement
            # Divide image into grid and check each cell
            grid_size = 3
            cell_width = width // grid_size
            cell_height = height // grid_size

            for i in range(grid_size):
                for j in range(grid_size):
                    x1 = j * cell_width
                    y1 = i * cell_height
                    x2 = min(x1 + cell_width, width)
                    y2 = min(y1 + cell_height, height)

                    # Count detections in this cell
                    cell_detections = 0
                    for box in grounding_results.boxes:
                        box_center_x = (box[0] + box[2]) / 2
                        box_center_y = (box[1] + box[3]) / 2
                        if x1 <= box_center_x < x2 and y1 <= box_center_y < y2:
                            cell_detections += 1

                    # Check if this region needs refinement
                    # (You could make this more sophisticated)
                    if cell_detections == 0 and vlm_global['count'] > grounding_results.count:
                        # Likely missing detections
                        problem_regions.append((x1, y1, x2, y2))
                    elif cell_detections > 10:  # Dense region
                        # Likely needs verification
                        problem_regions.append((x1, y1, x2, y2))

        return problem_regions

    def _refine_with_vlm(
        self,
        image: Image.Image,
        category: str,
        regions: List[Tuple[int, int, int, int]]
    ) -> Dict:
        """Use VLM to carefully count in problem regions."""
        refined_counts = {}

        for idx, (x1, y1, x2, y2) in enumerate(regions):
            crop = image.crop((x1, y1, x2, y2))

            # Multiple prompts for robustness
            prompts = [
                f"Count the {category} in this image region. Be very careful with overlapping objects.",
                f"How many individual {category} can you see? Count each one separately.",
                f"Look closely and count all {category}, including partially visible ones."
            ]

            counts = []
            for prompt in prompts:
                count = self._vlm_count(crop, prompt)
                counts.append(count)

            # Use median for robustness
            refined_count = int(np.median(counts))
            refined_counts[idx] = {
                'region': (x1, y1, x2, y2),
                'count': refined_count,
                'all_counts': counts
            }

        return refined_counts

    def _combine_results(
        self,
        vlm_global: Dict,
        grounding_results: DetectionResult,
        refined_counts: Dict
    ) -> Dict:
        """
        Intelligently combine results from all sources.
        """
        # Start with GroundingDINO as base (provides locations)
        base_count = grounding_results.count

        # Adjust based on refined regions
        adjustment = 0
        for region_data in refined_counts.values():
            region = region_data['region']
            vlm_region_count = region_data['count']

            # Count GroundingDINO detections in this region
            grounding_region_count = 0
            for box in grounding_results.boxes:
                box_center_x = (box[0] + box[2]) / 2
                box_center_y = (box[1] + box[3]) / 2
                if (region[0] <= box_center_x < region[2] and
                    region[1] <= box_center_y < region[3]):
                    grounding_region_count += 1

            # Adjust if VLM found more in this region
            if vlm_region_count > grounding_region_count:
                adjustment += (vlm_region_count - grounding_region_count)

        # Final count combines base + adjustments
        final_count = base_count + adjustment

        # Use VLM global as sanity check
        if abs(final_count - vlm_global['count']) > 10:
            # Large discrepancy - weight towards VLM
            final_count = int(0.6 * vlm_global['count'] + 0.4 * final_count)

        # Calculate confidence
        consistency = 1.0 - (abs(vlm_global['count'] - grounding_results.count) /
                            max(vlm_global['count'], grounding_results.count, 1))
        confidence = min(consistency + 0.3, 1.0)  # Boost confidence a bit

        return {
            'count': final_count,
            'confidence': confidence,
            'adjustment': adjustment
        }

    def _vlm_count(self, image: Image.Image, prompt: str) -> int:
        """Get count from VLM."""
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': prompt}
            ]
        }]

        text = self.vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs = []
        for message in messages:
            if isinstance(message['content'], list):
                for item in message['content']:
                    if isinstance(item, dict) and item.get('type') == 'image':
                        image_inputs.append(item['image'])

        inputs = self.vlm_processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.vlm_model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in
            zip(inputs.input_ids, generated_ids)
        ]
        response = self.vlm_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Extract number
        import re
        numbers = re.findall(r'\d+', response.strip())
        if numbers:
            return int(numbers[0])
        return 0

    def _vlm_query(self, image: Image.Image, prompt: str) -> str:
        """Get text response from VLM."""
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': prompt}
            ]
        }]

        text = self.vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs = []
        for message in messages:
            if isinstance(message['content'], list):
                for item in message['content']:
                    if isinstance(item, dict) and item.get('type') == 'image':
                        image_inputs.append(item['image'])

        inputs = self.vlm_processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.vlm_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in
            zip(inputs.input_ids, generated_ids)
        ]
        response = self.vlm_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response

    def _extract_attention_maps(self, image: Image.Image, prompt: str) -> Optional[np.ndarray]:
        """
        Extract attention maps from VLM if possible.
        Note: This is model-specific and may need adjustment.
        """
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': prompt}
            ]
        }]

        text = self.vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs = []
        for message in messages:
            if isinstance(message['content'], list):
                for item in message['content']:
                    if isinstance(item, dict) and item.get('type') == 'image':
                        image_inputs.append(item['image'])

        inputs = self.vlm_processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # Try to get attention weights
        try:
            with torch.no_grad():
                outputs = self.vlm_model(
                    **inputs,
                    output_attentions=True,
                    return_dict=True
                )

            # Extract cross-attention if available
            if hasattr(outputs, 'cross_attentions') and outputs.cross_attentions:
                # Get last layer attention
                attention = outputs.cross_attentions[-1]
                # Average over heads and process
                attention = attention.mean(dim=1)  # Average over heads
                # This would need more processing to map to image regions
                return attention.cpu().numpy()

        except Exception as e:
            print(f"Could not extract attention maps: {e}")

        return None

    def _create_visualizations(
        self,
        image: Image.Image,
        vlm_global: Dict,
        grounding_results: DetectionResult,
        problem_regions: List,
        final_result: Dict
    ) -> Dict:
        """Create comprehensive visualizations."""
        visualizations = {}

        # 1. Detection visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title(f"Original Image")
        axes[0, 0].axis('off')

        # GroundingDINO detections
        detection_img = image.copy()
        draw = ImageDraw.Draw(detection_img)
        for box, score in zip(grounding_results.boxes, grounding_results.scores):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            draw.text((x1, y1-10), f"{score:.2f}", fill='red')
        axes[0, 1].imshow(detection_img)
        axes[0, 1].set_title(f"GroundingDINO: {grounding_results.count} objects")
        axes[0, 1].axis('off')

        # Problem regions
        problem_img = image.copy()
        draw = ImageDraw.Draw(problem_img)
        for x1, y1, x2, y2 in problem_regions:
            draw.rectangle([x1, y1, x2, y2], outline='yellow', width=3)
        axes[0, 2].imshow(problem_img)
        axes[0, 2].set_title(f"Problem Regions: {len(problem_regions)}")
        axes[0, 2].axis('off')

        # Attention map (if available)
        if vlm_global.get('attention_maps') is not None:
            attention = vlm_global['attention_maps']
            # Process attention map (this is simplified)
            if len(attention.shape) > 2:
                attention = attention.mean(axis=0)
            axes[1, 0].imshow(attention, cmap='hot')
            axes[1, 0].set_title("VLM Attention Map")
        else:
            # Create a heatmap based on detection density
            heatmap = np.zeros((image.height, image.width))
            for box in grounding_results.boxes:
                x1, y1, x2, y2 = box.astype(int)
                heatmap[y1:y2, x1:x2] += 1

            # Gaussian blur for smoothing
            heatmap = gaussian_filter(heatmap, sigma=5)
            axes[1, 0].imshow(heatmap, cmap='hot', alpha=0.6)
            axes[1, 0].imshow(image, alpha=0.4)
            axes[1, 0].set_title("Detection Density Heatmap")
        axes[1, 0].axis('off')

        # Combined result
        final_img = image.copy()
        draw = ImageDraw.Draw(final_img)
        # Draw detections
        for box in grounding_results.boxes:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
        # Draw adjustments
        for region_data in problem_regions:
            if isinstance(region_data, tuple):
                x1, y1, x2, y2 = region_data
                draw.rectangle([x1, y1, x2, y2], outline='cyan', width=2)
        axes[1, 1].imshow(final_img)
        axes[1, 1].set_title(f"Final Result: {final_result['count']} objects")
        axes[1, 1].axis('off')

        # Results summary
        axes[1, 2].text(0.1, 0.9, f"VLM Global Count: {vlm_global['count']}",
                       transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].text(0.1, 0.7, f"GroundingDINO Count: {grounding_results.count}",
                       transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].text(0.1, 0.5, f"Adjustment: {final_result['adjustment']}",
                       transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].text(0.1, 0.3, f"Final Count: {final_result['count']}",
                       transform=axes[1, 2].transAxes, fontsize=14, weight='bold')
        axes[1, 2].text(0.1, 0.1, f"Confidence: {final_result['confidence']:.2%}",
                       transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].axis('off')

        plt.tight_layout()
        visualizations['combined'] = fig

        return visualizations


def main():
    parser = argparse.ArgumentParser(description="Hybrid VLM + GroundingDINO counting")
    parser.add_argument('--data_root', type=str, default='/media/M2SSD/FSC147',
                       help='Path to FSC147 dataset')
    parser.add_argument('--sample_idx', type=int, default=3,
                       help='Sample index to process')
    parser.add_argument('--output', type=str, default='hybrid_vlm_grounding.png',
                       help='Output visualization path')
    parser.add_argument('--output_json', type=str, default='hybrid_vlm_grounding.json',
                       help='Output JSON path')
    args = parser.parse_args()

    # Load sample from FSC147
    split_file = os.path.join(args.data_root, "Train_Test_Val_FSC_147.json")
    annotation_file = os.path.join(args.data_root, "annotation_FSC147_384.json")
    classes_file = os.path.join(args.data_root, "ImageClasses_FSC147.txt")

    with open(split_file, 'r') as f:
        split_data = json.load(f)

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    with open(classes_file, 'r') as f:
        class_lines = f.readlines()

    # Get validation samples
    val_images = split_data['val']
    if args.sample_idx >= len(val_images):
        print(f"Sample index {args.sample_idx} out of range")
        return

    image_name = val_images[args.sample_idx]
    ann = annotations[image_name]

    # Find category
    category = None
    for line in class_lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2 and parts[0] == image_name:
            category = parts[1]
            break

    if not category:
        category = "objects"

    # Load image
    image_path = os.path.join(args.data_root, "images_384_VarV2", image_name)
    image = Image.open(image_path)

    # Get ground truth count
    gt_count = len(ann['points'])

    print(f"\nProcessing {image_name}")
    print(f"Category: {category}")
    print(f"Ground Truth Count: {gt_count}")

    # Initialize hybrid counter
    counter = HybridVLMGroundingCounter()

    # Run counting with visualization
    results = counter.count_with_visualization(image, category, visualize_attention=True)

    print(f"\n" + "="*50)
    print(f"RESULTS:")
    print(f"  VLM Count: {results['vlm_count']}")
    print(f"  GroundingDINO Count: {results['grounding_count']}")
    print(f"  Final Hybrid Count: {results['final_count']}")
    print(f"  Ground Truth: {gt_count}")
    print(f"  Error: {abs(results['final_count'] - gt_count)}")
    print(f"  Confidence: {results['confidence']:.2%}")
    print(f"="*50)

    # Save visualization
    if results['visualizations']:
        results['visualizations']['combined'].savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to {args.output}")

    # Save JSON results
    json_results = {
        'image': image_name,
        'category': category,
        'ground_truth': gt_count,
        'final_count': results['final_count'],
        'vlm_count': results['vlm_count'],
        'grounding_count': results['grounding_count'],
        'confidence': results['confidence'],
        'error': abs(results['final_count'] - gt_count),
        'details': results['details']
    }

    with open(args.output_json, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()