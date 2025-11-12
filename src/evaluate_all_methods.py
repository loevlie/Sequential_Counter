#!/usr/bin/env python3
"""
Comprehensive evaluation of all counting methods on FSC147 dataset.
Tests VLM-based methods, GroundingDINO (with/without tiling), and direct VLM counting.
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from groundingdino.util.inference import load_model, load_image, predict
import supervision as sv
from torchvision.ops import nms
from typing import List, Dict, Tuple, Optional
import argparse
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import traceback
import re
import time

class CountingMethod:
    """Base class for counting methods."""

    def __init__(self, name: str):
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def count(self, image: Image.Image, category: str) -> int:
        """Count objects in image. To be implemented by subclasses."""
        raise NotImplementedError

class DirectVLMCounter(CountingMethod):
    """Direct VLM counting - just ask the model to count."""

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
        super().__init__("Direct VLM")
        print(f"Loading VLM model for {self.name}...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def count(self, image: Image.Image, category: str) -> int:
        """Simple direct counting."""
        prompt = f"""Count all the {category} in this image carefully.
Look at every part of the image and count each {category} exactly once.
Provide ONLY the total number as an integer."""

        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': prompt}
            ]
        }]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs = []
        for message in messages:
            if isinstance(message['content'], list):
                for item in message['content']:
                    if isinstance(item, dict) and item.get('type') == 'image':
                        image_inputs.append(item['image'])

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in
            zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return self._extract_number(response)

    def _extract_number(self, text: str) -> int:
        """Extract number from response."""
        numbers = re.findall(r'\d+', text.strip())
        if numbers:
            return int(numbers[0])
        return 0

class VLMDenseGridCounter(CountingMethod):
    """VLM with dense grid strategy and correction."""

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
        super().__init__("VLM Dense Grid")
        print(f"Loading VLM model for {self.name}...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def count(self, image: Image.Image, category: str) -> int:
        """Dense grid with correction - our best performing method."""
        width, height = image.size

        # First get global count for validation
        global_count = self._count_region(image, category, "global")

        # Create 4x4 grid with 15% overlap
        grid_counts = []
        overlap = 0.15
        grid_size = 4

        cell_width = int(width / (grid_size - overlap * (grid_size - 1)))
        cell_height = int(height / (grid_size - overlap * (grid_size - 1)))
        step_x = int(cell_width * (1 - overlap))
        step_y = int(cell_height * (1 - overlap))

        for i in range(grid_size):
            for j in range(grid_size):
                x1 = min(j * step_x, width - cell_width)
                y1 = min(i * step_y, height - cell_height)
                x2 = min(x1 + cell_width, width)
                y2 = min(y1 + cell_height, height)

                crop = image.crop((x1, y1, x2, y2))
                count = self._count_region(crop, category, "grid")
                grid_counts.append(count)

        # Calculate corrected sum
        grid_sum = sum(grid_counts)
        overlap_factor = 1 - (overlap * 0.3)  # Empirically determined
        corrected = int(grid_sum * overlap_factor)

        # Validate against global
        if abs(corrected - global_count) > 20:
            final = (corrected + global_count) // 2
        else:
            final = corrected

        # Apply overcount correction if needed
        if final > 85:  # Likely overcounting
            final = int(final * 0.9)

        return final

    def _count_region(self, image: Image.Image, category: str, context: str) -> int:
        """Count objects in a region."""
        if context == "global":
            prompt = f"Count ALL the {category} in this entire image. Provide ONLY the number."
        else:
            prompt = f"Count the {category} in this image region. Provide ONLY the number."

        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': prompt}
            ]
        }]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs = []
        for message in messages:
            if isinstance(message['content'], list):
                for item in message['content']:
                    if isinstance(item, dict) and item.get('type') == 'image':
                        image_inputs.append(item['image'])

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in
            zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return self._extract_number(response)

    def _extract_number(self, text: str) -> int:
        """Extract number from response."""
        numbers = re.findall(r'\d+', text.strip())
        if numbers:
            return int(numbers[0])
        return 0

class VLMAdaptiveCounter(CountingMethod):
    """VLM with adaptive validation strategy."""

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
        super().__init__("VLM Adaptive")
        print(f"Loading VLM model for {self.name}...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def count(self, image: Image.Image, category: str) -> int:
        """Adaptive counting with validation."""
        width, height = image.size

        # Multiple global assessments
        global_counts = []
        prompts = [
            f"Count ALL the {category} in this image. Provide ONLY the number.",
            f"Please carefully count every single {category} in this image. Provide ONLY the count.",
            f"Look at this densely packed image and count each {category}. Provide ONLY the number."
        ]

        for prompt in prompts:
            count = self._count_with_prompt(image, prompt)
            global_counts.append(count)

        # Use median for robustness
        global_count = int(np.median(global_counts))

        # Adaptive grid based on density
        if global_count < 10:
            grid_size = 2
        elif global_count < 50:
            grid_size = 3
        else:
            grid_size = 4

        # Grid counting with 10% overlap
        overlap = 0.1
        cell_width = int(width / (grid_size - overlap * (grid_size - 1)))
        cell_height = int(height / (grid_size - overlap * (grid_size - 1)))
        step_x = int(cell_width * (1 - overlap))
        step_y = int(cell_height * (1 - overlap))

        grid_counts = []
        for i in range(grid_size):
            for j in range(grid_size):
                x1 = min(j * step_x, width - cell_width)
                y1 = min(i * step_y, height - cell_height)
                x2 = min(x1 + cell_width, width)
                y2 = min(y1 + cell_height, height)

                crop = image.crop((x1, y1, x2, y2))
                count = self._count_with_prompt(crop, f"Count the {category} in this region. Provide ONLY the number.")
                grid_counts.append(count)

        # Calculate final count
        grid_sum = sum(grid_counts)
        overlap_factor = 0.1 * (grid_size - 1) / grid_size
        expected_sum = int(grid_sum * (1 - overlap_factor))

        # Weighted combination based on agreement
        if abs(expected_sum - global_count) > 15:
            weight_global = 0.6
            weight_grid = 0.4
        else:
            weight_global = 0.3
            weight_grid = 0.7

        final_count = int(weight_global * global_count + weight_grid * expected_sum)

        # Bounds checking
        if final_count > 95:
            final_count = min(final_count, int(global_count * 1.1))

        return final_count

    def _count_with_prompt(self, image: Image.Image, prompt: str) -> int:
        """Count with specific prompt."""
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': prompt}
            ]
        }]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs = []
        for message in messages:
            if isinstance(message['content'], list):
                for item in message['content']:
                    if isinstance(item, dict) and item.get('type') == 'image':
                        image_inputs.append(item['image'])

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in
            zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        numbers = re.findall(r'\d+', response.strip())
        if numbers:
            return int(numbers[0])
        return 0

class GroundingDINOCounter(CountingMethod):
    """GroundingDINO without tiling."""

    def __init__(self):
        super().__init__("GroundingDINO")
        print(f"Loading GroundingDINO model...")
        self.model = load_model(
            "/home/denny-loevlie/.local/lib/python3.10/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "weights/groundingdino_swint_ogc.pth"
        )

    def count(self, image: Image.Image, category: str) -> int:
        """Count using GroundingDINO."""
        # Save image temporarily for GroundingDINO
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Prepare for GroundingDINO
            image_source, image_tensor = load_image(tmp_path)

            # Predict
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_tensor,
                caption=category,
                box_threshold=0.35,
                text_threshold=0.25,
                device=self.device
            )

            # Apply NMS using torchvision
            h, w = image.size
            boxes = boxes * torch.tensor([w, h, w, h])
            nms_idx = nms(boxes, logits, 0.5)
            boxes = boxes[nms_idx]

            return len(boxes)
        finally:
            # Clean up temp file
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

class GroundingDINOTiledCounter(CountingMethod):
    """GroundingDINO with tiling for better detection."""

    def __init__(self):
        super().__init__("GroundingDINO Tiled")
        print(f"Loading GroundingDINO model...")
        self.model = load_model(
            "/home/denny-loevlie/.local/lib/python3.10/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "weights/groundingdino_swint_ogc.pth"
        )

    def count(self, image: Image.Image, category: str) -> int:
        """Count using tiled GroundingDINO."""
        width, height = image.size
        all_boxes = []

        # Use 3x3 tiling with overlap
        tile_size = 3
        overlap = 0.2

        cell_width = int(width / (tile_size - overlap * (tile_size - 1)))
        cell_height = int(height / (tile_size - overlap * (tile_size - 1)))
        step_x = int(cell_width * (1 - overlap))
        step_y = int(cell_height * (1 - overlap))

        for i in range(tile_size):
            for j in range(tile_size):
                x1 = min(j * step_x, width - cell_width)
                y1 = min(i * step_y, height - cell_height)
                x2 = min(x1 + cell_width, width)
                y2 = min(y1 + cell_height, height)

                # Crop and detect
                crop = image.crop((x1, y1, x2, y2))

                # Save crop temporarily for GroundingDINO
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    crop.save(tmp.name)
                    tmp_path = tmp.name

                try:
                    image_source, image_tensor = load_image(tmp_path)

                    boxes, logits, phrases = predict(
                        model=self.model,
                        image=image_tensor,
                        caption=category,
                        box_threshold=0.35,
                        text_threshold=0.25,
                        device=self.device
                    )

                    # Convert to global coordinates
                    h, w = crop.size
                    boxes = boxes * torch.tensor([w, h, w, h])

                    for box, logit in zip(boxes, logits):
                        x1_box, y1_box, x2_box, y2_box = box.tolist()
                        # Convert to global coordinates
                        global_box = [
                            x1 + x1_box,
                            y1 + y1_box,
                            x1 + x2_box,
                            y1 + y2_box,
                            logit.item()
                        ]
                        all_boxes.append(global_box)
                finally:
                    # Clean up temp file
                    import os
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

        # Apply global NMS
        if all_boxes:
            boxes_tensor = torch.tensor([b[:4] for b in all_boxes])
            scores_tensor = torch.tensor([b[4] for b in all_boxes])
            nms_idx = nms(boxes_tensor, scores_tensor, 0.5)
            final_boxes = [all_boxes[i] for i in nms_idx.tolist()]
            return len(final_boxes)

        return 0

class VLMHybridCounter(CountingMethod):
    """VLM with hybrid strategy (original best)."""

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
        super().__init__("VLM Hybrid")
        print(f"Loading VLM model for {self.name}...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def count(self, image: Image.Image, category: str) -> int:
        """Hybrid counting approach."""
        width, height = image.size

        # Global assessment
        global_count = self._count_with_prompt(
            image,
            f"Count ALL the {category} in this entire image. Consider objects that are fully visible, partially visible, overlapping, in background or foreground. Provide ONLY the number."
        )

        # 3x3 grid with 25% overlap
        grid_counts = []
        overlap = 0.25
        grid_size = 3

        cell_width = int(width / (grid_size - overlap * (grid_size - 1)))
        cell_height = int(height / (grid_size - overlap * (grid_size - 1)))
        step_x = int(cell_width * (1 - overlap))
        step_y = int(cell_height * (1 - overlap))

        for i in range(grid_size):
            for j in range(grid_size):
                x1 = min(j * step_x, width - cell_width)
                y1 = min(i * step_y, height - cell_height)
                x2 = min(x1 + cell_width, width)
                y2 = min(y1 + cell_height, height)

                crop = image.crop((x1, y1, x2, y2))
                count = self._count_with_prompt(crop, f"Count the {category} in this region. Provide ONLY the number.")
                grid_counts.append(count)

        # Aggregate
        grid_sum = sum(grid_counts)
        detailed_count = int(grid_sum * 0.7)  # Account for 25% overlap

        # Combine with weights
        if global_count > 0 and detailed_count > 0:
            coverage = len(grid_counts) / 9.0
            final_count = int(0.3 * global_count + 0.7 * detailed_count * min(1.0, coverage + 0.2))
        else:
            final_count = max(global_count, detailed_count)

        return final_count

    def _count_with_prompt(self, image: Image.Image, prompt: str) -> int:
        """Count with specific prompt."""
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': prompt}
            ]
        }]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs = []
        for message in messages:
            if isinstance(message['content'], list):
                for item in message['content']:
                    if isinstance(item, dict) and item.get('type') == 'image':
                        image_inputs.append(item['image'])

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in
            zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        numbers = re.findall(r'\d+', response.strip())
        if numbers:
            return int(numbers[0])
        return 0

def evaluate_methods(data_root: str, split: str, max_samples: int = None):
    """Evaluate all methods on FSC147 dataset."""

    # Load FSC147 dataset files
    split_file = os.path.join(data_root, "Train_Test_Val_FSC_147.json")
    annotation_file = os.path.join(data_root, "annotation_FSC147_384.json")
    classes_file = os.path.join(data_root, "ImageClasses_FSC147.txt")

    # Load split information
    with open(split_file, 'r') as f:
        splits = json.load(f)

    # Load annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Load image classes
    image_classes = {}
    with open(classes_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                image_classes[parts[0]] = parts[1]

    # Get images for the specified split
    split_images = splits[split]

    # Sample if requested
    if max_samples:
        import random
        random.seed(42)  # For reproducibility
        split_images = random.sample(split_images, min(max_samples, len(split_images)))

    # Initialize methods
    print("Initializing counting methods...")
    methods = [
        DirectVLMCounter(),
        VLMDenseGridCounter(),
        VLMAdaptiveCounter(),
        VLMHybridCounter(),
        GroundingDINOCounter(),
        GroundingDINOTiledCounter()
    ]

    # Results storage
    results = {method.name: [] for method in methods}
    results['ground_truth'] = []
    results['image_id'] = []
    results['category'] = []

    # Process each image
    print(f"\nEvaluating {len(split_images)} images from {split} split...")

    for image_name in tqdm(split_images, desc="Processing images"):
        # Get annotation
        if image_name not in annotations:
            print(f"No annotation found for {image_name}")
            continue

        ann = annotations[image_name]

        # Load image
        image_path = os.path.join(data_root, "images_384_VarV2", image_name.replace('.jpg', '') + '.jpg')
        if not os.path.exists(image_path):
            # Try without modification
            image_path = os.path.join(data_root, "images_384_VarV2", image_name)
            if not os.path.exists(image_path):
                print(f"Skipping {image_name} - image not found")
                continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            continue

        # Get ground truth and category
        gt_count = len(ann['points'])
        category = image_classes.get(image_name, 'objects')

        results['ground_truth'].append(gt_count)
        results['image_id'].append(image_name)
        results['category'].append(category)

        # Test each method
        for method in methods:
            try:
                start_time = time.time()
                pred_count = method.count(image, category)
                elapsed = time.time() - start_time

                results[method.name].append({
                    'count': pred_count,
                    'error': pred_count - gt_count,
                    'abs_error': abs(pred_count - gt_count),
                    'squared_error': (pred_count - gt_count) ** 2,
                    'time': elapsed
                })

            except Exception as e:
                print(f"Error with {method.name} on {image_name}: {e}")
                traceback.print_exc()
                results[method.name].append({
                    'count': 0,
                    'error': -gt_count,
                    'abs_error': gt_count,
                    'squared_error': gt_count ** 2,
                    'time': 0
                })

    return results

def create_summary_table(results: Dict) -> pd.DataFrame:
    """Create summary statistics table."""

    summary_data = []

    for method_name in results:
        if method_name in ['ground_truth', 'image_id', 'category']:
            continue

        method_results = results[method_name]

        if method_results:
            mae = np.mean([r['abs_error'] for r in method_results])
            rmse = np.sqrt(np.mean([r['squared_error'] for r in method_results]))
            mean_error = np.mean([r['error'] for r in method_results])
            std_error = np.std([r['error'] for r in method_results])
            avg_time = np.mean([r['time'] for r in method_results])

            # Calculate accuracy within different thresholds
            within_5 = sum(1 for r in method_results if r['abs_error'] <= 5) / len(method_results) * 100
            within_10 = sum(1 for r in method_results if r['abs_error'] <= 10) / len(method_results) * 100
            within_20pct = sum(1 for r, gt in zip(method_results, results['ground_truth'])
                             if r['abs_error'] <= max(1, gt * 0.2)) / len(method_results) * 100

            summary_data.append({
                'Method': method_name,
                'MAE': f"{mae:.2f}",
                'RMSE': f"{rmse:.2f}",
                'Mean Error': f"{mean_error:.2f}",
                'Std Error': f"{std_error:.2f}",
                'Within 5': f"{within_5:.1f}%",
                'Within 10': f"{within_10:.1f}%",
                'Within 20%': f"{within_20pct:.1f}%",
                'Avg Time (s)': f"{avg_time:.2f}"
            })

    # Sort by MAE
    summary_data.sort(key=lambda x: float(x['MAE']))

    return pd.DataFrame(summary_data)

def save_detailed_results(results: Dict, output_path: str):
    """Save detailed results to JSON."""

    # Convert to serializable format
    output = {
        'summary': {},
        'detailed': []
    }

    # Calculate summary for each method
    for method_name in results:
        if method_name in ['ground_truth', 'image_id', 'category']:
            continue

        method_results = results[method_name]
        if method_results:
            output['summary'][method_name] = {
                'mae': np.mean([r['abs_error'] for r in method_results]),
                'rmse': np.sqrt(np.mean([r['squared_error'] for r in method_results])),
                'mean_error': np.mean([r['error'] for r in method_results]),
                'std_error': np.std([r['error'] for r in method_results]),
                'avg_time': np.mean([r['time'] for r in method_results])
            }

    # Store detailed results for each image
    for i in range(len(results['ground_truth'])):
        img_result = {
            'image_id': results['image_id'][i],
            'category': results['category'][i],
            'ground_truth': results['ground_truth'][i],
            'predictions': {}
        }

        for method_name in results:
            if method_name in ['ground_truth', 'image_id', 'category']:
                continue
            if i < len(results[method_name]):
                img_result['predictions'][method_name] = results[method_name][i]

        output['detailed'].append(img_result)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Evaluate all counting methods on FSC147")
    parser.add_argument("--data_root", type=str, default="/media/M2SSD/FSC147",
                       help="Path to FSC147 dataset root")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                       help="Dataset split to evaluate")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (for testing)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run evaluation
    results = evaluate_methods(args.data_root, args.split, args.max_samples)

    # Create summary table
    summary_df = create_summary_table(results)

    # Print results
    print("\n" + "="*100)
    print("EVALUATION RESULTS")
    print("="*100)
    print(f"Dataset: FSC147 {args.split} split")
    print(f"Images evaluated: {len(results['ground_truth'])}")
    print("\n")
    print(summary_df.to_string(index=False))
    print("="*100)

    # Save results
    csv_path = os.path.join(args.output_dir, f"summary_{args.split}_{timestamp}.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")

    json_path = os.path.join(args.output_dir, f"detailed_{args.split}_{timestamp}.json")
    save_detailed_results(results, json_path)
    print(f"Detailed results saved to: {json_path}")

    # Create markdown report
    md_path = os.path.join(args.output_dir, f"report_{args.split}_{timestamp}.md")
    with open(md_path, 'w') as f:
        f.write(f"# Counting Methods Evaluation Report\n\n")
        f.write(f"**Dataset**: FSC147 {args.split} split\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Images evaluated**: {len(results['ground_truth'])}\n\n")
        f.write("## Summary Results\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n## Metrics Explanation\n\n")
        f.write("- **MAE**: Mean Absolute Error (lower is better)\n")
        f.write("- **RMSE**: Root Mean Square Error (lower is better)\n")
        f.write("- **Mean Error**: Average signed error (shows bias)\n")
        f.write("- **Within X**: Percentage of predictions within X of ground truth\n")
        f.write("- **Within 20%**: Percentage within 20% of ground truth count\n")

    print(f"Report saved to: {md_path}")

if __name__ == "__main__":
    main()