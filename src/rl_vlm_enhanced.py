#!/usr/bin/env python3
"""
Enhanced VLM-based counting using fine-grained adaptive cropping and improved prompting.
Combines multiple strategies to achieve better counting accuracy.
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import List, Dict, Tuple, Optional
import argparse
from dataclasses import dataclass
import re

@dataclass
class CropRegion:
    """Represents a crop region with its count and metadata."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    count: int
    confidence: float
    strategy: str
    level: int
    overlap_factor: float = 0.0

class EnhancedVLMCounter:
    """Enhanced VLM-based counting with multiple strategies."""

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
        """Initialize the VLM counter."""
        print(f"Loading VLM model: {model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and processor
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("VLM model loaded!")

    def count_objects(self, image: Image.Image, category: str, strategy: str = "hybrid") -> Dict:
        """
        Count objects using specified strategy.

        Args:
            image: Input PIL image
            category: Object category to count
            strategy: Counting strategy ('full_image', 'hybrid', 'dense_grid', 'adaptive_hierarchical',
                     'dense_with_validation', 'dense_explicit_overlap')

        Returns:
            Dictionary with counting results
        """
        if strategy == "full_image":
            return self._full_image_counting(image, category)
        elif strategy == "hybrid":
            return self._hybrid_counting(image, category)
        elif strategy == "dense_grid":
            return self._dense_grid_counting(image, category)
        elif strategy == "adaptive_hierarchical":
            return self._adaptive_hierarchical_counting(image, category)
        elif strategy == "dense_with_validation":
            return self._dense_with_validation_counting(image, category)
        elif strategy == "dense_explicit_overlap":
            return self._dense_explicit_overlap_counting(image, category)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _full_image_counting(self, image: Image.Image, category: str) -> Dict:
        """
        Simple baseline: ask VLM to count objects in the full image with a single prompt.
        No cropping, no grid, just one direct question.
        """
        width, height = image.size

        print("\n" + "="*80)
        print("Full Image Single Prompt VLM Counting")
        print("="*80)

        # Ask VLM for count on full image
        print("\nAsking VLM to count all objects in the full image...")
        count = self._count_with_context(image, category, "global")
        print(f"  Count: {count}")

        # Create single crop region for consistency with other strategies
        crop = CropRegion(
            bbox=(0, 0, width, height),
            count=count,
            confidence=1.0,
            strategy="full_image",
            level=0
        )

        return {
            "count": count,
            "crops": [crop],
            "strategy": "full_image"
        }

    def _hybrid_counting(self, image: Image.Image, category: str) -> Dict:
        """Hybrid approach combining multiple strategies."""
        width, height = image.size

        print("\n" + "="*80)
        print("Enhanced Hybrid VLM Counting")
        print("="*80)

        all_crops = []

        # Phase 1: Global assessment with improved prompt
        print("\nPhase 1: Global assessment")
        global_count = self._count_with_context(image, category, "global")
        print(f"  Global count: {global_count}")

        all_crops.append(CropRegion(
            bbox=(0, 0, width, height),
            count=global_count,
            confidence=0.8,
            strategy="global",
            level=0
        ))

        # Phase 2: Dense overlapping grid (3x3 with 25% overlap)
        print("\nPhase 2: Dense overlapping grid")
        grid_crops = self._create_overlapping_grid(image, category, 3, 3, overlap=0.25)
        all_crops.extend(grid_crops)

        # Phase 3: Adaptive refinement on high-density areas
        print("\nPhase 3: Adaptive refinement")
        dense_regions = self._identify_dense_regions(grid_crops, width, height)
        for region in dense_regions:
            refined_crops = self._refine_region(image, category, region)
            all_crops.extend(refined_crops)

        # Phase 4: Edge scanning for partial objects
        print("\nPhase 4: Edge scanning")
        edge_crops = self._scan_edges(image, category)
        all_crops.extend(edge_crops)

        # Phase 5: Intelligent aggregation
        print("\nPhase 5: Aggregating counts")
        final_count = self._aggregate_counts(all_crops, width, height)

        return {
            "count": final_count,
            "crops": all_crops,
            "strategy": "hybrid",
            "global_count": global_count
        }

    def _count_with_context(self, image: Image.Image, category: str, context: str) -> int:
        """Count objects with context-aware prompting."""
        # Different prompts based on context
        if context == "global":
            prompt = f"""Look at this image carefully. Count ALL the {category} you can see in the entire image.
Consider objects that are:
- Fully visible
- Partially visible at edges
- Overlapping with other objects
- In the background or foreground

Provide ONLY the total number as an integer."""
        elif context == "dense":
            prompt = f"""This is a densely packed region with many {category}.
Count each individual {category} carefully, even if they overlap.
Look for:
- Object boundaries and edges
- Color/texture differences between objects
- Shadows that indicate separate objects

Provide ONLY the count as an integer."""
        elif context == "edge":
            prompt = f"""Focus on the edge/border area of this image crop.
Count any {category} that are partially visible or cut off by the image boundary.

Provide ONLY the count as an integer."""
        else:
            prompt = f"Count the number of {category} in this image. Provide ONLY the number."

        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': prompt}
            ]
        }]

        # Process with model
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs if video_inputs else None,
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

        # Extract number from response
        return self._extract_number(response)

    def _create_overlapping_grid(self, image: Image.Image, category: str,
                                rows: int, cols: int, overlap: float) -> List[CropRegion]:
        """Create overlapping grid of crops."""
        width, height = image.size
        crops = []

        # Calculate cell size with overlap
        cell_width = int(width / (cols - overlap * (cols - 1)))
        cell_height = int(height / (rows - overlap * (rows - 1)))
        step_x = int(cell_width * (1 - overlap))
        step_y = int(cell_height * (1 - overlap))

        print(f"  Grid: {rows}x{cols}, overlap: {overlap*100}%")
        print(f"  Cell size: {cell_width}x{cell_height}")

        for i in range(rows):
            for j in range(cols):
                x1 = min(j * step_x, width - cell_width)
                y1 = min(i * step_y, height - cell_height)
                x2 = min(x1 + cell_width, width)
                y2 = min(y1 + cell_height, height)

                crop = image.crop((x1, y1, x2, y2))
                count = self._count_with_context(crop, category, "dense" if i == 1 and j == 1 else "normal")

                crops.append(CropRegion(
                    bbox=(x1, y1, x2, y2),
                    count=count,
                    confidence=0.7,
                    strategy="grid",
                    level=1,
                    overlap_factor=overlap
                ))
                print(f"    Region ({i},{j}): {count}")

        return crops

    def _identify_dense_regions(self, grid_crops: List[CropRegion],
                               width: int, height: int) -> List[Tuple[int, int, int, int]]:
        """Identify regions with high object density for refinement."""
        dense_regions = []

        # Find crops with above-average counts
        counts = [c.count for c in grid_crops]
        if not counts:
            return []

        avg_count = np.mean(counts)
        std_count = np.std(counts)
        threshold = avg_count + 0.5 * std_count

        for crop in grid_crops:
            if crop.count > threshold:
                # Expand region slightly for context
                x1, y1, x2, y2 = crop.bbox
                margin = 20
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(width, x2 + margin)
                y2 = min(height, y2 + margin)
                dense_regions.append((x1, y1, x2, y2))
                print(f"  Dense region identified: {crop.bbox}, count: {crop.count}")

        return dense_regions

    def _refine_region(self, image: Image.Image, category: str,
                       region: Tuple[int, int, int, int]) -> List[CropRegion]:
        """Refine counting in a dense region with finer grid."""
        x1, y1, x2, y2 = region
        region_crop = image.crop(region)
        crops = []

        # Use 2x2 fine grid for dense regions
        fine_crops = self._create_overlapping_grid(region_crop, category, 2, 2, overlap=0.3)

        # Adjust coordinates to global image space
        for crop in fine_crops:
            bx1, by1, bx2, by2 = crop.bbox
            crop.bbox = (x1 + bx1, y1 + by1, x1 + bx2, y1 + by2)
            crop.level = 2
            crop.strategy = "refinement"
            crops.append(crop)

        return crops

    def _scan_edges(self, image: Image.Image, category: str) -> List[CropRegion]:
        """Scan image edges for partial objects."""
        width, height = image.size
        crops = []
        edge_width = min(100, width // 4)
        edge_height = min(100, height // 4)

        edges = [
            ("top", (0, 0, width, edge_height)),
            ("bottom", (0, height - edge_height, width, height)),
            ("left", (0, 0, edge_width, height)),
            ("right", (width - edge_width, 0, width, height))
        ]

        for edge_name, bbox in edges:
            crop = image.crop(bbox)
            count = self._count_with_context(crop, category, "edge")
            if count > 0:
                crops.append(CropRegion(
                    bbox=bbox,
                    count=count,
                    confidence=0.5,
                    strategy="edge",
                    level=1
                ))
                print(f"  Edge {edge_name}: {count}")

        return crops

    def _aggregate_counts(self, crops: List[CropRegion], width: int, height: int) -> int:
        """Intelligently aggregate counts from overlapping regions."""
        if not crops:
            return 0

        # Sort crops by confidence and level
        crops.sort(key=lambda c: (c.confidence, -c.level), reverse=True)

        # Use the global count as a baseline
        global_crops = [c for c in crops if c.strategy == "global"]
        if global_crops:
            baseline = global_crops[0].count
        else:
            baseline = 0

        # Calculate coverage-weighted sum from detailed crops
        grid_crops = [c for c in crops if c.strategy in ["grid", "refinement"]]
        if grid_crops:
            # Create a density map
            density_map = np.zeros((height, width))
            weight_map = np.zeros((height, width))

            for crop in grid_crops:
                x1, y1, x2, y2 = crop.bbox
                area = (x2 - x1) * (y2 - y1)
                if area > 0:
                    density = crop.count / area
                    # Weighted by confidence and inversely by overlap
                    weight = crop.confidence * (1 - crop.overlap_factor * 0.5)

                    density_map[y1:y2, x1:x2] += density * weight
                    weight_map[y1:y2, x1:x2] += weight

            # Normalize by weights
            mask = weight_map > 0
            density_map[mask] /= weight_map[mask]

            # Estimate total count
            detailed_count = int(np.sum(density_map))
        else:
            detailed_count = baseline

        # Combine baseline and detailed with confidence weighting
        if baseline > 0 and detailed_count > 0:
            # Weight detailed count more if we have good coverage
            coverage = len(grid_crops) / 9.0  # Assuming 3x3 grid
            final_count = int(0.3 * baseline + 0.7 * detailed_count * min(1.0, coverage + 0.2))
        else:
            final_count = max(baseline, detailed_count)

        # Add edge corrections
        edge_crops = [c for c in crops if c.strategy == "edge"]
        edge_correction = sum(c.count for c in edge_crops) * 0.3  # Partial weight for edges

        final_count = int(final_count + edge_correction)

        print(f"\n  Baseline (global): {baseline}")
        print(f"  Detailed (grid): {detailed_count}")
        print(f"  Edge correction: {edge_correction:.1f}")
        print(f"  Final count: {final_count}")

        return final_count

    def _dense_grid_counting(self, image: Image.Image, category: str) -> Dict:
        """Dense grid approach with minimal overlap."""
        width, height = image.size

        print("\n" + "="*80)
        print("Dense Grid VLM Counting")
        print("="*80)

        # Use 4x4 grid with 10% overlap
        crops = self._create_overlapping_grid(image, category, 4, 4, overlap=0.1)

        # Simple sum with overlap correction
        total = sum(c.count for c in crops)
        # Correct for overlap (approximate)
        corrected = int(total * 0.85)  # 15% reduction for 10% overlap

        print(f"\nRaw sum: {total}")
        print(f"Corrected for overlap: {corrected}")

        return {
            "count": corrected,
            "crops": crops,
            "strategy": "dense_grid"
        }

    def _adaptive_hierarchical_counting(self, image: Image.Image, category: str) -> Dict:
        """Adaptive hierarchical approach."""
        width, height = image.size

        print("\n" + "="*80)
        print("Adaptive Hierarchical VLM Counting")
        print("="*80)

        all_crops = []

        # Level 0: Global
        global_count = self._count_with_context(image, category, "global")
        print(f"Level 0 (global): {global_count}")

        all_crops.append(CropRegion(
            bbox=(0, 0, width, height),
            count=global_count,
            confidence=1.0,
            strategy="hierarchical",
            level=0
        ))

        # Decide on granularity based on global count
        if global_count < 10:
            # Few objects, use 2x2
            grid_size = 2
        elif global_count < 50:
            # Moderate, use 3x3
            grid_size = 3
        else:
            # Many objects, use 4x4
            grid_size = 4

        print(f"Using {grid_size}x{grid_size} grid based on density")

        # Level 1: Adaptive grid
        grid_crops = self._create_overlapping_grid(
            image, category, grid_size, grid_size, overlap=0.15
        )
        all_crops.extend(grid_crops)

        # Aggregate with hierarchy weighting
        grid_sum = sum(c.count for c in grid_crops)
        overlap_factor = 0.15 * (grid_size - 1) / grid_size
        corrected_sum = int(grid_sum * (1 - overlap_factor))

        # Weighted average between levels
        final_count = int(0.4 * global_count + 0.6 * corrected_sum)

        print(f"\nGrid sum: {grid_sum}")
        print(f"Corrected sum: {corrected_sum}")
        print(f"Final weighted: {final_count}")

        return {
            "count": final_count,
            "crops": all_crops,
            "strategy": "adaptive_hierarchical"
        }

    def _dense_explicit_overlap_counting(self, image: Image.Image, category: str) -> Dict:
        """
        Dense grid with explicit overlap double-count detection.
        Shows VLM the overlapping regions and asks it to identify double-counted objects.
        """
        width, height = image.size

        print("\n" + "="*80)
        print("Dense Grid with Explicit Overlap Detection")
        print("="*80)

        all_crops = []
        overlap_regions = []

        # Phase 1: Dense grid counting (3x3 with 25% overlap)
        print("\nPhase 1: Dense grid counting (3×3 with 25% overlap)")
        grid_size = 3
        overlap = 0.25

        dense_crops = self._create_overlapping_grid(image, category, grid_size, grid_size, overlap=overlap)
        all_crops.extend(dense_crops)

        raw_sum = sum(c.count for c in dense_crops)
        print(f"  Raw sum from dense grid: {raw_sum}")

        # Phase 2: Extract and analyze overlapping regions
        print("\nPhase 2: Detecting double-counted objects in overlaps")

        cell_width = int(width / (grid_size - overlap * (grid_size - 1)))
        cell_height = int(height / (grid_size - overlap * (grid_size - 1)))
        step_x = int(cell_width * (1 - overlap))
        step_y = int(cell_height * (1 - overlap))

        overlap_width = cell_width - step_x
        overlap_height = cell_height - step_y

        double_count_total = 0

        # Horizontal overlaps (between columns)
        print("  Checking horizontal overlaps...")
        for i in range(grid_size):
            for j in range(grid_size - 1):
                # Overlap region between column j and j+1
                x1 = j * step_x + cell_width - overlap_width
                x2 = x1 + overlap_width
                y1 = i * step_y
                y2 = min(y1 + cell_height, height)

                overlap_crop = image.crop((x1, y1, x2, y2))

                # Ask VLM: "How many objects are fully or mostly contained in this overlap region?"
                prompt = f"""This is a narrow vertical strip from an overlap region between two tiles.
Count how many {category} are FULLY or MOSTLY visible in this strip.
Only count objects that would likely be counted in BOTH adjacent tiles.
Provide ONLY the count as an integer."""

                messages = [{
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': overlap_crop},
                        {'type': 'text', 'text': prompt}
                    ]
                }]

                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = self._process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs if video_inputs else None,
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

                double_count = self._extract_number(response)
                double_count_total += double_count

                if double_count > 0:
                    print(f"    H-overlap ({i},{j}->{j+1}): {double_count} objects")
                    overlap_regions.append({
                        'bbox': (x1, y1, x2, y2),
                        'count': double_count,
                        'type': 'horizontal'
                    })

        # Vertical overlaps (between rows)
        print("  Checking vertical overlaps...")
        for i in range(grid_size - 1):
            for j in range(grid_size):
                # Overlap region between row i and i+1
                x1 = j * step_x
                x2 = min(x1 + cell_width, width)
                y1 = i * step_y + cell_height - overlap_height
                y2 = y1 + overlap_height

                overlap_crop = image.crop((x1, y1, x2, y2))

                prompt = f"""This is a narrow horizontal strip from an overlap region between two tiles.
Count how many {category} are FULLY or MOSTLY visible in this strip.
Only count objects that would likely be counted in BOTH adjacent tiles.
Provide ONLY the count as an integer."""

                messages = [{
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': overlap_crop},
                        {'type': 'text', 'text': prompt}
                    ]
                }]

                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = self._process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs if video_inputs else None,
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

                double_count = self._extract_number(response)
                double_count_total += double_count

                if double_count > 0:
                    print(f"    V-overlap ({i}->{i+1},{j}): {double_count} objects")
                    overlap_regions.append({
                        'bbox': (x1, y1, x2, y2),
                        'count': double_count,
                        'type': 'vertical'
                    })

        # Phase 3: Compute final count
        print(f"\n  Total double-counted objects detected: {double_count_total}")
        final_count = raw_sum - double_count_total
        print(f"  Final count (raw - double): {final_count}")

        return {
            "count": final_count,
            "crops": all_crops,
            "overlap_regions": overlap_regions,
            "strategy": "dense_explicit_overlap",
            "raw_sum": raw_sum,
            "double_count_correction": double_count_total
        }

    def _dense_with_validation_counting(self, image: Image.Image, category: str) -> Dict:
        """
        Dense grid with global/sub-global validation.
        Combines thorough dense coverage with validation from larger patches.
        """
        width, height = image.size

        print("\n" + "="*80)
        print("Dense Grid with Validation VLM Counting")
        print("="*80)

        all_crops = []

        # Phase 1: Global validation count
        print("\nPhase 1: Global validation count")
        global_count = self._count_with_context(image, category, "global")
        print(f"  Global count: {global_count}")

        all_crops.append(CropRegion(
            bbox=(0, 0, width, height),
            count=global_count,
            confidence=0.8,
            strategy="global_validation",
            level=0
        ))

        # Phase 2: Sub-global validation (2x2 larger patches)
        print("\nPhase 2: Sub-global validation (2×2 patches)")
        subglobal_crops = self._create_overlapping_grid(image, category, 2, 2, overlap=0.1)
        subglobal_sum = sum(c.count for c in subglobal_crops)

        # Mark these as validation crops
        for crop in subglobal_crops:
            crop.strategy = "subglobal_validation"
            all_crops.append(crop)

        # Correct for overlap
        subglobal_corrected = int(subglobal_sum * 0.95)  # 5% reduction for 10% overlap
        print(f"  Sub-global raw sum: {subglobal_sum}")
        print(f"  Sub-global corrected: {subglobal_corrected}")

        # Phase 3: Dense grid counting (3x3 or 4x4 based on density)
        print("\nPhase 3: Dense grid counting")

        # Decide grid density based on validation counts
        avg_validation = (global_count + subglobal_corrected) / 2
        if avg_validation < 20:
            grid_size = 3
            overlap = 0.2
        elif avg_validation < 50:
            grid_size = 4
            overlap = 0.25
        else:
            grid_size = 5
            overlap = 0.3

        print(f"  Using {grid_size}×{grid_size} grid with {overlap*100}% overlap")

        dense_crops = self._create_overlapping_grid(image, category, grid_size, grid_size, overlap=overlap)

        # Mark as dense grid crops
        for crop in dense_crops:
            crop.strategy = "dense_grid"
            all_crops.append(crop)

        # Phase 4: Aggregate with density map
        print("\nPhase 4: Aggregating with density map")
        dense_count = self._aggregate_dense_crops(dense_crops, width, height, overlap)
        print(f"  Dense grid count: {dense_count}")

        # Phase 5: Cross-validation and fusion
        print("\nPhase 5: Cross-validation and fusion")

        # Calculate disagreement between methods
        counts = [global_count, subglobal_corrected, dense_count]
        mean_count = np.mean(counts)
        std_count = np.std(counts)

        print(f"  Global: {global_count}")
        print(f"  Sub-global: {subglobal_corrected}")
        print(f"  Dense: {dense_count}")
        print(f"  Mean: {mean_count:.1f}, Std: {std_count:.1f}")

        # Confidence-based weighting
        if std_count < mean_count * 0.15:  # Good agreement (<15% variation)
            print("  Good agreement - using weighted average")
            # Dense gets more weight when methods agree
            final_count = int(0.2 * global_count + 0.2 * subglobal_corrected + 0.6 * dense_count)
        elif std_count < mean_count * 0.3:  # Moderate agreement
            print("  Moderate agreement - balanced weighting")
            final_count = int(0.25 * global_count + 0.25 * subglobal_corrected + 0.5 * dense_count)
        else:  # Poor agreement
            print("  Poor agreement - conservative weighting")
            # Trust validation counts more when there's disagreement
            final_count = int(0.35 * global_count + 0.35 * subglobal_corrected + 0.3 * dense_count)

        print(f"  Final validated count: {final_count}")

        return {
            "count": final_count,
            "crops": all_crops,
            "strategy": "dense_with_validation",
            "global_count": global_count,
            "subglobal_count": subglobal_corrected,
            "dense_count": dense_count,
            "agreement_std": float(std_count)
        }

    def _aggregate_dense_crops(self, crops: List[CropRegion], width: int, height: int, overlap: float) -> int:
        """Aggregate dense grid crops using density map approach."""
        if not crops:
            return 0

        # Create density map
        density_map = np.zeros((height, width), dtype=np.float32)
        weight_map = np.zeros((height, width), dtype=np.float32)

        for crop in crops:
            x1, y1, x2, y2 = crop.bbox
            area = (x2 - x1) * (y2 - y1)
            if area > 0:
                density = crop.count / area
                # Weight inversely by overlap factor
                weight = crop.confidence * (1 - overlap * 0.5)

                density_map[y1:y2, x1:x2] += density * weight
                weight_map[y1:y2, x1:x2] += weight

        # Normalize overlapping regions
        mask = weight_map > 0
        density_map[mask] /= weight_map[mask]

        # Estimate total count
        total_count = int(np.sum(density_map))
        return total_count

    def _extract_number(self, text: str) -> int:
        """Extract number from VLM response."""
        # Clean the text
        text = text.strip()

        # Try to find a number
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[0])

        # Check for written numbers
        word_to_num = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
            'eighteen': 18, 'nineteen': 19, 'twenty': 20
        }

        text_lower = text.lower()
        for word, num in word_to_num.items():
            if word in text_lower:
                return num

        # Default to 0 if no number found
        print(f"Warning: Could not extract number from: {text}")
        return 0

    def _process_vision_info(self, messages):
        """Process vision information from messages."""
        image_inputs = []
        video_inputs = []

        for message in messages:
            if isinstance(message['content'], list):
                for item in message['content']:
                    if isinstance(item, dict) and item.get('type') == 'image':
                        image_inputs.append(item['image'])

        return image_inputs, video_inputs

    def visualize_results(self, image: Image.Image, result: Dict,
                         output_path: str = "enhanced_counting_result.png"):
        """Visualize counting results with crops."""
        # Create visualization
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)

        # Try to load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
            font_small = font

        # Draw crops with different colors based on strategy
        color_map = {
            "global": (0, 255, 0),
            "grid": (255, 255, 0),
            "refinement": (255, 128, 0),
            "edge": (128, 128, 255),
            "hierarchical": (255, 0, 255)
        }

        if "crops" in result:
            for crop in result["crops"]:
                if hasattr(crop, 'bbox'):
                    color = color_map.get(crop.strategy, (255, 255, 255))
                    # Draw rectangle with transparency based on level
                    alpha = 128 - crop.level * 30
                    x1, y1, x2, y2 = crop.bbox

                    # Draw border
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

                    # Draw count label
                    if crop.count > 0:
                        label = f"{crop.count}"
                        bbox = draw.textbbox((x1, y1), label, font=font_small)
                        draw.rectangle(bbox, fill=color)
                        draw.text((x1, y1), label, fill=(0, 0, 0), font=font_small)

        # Add summary text
        summary = f"Count: {result['count']} | Strategy: {result['strategy']}"
        if "global_count" in result:
            summary += f" | Global: {result['global_count']}"

        bbox = draw.textbbox((10, 10), summary, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0, 180))
        draw.text((10, 10), summary, fill=(255, 255, 255), font=font)

        img_draw.save(output_path)
        print(f"\nVisualization saved to: {output_path}")

        return img_draw


def main():
    parser = argparse.ArgumentParser(description="Enhanced VLM-based object counting")
    parser.add_argument("--image", type=str, default="images_384_VarV2/194.jpg",
                       help="Path to input image")
    parser.add_argument("--category", type=str, default="objects",
                       help="Category to count")
    parser.add_argument("--strategy", type=str, default="hybrid",
                       choices=["hybrid", "dense_grid", "adaptive_hierarchical", "dense_with_validation", "dense_explicit_overlap"],
                       help="Counting strategy")
    parser.add_argument("--output", type=str, default="enhanced_vlm_result.png",
                       help="Output visualization path")
    parser.add_argument("--output_json", type=str, default="enhanced_vlm_result.json",
                       help="Output JSON path")
    args = parser.parse_args()

    # Load image
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return

    image = Image.open(args.image).convert("RGB")
    print(f"Loaded image: {args.image}")
    print(f"Size: {image.size}")
    print(f"Category: {args.category}")

    # Initialize counter
    counter = EnhancedVLMCounter()

    # Count objects
    result = counter.count_objects(image, args.category, strategy=args.strategy)

    # Visualize results
    counter.visualize_results(image, result, args.output)

    # Save JSON results
    # Convert CropRegion objects to dicts for JSON serialization
    json_result = {
        "count": result["count"],
        "strategy": result["strategy"],
        "crops": [
            {
                "bbox": crop.bbox,
                "count": crop.count,
                "confidence": crop.confidence,
                "strategy": crop.strategy,
                "level": crop.level,
                "overlap_factor": crop.overlap_factor
            }
            for crop in result.get("crops", [])
        ]
    }

    if "global_count" in result:
        json_result["global_count"] = result["global_count"]

    with open(args.output_json, "w") as f:
        json.dump(json_result, f, indent=2)
    print(f"Results saved to: {args.output_json}")

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Strategy: {result['strategy']}")
    print(f"Final Count: {result['count']}")
    if "global_count" in result:
        print(f"Global Count: {result['global_count']}")
    print("="*80)


if __name__ == "__main__":
    main()