"""
OmniCount-191 Dataset Loader with Spatial Ordering

Loads images with point annotations and applies spatial sorting
for training sequential counting models.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import random
from typing import List, Tuple, Literal
from torch.utils.data import Dataset


class SpatialSorter:
    """Utilities for sorting points in different spatial orders."""

    @staticmethod
    def reading_order(points: List[Tuple[float, float]],
                     row_height: int = 50) -> List[Tuple[float, float]]:
        """Sort in reading order: top-to-bottom, left-to-right."""
        return sorted(points, key=lambda p: (int(p[1] // row_height), p[0]))

    @staticmethod
    def left_to_right(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Sort strictly left to right."""
        return sorted(points, key=lambda p: (p[0], p[1]))

    @staticmethod
    def nearest_neighbor(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Greedy nearest-neighbor (TSP heuristic)."""
        if len(points) <= 1:
            return points

        remaining = list(points)
        start = min(remaining, key=lambda p: (p[1], p[0]))
        path = [start]
        remaining.remove(start)

        while remaining:
            current = path[-1]
            nearest = min(remaining,
                         key=lambda p: np.sqrt((p[0]-current[0])**2 + (p[1]-current[1])**2))
            path.append(nearest)
            remaining.remove(nearest)

        return path


class OmniCountDataset(Dataset):
    """
    OmniCount-191 dataset with spatial ordering.

    Returns: (image, sorted_points, metadata)
    """

    def __init__(self,
                 dataset_root: str,
                 categories: List[str] = None,
                 split: Literal['train', 'valid', 'test'] = 'train',
                 spatial_order: Literal['reading_order', 'left_to_right', 'nearest_neighbor'] = 'reading_order',
                 min_objects: int = 1,
                 max_objects: int = 150,
                 image_size: Tuple[int, int] = None):
        """
        Args:
            dataset_root: Path to OmniCount-191 dataset
            categories: List of category folders (default: all)
            split: 'train', 'valid', or 'test'
            spatial_order: Ordering strategy
            min_objects: Min objects per image
            max_objects: Max objects per image
            image_size: Optional (W, H) to resize images
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.spatial_order = spatial_order
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.image_size = image_size

        if categories is None:
            categories = ['Supermarket', 'Urban', 'Fruits', 'Wild', 'Satellite',
                         'Vegetables', 'Birds', 'Pets', 'Household']

        self.categories = categories
        self.examples = []
        self._load_data()

    def _load_data(self):
        """Load COCO annotations from all categories."""
        for category in self.categories:
            cat_path = self.dataset_root / category / self.split

            if not cat_path.exists():
                continue

            annotation_file = cat_path / "_annotations.coco.json"
            if not annotation_file.exists():
                continue

            with open(annotation_file, 'r') as f:
                coco_data = json.load(f)

            # Group annotations by image
            image_to_anns = {}
            for ann in coco_data['annotations']:
                image_id = ann['image_id']
                if image_id not in image_to_anns:
                    image_to_anns[image_id] = []
                image_to_anns[image_id].append(ann)

            # Create examples
            for img_info in coco_data['images']:
                image_id = img_info['id']

                if image_id not in image_to_anns:
                    continue

                anns = image_to_anns[image_id]

                # Filter by count
                if len(anns) < self.min_objects or len(anns) > self.max_objects:
                    continue

                # Extract points from bbox centroids
                points = []
                for ann in anns:
                    bbox = ann['bbox']  # [x, y, w, h]
                    cx = bbox[0] + bbox[2] / 2
                    cy = bbox[1] + bbox[3] / 2
                    points.append((cx, cy))

                self.examples.append({
                    'category': category,
                    'image_path': cat_path / img_info['file_name'],
                    'image_width': img_info['width'],
                    'image_height': img_info['height'],
                    'points': points,
                    'num_objects': len(points)
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """Returns: (image, sorted_points, metadata)"""
        example = self.examples[idx]

        # Load image
        image = Image.open(example['image_path']).convert('RGB')

        # Resize if needed
        if self.image_size is not None:
            scale_x = self.image_size[0] / example['image_width']
            scale_y = self.image_size[1] / example['image_height']
            image = image.resize(self.image_size, Image.LANCZOS)
            points = [(x * scale_x, y * scale_y) for x, y in example['points']]
        else:
            points = example['points']

        # Apply spatial sorting
        sorter = SpatialSorter()

        if self.spatial_order == 'reading_order':
            sorted_points = sorter.reading_order(points, row_height=50)
        elif self.spatial_order == 'left_to_right':
            sorted_points = sorter.left_to_right(points)
        elif self.spatial_order == 'nearest_neighbor':
            sorted_points = sorter.nearest_neighbor(points)
        else:
            sorted_points = points

        metadata = {
            'category': example['category'],
            'num_objects': len(sorted_points),
            'image_size': image.size
        }

        return image, sorted_points, metadata
