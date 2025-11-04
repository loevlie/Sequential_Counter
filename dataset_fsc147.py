"""
FSC-147 Dataset Loader with Spatial Ordering

Loads images with point annotations from the FSC-147 dataset
and applies spatial sorting for training sequential counting models.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Literal
from torch.utils.data import Dataset
from collections import Counter


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


class FSC147Dataset(Dataset):
    """
    FSC-147 dataset with spatial ordering.

    Returns: (image, sorted_points, metadata)
    """

    def __init__(self,
                 dataset_root: str,
                 split: Literal['train', 'val', 'test'] = 'train',
                 spatial_order: Literal['reading_order', 'left_to_right', 'nearest_neighbor'] = 'reading_order',
                 min_objects: int = 1,
                 max_objects: int = 200,
                 image_size: Tuple[int, int] = None):
        """
        Args:
            dataset_root: Path to FSC-147 dataset root
            split: 'train', 'val', or 'test'
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

        self.examples = []
        self._load_data()

    def _load_data(self):
        """Load FSC-147 annotations."""
        # Load split information
        split_file = self.dataset_root / 'annotation_FSC147_384.json'
        splits_file = self.dataset_root / 'Train_Test_Val_FSC_147.json'
        classes_file = self.dataset_root / 'ImageClasses_FSC147.txt'

        if not split_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {split_file}")
        if not splits_file.exists():
            raise FileNotFoundError(f"Split file not found: {splits_file}")

        # Load annotations
        with open(split_file, 'r') as f:
            annotations = json.load(f)

        # Load splits
        with open(splits_file, 'r') as f:
            splits = json.load(f)

        # Load class information if available
        class_map = {}
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        img_name, class_name = parts
                        class_map[img_name] = class_name

        # Get image names for this split
        if self.split == 'val':
            image_names = splits['val']
        elif self.split == 'test':
            image_names = splits['test']
        else:
            image_names = splits['train']

        # Process each image
        images_dir = self.dataset_root / 'images_384_VarV2'

        for img_name in image_names:
            if img_name not in annotations:
                continue

            ann = annotations[img_name]
            points = ann['points']

            # Filter by count
            if len(points) < self.min_objects or len(points) > self.max_objects:
                continue

            # Get object class
            object_type = class_map.get(img_name, 'objects')
            # Clean up class name
            object_type = object_type.strip()
            if not object_type:
                object_type = 'objects'

            # Build image path
            img_path = images_dir / img_name

            self.examples.append({
                'image_path': img_path,
                'image_name': img_name,
                'image_width': ann['W'],
                'image_height': ann['H'],
                'points': [(float(p[0]), float(p[1])) for p in points],
                'object_type': object_type,
                'num_objects': len(points)
            })

        print(f"Loaded {len(self.examples)} images from {self.split} split")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """Returns: (image, sorted_points, metadata)"""
        # Try to load the sample
        max_retries = 10
        for attempt in range(max_retries):
            try:
                example = self.examples[idx]

                # Load image
                if not example['image_path'].exists():
                    raise FileNotFoundError(f"Image not found: {example['image_path']}")

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
                    'category': 'FSC147',
                    'object_type': example['object_type'],
                    'num_objects': len(sorted_points),
                    'image_size': image.size,
                    'image_name': example['image_name']
                }

                return image, sorted_points, metadata

            except Exception as e:
                # Handle file errors
                if attempt == 0:
                    print(f"\nWarning: Skipping sample {idx} due to error: {e}")
                # Try next sample (wrap around if needed)
                idx = (idx + 1) % len(self.examples)
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to load sample after {max_retries} attempts")


def get_dataset_stats(dataset_root: str):
    """Get statistics about the FSC-147 dataset."""
    dataset_root = Path(dataset_root)

    # Load annotations and splits
    with open(dataset_root / 'annotation_FSC147_384.json', 'r') as f:
        annotations = json.load(f)

    with open(dataset_root / 'Train_Test_Val_FSC_147.json', 'r') as f:
        splits = json.load(f)

    # Load classes
    class_map = {}
    classes_file = dataset_root / 'ImageClasses_FSC147.txt'
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    class_map[parts[0]] = parts[1]

    stats = {}
    for split_name, image_names in splits.items():
        counts = []
        object_types = []

        for img_name in image_names:
            if img_name in annotations:
                ann = annotations[img_name]
                counts.append(len(ann['points']))
                obj_type = class_map.get(img_name, 'objects')
                object_types.append(obj_type)

        stats[split_name] = {
            'num_images': len(image_names),
            'num_with_annotations': len(counts),
            'object_counts': counts,
            'object_types': Counter(object_types),
            'avg_count': np.mean(counts) if counts else 0,
            'median_count': np.median(counts) if counts else 0,
            'min_count': np.min(counts) if counts else 0,
            'max_count': np.max(counts) if counts else 0,
        }

    return stats


if __name__ == '__main__':
    # Test the dataset
    dataset = FSC147Dataset(
        dataset_root='/media/M2SSD/FSC147',
        split='train',
        min_objects=5,
        max_objects=50
    )

    print(f"\nDataset loaded: {len(dataset)} samples")

    # Show first sample
    if len(dataset) > 0:
        img, points, meta = dataset[0]
        print(f"\nFirst sample:")
        print(f"  Object type: {meta['object_type']}")
        print(f"  Count: {meta['num_objects']}")
        print(f"  Image size: {meta['image_size']}")
        print(f"  First 3 points: {points[:3]}")
