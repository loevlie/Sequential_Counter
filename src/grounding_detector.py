"""
GroundingDINO wrapper for object detection.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
import warnings

try:
    from groundingdino.util.inference import load_model, predict
    from groundingdino.util import box_ops
    GROUNDINGDINO_AVAILABLE = True
except ImportError:
    GROUNDINGDINO_AVAILABLE = False
    warnings.warn("GroundingDINO not installed. Install with: pip install groundingdino-py")


class GroundingDINODetector:
    """Wrapper for GroundingDINO object detection."""

    def __init__(
        self,
        model_config_path: str = None,
        model_checkpoint_path: str = None,
        device: str = "cuda"
    ):
        """
        Initialize GroundingDINO detector.

        Args:
            model_config_path: Path to model config (auto-detected if None)
            model_checkpoint_path: Path to pretrained weights (auto-detected if None)
            device: Device to run on
        """
        if not GROUNDINGDINO_AVAILABLE:
            raise ImportError("GroundingDINO not installed. Please install it first.")

        import os

        # Auto-detect config path
        if model_config_path is None:
            import groundingdino
            package_dir = os.path.dirname(groundingdino.__file__)
            model_config_path = os.path.join(
                package_dir, "config", "GroundingDINO_SwinT_OGC.py"
            )

        # Auto-detect checkpoint path
        if model_checkpoint_path is None:
            # Check common locations
            possible_paths = [
                "weights/groundingdino_swint_ogc.pth",
                "/home/denny-loevlie/Jivko/Sequential_Counter/weights/groundingdino_swint_ogc.pth"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_checkpoint_path = path
                    break

            if model_checkpoint_path is None:
                raise FileNotFoundError(
                    "Could not find GroundingDINO weights. "
                    "Please download them with: ./install_groundingdino.sh"
                )

        print(f"Loading GroundingDINO from:")
        print(f"  Config: {model_config_path}")
        print(f"  Weights: {model_checkpoint_path}")

        self.device = device
        self.model = load_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=device
        )

    def detect(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Detect objects in image using text prompt.

        Args:
            image: PIL Image
            text_prompt: Text description of objects to detect (e.g., "apple", "car")
            box_threshold: Confidence threshold for detections
            text_threshold: Text matching threshold

        Returns:
            boxes: numpy array of shape (N, 4) with normalized coords [0, 1]
            logits: numpy array of shape (N,) with confidence scores
            phrases: list of detected phrases
        """
        # Save PIL image to temp file (GroundingDINO's load_image needs a path)
        import tempfile
        from groundingdino.util.inference import load_image

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Load image using GroundingDINO's format
            image_source, image_tensor = load_image(tmp_path)

            # Run detection
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device
            )
        finally:
            # Clean up temp file
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Convert to numpy if needed
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()

        # GroundingDINO returns boxes in cxcywh format (center_x, center_y, width, height)
        # Convert to xyxy format (x1, y1, x2, y2) for consistency
        converted_boxes = []
        for box in boxes:
            cx, cy, w, h = box
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            converted_boxes.append([x1, y1, x2, y2])

        converted_boxes = np.array(converted_boxes) if len(converted_boxes) > 0 else np.array([])

        return converted_boxes, logits, phrases

    def detect_in_tile(
        self,
        tile: Image.Image,
        category: str,
        box_threshold: float = 0.3
    ) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
        """
        Detect objects in a single tile.

        Args:
            tile: PIL Image (tile)
            category: Object category to detect
            box_threshold: Detection threshold

        Returns:
            bboxes: List of (x1, y1, x2, y2) in normalized [0, 1] coords
            confidences: List of confidence scores
        """
        boxes, logits, phrases = self.detect(
            image=tile,
            text_prompt=category,
            box_threshold=box_threshold
        )

        # detect() already returns boxes in xyxy format
        bboxes = [tuple(box) for box in boxes]
        confidences = logits.tolist()

        return bboxes, confidences


class DummyDetector:
    """
    Dummy detector for testing when GroundingDINO is not available.
    Returns random detections for debugging.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn("Using DummyDetector - install GroundingDINO for real detections")

    def detect_in_tile(
        self,
        tile: Image.Image,
        category: str,
        box_threshold: float = 0.3
    ) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
        """
        Generate random detections for testing.

        Returns 2-5 random detections per tile.
        """
        num_detections = np.random.randint(2, 6)

        bboxes = []
        confidences = []

        for _ in range(num_detections):
            # Random bbox in normalized coords
            x1 = np.random.uniform(0.1, 0.6)
            y1 = np.random.uniform(0.1, 0.6)
            w = np.random.uniform(0.05, 0.2)
            h = np.random.uniform(0.05, 0.2)
            x2 = min(x1 + w, 1.0)
            y2 = min(y1 + h, 1.0)

            bboxes.append((x1, y1, x2, y2))
            confidences.append(np.random.uniform(0.3, 0.95))

        return bboxes, confidences


def get_detector(use_dummy: bool = False, **kwargs):
    """
    Factory function to get detector.

    Args:
        use_dummy: If True, use dummy detector. If False, try to use real GroundingDINO.
        **kwargs: Arguments to pass to detector constructor

    Returns:
        Detector instance
    """
    if use_dummy or not GROUNDINGDINO_AVAILABLE:
        return DummyDetector(**kwargs)
    else:
        return GroundingDINODetector(**kwargs)
