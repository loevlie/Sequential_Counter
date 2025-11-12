"""
Utility functions for visual marking.
"""

import cv2
import numpy as np
from typing import List, Tuple


class VisualMarker:
    """Applies visual marking overlays to images."""

    def __init__(self, strategy='heatmap', alpha=0.3):
        """
        Args:
            strategy: 'heatmap', 'numbers', or 'dots'
            alpha: Overlay transparency
        """
        self.strategy = strategy
        self.alpha = alpha

    def mark_image(self, image, points):
        """
        Mark image with given points.

        Args:
            image: RGB numpy array
            points: List of (x, y) tuples

        Returns:
            Marked RGB image
        """
        if isinstance(image, np.ndarray):
            image = image.copy()
        else:
            image = np.array(image)

        # Handle both list and numpy array points
        if isinstance(points, np.ndarray):
            if points.size == 0:
                return image
        elif not points:
            return image

        if self.strategy == 'heatmap':
            return self._mark_heatmap(image, points)
        elif self.strategy == 'numbers':
            return self._mark_numbers(image, points)
        elif self.strategy == 'dots':
            return self._mark_dots(image, points)
        else:
            return image

    def _mark_heatmap(self, image, points):
        """Overlay heatmap visualization."""
        H, W = image.shape[:2]
        heatmap = np.zeros((H, W), dtype=np.float32)

        for x, y in points:
            y_grid, x_grid = np.ogrid[:H, :W]
            gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * 10**2))
            heatmap = np.maximum(heatmap, gaussian)

        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_HOT
        )

        marked = cv2.addWeighted(image, 1.0, heatmap_colored, self.alpha, 0)
        return marked

    def _mark_numbers(self, image, points):
        """Mark with sequential numbers."""
        marked = image.copy()

        for i, (x, y) in enumerate(points, 1):
            # Background circle
            overlay = marked.copy()
            cv2.circle(overlay, (int(x), int(y)), 12, (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.6, marked, 0.4, 0, marked)

            # Number text
            text = str(i)
            cv2.putText(marked, text, (int(x)-6, int(y)+6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return marked

    def _mark_dots(self, image, points):
        """Mark with colored dots."""
        import colorsys

        marked = image.copy()
        n = len(points)

        for i, (x, y) in enumerate(points):
            hue = i / max(n, 1)
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            color = tuple(int(c * 255) for c in reversed(rgb))
            cv2.circle(marked, (int(x), int(y)), 6, color, -1)

        return marked
