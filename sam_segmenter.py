"""SAM (Segment Anything Model) interactive segmentation wrapper.

Two-step workflow:
1. load_image() — encode image (slow, ~10-30s) — call once per area
2. predict() — predict mask from points/box (fast, ~50ms) — call per click
"""

import os
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from pyproj import Transformer
from rasterio.transform import AffineTransformer

from geo_utils import mask_to_geojson

SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT", os.path.expanduser(
    "~/.cache/torch/hub/checkpoints/sam_vit_h_4b8939.pth"
))
SAM_MODEL_TYPE = "vit_h"


class SamSegmenter:
    """Interactive SAM segmentation with georeferencing."""

    def __init__(self):
        self.predictor = None
        self._loaded = False
        # Current image state
        self.geo_info = None
        self.image_shape = None
        self._image_set = False

    def load(self):
        """Load SAM model. Call once at startup."""
        if self._loaded:
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam.to(device)
        self.predictor = SamPredictor(sam)
        self._loaded = True

    def set_image(self, rgb_array, geo_info):
        """Encode an image for subsequent predictions.

        Args:
            rgb_array: [H, W, 3] uint8 array
            geo_info: dict with 'transform', 'crs', 'bbox', 'shape'
        """
        if not self._loaded:
            self.load()

        self.predictor.set_image(rgb_array)
        self.geo_info = geo_info
        self.image_shape = rgb_array.shape[:2]
        self._image_set = True

    def _lonlat_to_pixel(self, lon, lat):
        """Convert lon/lat to pixel coordinates in the current image."""
        gi = self.geo_info
        bbox = gi["bbox"]
        h, w = self.image_shape
        # Simple linear mapping from bbox to pixel coords
        px = (lon - bbox[0]) / (bbox[2] - bbox[0]) * w
        py = (lat - bbox[3]) / (bbox[1] - bbox[3]) * h  # y is flipped
        return px, py

    def predict(self, points=None, labels=None, box=None):
        """Predict segmentation mask from prompts.

        Args:
            points: list of [lon, lat] coordinates (positive/negative clicks)
            labels: list of 1 (foreground) or 0 (background) per point
            box: [west, south, east, north] bounding box prompt

        Returns:
            dict with:
              - geojson: GeoJSON Feature of the predicted mask
              - area_pixels: number of mask pixels
              - score: model confidence score
        """
        if not self._image_set:
            raise RuntimeError("No image loaded. Call set_image() first.")

        point_coords = None
        point_labels = None

        if points and labels:
            px_points = []
            for lon, lat in points:
                px, py = self._lonlat_to_pixel(lon, lat)
                px_points.append([px, py])
            point_coords = np.array(px_points, dtype=np.float32)
            point_labels = np.array(labels, dtype=np.int32)

        box_np = None
        if box:
            x1, y1 = self._lonlat_to_pixel(box[0], box[3])  # top-left
            x2, y2 = self._lonlat_to_pixel(box[2], box[1])  # bottom-right
            box_np = np.array([x1, y1, x2, y2], dtype=np.float32)

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_np,
            multimask_output=True,
        )

        # Pick the best mask (highest score)
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]
        score = float(scores[best_idx])

        # Convert mask to GeoJSON
        gi = self.geo_info
        feature = mask_to_geojson(mask, gi["transform"], gi.get("lonlat_crs", "EPSG:4326"))

        if feature:
            feature["properties"]["score"] = round(score, 4)
            feature["properties"]["area_pixels"] = int(mask.sum())

        return {
            "geojson": feature,
            "area_pixels": int(mask.sum()),
            "score": score,
        }

    @property
    def is_ready(self):
        return self._image_set
