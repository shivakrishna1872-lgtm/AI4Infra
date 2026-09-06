"""Asset attribution: provenance fields attached to every detected asset.

Everything here is measured from the LAS data itself. When a value cannot be
determined reliably the corresponding field is ``None`` in the output - never a
guess. Source point indices are preserved so every asset keeps a link to the
LiDAR points it was extracted from.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


def intensity_stats(values: np.ndarray) -> Optional[Dict[str, float]]:
    if not len(values):
        return None
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
        "std": float(values.std()),
    }


def rgb_stats(values: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    if values is None or not len(values):
        return None
    return {
        "red_mean": float(values[:, 0].mean()),
        "green_mean": float(values[:, 1].mean()),
        "blue_mean": float(values[:, 2].mean()),
        "brightness_mean": float(values.mean()),
    }


def orientation_deg(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Azimuth of the dominant XY axis, 0-180 degrees."""
    if len(x) < 3:
        return None
    covariance = np.cov(np.column_stack((x, y)), rowvar=False)
    if not np.isfinite(covariance).all():
        return None
    values, vectors = np.linalg.eigh(covariance)
    if values[-1] <= 0:
        return None
    vector = vectors[:, -1]
    return float((math.degrees(math.atan2(vector[1], vector[0])) + 360) % 180)


def scanner_label(point_source_id: Optional[np.ndarray], indices: np.ndarray) -> Tuple[Optional[int], Optional[str]]:
    """Majority point_source_id and a human label (Laser Left / Laser Right / other)."""
    if point_source_id is None or not len(indices):
        return None, None
    values = point_source_id[indices]
    if not len(values):
        return None, None
    counts = np.bincount(values.astype(np.int64))
    majority = int(np.argmax(counts))
    label: Optional[str]
    if majority == 1:
        label = "Laser Left"
    elif majority == 2:
        label = "Laser Right"
    else:
        label = f"point_source_{majority}"
    return majority, label


def run_label(gps_labels: Optional[np.ndarray], indices: np.ndarray) -> Optional[str]:
    """Run label from precomputed GPS-time k-means labels (1=first pass, 2=second pass)."""
    if gps_labels is None or not len(indices):
        return None
    values = gps_labels[indices]
    if not len(values):
        return None
    counts = np.bincount(values.astype(np.int64))
    if len(counts) < 3:
        return None
    majority = int(np.argmax(counts[1:]) + 1)
    return f"Run {majority}"


def sample_indices(indices: np.ndarray, limit: int) -> List[int]:
    """Deterministic, evenly spaced sample of source point indices (provenance)."""
    if not len(indices):
        return []
    stride = max(1, math.ceil(len(indices) / limit))
    return [int(index) for index in indices[::stride][:limit]]


def viewer_sample_indices(indices: np.ndarray, sample_stride: int, limit: int) -> List[int]:
    """Map source indices to viewer point rows (only points kept in the viewer sample)."""
    kept = indices[indices % sample_stride == 0] // sample_stride
    return [int(index) for index in kept[:limit]]