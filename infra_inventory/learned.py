"""Learned component classifier — a trained prior on top of the geometry detectors.

This is the lightweight learned layer of the pipeline. The geometry detectors
remain the gate (they decide *that* an asset exists); this module learns from
labeled ground truth which geometric signatures correspond to which asset
class, so every detection gets an honest, evidence-based ``model`` confidence
factor instead of the prior-free fallback.

What it is
----------
A multinomial logistic-regression (softmax) classifier over measured per-
component features, implemented directly in numpy (no scikit-learn / torch
required — the sandbox has neither and the disk is tight). It is trained by
``scripts/train_classifier.py`` on labeled components extracted from Quick
Simulation scenes, and shipped as ``configs/learned_classifier.json``.

What it is not
--------------
It is not a replacement for the Pointcept / PTv3 backend. Pointcept consumes
full CUDA point-cloud networks over the raw cloud; this classifier consumes
the same measured features the detectors already computed and adds class
evidence per candidate component. When a Pointcept/OpenPCSeg prior is present
it wins; this layer only fills the ``model`` factor on runs without one.

Feature design follows the geometric-feature conventions of the mobile-LiDAR
segmentation literature (Pointcept/PTv3 eigen-features, Toronto-3D MLS
features): per-component eigenvalue descriptors (linearity, planarity,
verticality, columnarity — the same features the Toronto-3D / RandLA-Net
lineage uses per point, aggregated per component here), normalized extents,
ground band, cross-section thickness, radiometric contrast, and compactness.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# The ten competition classes. Index = model id in the classifier.
LEARNED_CLASSES: Tuple[str, ...] = (
    "pavement", "pavement_marking", "utility_pole", "overhead_conductor",
    "utility_cabinet", "traffic_sign", "guardrail", "safety_barrier",
    "rumble_strip", "background",
)

#: Default location of the trained weights (shipped with the repo).
DEFAULT_MODEL_PATH = Path("configs/learned_classifier.json")

#: Feature vector: 14 measured, scale-normalized descriptors. This order is
#: part of the shipped model's contract — never reorder.
FEATURE_NAMES: Tuple[str, ...] = (
    "linearity", "planarity", "verticality", "columnarity",
    "log_length", "log_width", "log_height", "log_points",
    "ground_band", "log_cross_section", "intensity_contrast",
    "rgb_brightness_norm", "above_ground_ratio", "compactness",
)

_NUM_FEATURES = len(FEATURE_NAMES)  # 14


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def component_features(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, indices: np.ndarray,
    intensity: Optional[np.ndarray], rgb: Optional[np.ndarray],
    ground: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Measured 14-d feature vector for one component.

    Returns ``(features, extras)``: the normalized vector for the classifier
    and the raw human-readable values (for explanations). Never raises on
    degenerate components — every raw value is finite by construction.
    """
    pts = np.column_stack((x[indices], y[indices], z[indices]))
    n = len(pts)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    spans = np.maximum(hi - lo, 1e-6)  # metres
    length, width, height = float(spans[0]), float(spans[1]), float(spans[2])

    planarity = linearity = verticality = columnarity = 0.0
    cross_section = height  # fallback: bbox depth
    if n >= 3 and np.isfinite(pts).all():
        try:
            cov = np.cov(pts, rowvar=False)
            values, vectors = np.linalg.eigh(cov)
            values = np.clip(np.nan_to_num(values, nan=0.0), 0.0, None)
            if np.isfinite(values).all() and values[-1] > 1e-12:
                e0, e1, e2 = float(values[0]), float(values[1]), float(values[2])
                dominant = vectors[:, -1]
                planarity = max(0.0, min(1.0, (e1 - e0) / e2))
                linearity = max(0.0, min(1.0, (e2 - e1) / e2))
                verticality = max(0.0, min(1.0, abs(float(dominant[2]))))
                columnarity = max(0.0, min(1.0, e0 / e1)) if e1 > 1e-12 else 0.0
                cross_section = 2.0 * math.sqrt(max(e0, 0.0))
        except (np.linalg.LinAlgError, ValueError):  # pragma: no cover - defensive
            pass

    intensity_contrast = 0.0
    if intensity is not None and len(intensity):
        vals = intensity[indices]
        med = float(np.median(intensity)) if len(intensity) > 8 else 0.0
        intensity_contrast = float(np.clip((np.mean(vals) - med) / max(med, 1e-6), -2.0, 4.0))
    rgb_brightness = 0.0
    if rgb is not None and len(rgb):
        rgb_brightness = float(np.clip(np.mean(rgb[indices]) / 65535.0, 0.0, 1.0))

    above_ground_ratio = float(np.clip(np.mean(z[indices] > ground + 0.1), 0.0, 1.0))
    bbox_volume = max(length, 0.05) * max(width, 0.05) * max(height, 0.05)
    compactness = float(np.clip(n * 0.001 / bbox_volume, 0.0, 1.0))

    extras: Dict[str, float] = {
        "length_m": length, "width_m": width, "height_m": height,
        "cross_section_m": cross_section, "ground_band": ground_band(ground, lo[2], hi[2]),
        "intensity_contrast": intensity_contrast, "rgb_brightness": rgb_brightness,
        "point_count": n,
    }
    features = np.array([
        linearity, planarity, verticality, columnarity,
        math.log(length), math.log(width), math.log(height), math.log(max(n, 1)),
        ground_band(ground, lo[2], hi[2]), math.log(max(cross_section, 1e-3)),
        intensity_contrast, rgb_brightness, above_ground_ratio, compactness,
    ], dtype=np.float64)
    return features, extras


def ground_band(ground: float, z_min: float, z_max: float) -> float:
    """Where the component sits relative to the ground: 0=on ground, 1=high.

    Continuous version of the height-band gates the geometry detectors use:
    0.0 for objects starting at ground level, rising to 1.0 for objects whose
    base sits at/above ~5 m (overhead conductors).
    """
    return float(np.clip((z_min - ground) / 5.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Softmax classifier (pure numpy)
# ---------------------------------------------------------------------------

def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over axis -1."""
    shifted = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def train_softmax(
    X: np.ndarray, y: np.ndarray, num_classes: int, epochs: int = 400,
    lr: float = 0.35, l2: float = 1e-3, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full-batch softmax regression with L2 regularization.

    ``X`` must already be normalized with ``FeatureNormalizer``. Returns
    ``(W, b)`` with ``W`` shaped (num_features, num_classes).
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = rng.normal(0.0, 0.01, (d, num_classes))
    b = np.zeros(num_classes)
    onehot = np.eye(num_classes)[y]
    for _ in range(epochs):
        probs = softmax(X @ W + b)
        grad_W = X.T @ (probs - onehot) / n + l2 * W
        grad_b = (probs - onehot).sum(axis=0) / n
        W -= lr * grad_W
        b -= lr * grad_b
    return W, b


class FeatureNormalizer:
    """Standardize features to zero mean / unit variance (per feature)."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.maximum(np.asarray(std, dtype=np.float64), 1e-9)

    @classmethod
    def fit(cls, X: np.ndarray) -> "FeatureNormalizer":
        return cls(X.mean(axis=0), X.std(axis=0))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureNormalizer":
        return cls(np.asarray(data["mean"], dtype=np.float64), np.asarray(data["std"], dtype=np.float64))


class ComponentClassifier:
    """Trained softmax classifier over component features (numpy, no deps)."""

    def __init__(
        self, weights: np.ndarray, bias: np.ndarray, normalizer: FeatureNormalizer,
        classes: Sequence[str] = LEARNED_CLASSES, version: str = "learned-v1",
        training: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = np.asarray(bias, dtype=np.float64)
        self.normalizer = normalizer
        self.classes = tuple(classes)
        self.version = version
        self.training = training or {}

    # -- inference ----------------------------------------------------------

    def scores(self, features: np.ndarray) -> Dict[str, float]:
        """Class -> probability for one component's feature vector."""
        z = self.normalizer.transform(features[None, :]) @ self.weights + self.bias
        probs = softmax(z)[0]
        return {name: float(p) for name, p in zip(self.classes, probs)}

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """(best class, its probability) for one component."""
        probs = self.scores(features)
        best = max(probs, key=probs.get)  # type: ignore[arg-type]
        return best, probs[best]

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "classes": list(self.classes),
            "features": list(FEATURE_NAMES),
            "normalizer": self.normalizer.to_dict(),
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "training": self.training,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComponentClassifier":
        return cls(
            weights=np.asarray(data["weights"], dtype=np.float64),
            bias=np.asarray(data["bias"], dtype=np.float64),
            normalizer=FeatureNormalizer.from_dict(data["normalizer"]),
            classes=data.get("classes", LEARNED_CLASSES),
            version=data.get("version", "learned-v1"),
            training=data.get("training", {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ComponentClassifier":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_classifier(path: Optional[Path] = None) -> Optional[ComponentClassifier]:
    """Load the trained classifier, or None when absent/invalid (never crashes)."""
    target = Path(path) if path else DEFAULT_MODEL_PATH
    try:
        if not target.is_file():
            return None
        return ComponentClassifier.load(target)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Ground-truth collection hook (enabled with --collect-training)
# ---------------------------------------------------------------------------

def collect_component_example(
    ctx: Any, asset_class: str, indices: np.ndarray, ground_used: float
) -> None:
    """Append one labeled training example (component features + class).

    Called from ``build_asset`` when ``ctx.settings.collect_training`` is on.
    Writes one JSONL file per tile under ``<output>/training/`` — per-tile
    files (not one shared file) because tiles are extracted in fork worker
    processes and a shared append target could interleave.
    """
    features, extras = component_features(
        ctx.x, ctx.y, ctx.z, indices, ctx.intensity, ctx.rgb, ground_used,
    )
    extras["center_xyz"] = [
        round(float(np.mean(ctx.x[indices])), 4),
        round(float(np.mean(ctx.y[indices])), 4),
        round(float(np.mean(ctx.z[indices])), 4),
    ]
    _TRAINING_BUFFER.append({
        "tile": ctx.tile,
        "class": asset_class,
        "features": features.tolist(),
        "extras": extras,
    })
    if len(_TRAINING_BUFFER) >= 512:
        flush_training_buffer(ctx)


_TRAINING_BUFFER: List[dict] = []


def flush_training_buffer(ctx: Any) -> None:
    """Drain the in-memory example buffer to ``<output>/training/components_<tile>.jsonl``.

    ``ctx.settings.output_dir`` is set by the pipeline before the detection
    pass (a plain attribute; it is not a schema field so it never round-trips
    through API settings).
    """
    if not _TRAINING_BUFFER:
        return
    from pathlib import Path as _P

    out_dir = getattr(ctx.settings, "output_dir", None)
    if not out_dir:
        _TRAINING_BUFFER.clear()
        return
    training_dir = _P(out_dir) / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    # Tile name is filesystem-safe (tile_x_y); examples from several tiles may
    # share a worker process, so group by the tile recorded on each example.
    by_tile: Dict[str, List[dict]] = {}
    for example in _TRAINING_BUFFER:
        by_tile.setdefault(str(example["tile"]), []).append(example)
    _TRAINING_BUFFER.clear()
    for tile, examples in by_tile.items():
        path = training_dir / f"components_{tile}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(json.dumps(ex) for ex in examples) + "\n")


# ---------------------------------------------------------------------------
# Inference-time prior for build_asset
# ---------------------------------------------------------------------------

def learned_prior_factor(
    classifier: Optional["ComponentClassifier"], assigned_class: str,
    features: np.ndarray, scores: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """(model factor, prior class) for a detected component, or (None, None).

    The factor measures how strongly the trained classifier's evidence for the
    detector's assigned class agrees with the geometry decision: it is the
    calibrated blend of P(assigned) with the runner-up evidence. A component
    whose features strongly match its assigned class scores high; one the
    classifier considers ambiguous scores lower — exactly what the confidence
    blend should reflect.
    """
    if classifier is None:
        return None, None
    probs = scores if scores is not None else classifier.scores(features)
    p_assigned = float(np.clip(probs.get(assigned_class, 0.0), 0.0, 1.0))
    others = [p for cls, p in probs.items() if cls != assigned_class]
    p_runner = max(others) if others else 0.0
    # Calibrated blend: an unambiguous component keeps its P(assigned); an
    # ambiguous one is pulled toward the runner-up evidence.
    factor = float(np.clip(p_assigned - 0.25 * p_runner, 0.0, 1.0))
    prior_name = classifier.classes[int(np.argmax([probs.get(c, 0.0) for c in classifier.classes]))]
    return factor, prior_name
