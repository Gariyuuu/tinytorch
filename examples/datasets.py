"""Small, fully synthetic datasets for the examples.

Everything here is generated from a seeded ``np.random.Generator`` -- there are
no downloads, no cached files and no hidden state, so every example is exactly
reproducible and the repository stays dependency-free.

The digits set is deliberately described as *MNIST-like*, not MNIST: it is
procedurally drawn 8x8 glyphs with noise and jitter. It exercises the same
machinery (multi-class cross-entropy over flattened images) at a size that
trains in seconds on a CPU, and no claim is made that accuracy on it
transfers to real MNIST.
"""

from __future__ import annotations

import numpy as np

__all__ = ["linear_regression_data", "make_digits", "spiral_data", "train_test_split", "xor_data"]


def linear_regression_data(
    n_samples: int = 200,
    n_features: int = 3,
    noise: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """A linear problem with a known ground-truth solution.

    Returns ``(X, y, true_weight, true_bias)`` so the fitted parameters can be
    compared against the values that generated the data -- convergence is then
    a checkable claim, not a vibe.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features))
    weight = rng.uniform(-2.0, 2.0, size=(n_features, 1))
    bias = float(rng.uniform(-1.0, 1.0))
    y = x @ weight + bias + noise * rng.standard_normal((n_samples, 1))
    return x, y, weight, bias


def xor_data(n_per_quadrant: int = 100, spread: float = 0.35, seed: int = 1):
    """Noisy XOR: four Gaussian blobs at (+-1, +-1), labelled by sign agreement.

    Not linearly separable, so a network without a hidden layer cannot beat
    50% -- which makes it a real test of whether backprop through a
    nonlinearity works.
    """
    rng = np.random.default_rng(seed)
    centers = [(-1.0, -1.0, 0), (1.0, 1.0, 0), (-1.0, 1.0, 1), (1.0, -1.0, 1)]
    xs, ys = [], []
    for cx, cy, label in centers:
        xs.append(rng.normal([cx, cy], spread, size=(n_per_quadrant, 2)))
        ys.append(np.full(n_per_quadrant, label))
    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)
    order = rng.permutation(len(x))
    return x[order], y[order]


def spiral_data(n_per_class: int = 150, n_classes: int = 3, noise: float = 0.15, seed: int = 2):
    """Interleaved spirals -- a harder nonlinear multi-class benchmark."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n_per_class * n_classes, 2), dtype=np.float32)
    y = np.zeros(n_per_class * n_classes, dtype=np.int64)
    for c in range(n_classes):
        index = slice(n_per_class * c, n_per_class * (c + 1))
        radius = np.linspace(0.0, 1.0, n_per_class)
        theta = np.linspace(c * 4.0, (c + 1) * 4.0, n_per_class) + rng.normal(0, noise, n_per_class)
        x[index] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
        y[index] = c
    order = rng.permutation(len(x))
    return x[order], y[order]


# Seven-segment-style strokes on an 8x8 grid, one entry per digit. Each stroke
# is (row_slice, col_slice) in a 8x8 image.
_SEGMENTS = {
    "top": (slice(1, 2), slice(2, 6)),
    "upper_left": (slice(1, 4), slice(1, 2)),
    "upper_right": (slice(1, 4), slice(5, 6)),
    "middle": (slice(3, 4), slice(2, 6)),
    "lower_left": (slice(4, 7), slice(1, 2)),
    "lower_right": (slice(4, 7), slice(5, 6)),
    "bottom": (slice(6, 7), slice(2, 6)),
}
_DIGIT_SEGMENTS = {
    0: ["top", "upper_left", "upper_right", "lower_left", "lower_right", "bottom"],
    1: ["upper_right", "lower_right"],
    2: ["top", "upper_right", "middle", "lower_left", "bottom"],
    3: ["top", "upper_right", "middle", "lower_right", "bottom"],
    4: ["upper_left", "upper_right", "middle", "lower_right"],
    5: ["top", "upper_left", "middle", "lower_right", "bottom"],
    6: ["top", "upper_left", "middle", "lower_left", "lower_right", "bottom"],
    7: ["top", "upper_right", "lower_right"],
    8: ["top", "upper_left", "upper_right", "middle", "lower_left", "lower_right", "bottom"],
    9: ["top", "upper_left", "upper_right", "middle", "lower_right", "bottom"],
}


def make_digits(
    n_per_class: int = 120,
    n_classes: int = 10,
    noise: float = 0.15,
    seed: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """An MNIST-*like* 8x8 handwritten-digit set, drawn procedurally.

    Each sample is a seven-segment glyph with per-pixel Gaussian noise, a
    random intensity and a one-pixel jitter, then clipped to ``[0, 1]``.
    Returns ``(X, y)`` with ``X`` of shape ``(n, 64)`` in float32.

    This is a *synthetic stand-in* for MNIST, chosen so the repository needs no
    download and the example runs in seconds. Accuracy numbers on it say
    something about the framework, not about MNIST.
    """
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for digit in range(n_classes):
        template = np.zeros((8, 8), dtype=np.float32)
        for segment in _DIGIT_SEGMENTS[digit]:
            rows, cols = _SEGMENTS[segment]
            template[rows, cols] = 1.0
        for _ in range(n_per_class):
            image = template * rng.uniform(0.7, 1.0)
            shift_r, shift_c = rng.integers(-1, 2, size=2)
            image = np.roll(np.roll(image, shift_r, axis=0), shift_c, axis=1)
            image = image + noise * rng.standard_normal((8, 8)).astype(np.float32)
            images.append(np.clip(image, 0.0, 1.0).reshape(-1))
            labels.append(digit)
    x = np.stack(images).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    order = rng.permutation(len(x))
    return x[order], y[order]


def train_test_split(x: np.ndarray, y: np.ndarray, test_fraction: float = 0.25, seed: int = 4):
    """Deterministic shuffle-and-split into ``(x_train, y_train, x_test, y_test)``."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    cut = int(len(x) * (1.0 - test_fraction))
    train, test = order[:cut], order[cut:]
    return x[train], y[train], x[test], y[test]
