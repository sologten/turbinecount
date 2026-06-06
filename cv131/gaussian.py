"""Gaussian smoothing (CS 131 Lec 2).

Separable Gaussian: build a 1-D kernel, apply it along rows then columns. This
is O(N*k) instead of O(N*k^2) and is numerically identical to the 2-D kernel up
to floating point.
"""

from __future__ import annotations

import numpy as np

from ._conv import correlate2d


def gaussian_kernel(sigma: float, radius: int | None = None) -> np.ndarray:
    """1-D normalized Gaussian kernel.

    radius defaults to ceil(3*sigma), capturing >99.7% of the mass.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if radius is None:
        radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    k /= k.sum()
    return k


def gaussian_smooth(img: np.ndarray, sigma: float, radius: int | None = None) -> np.ndarray:
    """Separable Gaussian blur with reflect padding (same-size output)."""
    k = gaussian_kernel(sigma, radius)
    row = k.reshape(1, -1)   # horizontal pass
    col = k.reshape(-1, 1)   # vertical pass
    out = correlate2d(np.asarray(img, dtype=np.float64), row)
    out = correlate2d(out, col)
    return out
