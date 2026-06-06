"""Shared 2-D correlation/convolution core (pure NumPy).

We implement one well-tested sliding-window operator and build the rest of the
syllabus primitives on top of it. The implementation is vectorized with
stride tricks so it stays fast enough for full Sentinel-2 chips without leaving
NumPy.
"""

from __future__ import annotations

import numpy as np


def pad_reflect(img: np.ndarray, py: int, px: int) -> np.ndarray:
    """Symmetric (reflect) padding. Mirrors edge pixels so that gradients and
    blurs do not invent a dark border around the image."""
    if py == 0 and px == 0:
        return img.astype(np.float64, copy=True)
    return np.pad(img.astype(np.float64), ((py, py), (px, px)), mode="reflect")


def _sliding_windows(img: np.ndarray, kh: int, kw: int) -> np.ndarray:
    """Return an (H, W, kh, kw) view of every kh x kw window of a padded image.

    Uses as_strided on a contiguous array. The returned array is a *view*; do
    not write to it.
    """
    img = np.ascontiguousarray(img, dtype=np.float64)
    H = img.shape[0] - kh + 1
    W = img.shape[1] - kw + 1
    s0, s1 = img.strides
    shape = (H, W, kh, kw)
    strides = (s0, s1, s0, s1)
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)


def correlate2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D cross-correlation with reflect padding, output same size as input.

    Cross-correlation (no kernel flip). For symmetric kernels (Gaussian) this
    equals convolution; for asymmetric ones (Sobel) we deliberately want
    correlation so the sign convention matches the textbook Sobel masks.
    """
    img = np.asarray(img, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("kernel dimensions must be odd")
    padded = pad_reflect(img, kh // 2, kw // 2)
    win = _sliding_windows(padded, kh, kw)
    return np.einsum("ijkl,kl->ij", win, kernel)


def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """True convolution (kernel flipped on both axes), reflect padding."""
    return correlate2d(img, np.flip(np.asarray(kernel, dtype=np.float64)))
