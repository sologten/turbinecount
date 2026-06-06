"""Harris corner detection (CS 131 Lec 6), from scratch.

Structure tensor M = G_sigma * [[Ix^2, IxIy],[IxIy, Iy^2]], corner response
R = det(M) - k*trace(M)^2. The bright compact nacelle hub gives a strong
corner-like response against the surrounding field.
"""

from __future__ import annotations

import numpy as np

from .gaussian import gaussian_smooth
from .sobel import sobel_gradients


def harris_response(img: np.ndarray, *, sigma: float = 1.0, k: float = 0.04) -> np.ndarray:
    """Dense Harris response map R (same size as img)."""
    img = np.asarray(img, dtype=np.float64)
    g = sobel_gradients(img)
    Ix, Iy = g.gx, g.gy
    Sxx = gaussian_smooth(Ix * Ix, sigma)
    Syy = gaussian_smooth(Iy * Iy, sigma)
    Sxy = gaussian_smooth(Ix * Iy, sigma)
    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    return det - k * trace * trace


def harris_corners(
    img: np.ndarray,
    *,
    sigma: float = 1.0,
    k: float = 0.04,
    rel_threshold: float = 0.01,
    nms_radius: int = 2,
    max_corners: int | None = None,
) -> np.ndarray:
    """Return corner coordinates as an (n, 2) array of (row, col).

    rel_threshold is a fraction of the maximum response. NMS keeps only local
    maxima within a (2*nms_radius+1) window.
    """
    R = harris_response(img, sigma=sigma, k=k)
    if R.max() <= 0:
        return np.empty((0, 2), dtype=np.int64)
    thr = rel_threshold * R.max()
    cand = R > thr

    H, W = R.shape
    keep = np.zeros_like(cand)
    ys, xs = np.nonzero(cand)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - nms_radius), min(H, y + nms_radius + 1)
        x0, x1 = max(0, x - nms_radius), min(W, x + nms_radius + 1)
        if R[y, x] >= R[y0:y1, x0:x1].max():
            keep[y, x] = True

    pts = np.argwhere(keep)
    if max_corners is not None and len(pts) > max_corners:
        order = np.argsort(R[pts[:, 0], pts[:, 1]])[::-1][:max_corners]
        pts = pts[order]
    return pts
