"""Template matching via normalized cross-correlation (CS 131 Lec 4-5).

NCC is invariant to local affine changes in brightness/contrast, which matters
because nacelle brightness varies with sun angle and sensor gain across scenes.
Returns a dense correlation map in [-1, 1], same size as the image (the border
where the template overhangs is filled with the minimum valid score).
"""

from __future__ import annotations

import numpy as np

from ._conv import _sliding_windows, correlate2d, pad_reflect


def matched_filter(img: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Plain cross-correlation of a zero-mean template with the image (Lec 4-5).

    Unlike NCC this is NOT contrast-invariant: a zero-mean bright-center
    template acts as a center-surround matched filter that responds strongly to
    bright compact blobs (the nacelle hub) and weakly to flat terrain. That
    brightness sensitivity is precisely what NCC throws away.
    """
    template = np.asarray(template, dtype=np.float64)
    template = template - template.mean()
    return correlate2d(np.asarray(img, dtype=np.float64), template)


def normalized_cross_correlation(img: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Dense NCC map, same size as img.

    score(y,x) = sum( (W - meanW)(T - meanT) ) / (||W-meanW|| ||T-meanT||)
    where W is the image window centered at (y,x) and T is the template.
    """
    img = np.asarray(img, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    th, tw = template.shape
    if th % 2 == 0 or tw % 2 == 0:
        raise ValueError("template dimensions must be odd")

    t_zm = template - template.mean()
    t_norm = np.sqrt((t_zm ** 2).sum())
    if t_norm == 0:
        raise ValueError("template has zero variance")

    padded = pad_reflect(img, th // 2, tw // 2)
    win = _sliding_windows(padded, th, tw)              # (H, W, th, tw)
    win_mean = win.mean(axis=(2, 3), keepdims=True)
    win_zm = win - win_mean
    num = np.einsum("ijkl,kl->ij", win_zm, t_zm)
    win_norm = np.sqrt((win_zm ** 2).sum(axis=(2, 3)))
    denom = win_norm * t_norm
    out = np.zeros(img.shape, dtype=np.float64)
    nz = denom > 1e-12
    out[nz] = num[nz] / denom[nz]
    return out


def match_template(
    img: np.ndarray,
    template: np.ndarray,
    *,
    threshold: float = 0.5,
    nms_radius: int = 2,
) -> np.ndarray:
    """Return match locations (row, col) where NCC exceeds threshold, after NMS."""
    ncc = normalized_cross_correlation(img, template)
    cand = ncc >= threshold
    H, W = ncc.shape
    keep = []
    ys, xs = np.nonzero(cand)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - nms_radius), min(H, y + nms_radius + 1)
        x0, x1 = max(0, x - nms_radius), min(W, x + nms_radius + 1)
        if ncc[y, x] >= ncc[y0:y1, x0:x1].max():
            keep.append((y, x))
    return np.array(keep, dtype=np.int64).reshape(-1, 2)
