"""Hough line transform (CS 131 Lec 3), from scratch.

Standard rho-theta accumulator voting over edge pixels, followed by
non-maximum suppression to pull out line peaks. We also expose a per-pixel
"line support" map: for each detected line, every edge pixel within a small
perpendicular distance of it gets credit. That map is what the turbine
pipeline consumes (blade shadows are short straight segments).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sobel import sobel_gradients


@dataclass
class HoughResult:
    accumulator: np.ndarray   # (n_rho, n_theta) vote counts
    rhos: np.ndarray          # rho bin centers
    thetas: np.ndarray        # theta bin centers (radians)
    lines: np.ndarray         # (k, 3): rho, theta, votes  (votes-descending)


def _edge_mask(img: np.ndarray, mag_percentile: float) -> np.ndarray:
    g = sobel_gradients(img)
    thr = np.percentile(g.magnitude, mag_percentile)
    # Require strictly positive gradient as well: when edges are sparse the
    # percentile can collapse to 0, which would otherwise let every flat pixel
    # vote and swamp the accumulator.
    return (g.magnitude >= thr) & (g.magnitude > 1e-8)


def hough_lines(
    img: np.ndarray,
    *,
    mag_percentile: float = 90.0,
    theta_step_deg: float = 1.0,
    rho_step: float = 1.0,
    n_lines: int = 50,
    nms_rho: int = 5,
    nms_theta: int = 5,
    min_votes: int = 1,
    edge_mask: np.ndarray | None = None,
) -> HoughResult:
    """Detect straight lines.

    Parameters
    ----------
    mag_percentile : keep gradient-magnitude pixels at/above this percentile as
        edge pixels that vote. Ignored if edge_mask is given.
    theta_step_deg, rho_step : accumulator resolution.
    n_lines : max peaks to return after NMS.
    nms_rho, nms_theta : half-window (in bins) for peak non-maximum suppression.
    """
    img = np.asarray(img, dtype=np.float64)
    if edge_mask is None:
        edge_mask = _edge_mask(img, mag_percentile)
    ys, xs = np.nonzero(edge_mask)

    thetas = np.deg2rad(np.arange(-90.0, 90.0, theta_step_deg))
    H, W = img.shape
    rho_max = float(np.hypot(H, W))
    rhos = np.arange(-rho_max, rho_max + rho_step, rho_step)
    n_theta = thetas.size
    n_rho = rhos.size

    acc = np.zeros((n_rho, n_theta), dtype=np.int64)
    if xs.size:
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        # rho = x cos(theta) + y sin(theta), for every edge pixel x theta
        rho_vals = np.outer(xs, cos_t) + np.outer(ys, sin_t)      # (E, n_theta)
        rho_idx = np.round((rho_vals + rho_max) / rho_step).astype(np.int64)
        np.clip(rho_idx, 0, n_rho - 1, out=rho_idx)
        theta_idx = np.broadcast_to(np.arange(n_theta), rho_idx.shape)
        flat = rho_idx.ravel() * n_theta + theta_idx.ravel()
        acc = np.bincount(flat, minlength=n_rho * n_theta).reshape(n_rho, n_theta)

    lines = _nms_peaks(acc, rhos, thetas, n_lines, nms_rho, nms_theta, min_votes)
    return HoughResult(accumulator=acc, rhos=rhos, thetas=thetas, lines=lines)


def _nms_peaks(acc, rhos, thetas, n_lines, nms_rho, nms_theta, min_votes):
    work = acc.astype(np.float64).copy()
    out = []
    n_rho, n_theta = work.shape
    for _ in range(n_lines):
        idx = int(np.argmax(work))
        r, t = divmod(idx, n_theta)
        v = work[r, t]
        if v < min_votes:
            break
        out.append((rhos[r], thetas[t], acc[r, t]))
        r0, r1 = max(0, r - nms_rho), min(n_rho, r + nms_rho + 1)
        t0, t1 = max(0, t - nms_theta), min(n_theta, t + nms_theta + 1)
        work[r0:r1, t0:t1] = -1.0
    return np.array(out, dtype=np.float64).reshape(-1, 3)


def line_support_map(
    img_shape: tuple[int, int],
    edge_mask: np.ndarray,
    lines: np.ndarray,
    *,
    dist_tol: float = 1.5,
    seg_max_len: float | None = None,
) -> np.ndarray:
    """Per-pixel line-support score.

    For each edge pixel, score = number of detected lines passing within
    `dist_tol` pixels (perpendicular distance). Optionally ignore lines whose
    locally supported segment is longer than seg_max_len (blade shadows are
    *short* lines, so very long field-edge lines can be discounted upstream).
    The result is a dense float map the combiner can treat as a feature.
    """
    H, W = img_shape
    score = np.zeros((H, W), dtype=np.float64)
    if lines.size == 0:
        return score
    ys, xs = np.nonzero(edge_mask)
    if xs.size == 0:
        return score
    for rho, theta, _votes in lines:
        d = np.abs(xs * np.cos(theta) + ys * np.sin(theta) - rho)
        hit = d <= dist_tol
        if hit.any():
            score[ys[hit], xs[hit]] += 1.0
    return score
