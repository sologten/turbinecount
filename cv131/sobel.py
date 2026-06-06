"""Sobel gradients (CS 131 Lec 3).

Returns gx, gy, magnitude, and orientation. The Sobel masks are the standard
separable [1 2 1] smoothing x [-1 0 1] differencing pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._conv import correlate2d

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)

SOBEL_Y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], dtype=np.float64)


@dataclass
class Gradients:
    gx: np.ndarray          # d/dx (positive = brightness increases to the right)
    gy: np.ndarray          # d/dy (positive = brightness increases downward)
    magnitude: np.ndarray
    orientation: np.ndarray  # radians in (-pi, pi], gradient direction


def sobel_gradients(img: np.ndarray) -> Gradients:
    img = np.asarray(img, dtype=np.float64)
    gx = correlate2d(img, SOBEL_X)
    gy = correlate2d(img, SOBEL_Y)
    mag = np.hypot(gx, gy)
    ori = np.arctan2(gy, gx)
    return Gradients(gx=gx, gy=gy, magnitude=mag, orientation=ori)
