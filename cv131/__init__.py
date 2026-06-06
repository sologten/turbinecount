"""From-scratch CS 131 computer-vision primitives (pure NumPy).

Every function here is implemented from first principles using only NumPy.
OpenCV / scikit-learn are NEVER imported in this package. They appear only in
the ablation baselines (turbinecount/ablations.py), where swapping a scratch
primitive for a library call is the explicit object of study.

Lecture mapping:
  gaussian.py  -> Lec 2  (images and filters)
  sobel.py     -> Lec 3  (edges)
  hough.py     -> Lec 3  (lines)
  harris.py    -> Lec 6  (local features / corners)
  template.py  -> Lec 4-5 (cross-correlation / template matching)
"""

from .gaussian import gaussian_kernel, gaussian_smooth
from .sobel import sobel_gradients
from .hough import hough_lines, HoughResult
from .harris import harris_response, harris_corners
from .template import normalized_cross_correlation, matched_filter, match_template

__all__ = [
    "gaussian_kernel",
    "gaussian_smooth",
    "sobel_gradients",
    "hough_lines",
    "HoughResult",
    "harris_response",
    "harris_corners",
    "normalized_cross_correlation",
    "matched_filter",
    "match_template",
]
