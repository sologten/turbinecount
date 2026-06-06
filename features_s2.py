"""Feature extraction for Sentinel-2 R/G/B/NIR patches (4 x 48 x 48 reflectance).

Three named feature families (prefixes let the modeling ablate by family):

  stat_  : basic salience / brightness statistics, incl. the local-contrast
           z-score that the original project showed is the binding cue at 10 m.
  cv131_ : CS 131 static primitives (Gaussian, Sobel, Hough line support, Harris
           response, matched-filter template) summarized over the patch. This is
           the from-scratch syllabus baseline / documented fallback.
  plx_   : *band-parallax / blade-motion* features (the novel contribution).
           Sentinel-2 captures B02(blue,t=0), B03(green,+0.324s), B04(red,+1.005s)
           at offset times, so a spinning blade sits at displaced pixels across
           bands. After per-band contrast normalization, a MOVING object leaves a
           localized multi-band dipole/fringe in the band-difference maps that a
           static object (building, pad, field) does not. These features measure
           that fringe; they do not depend on the static structure being locally
           salient, which is exactly the 10 m failure mode of the baseline.

Optional compact HOG descriptor is computed separately (hog_*) for the
advanced-ML comparison.
"""
from __future__ import annotations
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gt
from cv131.gaussian import gaussian_smooth
from cv131.sobel import sobel_gradients
from cv131.harris import harris_response
from cv131.hough import _edge_mask, hough_lines, line_support_map
from cv131.template import matched_filter

H = 48; C = H // 2
def _center(a, r=12):
    return a[C - r:C + r, C - r:C + r]

def _znorm(a):
    s = a.std()
    return (a - a.mean()) / (s + 1e-8)

# ---------- static families ----------
def stat_feats(p):
    red, grn, blu, nir = p
    gray = 0.299 * red + 0.587 * grn + 0.114 * blu
    sm = gaussian_smooth(gray, 1.0)
    cen = _center(sm, 4); sur = sm.copy(); sur[C-8:C+8, C-8:C+8] = np.nan
    mu, sd = np.nanmean(sur), np.nanstd(sur) + 1e-8
    z_bright = (np.nanmax(cen) - mu) / sd
    z_dark = (mu - np.nanmin(cen)) / sd
    nr = np.median(_center(nir, 4)) / (np.median(_center(red, 4)) + 1e-6)
    return {
        "stat_zbright": float(z_bright),
        "stat_zdark": float(z_dark),
        "stat_zabs": float(max(z_bright, z_dark)),
        "stat_center_bright": float(np.median(cen)),
        "stat_contrast": float(sm.std()),
        "stat_nir_red": float(nr),
    }

def cv131_feats(p):
    red, grn, blu, nir = p
    gray = 0.299 * red + 0.587 * grn + 0.114 * blu
    sm = gaussian_smooth(gray, 1.2)
    smn = (sm - sm.min()) / (np.ptp(sm) + 1e-8)
    # Hough short-line support
    em = _edge_mask(smn, 90.0)
    hough = 0.0
    if em.sum() >= 5:
        res = hough_lines(smn, edge_mask=em, n_lines=8, min_votes=5)
        if res.lines.size:
            hough = float(line_support_map(smn.shape, em, res.lines, dist_tol=1.5).max())
    harris = float(harris_response(smn, sigma=1.5).max())
    # matched filter to a small bright bump (nacelle proxy)
    r = 4; y, x = np.mgrid[-r:r+1, -r:r+1]
    t = np.exp(-(x*x+y*y)/(2*2.0**2)); t -= t.mean()
    tmpl = float(matched_filter(smn, t).max())
    g = sobel_gradients(smn)
    sob = float(np.mean(_center(g.magnitude, 8)))
    return {"cv131_hough": hough, "cv131_harris": harris,
            "cv131_template": tmpl, "cv131_sobel_center": sob}

# ---------- parallax / motion family (novel) ----------
def _fft_xcorr_offset(a, b, maxs=8):
    """Best integer shift aligning b to a (within +/-maxs), and peak corr."""
    a = _znorm(a); b = _znorm(b)
    fa = np.fft.rfft2(a); fb = np.fft.rfft2(b)
    cc = np.fft.irfft2(fa * np.conj(fb), s=a.shape)
    cc = np.fft.fftshift(cc) / a.size
    cy, cx = np.array(cc.shape) // 2
    win = cc[cy-maxs:cy+maxs+1, cx-maxs:cx+maxs+1]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    dy, dx = iy - maxs, ix - maxs
    return dy, dx, float(win.max())

def plx_feats(p):
    red, grn, blu, nir = p
    # per-band high-pass + contrast normalization (compare structure, not albedo)
    def hp(a):
        return _znorm(a - gaussian_smooth(a, 3.0))
    R, G, B = hp(red), hp(grn), hp(blu)
    f = {}
    # band-difference fringe maps (red lags blue most -> strongest motion fringe)
    for nm, D in (("rb", R - B), ("gb", G - B), ("rg", R - G)):
        Dc = _center(D, 12)
        f[f"plx_{nm}_energy"] = float(np.mean(Dc**2))
        f[f"plx_{nm}_peak"] = float(np.max(np.abs(Dc)))
        f[f"plx_{nm}_ptp"] = float(Dc.max() - Dc.min())
        # dipole separation: distance between strongest + and - lobes
        yp, xp = np.unravel_index(np.argmax(Dc), Dc.shape)
        yn, xn = np.unravel_index(np.argmin(Dc), Dc.shape)
        f[f"plx_{nm}_dipole_sep"] = float(np.hypot(yp-yn, xp-xn))
    # chromatic dispersion: per-pixel spread across visible bands, center
    disp = np.std(np.stack([R, G, B], 0), axis=0)
    f["plx_chroma_disp"] = float(np.mean(_center(disp, 12)))
    f["plx_chroma_disp_peak"] = float(np.max(_center(disp, 12)))
    # cross-correlation parallax: blue->red offset magnitude + decorrelation
    dy, dx, pk = _fft_xcorr_offset(_center(B, 16), _center(R, 16), maxs=6)
    f["plx_xcorr_offset"] = float(np.hypot(dy, dx))
    f["plx_xcorr_peak"] = pk            # lower => more decorrelated => motion
    f["plx_decorr"] = float(1.0 - pk)
    # physical time-ordering consistency: rb dipole should align with gb dipole
    # (same direction, larger magnitude ~3x). Use central-map correlation.
    Drb = _center(R - B, 12).ravel(); Dgb = _center(G - B, 12).ravel()
    cc = np.corrcoef(Drb, Dgb)[0, 1] if Drb.std() > 0 and Dgb.std() > 0 else 0.0
    f["plx_time_consistency"] = float(cc if np.isfinite(cc) else 0.0)
    # ratio of fringe energy red-blue (1.005s) to green-blue (0.324s): motion grows w/ time
    f["plx_energy_ratio_rb_gb"] = float((f["plx_rb_energy"] + 1e-6) / (f["plx_gb_energy"] + 1e-6))
    return f

def hog_feats(p):
    from skimage.feature import hog
    red, grn, blu, nir = p
    gray = 0.299 * red + 0.587 * grn + 0.114 * blu
    g = (gray - gray.min()) / (np.ptp(gray) + 1e-8)
    h = hog(g, orientations=6, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
            feature_vector=True)
    return {f"hog_{i}": float(v) for i, v in enumerate(h)}

def patch_features(p, with_hog=True):
    d = {}
    d.update(stat_feats(p)); d.update(cv131_feats(p)); d.update(plx_feats(p))
    if with_hog:
        d.update(hog_feats(p))
    return d
