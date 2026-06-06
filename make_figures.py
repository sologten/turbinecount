"""Diagnostic figures: (1) the band-parallax fringe on a real turbine patch vs a
background patch; (2) locked-test AUC by detector and by salience regime."""
from __future__ import annotations
import os, sys, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gt, features_s2 as F

OUT = os.path.dirname(os.path.abspath(__file__))

def _norm(a):
    a = a.astype(float); lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)

def hp(a):
    from cv131.gaussian import gaussian_smooth
    z = a - gaussian_smooth(a, 3.0); s = z.std()
    return (z - z.mean()) / (s + 1e-8)

def fig_fringe():
    # find a turbine patch with a strong red-blue fringe and a calm background patch
    files = sorted(glob.glob(f"{gt.WORK}/cache/s2/*.npz"))
    best = None
    for f in files[:40]:
        z = np.load(f); pos = z["pos"]
        for i in range(pos.shape[0]):
            v = F.plx_feats(pos[i]).get("plx_rb_peak", 0)
            if best is None or v > best[0]:
                best = (v, f, i)
    z = np.load(best[1]); tp = z["pos"][best[2]]; bg = z["neg"][0]
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.8))
    for row, (patch, tag) in enumerate([(tp, "turbine (operating)"), (bg, "background")]):
        r, g, b, n = patch
        rgb = np.stack([_norm(r), _norm(g), _norm(b)], -1)
        ax[row, 0].imshow(rgb); ax[row, 0].set_title(f"{tag}\nRGB composite", fontsize=10)
        R, B = hp(r), hp(b)
        d = R - B
        m = np.abs(d).max()
        im = ax[row, 1].imshow(d, cmap="RdBu_r", vmin=-m, vmax=m)
        ax[row, 1].set_title("red - blue band difference\n(1.0 s apart)", fontsize=10)
        G = hp(g); dg = G - B; mg = np.abs(dg).max() + 1e-9
        ax[row, 2].imshow(dg, cmap="RdBu_r", vmin=-mg, vmax=mg)
        ax[row, 2].set_title("green - blue band difference\n(0.32 s apart)", fontsize=10)
        disp = np.std(np.stack([hp(r), hp(g), hp(b)], 0), 0)
        ax[row, 3].imshow(disp, cmap="magma"); ax[row, 3].set_title("chromatic dispersion", fontsize=10)
        for c in range(4):
            ax[row, c].set_xticks([]); ax[row, c].set_yticks([])
    fig.suptitle("Band-parallax cue: a spinning turbine leaves a localized multi-band fringe; flat background does not",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = f"{OUT}/fig_parallax_fringe.png"; fig.savefig(p, dpi=130); plt.close(fig)
    print("saved", p, "| source", os.path.basename(best[1]), "rb_peak", round(best[0], 1))

def fig_results():
    r = json.load(open(f"{gt.WORK}/artifacts/final_test.json"))
    labels = ["CS131\nprimitives", "HOG+SVM\n(literature)", "static\nsalience",
              "parallax\n(novel)", "parallax+static\n(headline)"]
    vals = [r["cv131"]["per_proj"], r["hog"]["per_proj"], r["static"]["per_proj"],
            r["plx"]["per_proj"], r["headline"]["per_proj"]]
    colors = ["#999", "#bbb", "#6aa", "#e8743b", "#1f77b4"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax[0].text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    ax[0].set_ylim(0.7, 1.0); ax[0].set_ylabel("per-project AUC")
    ax[0].set_title("Held-out test (33 projects, 35 states)\nsingle locked evaluation", fontsize=11)
    ax[0].axhline(r["headline"]["per_proj"], ls="--", c="#1f77b4", alpha=0.4)
    # salience-stratified (hardcoded from final_test output)
    grp = ["low-salience\n(bright terrain)", "high-salience"]
    plx = [0.928, 0.977]; stat = [0.909, 0.990]; hog = [0.899, 0.980]
    x = np.arange(2); w = 0.25
    ax[1].bar(x - w, plx, w, label="parallax", color="#e8743b")
    ax[1].bar(x, stat, w, label="static", color="#6aa")
    ax[1].bar(x + w, hog, w, label="HOG", color="#bbb")
    ax[1].set_xticks(x); ax[1].set_xticklabels(grp); ax[1].set_ylim(0.85, 1.0)
    ax[1].set_ylabel("per-project AUC"); ax[1].legend(fontsize=9)
    ax[1].set_title("Where each cue wins (test)\nparallax leads on bright terrain", fontsize=11)
    fig.tight_layout()
    p = f"{OUT}/fig_test_results.png"; fig.savefig(p, dpi=130); plt.close(fig)
    print("saved", p)

if __name__ == "__main__":
    fig_fringe(); fig_results()
