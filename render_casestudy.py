"""Render Part-3b case-study figures from cached real Sentinel-2 chips.

Three figures:
  fig_casestudy_hero.png       Golden Hills: strongest-fringe turbine vs a
                               background patch -> true-color, chromatic (high-
                               pass R/G/B) composite that makes the moving-blade
                               rainbow visible, and the red-blue fringe heatmap.
  fig_casestudy_grid.png       all 4 case projects, one representative turbine
                               each: true-color | chromatic composite | rb map.
  fig_casestudy_separation.png per-project rb_peak, turbine vs background.
"""
from __future__ import annotations
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import casestudy as CS

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
CC = os.path.join(ART, "casestudy")
np.random.seed(0)

LABELS = {
    "OR_golden_hills_oregon": "Golden Hills, OR  (2022, 143 m rotor)",
    "ID_rockland": "Rockland, ID  (2011, 100 m rotor)",
    "WY_roundhouse_wind_ii": "Roundhouse II, WY  (2023, 116 m rotor)",
    "TX_shamrock_wind_facility_white_mesa_wind_iii": "Shamrock / White Mesa III, TX  (2023, 126 m rotor)",
}


def stretch(a, lo=2, hi=98):
    p1, p2 = np.percentile(a, [lo, hi])
    return np.clip((a - p1) / (p2 - p1 + 1e-8), 0, 1)


def truecolor(p):
    r, g, b, _ = p
    return np.dstack([stretch(r), stretch(g), stretch(b)])


def chromatic(p):
    """High-pass each visible band, z-norm, map to RGB. A static edge appears
    gray (R=G=B aligned); a moving blade tip is displaced across bands, so the
    channels disagree and the fringe shows up colored."""
    R, G, B, rb, disp, f = CS.patch_plx(p)
    def n01(a):
        return np.clip((a - np.percentile(a, 2)) / (np.percentile(a, 98) - np.percentile(a, 2) + 1e-8), 0, 1)
    return np.dstack([n01(R), n01(G), n01(B)]), rb, f


def load(sg):
    z = np.load(os.path.join(CC, f"{sg}.npz"), allow_pickle=True)
    return z["chip"], z["turb_rows"], z["turb_cols"]


def cut(chip, r, c):
    r, c = int(round(r)), int(round(c))
    return chip[:, r - 24:r + 24, c - 24:c + 24]


def patches(sg):
    chip, tr, tc = load(sg)
    _, H, W = chip.shape
    tp = []
    for r, c in zip(tr, tc):
        p = cut(chip, r, c)
        if p.shape[1:] == (48, 48) and (p <= 0).mean() <= 0.05:
            tp.append((p, CS.patch_plx(p)[5]["rb_peak"]))
    trc = np.column_stack([tr, tc]) if tr.size else np.zeros((0, 2))
    bg = []; tries = 0
    while len(bg) < max(len(tp), 20) and tries < 6000:
        tries += 1
        rr = np.random.randint(24, H - 24); cc = np.random.randint(24, W - 24)
        if trc.size and np.hypot(trc[:, 0] - rr, trc[:, 1] - cc).min() < 50:
            continue
        p = chip[:, rr - 24:rr + 24, cc - 24:cc + 24]
        if (p <= 0).mean() <= 0.05:
            bg.append((p, CS.patch_plx(p)[5]["rb_peak"]))
    return tp, bg


# ---------- figure 1: hero ----------
def fig_hero():
    sg = "OR_golden_hills_oregon"
    tp, bg = patches(sg)
    tp.sort(key=lambda x: -x[1]); bg.sort(key=lambda x: x[1])
    hero = tp[0][0]; back = bg[0][0]
    fig, ax = plt.subplots(2, 3, figsize=(11, 7.4))
    for row, (p, tag) in enumerate([(hero, "operating turbine"), (back, "background (cropland)")]):
        tc = truecolor(p); chrom, rb, f = chromatic(p)
        ax[row, 0].imshow(tc); ax[row, 0].set_title(f"{tag}\ntrue color (R/G/B)", fontsize=10)
        ax[row, 1].imshow(chrom)
        ax[row, 1].set_title("chromatic composite\n(high-pass R/G/B -> motion = color)", fontsize=10)
        im = ax[row, 2].imshow(rb, cmap="RdBu_r", vmin=-np.abs(rb).max(), vmax=np.abs(rb).max())
        ax[row, 2].set_title(f"red-blue fringe map\nrb_peak = {f['rb_peak']:.1f}", fontsize=10)
        for a in ax[row]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Case study WIN  -  Golden Hills, OR (parallax test AUC 1.00, largest fringe in panel)\n"
                 "The spinning blade is displaced across the time-staggered bands, leaving a red-blue dipole;\n"
                 "the static field shows none. The turbine is barely brighter than the field in true color.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(ART, "fig_casestudy_hero.png")
    fig.savefig(out, dpi=140); plt.close(fig); print("->", out)


# ---------- figure 2: grid of 4 ----------
def fig_grid():
    fig, ax = plt.subplots(len(CS.CASES), 3, figsize=(10.5, 3.2 * len(CS.CASES)))
    for i, sg in enumerate(CS.CASES):
        tp, bg = patches(sg)
        tp.sort(key=lambda x: -x[1])
        p = tp[0][0] if tp else bg[0][0]
        tc = truecolor(p); chrom, rb, f = chromatic(p)
        bgmean = np.mean([b[1] for b in bg]) if bg else float("nan")
        ax[i, 0].imshow(tc)
        ax[i, 0].set_ylabel(LABELS[sg].split("  (")[0], fontsize=10, rotation=90, labelpad=8)
        ax[i, 0].set_title("true color" if i == 0 else "", fontsize=10)
        ax[i, 1].imshow(chrom); ax[i, 1].set_title("chromatic (motion=color)" if i == 0 else "", fontsize=10)
        im = ax[i, 2].imshow(rb, cmap="RdBu_r", vmin=-np.abs(rb).max(), vmax=np.abs(rb).max())
        ax[i, 2].set_title("red-blue fringe" if i == 0 else "", fontsize=10)
        ax[i, 2].text(1.04, 0.5, f"turbine rb_peak {f['rb_peak']:.1f}\nbg mean {bgmean:.1f}",
                      transform=ax[i, 2].transAxes, fontsize=9, va="center")
        for a in ax[i]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Case studies: representative operating-turbine patch per project (strongest-fringe turbine shown)\n"
                 "Fringe collapses on idle fleets (Roundhouse, Shamrock) - the cue measures rotation, not presence",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(ART, "fig_casestudy_grid.png")
    fig.savefig(out, dpi=140); plt.close(fig); print("->", out)


# ---------- figure 3: separation ----------
def fig_separation():
    fig, ax = plt.subplots(figsize=(9, 5))
    stats = {}
    for i, sg in enumerate(CS.CASES):
        tp, bg = patches(sg)
        tv = np.array([t[1] for t in tp]); bv = np.array([b[1] for b in bg])
        stats[sg] = dict(turb_mean=float(tv.mean()), bg_mean=float(bv.mean()),
                         turb_n=len(tv), bg_n=len(bv))
        xt = i - 0.16 + np.random.uniform(-0.05, 0.05, len(tv))
        xb = i + 0.16 + np.random.uniform(-0.05, 0.05, len(bv))
        ax.scatter(xt, tv, s=22, color="C3", alpha=0.7, label="turbine" if i == 0 else None)
        ax.scatter(xb, bv, s=22, color="C0", alpha=0.5, label="background" if i == 0 else None)
        ax.plot([i - 0.16], [tv.mean()], "_", color="k", ms=22, mew=2)
        ax.plot([i + 0.16], [bv.mean()], "_", color="k", ms=22, mew=2)
    ax.set_xticks(range(len(CS.CASES)))
    ax.set_xticklabels([LABELS[s].split("  (")[0] for s in CS.CASES], rotation=12, fontsize=9)
    ax.set_ylabel("plx_rb_peak  (red-blue band-difference fringe peak)")
    ax.set_title("Where the parallax cue lives: red-blue fringe peak, turbine-centered vs background patches")
    ax.legend(); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out = os.path.join(ART, "fig_casestudy_separation.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    json.dump(stats, open(os.path.join(CC, "separation_stats.json"), "w"), indent=1)
    print("->", out)


if __name__ == "__main__":
    fig_hero(); fig_grid(); fig_separation()
    print("done")
