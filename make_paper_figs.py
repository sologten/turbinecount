"""Compact, paper-ready figures for the CVPR report (honest method-effectiveness study)."""
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
FIG = os.path.join(HERE, "report", "figs")
os.makedirs(FIG, exist_ok=True)
ft = json.load(open(f"{ART}/final_test.json"))
ab = json.load(open(f"{ART}/ablations/ablations_summary.json"))
db = json.load(open(f"{ART}/detection_benchmark.json"))["summary"]
plt.rcParams.update({"font.size": 12})

# ---- Figure: effectiveness in one row: (a) per-method AUC, (b) salience strat, (c) AUC vs detection F1 ----
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))

# (a) per-method classification per-proj AUC
methods = [("CS131\nprimitives", ft["cv131"]["per_proj"], "#d62728"),
           ("HOG+SVM", ft["hog"]["per_proj"], "#9467bd"),
           ("static\nsalience", ft["static"]["per_proj"], "#1f77b4"),
           ("parallax\n(novel)", ft["plx"]["per_proj"], "#2ca02c"),
           ("fused\n(RF)", ft["headline"]["per_proj"], "#111111")]
x = np.arange(len(methods))
ax[0].bar(x, [m[1] for m in methods], color=[m[2] for m in methods])
for i, m in enumerate(methods):
    ax[0].text(i, m[1] + 0.006, f"{m[1]:.3f}", ha="center", fontsize=10)
ax[0].axhline(0.5, color="0.5", ls=":", lw=1)
ax[0].set_xticks(x); ax[0].set_xticklabels([m[0] for m in methods], fontsize=10)
ax[0].set_ylim(0.5, 1.0); ax[0].set_ylabel("held-out per-project AUC")
ax[0].set_title("(a) Method effectiveness (classification)")

# (b) salience-stratified
st = ab["salience_stratified"]; fams = ["parallax", "static", "HOG", "CS131"]
keys = ["plx", "static", "hog", "cv131"]
low = [st["low"][k] for k in keys]; high = [st["high"][k] for k in keys]
xx = np.arange(len(fams))
ax[1].bar(xx - 0.2, low, 0.38, color="#8c6d31", label=f"low salience (n={st['low']['n_proj']})")
ax[1].bar(xx + 0.2, high, 0.38, color="#e7ba52", label=f"high salience (n={st['high']['n_proj']})")
ax[1].set_xticks(xx); ax[1].set_xticklabels(fams, fontsize=10)
ax[1].set_ylim(0.6, 1.0); ax[1].set_ylabel("per-project AUC")
ax[1].set_title("(b) Bright-terrain robustness"); ax[1].legend(fontsize=9, loc="lower left")

# (c) classification AUC vs honest detection F1
labels = ["classification\nAUC (fused)", "detection\nF1 (fused)", "count fit\nR-squared"]
vals = [ft["headline"]["per_proj"], db["micro"]["f1"], max(db["count_r2_fit"], 0)]
cols = ["#2ca02c", "#d62728", "#d62728"]
ax[2].bar(np.arange(3), vals, color=cols)
for i, v in enumerate([ft["headline"]["per_proj"], db["micro"]["f1"], db["count_r2_fit"]]):
    ax[2].text(i, max(v, 0) + 0.02, f"{v:.2f}", ha="center", fontsize=10)
ax[2].set_xticks(np.arange(3)); ax[2].set_xticklabels(labels, fontsize=10)
ax[2].set_ylim(0, 1.05); ax[2].set_title("(c) The honest gap: ranking vs detecting")

fig.tight_layout()
fig.savefig(f"{FIG}/fig_effectiveness.png", dpi=150)
print("-> fig_effectiveness.png")

# copy/resize existing figures into report/figs
from PIL import Image
for src, dst, w in [("fig_casestudy_hero.png", "fig_hero.png", 1300),
                    ("fig_casestudy_separation.png", "fig_separation.png", 1200),
                    ("fig_casestudy_grid.png", "fig_grid.png", 1100)]:
    im = Image.open(f"{ART}/{src}")
    h = round(w * im.size[1] / im.size[0])
    im.resize((w, h)).save(f"{FIG}/{dst}")
    print("->", dst)
print("done")
