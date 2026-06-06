"""Build one summary figure for the Part-3a ablations, from saved JSON traces."""
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "artifacts", "ablations")
S = json.load(open(os.path.join(A, "ablations_summary.json")))
FAMC = {"stat": "#1f77b4", "cv131": "#d62728", "plx": "#2ca02c", "hog": "#9467bd"}

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# (a) single-feature univariate test per-proj AUC, top 14
rows = sorted(S["single_feature_auc"], key=lambda r: r["perproj_test"], reverse=True)[:14][::-1]
y = np.arange(len(rows))
ax[0, 0].barh(y, [r["perproj_test"] for r in rows],
              color=[FAMC[r["family"]] for r in rows])
ax[0, 0].set_yticks(y); ax[0, 0].set_yticklabels([r["feature"] for r in rows], fontsize=8)
ax[0, 0].axvline(0.5, color="0.5", ls=":", lw=1)
ax[0, 0].set_xlim(0.45, 1.0)
ax[0, 0].set_xlabel("held-out test per-project AUC (single feature, oriented)")
ax[0, 0].set_title("(a) Single-feature separability (top 14, non-HOG)\ncolor = family", fontsize=11)
for fam, c in FAMC.items():
    if fam != "hog":
        ax[0, 0].plot([], [], "s", color=c, label=fam)
ax[0, 0].legend(fontsize=8, loc="lower right")

# (b) CS131 primitives: solo univariate AUC + leave-one-out drop
eff = S["cs131_effect"]
prims = ["cv131_sobel_center", "cv131_harris", "cv131_template", "cv131_hough"]
solo = [eff["solo_univariate"][p]["perproj_test"] for p in prims]
loo = [eff["leave_one_out"][p]["drop_vs_all_test_pp"] for p in prims]
yy = np.arange(len(prims))
ax[0, 1].barh(yy - 0.2, solo, height=0.38, color="#d62728", label="solo test AUC")
ax[0, 1].barh(yy + 0.2, loo, height=0.38, color="#ff9896",
              label="AUC drop if removed from cv131 family")
ax[0, 1].axvline(0.5, color="0.5", ls=":", lw=1)
ax[0, 1].set_yticks(yy); ax[0, 1].set_yticklabels([p.replace("cv131_", "") for p in prims], fontsize=9)
ax[0, 1].set_xlabel("AUC (solo)  /  AUC drop (leave-one-out)")
cv131_all = eff["cv131_all"]["test_perproj"]
ax[0, 1].set_title(f"(b) CS131 primitive effect (family AUC {cv131_all:.3f})\n"
                   "sobel carries the family; hough ~ chance (0.52)", fontsize=11)
ax[0, 1].legend(fontsize=8, loc="center right")

# (c) family ablation: each alone (RF) + leave-one-family-out drop (RF)
fam = S["family_ablation"]
fams = ["plx", "stat", "hog", "cv131"]
alone = [fam["alone"][f]["rf"]["test_perproj"] for f in fams]
allrf = fam["all_rf"]["test_perproj"]
loo_f = [fam["leave_one_out"][f]["drop_vs_all"] for f in fams]
xx = np.arange(len(fams))
ax[1, 0].bar(xx - 0.2, alone, width=0.38, color=[FAMC[f] for f in fams], label="family alone (RF)")
ax[1, 0].bar(xx + 0.2, loo_f, width=0.38, color="0.7", label="drop when removed from ALL")
ax[1, 0].axhline(allrf, color="k", ls="--", lw=1, label=f"ALL families (RF) = {allrf:.3f}")
ax[1, 0].set_xticks(xx); ax[1, 0].set_xticklabels(fams)
ax[1, 0].set_ylabel("test per-project AUC  /  AUC drop")
ax[1, 0].set_title("(c) Family ablation (random forest)\nparallax is the load-bearing family; cv131 & hog redundant", fontsize=11)
ax[1, 0].legend(fontsize=8)

# (d) salience-stratified per family
st = S["salience_stratified"]
fams2 = ["plx", "static", "hog", "cv131"]
low = [st["low"][f] for f in fams2]; high = [st["high"][f] for f in fams2]
xx = np.arange(len(fams2))
ax[1, 1].bar(xx - 0.2, low, width=0.38, color="#8c6d31", label=f"low salience ({st['low']['n_proj']} proj)")
ax[1, 1].bar(xx + 0.2, high, width=0.38, color="#e7ba52", label=f"high salience ({st['high']['n_proj']} proj)")
ax[1, 1].set_xticks(xx); ax[1, 1].set_xticklabels(fams2)
ax[1, 1].set_ylim(0.6, 1.0)
ax[1, 1].set_ylabel("test per-project AUC")
ax[1, 1].set_title("(d) Salience-stratified test\nparallax wins on bright/textured (low-salience) terrain", fontsize=11)
ax[1, 1].legend(fontsize=8)

fig.suptitle("TurbineCount v2 - ablation summary (held-out test, GroupKFold-developed, cached feature matrix)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(HERE, "artifacts", "fig_ablation_summary.png")
fig.savefig(out, dpi=140); print("->", out)
