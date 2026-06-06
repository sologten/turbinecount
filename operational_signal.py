"""Can our EXISTING data tell us whether the parallax cue behaves like an
operational (blade-motion) signal, without any external operational ground truth?

Four honest consistency tests (none is proof; all stated with caveats):
  1. Apparent fringe-present fraction + bimodality of the turbine fringe vs background.
  2. Within-project coherence: does the fringe cluster by farm MORE than appearance
     does? A shared per-scene factor (plausibly wind/operation at the instant of
     capture) should make turbines in one farm behave alike.
  3. Per-project detection recall vs per-project mean fringe (do we catch the ones
     that show motion?).
  4. Rotor-size scaling: conditional on showing a fringe, does it scale with rotor
     diameter, as blade-tip-speed physics predicts?
"""
from __future__ import annotations
import os, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(f"{HERE}/artifacts/s2_matrix.npz", allow_pickle=True)
X = z["X"].astype(float); y = z["y"].astype(int); g = z["groups"]; sp = z["split"]
names = [str(c) for c in z["feature_names"]]; NID = {n: i for i, n in enumerate(names)}
RB = NID["plx_rb_peak"]; ZAB = NID["stat_zabs"]
plx = [i for i, n in enumerate(names) if n.startswith("plx_")]

def slug(name, state):
    s = name.lower().replace(" ", "_")
    for ch in "()/.,'\"":
        s = s.replace(ch, "")
    return f"{state}_{s}"[:60]
panel = {slug(d["name"], d["state"]): d for d in json.load(open(f"{HERE}/panel.json"))}
det = {d["slug"]: d for d in json.load(open(f"{HERE}/artifacts/detection_benchmark.json"))["per_project"]}

pos = y == 1; neg = y == 0
rb_pos = X[pos, RB]; rb_neg = X[neg, RB]
print(f"positives {pos.sum()}  background {neg.sum()}")

# ---- 1. apparent fringe-present fraction + bimodality ----
thr = np.percentile(rb_neg, 95)   # background 95th pct = "fringe clearly present"
frac_pos = float((rb_pos > thr).mean()); frac_neg = float((rb_neg > thr).mean())
print("\n[1] Apparent fringe-present (rb_peak > background 95th pct = %.2f)" % thr)
print(f"    turbines above: {frac_pos:.1%}   background above: {frac_neg:.1%} (by construction ~5%)")
# 2-component mixture on log fringe of turbines
lr = np.log(rb_pos + 1e-3).reshape(-1, 1)
gm = GaussianMixture(2, random_state=0).fit(lr)
mu = np.exp(gm.means_.ravel()); w = gm.weights_; order = np.argsort(mu)
lo, hi = order
bg_med = np.median(rb_neg)
print(f"    GMM low mode: mean rb {mu[lo]:.2f}, weight {w[lo]:.1%}  (background median {bg_med:.2f})")
print(f"    GMM high mode: mean rb {mu[hi]:.2f}, weight {w[hi]:.1%}")

# ---- 2. within-project coherence (variance explained by project) ----
def eta2(values, groups):
    """Fraction of variance in `values` explained by group membership (one-way)."""
    gm_ = values.mean(); ssb = 0.0; ssw = 0.0
    for k in np.unique(groups):
        v = values[groups == k]
        if v.size < 2:
            continue
        ssb += v.size * (v.mean() - gm_) ** 2
        ssw += ((v - v.mean()) ** 2).sum()
    return ssb / (ssb + ssw)
gp = g[pos]
e_plx = eta2(np.log(rb_pos + 1e-3), gp)
e_sal = eta2(X[pos, ZAB], gp)
# also for a pure-contrast control: center brightness
e_bri = eta2(X[pos, NID["stat_center_bright"]], gp)
print("\n[2] Variance among TURBINES explained by which farm they are in:")
print(f"    parallax fringe (log rb_peak): {e_plx:.1%}")
print(f"    appearance salience (z_abs):   {e_sal:.1%}")
print(f"    raw center brightness:         {e_bri:.1%}")
print("    -> higher for parallax => fringe shares a per-scene factor (consistent with shared wind/operation)")

# ---- 3. per-project detection recall vs mean fringe ----
slugs = [k for k in np.unique(gp) if k in det]
mean_rb = np.array([X[pos & (g == k), RB].mean() for k in slugs])
recall = np.array([det[k]["recall"] for k in slugs])
sal_pp = np.array([X[pos & (g == k), ZAB].mean() for k in slugs])
def pear(a, b):
    return float(np.corrcoef(a, b)[0, 1])
print("\n[3] Per-project detection recall vs per-project signal (n=%d projects):" % len(slugs))
print(f"    recall vs mean fringe (rb_peak): r = {pear(mean_rb, recall):+.2f}")
print(f"    recall vs mean salience (z_abs): r = {pear(sal_pp, recall):+.2f}")

# ---- 4. rotor-size scaling ----
rot = np.array([panel[k]["rotor_d"] for k in slugs if k in panel])
mean_rb_r = np.array([X[pos & (g == k), RB].mean() for k in slugs if k in panel])
hi_rb_r = np.array([np.percentile(X[pos & (g == k), RB], 75) for k in slugs if k in panel])
print("\n[4] Rotor diameter vs fringe across projects (n=%d):" % len(rot))
print(f"    rotor vs MEAN fringe:        r = {pear(rot, mean_rb_r):+.2f}  (operation dominates, so weak/noisy expected)")
print(f"    rotor vs 75th-pct fringe:    r = {pear(rot, hi_rb_r):+.2f}  (conditional on stronger-fringe turbines)")

# ---- detected vs missed positives (test only, frozen-style parallax model) ----
tr = sp == "train"; te = sp == "test"
sc = StandardScaler().fit(X[tr][:, plx]); m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr][:, plx]), y[tr])
pte = m.predict_proba(sc.transform(X[te][:, plx]))[:, 1]
poste = (y[te] == 1)
hi_mask = poste & (pte >= 0.5); lo_mask = poste & (pte < 0.5)
print("\n[5] Held-out turbines split by parallax model (test):")
print(f"    scored-high turbines: {hi_mask.sum()}  median rb {np.median(X[te][hi_mask, RB]):.2f}")
print(f"    scored-low  turbines: {lo_mask.sum()}  median rb {np.median(X[te][lo_mask, RB]):.2f}")
print(f"    background (test):    median rb {np.median(X[te][(y[te]==0), RB]):.2f}")
print("    -> if missed turbines look like background in fringe, the recall ceiling is 'no motion present', not detector failure")

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
bins = np.linspace(0, 12, 50)
ax[0].hist(np.clip(rb_neg, 0, 12), bins=bins, density=True, alpha=0.6, color="#1f77b4", label="background")
ax[0].hist(np.clip(rb_pos, 0, 12), bins=bins, density=True, alpha=0.55, color="#2ca02c", label="turbines")
ax[0].axvline(thr, color="k", ls="--", lw=1, label=f"bg 95th pct = {thr:.1f}")
ax[0].set_xlabel("plx_rb_peak (red-blue fringe)"); ax[0].set_ylabel("density")
ax[0].set_title(f"(a) Fringe present on {frac_pos:.0%} of turbines\nvs {frac_neg:.0%} of background")
ax[0].legend()
ax[1].scatter(mean_rb, recall, s=45, color="#2ca02c", edgecolor="k", linewidth=0.4)
b, a = np.polyfit(mean_rb, recall, 1)
xs = np.linspace(mean_rb.min(), mean_rb.max(), 20)
ax[1].plot(xs, b * xs + a, "r-", lw=1.2)
ax[1].set_xlabel("per-project mean fringe (rb_peak)"); ax[1].set_ylabel("detection recall")
ax[1].set_title(f"(b) Recall tracks fringe strength\nPearson r = {pear(mean_rb, recall):+.2f}")
ax[1].grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{HERE}/artifacts/fig_operational_signal.png", dpi=140)
print("\nsaved -> artifacts/fig_operational_signal.png")
