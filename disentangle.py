"""Disentangle: does the band-parallax family carry signal BEYOND static
local-contrast, or is it just re-encoded contrast? Two tests, train-only,
GroupKFold-by-project.

  Test 1 (orthogonalization): inside each CV fold, linearly regress every plx_
  feature on the static features (stat_+cv131_) fit on the training fold, and
  keep the RESIDUAL. If a model on residual-plx still has AUC >> 0.5, the
  parallax family contains structure not explained by static contrast.

  Test 2 (contrast-conditional AUC): bin all patches by the static salience cue
  stat_zabs into deciles; within each bin (where positives and negatives have
  ~equal static contrast, so static salience cannot separate them) compute the
  pos-vs-neg AUC of (a) the best single parallax feature and (b) stat_zabs
  itself. If parallax stays >0.5 within bins while stat_zabs is ~0.5, parallax
  carries information beyond local contrast.
"""
from __future__ import annotations
import numpy as np, json
import harness as H
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

d = H.load(); X, y, g = H.train_view(d); names = d["names"]
PLX = H.cols(names, ["plx_"]); STAT = H.cols(names, ["stat_", "cv131_"])
zabs_i = names.index("stat_zabs")

def auc_pp(yv, p, gv):
    a = [roc_auc_score(yv[gv == k], p[gv == k]) for k in np.unique(gv)
         if len(np.unique(yv[gv == k])) == 2]
    return roc_auc_score(yv, p), float(np.mean(a))

# ---- Test 1: orthogonalize plx against static, inside CV ----
gkf = GroupKFold(5)
oof_raw = np.full(len(y), np.nan); oof_res = np.full(len(y), np.nan)
for tr, va in gkf.split(X, y, g):
    # residualize each plx feature on static (fit on train fold only)
    S = StandardScaler().fit(X[tr][:, STAT])
    Str, Sva = S.transform(X[tr][:, STAT]), S.transform(X[va][:, STAT])
    Rtr = np.zeros((len(tr), len(PLX))); Rva = np.zeros((len(va), len(PLX)))
    for j, c in enumerate(PLX):
        lr = LinearRegression().fit(Str, X[tr][:, c])
        Rtr[:, j] = X[tr][:, c] - lr.predict(Str)
        Rva[:, j] = X[va][:, c] - lr.predict(Sva)
    sc = StandardScaler().fit(Rtr)
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Rtr), y[tr])
    oof_res[va] = m.predict_proba(sc.transform(Rva))[:, 1]
    # raw plx for reference
    scr = StandardScaler().fit(X[tr][:, PLX])
    mr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scr.transform(X[tr][:, PLX]), y[tr])
    oof_raw[va] = mr.predict_proba(scr.transform(X[va][:, PLX]))[:, 1]

raw = auc_pp(y, oof_raw, g); res = auc_pp(y, oof_res, g)
print("=== Test 1: parallax orthogonalized against static (stat_+cv131_) ===")
print(f"  raw parallax           pooled {raw[0]:.3f}  per-proj {raw[1]:.3f}")
print(f"  residual parallax      pooled {res[0]:.3f}  per-proj {res[1]:.3f}")
print(f"  -> signal surviving removal of static-contrast: {'YES' if res[1]>0.6 else 'weak'}")

# ---- Test 2: contrast-conditional AUC (within stat_zabs deciles) ----
zabs = X[:, zabs_i]
# best single parallax feature by |pointbiserial| with y
best_j = max(PLX, key=lambda c: abs(np.corrcoef(X[:, c], y)[0, 1]))
best_name = names[best_j]
edges = np.quantile(zabs, np.linspace(0, 1, 11))
print(f"\n=== Test 2: within-contrast-bin AUC (static cue held ~constant per bin) ===")
print(f"  best single parallax feature = {best_name}")
print(f"  {'zabs bin':>14s} {'n':>5s} {'pos%':>5s} {best_name[:16]:>16s} {'stat_zabs':>9s}")
rows = []
for b in range(10):
    m = (zabs >= edges[b]) & (zabs <= edges[b+1] if b == 9 else zabs < edges[b+1])
    if m.sum() < 50 or len(np.unique(y[m])) < 2:
        continue
    a_plx = roc_auc_score(y[m], X[m, best_j]); a_z = roc_auc_score(y[m], zabs[m])
    rows.append((float(edges[b]), float(edges[b+1]), int(m.sum()), float(y[m].mean()), a_plx, a_z))
    print(f"  [{edges[b]:5.1f},{edges[b+1]:5.1f}) {m.sum():5d} {y[m].mean()*100:4.0f}% {a_plx:16.3f} {a_z:9.3f}")
within_plx = np.mean([r[4] for r in rows]); within_z = np.mean([r[5] for r in rows])
print(f"  mean within-bin AUC: parallax {within_plx:.3f}  vs  stat_zabs {within_z:.3f}")
print(f"  -> parallax separates pos/neg within fixed-contrast bins: {'YES' if within_plx>0.6 else 'no'}")

json.dump({"test1_raw_plx": raw, "test1_residual_plx": res,
           "test2_best_feature": best_name,
           "test2_within_bin_auc_parallax": within_plx,
           "test2_within_bin_auc_zabs": within_z,
           "test2_bins": rows},
          open(f"{H.MATRIX.rsplit('/',1)[0]}/disentangle.json", "w"), indent=1)
