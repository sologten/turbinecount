"""THE single, final locked-test evaluation. Run exactly once.

Every model below was selected/frozen using train-only GroupKFold CV (see
sweep_s2.py, mechanism_audit.py, disentangle.py). Here we fit each frozen
configuration on the FULL train split and evaluate ONCE on the 33 held-out test
projects, which were never touched during development. We report pooled AUC,
mean per-project AUC, average precision, a salience-stratified breakdown, and a
project-level bootstrap CI for the headline detector.
"""
from __future__ import annotations
import os, numpy as np, json, joblib
import harness as H
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

d = H.load(); names = d["names"]
tr = d["split"] == "train"; te = d["split"] == "test"
Xtr, ytr, gtr = d["X"][tr], d["y"][tr], d["groups"][tr]
Xte, yte, gte = d["X"][te], d["y"][te], d["groups"][te]
print(f"TRAIN {len(ytr)} ({len(set(gtr))} proj)  |  TEST {len(yte)} ({len(set(gte))} proj)  [touched once]")

def per_proj(yv, p, gv):
    a = [roc_auc_score(yv[gv == k], p[gv == k]) for k in np.unique(gv)
         if len(np.unique(yv[gv == k])) == 2]
    return float(np.mean(a)), np.array(a), list(np.unique(gv))

def fit_score(cols, make, name):
    sc = StandardScaler().fit(Xtr[:, cols])
    m = make().fit(sc.transform(Xtr[:, cols]), ytr)
    s = (m.decision_function(sc.transform(Xte[:, cols])) if not hasattr(m, "predict_proba")
         else m.predict_proba(sc.transform(Xte[:, cols]))[:, 1])
    pooled = roc_auc_score(yte, s); ap = average_precision_score(yte, s)
    mpp, aucs, _ = per_proj(yte, s, gte)
    print(f"  {name:34s} pooledAUC {pooled:.3f} | per-proj {mpp:.3f} | AP {ap:.3f}")
    return dict(name=name, pooled=pooled, per_proj=mpp, ap=ap), s

C = lambda *p: H.cols(names, list(p))
print("\n=== Held-out TEST results (33 projects, 35 states represented in panel) ===")
res = {}
res["cv131"], _ = fit_score(C("cv131_"), lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
                            "CS131 primitives (old fallback)")
res["static"], _ = fit_score(C("stat_", "cv131_"), lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
                             "Static salience (stat+cv131)")
res["hog"], _ = fit_score(C("hog_"), lambda: LinearSVC(C=1.0, class_weight="balanced", max_iter=5000),
                          "HOG + linear SVM (literature)")
res["plx"], splx = fit_score(C("plx_"), lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
                             "Parallax/motion only (novel)")

# headline = the frozen best model saved by the sweep (RF plx+static)
best = joblib.load(f"{ART}/best_s2_model.joblib")
bcols = np.array([names.index(c) for c in best["cols"]])
sb = best["model"].predict_proba(best["scaler"].transform(Xte[:, bcols]))[:, 1]
pooled_b = roc_auc_score(yte, sb); ap_b = average_precision_score(yte, sb)
mpp_b, aucs_b, projs_b = per_proj(yte, sb, gte)
print(f"  {'HEADLINE: '+type(best['model']).__name__+' plx+static':34s} pooledAUC {pooled_b:.3f} | per-proj {mpp_b:.3f} | AP {ap_b:.3f}")
res["headline"] = dict(name="RF plx+static (frozen best)", pooled=pooled_b, per_proj=mpp_b, ap=ap_b)

# project-level bootstrap CI of headline per-project AUC
rng = np.random.default_rng(0)
boot = [np.mean(rng.choice(aucs_b, len(aucs_b), replace=True)) for _ in range(2000)]
ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
print(f"  headline per-project AUC 95% bootstrap CI: [{ci[0]:.3f}, {ci[1]:.3f}]  (n_test_proj={len(aucs_b)})")

# salience-stratified test (does parallax hold on low-salience test projects?)
zi = names.index("stat_zabs")
sal = {k: Xte[(gte == k) & (yte == 1), zi].mean() for k in np.unique(gte)}
med = np.median(list(sal.values()))
print("\n=== TEST split by appearance-salience (median of mean pos stat_zabs) ===")
for lab, sel in (("LOW-salience", lambda k: sal[k] <= med), ("HIGH-salience", lambda k: sal[k] > med)):
    ks = [k for k in np.unique(gte) if sel(k)]
    mask = np.isin(gte, ks)
    def a(cols, make):
        scl = StandardScaler().fit(Xtr[:, cols]); mm = make().fit(scl.transform(Xtr[:, cols]), ytr)
        pr = mm.predict_proba(scl.transform(Xte[mask][:, cols]))[:, 1] if hasattr(mm, "predict_proba") else mm.decision_function(scl.transform(Xte[mask][:, cols]))
        mp, _, _ = per_proj(yte[mask], pr, gte[mask]); return mp
    ap_ = a(C("plx_"), lambda: LogisticRegression(max_iter=2000, class_weight="balanced"))
    as_ = a(C("stat_", "cv131_"), lambda: LogisticRegression(max_iter=2000, class_weight="balanced"))
    ah_ = a(C("hog_"), lambda: LogisticRegression(max_iter=2000, class_weight="balanced"))
    print(f"  {lab} ({len(ks)} proj): parallax {ap_:.3f} | static {as_:.3f} | HOG {ah_:.3f}")

# NAIP locked test
print("\n=== NAIP high-res benchmark, held-out test projects (single eval) ===")
try:
    nb = joblib.load(f"{ART}/best_naip_model.joblib")
    z = np.load(f"{ART}/naip_test_matrix.npz", allow_pickle=True)
    Xn, yn, gn = z["X"], z["y"].astype(int), z["groups"]
    nnames = list(z["feature_names"]); ncols = [nnames.index(c) for c in nb.get("cols", nnames)] if "cols" in nb else slice(None)
    mdl = nb["model"] if isinstance(nb, dict) else nb
    Xnc = Xn[:, ncols] if isinstance(ncols, list) else Xn
    if isinstance(nb, dict) and "scaler" in nb and nb["scaler"] is not None:
        Xnc = nb["scaler"].transform(Xnc)
    sn = mdl.predict_proba(Xnc)[:, 1]
    mppn, _, _ = per_proj(yn, sn, gn)
    print(f"  NAIP best ({type(mdl).__name__}): pooledAUC {roc_auc_score(yn,sn):.3f} | per-proj {mppn:.3f} | n_proj {len(set(gn))}")
    res["naip"] = dict(pooled=float(roc_auc_score(yn, sn)), per_proj=mppn, n_proj=len(set(gn)))
except Exception as e:
    print("  NAIP eval skipped:", type(e).__name__, str(e)[:100])

res["headline_ci"] = ci
json.dump(res, open(f"{ART}/final_test.json", "w"), indent=1)
print("\nsaved -> final_test.json")
