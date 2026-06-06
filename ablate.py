"""Rigorous, leakage-free ablation harness for TurbineCount v2 (Part 3a).

Runs entirely on the CACHED feature matrix artifacts/s2_matrix.npz (no network,
no re-featurization), so it is exactly reproducible. Mirrors final_test.py /
harness.py methodology:
  * model development & CV use split=='train' ONLY, GroupKFold(group=project)
  * StandardScaler fit on train folds only
  * the held-out test split is scored once per configuration
  * single-feature "AUC" is the model-free univariate AUC (oriented by TRAIN sign)

Phased so each phase fits the runtime budget. Usage:
  python ablate.py sanity | single | cs131 | family | perm | strat | all
All phases append to artifacts/ablations/ablations_summary.json and write their
own JSON/CSV trace.
"""
from __future__ import annotations
import os, sys, json, csv, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

PHASE = sys.argv[1] if len(sys.argv) > 1 else "all"
HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(HERE, "artifacts", "s2_matrix.npz")
OUT = os.path.join(HERE, "artifacts", "ablations")
os.makedirs(OUT, exist_ok=True)

z = np.load(MATRIX, allow_pickle=True)
X = z["X"].astype(float); y = z["y"].astype(int)
groups = z["groups"]; split = z["split"]
names = [str(c) for c in z["feature_names"]]
NIDX = {n: i for i, n in enumerate(names)}
tr = split == "train"; te = split == "test"
Xtr, ytr, gtr = X[tr], y[tr], groups[tr]
Xte, yte, gte = X[te], y[te], groups[te]

NONHOG = [n for n in names if not n.startswith("hog_")]
CV131 = [n for n in names if n.startswith("cv131_")]
STAT = [n for n in names if n.startswith("stat_")]
PLX = [n for n in names if n.startswith("plx_")]

LR = lambda: LogisticRegression(max_iter=2000, class_weight="balanced")
RF = lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                    n_jobs=-1, random_state=0)
SUMPATH = os.path.join(OUT, "ablations_summary.json")


def cols(prefixes):
    return np.array([i for i, n in enumerate(names)
                     if any(n.startswith(p) for p in prefixes)])

def colset(feats):
    return np.array([NIDX[f] for f in feats])

def per_proj_auc(yv, p, gv):
    a = [roc_auc_score(yv[gv == k], p[gv == k]) for k in np.unique(gv)
         if len(np.unique(yv[gv == k])) == 2]
    return float(np.mean(a)) if a else float("nan")

def univariate_auc(col_idx):
    f_tr = Xtr[:, col_idx]; f_te = Xte[:, col_idx]
    sign = 1.0 if roc_auc_score(ytr, f_tr) >= 0.5 else -1.0
    return dict(pooled_train=float(roc_auc_score(ytr, sign * f_tr)),
                perproj_train=per_proj_auc(ytr, sign * f_tr, gtr),
                pooled_test=float(roc_auc_score(yte, sign * f_te)),
                perproj_test=per_proj_auc(yte, sign * f_te, gte), sign=sign)

def fit_eval(feat_cols, make, label="", do_cv=True):
    sc = StandardScaler().fit(Xtr[:, feat_cols])
    m = make().fit(sc.transform(Xtr[:, feat_cols]), ytr)
    sco = lambda M, Xs: (M.predict_proba(Xs)[:, 1] if hasattr(M, "predict_proba")
                         else M.decision_function(Xs))
    s_te = sco(m, sc.transform(Xte[:, feat_cols]))
    out = dict(label=label, n_features=int(len(feat_cols)),
               test_pooled=float(roc_auc_score(yte, s_te)),
               test_perproj=per_proj_auc(yte, s_te, gte),
               test_ap=float(average_precision_score(yte, s_te)))
    if do_cv:
        gkf = GroupKFold(n_splits=5); oof = np.full(len(ytr), np.nan)
        for a, b in gkf.split(Xtr, ytr, gtr):
            scf = StandardScaler().fit(Xtr[a][:, feat_cols])
            mf = make().fit(scf.transform(Xtr[a][:, feat_cols]), ytr[a])
            oof[b] = sco(mf, scf.transform(Xtr[b][:, feat_cols]))
        out["cv_pooled"] = float(roc_auc_score(ytr, oof))
        out["cv_perproj"] = per_proj_auc(ytr, oof, gtr)
    return out

def load_sum():
    return json.load(open(SUMPATH)) if os.path.exists(SUMPATH) else {}
def save_sum(s):
    json.dump(s, open(SUMPATH, "w"), indent=1)


def ph_sanity():
    print(f"TRAIN {len(ytr)} / {len(set(gtr))} proj | TEST {len(yte)} / {len(set(gte))} proj")
    print("=== 0. SANITY reproduce final_test families (LR) ===")
    s = load_sum(); out = {}
    for key, pref, lab in [("cv131", ["cv131_"], "CS131 primitives only"),
                           ("static", ["stat_", "cv131_"], "Static (stat+cv131)"),
                           ("plx", ["plx_"], "Parallax only")]:
        r = fit_eval(cols(pref), LR, lab, do_cv=False); out[key] = r
        print(f"  {lab:24s} test pooled {r['test_pooled']:.3f} per-proj {r['test_perproj']:.3f} AP {r['test_ap']:.3f}")
    import joblib
    best = joblib.load(os.path.join(HERE, "artifacts", "best_s2_model.joblib"))
    bc = colset([str(c) for c in best["cols"]])
    sb = best["model"].predict_proba(best["scaler"].transform(Xte[:, bc]))[:, 1]
    out["headline_frozen"] = dict(test_pooled=float(roc_auc_score(yte, sb)),
        test_perproj=per_proj_auc(yte, sb, gte), test_ap=float(average_precision_score(yte, sb)))
    print(f"  {'HEADLINE frozen RF':24s} test pooled {out['headline_frozen']['test_pooled']:.3f} "
          f"per-proj {out['headline_frozen']['test_perproj']:.3f} AP {out['headline_frozen']['test_ap']:.3f}")
    s["sanity"] = out; save_sum(s)


def ph_single():
    print("=== 1. Single-feature univariate AUC (non-HOG), ranked ===")
    s = load_sum(); rows = []
    for n in NONHOG:
        rows.append(dict(feature=n, family=n.split("_")[0], **univariate_auc(NIDX[n])))
    rows.sort(key=lambda r: -r["perproj_test"])
    with open(os.path.join(OUT, "single_feature_auc.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "family", "pooled_train",
            "perproj_train", "pooled_test", "perproj_test", "sign"]); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    for r in rows[:12]:
        print(f"  {r['feature']:24s} testPP {r['perproj_test']:.3f} testPool {r['pooled_test']:.3f}")
    s["single_feature_auc"] = rows; save_sum(s)


def ph_cs131():
    print("=== 2. CS131 primitive effect ===")
    s = load_sum(); eff = {}
    eff["solo_univariate"] = {n: univariate_auc(NIDX[n]) for n in CV131}
    for n in CV131:
        u = eff["solo_univariate"][n]
        print(f"  solo {n:20s} testPP {u['perproj_test']:.3f} testPool {u['pooled_test']:.3f}")
    eff["solo_logistic"] = {n: fit_eval(colset([n]), LR, n, do_cv=True) for n in CV131}
    full = fit_eval(colset(CV131), LR, "cv131 all", do_cv=True); eff["cv131_all"] = full
    print(f"  cv131 ALL: test per-proj {full['test_perproj']:.3f} CV per-proj {full['cv_perproj']:.3f}")
    loo = {}
    for n in CV131:
        r = fit_eval(colset([c for c in CV131 if c != n]), LR, f"cv131-{n}", do_cv=True)
        loo[n] = dict(test_perproj=r["test_perproj"], cv_perproj=r["cv_perproj"],
                      drop_vs_all_test_pp=full["test_perproj"] - r["test_perproj"])
        print(f"  cv131 minus {n:16s} test pp {r['test_perproj']:.3f} (drop {loo[n]['drop_vs_all_test_pp']:+.3f})")
    eff["leave_one_out"] = loo
    remaining = list(CV131); chosen = []; greedy = []
    while remaining:
        bn, br = None, None
        for n in remaining:
            r = fit_eval(colset(chosen + [n]), LR, "", do_cv=True)
            if br is None or r["cv_perproj"] > br["cv_perproj"]:
                bn, br = n, r
        chosen.append(bn); remaining.remove(bn)
        greedy.append(dict(added=bn, set=list(chosen), cv_perproj=br["cv_perproj"],
                           test_perproj=br["test_perproj"]))
        print(f"  greedy + {bn:18s} CV pp {br['cv_perproj']:.3f} test pp {br['test_perproj']:.3f}")
    eff["greedy_add"] = greedy
    bs = fit_eval(cols(["stat_"]), LR, "", do_cv=False)
    spc = fit_eval(cols(["stat_", "cv131_"]), LR, "", do_cv=False)
    po = fit_eval(cols(["plx_"]), LR, "", do_cv=False)
    ppc = fit_eval(cols(["plx_", "cv131_"]), LR, "", do_cv=False)
    eff["addon_lift"] = dict(stat_only=bs["test_perproj"], stat_plus_cv131=spc["test_perproj"],
        lift_on_stat=spc["test_perproj"] - bs["test_perproj"], plx_only=po["test_perproj"],
        plx_plus_cv131=ppc["test_perproj"], lift_on_plx=ppc["test_perproj"] - po["test_perproj"])
    print(f"  cv131 add-on lift: on stat {eff['addon_lift']['lift_on_stat']:+.3f} | on plx {eff['addon_lift']['lift_on_plx']:+.3f}")
    json.dump(eff, open(os.path.join(OUT, "cs131_primitive_effect.json"), "w"), indent=1)
    s["cs131_effect"] = eff; save_sum(s)


def ph_family():
    print("=== 3. Family ablation (LR + RF) ===")
    s = load_sum()
    fams = {"stat": ["stat_"], "cv131": ["cv131_"], "plx": ["plx_"], "hog": ["hog_"]}
    fam = {"alone": {}, "leave_one_out": {}}
    for f, pref in fams.items():
        rL = fit_eval(cols(pref), LR, f, do_cv=True)
        rR = fit_eval(cols(pref), RF, f, do_cv=False)
        fam["alone"][f] = dict(lr=rL, rf=rR)
        print(f"  alone {f:6s} LR testPP {rL['test_perproj']:.3f} CV {rL['cv_perproj']:.3f} | RF testPP {rR['test_perproj']:.3f}")
    allp = ["stat_", "cv131_", "plx_", "hog_"]
    full = fit_eval(cols(allp), RF, "ALL", do_cv=False); fam["all_rf"] = full
    print(f"  ALL (RF) testPP {full['test_perproj']:.3f} pooled {full['test_pooled']:.3f}")
    for f, pref in fams.items():
        r = fit_eval(cols([p for p in allp if p not in pref]), RF, f"ALL-{f}", do_cv=False)
        fam["leave_one_out"][f] = dict(test_perproj=r["test_perproj"],
            drop_vs_all=full["test_perproj"] - r["test_perproj"])
        print(f"  ALL minus {f:6s} (RF) testPP {r['test_perproj']:.3f} (drop {fam['leave_one_out'][f]['drop_vs_all']:+.3f})")
    json.dump(fam, open(os.path.join(OUT, "family_ablation.json"), "w"), indent=1)
    s["family_ablation"] = fam; save_sum(s)


def ph_perm():
    print("=== 4. Permutation importance (RF plx+static, test AUC drop) ===")
    s = load_sum(); ps = STAT + CV131 + PLX; pc = colset(ps)
    sc = StandardScaler().fit(Xtr[:, pc]); rf = RF().fit(sc.transform(Xtr[:, pc]), ytr)
    Xs = sc.transform(Xte[:, pc]); base = roc_auc_score(yte, rf.predict_proba(Xs)[:, 1])
    rng = np.random.default_rng(0); perm = []
    for j, n in enumerate(ps):
        dr = []
        for _ in range(5):
            Xp = Xs.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            dr.append(base - roc_auc_score(yte, rf.predict_proba(Xp)[:, 1]))
        perm.append(dict(feature=n, mean_auc_drop=float(np.mean(dr)), sd=float(np.std(dr))))
    perm.sort(key=lambda r: -r["mean_auc_drop"])
    print(f"  base test pooled AUC {base:.4f}")
    for r in perm[:10]:
        print(f"    {r['feature']:24s} drop {r['mean_auc_drop']:+.4f}")
    json.dump(dict(base_test_auc=float(base), importance=perm),
              open(os.path.join(OUT, "perm_importance.json"), "w"), indent=1)
    s["perm_importance"] = dict(base_test_auc=float(base), top=perm[:12]); save_sum(s)


def ph_strat():
    print("=== 5. Salience-stratified test ===")
    s = load_sum(); zi = NIDX["stat_zabs"]
    sal = {k: Xte[(gte == k) & (yte == 1), zi].mean() for k in np.unique(gte)}
    med = float(np.median(list(sal.values()))); strat = {"median": med, "low": {}, "high": {}}
    for lab, sel in (("low", lambda k: sal[k] <= med), ("high", lambda k: sal[k] > med)):
        ks = [k for k in np.unique(gte) if sel(k)]; mask = np.isin(gte, ks)
        for f, pref in [("plx", ["plx_"]), ("static", ["stat_", "cv131_"]),
                        ("cv131", ["cv131_"]), ("hog", ["hog_"])]:
            c = cols(pref); sc = StandardScaler().fit(Xtr[:, c])
            m = LR().fit(sc.transform(Xtr[:, c]), ytr)
            p = m.predict_proba(sc.transform(Xte[mask][:, c]))[:, 1]
            strat[lab][f] = per_proj_auc(yte[mask], p, gte[mask])
        strat[lab]["n_proj"] = len(ks)
        print(f"  {lab:4s} ({len(ks)}): plx {strat[lab]['plx']:.3f} static {strat[lab]['static']:.3f} "
              f"cv131 {strat[lab]['cv131']:.3f} hog {strat[lab]['hog']:.3f}")
    json.dump(strat, open(os.path.join(OUT, "salience_stratified.json"), "w"), indent=1)
    s["salience_stratified"] = strat; save_sum(s)


PHASES = {"sanity": ph_sanity, "single": ph_single, "cs131": ph_cs131,
          "family": ph_family, "perm": ph_perm, "strat": ph_strat}
if PHASE == "all":
    for fn in PHASES.values():
        fn()
elif PHASE in PHASES:
    PHASES[PHASE]()
else:
    print("phases:", list(PHASES))
print("done phase:", PHASE)
