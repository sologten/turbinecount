"""Rigor/verification audit for the band-parallax wind-turbine claim.

Central claim under test:
  Band-parallax/motion features (plx_) detect turbines EVEN WHERE static
  appearance-salience is low -- i.e., in the bright/textured-terrain regime
  that defeats appearance-based detectors. If parallax only wins where static
  already wins, the contribution is not supported.

ALL analysis uses split=='train' only, via harness.train_view.
ALL CV is GroupKFold by project, scaling fit on train folds only (harness.cv_oof).
"""
from __future__ import annotations
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

import harness as H

RNG = np.random.RandomState(131)
OUT = "/sessions/upbeat-wizardly-ramanujan/work/artifacts/mechanism_audit.json"


def per_project_auc(y, p, groups, keep=None):
    """Dict project->AUC for projects with both classes (optionally restricted)."""
    out = {}
    projs = np.unique(groups) if keep is None else keep
    for gp in projs:
        m = groups == gp
        if m.sum() == 0:
            continue
        if len(np.unique(y[m])) == 2:
            out[str(gp)] = float(roc_auc_score(y[m], p[m]))
    return out


def main():
    d = H.load()
    X, y, groups = H.train_view(d)
    names = d["names"]

    plx_idx = H.cols(names, ["plx_"])
    static_idx = H.cols(names, ["stat_", "cv131_"])
    hog_idx = H.cols(names, ["hog_"])
    stat_idx = H.cols(names, ["stat_"])
    zabs_col = names.index("stat_zabs")
    bright_col = names.index("stat_center_bright")
    contrast_col = names.index("stat_contrast")

    results = {}

    # ---- Project salience = mean stat_zabs over POSITIVE patches ----
    proj_list = np.unique(groups)
    salience = {}
    for gp in proj_list:
        m = (groups == gp) & (y == 1)
        if m.sum() > 0:
            salience[gp] = float(np.mean(X[m, zabs_col]))
    sal_projs = np.array([gp for gp in proj_list if gp in salience])
    sal_vals = np.array([salience[gp] for gp in sal_projs])
    med = float(np.median(sal_vals))
    low_projs = set(sal_projs[sal_vals <= med])
    high_projs = set(sal_projs[sal_vals > med])
    results["n_train_projects"] = int(len(proj_list))
    results["n_projects_with_positives"] = int(len(sal_projs))
    results["salience_median"] = med
    results["n_low_salience_projects"] = int(len(low_projs))
    results["n_high_salience_projects"] = int(len(high_projs))

    def mk_logreg():
        return LogisticRegression(max_iter=2000, C=1.0)

    # ---- TASK A: parallax-only vs static-only OOF, split by salience regime ----
    oof_plx, fa_plx = H.cv_oof(X[:, plx_idx], y, groups, mk_logreg, n_splits=5, scale=True)
    oof_stat, fa_stat = H.cv_oof(X[:, static_idx], y, groups, mk_logreg, n_splits=5, scale=True)

    pooled_plx, mpp_plx, _ = H.grouped_auc(y, oof_plx, groups)
    pooled_stat, mpp_stat, _ = H.grouped_auc(y, oof_stat, groups)

    ppa_plx = per_project_auc(y, oof_plx, groups)
    ppa_stat = per_project_auc(y, oof_stat, groups)

    def mean_in(ppa, projset):
        vals = [ppa[str(gp)] for gp in projset if str(gp) in ppa]
        return float(np.mean(vals)) if vals else float("nan"), len(vals)

    plx_low, n_low_p = mean_in(ppa_plx, low_projs)
    plx_high, n_high_p = mean_in(ppa_plx, high_projs)
    stat_low, _ = mean_in(ppa_stat, low_projs)
    stat_high, _ = mean_in(ppa_stat, high_projs)

    results["taskA"] = {
        "parallax_only": {"pooled_auc": pooled_plx, "mean_per_proj_auc": mpp_plx,
                           "cv_fold_auc": fa_plx.tolist()},
        "static_only": {"pooled_auc": pooled_stat, "mean_per_proj_auc": mpp_stat,
                        "cv_fold_auc": fa_stat.tolist()},
        "mean_per_proj_auc_LOW_salience": {"parallax": plx_low, "static": stat_low,
                                           "n_proj": n_low_p},
        "mean_per_proj_auc_HIGH_salience": {"parallax": plx_high, "static": stat_high,
                                            "n_proj": n_high_p},
    }

    # correlation of per-project AUC vs project salience
    common = [gp for gp in sal_projs if str(gp) in ppa_plx and str(gp) in ppa_stat]
    cs = np.array([salience[gp] for gp in common])
    cp = np.array([ppa_plx[str(gp)] for gp in common])
    ct = np.array([ppa_stat[str(gp)] for gp in common])
    def corr(a, b):
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])
    results["taskA"]["corr_salience_vs_auc"] = {
        "parallax": corr(cs, cp), "static": corr(cs, ct),
        "n_projects_in_corr": len(common),
    }
    results["taskA"]["scatter"] = {
        "salience": cs.tolist(), "parallax_auc": cp.tolist(),
        "static_auc": ct.tolist(), "projects": [str(g) for g in common],
    }

    # ---- TASK B: HOG + linear SVM literature baseline ----
    def mk_svm():
        return LinearSVC(C=1.0, max_iter=5000)
    oof_hog, fa_hog = H.cv_oof(X[:, hog_idx], y, groups, mk_svm, n_splits=5, scale=True)
    pooled_hog, mpp_hog, _ = H.grouped_auc(y, oof_hog, groups)
    ppa_hog = per_project_auc(y, oof_hog, groups)
    hog_low, _ = mean_in(ppa_hog, low_projs)
    hog_high, _ = mean_in(ppa_hog, high_projs)
    results["taskB"] = {
        "hog_svm": {"pooled_auc": pooled_hog, "mean_per_proj_auc": mpp_hog,
                    "cv_fold_auc": fa_hog.tolist(),
                    "mean_per_proj_auc_LOW_salience": hog_low,
                    "mean_per_proj_auc_HIGH_salience": hog_high},
    }

    # ---- TASK C: permutation importance for combined plx_+stat_ model ----
    comb_idx = np.concatenate([plx_idx, stat_idx])
    comb_names = [names[i] for i in comb_idx]
    Xc = X[:, comb_idx]
    gkf = GroupKFold(n_splits=5)
    base_aucs, imp_drops = [], np.zeros((5, len(comb_idx)))
    for fi, (tr, va) in enumerate(gkf.split(Xc, y, groups)):
        sc = StandardScaler().fit(Xc[tr])
        Xtr, Xva = sc.transform(Xc[tr]), sc.transform(Xc[va])
        rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                    min_samples_leaf=5, n_jobs=-1, random_state=131)
        rf.fit(Xtr, y[tr])
        base = roc_auc_score(y[va], rf.predict_proba(Xva)[:, 1])
        base_aucs.append(base)
        for j in range(Xva.shape[1]):
            Xp = Xva.copy()
            Xp[:, j] = RNG.permutation(Xp[:, j])
            a = roc_auc_score(y[va], rf.predict_proba(Xp)[:, 1])
            imp_drops[fi, j] = base - a
    mean_drop = imp_drops.mean(axis=0)
    order = np.argsort(mean_drop)[::-1]
    top = [{"feature": comb_names[k], "mean_auc_drop": float(mean_drop[k])}
           for k in order[:12]]
    results["taskC"] = {
        "model": "RandomForest on plx_+stat_, GroupKFold permutation importance",
        "base_cv_auc": float(np.mean(base_aucs)),
        "top_features": top,
        "top_plx_features": [t for t in top if t["feature"].startswith("plx_")][:10],
    }

    # ---- TASK D1: label-shuffle control ----
    yshuf = y.copy()
    for gp in proj_list:  # shuffle within train (global shuffle); also break group structure
        pass
    yshuf = RNG.permutation(y)
    Xall = X[:, np.concatenate([plx_idx, static_idx])]
    oof_sh, fa_sh = H.cv_oof(Xall, yshuf, groups, mk_logreg, n_splits=5, scale=True)
    pooled_sh, mpp_sh, _ = H.grouped_auc(yshuf, oof_sh, groups)
    results["taskD"] = {"label_shuffle": {
        "pooled_auc": pooled_sh, "mean_per_proj_auc": mpp_sh,
        "cv_fold_auc": fa_sh.tolist(),
        "note": "should be ~0.5 if no leakage"}}

    # ---- TASK D2: brightness-only & confound single-feature AUCs (CV OOF) ----
    def single_feat_cv(col):
        oof, _ = H.cv_oof(X[:, [col]], y, groups, mk_logreg, n_splits=5, scale=True)
        pooled, mpp, _ = H.grouped_auc(y, oof, groups)
        # orient: take max(auc, 1-auc) at pooled to report separability magnitude
        return {"pooled_auc": pooled, "mean_per_proj_auc": mpp,
                "pooled_auc_oriented": float(max(pooled, 1 - pooled))}
    results["taskD"]["brightness_only"] = single_feat_cv(bright_col)
    results["taskD"]["contrast_only_edge_proxy"] = single_feat_cv(contrast_col)
    results["taskD"]["zabs_salience_only"] = single_feat_cv(zabs_col)
    # HOG total energy as a texture/edge proxy
    hog_energy = X[:, hog_idx].sum(axis=1, keepdims=True)
    oof_he, _ = H.cv_oof(hog_energy, y, groups, mk_logreg, n_splits=5, scale=True)
    p_he, m_he, _ = H.grouped_auc(y, oof_he, groups)
    results["taskD"]["hog_energy_texture_proxy"] = {
        "pooled_auc": p_he, "mean_per_proj_auc": m_he,
        "pooled_auc_oriented": float(max(p_he, 1 - p_he))}

    # ---- VERDICT logic ----
    delta_low = plx_low - stat_low
    delta_high = plx_high - stat_high
    if plx_low > stat_low + 0.02 and plx_low >= 0.6:
        verdict = "SUPPORTED: parallax beats static on low-salience terrain"
    elif plx_low > stat_low + 0.02:
        verdict = "PARTIAL: parallax edges static on low-salience but absolute AUC modest"
    elif abs(delta_low) <= 0.02:
        verdict = "NOT SUPPORTED: parallax ~ static on low-salience terrain"
    else:
        verdict = "REFUTED: static beats parallax even on low-salience terrain"
    results["VERDICT"] = {
        "text": verdict,
        "parallax_minus_static_LOW": float(delta_low),
        "parallax_minus_static_HIGH": float(delta_high),
    }

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
