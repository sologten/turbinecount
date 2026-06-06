"""Diverse, leakage-free classifier sweep for S2 wind-turbine detection.

Rigor guarantees (do not violate):
  * Only split=='train' rows are used (via harness.train_view).
  * GroupKFold(5) by project (via harness.cv_oof). No farm in train & val.
  * StandardScaler fit inside each fold on train rows only (scale=True).
  * Test rows are NEVER loaded or scored anywhere in this script.

For each classifier we run a small hyperparameter grid, scoring each config by
per-project CV AUC (grouped_auc mean). We do this on 4 feature subsets:
  ['plx_'], ['stat_','cv131_'], ['plx_','stat_','cv131_'], ALL.
We then refit the single best (classifier, subset, hyperparams) by per-project
CV AUC on all train rows with a StandardScaler, and persist scaler+model+cols.
"""
from __future__ import annotations
import json
import os
import sys
import itertools
import numpy as np
import joblib

import harness as H
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ART = "/sessions/upbeat-wizardly-ramanujan/work/artifacts"
SWEEP_JSON = f"{ART}/sweep_s2.json"
BEST_MODEL = f"{ART}/best_s2_model.joblib"

SUBSETS = {
    "plx": ["plx_"],
    "static": ["stat_", "cv131_"],
    "plx+static": ["plx_", "stat_", "cv131_"],
    "ALL": ["stat_", "cv131_", "plx_", "hog_"],
}


def grid_dicts(grid):
    """Expand a dict of {param: [values]} into a list of config dicts."""
    keys = list(grid.keys())
    out = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        out.append(dict(zip(keys, combo)))
    return out


# Each entry: name -> (factory(**cfg) -> model, grid_dict)
def lr_factory(C):
    return lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=C)


def svc_factory(C, gamma):
    # probability via decision_function path in harness (no predict_proba).
    return lambda: SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced")


def rf_factory(n_estimators, max_depth):
    return lambda: RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        class_weight="balanced", n_jobs=-1, random_state=0)


def xgb_factory(n_estimators, max_depth, learning_rate):
    return lambda: XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, scale_pos_weight=1,
        subsample=0.9, colsample_bytree=0.9,
        tree_method="hist", eval_metric="logloss",
        n_jobs=-1, random_state=0)


def mlp_factory(hidden, alpha):
    return lambda: MLPClassifier(
        hidden_layer_sizes=(hidden,), alpha=alpha,
        max_iter=400, early_stopping=True, random_state=0)


CLASSIFIERS = {
    "LogisticRegression": (lr_factory, {"C": [0.1, 1, 10]}),
    "SVC_rbf": (svc_factory, {"C": [1, 10], "gamma": ["scale", 0.01]}),
    "RandomForest": (rf_factory, {"n_estimators": [200, 400], "max_depth": [None, 16]}),
    "XGBoost": (xgb_factory, {"n_estimators": [200, 400], "max_depth": [4, 6], "learning_rate": [0.05, 0.1]}),
    "MLP": (mlp_factory, {"hidden": [64, 128], "alpha": [1e-4, 1e-2]}),
}


PARTIAL = f"{ART}/sweep_partial.json"


def run_one(clf_name, only_subset=None, cfg_slice=None):
    """CV-sweep a single classifier over subsets; append rows to PARTIAL.

    If only_subset is given, restrict to that single subset name (used to keep
    heavy classifiers like SVC-RBF inside the per-call compute budget).
    If cfg_slice='a:b' is given, only grid configs [a:b] are evaluated this call;
    the per-cell best is merged with any prior best already stored for that cell
    (so a heavy cell can be split across calls without losing the running best).
    """
    factory, grid = CLASSIFIERS[clf_name]
    d = H.load()
    X, y, g = H.train_view(d)
    names = d["names"]

    all_cfgs = grid_dicts(grid)
    if cfg_slice:
        a, b = cfg_slice.split(":")
        cfgs = all_cfgs[int(a):int(b)]
    else:
        cfgs = all_cfgs

    # load prior bests for merge (so split cells keep the running best)
    prior = {}
    if os.path.exists(PARTIAL):
        with open(PARTIAL) as f:
            for r in json.load(f):
                prior[(r["classifier"], r["subset"])] = r

    subsets = SUBSETS if only_subset is None else {only_subset: SUBSETS[only_subset]}
    rows = []
    for sub_name, prefixes in subsets.items():
        idx = H.cols(names, prefixes)
        Xs = X[:, idx]
        best_cfg = None
        best_metrics = None
        # seed with any previously-stored best for this cell
        p = prior.get((clf_name, sub_name))
        if p is not None:
            best_cfg = p["best_hyperparams"]
            best_metrics = {
                "pooled_auc": p["pooled_auc"],
                "perproject_auc": p["perproject_auc"],
                "fold_mean": p["fold_mean"],
                "fold_sd": p["fold_sd"],
                "n_proj": p["n_proj"],
            }
        for cfg in cfgs:
            make = factory(**cfg)
            oof, fa = H.cv_oof(Xs, y, g, make, n_splits=5, scale=True)
            pooled, perproj, npj = H.grouped_auc(y, oof, g)
            metrics = {
                "pooled_auc": float(pooled),
                "perproject_auc": float(perproj),
                "fold_mean": float(fa.mean()),
                "fold_sd": float(fa.std()),
                "n_proj": int(npj),
            }
            if best_metrics is None or perproj > best_metrics["perproject_auc"]:
                best_metrics = metrics
                best_cfg = cfg
        row = {
            "classifier": clf_name,
            "subset": sub_name,
            "subset_prefixes": prefixes,
            "n_features": int(len(idx)),
            "grid": grid,
            "best_hyperparams": best_cfg,
            **best_metrics,
        }
        rows.append(row)
        print("%-18s %-12s pooled=%.4f perproj=%.4f fold=%.4f+/-%.4f best=%s" % (
            clf_name, sub_name, row["pooled_auc"], row["perproject_auc"],
            row["fold_mean"], row["fold_sd"], best_cfg), flush=True)

    done_subs = {r["subset"] for r in rows}
    existing = []
    if os.path.exists(PARTIAL):
        with open(PARTIAL) as f:
            existing = json.load(f)
    # drop only the (classifier, subset) cells we just recomputed
    existing = [r for r in existing
                if not (r["classifier"] == clf_name and r["subset"] in done_subs)]
    existing.extend(rows)
    with open(PARTIAL, "w") as f:
        json.dump(existing, f, indent=2)
    print("Appended %d rows for %s -> %s" % (len(rows), clf_name, PARTIAL), flush=True)


def finalize():
    """Assemble full results table, pick best by per-project AUC, refit & save."""
    d = H.load()
    X, y, g = H.train_view(d)
    names = d["names"]

    with open(PARTIAL) as f:
        results = json.load(f)

    best_row = max(results, key=lambda r: r["perproject_auc"])

    out = {
        "n_train_rows": int(len(y)),
        "n_train_projects": int(len(np.unique(g))),
        "subsets": SUBSETS,
        "selection_metric": "per-project CV AUC (GroupKFold=5)",
        "results": results,
        "best": {
            "classifier": best_row["classifier"],
            "subset": best_row["subset"],
            "subset_prefixes": best_row["subset_prefixes"],
            "best_hyperparams": best_row["best_hyperparams"],
            "metrics": {
                "pooled_auc": best_row["pooled_auc"],
                "perproject_auc": best_row["perproject_auc"],
                "fold_mean": best_row["fold_mean"],
                "fold_sd": best_row["fold_sd"],
                "n_proj": best_row["n_proj"],
            },
        },
    }
    with open(SWEEP_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote", SWEEP_JSON, flush=True)

    # ---- refit best on ALL train rows, StandardScaler fit on all train rows ----
    factory, _ = CLASSIFIERS[best_row["classifier"]]
    prefixes = best_row["subset_prefixes"]
    idx = H.cols(names, prefixes)
    Xbest = X[:, idx]
    scaler = StandardScaler().fit(Xbest)
    Xb = scaler.transform(Xbest)
    model = factory(**best_row["best_hyperparams"])()
    model.fit(Xb, y)
    cols_used = [names[i] for i in idx]
    joblib.dump(
        {"scaler": scaler, "model": model, "cols": cols_used, "subset": prefixes},
        BEST_MODEL)
    print("Saved best model+scaler to", BEST_MODEL, flush=True)
    print("BEST: %s on %s  per-project CV AUC=%.4f  hyperparams=%s" % (
        best_row["classifier"], best_row["subset"],
        best_row["perproject_auc"], best_row["best_hyperparams"]), flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    only_sub = sys.argv[2] if len(sys.argv) > 2 else None
    cfg_slice = sys.argv[3] if len(sys.argv) > 3 else None
    if arg == "finalize":
        finalize()
    elif arg == "all":
        for name in CLASSIFIERS:
            run_one(name)
        finalize()
    else:
        run_one(arg, only_sub, cfg_slice)
