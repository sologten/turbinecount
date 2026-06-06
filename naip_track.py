"""NAIP high-resolution (~0.6 m GSD) wind-turbine benchmark track.

Featurizes cached 160x160x4 NAIP tiles (bands R,G,B,NIR), builds train/test
feature matrices under the frozen project split, and runs a GroupKFold(5)
cross-validated comparison of LogisticRegression / RandomForest / XGBoost over
feature subsets (CS131-static-only, static+HOG, ALL).

ABSOLUTE RIGOR
  * split=='test' projects are NEVER used for CV / tuning / any metric. The
    test feature matrix is built and saved for later use only.
  * CV groups = project slug (GroupKFold) so a project never spans folds.
  * StandardScaler is fit on training folds only (inside each CV pipeline).

Run:  python3 naip_track.py
"""
from __future__ import annotations

import os
import sys
import json
import time
import warnings

import numpy as np

# --- paths -------------------------------------------------------------------
WORK = "/sessions/upbeat-wizardly-ramanujan/work"
CACHE = f"{WORK}/cache/naip"
ARTIFACTS = f"{WORK}/artifacts"
CV131_PARENT = "/sessions/upbeat-wizardly-ramanujan/mnt/cs131/turbinecount"
CODE_DIR = "/sessions/upbeat-wizardly-ramanujan/mnt/cs131/turbinecount_v2"

sys.path.insert(0, CV131_PARENT)   # for `from cv131...`
sys.path.insert(0, CODE_DIR)       # for gt

import gt  # noqa: E402
from cv131.harris import harris_response  # noqa: E402
from cv131.sobel import sobel_gradients  # noqa: E402
from cv131.gaussian import gaussian_smooth  # noqa: E402
from cv131.template import matched_filter  # noqa: E402
from cv131.hough import _edge_mask, hough_lines, line_support_map  # noqa: E402

from skimage.feature import hog  # noqa: E402

warnings.filterwarnings("ignore")

# At ~0.6 m GSD a 160x160 tile is ~96 m across. A turbine hub/nacelle is a
# bright compact blob a few px wide; tower / blade / shadow are long thin
# lines tens of px long. These constants are scaled for that resolution.
NACELLE_RADIUS = 3      # ~1.8 m bright bump template radius
CENTER_HALF = 20        # center window half-size for local-contrast z-score
SHORT_LINE_MAX = 60.0   # px; tower/blade/shadow are long thin lines


# ============================================================================
# Featurizer
# ============================================================================
def _gray(tile: np.ndarray) -> np.ndarray:
    """Luminance from R,G,B bands of a (4,H,W) tile, scaled to [0,1]."""
    r, g, b = tile[0], tile[1], tile[2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return (lum / 255.0).astype(np.float64)


def _nacelle_template(radius: int) -> np.ndarray:
    """Small radially-bright Gaussian bump (zero-meaned inside matched_filter)."""
    n = 2 * radius + 1
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    rr = np.sqrt(xx * xx + yy * yy)
    t = np.exp(-(rr ** 2) / (2.0 * (radius / 1.5) ** 2))
    return t.astype(np.float64)


_TEMPLATE = _nacelle_template(NACELLE_RADIUS)

# Feature names (must match the order produced by featurize_tile).
STATIC_NAMES = [
    "stat_contrast_bright_z",   # bright-polarity center-vs-surround z-score
    "stat_contrast_dark_z",     # dark-polarity (shadow) center-vs-surround z-score
    "cv131_harris_max",         # max Harris corner response
    "cv131_matchfilt_max",      # max matched-filter response to nacelle bump
    "cv131_hough_support_max",  # max short-line Hough support (tower/blade/shadow)
    "cv131_hough_support_sum",  # total short-line support
    "stat_sobel_center_energy", # Sobel gradient energy in center window
    "stat_nir_red_ratio",       # mean NIR/Red ratio (vegetation/material cue)
]


def featurize_tile(tile: np.ndarray, n_hog: int):
    """Return (static_vec, hog_vec) for one (4,160,160) tile."""
    tile = np.asarray(tile, dtype=np.float64)
    gray = _gray(tile)
    H, W = gray.shape
    cy, cx = H // 2, W // 2

    # --- local-contrast z-score: center window vs. surround ---
    c0, c1 = cy - CENTER_HALF, cy + CENTER_HALF
    center = gray[c0:c1, c0:c1]
    surround_mean = gray.mean()
    surround_std = gray.std() + 1e-8
    # bright polarity: how far the brightest center pixel sits above surround
    bright_z = (center.max() - surround_mean) / surround_std
    # dark polarity (shadow): how far the darkest center pixel sits below
    dark_z = (surround_mean - center.min()) / surround_std

    # --- Harris (CS131): bright compact hub -> strong corner response ---
    R = harris_response(gaussian_smooth(gray, 1.0), sigma=1.5, k=0.04)
    harris_max = float(R.max())

    # --- matched filter (CS131): center-surround nacelle-bump response ---
    mf = matched_filter(gray, _TEMPLATE)
    matchfilt_max = float(mf.max())

    # --- Hough short-line support (CS131): tower/blade/shadow lines ---
    emask = _edge_mask(gray, 92.0)
    hl = hough_lines(gray, edge_mask=emask, n_lines=30, min_votes=5)
    if hl.lines.size:
        supp = line_support_map(gray.shape, emask, hl.lines,
                                dist_tol=1.5, seg_max_len=SHORT_LINE_MAX)
        hough_support_max = float(supp.max())
        hough_support_sum = float(supp.sum())
    else:
        hough_support_max = 0.0
        hough_support_sum = 0.0

    # --- Sobel gradient energy in center window ---
    g = sobel_gradients(gray)
    sobel_center_energy = float((g.magnitude[c0:c1, c0:c1] ** 2).mean())

    # --- NIR/Red ratio (robust: guard near-zero Red, clip outliers) ---
    red = tile[0]
    nir = tile[3]
    ratio = nir / (red + 1.0)          # +1 DN floor (bands are 0..255 DN)
    nir_red_ratio = float(np.clip(ratio, 0.0, 10.0).mean())

    static = np.array([
        bright_z, dark_z, harris_max, matchfilt_max,
        hough_support_max, hough_support_sum,
        sobel_center_energy, nir_red_ratio,
    ], dtype=np.float64)

    # --- compact HOG descriptor on the grayscale tile ---
    hog_vec = hog(
        gray, orientations=8, pixels_per_cell=(20, 20),
        cells_per_block=(2, 2), block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float64)
    assert hog_vec.size == n_hog, (hog_vec.size, n_hog)
    return static, hog_vec


def _hog_length() -> int:
    dummy = np.zeros((160, 160), dtype=np.float64)
    v = hog(dummy, orientations=8, pixels_per_cell=(20, 20),
            cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
    return int(v.size)


# ============================================================================
# Matrix building
# ============================================================================
PROGRESS_LOG = f"{ARTIFACTS}/naip_track_progress.log"


def _log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(line + "\n")


def _cache_files() -> set:
    return {f[:-4] for f in os.listdir(CACHE)
            if f.endswith(".npz")
            and not any(k in f for k in ("_pos_", "_neg_", "features"))}


def build_matrix(split: str):
    """Featurize every tile of every project on `split`. Returns
    (X, y, groups, feature_names)."""
    files = _cache_files()
    n_hog = _hog_length()
    hog_names = [f"hog_{i:03d}" for i in range(n_hog)]
    feature_names = STATIC_NAMES + hog_names

    Xs, ys, groups = [], [], []
    n_proj = 0
    n_tiles = 0
    for d in gt.panel():
        if d["split"] != split:
            continue
        slug = gt.slug(d["name"], d["state"])
        if slug not in files:
            continue
        z = np.load(f"{CACHE}/{slug}.npz", allow_pickle=True)
        X = z["X"]
        y = z["y"].astype(int)
        n_proj += 1
        _log(f"[{split}] featurizing {slug} ({len(y)} tiles)")
        for i in range(len(y)):
            stat, hg = featurize_tile(X[i], n_hog)
            Xs.append(np.concatenate([stat, hg]))
            ys.append(int(y[i]))
            groups.append(slug)
            n_tiles += 1
    X = np.asarray(Xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    groups = np.asarray(groups)
    print(f"[{split}] projects={n_proj} tiles={n_tiles} "
          f"pos={int(y.sum())} neg={int((y == 0).sum())} "
          f"X.shape={X.shape}")
    return X, y, groups, feature_names


# ============================================================================
# CV comparison
# ============================================================================
def run_cv():
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    import joblib
    try:
        from xgboost import XGBClassifier
        have_xgb = True
    except Exception as e:  # pragma: no cover
        print("xgboost unavailable:", e)
        have_xgb = False

    # ---- load / build train matrix (reuse cache to allow fast re-runs) ----
    train_path = f"{ARTIFACTS}/naip_train_matrix.npz"
    if os.path.exists(train_path):
        z = np.load(train_path, allow_pickle=True)
        Xtr, ytr, gtr = z["X"], z["y"], z["groups"]
        names = [str(s) for s in z["feature_names"]]
        _log(f"loaded cached train matrix {train_path} X={Xtr.shape}")
    else:
        Xtr, ytr, gtr, names = build_matrix("train")
        np.savez_compressed(train_path, X=Xtr, y=ytr, groups=gtr,
                            feature_names=np.array(names))
        _log("saved " + train_path)

    # ---- build + save test matrix (NEVER read for metrics) ----
    test_path = f"{ARTIFACTS}/naip_test_matrix.npz"
    if os.path.exists(test_path):
        zt = np.load(test_path, allow_pickle=True)
        Xte, yte, gte = zt["X"], zt["y"], zt["groups"]
        names_te = [str(s) for s in zt["feature_names"]]
        _log(f"loaded cached test matrix {test_path} (NOT used for metrics)")
    else:
        Xte, yte, gte, names_te = build_matrix("test")
        np.savez_compressed(test_path, X=Xte, y=yte, groups=gte,
                            feature_names=np.array(names))
        _log("saved " + test_path + " (stored only; excluded from all metrics)")
    assert names_te == names

    names = list(names)
    static_cols = [i for i, n in enumerate(names) if not n.startswith("hog_")]
    hog_cols = [i for i, n in enumerate(names) if n.startswith("hog_")]
    all_cols = list(range(len(names)))
    subsets = {
        "static_only": static_cols,
        "static+HOG": static_cols + hog_cols,
        "ALL": all_cols,
    }

    # ---- model grids ----
    def model_specs():
        specs = {
            "LogReg": [
                ("C=0.1", LogisticRegression(C=0.1, max_iter=2000,
                                             class_weight="balanced")),
                ("C=1", LogisticRegression(C=1.0, max_iter=2000,
                                           class_weight="balanced")),
                ("C=10", LogisticRegression(C=10.0, max_iter=2000,
                                            class_weight="balanced")),
            ],
            "RandomForest": [
                ("n200_d6", RandomForestClassifier(
                    n_estimators=200, max_depth=6, random_state=0,
                    class_weight="balanced", n_jobs=1)),
                ("n300_d10", RandomForestClassifier(
                    n_estimators=300, max_depth=10, random_state=0,
                    class_weight="balanced", n_jobs=1)),
            ],
        }
        if have_xgb:
            pos = int(ytr.sum())
            neg = int((ytr == 0).sum())
            spw = max(1.0, neg / max(1, pos))
            specs["XGBoost"] = [
                ("d3_lr0.1", XGBClassifier(
                    n_estimators=200, max_depth=3, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=spw, eval_metric="auc",
                    tree_method="hist",
                    n_jobs=1, random_state=0, verbosity=0)),
                ("d5_lr0.05", XGBClassifier(
                    n_estimators=300, max_depth=5, learning_rate=0.05,
                    subsample=0.7, colsample_bytree=0.7,
                    scale_pos_weight=spw, eval_metric="auc",
                    tree_method="hist",
                    n_jobs=1, random_state=0, verbosity=0)),
            ]
        return specs

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(Xtr, ytr, groups=gtr))

    def cv_oof(cols, est):
        """Out-of-fold predictions; scaler fit on train folds only."""
        oof = np.full(len(ytr), np.nan)
        for tr_idx, va_idx in folds:
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("clf", est)])
            pipe.fit(Xtr[np.ix_(tr_idx, cols)], ytr[tr_idx])
            oof[va_idx] = pipe.predict_proba(Xtr[np.ix_(va_idx, cols)])[:, 1]
        return oof

    def pooled_auc(oof):
        return roc_auc_score(ytr, oof)

    def per_project_auc(oof):
        """Mean per-project AUC over projects that have both classes."""
        accs = {}
        for slug in np.unique(gtr):
            m = gtr == slug
            yy = ytr[m]
            if len(np.unique(yy)) < 2:
                continue
            accs[slug] = roc_auc_score(yy, oof[m])
        mean = float(np.mean(list(accs.values()))) if accs else float("nan")
        return mean, accs

    # ---- run the full comparison ----
    from sklearn.base import clone
    # ---- resumable CV: checkpoint each completed cell to disk so the run
    #      survives a VM recycle. On restart, finished cells are skipped. ----
    ckpt_path = f"{ARTIFACTS}/naip_cv_checkpoint.json"
    ckpt = {}
    if os.path.exists(ckpt_path):
        try:
            ckpt = json.load(open(ckpt_path))
            _log(f"resuming from checkpoint with {len(ckpt)} completed cells")
        except Exception:
            ckpt = {}

    results = []   # (subset, family, grid_label, pooled_auc, mean_pp_auc)
    best = None
    t0 = time.time()
    # Reuse by column-set within a single run so identical subsets
    # (ALL == static+HOG when there are no extra feature groups) are not recomputed.
    colcache = {}

    def _ckkey(family, label, colkey):
        return f"{family}|{label}|{len(colkey)}"

    for sname, cols in subsets.items():
        colkey = tuple(cols)
        for family, grid in model_specs().items():
            for label, est in grid:
                cc = (colkey, family, label)
                cell = _ckkey(family, label, colkey)
                if cc in colcache:
                    pa, mpp, pp = colcache[cc]
                    note = " (reused)"
                elif cell in ckpt:
                    d = ckpt[cell]
                    pa, mpp, pp = d["pooled"], d["mean_pp"], d["pp"]
                    colcache[cc] = (pa, mpp, pp)
                    note = " (ckpt)"
                else:
                    oof = cv_oof(cols, clone(est))
                    pa = pooled_auc(oof)
                    mpp, pp = per_project_auc(oof)
                    colcache[cc] = (pa, mpp, pp)
                    note = ""
                    ckpt[cell] = {"pooled": float(pa), "mean_pp": float(mpp),
                                  "pp": {k: float(v) for k, v in pp.items()}}
                    with open(ckpt_path, "w") as f:
                        json.dump(ckpt, f)
                results.append((sname, family, label, pa, mpp))
                _log(f"  {sname:12s} {family:13s} {label:10s} "
                     f"pooled={pa:.4f} mean_pp={mpp:.4f}{note}")
                # select best by pooled AUC (tie-break mean per-project)
                key = (pa, mpp)
                if best is None or key > best["key"]:
                    best = dict(key=key, subset=sname, family=family,
                                label=label, est=clone(est), cols=cols,
                                pooled=pa, mean_pp=mpp, pp=pp)
    _log(f"CV finished in {time.time() - t0:.1f}s")

    # ---- markdown table (best grid per family x subset, by pooled AUC) ----
    by_cell = {}
    for sname, family, label, pa, mpp in results:
        k = (family, sname)
        if k not in by_cell or pa > by_cell[k][2]:
            by_cell[k] = (label, sname, pa, mpp)
    families = ["LogReg", "RandomForest"] + (["XGBoost"] if have_xgb else [])
    sub_order = ["static_only", "static+HOG", "ALL"]
    lines = []
    header = "| Model | " + " | ".join(sub_order) + " |"
    sep = "|" + "---|" * (len(sub_order) + 1)
    lines.append(header)
    lines.append(sep)
    for fam in families:
        row = [fam]
        for sub in sub_order:
            cell = by_cell.get((fam, sub))
            row.append(f"{cell[2]:.4f}" if cell else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    table_md = "\n".join(lines)
    print("\n=== Pooled CV AUC (best grid per cell) ===")
    print(table_md)

    # ---- refit best on all train data, save model+scaler+cols ----
    from sklearn.pipeline import Pipeline as Pipe
    final = Pipe([("scaler", StandardScaler()), ("clf", best["est"])])
    final.fit(Xtr[:, best["cols"]], ytr)
    payload = {
        "pipeline": final,
        "scaler": final.named_steps["scaler"],
        "model": final.named_steps["clf"],
        "cols": best["cols"],
        "feature_names": [names[i] for i in best["cols"]],
        "all_feature_names": names,
        "subset": best["subset"],
        "family": best["family"],
        "grid_label": best["label"],
        "cv_pooled_auc": best["pooled"],
        "cv_mean_per_project_auc": best["mean_pp"],
    }
    joblib.dump(payload, f"{ARTIFACTS}/best_naip_model.joblib")
    _log("saved " + f"{ARTIFACTS}/best_naip_model.joblib")

    _log(f"BEST: {best['family']} ({best['label']}) on {best['subset']}  "
         f"pooled={best['pooled']:.4f} mean_pp={best['mean_pp']:.4f}")
    for slug in sorted(best["pp"]):
        _log(f"  PP {slug:40s} {best['pp'][slug]:.4f}")

    # Persist a machine-readable summary so results survive VM recycles.
    summary = {
        "n_train_projects": int(len(np.unique(gtr))),
        "n_train_tiles": int(len(ytr)),
        "n_test_projects": int(len(np.unique(gte))),
        "n_test_tiles": int(len(yte)),
        "results": [
            {"subset": s, "family": f, "grid": l,
             "pooled_auc": float(pa), "mean_per_project_auc": float(mpp)}
            for (s, f, l, pa, mpp) in results
        ],
        "pooled_table_markdown": table_md,
        "best": {
            "family": best["family"], "grid": best["label"],
            "subset": best["subset"],
            "pooled_auc": float(best["pooled"]),
            "mean_per_project_auc": float(best["mean_pp"]),
            "per_project_auc": {k: float(v) for k, v in best["pp"].items()},
        },
        "test_used_for_metrics": False,
    }
    with open(f"{ARTIFACTS}/naip_track_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _log("saved " + f"{ARTIFACTS}/naip_track_summary.json")

    return table_md, best


if __name__ == "__main__":
    run_cv()
