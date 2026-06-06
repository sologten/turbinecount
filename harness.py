"""Rigorous, leakage-free evaluation harness.

Guarantees:
  * Only rows with split=='train' are ever used for model development / CV.
  * CV uses GroupKFold(group=project): a wind farm is never split across folds.
  * Feature standardization is fit on the training folds only (inside each fold).
  * The test set (split=='test') is loaded by a SEPARATE function that must be
    called exactly once, at the very end, after all model selection is frozen.
"""
from __future__ import annotations
import os
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

MATRIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "s2_matrix.npz")

def load(matrix=MATRIX):
    z = np.load(matrix, allow_pickle=True)
    return dict(X=z["X"], y=z["y"].astype(int), groups=z["groups"],
                split=z["split"], names=list(z["feature_names"]))

def train_view(d):
    m = d["split"] == "train"
    return d["X"][m], d["y"][m], d["groups"][m]

def cols(names, prefixes):
    return np.array([i for i, n in enumerate(names)
                     if any(n.startswith(p) for p in prefixes)])

def cv_oof(X, y, groups, make_model, n_splits=5, scale=True):
    """Return out-of-fold predictions and per-fold grouped AUC."""
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan)
    fold_auc = []
    for tr, va in gkf.split(X, y, groups):
        Xs = X
        if scale:
            sc = StandardScaler().fit(X[tr])
            Xs = sc.transform(X)
        m = make_model()
        m.fit(Xs[tr], y[tr])
        p = m.predict_proba(Xs[va])[:, 1] if hasattr(m, "predict_proba") else m.decision_function(Xs[va])
        oof[va] = p
        fold_auc.append(roc_auc_score(y[va], p))
    return oof, np.array(fold_auc)

def grouped_auc(y, p, groups):
    """AUC computed per project then averaged (robust to project size), plus pooled."""
    pooled = roc_auc_score(y, p)
    aucs = []
    for g in np.unique(groups):
        m = groups == g
        if len(np.unique(y[m])) == 2:
            aucs.append(roc_auc_score(y[m], p[m]))
    return pooled, float(np.mean(aucs)) if aucs else float("nan"), len(aucs)
