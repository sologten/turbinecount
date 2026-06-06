"""Featurize every cached Sentinel-2 project into train/test matrices.

Writes one npz with X (N,F), feature_names, y (1=turbine), groups (project slug),
split ('train'/'test'). The split column is the FROZEN project-level split from
panel.json; the modeling code must only ever fit on split=='train'.
"""
from __future__ import annotations
import os, sys, glob, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gt, features_s2 as F

OUT = f"{gt.WORK}/artifacts/s2_matrix.npz"

def main():
    slug2split = {gt.slug(d["name"], d["state"]): d["split"] for d in gt.panel()}
    files = sorted(glob.glob(f"{gt.WORK}/cache/s2/*.npz"))
    names = None
    X, y, groups, split = [], [], [], []
    t0 = time.time()
    for fi, f in enumerate(files):
        slug = os.path.basename(f)[:-4]
        sp = slug2split.get(slug)
        if sp is None:
            continue
        z = np.load(f)
        for kind, lab in (("pos", 1), ("neg", 0)):
            arr = z[kind]
            for i in range(arr.shape[0]):
                d = F.patch_features(arr[i], with_hog=True)
                if names is None:
                    names = list(d.keys())
                X.append([d[k] for k in names]); y.append(lab)
                groups.append(slug); split.append(sp)
        if fi % 25 == 0:
            print(f"  {fi+1}/{len(files)} {slug} ({time.time()-t0:.0f}s)", flush=True)
    X = np.array(X, np.float32); y = np.array(y, np.int8)
    groups = np.array(groups); split = np.array(split)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    np.savez_compressed(OUT, X=X, feature_names=np.array(names), y=y,
                        groups=groups, split=split)
    tr = split == "train"; te = split == "test"
    print(f"\nS2 matrix: X{X.shape} | {len(set(groups))} projects | "
          f"train {tr.sum()} (pos {y[tr].sum()}) | test {te.sum()} (pos {y[te].sum()})")
    fam = {p: sum(n.startswith(p) for n in names) for p in ("stat_","cv131_","plx_","hog_")}
    print("feature families:", fam)
    print("saved ->", OUT)

if __name__ == "__main__":
    main()
