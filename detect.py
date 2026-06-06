"""Detection-mode evaluation for the frozen Sentinel-2 turbine detector.

The existing model was only evaluated as candidate-point classification
(ROC-AUC). This module runs it as a *detector* on whole scenes and reports the
SOTA-comparable metrics: object-level precision/recall/F1 and a country-style
count R^2 vs USWTDB, so it can be compared apples-to-apples with the literature
(GRW count-r2 0.93; offshore object-level F1).

LEAKAGE RULES (strict):
  * The operating threshold is derived from TRAIN projects ONLY (phase
    `threshold`, micro-F1 maximizing). TEST projects are scored exactly once
    with that frozen threshold (phase `test`).
  * The frozen model is used as-is; nothing is retrained.

Pipeline phases (each resumable + time-boxed, because bash calls die at ~45s):
  fetch <split> [budget_s]   STAC-pull the lowest-cloud 2024-summer ~6km chip per
                             project, windowed-read R/G/B/NIR (4,H,W) reflectance,
                             cache (chip, transform, crs, turb_rows, turb_cols)
                             to cache/s2_detect/<slug>.npz. Skips cached.
  score <split> [n]          Dense sliding-window detection on cached chips:
                             stride 8, skip 24px border, 48x48 patch -> 29 model
                             features -> predict_proba -> score map -> NMS
                             (min-sep 6px=60m). Cache ALL peaks (row,col,score)
                             to cache/s2_detect/det_<slug>.npz so threshold/test
                             phases just apply a cutoff. Processes n projects/run.
  threshold                  Sweep thresholds on TRAIN detection caches, greedy-
                             match peaks to USWTDB within 6px, pick the single
                             threshold maximizing micro-F1. Save to
                             artifacts/detection_threshold.json.
  test                       Apply the frozen TRAIN threshold to TEST caches ONCE.
                             Per-project TP/FP/FN/P/R/F1/counts, micro+macro P/R/F1,
                             count R^2 (+slope, MAPE). Save detection_benchmark.json
                             and fig_count_scatter.png.
"""
from __future__ import annotations
import os, sys, time, json, math
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform, transform_bounds
from rasterio.windows import from_bounds
from pystac_client import Client
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gt
import features_s2 as F

# ---- config (mirrors fetch_s2 so chips are pulled the SAME way) ----
STAC = "https://earth-search.aws.element84.com/v1"
COLL = "sentinel-2-l2a"
BANDS = ("red", "green", "blue", "nir")        # B04,B03,B02,B08 @10m
DATE = "2024-05-01/2024-09-30"
MAX_CLOUD = 20.0
PATCH = 48          # px (480 m)
HALF = PATCH // 2
MAX_BOX_DEG = 0.06  # ~6 km cap around centroid
PAD = 0.006

STRIDE = 8          # px between window centers
BORDER = 24         # skip this many px at chip edge (== HALF, full patch needed)
NMS_SEP = 6         # min separation between kept peaks (px) = 60 m
MATCH_PX = 6        # greedy match radius detection<->USWTDB (px) = 60 m

CACHE = f"{gt.WORK}/cache/s2_detect"
ART = f"{gt.WORK}/artifacts"
# host-readable mirror (code dir is mounted on the host) for progress polling
HOST_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(HOST_ART, exist_ok=True)
THRESH_JSON = f"{ART}/detection_threshold.json"
BENCH_JSON = f"{ART}/detection_benchmark.json"
HOST_FIG = "/Users/sophie/Downloads/cs131/turbinecount_v2/artifacts/fig_count_scatter.png"
os.makedirs(CACHE, exist_ok=True)

_MODEL = None
def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(f"{ART}/best_s2_model.joblib")
        _MODEL["cols"] = [str(c) for c in _MODEL["cols"]]
    return _MODEL


# ---------------- phase: fetch whole chips ----------------
def _scene(cat, bbox):
    s = cat.search(collections=[COLL], bbox=bbox, datetime=DATE,
                   query={"eo:cloud_cover": {"lt": MAX_CLOUD}}, max_items=30)
    items = list(s.items())
    if not items:
        return None
    return sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))[0]


def _bounded_bbox(lon, lat):
    clon, clat = float(np.median(lon)), float(np.median(lat))
    h = MAX_BOX_DEG / 2
    lo_x, hi_x = max(lon.min(), clon - h), min(lon.max(), clon + h)
    lo_y, hi_y = max(lat.min(), clat - h), min(lat.max(), clat + h)
    return (lo_x - PAD, lo_y - PAD, hi_x + PAD, hi_y + PAD)


def fetch_project(name, state):
    """Pull one whole chip (4,H,W) reflectance + transform/crs + turbine (row,col)."""
    lon, lat, yr, rd = gt.turbines(name, state)
    if lon.size == 0:
        return "no_turbines"
    bbox = _bounded_bbox(lon, lat)
    cat = Client.open(STAC)
    item = _scene(cat, list(bbox))
    if item is None:
        return "no_scene"
    arrs, transform, crs = [], None, None
    env = rasterio.Env(AWS_NO_SIGN_REQUEST="YES",
                       GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                       GDAL_HTTP_MULTIRANGE="YES")
    with env:
        for b in BANDS:
            with rasterio.open(item.assets[b].href) as ds:
                bp = transform_bounds("EPSG:4326", ds.crs, *bbox)
                win = from_bounds(*bp, ds.transform)
                a = ds.read(1, window=win).astype(np.float32)
                if transform is None:
                    transform, crs = ds.window_transform(win), ds.crs
                arrs.append(a)
    h = min(a.shape[0] for a in arrs); w = min(a.shape[1] for a in arrs)
    chip = np.stack([a[:h, :w] for a in arrs], 0) / 10000.0   # (4,H,W) reflectance
    if h < PATCH or w < PATCH:
        return "degenerate_chip"

    # map ALL high-conf turbine lon/lat -> (row,col); keep those whose center can
    # carry a full patch (matches what the detector can possibly hit)
    xs, ys = warp_transform("EPSG:4326", crs, lon.tolist(), lat.tolist())
    inv = ~transform
    cols, rows = [], []
    for x, y in zip(xs, ys):
        c, r = inv * (x, y); cols.append(c); rows.append(r)
    rows = np.array(rows); cols = np.array(cols)
    inb = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    rows, cols = rows[inb], cols[inb]
    if rows.size == 0:
        return "no_turbine_in_box"

    np.savez_compressed(f"{CACHE}/{gt.slug(name,state)}.npz",
                        chip=chip.astype(np.float32),
                        transform=np.array(transform).reshape(3, 3) if hasattr(transform, "__len__") else
                                  np.array([transform.a, transform.b, transform.c,
                                            transform.d, transform.e, transform.f, 0, 0, 1]).reshape(3, 3),
                        crs=str(crs),
                        turb_rows=rows.astype(np.float32),
                        turb_cols=cols.astype(np.float32),
                        scene=item.id,
                        cloud=float(item.properties.get("eo:cloud_cover", -1)),
                        name=name, state=state)
    return f"ok chip={h}x{w} turb={rows.size} cloud={item.properties.get('eo:cloud_cover',-1):.1f}"


def cmd_fetch(split, budget_s=38.0, workers=6):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    panel = gt.panel()
    todo = [d for d in panel if d["split"] == split
            and not os.path.exists(f"{CACHE}/{gt.slug(d['name'],d['state'])}.npz")]
    t0 = time.time(); lock = threading.Lock(); did = [0]

    def work(d):
        if time.time() - t0 > budget_s:
            return
        slug = gt.slug(d["name"], d["state"])
        try:
            res = fetch_project(d["name"], d["state"])
        except Exception as e:
            res = f"ERR {type(e).__name__}: {str(e)[:80]}"
        with lock:
            print(f"{slug:42s} {res}", flush=True); did[0] += 1

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, d) for d in todo]
        for _ in as_completed(futs):
            pass
    cached = [f for f in os.listdir(CACHE) if f.endswith(".npz") and not f.startswith("det_")]
    print(f"--- fetch {split}: {did[0]} this run | {len(cached)} chips cached ---")


# ---------------- phase: dense detection scoring ----------------
def _score_chip(chip):
    """Dense sliding-window model scores. Returns (rows, cols, scores) on grid."""
    m = model()
    cols_order = m["cols"]; scaler = m["scaler"]; mdl = m["model"]
    _, H, W = chip.shape
    rc = list(range(BORDER, H - BORDER, STRIDE))
    cc = list(range(BORDER, W - BORDER, STRIDE))
    feats = []; centers = []
    for r in rc:
        for c in cc:
            patch = chip[:, r - HALF:r + HALF, c - HALF:c + HALF]
            if (patch <= 0).mean() > 0.05:      # nodata -> skip
                continue
            d = F.patch_features(patch, with_hog=False)
            feats.append([d[k] for k in cols_order])
            centers.append((r, c))
    if not feats:
        return np.array([]), np.array([]), np.array([])
    X = np.asarray(feats, float)
    X = scaler.transform(X)
    s = mdl.predict_proba(X)[:, 1]
    centers = np.asarray(centers, float)
    return centers[:, 0], centers[:, 1], s


def _nms(rows, cols, scores, sep=NMS_SEP):
    """Keep local maxima: sort by score desc, drop any peak within `sep` px of a
    higher-scoring kept peak."""
    order = np.argsort(-scores)
    keep_r, keep_c, keep_s = [], [], []
    for i in order:
        r, c, sc = rows[i], cols[i], scores[i]
        ok = True
        for kr, kc in zip(keep_r, keep_c):
            if math.hypot(r - kr, c - kc) < sep:
                ok = False; break
        if ok:
            keep_r.append(r); keep_c.append(c); keep_s.append(sc)
    return np.array(keep_r), np.array(keep_c), np.array(keep_s)


def score_project(slug, force=False):
    out = f"{CACHE}/det_{slug}.npz"
    if os.path.exists(out) and not force:
        return "cached"
    z = np.load(f"{CACHE}/{slug}.npz", allow_pickle=True)
    chip = z["chip"]
    rows, cols, scores = _score_chip(chip)
    if rows.size == 0:
        return "empty"
    pr, pc, ps = _nms(rows, cols, scores)   # NMS over ALL scored peaks (threshold applied later)
    np.savez_compressed(out, peak_rows=pr, peak_cols=pc, peak_scores=ps,
                        turb_rows=z["turb_rows"], turb_cols=z["turb_cols"],
                        name=str(z["name"]), state=str(z["state"]))
    return f"ok peaks={pr.size} turb={z['turb_rows'].size}"


def cmd_score(split, n=999, budget_s=38.0):
    """Score as many un-scored cached chips of `split` as fit in budget_s."""
    panel = gt.panel()
    todo = [d for d in panel if d["split"] == split]
    t0 = time.time(); did = 0
    for d in todo:
        if did >= n or time.time() - t0 > budget_s:
            break
        slug = gt.slug(d["name"], d["state"])
        if not os.path.exists(f"{CACHE}/{slug}.npz"):
            continue
        if os.path.exists(f"{CACHE}/det_{slug}.npz"):
            continue
        try:
            res = score_project(slug)
        except Exception as e:
            res = f"ERR {type(e).__name__}: {str(e)[:80]}"
        print(f"{slug:42s} {res}", flush=True); did += 1
    n_split = sum(os.path.exists(f"{CACHE}/det_{gt.slug(d['name'],d['state'])}.npz") for d in todo)
    n_chip = sum(os.path.exists(f"{CACHE}/{gt.slug(d['name'],d['state'])}.npz") for d in todo)
    print(f"--- score {split}: {did} this run | scored {n_split}/{len(todo)} | chips {n_chip}/{len(todo)} ---")


# ---------------- matching + metrics ----------------
def _greedy_match(pr, pc, ps, tr, tc, thr, radius=MATCH_PX):
    """Detections above `thr` greedily matched (highest score first) to nearest
    unused USWTDB turbine within `radius`. Returns TP, FP, FN."""
    keep = ps >= thr
    dr, dc, ds = pr[keep], pc[keep], ps[keep]
    order = np.argsort(-ds)
    used = np.zeros(tr.size, bool)
    tp = 0
    for i in order:
        # nearest unused turbine
        best_j, best_d = -1, radius + 1e-9
        for j in range(tr.size):
            if used[j]:
                continue
            dd = math.hypot(dr[i] - tr[j], dc[i] - tc[j])
            if dd <= best_d:
                best_d = dd; best_j = j
        if best_j >= 0:
            used[best_j] = True; tp += 1
    fp = int(keep.sum()) - tp
    fn = int(tr.size) - tp
    return tp, fp, fn


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _load_dets(split):
    panel = gt.panel()
    out = []
    for d in panel:
        if d["split"] != split:
            continue
        slug = gt.slug(d["name"], d["state"])
        f = f"{CACHE}/det_{slug}.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        out.append(dict(slug=slug, name=str(z["name"]), state=str(z["state"]),
                        pr=z["peak_rows"], pc=z["peak_cols"], ps=z["peak_scores"],
                        tr=z["turb_rows"], tc=z["turb_cols"]))
    return out


# ---------------- phase: threshold from TRAIN ----------------
def cmd_threshold():
    dets = _load_dets("train")
    if not dets:
        print("no train detection caches yet"); return
    all_scores = np.concatenate([d["ps"] for d in dets if d["ps"].size])
    grid = np.unique(np.quantile(all_scores, np.linspace(0.0, 1.0, 201)))
    best = None
    for thr in grid:
        TP = FP = FN = 0
        for d in dets:
            tp, fp, fn = _greedy_match(d["pr"], d["pc"], d["ps"], d["tr"], d["tc"], thr)
            TP += tp; FP += fp; FN += fn
        p, r, f = _prf(TP, FP, FN)
        if best is None or f > best["f1"]:
            best = dict(threshold=float(thr), f1=f, precision=p, recall=r,
                        TP=TP, FP=FP, FN=FN)
    best["n_train_projects"] = len(dets)
    json.dump(best, open(THRESH_JSON, "w"), indent=1)
    print(f"TRAIN threshold (micro-F1 max) over {len(dets)} projects:")
    print(f"  thr={best['threshold']:.4f}  microF1={best['f1']:.3f}  P={best['precision']:.3f}  R={best['recall']:.3f}")
    print(f"  TP={best['TP']} FP={best['FP']} FN={best['FN']}  -> {THRESH_JSON}")


# ---------------- phase: TEST eval (once) ----------------
def cmd_test():
    if not os.path.exists(THRESH_JSON):
        print("run `threshold` first"); return
    thr = json.load(open(THRESH_JSON))["threshold"]
    dets = _load_dets("test")
    if not dets:
        print("no test detection caches yet"); return
    rows = []
    TP = FP = FN = 0
    ps_list, rs_list, fs_list = [], [], []
    det_counts, true_counts = [], []
    for d in dets:
        tp, fp, fn = _greedy_match(d["pr"], d["pc"], d["ps"], d["tr"], d["tc"], thr)
        p, r, f = _prf(tp, fp, fn)
        det_count = int((d["ps"] >= thr).sum())
        true_count = int(d["tr"].size)
        rows.append(dict(slug=d["slug"], name=d["name"], state=d["state"],
                         TP=tp, FP=fp, FN=fn, precision=round(p, 4), recall=round(r, 4),
                         f1=round(f, 4), detected_count=det_count, true_count=true_count))
        TP += tp; FP += fp; FN += fn
        ps_list.append(p); rs_list.append(r); fs_list.append(f)
        det_counts.append(det_count); true_counts.append(true_count)

    micro_p, micro_r, micro_f = _prf(TP, FP, FN)
    macro_p, macro_r, macro_f = float(np.mean(ps_list)), float(np.mean(rs_list)), float(np.mean(fs_list))

    dc = np.array(det_counts, float); ttc = np.array(true_counts, float)
    # (a) identity-line R2: how close detected_count is to true_count (1 - SSE about y=x).
    ss_res = np.sum((ttc - dc) ** 2)
    ss_tot = np.sum((ttc - ttc.mean()) ** 2)
    r2_identity = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # (b) regression-fit R2 == pearson^2 (how well a *linear calibration* of detected
    #     count tracks true count; this is the GRW-style "count r2" that allows a slope/intercept).
    slope_ls, intercept_ls = (np.polyfit(ttc, dc, 1) if ttc.size > 1 else (float("nan"), 0.0))
    slope_ls = float(slope_ls); intercept_ls = float(intercept_ls)
    pearson = float(np.corrcoef(ttc, dc)[0, 1]) if ttc.size > 1 else float("nan")
    r2_fit = float(pearson ** 2)
    nz = ttc > 0
    mape = float(np.mean(np.abs(dc[nz] - ttc[nz]) / ttc[nz])) if nz.any() else float("nan")

    summary = dict(
        operating_threshold=thr, threshold_source="TRAIN micro-F1 max (frozen)",
        n_test_projects=len(dets),
        micro=dict(precision=round(micro_p, 4), recall=round(micro_r, 4), f1=round(micro_f, 4),
                   TP=TP, FP=FP, FN=FN),
        macro=dict(precision=round(macro_p, 4), recall=round(macro_r, 4), f1=round(macro_f, 4)),
        count_r2_identity=round(r2_identity, 4),
        count_r2_fit=round(r2_fit, 4),
        count_slope=round(slope_ls, 4), count_intercept=round(intercept_ls, 4),
        count_pearson_r=round(pearson, 4), count_mape=round(mape, 4),
        total_detected=int(dc.sum()), total_true=int(ttc.sum()),
        match_radius_px=MATCH_PX, nms_sep_px=NMS_SEP, stride_px=STRIDE,
    )
    out = dict(summary=summary, per_project=rows)
    json.dump(out, open(BENCH_JSON, "w"), indent=1)
    try:
        json.dump(out, open(f"{HOST_ART}/detection_benchmark.json", "w"), indent=1)
    except Exception:
        pass

    # ---- count scatter ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        mx = max(ttc.max(), dc.max()) * 1.08 + 1
        ax.plot([0, mx], [0, mx], "--", color="0.6", lw=1, label="y = x")
        xs = np.linspace(0, mx, 50)
        ax.plot(xs, slope_ls * xs + intercept_ls, "-", color="C3", lw=1.2,
                label=f"linear fit (slope {slope_ls:.2f})")
        ax.scatter(ttc, dc, s=40, alpha=0.8, edgecolor="k", linewidth=0.4, color="C0")
        ax.set_xlabel("USWTDB true turbine count in chip (per project)")
        ax.set_ylabel("Detected turbine count (per project)")
        ax.set_title(f"Detection count vs ground truth (n={len(dets)} test projects)\n"
                     f"fit R$^2$={r2_fit:.3f}  identity R$^2$={r2_identity:.3f}  slope={slope_ls:.2f}  "
                     f"MAPE={mape*100:.0f}%  thr={thr:.2f}")
        ax.set_xlim(0, mx); ax.set_ylim(0, mx); ax.legend(loc="upper left")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        # HOST_ART is the mounted code-dir artifacts folder (== host /Users/.../artifacts)
        fig.savefig(f"{HOST_ART}/fig_count_scatter.png", dpi=130)
        fig.savefig(f"{ART}/fig_count_scatter.png", dpi=130)
        print(f"  scatter -> {HOST_ART}/fig_count_scatter.png")
    except Exception as e:
        print("  scatter skipped:", type(e).__name__, str(e)[:120])

    print(f"\n=== TEST detection benchmark (n={len(dets)}, threshold {thr:.4f} [TRAIN-derived, frozen]) ===")
    print(f"  micro  P={micro_p:.3f} R={micro_r:.3f} F1={micro_f:.3f}  (TP={TP} FP={FP} FN={FN})")
    print(f"  macro  P={macro_p:.3f} R={macro_r:.3f} F1={macro_f:.3f}")
    print(f"  count  fitR2={r2_fit:.3f}  identityR2={r2_identity:.3f}  slope={slope_ls:.3f}  pearson={pearson:.3f}  MAPE={mape*100:.1f}%")
    print(f"  totals detected={int(dc.sum())} true={int(ttc.sum())}")
    print(f"  saved -> {BENCH_JSON}")


# ---------------- full pipeline (single detached process) ----------------
def _slug(d):
    return gt.slug(d["name"], d["state"])


def fetch_one(d, log):
    slug = _slug(d)
    if os.path.exists(f"{CACHE}/{slug}.npz"):
        return "chip_cached"
    try:
        res = fetch_project(d["name"], d["state"])
    except Exception as e:
        res = f"ERR {type(e).__name__}: {str(e)[:80]}"
    log(f"FETCH {slug:42s} {res}")
    return res


def score_one(d, log):
    slug = _slug(d)
    if os.path.exists(f"{CACHE}/det_{slug}.npz"):
        return "det_cached"
    if not os.path.exists(f"{CACHE}/{slug}.npz"):
        return "no_chip"
    try:
        res = score_project(slug)
    except Exception as e:
        res = f"ERR {type(e).__name__}: {str(e)[:80]}"
    log(f"SCORE {slug:42s} {res}")
    return res


def cmd_all(budget_s=600.0, min_train=18):
    """Resumable end-to-end: fetch missing chips, score missing chips, and once
    >= min_train train projects and ALL available test projects are scored,
    derive the TRAIN threshold and run the TEST eval. Idempotent: re-running
    advances progress and re-emits results. Writes a host-readable status file."""
    import datetime
    logf = open(f"{HOST_ART}/detection_run.log", "a", buffering=1)
    t_start = time.time()

    def log(msg):
        logf.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}\n")
        logf.flush()
        print(msg, flush=True)

    panel = gt.panel()
    train = [d for d in panel if d["split"] == "train"]
    test = [d for d in panel if d["split"] == "test"]
    log(f"=== pipeline tick: {len(train)} train, {len(test)} test ===")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    # 1) FETCH missing chips (prioritize test + first min_train*2 train)
    pri = test + train
    todo = [d for d in pri if not os.path.exists(f"{CACHE}/{_slug(d)}.npz")]
    if todo and time.time() - t_start < budget_s:
        log(f"fetching {len(todo)} chips ...")
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(fetch_one, d, log): d for d in todo}
            for _ in as_completed(futs):
                if time.time() - t_start > budget_s:
                    break

    # 2) SCORE missing chips: ALL test, then train up to a healthy margin
    todo = [d for d in test if os.path.exists(f"{CACHE}/{_slug(d)}.npz")
            and not os.path.exists(f"{CACHE}/det_{_slug(d)}.npz")]
    todo += [d for d in train if os.path.exists(f"{CACHE}/{_slug(d)}.npz")
             and not os.path.exists(f"{CACHE}/det_{_slug(d)}.npz")]
    log(f"scoring up to {len(todo)} chips ...")
    for i, d in enumerate(todo):
        if time.time() - t_start > budget_s:
            log("budget hit during scoring"); break
        score_one(d, log)
        if (i + 1) % 5 == 0:
            ndet_tr = sum(os.path.exists(f"{CACHE}/det_{_slug(x)}.npz") for x in train)
            ndet_te = sum(os.path.exists(f"{CACHE}/det_{_slug(x)}.npz") for x in test)
            log(f"  scored {i+1}/{len(todo)} | det train {ndet_tr} test {ndet_te}")

    ndet_tr = sum(os.path.exists(f"{CACHE}/det_{_slug(d)}.npz") for d in train)
    ndet_te = sum(os.path.exists(f"{CACHE}/det_{_slug(d)}.npz") for d in test)
    nchip_te = sum(os.path.exists(f"{CACHE}/{_slug(d)}.npz") for d in test)
    log(f"det caches: train {ndet_tr}/{len(train)}, test {ndet_te}/{len(test)} (test chips {nchip_te})")

    # 3+4) once enough train + all available test are scored, do threshold + test
    if ndet_tr >= min_train and ndet_te >= nchip_te and ndet_te > 0:
        log("deriving TRAIN threshold (micro-F1 max) ...")
        try:
            cmd_threshold()
            thr = json.load(open(THRESH_JSON))
            log(f"TRAIN threshold = {thr['threshold']:.4f}  microF1={thr['f1']:.3f}")
            log("running TEST eval at frozen threshold ...")
            cmd_test()
            log("TEST eval done -> detection_benchmark.json")
        except Exception as e:
            import traceback
            log(f"EVAL ERR {type(e).__name__}: {e}\n{traceback.format_exc()[:400]}")
    else:
        log(f"not ready for eval yet (need train>={min_train}, test all {nchip_te})")
    try:
        write_status()
    except Exception:
        pass
    log("=== tick done ===")
    logf.close()


def write_status():
    """Dump fetch/score progress to a host-readable JSON for polling."""
    panel = gt.panel()
    te = [d for d in panel if d["split"] == "test"]
    tr = [d for d in panel if d["split"] == "train"]
    def cnt(lst, pre=""):
        return sum(os.path.exists(f"{CACHE}/{pre}{gt.slug(d['name'],d['state'])}.npz") for d in lst)
    st = dict(
        chip_test=cnt(te), n_test=len(te), chip_train=cnt(tr), n_train=len(tr),
        det_test=cnt(te, "det_"), det_train=cnt(tr, "det_"),
        thresh_done=os.path.exists(THRESH_JSON),
        bench_done=os.path.exists(BENCH_JSON),
    )
    json.dump(st, open(f"{HOST_ART}/detection_status.json", "w"), indent=1)
    print(json.dumps(st))
    return st


# ---------------- cli ----------------
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "all":
        cmd_all(budget_s=float(sys.argv[2]) if len(sys.argv) > 2 else 40.0); sys.exit(0)
    if cmd == "scoreboth":
        cmd_score("train", budget_s=float(sys.argv[2]) if len(sys.argv) > 2 else 38.0)
        cmd_score("test", budget_s=10.0)
        write_status(); sys.exit(0)
    if cmd == "writestatus":
        write_status(); sys.exit(0)
    if cmd == "fetch":
        cmd_fetch(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 38.0)
    elif cmd == "score":
        cmd_score(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2)
    elif cmd == "threshold":
        cmd_threshold()
    elif cmd == "test":
        cmd_test()
    elif cmd == "status":
        chips = [f for f in os.listdir(CACHE) if f.endswith(".npz") and not f.startswith("det_")]
        dets = [f for f in os.listdir(CACHE) if f.startswith("det_")]
        print(f"chips cached: {len(chips)} | detection caches: {len(dets)}")
    else:
        print("usage: detect.py [fetch <split>|score <split> [n]|threshold|test|status]")
