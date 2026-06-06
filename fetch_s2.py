"""Resumable Sentinel-2 L2A patch fetcher for the band-parallax track.

For each project we find the lowest-cloud summer-2024 scene over a bounded box
around the project centroid, do a windowed COG read of the 10 m R/G/B/NIR bands
(co-registered to the same grid; moving-object inter-band parallax is preserved
because L2A does NOT motion-correct), then cut fixed PATCH x PATCH patches:
  * positives  : centered on each high-confidence USWTDB turbine in the box
  * negatives  : random points >= NEG_MIN_PX from every turbine (>=500 m), no nodata
Each project's patches are cached as one .npz. Re-running skips cached projects,
so the pull can be driven in short time-boxed chunks.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform, transform_bounds
from rasterio.windows import from_bounds
from pystac_client import Client

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gt

STAC = "https://earth-search.aws.element84.com/v1"
COLL = "sentinel-2-l2a"
BANDS = ("red", "green", "blue", "nir")        # B04,B03,B02,B08 @10m
DATE = "2024-05-01/2024-09-30"
MAX_CLOUD = 20.0
PATCH = 48          # px (480 m)
HALF = PATCH // 2
MAX_BOX_DEG = 0.06  # ~6 km cap around centroid
NEG_MIN_PX = 50     # >=500 m from any turbine
MAX_POS = 70        # cap positives per project
CACHE = f"{gt.WORK}/cache/s2"
LOG = f"{gt.WORK}/artifacts/s2_fetch_log.json"
os.makedirs(CACHE, exist_ok=True)


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
    # pad so edge turbines get full patches
    pad = 0.006
    return (lo_x - pad, lo_y - pad, hi_x + pad, hi_y + pad)


def fetch_project(name, state):
    lon, lat, yr, rd = gt.turbines(name, state)
    if lon.size == 0:
        return "no_turbines"
    bbox = _bounded_bbox(lon, lat)
    cat = Client.open(STAC)
    item = _scene(cat, list(bbox))
    if item is None:
        return "no_scene"
    arrs, transform, crs = [], None, None
    env = rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
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
    bands = np.stack([a[:h, :w] for a in arrs], 0) / 10000.0   # (4,H,W) reflectance
    if h < PATCH or w < PATCH:
        return "degenerate_chip"

    # map turbine lon/lat -> (row,col)
    xs, ys = warp_transform("EPSG:4326", crs, lon.tolist(), lat.tolist())
    inv = ~transform
    cols, rows = [], []
    for x, y in zip(xs, ys):
        c, r = inv * (x, y); cols.append(c); rows.append(r)
    rows = np.array(rows); cols = np.array(cols)
    inb = (rows >= HALF) & (rows < h - HALF) & (cols >= HALF) & (cols < w - HALF)
    rows, cols, yr2 = rows[inb], cols[inb], yr[inb]
    if rows.size == 0:
        return "no_turbine_in_box"
    if rows.size > MAX_POS:
        sel = np.linspace(0, rows.size - 1, MAX_POS).round().astype(int)
        rows, cols, yr2 = rows[sel], cols[sel], yr2[sel]

    def cut(r, c):
        r, c = int(round(r)), int(round(c))
        return bands[:, r - HALF:r + HALF, c - HALF:c + HALF]

    pos = np.stack([cut(r, c) for r, c in zip(rows, cols)], 0)  # (P,4,48,48)
    # negatives: random, >=NEG_MIN_PX from any turbine, no nodata
    gray = bands.mean(0)
    rng = np.random.default_rng(hash((name, state)) % (2**32))
    tr = np.column_stack([rows, cols])
    negs = []; tries = 0
    target = pos.shape[0]
    while len(negs) < target and tries < target * 60:
        tries += 1
        rr = rng.integers(HALF, h - HALF); cc = rng.integers(HALF, w - HALF)
        d = np.hypot(tr[:, 0] - rr, tr[:, 1] - cc).min()
        if d < NEG_MIN_PX:
            continue
        patch = bands[:, rr - HALF:rr + HALF, cc - HALF:cc + HALF]
        if (patch <= 0).mean() > 0.05:      # skip nodata
            continue
        negs.append(patch)
    if not negs:
        return "no_negatives"
    neg = np.stack(negs, 0)

    np.savez_compressed(f"{CACHE}/{gt.slug(name,state)}.npz",
                        pos=pos.astype(np.float32), neg=neg.astype(np.float32),
                        scene=item.id, cloud=float(item.properties.get("eo:cloud_cover", -1)),
                        gsd=10.0, name=name, state=state)
    return f"ok pos={pos.shape[0]} neg={neg.shape[0]} cloud={item.properties.get('eo:cloud_cover',-1):.1f}"


def main(budget_s=40.0, workers=8):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    panel = gt.panel()
    log = json.load(open(LOG)) if os.path.exists(LOG) else {}
    todo = [d for d in panel
            if not os.path.exists(f"{CACHE}/{gt.slug(d['name'],d['state'])}.npz")
            and not log.get(gt.slug(d['name'], d['state']), "").startswith(("no_", "degenerate"))]
    t0 = time.time(); lock = threading.Lock(); done = []

    def work(d):
        if time.time() - t0 > budget_s:
            return None
        key = gt.slug(d["name"], d["state"])
        try:
            res = fetch_project(d["name"], d["state"])
        except Exception as e:
            res = f"ERR {type(e).__name__}: {str(e)[:80]}"
        with lock:
            log[key] = res; done.append(key)
            print(f"{key:42s} {res}", flush=True)
        return key

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, d) for d in todo]
        for _ in as_completed(futs):
            pass
    json.dump(log, open(LOG, "w"))
    n_ok = sum(1 for v in log.values() if v.startswith("ok"))
    n_cached = len([f for f in os.listdir(CACHE) if f.endswith(".npz")])
    print(f"--- this run: {len(done)} | total ok-logged: {n_ok} | cached npz: {n_cached} / {len(panel)} ---")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 38.0)
