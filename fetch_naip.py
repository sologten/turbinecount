"""Resumable NAIP 0.6 m tile fetcher for the static-structure benchmark track.

Per project we sample up to N_POS high-confidence operational turbines and an
equal number of background points (>= NEG_MIN_M from any turbine), then batch-
read fixed-metric windows from the Microsoft Planetary Computer NAIP COGs (one
STAC search + one open per covering scene). Each tile is resampled to a fixed
PX x PX grid so scale is normalized across NAIP's 0.6/1.0 m vintages. One .npz
per project; re-running skips cached projects.
"""
from __future__ import annotations
import os, sys, time, json, urllib.parse, urllib.request
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds
from skimage.transform import resize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gt

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href="
SIZE_M = 120.0          # physical tile size
PX = 160                # normalized pixel grid
N_POS = 12              # positives sampled per project
NEG_MIN_M = 150.0
CACHE = f"{gt.WORK}/cache/naip"
LOG = f"{gt.WORK}/artifacts/naip_fetch_log.json"
os.makedirs(CACHE, exist_ok=True)
_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/124 Safari/537.36"}


def _search(lon, lat, d=0.012):
    body = json.dumps({"collections": ["naip"],
                       "bbox": [min(lon) - d, min(lat) - d, max(lon) + d, max(lat) + d],
                       "limit": 60}).encode()
    req = urllib.request.Request(PC_STAC, data=body,
                                 headers={**_UA, "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read()).get("features", [])


def _sign(href):
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(PC_SIGN + urllib.parse.quote(href, safe=""), headers=_UA),
        timeout=60).read())["href"]


def _read(ds, lon, lat, year):
    xs, ys = warp_transform("EPSG:4326", ds.crs, [lon], [lat])
    cx, cy = xs[0], ys[0]; h = SIZE_M / 2
    win = from_bounds(cx - h, cy - h, cx + h, cy + h, ds.transform)
    # out_shape lets GDAL decode from the nearest COG overview and resample to a
    # fixed PX grid in one step -> far faster than full-res read + skimage resize.
    nb = min(ds.count, 4)
    arr = ds.read(indexes=list(range(1, nb + 1)), out_shape=(nb, PX, PX),
                  window=win, boundless=True, fill_value=0).astype(np.float32)
    if arr.shape[0] < 3:
        return None
    if arr.shape[0] == 3:  # pad missing NIR with zeros
        arr = np.concatenate([arr, np.zeros((1, PX, PX), np.float32)], 0)
    return arr


def fetch_project(name, state):
    lon, lat, yr, rd = gt.turbines(name, state)
    if lon.size == 0:
        return "no_turbines"
    # sample positives spread across the farm
    if lon.size > N_POS:
        sel = np.linspace(0, lon.size - 1, N_POS).round().astype(int)
    else:
        sel = np.arange(lon.size)
    plon, plat, pyr = lon[sel], lat[sel], yr[sel]
    # negatives: random in farm bbox, >= NEG_MIN_M from any turbine
    rng = np.random.default_rng(hash((name, state, "naip")) % (2**32))
    deg = NEG_MIN_M / 111000.0
    nlon, nlat = [], []
    lo_x, hi_x, lo_y, hi_y = lon.min(), lon.max(), lat.min(), lat.max()
    pad = 0.01
    tries = 0
    while len(nlon) < len(sel) and tries < 400:
        tries += 1
        gx = rng.uniform(lo_x - pad, hi_x + pad); gy = rng.uniform(lo_y - pad, hi_y + pad)
        if np.hypot((lon - gx), (lat - gy)).min() < deg:
            continue
        nlon.append(gx); nlat.append(gy)
    pts = list(zip(plon, plat)) + list(zip(nlon, nlat))
    labels = [1] * len(plon) + [0] * len(nlon)
    if not pts:
        return "no_points"

    feats = _search([p[0] for p in pts], [p[1] for p in pts])
    if not feats:
        return "no_naip"
    tiles = [None] * len(pts); yrs = [0] * len(pts)
    # rank scenes by how many of our points they cover; open at most the top 2
    def cover(f):
        bb = f.get("bbox") or [0, 0, 0, 0]
        return sum(1 for p in pts if bb[0] <= p[0] <= bb[2] and bb[1] <= p[1] <= bb[3])
    feats = sorted([f for f in feats if cover(f) > 0], key=cover, reverse=True)[:2]

    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES",
                       GDAL_HTTP_TIMEOUT="25", GDAL_HTTP_MAX_RETRY="2", GDAL_HTTP_RETRY_DELAY="1",
                       VSI_CACHE="TRUE")
    with env:
        for f in feats:
            bb = f.get("bbox")
            if not bb:
                continue
            rem = [i for i in range(len(pts)) if tiles[i] is None
                   and bb[0] <= pts[i][0] <= bb[2] and bb[1] <= pts[i][1] <= bb[3]]
            if not rem:
                continue
            year = int(f["properties"].get("naip:year", 0))
            try:
                href = _sign(f["assets"]["image"]["href"])
                with rasterio.open(href) as ds:
                    for i in rem:
                        try:
                            t = _read(ds, pts[i][0], pts[i][1], year)
                        except Exception:
                            t = None
                        if t is not None:
                            tiles[i] = t; yrs[i] = year
            except Exception:
                continue
    keep = [i for i in range(len(pts)) if tiles[i] is not None]
    if not keep:
        return "no_tiles_read"
    X = np.stack([tiles[i] for i in keep], 0)
    y = np.array([labels[i] for i in keep])
    ty = np.array([yrs[i] for i in keep])
    # operational filter: positive only if turbine built by NAIP year
    pj = np.array([pyr[i] if i < len(plon) else 0 for i in keep])
    np.savez_compressed(f"{CACHE}/{gt.slug(name,state)}.npz",
                        X=X.astype(np.float32), y=y, tile_year=ty, p_year=pj,
                        name=name, state=state)
    return f"ok n={len(keep)} pos={int(y.sum())} neg={int((1-y).sum())} yrs={sorted(set(ty.tolist()))}"


SUBSET = f"{gt.WORK}/artifacts/naip_subset.json"


def main(budget_s=40.0, workers=6):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    panel = gt.panel()
    if os.path.exists(SUBSET):
        keep = set(json.load(open(SUBSET)))
        panel = [d for d in panel if gt.slug(d["name"], d["state"]) in keep]
    log = json.load(open(LOG)) if os.path.exists(LOG) else {}
    todo = [d for d in panel
            if not os.path.exists(f"{CACHE}/{gt.slug(d['name'],d['state'])}.npz")
            and not log.get(gt.slug(d['name'], d['state']), "").startswith(("no_",))]
    t0 = time.time(); lock = threading.Lock(); done = []

    def work(d):
        if time.time() - t0 > budget_s:
            return
        key = gt.slug(d["name"], d["state"])
        try:
            res = fetch_project(d["name"], d["state"])
        except Exception as e:
            res = f"ERR {type(e).__name__}: {str(e)[:70]}"
        with lock:
            log[key] = res; done.append(key)
            print(f"{key:42s} {res}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(as_completed([ex.submit(work, d) for d in todo]))
    json.dump(log, open(LOG, "w"))
    n_ok = sum(1 for v in log.values() if v.startswith("ok"))
    n_cached = len([f for f in os.listdir(CACHE) if f.endswith(".npz") and not f.startswith("_")
                    and "_pos_" not in f and "_neg_" not in f and "features" not in f])
    print(f"--- this run {len(done)} | ok-logged {n_ok} | project npz {n_cached}/{len(panel)} ---")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 40.0,
         int(sys.argv[2]) if len(sys.argv) > 2 else 6)
