"""Part 3b: re-fetch real Sentinel-2 chips for selected case-study projects and
render RGB + band-parallax-fringe panels showing where the cue wins/fails.

Self-contained (does NOT use the dead gt.WORK path). Turbine points are pulled
live from the USWTDB PostgREST API; the S2 R/G/B/NIR bands are windowed-read
from AWS Earth Search exactly as fetch_s2.py does (no motion correction, so the
inter-band blade fringe is preserved).

Usage:
  python casestudy.py fetch <slug>     # cache one project's chip + turbine px
  python casestudy.py render            # build all panels from cached chips
"""
from __future__ import annotations
import os, sys, json, time, urllib.parse, urllib.request
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform, transform_bounds
from rasterio.windows import from_bounds
from pystac_client import Client

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "artifacts", "casestudy")
os.makedirs(CACHE, exist_ok=True)
PANEL = json.load(open(os.path.join(HERE, "panel.json")))

STAC = "https://earth-search.aws.element84.com/v1"
COLL = "sentinel-2-l2a"
BANDS = ("red", "green", "blue", "nir")
DATE = "2024-05-01/2024-09-30"
MAX_CLOUD = 20.0
PATCH = 48; HALF = 24
MAX_BOX_DEG = 0.06; PAD = 0.006
USWTDB = "https://eersc.usgs.gov/api/uswtdb/v1/turbines"

# selected case studies (slug -> role)
CASES = {
    "OR_golden_hills_oregon": "parallax WIN (largest fringe, AUC 1.00)",
    "ID_rockland": "low-salience WIN (turbine not salient, plx AUC 0.96)",
    "WY_roundhouse_wind_ii": "parallax FAIL (no fringe, plx AUC 0.29, detF1 0.00)",
    "TX_shamrock_wind_facility_white_mesa_wind_iii": "idle-fleet ceiling (48 big rotors, 0 detected)",
}


def slug(name, state):
    s = name.lower().replace(" ", "_")
    for ch in "()/.,'\"":
        s = s.replace(ch, "")
    return f"{state}_{s}"[:60]


def proj_meta(target_slug):
    for d in PANEL:
        if slug(d["name"], d["state"]) == target_slug:
            return d
    raise KeyError(target_slug)


def uswtdb_turbines(name, state):
    params = {"select": "xlong,ylat,t_conf_loc,t_rd,p_year",
              "p_name": f"eq.{name}", "t_state": f"eq.{state}", "t_offshore": "eq.0"}
    url = USWTDB + "?" + urllib.parse.urlencode(params, safe="*().,")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=40) as r:
        rows = json.load(r)
    rows = [x for x in rows if (x.get("t_conf_loc") or 0) >= 2]
    lon = np.array([x["xlong"] for x in rows], float)
    lat = np.array([x["ylat"] for x in rows], float)
    rd = np.array([x.get("t_rd") or np.nan for x in rows], float)
    return lon, lat, rd


def bbox_of(lon, lat):
    clon, clat = float(np.median(lon)), float(np.median(lat))
    h = MAX_BOX_DEG / 2
    lo_x, hi_x = max(lon.min(), clon - h), min(lon.max(), clon + h)
    lo_y, hi_y = max(lat.min(), clat - h), min(lat.max(), clat + h)
    return (lo_x - PAD, lo_y - PAD, hi_x + PAD, hi_y + PAD)


def fetch(target_slug):
    out = os.path.join(CACHE, f"{target_slug}.npz")
    if os.path.exists(out):
        return "cached"
    d = proj_meta(target_slug)
    lon, lat, rd = uswtdb_turbines(d["name"], d["state"])
    if lon.size == 0:
        return "no_turbines"
    bbox = bbox_of(lon, lat)
    cat = Client.open(STAC)
    s = cat.search(collections=[COLL], bbox=list(bbox), datetime=DATE,
                   query={"eo:cloud_cover": {"lt": MAX_CLOUD}}, max_items=30)
    items = list(s.items())
    if not items:
        return "no_scene"
    item = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))[0]
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
    chip = np.stack([a[:h, :w] for a in arrs], 0) / 10000.0
    xs, ys = warp_transform("EPSG:4326", crs, lon.tolist(), lat.tolist())
    inv = ~transform
    cols, rows = [], []
    for x, y in zip(xs, ys):
        c, r = inv * (x, y); cols.append(c); rows.append(r)
    rows = np.array(rows); cols = np.array(cols)
    inb = (rows >= HALF) & (rows < h - HALF) & (cols >= HALF) & (cols < w - HALF)
    rows, cols = rows[inb], cols[inb]
    np.savez_compressed(out, chip=chip.astype(np.float32),
                        turb_rows=rows.astype(np.float32), turb_cols=cols.astype(np.float32),
                        scene=item.id, cloud=float(item.properties.get("eo:cloud_cover", -1)),
                        name=d["name"], state=d["state"])
    return f"ok chip={h}x{w} turb={rows.size} cloud={item.properties.get('eo:cloud_cover',-1):.1f} scene={item.id}"


# ---------- parallax pipeline (mirrors features_s2.plx) ----------
def _znorm(a):
    return (a - a.mean()) / (a.std() + 1e-8)

def _gsmooth(a, s):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(a, s)

def hp(a):
    return _znorm(a - _gsmooth(a, 3.0))

def patch_plx(p):
    """Return (R,G,B highpass), rb-diff map, and key scalar features."""
    red, grn, blu, nir = p
    R, G, B = hp(red), hp(grn), hp(blu)
    C = 24
    def center(a, r=12):
        return a[C - r:C + r, C - r:C + r]
    rb = R - B
    Dc = center(rb, 12)
    disp = np.std(np.stack([R, G, B], 0), axis=0)
    feats = dict(
        rb_peak=float(np.max(np.abs(Dc))),
        rb_ptp=float(Dc.max() - Dc.min()),
        rb_energy=float(np.mean(Dc ** 2)),
        chroma_disp_peak=float(np.max(center(disp, 12))),
    )
    return R, G, B, rb, disp, feats


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "fetch":
        print(sys.argv[2], "->", fetch(sys.argv[2]))
    elif cmd == "fetchall":
        for sg in CASES:
            t = time.time()
            try:
                print(f"{sg:46s}", fetch(sg), f"({time.time()-t:.0f}s)")
            except Exception as e:
                print(f"{sg:46s} ERR {type(e).__name__}: {str(e)[:90]}")
    elif cmd == "list":
        for sg, role in CASES.items():
            ex = os.path.exists(os.path.join(CACHE, f"{sg}.npz"))
            print(f"  [{'x' if ex else ' '}] {sg:46s} {role}")
