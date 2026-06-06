"""Ground-truth helpers: load the frozen panel and per-project turbine points
from the locally cached full USWTDB dump (no network)."""
from __future__ import annotations
import os, json
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
# USWTDB dump is only needed by the fetch scripts; override with TC_WORK if you have it.
WORK = os.environ.get("TC_WORK", os.path.join(ROOT, "data"))
ALL = f"{WORK}/cache/uswtdb/all_onshore.json"
PANEL = os.path.join(ROOT, "panel.json")

_TURB = None

def panel():
    return json.load(open(PANEL))

def _index():
    global _TURB
    if _TURB is None:
        data = json.load(open(ALL))
        d = defaultdict(list)
        for r in data:
            d[(r["p_name"], r["t_state"])].append(r)
        _TURB = d
    return _TURB

def turbines(name, state, conf_min=2):
    """High-confidence turbine (lon,lat,year,rotor_d) arrays for one project."""
    rs = [r for r in _index()[(name, state)] if (r.get("t_conf_loc") or 0) >= conf_min]
    lon = np.array([r["xlong"] for r in rs], float)
    lat = np.array([r["ylat"] for r in rs], float)
    yr = np.array([r.get("p_year") or 0 for r in rs], float)
    rd = np.array([r.get("t_rd") or np.nan for r in rs], float)
    return lon, lat, yr, rd

def slug(name, state):
    s = name.lower().replace(" ", "_")
    for ch in "()/.,'\"":
        s = s.replace(ch, "")
    return f"{state}_{s}"[:60]
