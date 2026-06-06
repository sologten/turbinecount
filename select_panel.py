"""Select a diverse multi-state panel of USWTDB wind projects and freeze a
leakage-free train/test split.

Design choices (all made on metadata only, BEFORE any imagery or feature is
seen, so the split cannot leak through the modeling):
  * Candidate projects: commissioned 2008-2023 (so operational in 2024 imagery),
    >=20 high-confidence (t_conf_loc>=2) turbines, <=400 turbines total.
  * Per-state cap to prevent Texas from dominating; within a state we spread the
    picks across commission years for vintage diversity.
  * Split: whole projects are assigned to train or test (grouped split, no farm
    is ever split across the two), stratified by census region x vintage bin,
    ~20% of projects held out as a LOCKED test set with a fixed seed.
  * Cross-validation on train uses GroupKFold with group=project, so CV folds
    also never split a farm.
"""
from __future__ import annotations
import json
import numpy as np
from collections import defaultdict

ALL = "/sessions/upbeat-wizardly-ramanujan/work/cache/uswtdb/all_onshore.json"
OUT = "/sessions/upbeat-wizardly-ramanujan/work/artifacts"

REGION = {  # coarse census-ish regions for stratification + terrain diversity
 'TX':'S_plains','OK':'S_plains','KS':'S_plains','NM':'S_plains','CO':'mountain_w',
 'WY':'mountain_w','MT':'mountain_w','ID':'mountain_w','UT':'mountain_w','AZ':'mountain_w','NV':'mountain_w',
 'IA':'midwest','IL':'midwest','MN':'midwest','MO':'midwest','IN':'midwest','MI':'midwest',
 'WI':'midwest','OH':'midwest','ND':'n_plains','SD':'n_plains','NE':'n_plains',
 'CA':'pacific','OR':'pacific','WA':'pacific','HI':'pacific',
 'NY':'northeast','PA':'northeast','ME':'northeast','WV':'northeast','NH':'northeast',
 'VT':'northeast','MD':'northeast','NC':'southeast','PR':'southeast'}

PER_STATE_CAP = 6   # tuned to land ~140 projects
SEED = 20260530

def vintage_bin(y):
    return '08-13' if y<=2013 else ('14-18' if y<=2018 else '19-23')

def main():
    data=json.load(open(ALL))
    proj=defaultdict(list)
    for r in data: proj[(r["p_name"], r["t_state"])].append(r)
    rows=[]
    for (name,state),rs in proj.items():
        yrs=[x["p_year"] for x in rs if x.get("p_year")]
        if not yrs: continue
        yr=int(round(np.median(yrs)))
        hi=[x for x in rs if (x.get("t_conf_loc") or 0)>=2]
        if not (2008<=yr<=2023 and len(hi)>=20 and len(rs)<=400): continue
        rds=[x["t_rd"] for x in rs if x.get("t_rd")]
        rows.append(dict(name=name,state=state,year=yr,n=len(rs),n_hi=len(hi),
                         lat=float(np.mean([x["ylat"] for x in rs])),
                         lon=float(np.mean([x["xlong"] for x in rs])),
                         rotor_d=float(np.mean(rds)) if rds else None,
                         region=REGION.get(state,'other')))
    # per-state selection, spread across years, deterministic
    by_state=defaultdict(list)
    for d in rows: by_state[d["state"]].append(d)
    panel=[]
    for st, ds in by_state.items():
        ds=sorted(ds, key=lambda d:(d["year"], d["name"]))
        k=min(PER_STATE_CAP, len(ds))
        if k==0: continue
        idx=np.linspace(0, len(ds)-1, k).round().astype(int)
        panel+=[ds[i] for i in sorted(set(idx.tolist()))]
    panel=sorted(panel, key=lambda d:(d["state"], d["name"]))

    # ---- frozen grouped split: ~20% of projects to test, stratified ----
    rng=np.random.default_rng(SEED)
    strata=defaultdict(list)
    for d in panel: strata[(d["region"], vintage_bin(d["year"]))].append(d)
    test=set()
    for key, ds in strata.items():
        ds=sorted(ds, key=lambda d:d["name"]) ; rng.shuffle(ds)
        ntest=max(1, round(0.2*len(ds))) if len(ds)>=3 else (1 if len(ds)>=2 else 0)
        for d in ds[:ntest]: test.add((d["name"], d["state"]))
    for d in panel:
        d["split"]="test" if (d["name"],d["state"]) in test else "train"

    n_tr=sum(d["split"]=="train" for d in panel); n_te=len(panel)-n_tr
    json.dump(panel, open(f"{OUT}/panel.json","w"), indent=1)
    print(f"PANEL: {len(panel)} projects | train {n_tr} | test {n_te}")
    print(f"states: {len(set(d['state'] for d in panel))} | turbines(high-conf): {sum(d['n_hi'] for d in panel)}")
    sd=defaultdict(lambda:[0,0])
    for d in panel: sd[d['state']][0 if d['split']=='train' else 1]+=1
    print("per-state (train,test):")
    print({k:tuple(v) for k,v in sorted(sd.items())})
    print("\nTEST projects:")
    for d in sorted(panel, key=lambda d:(d['region'],d['state'],d['name'])):
        if d['split']=='test':
            print(f"  {d['name'][:34]:34s} {d['state']} {d['year']} n_hi={d['n_hi']:3d} {d['region']}")
    vb=defaultdict(lambda:[0,0])
    for d in panel: vb[vintage_bin(d['year'])][0 if d['split']=='train' else 1]+=1
    print("\nvintage (train,test):", {k:tuple(v) for k,v in sorted(vb.items())})

if __name__=="__main__":
    main()
