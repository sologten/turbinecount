# TurbineCount

**What actually works at 10 m? An honest study of static, texture, and blade-motion cues for wind-turbine detection in Sentinel-2.**

TurbineCount asks a narrow, practical question: at Sentinel-2's 10 m ground sample distance, which computer-vision cues actually separate operating wind turbines from background, and which only appear to? We compare four cue families head to head on a clean, leakage-free, project-grouped benchmark, and we introduce a novel **band-parallax (blade-motion)** cue that reads the faint multi-band fringe a *spinning* blade leaves in Sentinel-2 imagery.

The honest one-line takeaway: this is not a turbine detector, it is a **turbine-rotation detector**. It measures whether a turbine was spinning at the instant of capture, which is a proxy for whether it was generating.

## Headline results

On 33 held-out wind projects (never seen during development), single locked evaluation:

| Detector | pooled AUC | per-project AUC | AP |
|---|---|---|---|
| CS131 primitives only (from-scratch fallback) | 0.787 | 0.792 | 0.777 |
| HOG + linear SVM (appearance baseline) | 0.933 | 0.938 | 0.928 |
| Static salience (brightness z-score + CS131) | 0.956 | 0.948 | 0.954 |
| Band-parallax / motion only (**novel**) | 0.963 | 0.952 | 0.946 |
| **Fused random forest (parallax + salience)** | **0.984** | **0.971** | **0.985** |

The motion cue, on its own, beats both appearance baselines and is the most salience-robust family on the bright, textured terrain where appearance fails (per-project AUC 0.928 vs 0.909 for static).

**The honest gap.** Those are classification AUCs with random-background negatives. Run as a real detector (dense sliding window, NMS, frozen threshold), the same model scores micro **F1 0.39** with count **R-squared 0.13**. That is the apples-to-apples object-level number, and it is far below the classification AUC. Idle turbines are invisible to the cue by design, which caps recall against installed-count ground truth.

## Why it matters

Static maps already tell you what wind is *installed*. Almost nothing gives you a free, repeatable read on what is *actually turning*. That operational signal feeds grid balancing, curtailment and outage detection, and energy / carbon analysis. A concrete, fast-growing case is **data-center investment**: pairing which turbines near a project are spinning with their nameplate capacity lets an investor estimate the operational wind share feeding a project's grid region or contracted portfolio, and independently stress-test its clean-power claims.

## Repository layout

```
cv131/              From-scratch CS131 primitives (no OpenCV / skimage):
                    convolution, Gaussian, Sobel, Harris, Hough, template matching.
features_s2.py      The four feature families, incl. the 19 band-parallax features.
gt.py, harness.py   Ground-truth helpers and the leakage-free GroupKFold evaluation.
select_panel.py     Frozen 173-project panel + project-grouped split (metadata only).
fetch_s2.py         Resumable Sentinel-2 L2A patch fetch (AWS Earth Search).
fetch_naip.py       NAIP 0.6 m tile fetch (Microsoft Planetary Computer).
build_matrix.py     Featurize patches -> artifacts/s2_matrix.npz.
sweep_s2.py         Five classifiers x feature subsets, GroupKFold CV (train only).
mechanism_audit.py  Salience stratification + literature baseline + audit.
disentangle.py      Motion-vs-contrast controls (orthogonalization, within-bin AUC).
naip_track.py       NAIP 0.6 m high-resolution benchmark.
final_test.py       The single locked-test evaluation.
detect.py           Detection-mode benchmark (sliding window + NMS, object F1, count R^2).
ablate.py           Rigorous per-feature / per-family ablations -> artifacts/ablations/.
operational_signal.py  Tests whether the cue behaves like an operational (motion) signal.
casestudy.py        Re-fetch real chips for win/fail case studies.
render_casestudy.py, make_paper_figs.py, fig_ablations.py   Figure generation.

artifacts/          Cached results: feature matrices (.npz), frozen models (.joblib),
                    result JSON/CSV, and all figures. The held-out test split is read
                    only by final_test.py.
report/             4-page CVPR write-up (main.tex, cvpr.sty, main.pdf, figs/).
slides/             Demo-day deck (.pptx + .pdf).
docs/               Plain-English explainer, full analysis memo, verified related work,
                    and RESULTS_v2.md.
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Reproduce (from cached artifacts, no network)

These run directly against the cached feature matrix and frozen model:

```bash
python ablate.py all          # per-primitive + per-family ablations, traces -> artifacts/ablations/
python final_test.py          # reproduces the headline table above (to 3 decimals)
python operational_signal.py  # rotation-signal consistency tests + figure
python make_paper_figs.py     # report figures
python fig_ablations.py       # ablation summary figure
```

The data-collection scripts (`select_panel.py`, `fetch_s2.py`, `fetch_naip.py`, `build_matrix.py`, `sweep_s2.py`, `detect.py`) were run in the original environment and need network access plus a local USWTDB dump. Point `TC_WORK` at a directory holding `cache/uswtdb/all_onshore.json` to re-run them; the cached outputs in `artifacts/` let you skip this for everything above.

## Honest limitations

- This is candidate-point classification, not a deployed counter; the real detection number is F1 0.39.
- The parallax cue fires only on spinning turbines, so recall against installed-turbine ground truth has a hard ceiling.
- A single satellite pass is one instant; estimating utilization needs many passes.
- Negatives are random background (mostly cropland), not hard confusers (buildings, towers), so the AUCs are optimistic relative to a real scan.

The cleanest next experiment, using only free Sentinel-2: image the same turbines on two dates. If a farm's fringe flips on and off across dates while its brightness stays constant, that demonstrates the cue tracks motion (operation), not appearance.

## Data sources

- Ground truth: USGS U.S. Wind Turbine Database (USWTDB).
- Imagery: Sentinel-2 L2A via AWS Earth Search; NAIP via the Microsoft Planetary Computer.

## License

MIT. See `LICENSE`.
