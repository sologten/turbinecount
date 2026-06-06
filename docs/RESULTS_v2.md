# TurbineCount v2: a band-parallax (blade-motion) wind-turbine detector for Sentinel-2

A follow-on to the original CS131 TurbineCount project. The original project's
honest headline was a *limitation*: at Sentinel-2's 10 m ground sample distance,
turbines on bright or textured terrain are spatially resolved but not locally
salient, so an appearance-based detector fails on most scenes, and only 2 of 7
panel scenes passed a salience screen. That collapsed the split to train-on-1 /
test-on-1 scene (illustrative detection AP 0.125, per-pixel AUC 0.871) and could
only be rescued with 0.3 m NAIP, which is flown every 2 to 3 years and cannot
nowcast capacity.

This v2 attacks that limitation directly. It (1) builds a clean, large,
leakage-free dataset, (2) introduces a genuinely different cue that does not
depend on static local salience, and (3) evaluates everything with grouped
cross-validation and a single locked-test pass.

All numbers below were produced by the scripts in this folder on real data
(USWTDB ground truth, Sentinel-2 L2A from AWS Earth Search, NAIP from the
Microsoft Planetary Computer). The held-out test set was scored exactly once.

## Headline

On 33 held-out wind projects (never seen during any development), spanning the
full national panel of 35 states:

| Detector | pooled AUC | per-project AUC | AP |
|---|---|---|---|
| CS131 primitives only (the documented fallback) | 0.787 | 0.792 | 0.777 |
| Static salience (z-score + CS131) | 0.956 | 0.948 | 0.954 |
| HOG + linear SVM (generic appearance baseline) | 0.933 | 0.938 | 0.928 |
| **Band-parallax / motion only (novel)** | **0.963** | **0.952** | 0.946 |
| **Headline: random forest, parallax + static** | **0.984** | **0.971** | **0.985** |

Headline per-project AUC 95% project-bootstrap CI: [0.928, 0.995], n = 33 test
projects. The novel motion cue, on its own, beats both the static salience model
and a standard HOG+SVM appearance detector, and it lifts the combined detector
to 0.984 pooled AUC. This is a far more robust result than the original
train-1 / test-1 setup, and it holds out-of-sample across the whole country.

## What is new: the band-parallax cue

Sentinel-2's MSI records spectral bands at slightly different times because the
detectors are physically offset on the focal plane. Published inter-band delays
are about 0.324 s (green B03 after blue B02) and 1.005 s (red B04 after blue
B02), up to 2.6 s across all bands. A spinning blade tip on a 100 to 130 m rotor
travels tens of meters in that window, so an *operating* turbine appears at
displaced pixels across the visible bands, leaving a localized multi-band fringe.
Standard L1C/L2A processing geo-rectifies each band to a common grid but does not
motion-correct, so the fringe survives in the product. This is the same physics
that paints moving ships and aircraft in rainbow colors in Sentinel-2 imagery.

The detector turns this into 19 features (file `features_s2.py`, prefix `plx_`):
per-band high-pass and contrast normalization, then band-difference fringe maps
(red-blue, green-blue, red-green) summarized by peak, peak-to-peak, energy, and
dipole geometry; a chromatic-dispersion map; a blue-to-red cross-correlation
offset; and a time-ordering consistency term (the red-blue fringe should align
with, and be larger than, the green-blue fringe in proportion to the time gap).

Closest prior art is "Detection of wind turbine motion between satellite bands
with CNNs"; the oil-storage-tank-from-shadow literature supplied the
methodological template of turning a geometric side effect (shadow length there,
inter-band displacement here) into the discriminative measurement. The
contribution here is a from-scratch, interpretable feature implementation of the
cue, a controlled comparison against appearance baselines, and an explicit test
of whether the cue is motion or merely re-encoded contrast.

## Dataset and split (clean, frozen, leakage-free)

Ground truth is the USGS USWTDB (high-confidence locations only, t_conf_loc >= 2).
The panel is 173 projects selected on metadata alone, before any imagery was
seen, spanning all 35 USWTDB wind states, commissioned 2008 to 2023 so every
turbine is operational in the 2024 imagery (the operational-only label the
analysis requires). The split is grouped by whole project (no farm is ever split),
stratified by census region and vintage, seed-locked: 139 train / 34 test
projects (33 yielded Sentinel-2 patches). Cross-validation on train uses
GroupKFold with group = project, so CV folds also never split a farm. The test
projects span every region and all three vintage bins.

The Sentinel-2 dataset is 171 projects, 10,634 patches (5,339 turbine-centered
positives, 5,295 background negatives >= 500 m from any turbine), each a 48x48
R/G/B/NIR chip at 10 m. See `select_panel.py`, `fetch_s2.py`, `build_matrix.py`.

## Methods tried, with cross-validation and tuning

Feature families (by prefix): `stat_` salience statistics (6), `cv131_` the
from-scratch CS131 primitives Gaussian/Sobel/Hough/Harris/template (4), `plx_`
band-parallax (19), `hog_` a compact HOG descriptor (96). Five classifiers were
swept with small CV-tuned grids (logistic regression, RBF SVM, random forest,
XGBoost, MLP) across feature subsets, all on train only with GroupKFold(5). The
sweep was run by parallel sub-agents and the chosen model was independently
re-cross-validated and reproduced to four decimals before any test use.

GroupKFold(5) CV AUC on train (pooled), abridged:

| family | logistic | best of 5 models |
|---|---|---|
| CS131 primitives only | 0.782 | 0.782 |
| static (stat + cv131) | 0.943 | 0.960 (XGB) |
| HOG | 0.934 | 0.934 |
| parallax only | 0.952 | 0.960 (RF) |
| parallax + static | 0.973 | 0.979 (XGB pooled) / 0.967 per-proj (RF) |
| all families | 0.974 | 0.980 (XGB) |

Selected headline by per-project CV AUC: random forest on parallax + static
(per-project CV 0.967, pooled 0.979). Full table in `artifacts/sweep_s2.json`.

## Is the parallax cue real motion or re-encoded contrast? (the honest core)

The most important risk is that the band-difference features simply re-encode
local contrast: a bright high-contrast turbine produces a large band-difference
even if its blades are not moving, and `plx_rb_peak` does correlate with the
static salience cue (r about 0.4 to 0.47). Two tests addressed this, both on
train with GroupKFold (`disentangle.py`):

1. Orthogonalization. Inside each fold, every parallax feature was linearly
   regressed on the static features and only the residual was kept. The residual
   parallax still scores per-project AUC 0.763 (pooled 0.777). Most of the
   parallax signal is orthogonal to static contrast.

2. Within-contrast-bin AUC. Binning all patches into deciles of the static
   salience cue (so contrast is held roughly constant inside a bin), the best
   parallax feature separates turbines from background at mean within-bin AUC
   0.759 (0.82 to 0.90 in the mid-contrast bins), while the static cue itself
   falls to 0.527, essentially chance, within bins.

Both show the parallax family carries information beyond local contrast,
consistent with genuine inter-band motion. Permutation importance ranks
`plx_rb_peak`, `plx_chroma_disp_peak`, and `plx_rb_ptp` highest, exactly the
band-difference and chromatic-dispersion terms the physics predicts.

A label-shuffle control collapses CV AUC to 0.490, confirming no leakage.

## Where each cue wins (locked test, salience-stratified)

Splitting the 33 test projects at the median appearance-salience:

| regime | parallax | static | HOG |
|---|---|---|---|
| low-salience (17 projects, bright/textured terrain) | **0.928** | 0.909 | 0.899 |
| high-salience (16 projects) | 0.977 | **0.990** | 0.980 |

On exactly the bright-terrain regime that defeated the original detector, the
motion cue is the most robust single family and beats the appearance baseline.
On easy high-salience scenes, static appearance is best and parallax is
marginally behind. The two are complementary, which is why the fused model wins
overall. The low-salience margin is real but modest (+0.02 over static, +0.03
over HOG), and is reported as "more salience-robust," not "detects where static
collapses to chance."

## NAIP 0.6 m secondary benchmark

To confirm the static track also generalizes at high resolution beyond the
original 2-scene experiment, the from-scratch CS131 static features were run on
39 NAIP projects (29 train / 10 test, 23 states). Held-out test: pooled AUC
0.990, per-project AUC 1.000 (CV per-project 0.944). At 0.6 m the static cues
(Harris, matched-filter template) are individually excellent and HOG does not
help. See `naip_track.py`.

## What failed or is limited (read this)

- The original CS131 primitives alone remain weak at 10 m (test AUC 0.787). They
  are the documented fallback and they behave as a fallback. The strength comes
  from the salience z-score and, above all, from the new parallax features.
- The whole-patch inter-band cross-correlation offset (`plx_xcorr_offset`) was
  essentially useless: it is about 0 for both classes because the static ground
  dominates the alignment. The signal lives in the *localized* fringe energy, not
  in a global shift. This feature was kept for completeness but carries almost no
  weight.
- HOG added nothing on NAIP and little on Sentinel-2 beyond the engineered cues.
- The negative pool is random background (mostly cropland), not hard negatives
  such as buildings. The fused AUCs are therefore optimistic relative to a real
  deployment whose false positives are bright rooftops. The parallax cue is
  *expected* to resist building confusers because static buildings produce no
  blade fringe, but with the current negatives this is a hypothesis, not a
  demonstrated result.
- The parallax cue only fires on *spinning* turbines. Operational turbines that
  were idle at image time (low wind, curtailment, maintenance) show no fringe, so
  parallax recall against the installed-turbine ground truth has a ceiling below
  1.0. For a generation nowcast this is arguably a feature (it measures what is
  actually turning), but against USWTDB it is a recall limit.
- NAIP could not be pulled at full panel scale in this environment because COG
  read latency frequently exceeded the available compute window; the NAIP track
  is therefore 39 projects, not 173. The Sentinel-2 parallax track is the
  primary, fully-scaled benchmark.
- Detection here is patch-level classification of a candidate point, matching the
  proposal's task framing. It is not a full dense-scan detector with
  non-max-suppression; absolute precision in a sliding-window deployment would
  differ.

## Bearing on the alt-data thesis

The parallax detector uses only free Sentinel-2 at its native 5-day cadence, so
unlike the NAIP rescue it can in principle nowcast EIA Form 860. It is the most
salience-robust cue on bright terrain, the regime where an appearance-only
Sentinel-2 nowcast is biased, and it keys on rotation, which is a direct proxy
for whether a turbine is generating. The honest framing is that it narrows, not
erases, the bright-terrain gap the original project identified, and it does so
with an interpretable, from-scratch cue rather than a black-box model.

## Reproduce

```
python select_panel.py        # frozen 173-project panel + split (metadata only)
python fetch_s2.py 40 8        # resumable Sentinel-2 patch pull (cached)
python fetch_naip.py 36 12     # resumable NAIP tile pull (subset, cached)
python build_matrix.py         # featurize -> artifacts/s2_matrix.npz
python sweep_s2.py             # diverse classifiers, GroupKFold CV, train only
python mechanism_audit.py      # salience stratification + literature baseline + audit
python disentangle.py          # motion-vs-contrast tests
python naip_track.py           # NAIP high-res benchmark
python final_test.py           # the single locked-test evaluation
```

Models and result JSONs are cached in `artifacts/`. The held-out test split is
read only by `final_test.py`.
