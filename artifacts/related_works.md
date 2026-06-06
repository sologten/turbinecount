# Related Works: Satellite-Based Wind Turbine Detection — Literature Review and Comparison

**Scope.** An objective map of the satellite wind-turbine detection field, and an honest assessment of where our project ("ours") stands. Our project is **candidate-point classification** on **Sentinel-2 L2A 10 m** optical imagery (onshore US): given a (lat, lon), output P(operational turbine), using an engineered **band-parallax / blade-motion** cue (19 interpretable features) fused with CS131 static primitives and a salience z-score via a random forest. Ground truth: USGS USWTDB; 173 US projects, 35 states; project-grouped train/test split (139 train / 33 test); operational-only labels. Held-out single locked eval: **per-project ROC-AUC 0.971** (pooled 0.984, AP 0.985) for the fused model; parallax-only 0.952; static salience 0.948; HOG+linear-SVM baseline 0.938; old CS131-primitives-only 0.787. NAIP 0.6 m static benchmark test AUC 0.99.

> **Caveat carried throughout.** Our headline metric is ROC-AUC / AP on a *candidate-point classification* task where negatives are **random background** (not buildings/towers/roads). This is fundamentally *easier* than full-scene object detection, where false positives scale with image area and hard negatives dominate. ROC-AUC on this setup is **not numerically comparable** to object-level mAP/F1 or to count-level r². Every comparison below respects that.
>
> *Sourcing: all numbers below were read directly from the primary papers' results sections/tables (PDFs fetched in-session). Each numeric claim carries a URL.*

---

## 1. Catalog of Key Related Works

### Bucket A — Global / continental onshore mapping (the "product" comparisons)

#### A1. Microsoft Global Renewables Watch (GRW)
- **Citation:** Robinson, Ortiz, Kim, Dodhia, Zolli, Nagaraju, Oakleaf, Kiesecker, Lavista Ferres. *Global Renewables Watch: A Temporal Dataset of Solar and Wind Energy Derived from Satellite Imagery.* arXiv:2503.14860, 2025 (Microsoft AI for Good / The Nature Conservancy / Planet Labs). https://arxiv.org/abs/2503.14860
- **Imagery + GSD:** PlanetScope quarterly RGB visual basemaps, **4.7 m/px** at equator (zoom 15), Q4 2017–Q2 2024. https://arxiv.org/html/2503.14860v1
- **Task:** Point-based **semantic segmentation → counting** (wind). Wind model = **FCN + ResNet-50 backbone**, localization-based counting (LC) loss with four terms (patch/point/split/false-positive); MOSAIKS-SVM false-positive filter. Solar = U-Net + ResNeXt-50. https://arxiv.org/html/2503.14860v1
- **Dataset & region:** Global; wind labels from Dunnett et al. OSM points (272,503 → 17,971 tiles, 80/10/10 split); deployed on ~13.98 trillion pixels. Final product: **375,197 wind turbines**, 86,410 solar PV installations. https://arxiv.org/html/2503.14860v1
- **EXACT metrics (Table 1, wind, OBJECT level):**
  - No filtering: **precision 59.63%, recall 71.48%, F2 68.75%**.
  - Global model + filtering + hard negatives: **precision 90.81%, recall 81.63%, F2 83.31%**. (No pixel-level reported for wind.) https://arxiv.org/html/2503.14860v1
  - Solar (context): best pixel **F2 74.48%**; object precision 23.97% → **50.93%** after FP filter. https://arxiv.org/html/2503.14860v1
  - **Country-level capacity vs IRENA 2023 (Table 2):** their onshore wind **Pearson r² = 0.932**, Kendall τ = 0.877; (Satlas onshore wind r² = 0.978); their solar PV r² = 0.960. Abstract rounds to 0.96/0.93. https://arxiv.org/html/2503.14860v1
- **One-liner:** Data-centric OSM-label cleaning + FCN point segmentation + hard-negative mining + MOSAIKS-SVM FP filter, run globally/temporally.
- **Strengths:** True global object-level product; honest object precision/recall/F2; IRENA-validated; temporal (construction dates). The benchmark for onshore wind mapping at scale.
- **Limitations:** 4.7 m basemaps miss small/old turbines; object metrics rest on noisy OSM labels (authors note they *underestimate* true performance); static appearance only, no motion cue.

#### A2. Satlas renewable-energy layer (baseline compared inside GRW)
- **Citation:** Bastani, Wolters, Gupta, Ferdinando, Kembhavi. *SatlasPretrain.* ICCV 2023, 16772–16782 (as compared in GRW Table 2).
- **Metric:** Onshore wind country-level **r² = 0.978** (all-wind 0.975) vs IRENA 2023 — the **best count-level wind agreement** in the literature. https://arxiv.org/html/2503.14860v1
- **Note:** Strongest count r² for wind; per-object detection metrics not reported in GRW.

### Bucket B — Sentinel-2 a contrario shadow/hub detectors (Mandroux) — most directly comparable on imagery/GSD/region

#### B1. Mandroux et al., IPOL 2022 (single-date)
- **Citation:** N. Mandroux, T. Dagobert, S. Drouyer, R. Grompone von Gioi. *Single Date Wind Turbine Detection on Sentinel-2 Optical Images.* Image Processing On Line, 12 (2022), 198–217. https://doi.org/10.5201/ipol.2022.384
- **Imagery + GSD:** Sentinel-2, **10 m**, **blue band B02 only** — deliberately, "to avoid blurring effects due to rotating blades." https://www.ipol.im/pub/art/2022/384/article.pdf
- **Task:** Single-scene object **detection** via a contrario (NFA): dark-shadow detector + bright-hub detector fused; tower height assumed 80 m. https://www.ipol.im/pub/art/2022/384/article.pdf
- **Dataset & region:** ~300 Sentinel-2 chips with a centered turbine + ~300 without; detect/no-detect in the central square. https://www.ipol.im/pub/art/2022/384/article.pdf
- **EXACT metrics:** Parameter sweeps (Fig. 6) peak at **F1 ≈ 0.9** at (t_shadow 25, t_hub 50, t_NFA 1), with precision/recall curves across thresholds. https://www.ipol.im/pub/art/2022/384/article.pdf
- **One-liner:** Statistical (NFA) fusion of a shadow detector and a hub detector on the blue band only.
- **Strengths:** Same imagery/GSD/region scale as ours; interpretable; principled false-alarm control; open source.
- **Limitations:** Static only; **explicitly discards blade motion** by using B02 only — the exact gap our parallax cue fills. Shadow detector fooled by roads/crops/relief; hub detector very loose (1.6% of all pixels fire); small hand-built test set.

#### B2. Mandroux et al., ISPRS Annals 2022 (multi-date)
- **Citation:** N. Mandroux, S. Drouyer, R. Grompone von Gioi. *Multi-Date Wind Turbine Detection on Optical Satellite Images.* ISPRS Annals V-2-2022, 383–390 (XXIV ISPRS Congress, Nice). https://doi.org/10.5194/isprs-annals-V-2-2022-383-2022
- **Imagery + GSD:** Sentinel-2 **10 m**, B02 only, 4 dates of the same scene (temporal coherence). https://isprs-annals.copernicus.org/articles/V-2-2022/383/2022/isprs-annals-V-2-2022-383-2022.pdf
- **Task:** Multi-date object **detection** (temporal aggregation of shadow/hub a contrario scores).
- **EXACT metrics:**
  - Controlled scene (40 turbines): **mean Average Precision = 0.939** (PR curve); ~**75% of turbines found with zero false positives**; ~**80% detected at 0.01 false positives / km²**. https://isprs-annals.copernicus.org/articles/V-2-2022/383/2022/isprs-annals-V-2-2022-383-2022.pdf
  - Large-scale 10,000 km² (realistic): ε=1 → ~**60%** detected; ε=0.1 → ~**45%**; ε=0.01 → ~**40%**; observed FP ran ~3–4× the nominal NFA budget per 1000×1000 tile. https://isprs-annals.copernicus.org/articles/V-2-2022/383/2022/isprs-annals-V-2-2022-383-2022.pdf
  - Authors' DL note: deep methods reach **AP 0.9–0.98 but on 0.5–2 m imagery** (Zhang et al. 2020) vs their 10 m. https://isprs-annals.copernicus.org/articles/V-2-2022/383/2022/isprs-annals-V-2-2022-383-2022.pdf
- **Strengths:** Best apples-to-apples published Sentinel-2 10 m onshore detector with a real object-level mAP (0.939) and a realistic large-scale false-alarm rate; honest that ROC is misleading on sparse-positive pixel grids.
- **Limitations:** Static; large-scale recall collapses to 40–60%; mAP 0.939 is on a *favorable* hand-picked scene (authors say so).

### Bucket C — Band-motion / parallax (closest conceptual neighbor to ours)

#### C1. Nahrstedt, Gärtner, Wittmann — band-motion CNN
- **Citation:** F. Nahrstedt, P. Gärtner, J. Wittmann. *Detection of wind turbine motion between satellite bands with convolutional neural networks.* EnviroInfo 2023, Lecture Notes in Informatics, Gesellschaft für Informatik, Bonn. https://dl.gi.de/bitstreams/3fb4a95e-8c19-4d2c-9652-d51770859c27/download
- **Imagery + GSD:** Sentinel-2 **10 m**, bands B02/B03/B04, exploiting the staggered acquisition (**max offset 1.005 s**) — the same physics our parallax cue uses; theoretical max blade displacement ~45 m (4.5 px) for a 90 m blade. https://dl.gi.de/bitstreams/3fb4a95e-8c19-4d2c-9652-d51770859c27/download
- **Task:** Binary **classification** — spinning turbine vs non-turbine — with a pretrained **DenseNet-121** on 40×40×3 RGB chips. Labels from 4,800 Enercon turbines (Germany); **2,996 spinning / 110 standing / 1,662 undetectable** shadows. https://dl.gi.de/bitstreams/3fb4a95e-8c19-4d2c-9652-d51770859c27/download
- **EXACT metrics:** **~99.4% validation accuracy** (best DenseNet-121 config); training CE loss stagnated ~0.24 nats. **No precision/recall/AP/F1; no held-out cross-region test; no grouped split; near-degenerate negative class (only 110 standing).** Plus an occlusion-sensitivity analysis showing the model attends to the turbine. https://dl.gi.de/bitstreams/3fb4a95e-8c19-4d2c-9652-d51770859c27/download
- **One-liner:** Treat the inter-band time offset as motion; train a CNN to recognize the spinning-blade band signature.
- **Strengths:** Same physical cue as ours; demonstrates the cue is learnable; free imagery; the only prior motion-based wind detector.
- **Limitations:** Essentially **qualitative / single-accuracy**; ~99.4% is in-distribution validation with a degenerate class balance, no AUC/F1/AP, no FP analysis. Our project is the more rigorous quantification of the same cue (locked test, GroupKFold, residual-orthogonalization controls, 19 interpretable features vs a black box).

#### C2. Methodological cousin — Sentinel-2 MSI parallax for moving objects
- **Citation:** Liu et al., *Space eye on flying aircraft: From Sentinel-2 MSI parallax to hybrid computing.* Remote Sensing of Environment 246:111867, 2020; Fisser et al., *Detecting Moving Trucks on Roads Using Sentinel-2 Data*, Remote Sensing 14(7):1595, 2022 (both cited by C1).
- **Relevance:** Establish the same inter-band parallax physics (staggered band acquisition → moving-object displacement) for aircraft/trucks. Different target, not a detection-performance comparator.

### Bucket D — Offshore Sentinel-2 / SAR detectors

#### D1. Seasonally-robust offshore (Geowise-Net / Contrast-Net / Composite)
- **Citation:** *Seasonally Robust Offshore Wind Turbine Detection in Sentinel-2 Imagery Using Imaging Geometry-Aware Deep Learning.* Remote Sensing 17(14):2482, 2025. https://doi.org/10.3390/rs17142482
- **Imagery + GSD:** Sentinel-2 **10 m**, offshore China; 887 scenes (<5% cloud), Jan 2023–Mar 2025; 7,369 OWTs detected.
- **Task:** Object **detection** (Faster R-CNN baseline + geometry-aware variants: Geowise-Net injects solar/view angles; Contrast-Net uses contrastive learning on seasonal pairs; Composite fuses both).
- **EXACT metrics (single-date F1):** Faster R-CNN **0.947** < Geowise-Net **0.958** < Contrast-Net **0.961** < Composite **0.966**; summer–winter gap reduced 3.7% → 0.8%. https://doi.org/10.3390/rs17142482
- **Strengths:** Sentinel-2 10 m object detection with F1; explicitly models imaging geometry (conceptually adjacent to our time-ordering/parallax features).
- **Limitations:** **Offshore** (water = near-trivial background vs onshore confusers); not transferable to onshore clutter.

#### D2. DeepOWT (Sentinel-1 SAR, global offshore)
- **Citation:** T. Hoeser, S. Feuerstein, C. Kuenzer. *DeepOWT: a global offshore wind turbine data set derived with deep learning from Sentinel-1 data.* Earth System Science Data 14:4251–4270, 2022. https://essd.copernicus.org/articles/14/4251/2022/
- **Imagery + GSD:** Sentinel-1 SAR (~10 m), global ocean; **cascade of two ResNet-50 Faster R-CNN detectors** (farm stage → turbine stage); **9,941 offshore infrastructure locations**. https://essd.copernicus.org/articles/14/4251/2022/
- **Task:** Object **detection** + global dataset; TP = predicted point within a 100 m-radius ground-truth polygon; reports class-wise **precision–recall curves and AP** on 2021Q2 North-Sea-Basin and East-China-Sea test sets. https://essd.copernicus.org/articles/14/4251/2022/
- **Related Sentinel-1 result (Shandong, ResNet34-SSD):** **precision 96.58%, recall 91.59%, F1 94.02%** (Zhang et al., J. Cleaner Production 2023); another S1 model reports **P/R = 95.97% / 91.18%**. https://www.sciencedirect.com/science/article/abs/pii/S0959652623045730
- **Strengths:** Global, all-weather (SAR), object-level PR/AP; strong precision/recall on water.
- **Limitations:** Offshore SAR (water clutter ≪ onshore optical clutter); not comparable to onshore optical classification.

### Bucket E — Onshore high-resolution object detectors (YOLO / Faster-RCNN on sub-meter imagery)

#### E1. WT-YOLO
- **Citation:** Zhai, Chen, et al. *Identifying wind turbines from multiresolution and multibackground remote sensing imagery.* Int. J. Applied Earth Observation & Geoinformation 122:103613, 2023. https://www.sciencedirect.com/science/article/pii/S1569843223004375
- **Imagery + GSD:** Google Earth high-res, **0.6–5.4 m**, multi-background dataset; YOLOv5-based.
- **Task:** Object **detection** + positioning. **EXACT metric:** AP **5.92%–15.43% higher** than prior wind-turbine detectors across 0.6–5.4 m (relative improvement; code github.com/zyyyccc/WT-YOLO). https://www.sciencedirect.com/science/article/pii/S1569843223004375
- **Strengths:** Onshore, multi-resolution, real object detection with AP. **Limitations:** Needs sub-meter imagery (expensive, not free-global); static appearance, no motion cue. This is the regime our **NAIP 0.6 m static benchmark (test AUC 0.99)** sits in.

#### E2. CGA-YOLO (Gaofen-2, ~1 m, Shandong)
- **Citation:** *Wind Turbines Small Object Detection in Remote Sensing Images Based on CGA-YOLO.* Remote Sensing 18(2):324. https://doi.org/10.3390/rs18020324 — onshore high-res small-object detection (SDWT Gaofen-2 dataset); improved recall/precision over YOLO baselines. Same regime as E1.

### Bucket F — Methodological analog: oil-storage-tank-from-shadow

- **YOLOv3 (optical, floating-head tanks + shadow-volume):** test **AP 0.84** (train 0.942). https://towardsdatascience.com/oil-storage-tanks-volume-occupancy-on-satellite-imagery-using-yolov3-3cf251362d9d/
- **YOLOX-TR (dense oil tanks, SAR):** **mAP@0.5 = 94.8%**, mAP = 60.8%. https://www.mdpi.com/2072-4292/14/14/3246/htm
- **Relevance:** Like turbines, oil tanks are detected partly via **shadow geometry** (sun-angle reasoning to recover height/volume) — the canonical "shadow-as-signal" analog to our shadow/hub/parallax features; shows engineered shadow cues + a detector reach AP ~0.84–0.95 object-level.

### Ground truth used by our project
- **USGS USWTDB (U.S. Wind Turbine Database):** standard high-completeness U.S. onshore registry; appropriate gold standard for our labels. https://eerscmap.usgs.gov/uswtdb/

---

## 2. Comparability Matrix

Legend: ✅ directly comparable to ours; ⚠️ partial; ❌ not comparable.

| Work | Imagery / GSD | Task | Region | Headline metric (number) | Same task? | Same metric? | Same GSD? | Same region? | Directly comparable? |
|---|---|---|---|---|---|---|---|---|---|
| **Ours** | S2 / 10 m | **classification** (cand. pt, easy negs) | Onshore US | per-proj ROC-AUC **0.971**; AP 0.985 | — | — | — | — | — |
| **GRW (A1)** | PlanetScope / 4.7 m | seg→count | Global onshore | obj **P 90.8 / R 81.6 / F2 83.3**; r² **0.932** | ❌ | ❌ (F2/r² ≠ AUC) | ❌ | ⚠️ (incl. US) | ❌ — different task & metric |
| **Satlas (A2)** | high-res / global | count | Global | r² **0.978** vs IRENA | ❌ | ❌ | ❌ | ⚠️ | ❌ — count-level only |
| **Mandroux single (B1)** | S2 / 10 m, B02 only | detection | Onshore | **F1 ≈ 0.9** (chip detect) | ⚠️ | ⚠️ (F1 vs AUC) | ✅ | ⚠️ | ⚠️ — closest on imagery; metric differs; discards motion cue |
| **Mandroux multi (B2)** | S2 / 10 m, B02 only | detection | Onshore | **mAP 0.939**; ~80% @0.01 FP/km²; 40–60% large-scale | ⚠️ | ⚠️ (mAP) | ✅ | ⚠️ | ⚠️ — best S2 10 m object-level reference |
| **Nahrstedt CNN (C1)** | S2 / 10 m | classification (spin) | Germany | **~99.4% val acc** (no P/R/F1/AUC) | ✅ (same cue+task) | ⚠️ (acc, no test) | ✅ | ⚠️ | ⚠️ — same cue, only qualitative/in-dist accuracy |
| **Geowise/Contrast (D1)** | S2 / 10 m | detection | **Offshore** | **F1 0.947–0.966** | ⚠️ | ⚠️ | ✅ | ❌ | ❌ — water trivializes negatives |
| **DeepOWT (D2)** | S1 SAR / 10 m | detection | **Offshore** | PR/AP curves; related S1 **F1 94.0** | ❌ (SAR) | ⚠️ | ✅ | ❌ | ❌ — SAR + offshore |
| **WT-YOLO (E1)** | GE / 0.6–5.4 m | detection | Onshore | AP **+5.9–15.4%** vs priors | ⚠️ | ⚠️ | ❌ (sub-m) | ⚠️ | ⚠️ — matches our **NAIP** regime, not S2 |
| **CGA-YOLO (E2)** | Gaofen-2 / ~1 m | detection | Onshore | improved P/R vs YOLO | ⚠️ | ⚠️ | ❌ | ❌ | ❌ — high-res only |
| **Oil tanks (F)** | optical/SAR / sub-m | detection | n/a | AP **0.84**; mAP@.5 **94.8** | ⚠️ (analog) | ⚠️ | ❌ | ❌ | ⚠️ — methodological analog only |

**Why ROC-AUC here is not comparable to mAP/F1/r².** Our negatives are random background, so the classifier separates "turbine vs empty field," not "turbine vs tower/road/bright-building/crop-shadow." Object detectors (GRW, Mandroux, Geowise, WT-YOLO) and the oil-tank analog are penalized by exactly those hard negatives, and their false positives scale with imagery area — which is why GRW's *object* precision is 90.8% (not ~98%) and Mandroux's large-scale recall collapses to 40–60%. ROC-AUC is also inflated on sparse-positive grids — Mandroux explicitly warn their near-perfect ROC "does not give a correct vision of absolute performance." Count-level r² (Satlas 0.978, GRW 0.932) measures aggregate capacity agreement, a different question from per-instance correctness.

---

## 3. Honest Verdict

**Who is genuine SOTA, by regime:**
- **Global onshore object-level product:** **Microsoft GRW (A1)** — object precision 90.8% / recall 81.6% / F2 83.3% at 4.7 m, IRENA-validated. The reference for "find and count onshore turbines worldwide."
- **Count-level capacity agreement:** **Satlas (A2)**, onshore wind r² = 0.978, narrowly ahead of GRW's 0.932.
- **Sentinel-2 10 m onshore object detection (interpretable, classical):** **Mandroux multi-date (B2)** — mAP 0.939 on a favorable scene; the honest large-scale numbers (40–60% recall, thousands of FP/country) show how hard the real onshore task is at 10 m.
- **Offshore detection:** **Geowise/Contrast-Net (D1)** for Sentinel-2 optical (F1 up to 0.966) and **DeepOWT (D2)** for global SAR — but on water, a categorically easier background.
- **High-res onshore detection:** **WT-YOLO (E1)** — the regime our NAIP 0.6 m benchmark (test AUC 0.99) lives in.

**Where our approach is genuinely novel.** The **band-parallax / blade-motion cue as an engineered, interpretable signal** is novel relative to every static detector. Mandroux et al. **explicitly throw this signal away** — both papers use the blue band B02 *only*, deliberately, "to avoid blurring effects due to rotating blades." We do the opposite: we treat the across-band displacement of the spinning blade (green +0.324 s, red +1.005 s after blue) as *the feature*, engineering 19 interpretable parallax descriptors. The only prior work using the same physics is **Nahrstedt et al. (C1)** — a black-box DenseNet reporting a single ~99.4% in-distribution validation accuracy with no AUC/F1/AP, no grouped split, and a degenerate 110-sample negative class. Our contribution over C1 is **rigor and interpretability**: a project-grouped frozen split, GroupKFold CV, a single locked eval, residual-after-orthogonalization controls (parallax AUC **0.76** after removing contrast; within-contrast-bin AUC **0.76** vs static **0.53**) showing the cue carries signal *beyond* contrast, and 19 hand-engineered features rather than an opaque net. **No prior work has demonstrated the parallax cue is separable from contrast — that is the defensible novelty.**

**Where we are NOT (yet) SOTA, and our likely standing on benchmarks we haven't run.** Our 0.971 ROC-AUC is on the easy-negatives classification task; it is **not** an object-detection or counting result and should never be quoted against GRW's F2 or Satlas's r².
- **If we measured object-level F1** with realistic hard negatives on full scenes: expect a **substantial drop**. Mandroux's own trajectory (mAP 0.939 favorable chip → ~40–60% large-scale recall) and GRW's object precision of 90.8% at finer 4.7 m suggest a 10 m onshore object-F1 in the **~0.6–0.85** band is a realistic target — *plausibly competitive with Mandroux at equal GSD, but unproven*. Our parallax cue cannot fire on **static/parked turbines** (a hard recall ceiling vs installed count): we would likely beat the static-only a contrario baseline on *spinning* turbines while losing recall on non-spinning ones.
- **If we measured count-level r² vs USWTDB/IRENA:** untested; our operational-only, spinning-only signal would **systematically undercount** (parked turbines invisible), so we would likely **trail Satlas (0.978) and GRW (0.932)** unless fused with a static detector.
- **Honest residual caveats (ours):** negatives are random background → no object-level precision yet; parallax only fires on spinning turbines → recall ceiling; salience margin over static is modest (**+0.02 to +0.03** AUC).

**Bottom line.** Our work is best framed not as a competitor to GRW/Satlas/Mandroux on their object/count benchmarks, but as the **first rigorous, interpretable demonstration that Sentinel-2 inter-band parallax is a contrast-independent operational-turbine signal** — precisely the cue the dominant 10 m classical detector (Mandroux) deliberately discards, and which the only prior motion work (C1) showed only qualitatively. The object-F1 and count-r² benchmarks that would let us claim detection SOTA remain to be run, and on those we expect to be competitive-but-not-leading at 10 m, with a structural recall ceiling on non-spinning turbines.

---

### Source URLs (every numeric claim)
- GRW abstract/arXiv: https://arxiv.org/abs/2503.14860 · full text (Table 1 object P/R/F2, Table 2 r²): https://arxiv.org/html/2503.14860v1
- Mandroux single-date (IPOL 2022, F1≈0.9, B02-only): https://www.ipol.im/pub/art/2022/384/article.pdf · DOI https://doi.org/10.5201/ipol.2022.384
- Mandroux multi-date (ISPRS 2022, mAP 0.939, 80%@0.01 FP/km², 40–60% large-scale, DL AP 0.9–0.98 @0.5–2 m): https://isprs-annals.copernicus.org/articles/V-2-2022/383/2022/isprs-annals-V-2-2022-383-2022.pdf · DOI https://doi.org/10.5194/isprs-annals-V-2-2022-383-2022
- Nahrstedt band-motion CNN (99.4% val acc, 1.005 s offset, qualitative): https://dl.gi.de/bitstreams/3fb4a95e-8c19-4d2c-9652-d51770859c27/download
- Geowise/Contrast/Composite offshore S2 (F1 0.947–0.966): https://doi.org/10.3390/rs17142482
- DeepOWT (Sentinel-1, PR/AP, 9,941 OWTs): https://essd.copernicus.org/articles/14/4251/2022/
- Sentinel-1 Shandong (P 96.58 / R 91.59 / F1 94.02): https://www.sciencedirect.com/science/article/abs/pii/S0959652623045730
- WT-YOLO (0.6–5.4 m, AP +5.9–15.4%): https://www.sciencedirect.com/science/article/pii/S1569843223004375
- CGA-YOLO (Gaofen-2): https://doi.org/10.3390/rs18020324
- Oil tanks YOLOv3 (AP 0.84): https://towardsdatascience.com/oil-storage-tanks-volume-occupancy-on-satellite-imagery-using-yolov3-3cf251362d9d/ · YOLOX-TR SAR (mAP@.5 94.8): https://www.mdpi.com/2072-4292/14/14/3246/htm
- USGS USWTDB: https://eerscmap.usgs.gov/uswtdb/
