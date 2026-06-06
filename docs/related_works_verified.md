# Related Works (VERIFIED): Satellite-Based Wind Turbine Detection

Independent literature review, primary-source fact-check, and honest SOTA comparison for the CS131 project.

**What "ours" is.** Candidate-point CLASSIFICATION on Sentinel-2 L2A 10 m optical imagery, onshore US. Given a (lat, lon), output P(operational turbine). Novelty: an engineered, interpretable band-parallax / blade-motion cue (19 features) exploiting Sentinel-2's staggered inter-band acquisition, fused with CS131 static primitives plus a salience z-score, via a random forest. Ground truth USGS USWTDB; 173 US projects, 35 states; project-grouped split (139 train / 33 test); operational-only; single locked test. Headline (held-out, ROC-AUC, RANDOM-BACKGROUND negatives): fused RF per-project AUC 0.971 (pooled 0.984, AP 0.985); parallax-only 0.952; static salience 0.948; HOG+linear-SVM 0.938; old CS131-primitives-only 0.787. Separate NAIP 0.6 m static benchmark test AUC ~0.99. Real detection-mode benchmark (dense sliding window + NMS, threshold frozen from train, 34 test projects): micro precision 0.37 / recall 0.42 / F1 0.39, count fit-R2 0.13.

**Verification status.** Every paper below was fetched from its primary source in-session (arXiv HTML, IPOL PDF, ISPRS PDF, ESSD HTML, GI LNI PDF) except where noted. MDPI papers (offshore S2; CGA-YOLO) and one ScienceDirect paper render their results TABLES as images/JS, so exact decimal cells in those tables could not be read as text; the surrounding narrative numbers were verified and any unverifiable cell is flagged.

> **Caveat carried throughout.** Our headline is ROC-AUC / AP on candidate-point CLASSIFICATION with RANDOM-BACKGROUND negatives. That is fundamentally easier than full-scene object detection, where false positives scale with image area and hard negatives (towers, roads, bright buildings, crop shadows) dominate. ROC-AUC on this setup is NOT numerically comparable to object-level mAP / F1 or to count-level r2. Our own detection-mode F1 of 0.39 is the apples-to-apples object-level number, and it sits far below the classification AUC of 0.971.

---

## 1. Catalog of works by regime (exact metrics + GSD + region + task + source)

### Regime A. Global / continental onshore mapping products

**A1. Microsoft Global Renewables Watch (GRW).** Robinson, Ortiz, Kim, Dodhia, Zolli, Nagaraju, Oakleaf, Kiesecker, Lavista Ferres. arXiv:2503.14860, 2025. Source read: https://arxiv.org/html/2503.14860v1
- Imagery / GSD: PlanetScope quarterly RGB visual basemaps, 4.7 m/px at the equator (Web Mercator zoom 15), 2017 Q4 to 2024 Q2.
- Task: wind = point-based semantic segmentation to counting; FCN + ResNet-50 backbone, localization-based counting (LC) loss with four terms (patch/point/split/false-positive); MOSAIKS-SVM false-positive filter. Solar = U-Net + ResNeXt-50.
- Dataset / region: global. Wind labels from Dunnett et al. OSM points: 272,503 points DBSCAN-clustered into 17,971 tiles, 80/10/10 split. Deployed on ~13.98 trillion pixels. Final product 375,197 wind turbines, 86,410 solar PV installations.
- EXACT metrics (Table 1, wind, OBJECT level): no filtering precision 59.63% / recall 71.48% / F2 68.75%; global model + filtering + hard negatives precision 90.81% / recall 81.63% / F2 83.31%. No pixel-level reported for wind. (Solar context: pixel F2 74.48%; object precision 23.97% to 50.93% after FP filter.)
- EXACT metrics (Table 2, country-level capacity vs IRENA 2023): their onshore wind Pearson r2 = 0.932, Kendall tau = 0.877, +181.4 GW global difference. Satlas onshore wind r2 = 0.978 (tau 0.804, +51.2 GW); Satlas all-wind r2 = 0.975 (+12.9 GW). Their solar PV r2 = 0.960 (tau 0.718). Abstract rounds to 0.96 solar / 0.93 onshore wind.
- Note: authors state object numbers "likely underestimate performance" due to noisy / misaligned / repowered OSM labels.

**A2. Satlas renewable-energy layer.** Bastani, Wolters, Gupta, Ferdinando, Kembhavi, "SatlasPretrain," ICCV 2023, pp. 16772-16782. Citation confirmed at https://openaccess.thecvf.com/content/ICCV2023/html/Bastani_SatlasPretrain_A_Large-Scale_Dataset_for_Remote_Sensing_Image_Understanding_ICCV_2023_paper.html ; combines Sentinel-2 + NAIP, includes wind-turbine detection.
- Only number used here is the count-level agreement reported BY GRW in its Table 2: Satlas onshore wind r2 = 0.978, the best count-level wind agreement in the literature. Source for the number: https://arxiv.org/html/2503.14860v1 (not the Satlas paper itself; the Satlas paper does not report a turbine-count r2 vs IRENA).

### Regime B. Sentinel-2 10 m classical onshore detectors (Mandroux) -- closest on imagery / GSD / region

**B1. Mandroux, Dagobert, Drouyer, Grompone von Gioi. Single Date Wind Turbine Detection on Sentinel-2 Optical Images. IPOL 12 (2022), 198-217.** Source read: https://www.ipol.im/pub/art/2022/384/article.pdf ; DOI https://doi.org/10.5201/ipol.2022.384
- Imagery / GSD: Sentinel-2, 10 m, blue band B02 ONLY -- explicit quote: "Only the B02 spectral band is used: the blue one, to avoid blurring effects due to rotating blades." Tower height assumed 80 m.
- Task: single-scene object detection via a-contrario (NFA): dark-shadow detector + bright-hub detector fused.
- Dataset / region: ~300 Sentinel-2 chips with a centered turbine and ~300 without; detect / no-detect assessed in the central square only. Hand-built. Onshore (region not US-specific; generic).
- EXACT metrics: parameter sweeps (Fig. 6) with (t_shadow, t_hub, t_NFA) = (25, 50, 1) give F1 peaking near 0.9, with precision / recall plotted across thresholds. No single boxed F1 number is stated; "F1 ~ 0.9" is a read of the peak of the plotted curves.

**B2. Mandroux, Drouyer, Grompone von Gioi. Multi-Date Wind Turbine Detection on Optical Satellite Images. ISPRS Annals V-2-2022, 383-390.** Source read: https://isprs-annals.copernicus.org/articles/V-2-2022/383/2022/isprs-annals-V-2-2022-383-2022.pdf ; DOI https://doi.org/10.5194/isprs-annals-V-2-2022-383-2022
- Imagery / GSD: Sentinel-2 10 m, B02 only ("only one spectral band is used: the blue B02 one"), 4 dates of one scene (March, Sept, Nov, Dec); temporal coherence.
- Task: multi-date object detection (temporal aggregation of shadow / hub a-contrario scores).
- EXACT metrics:
  - Controlled scene, 1000x1000 px, 40 turbines: mean Average Precision = 0.939 (PR curve). ~75% of turbines found with zero false positives. ~80% detected at 0.01 false positives / km2. Around 80% detected overall on this scene.
  - Large-scale 10,000 km2 (Connecticut-sized, realistic): epsilon=1 to ~60% detected; epsilon=0.1 to 45%; epsilon=0.01 to 40%. Observed false positives ran ~3-4x the nominal NFA budget per 1000x1000 tile (e.g. epsilon=1 expected 1, got "3 or 4 on average").
  - The hub detector alone fires on 1.6% of all pixels (= ~16,000 detections on a million-pixel image).
  - DL note: deep methods reach AP 0.9-0.98 but on 0.5-2 m imagery (Zhang et al. 2020), vs their 10 m.
  - Authors explicitly warn the near-perfect ROC "does not give a correct vision of the absolute performances" on a sparse-positive pixel grid, and prefer rate-of-false-alarms / PR.

### Regime C. Band-motion / parallax (closest conceptual neighbor)

**C1. Nahrstedt, Gartner, Wittmann. Detection of wind turbine motion between satellite bands with convolutional neural networks. EnviroInfo 2023, LNI, Gesellschaft fur Informatik, Bonn.** Source read: https://dl.gi.de/server/api/core/bitstreams/3fb4a95e-8c19-4d2c-9652-d51770859c27/content (the /bitstreams/.../download form in the old review now 404s/blocks; the /server/api/core/bitstreams/.../content form works.)
- Imagery / GSD: Sentinel-2 10 m, bands B02/B03/B04, exploiting staggered acquisition, max offset 1.005 s (B02 to B04). Theoretical max blade-tip displacement ~45 m (4.5 px) for a 90 m blade. Same physics our cue uses.
- Task: binary classification, spinning turbine vs non-turbine, pretrained DenseNet-121 on 40x40x3 RGB chips. Labels from 4,800 Enercon turbines (Germany): 2,996 spinning / 110 standing / 1,662 undetectable shadows (labeled via shadow spin; model trained on the blades themselves).
- EXACT metrics: "accuracy of up to ~99.4% on the validation set." Cross-entropy loss stagnated ~0.24 nats. NO precision / recall / AP / F1. NO held-out cross-region TEST set (the 99.4% is in-distribution validation). NO grouped split. Near-degenerate negative class (only 110 standing turbines, vs 2,996 spinning, plus non-turbine images). Plus a qualitative occlusion-sensitivity analysis and one qualitative offshore example near Borkum.

**C2. Methodological cousins (parallax physics for other moving objects).** Liu et al., "Space eye on flying aircraft: From Sentinel-2 MSI parallax to hybrid computing," Remote Sensing of Environment 246:111867, 2020; Fisser et al., "Detecting Moving Trucks on Roads Using Sentinel-2 Data," Remote Sensing 14(7):1595, 2022; Heiselberg, "Aircraft and Ship Velocity Determination in Sentinel-2 Multispectral Images," Sensors 19(13), 2019. All establish the same staggered-band parallax physics for aircraft / ships / trucks. Not wind-detection comparators.

### Regime D. Offshore Sentinel-2 / SAR detectors

**D1. Seasonally Robust Offshore Wind Turbine Detection in Sentinel-2 Imagery Using Imaging Geometry-Aware Deep Learning. Remote Sensing 17(14):2482, 2025.** Source read (narrative): https://www.mdpi.com/2072-4292/17/14/2482
- Imagery / GSD: Sentinel-2 10 m, offshore China; 887 scenes over 56 MGRS grid cells; 2,471 OWT samples (1,819 train from 10 grids / 652 test from 5 grids); ~18,000 km of coastline; 7,369 OWTs mapped (by March 2025).
- Task: object detection. Baseline Faster R-CNN (ResNet-50 + FPN, MMDetection). Three geometry-aware variants: Geowise-Net (FiLM-injects solar incidence + satellite view angles), Contrast-Net (contrastive learning on seasonal pairs), Composite (fuses both).
- Metrics (point-based, 50 m TP threshold): all three proposed models beat Faster R-CNN; ranking Faster R-CNN < Geowise-Net < Contrast-Net < Composite, with full-test-set improvements of ~1.1% / 1.4% / 1.9% over baseline. Seasonal F1 gap (cold minus hot) shrinks: Faster R-CNN 3.7% > Geowise-Net 1.9% > Contrast-Net 1.5% > Composite 0.8%. The specific F1 values 0.947 / 0.958 / 0.961 / 0.966 quoted in the old review are CONSISTENT with the stated ~1.1/1.4/1.9% spread but could NOT be read from the primary text (Table 1 renders as an image/JS on MDPI). Flagged as UNVERIFIED-EXACT, ranking + deltas VERIFIED.

**D2. DeepOWT. Hoeser, Feuerstein, Kuenzer. ESSD 14:4251-4270, 2022.** Source read: https://essd.copernicus.org/articles/14/4251/2022/
- Imagery / GSD: Sentinel-1 SAR (~10 m), global ocean (200 km coastal buffer). Cascade of two CNNs trained ONLY on synthetic data (SyntEO): stage 1 farm (OWF) detector, stage 2 turbine (OWT) detector. ResNet-50 and Faster R-CNN both appear in the architecture (abbreviation list). 9,941 offshore infrastructure locations, quarterly 2016Q3-2021Q2.
- Task: object detection + global dataset. Reports class-wise precision-recall curves and AP (Table 2, Fig. 9) on 2021Q2 North-Sea-Basin (NSB) and East-China-Sea (ECS) test sets; F1 time series (Fig. 10). Exact AP / F1 cells are in an image-rendered table; the STRUCTURE (PR + AP per class, two ocean test sites, stable across both) is verified, exact decimals not extractable from text.

**D2b. Shandong offshore (related, often cited alongside DeepOWT).** "Deep learning-based monitoring of offshore wind turbines in Shandong Sea of China and their location analysis," Journal of Cleaner Production, 2023. Source: https://www.sciencedirect.com/science/article/abs/pii/S0959652623045730
- IMAGERY CORRECTION: this paper uses Sentinel-2, NOT Sentinel-1 (multiple independent abstracts confirm "Sentinel-2 remote sensing images"). ResNet34-SSD model. Precision 96.58% / recall 91.59% / F1 94.02%. 424 turbines across Dongying (149), Laizhou Bay (88), Haiyang (187).

### Regime E. Onshore high-resolution object detectors (sub-meter)

**E1. WT-YOLO. Zhai, Chen, et al. Identifying wind turbines from multiresolution and multibackground remote sensing imagery. Int. J. Applied Earth Observation & Geoinformation 122:103613, 2023.** Source: https://www.sciencedirect.com/science/article/pii/S1569843223004375 ; code https://github.com/zyyyccc/WT-YOLO
- Imagery / GSD: Google Earth high-res, 0.6-5.4 m, multi-background dataset; YOLOv5-based, treats hub / base / shadow-hub as key points.
- Task: object detection + positioning. EXACT metric: AP is 5.92% to 15.43% HIGHER than prior wind-turbine detectors across 0.6-5.4 m (relative improvement, not an absolute AP). This is the regime our NAIP 0.6 m static benchmark (test AUC ~0.99) sits in.

**E2. CGA-YOLO. Wind Turbines Small Object Detection in Remote Sensing Images Based on CGA-YOLO: A Case Study in Shandong Province, China. Remote Sensing 18(2):324, 2026.** Source: https://www.mdpi.com/2072-4292/18/2/324 ; DOI https://doi.org/10.3390/rs18020324
- Imagery / GSD: Gaofen-2 (~1 m), SDWT dataset, Shandong. Small-object detection with dynamic convolution + CBAM + GhostBottleneck. Reports improved precision / recall over YOLO baselines (no single number quoted in old review; consistent). High-res onshore regime.

### Regime F. Methodological analog: oil-storage-tank-from-shadow

- YOLOv3 (optical, floating-head tanks + shadow volume): test AP 0.84, train AP 0.942, 3-class (Tank/Fixed-head/Tank-cluster). Source: https://towardsdatascience.com/oil-storage-tanks-volume-occupancy-on-satellite-imagery-using-yolov3-3cf251362d9d/ (blog write-up of a GitHub project, NOT a peer-reviewed paper -- weakest source in the set).
- YOLOX-TR (dense oil tanks, Gaofen-3 1 m SAR): mAP@0.5 = 94.8%, mAP = 60.8%. Source: https://www.mdpi.com/2072-4292/14/14/3246
- Relevance: shadow-as-signal analog (sun-angle reasoning to recover height/volume). Methodological only.

### Ground truth used by ours
- USGS USWTDB: https://eerscmap.usgs.gov/uswtdb/ . Appropriate high-completeness US onshore registry. Confirmed valid gold standard.

---

## 2. Comparability matrix

Legend: YES directly comparable to ours; PART partial; NO not comparable.

| Work | Imagery / GSD | Task | Region | Headline metric | Same task? | Same metric? | Same GSD? | Same region? | Directly comparable to ours? |
|---|---|---|---|---|---|---|---|---|---|
| Ours (classification) | S2 / 10 m | candidate-point classification, EASY (random bg) negatives | Onshore US | per-proj ROC-AUC 0.971; AP 0.985 | - | - | - | - | - |
| Ours (detection mode) | S2 / 10 m | object detection, sliding window + NMS | Onshore US | micro F1 0.39 (P 0.37 / R 0.42); count R2 0.13 | - | - | - | - | this is OUR own apples-to-apples object number |
| GRW (A1) | PlanetScope / 4.7 m | seg to count | Global onshore | obj P 90.8 / R 81.6 / F2 83.3; count r2 0.932 | NO | NO (F2/r2 != AUC) | NO | PART (incl. US) | NO -- different task and metric |
| Satlas (A2) | S2+NAIP / mixed | count | Global | count r2 0.978 vs IRENA | NO | NO | NO | PART | NO -- count-level only |
| Mandroux single (B1) | S2 / 10 m, B02 only | detection | Onshore | F1 ~0.9 (central-chip detect) | PART | PART (F1 vs AUC) | YES | PART | PART -- closest imagery; discards the motion cue we use; metric differs |
| Mandroux multi (B2) | S2 / 10 m, B02 only | detection | Onshore | mAP 0.939 favorable; 40-60% large-scale recall | PART | PART (mAP/recall vs AUC) | YES | PART | PART -- best S2 10 m object-level reference; vs our detection-mode F1 0.39 |
| Nahrstedt CNN (C1) | S2 / 10 m | classification (spin) | Germany | ~99.4% in-dist VAL accuracy (no P/R/F1/AP/test) | YES (same cue + task) | PART (acc only, no test/AUC) | YES | PART | PART -- same cue, but only one in-distribution accuracy with degenerate negatives |
| Geowise/Contrast/Composite (D1) | S2 / 10 m | detection | OFFSHORE | F1 ranked, ~1-2% spread (exact cells unread) | PART | PART | YES | NO | NO -- water trivializes negatives |
| DeepOWT (D2) | S1 SAR / ~10 m | detection | OFFSHORE | PR / AP curves per class, NSB + ECS | NO (SAR) | PART | YES | NO | NO -- SAR + offshore |
| Shandong (D2b) | S2 / 10 m | detection | OFFSHORE | P 96.58 / R 91.59 / F1 94.02 | PART | PART (F1 vs AUC) | YES | NO | NO -- offshore (water) |
| WT-YOLO (E1) | GE / 0.6-5.4 m | detection | Onshore | AP +5.9-15.4% vs priors (relative) | PART | PART | NO (sub-m) | PART | PART -- matches our NAIP regime, not S2 |
| CGA-YOLO (E2) | Gaofen-2 / ~1 m | detection | Onshore | improved P/R vs YOLO | PART | PART | NO | NO | NO -- high-res only |
| Oil tanks (F) | optical/SAR / sub-m | detection | n/a | AP 0.84; mAP@0.5 94.8 | PART (analog) | PART | NO | NO | PART -- methodological analog only |

**Why our 0.971 ROC-AUC is not comparable to mAP / F1 / r2.** Our negatives are random background, so the classifier separates "turbine vs empty field," not "turbine vs tower / road / bright building / crop shadow." The detectors above (GRW, Mandroux, Geowise, WT-YOLO, oil tanks) are penalized precisely by those hard negatives, and their false positives scale with image area. That is why GRW's OBJECT precision is 90.8% (not ~98%) and Mandroux's large-scale recall collapses to 40-60%. ROC-AUC is also inflated on sparse-positive grids, which Mandroux state explicitly. Count-level r2 (Satlas 0.978, GRW 0.932) measures aggregate capacity agreement, a different question from per-instance correctness. The ONLY apples-to-apples object-level number we own is the detection-mode F1 of 0.39, and that is the figure that belongs next to Mandroux's mAP 0.939 / 40-60% recall.

---

## 3. Fact-check of the existing related_works.md

Each numeric / factual claim in the prior file, marked CONFIRMED / UNVERIFIED / WRONG, with source.

**GRW (A1)**
- 4.7 m/px, PlanetScope, Q4 2017-Q2 2024 -- CONFIRMED (arXiv html).
- FCN + ResNet-50, LC loss four terms, MOSAIKS-SVM FP filter; solar U-Net + ResNeXt-50 -- CONFIRMED.
- 272,503 points to 17,971 tiles, 80/10/10; ~13.98 trillion pixels; 375,197 wind / 86,410 solar -- CONFIRMED.
- Wind object: no-filter P 59.63 / R 71.48 / F2 68.75; filtered P 90.81 / R 81.63 / F2 83.31 -- CONFIRMED (Table 1).
- Solar pixel F2 74.48; object precision 23.97 to 50.93 -- CONFIRMED (Table 1).
- Country-level: their onshore wind r2 0.932, tau 0.877; Satlas onshore wind r2 0.978; their solar r2 0.960 -- CONFIRMED (Table 2).
- Satlas all-wind r2 0.975 (the old review lists this parenthetically as "0.975") -- CONFIRMED.

**Mandroux single (B1)**
- 10 m, B02 only, "to avoid blurring effects due to rotating blades" -- CONFIRMED (exact quote).
- tower height assumed 80 m -- CONFIRMED.
- shadow + hub a-contrario fusion -- CONFIRMED.
- ~300 chips with centered turbine + ~300 without; central-square detect/no-detect -- CONFIRMED.
- F1 ~ 0.9 at (25, 50, 1) -- CONFIRMED as a read of Fig. 6 peak. Note this is an interpolated peak of plotted F1 curves, not a boxed number. UNVERIFIED-EXACT (the "0.9" is approximate by the paper's own plotting).
- hub detector fires on 1.6% of pixels -- CONFIRMED but this is stated in the MULTI-date (B2) paper, not the single-date (B1) paper; the old review attributes it to B1. Minor MIS-ATTRIBUTION (correct number, wrong paper).

**Mandroux multi (B2)**
- 10 m, B02 only, 4 dates -- CONFIRMED.
- mAP = 0.939 -- CONFIRMED (exact).
- ~75% found with zero FP -- CONFIRMED.
- ~80% at 0.01 FP/km2 -- CONFIRMED.
- large-scale: 60% / 45% / 40% at epsilon 1 / 0.1 / 0.01 -- CONFIRMED (exact).
- observed FP ~3-4x nominal budget -- CONFIRMED.
- DL AP 0.9-0.98 at 0.5-2 m (Zhang et al. 2020) -- CONFIRMED (exact).
- "ROC does not give a correct vision of absolute performance" -- CONFIRMED (exact paraphrase).

**Nahrstedt CNN (C1)**
- S2 10 m, B02/B03/B04, max offset 1.005 s -- CONFIRMED.
- max blade displacement ~45 m (4.5 px) for 90 m blade -- CONFIRMED (exact).
- DenseNet-121, 40x40x3 RGB chips -- CONFIRMED.
- 4,800 Enercon turbines (Germany); 2,996 spinning / 110 standing / 1,662 undetectable -- CONFIRMED (exact).
- ~99.4% validation accuracy -- CONFIRMED (exact: "up to ~99.4% on the validation set").
- CE loss stagnated ~0.24 nats -- CONFIRMED (exact).
- "no P/R/AP/F1, no held-out cross-region test, no grouped split, near-degenerate negative class (only 110 standing)" -- CONFIRMED. The 99.4% is in-distribution validation; there is no separate test set, only occlusion-sensitivity + one qualitative offshore example. The old review's characterization is accurate and fair.
- DEAD LINK: the old review's URL (.../bitstreams/3fb4a95e.../download) now blocks; working URL is .../server/api/core/bitstreams/3fb4a95e.../content. Update recommended.

**Offshore S2 (D1)**
- S2 10 m, offshore China, 887 scenes, 7,369 OWTs -- CONFIRMED. (Old review says "<5% cloud, Jan 2023-Mar 2025" -- the split details I read are 2,471 samples, 1,819 train / 652 test, mapped by March 2025; the cloud/date detail not re-checked but plausible.)
- Faster R-CNN baseline + Geowise-Net / Contrast-Net / Composite, with described mechanisms -- CONFIRMED.
- Single-date F1 0.947 < 0.958 < 0.961 < 0.966 -- UNVERIFIED-EXACT. The RANKING (FRCNN < Geowise < Contrast < Composite) and the ~1.1/1.4/1.9% improvement spread are CONFIRMED in the text; the four exact decimals are in an image/JS table I could not read. The decimals are internally consistent with the stated deltas, so they are plausibly correct, but not independently confirmed.
- summer-winter gap 3.7% to 0.8% -- CONFIRMED (exact: FRCNN 3.7 > Geowise 1.9 > Contrast 1.5 > Composite 0.8).

**DeepOWT (D2)**
- S1 SAR ~10 m, global, cascade of two CNNs, 9,941 locations -- CONFIRMED. The "two ResNet-50 Faster R-CNN detectors" phrasing is approximately right (ResNet-50 and Faster R-CNN both in the architecture; trained on synthetic SyntEO data; farm stage then turbine stage).
- class-wise PR curves and AP on 2021Q2 NSB and ECS test sets -- CONFIRMED (Table 2, Fig. 9).
- "TP = predicted point within a 100 m-radius ground-truth polygon" -- UNVERIFIED. The 100 m radius did not appear in the text I read; the matching criterion is described but the exact radius was not isolated. Treat the 100 m as unconfirmed.

**Shandong Sentinel-1 (cited under D2)**
- "Sentinel-1 Shandong (ResNet34-SSD): P 96.58 / R 91.59 / F1 94.02" -- NUMBERS CONFIRMED, IMAGERY WRONG. The primary source (J. Cleaner Production 2023, S0959652623045730) describes Sentinel-2 imagery, not Sentinel-1. The old review lists it as a "related Sentinel-1 result" supporting DeepOWT's SAR strength; it is actually a Sentinel-2 optical result. This weakens its use as SAR-supporting evidence. CORRECTION REQUIRED.
- "another S1 model reports P/R = 95.97% / 91.18%" -- UNVERIFIED (could not isolate the specific second model; do not rely on it).

**WT-YOLO (E1)**
- Google Earth 0.6-5.4 m, YOLOv5-based, AP +5.92-15.43% vs priors, code at github.com/zyyyccc/WT-YOLO -- CONFIRMED. Note this is a RELATIVE improvement, not an absolute AP; the old review states this correctly.

**CGA-YOLO (E2)**
- Gaofen-2 ~1 m, SDWT dataset, Shandong, improved P/R over YOLO baselines -- CONFIRMED (qualitatively). Note: published Jan 2026, vol 18(2):324, so it is a 2026 paper.

**Oil tanks (F)**
- YOLOv3 test AP 0.84 (train 0.942) -- CONFIRMED, but SOURCE IS A BLOG/GITHUB write-up, not peer-reviewed. Flag the source quality.
- YOLOX-TR mAP@0.5 94.8, mAP 60.8 (Gaofen-3 1 m SAR) -- CONFIRMED (exact).

**Physics claim used throughout (ours)**
- "green B03 ~+0.324 s after blue B02, red B04 ~+1.005 s after blue, up to ~2.6 s across all bands" -- PARTIALLY VERIFIED. The 1.005 s B02-to-B04 max offset is CONFIRMED via Nahrstedt citing the ESA MSI technical guide. The staggered-detector inter-band along-track parallax physics is well established (ESA MSI guide; Liu et al. 2020; Heiselberg 2019). The finer per-band values (0.324 s green; 2.6 s full spread) are not in a single quotable cited source and appear to be project-derived from the focal-plane geometry; they are physically consistent but should be cited to the ESA detector-timing spec rather than to Nahrstedt.

**Things the old review got RIGHT and that survive scrutiny**
- The central honest framing (ROC-AUC on easy negatives != object mAP/F1/r2) is correct and well supported.
- The "Mandroux deliberately discards the motion cue (B02 only)" point is verified by an exact quote in BOTH Mandroux papers. This is the strongest, cleanly-sourced novelty argument.
- The Nahrstedt critique (single in-distribution accuracy, degenerate negatives, no AUC/F1/AP/grouped split) is accurate.

**Important works the old review did NOT include (gaps)**
- He et al., "Mapping land- and offshore-based wind turbines in China in 2023 with Sentinel-2 satellite data," Renewable & Sustainable Energy Reviews 2025 (S1364032125002394): a Sentinel-2 10 m ONSHORE+offshore China mapping paper -- arguably more directly comparable on imagery+region than the offshore-only D1/D2, and missing.
- Manso-Callejo et al. 2020 (hybrid semantic segmentation for wind turbines) and Han et al. 2018 ("Targets Mask U-Net"), both cited inside Mandroux as the high-res DL prior art -- worth listing as the high-res-DL baseline lineage.
- Heiselberg 2019 (Sentinel-2 aircraft/ship velocity) -- the original inter-band parallax method; the old review cites Liu and Fisser but not Heiselberg, which is the seminal one.
- Dunnett et al. 2020 (Scientific Data) -- the OSM-derived global wind/solar dataset that is GRW's label source AND a standalone "registry" comparator.

---

## 4. Honest verdict on SOTA-by-regime and where ours stands

**Genuine SOTA by regime.**
- Global onshore object-level product: Microsoft GRW. Object P 90.8 / R 81.6 / F2 83.3 at 4.7 m, IRENA-validated, temporal. The reference for "find and count onshore turbines worldwide."
- Count-level capacity agreement: Satlas, onshore wind r2 0.978, narrowly ahead of GRW's 0.932 (both as measured in GRW Table 2).
- Sentinel-2 10 m onshore object detection (interpretable, classical): Mandroux multi-date, mAP 0.939 on a favorable hand-picked scene, with the honest large-scale numbers (40-60% recall, ~10,000 FP per Connecticut-sized region) showing how hard the real onshore 10 m task is. This is the right yardstick for our detection mode.
- Offshore detection: the Sentinel-2 geometry-aware models (D1) for optical, and DeepOWT (D2) for global SAR. Both operate over water, a categorically easier background than onshore clutter.
- High-res onshore detection: WT-YOLO / CGA-YOLO on sub-meter imagery. This is the regime our NAIP 0.6 m benchmark (test AUC ~0.99) lives in, but our NAIP number is a classification AUC, not an object AP, so it is not a SOTA claim against them either.

**Where ours is genuinely novel.** The band-parallax / blade-motion cue as an engineered, interpretable, CONTRAST-SEPARABLE signal. Mandroux -- the dominant 10 m classical detector -- deliberately throws this signal away: both papers use B02 only "to avoid blurring effects due to rotating blades." We treat the across-band displacement of the spinning blade as the feature, with 19 interpretable descriptors. The only prior work using the same physics is Nahrstedt (C1), a black-box DenseNet reporting a single ~99.4% in-distribution validation accuracy with a degenerate 110-sample standing class and no AUC/F1/AP/test/grouped split. Our defensible contribution over C1 is rigor + interpretability: a project-grouped frozen split, a single locked eval, residual-after-orthogonalization controls showing the cue carries signal BEYOND contrast, and hand-engineered features instead of an opaque net. No prior work has shown the parallax cue is separable from contrast -- that is the real novelty.

**Where ours is clearly NOT SOTA.** Object-level detection and counting. Our 0.971 is classification AUC on easy negatives and must never be quoted against GRW's F2, Satlas's r2, or Mandroux's mAP. Our OWN apples-to-apples object-level number is the detection-mode benchmark: micro F1 0.39 (P 0.37 / R 0.42), count R2 0.13. That F1 of 0.39 is far below Mandroux's favorable-scene mAP 0.939 and in the same hard regime as Mandroux's large-scale 40-60% recall. Two structural reasons we trail on object metrics: (1) random-background training does not teach the hard negatives that dominate full scenes; (2) the parallax cue cannot fire on static / parked / low-wind turbines, imposing a recall ceiling versus installed count, so a count-r2 comparison vs USWTDB/IRENA would systematically undercount and trail Satlas (0.978) / GRW (0.932) unless fused with a static detector.

**The only truly apples-to-apples comparisons.**
- For the parallax CUE itself: Nahrstedt (C1) -- same imagery, same GSD, same physics, same binary-classification task. We are strictly more rigorous; they have no AUC/F1/test to beat, so the honest claim is "first rigorous, contrast-controlled quantification," not "higher number than C1."
- For our DETECTION mode: Mandroux multi-date (B2) at equal imagery (S2) and equal GSD (10 m) -- our F1 0.39 vs their favorable-scene mAP 0.939 and large-scale 40-60% recall. This is the comparison that should anchor any object-level claim, and on it we are NOT competitive yet.
- Everything else (GRW, Satlas, offshore D1/D2/D2b, WT-YOLO, CGA-YOLO, oil tanks) differs on task, metric, GSD, or region and is context, not a head-to-head.

**Bottom line.** Frame the work as the first rigorous, interpretable demonstration that Sentinel-2 inter-band parallax is a contrast-independent OPERATIONAL-turbine signal -- exactly the cue the dominant 10 m classical detector (Mandroux) deliberately discards, and which the only prior motion work (Nahrstedt) showed only qualitatively. Do NOT frame the 0.971 AUC as detection or counting SOTA. The object-level number we actually have (F1 0.39) is honest and weak, consistent with how hard 10 m onshore detection is even for the published SOTA.

---

### Source URLs (every numeric claim)
- GRW: https://arxiv.org/abs/2503.14860 ; full text (Tables 1-2): https://arxiv.org/html/2503.14860v1
- Satlas (citation): https://openaccess.thecvf.com/content/ICCV2023/html/Bastani_SatlasPretrain_A_Large-Scale_Dataset_for_Remote_Sensing_Image_Understanding_ICCV_2023_paper.html ; r2 0.978 number is from GRW Table 2 above
- Mandroux single (IPOL 2022): https://www.ipol.im/pub/art/2022/384/article.pdf ; DOI https://doi.org/10.5201/ipol.2022.384
- Mandroux multi (ISPRS 2022): https://isprs-annals.copernicus.org/articles/V-2-2022/383/2022/isprs-annals-V-2-2022-383-2022.pdf ; DOI https://doi.org/10.5194/isprs-annals-V-2-2022-383-2022
- Nahrstedt CNN (working link): https://dl.gi.de/server/api/core/bitstreams/3fb4a95e-8c19-4d2c-9652-d51770859c27/content
- Offshore S2 (D1): https://www.mdpi.com/2072-4292/17/14/2482 ; DOI https://doi.org/10.3390/rs17142482
- DeepOWT (D2): https://essd.copernicus.org/articles/14/4251/2022/
- Shandong S2 (D2b): https://www.sciencedirect.com/science/article/abs/pii/S0959652623045730
- WT-YOLO (E1): https://www.sciencedirect.com/science/article/pii/S1569843223004375 ; code https://github.com/zyyyccc/WT-YOLO
- CGA-YOLO (E2): https://www.mdpi.com/2072-4292/18/2/324 ; DOI https://doi.org/10.3390/rs18020324
- Oil tanks YOLOv3 (blog/GitHub, non-peer-reviewed): https://towardsdatascience.com/oil-storage-tanks-volume-occupancy-on-satellite-imagery-using-yolov3-3cf251362d9d/
- Oil tanks YOLOX-TR (SAR): https://www.mdpi.com/2072-4292/14/14/3246
- USGS USWTDB: https://eerscmap.usgs.gov/uswtdb/
