# C12 Hackathon Solution Report

## Challenge 1 detection and Challenge 2 contrast optimization

**Purpose.** This report explains the implemented algorithms, their physical motivation, the code paths, the held-out evaluation design, and what the measured results do—and do not—show. The work keeps the organiser's default simulator configuration unchanged. The detector and optimizers make decisions from measured images and public experiment APIs; `reveal()` is used only by the separate post-run benchmark evaluator.

---

## Executive summary

| Challenge | Initial method | Added method | Main held-out outcome |
|---|---|---|---|
| 1 — detect interdot sticks | Shape-based matched filter with its baseline angle bank | Periodic lattice filter and a wide-angle, charge-slope-informed detector | Default-test macro F1 rises from **0.7460** to **0.7786**; false-positive pixels fall from 4,233 to 2,855. |
| 1 — orientation robustness | Same baseline detector | Charge-slope detector on a separate charge-consistent stress set | Stress-test macro F1 rises from **0.7307** to **0.7870**; false-positive pixels fall by 32.9%. |
| 2 — maximize contrast | Full-frame global search plus local refinement | Drift-calibrated search of selected small ROIs | Across 10 paired devices, median score ratio is 0.842 vs 0.857, with **51.8% fewer integrated pixels**. |

The detector changes produce a measurable improvement on the default test split and a larger gain on the orientation stress split. The Challenge 2 zoom method trades a small decrease in median score ratio for approximately half the pixels. Its worst-device result is weaker, and it makes more individual scan calls; the pixel-time estimate excludes per-call overhead.

---

# 1. Challenge 1 — detect the interdot pixels

## 1.1 Problem definition and constraints

Input is one raw charge-stability-diagram (CSD) image. Output is a binary mask of the short interdot features (“sticks”). A target is sparse relative to the image; background pixels dominate, and raw image standard deviation is not a useful segmentation metric. Horizontal row noise and short line connectors can resemble dark features.

The implementation is analytic and training-free. At inference it does not read `sticks.jsonl`, masks, generator labels, or simulator state. It uses the public generator's appearance scale and the image itself. Evaluation is reported on foreground pixel metrics and connected-component object metrics—not pixel accuracy, which would be misleading for such a sparse foreground.

## 1.2 Initial detector: local matched filtering

Implemented in `solutions/detection.py` as `InterdotDetector`.

1. **Remove row-wise acquisition offset.** Subtract the median of each image row. This suppresses a shared horizontal stripe component while leaving a small dark stick largely intact.
2. **Build a physical template bank.** Templates are short rectangles parameterized by length and width, blurred with the configured Gaussian blur, and compared against a local background annulus. Each kernel is zero-sum and L2-normalized, making its response less sensitive to absolute background and comparable across shapes. The baseline bank uses the simulator's nominal 45-degree angle and small angle offsets around it; it is deliberately a good starting point for the default simulator, not a claim that all physical transitions are 45 degrees.
3. **Find candidate centres.** Correlate the row-corrected image with the bank, retain the best response at each pixel, apply peak non-maximum suppression, and record the selected template index and angle for every candidate.
4. **Rasterize the mask.** Convert each retained peak to the blurred footprint of its winning template. The default matched-filter score threshold is 4 noise-standard-deviation units.
5. **Estimate stick contrast.** Divide the matched response by the gain of the winning template. This compensates for much of the square-root-of-area advantage that a larger template has in the raw correlation score.

The initial method is a strong high-recall baseline, but it has a limited orientation bank and does not use the fact that real interdot candidates should lie on a repeated charge lattice.

## 1.3 Periodicity filter: reject candidates inconsistent with a charge lattice

Implemented in `solutions/periodicity.py` as `fit_charge_lattice` and `PeriodicInterdotDetector`.

Each candidate centre is converted from pixel `(row, column)` to physical plunger coordinates `(V_h, V_v)`. The fitting routine then:

- merges near-duplicate candidate centres (default radius 0.020 V);
- forms pair-difference vectors whose lengths are compatible with the nominal charging spacing;
- uses pairs of non-collinear vectors as candidate 2-D lattice bases;
- scores each basis by how many candidate centres fall within 0.015 V of integer lattice coordinates;
- refits the basis and origin by least squares; and
- rejects an unsupported or poorly conditioned fit.

The default periodic filter requires at least four inliers and confidence at least 0.10. Candidates on a fitted lattice are retained; unusually strong isolated responses (score at least 18) are rescued because a genuine stick may be isolated by missing neighbours or lie near the scan edge. If the image does not support a reliable 2-D fit, the implementation falls back to the unfiltered detector rather than applying a guessed lattice.

**Why this helps:** pixel matching asks whether a local patch resembles a stick. Periodicity adds a second, independent question: does this candidate occupy a plausible charge-state crossing in the repeated 2-D pattern? On the default test split this substantially improves object precision and reduces false-positive pixels.

## 1.4 Charge quantization and stick-slope constraint

A fixed 45-degree assumption is not a general consequence of charge quantization. In a constant-interaction electrostatic model, electrochemical potentials depend on the dot occupations, charging energies, mutual charging energy, and gate-capacitance/lever-arm matrix. For example, in a two-plunger-coordinate notation:
$$
\[
\mu_1=(N_1-\tfrac12)E_{C1}+N_2E_{Cm}
-\frac{C_{g1}V_1E_{C1}+C_{g2}V_2E_{Cm}}{|e|},
\]
\[
\mu_2=(N_2-\tfrac12)E_{C2}+N_1E_{Cm}
-\frac{C_{g1}V_1E_{Cm}+C_{g2}V_2E_{C2}}{|e|}.
\]
$$
For this convention, constant $-\(\mu_1\)$ and constant $-\(\mu_2\)$ boundaries have slopes
$$
\[
\frac{dV_2}{dV_1}=-\frac{C_{g1}E_{C1}}{C_{g2}E_{Cm}},
\qquad
\frac{dV_2}{dV_1}=-\frac{C_{g1}E_{Cm}}{C_{g2}E_{C2}}.
\]
$$
The interdot degeneracy condition is $\(\mu_1-\mu_2=0\)$. Its direction depends on the capacitances and lever arms; it is not universally 45 degrees. Charge quantization creates stable charge domains and a honeycomb-like lattice, but an unlabeled grayscale lattice alone does not identify every charge-axis polarity or lever arm.

The implemented image model represents the fitted charging lattice by a basis matrix \(B\) whose columns are the two inferred charge translations. It defines charge coordinates
$$
\[
q= B^{-1}(V-V_0).
\]
$$
A constant charge-transfer coordinate $\(q_1-q_2\)$ has normal vector $\(\nabla q_1-\nabla q_2\)$; the stick tangent is perpendicular to this normal. Because the image does not label the polarity of either charge axis, the fitter evaluates both relative-sign hypotheses. The formulas motivate the slope prior, but the code does **not** pretend that the lattice uniquely determines the true capacitance matrix.

### Implementation of the wide-angle charge-slope method

`ChargeConstrainedInterdotDetector` uses these steps:

1. Run a broad matched-filter bank with 15-degree samples over the unoriented 0–180-degree range; do not restrict candidates to the baseline 45-degree neighborhood.
2. Estimate the dominant image orientation by a weighted axial circular mean of candidate template angles. Axial means treat a stick at angle \(\theta\) as equivalent to \(\theta+\pi\). High matched-filter margins receive more weight.
3. Fit the candidate-centre lattice. Supply the measured angle as a **soft RANSAC tie-breaker** between equivalent bases; the dual-basis charge relation is used to select a slope hypothesis only when it agrees with the measured orientation within 15 degrees.
4. Retain candidates within 25 degrees of the selected direction; allow lattice-supported candidates up to 35 degrees. Extremely strong peaks have a safety-valve path through the filter. If orientation consensus is weak, fall back rather than imposing a confident but unsupported angle.

The soft prior is intentional: real devices may have cross-capacitance, finite tunnel coupling, nonlinearity, or an imperfect fitted basis. A hard equality between the lattice formula and a measured angle would overstate what the image identifies.

## 1.5 Challenge 1 evaluation design

The primary held-out benchmark uses 120 default-configuration images generated with seed 2026. The baseline detection threshold is 4.0 and was fixed from a separate 100-image validation split with seed 999. All methods in the following table use the same held-out images.

- **Macro pixel P/R/F1/IoU:** compute each image's foreground metrics, then average across images. This weights scenes equally.
- **Micro pixel P/R/F1/IoU:** aggregate foreground pixel counts across the dataset, then compute the metric.
- **Object precision/recall:** label 8-connected target and predicted components; a target object is found if any predicted foreground overlaps it. A predicted component is counted as correct if it overlaps target foreground.
- **FP pixels:** number of predicted foreground pixels outside the target mask.

### Default test results

| Method | Macro precision | Macro recall | Macro F1 | Macro IoU | Object precision | Object recall | FP pixels |
|---|---:|---:|---:|---:|---:|---:|---:|
| Initial matched filter | 0.6400 | **0.9176** | 0.7460 | 0.6078 | 0.7184 | **0.9308** | 4,233 |
| Matched filter + periodicity | 0.6833 | 0.9015 | 0.7692 | 0.6389 | 0.8784 | 0.9124 | 3,339 |
| Wide-angle + periodicity (negative control) | 0.5881 | 0.8583 | 0.6842 | 0.5345 | 0.8222 | 0.8959 | 5,338 |
| Charge-quantized slope + angle consensus | **0.7094** | 0.8730 | **0.7786** | **0.6490** | **0.9947** | 0.9018 | **2,855** |

Aggregate micro metrics are shown separately because they weight images by foreground-pixel count rather than weighting each image equally:

| Method | Micro precision | Micro recall | Micro F1 | Micro IoU |
|---|---:|---:|---:|---:|
| Initial matched filter | 0.6440 | 0.9207 | 0.7579 | 0.6102 |
| Matched filter + periodicity | 0.6929 | 0.9055 | 0.7850 | 0.6461 |
| Wide-angle + periodicity | 0.5731 | 0.8616 | 0.6884 | 0.5248 |
| Charge-quantized slope + angle consensus | **0.7188** | 0.8773 | **0.7901** | **0.6531** |

**Interpretation.** Periodicity raises macro F1 by 0.0232 absolute over the initial matched filter and cuts FP pixels by 21.1%. The charge-slope method raises F1 by 0.0326 over the initial method and 0.0094 over periodicity alone; it cuts FP pixels by 32.6% relative to the initial detector. The trade-off is lower pixel and object recall than the initial high-recall detector. The wide-angle negative control shows that simply adding many orientations creates more multiple-comparison false positives; the angle-consensus and charge-lattice reasoning matter.

## 1.6 Orientation stress test

The optional generator `solutions/generate_charge_stress.py` changes only the stick-angle rule for an evaluation counterfactual. It uses broader charging-line angle parameters (`angle_alpha=angle_beta=0.7`), derives each scene's stick angle from the generated charge-lattice basis, and leaves `csd/config.py` and the default benchmark untouched. Rejection sampling keeps only lattices with crossing sine at least 0.5 and scenes with 6–30 visible sticks. This conditioning avoids a misleading macro score dominated by empty scenes.

The held-out stress split contains 120 images with seed 404; the validation split contains 100 images with seed 303. On the stress test, the mean stick-angle range is 16.8–74.1 degrees; the 10th, 50th, and 90th percentiles are 24.9, 46.9, and 65.5 degrees. About 46.7% of images are outside the baseline bank's ±11.46-degree range around 45 degrees.

| Method | Macro precision | Macro recall | Macro F1 | Macro IoU | Object precision | Object recall | FP pixels |
|---|---:|---:|---:|---:|---:|---:|---:|
| Initial matched filter | 0.6479 | 0.8614 | 0.7307 | 0.5866 | 0.7212 | 0.8931 | 4,476 |
| Matched filter + periodicity | 0.6849 | 0.8477 | 0.7486 | 0.6106 | 0.8864 | 0.8703 | 3,643 |
| Wide-angle + periodicity | 0.6093 | 0.8637 | 0.7002 | 0.5502 | 0.8168 | **0.9002** | 5,814 |
| Charge-quantized slope + angle consensus | **0.7335** | **0.8662** | **0.7870** | **0.6582** | **0.9590** | 0.8926 | **3,005** |

| Method | Micro precision | Micro recall | Micro F1 | Micro IoU |
|---|---:|---:|---:|---:|
| Initial matched filter | 0.6473 | 0.8509 | 0.7353 | 0.5813 |
| Matched filter + periodicity | 0.6896 | 0.8381 | 0.7566 | 0.6085 |
| Wide-angle + periodicity | 0.5896 | 0.8649 | 0.7012 | 0.5399 |
| Charge-quantized slope + angle consensus | **0.7354** | 0.8650 | **0.7950** | **0.6597** |

**Interpretation.** The charge-slope method improves stress-test F1 by 0.0563 over the initial detector and 0.0384 over periodicity alone; it reduces FP pixels by 32.9% relative to the initial detector. This supports robustness to a broader orientation distribution inside this simulator model. It is not evidence that all real devices share the same electrostatic model or lever arms.

---

# 2. Challenge 2 — maximize contrast while tracking drift

## 2.1 Objective and public-API boundary

A fresh device has a hidden contrast landscape over barrier gates `g1`, `g3`, and `g5`. These barriers both change stick contrast and move the pattern in the plunger plane `(g2, g4)`. The plungers set the centre of the measurement window. A good optimizer therefore must do two things: explore barrier space and keep useful sticks in view.

Both implementations in `solutions/optimization.py` use only participant-facing `experiment.start`, `experiment.extent`, and `experiment.measure`. They do not read the simulator's hidden contrast model, drift matrix, region labels, or optimum. `reveal()` appears only in the post-run evaluator.

The image objective is the strongest detector-derived stick amplitude: matched response divided by the selected template's gain, using `top_k=1`. This is closer to “maximum interdot contrast” than whole-image standard deviation, which is diluted by background and can prefer a region with more pixels. The score is still an image-derived proxy, not a direct measurement of the simulator's internal contrast factor.

## 2.2 Full-frame reference optimizer

`ContrastOptimizer` is the comparison method for Challenge 2.

1. Measure the starting full frame and six orthogonal barrier probes (±0.04 V on each barrier axis under default bounds).
2. Register consecutive full-frame edge maps. Preprocessing removes row medians, smooths the image, and uses gradient magnitude so local contrast changes have less effect on the geometric reference. Bounded cross-correlation estimates image motion; subpixel parabolic interpolation refines the peak. Ambiguous registrations trigger a repeated frame and averaging.
3. Combine image motion with the known change in commanded plunger position to estimate barrier-induced drift. Fit an empirical robust model from the observed barriers `(g1,g3,g5)` to the displacement `(Δg2,Δg4)`. The linear model is used first; a regularized quadratic model is enabled when enough observations span the quadratic terms.
4. Explore 48 scrambled Sobol barrier samples in nearest-neighbour order. Interpolate large moves so each barrier step is no larger than 0.08 V; intermediate scans are scored as well as Sobol endpoints.
5. Take two spatially separated high-scoring starts into coordinate-pattern refinement. Local comparisons and final confirmation use repeated frames to reduce measurement noise.

This full-frame strategy gives a strong, robust reference but integrates 150×150 = 22,500 pixels on every full-resolution scan.

## 2.3 Zoom optimizer: fit displacement, predict ROI positions

`ZoomedContrastOptimizer` changes the measurement design rather than changing the contrast objective.

### A. Find candidate windows from one overview

One full-frame overview is passed through the periodic detector. Eligible candidate centres are converted from image coordinates to the experiment's absolute `(g2,g4)` extent. Near-duplicate maxima are removed; farthest-point selection chooses up to four spatially separated centres. This uses measured stick locations, not hidden spatial-region labels. The overview also anchors later drift registration.

### B. Calibrate barrier-induced displacement with full frames

Narrow ROI crops can be ambiguous for image registration, so drift calibration is performed with full-frame images before the optimizer relies on small windows. The design contains six orthogonal ±0.04-V probes and five Sobol-distributed barrier points spanning approximately ±0.45 V under the default bounds. Short intermediate moves (≤0.08 V per barrier) maintain image overlap.

The zoom tracker registers each calibration image to the initial overview, rather than adding pairwise shifts indefinitely. If `p` is the commanded panner displacement and `s` is measured image motion relative to the origin, the absolute barrier drift observation is `s + p`. Those pairs train a robust, regularized model that can include quadratic terms:

\[
\widehat{\Delta p}(b)=\widehat A b + \text{quadratic terms},\qquad b=(g_1,g_3,g_5).
\]

At a new barrier setting `b`, ROI `i` is scanned at

\[
(g_2,g_4)_i=(g_{2,0},g_{4,0})+\widehat{\Delta p}(b)+\text{ROI-offset}_i.
\]

This predicts where the original measured patch should land after the barrier-induced drift and adds the fixed offset to put that patch inside the zoom window.

### C. Search locally, then refine

The default ROI span is 0.12 V per axis at 2 mV per pixel, or 60×60 = 3,600 pixels per window. With four ROIs, an ordinary barrier candidate uses 14,400 ROI pixels rather than 22,500 pixels for one full frame (36% fewer before calibration overhead). The optimizer evaluates its Sobol/path candidates at the predicted ROIs, selects high-scoring `(ROI, barrier)` pairs, locally refines two starts, and confirms the best point with repeated ROI scans.

The full-frame calibration and initial overview are included in the measured pixel budget. The benchmark's separate oracle measurements are not included. Since the optimizer visits multiple ROIs per barrier setting, it makes more scan calls than the full-frame version even though it integrates fewer pixels overall.

## 2.4 Challenge 2 evaluation design

`solutions/evaluate_optimization.py` runs each optimizer on a fresh experiment with the same device seed and optimizer seed. For a fair pair, the two optimizers see independently recreated but identically seeded devices. After optimization has returned, the evaluator calls `reveal()` and performs three full-frame repeat measurements both at the found point and at the post-run revealed optimum.

The reported accuracy measure is

\[
\text{score ratio}=\frac{\text{mean repeated image score at found point}}
{\text{mean repeated image score at the revealed optimum}}.
\]

This oracle comparison is strictly post-run; it is not part of candidate selection. Finite noisy image scores can produce a measured ratio above 1.0. It is a reproducible check of the detector-derived objective, not a claim that the physical system exceeds its true maximum.

## 2.5 Full-frame versus zoom results

The checked-in `results/optimization_comparison.csv` contains 10 paired fresh-device seeds (0–9), optimizer seed 0, and 48 global Sobol samples. Barrier distance is Euclidean distance in the three-dimensional `(g1,g3,g5)` space. Pixel cost uses the challenge convention of 1 ms per pixel. Time below is a proxy; it does not include fixed overhead per scan call.

| Optimizer | Mean score ratio | Median score ratio | Range | Median barrier distance | Median scan calls | Median pixels | 1-ms/pixel estimate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full-frame global + local | 0.874 | 0.857 | 0.745–1.121 | 0.216 V | 334.5 | 7,526,250 | 125.4 min |
| Calibrated ROI zoom | 0.822 | 0.842 | 0.357–1.121 | 0.248 V | 792 | 3,626,100 | 60.4 min |

The median zoom run integrates 51.8% fewer pixels, saving about 65.0 minutes under the stated per-pixel rate. The median score ratio is 0.0155 lower than the full-frame median. Zoom uses about 2.37 times as many scan calls, so instruments with significant per-call latency may realize less wall-clock savings than the pixel-only estimate. The lowest zoom score ratio (0.357) is materially worse than the full-frame minimum (0.745); ROI coverage, registration error, and model extrapolation remain meaningful risks.

**Meaning.** This is a real efficiency–robustness trade-off, not a universal win in both metrics. The method is useful when per-pixel acquisition time dominates and the image-derived drift model is accurate enough to keep the selected patches in view. A practical instrument should include occasional full-frame refreshes or uncertainty-triggered ROI expansion.

---

# 3. Reproduction guide and code map

## 3.1 Re-run the data and score tables

Run from the repository root in the project's `uv` environment:

```bash
# Default detection validation and held-out test splits.
uv run python starter/stage1_detection/generate_data.py --n 100 --out data/val  --seed 999
uv run python starter/stage1_detection/generate_data.py --n 120 --out data/test --seed 2026
uv run python -m solutions.evaluate_detection \
  --dataset data/test --method all --csv results/detection_comparison.csv

# Separate counterfactual charge-slope stress split.
uv run python -m solutions.generate_charge_stress --n 100 --out data/charge_val  --seed 303
uv run python -m solutions.generate_charge_stress --n 120 --out data/charge_test --seed 404
uv run python -m solutions.evaluate_detection \
  --dataset data/charge_test --method all --csv results/charge_stress_comparison.csv

# Matched 10-device optimizer benchmark.
uv run python -m solutions.evaluate_optimization \
  --count 10 --start-seed 0 --optimizer-seed 0 --global-samples 48 \
  --method all --csv results/optimization_comparison.csv

# Focused unit/API tests.
uv run python -m pytest -q
```

The optional stress generator does not alter the organiser's default simulator. Keep validation and test seeds separate when changing thresholds or angular tolerances. `--method all` compares `matched-filter`, `periodic`, `wide-periodic`, and `charge-quantized`.

## 3.2 Relevant files

| Path | Role |
|---|---|
| `solutions/detection.py` | Matched-filter template bank, candidate angles/centres, binary masks, contrast proxy, pixel metrics. |
| `solutions/periodicity.py` | Integer charge-lattice RANSAC, periodic filtering, dual-basis slope hypotheses, wide-angle charge-slope detector. |
| `solutions/evaluate_detection.py` | Pixel, object, lattice-fit, and false-positive comparison CLI. |
| `solutions/generate_charge_stress.py` | Seeded orientation stress-set generator, with populated-scene rejection sampling. |
| `solutions/optimization.py` | Full-frame optimizer, absolute and incremental image registration, robust displacement regression, ROI selection and zoom search. |
| `solutions/evaluate_optimization.py` | Paired full-frame/zoom evaluation; uses reveal only after optimization returns. |
| `tests/test_solution.py` | Detector, lattice-slope, optimizer API, and drift-registration tests. |
| `results/detection_comparison.csv` | Default held-out Challenge 1 metrics. |
| `results/charge_stress_comparison.csv` | Separate orientation stress-test metrics. |
| `results/optimization_comparison.csv` | Paired 10-device Challenge 2 metrics and pixel budgets. |

---

# 4. Limitations and next steps

1. **Charge-axis identifiability.** The capacitance matrix, charge labels, and polarity are not uniquely recoverable from an unlabeled image. On real data, measure lever arms/capacitances or use controlled gate-bias response before relying more strongly on the charge-slope prior.
2. **Stress-test scope.** The orientation stress data are a transparent simulator counterfactual with conditioned crossing angles and visible-stick counts. They do not reproduce all experimental effects, such as nonlinear lever arms, tunnel-coupling changes, hysteresis, or nonuniform noise.
3. **Detector calibration.** Stick appearance can vary with device, scan resolution, and contrast. A local contrast-to-noise estimate with uncertainty could reduce selection bias and improve transfer to different stick widths/blur.
4. **Drift-model extrapolation.** Quadratic regression is empirical. Large unobserved shifts, scan-window boundaries, or weak features can invalidate its prediction. Reacquisition and larger windows should be triggered by registration uncertainty.
5. **ROI coverage.** Four windows are a budget choice, not guaranteed coverage of the eventual best spatial patch. Adaptive expansion, periodic full-frame checks, and a per-pixel/per-call cost model would make zoom scheduling more reliable.
6. **Search strategy.** The global Sobol plus coordinate-pattern search is interpretable but does not model uncertainty in the contrast landscape. A trust-region or Bayesian strategy could allocate measurements more efficiently, provided its uncertainty estimates are calibrated to noisy image scores.

## Background reference

W. G. van der Wiel et al., “Electron transport through double quantum dots,” *Reviews of Modern Physics* 75, 1 (2003), DOI [10.1103/RevModPhys.75.1](https://doi.org/10.1103/RevModPhys.75.1). The constant-interaction model motivates the dependence of charge-boundary slopes on capacitances and charging energies; it does not establish a universal 45-degree interdot orientation.