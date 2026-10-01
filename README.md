# C12-hackathon

Welcome to C12's hackathon! The goal for you is to learn some aspects of qubit calibration and, as scientists, propose solutions to automate it.

In this repository you will find a simulation of a **5-gate double-quantum-dot (DQD)** device and a two-part challenge built on top of it. The device is imaged through its **charge-stability diagram (CSD)** — a 2D map in the **(g2, g4)** plunger plane. The three barriers **g1, g3, g5** drift the interdots ("sticks") across that plane *and* set their
**contrast**.

Your job is to build tools a real experimentalist would actually use: first to
**detect** the interdots automatically, then to **tune the device** to the point of
best contrast.

> **Read this first — the spirit of the challenge.**
> Approach this like a researcher: the simulator is a *stand-in* for a real machine,
> so the real prize is methods that would **transfer to an actual lab**. There, you
> don't get `reveal()` and every measurement costs time and money — so the
> interesting question is what you'd do with only what a real experimentalist can
> see. Solutions that lean on hidden state, reverse-engineer the noise model, or
> brute-force every pixel win here but teach you nothing about a real device.
> Build something you could defend to a physicist, and justify *why* it works: this
> is graded much more on **scientific approach** than on a leaderboard number.
>
> Treat the repo as a **sandbox**: solve the two challenges on the defaults first,
> then change whatever you can scientifically justify. See
> [How far can I go?](#how-far-can-i-go) for exactly what that means.

---

## The two challenges

1. **Detection** — implement an algorithm that takes a CSD image and outputs the **interdot pixels** (a binary
   mask).
2. **Optimization** — implement an algorithm that finds the gate voltages that **maximize interdot contrast** for any fresh device simulated by the simulator. You may (and probably
   should) reuse your stage-1 detector to build a better objective.

These two challenges are the **mandatory baseline**, and doing them well is
entirely enough. They can also simply be a starting point, though: if you want to,
and as long as it is motivated by a sound **scientific approach**, you are free to
use the backend as a sandbox — explore it, extend the problems, and push things
further in whatever direction you find interesting.

---

## How far can I go?

Think of the repo as a **sandbox with one common baseline**. There are three tiers,
and they are about *reproducibility*, not permission:

1. **Baseline (required).** Solve Challenge 1 & 2 against the **default**
   `csd/config.py`. This is the common ground everyone is measured on, so leave the
   defaults in place for these results.
2. **Sandbox (encouraged).** Beyond the baseline, change *anything* if you can
   justify it scientifically — generator parameters, the simulator physics, the
   contrast model, the problem itself. Just **keep the default baseline run too**,
   and **document what you changed and why**. A well-motivated experiment that
   "fails" is worth more than an unexplained tweak that helps.
3. **One thing to avoid.** Please don't lean on the simulator's hidden state to
   boost your score — e.g. `reveal()` inside your optimizer, reverse-engineering the
   noise model, or brute-forcing every pixel. Reshaping the sandbox for good science
   is very different from gaming the number, and only the latter misses the point.

---

## Installation

Requires Python ≥ 3.10. We use [uv](https://docs.astral.sh/uv/) for environments
and dependencies:

```bash
# install uv once (see the uv docs for other platforms)
curl -LsSf https://astral.sh/uv/install.sh | sh

# create the venv and install the project (uv fetches a compatible Python if needed)
uv sync
```

`uv sync` creates `.venv/` and installs the project plus its dev tooling. Either
prefix commands with `uv run` (e.g. `uv run python starter/...`), or activate the
environment in your terminal:

```bash
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows (PowerShell / cmd)
```

The stack is left to you — pick any framework (PyTorch,
JAX, scikit-learn, ...) and add it with `uv add <package>`.

---

## Step 0 — get to know the backend

Before writing any solution, **play with the provided explorer scripts**. They
are interactive tools whose only purpose is to build your intuition for how the
device behaves — what an interdot looks like, how barriers drift and brighten the
diagram, how noise and window size affect what you measure.

```bash
# Stage 1: generate a dataset first (see Challenge 1 below), then browse it
# with its ground-truth masks overlaid
uv run python starter/stage1_detection/generate_data.py --n 2000 --out data/train
uv run python starter/stage1_detection/explore_data.py --out data/train

# Stage 2: live sliders over the five gates (g1..g5) + scan settings
# (no data needed — the simulator builds a fresh device on the fly)
uv run python starter/stage2_optimization/explore_simulator.py
```

Everything under `starter/` is **illustrative, not prescriptive** — copy it, edit
it, or throw it away.

---

## Challenge 1 — detection

**Goal:** Implement an algorithm that takes a CSD (Charge Stability Diagram) measurement as input and outputs the pixels corresponding to sticks.

**How?** Generate a dataset locally using the generator, then train whatever you like to map `images → masks`.
The data is a plain folder (`images.npy`, `masks.npy`, `sticks.jsonl`,
`meta.json`) — memory-mapped, no exotic dependencies.

```bash
uv run python starter/stage1_detection/generate_data.py --n 2000 --out data/train
uv run python starter/stage1_detection/generate_data.py --n 400  --out data/val --seed 999
```

Generation runs at roughly **1000 samples/minute** (so the 2000+400 above take ~2–3 min).

```python
from csd import load_dataset

ds = load_dataset("data/train")
images, masks = ds["images"], ds["masks"]   # (N, 150, 150); memory-mapped
sticks = ds["sticks"]                        # per-image metadata: positions, width, angle, ...
```

Images are **raw**, exactly as the stage-2 simulator emits them, so
a detector trained here transfers to stage 2.

---

## Challenge 2 — optimization

**Definition:** We call "contrast" of an interdot the absolute value of the difference between the intensity on the stick and the background.

**Goal:** Implement an algorithm that, for any fresh device, finds the gate configuration (g1, g2, g3, g4, g5) corresponding to the inderdot with highest possible contrast.

**How?** Each `new_experiment()` is a **fresh device** with a **hidden, randomised**
contrast sweet-spot. You propose gate voltages, measure images, and keep what
improves — you can track your **budget** (measurements and pixels integrated;
pixels are the proxy for acquisition time).

```python
from csd import new_experiment

exp = new_experiment()                                   # new hidden optimum each call
img = exp.measure(g1=0.0, g2=0.15, g3=0.0, g4=0.15, g5=0.0)  # a raw CSD image
score = img.std()                                        # a simple (weak) objective

# ... your loop: propose gates -> measure -> keep what improves ...

print(exp.reveal())   # hidden optimum + budget — for SELF-CHECK only, not for your algorithm
```

- `g2, g4` **pan** the measurement window; `g1, g3, g5` are the **barriers** that
  set contrast (and drift the sticks, so you may need to re-center `g2, g4`).
- `measure(...)` accepts `span_h, span_v, step_h, step_v` — measure the same
  device through a **wide low-res overview** or a **zoomed fine-step scan**.
- `img.std()` is only a starting objective; a contrast-to-noise ratio computed on
  your **detected** interdot pixels is far less noisy. This is where stage 1 pays
  off.

A baseline coordinate-ascent optimizer (designed to stall in a local optimum — a
starting point to beat) is provided:

```bash
uv run python starter/stage2_optimization/optimize.py
```

> `reveal()` exists so you can **check your own results**. Using it *inside* your
> optimizer defeats the point and is exactly the kind of simulator-exploitation
> we're not looking for.

---
## What you have to submit before the end of the event

A documented repository with:
- your solution to challenges 1 and 2;
- a README.md so that we understand how your code is structured and how to run it;
- the slides you'll be using to present your work to us on Friday.

Thank you very much for your participation and effort!

---
## What we're evaluating

You'll give a **10-minute presentation in English** followed by a **5-minute
Q&A**. You likely won't cover everything in 10 minutes — the Q&A is
where the jury fills the gaps, so be ready to defend your choices.

Marks weigh the **scientific approach** far above raw performance. In order of
importance:

**Scientific approach (the bulk of the grade)**
- **Justify your choices** — *why* this detector architecture, *why* this
  optimization strategy. Motivated decisions matter more than the final score.
- **Metrics** — define how you measure success for both detection and optimization, and use them
  to validate your results.
- **Data management & reproducibility** — clean train/validation/test split,
  fixed seeds, results anyone can re-run.
- **Clean repository** — readable, organised, documented code.
- **Limitations & next steps** — what your solution can't do yet, and how you'd
  improve or extend it.

**Solution quality**
- A **robust interdot detector**, assessed on the metrics *you* present.
- A **contrast-optimization algorithm** that finds the maximum with efficiency.

**Communication**
- Clear, pedagogical delivery: good intro, well-explained concepts.
- Strong visual support: readable slides, well-chosen figures and plots.

**Originality** of the approach is rewarded.

Remember the guiding principle above: solutions are judged on whether they'd
**work on a real device**, not on how thoroughly they exploit this simulator.

---

## Reference solution (implemented)

The two required challenges are implemented without changing the organiser's
`csd/config.py` defaults or reading simulator internals.

### Challenge 1 — training-free, physics-guided pixel detector

`solutions/detection.py` builds a normalized matched-filter bank from the public
appearance ranges: short dark rectangles, the nominal stick angle and its
orientation jitter, and the configured blur. Before filtering, it subtracts a
robust median from each row to suppress the shared horizontal acquisition noise.
Each zero-sum template compares the putative stick to a local background annulus;
non-maximum suppression keeps one centre per feature, and the selected blurred
rectangle footprints form the binary pixel mask. The default threshold is 4
matched-filter noise units. No generated training images, stick metadata, or
hidden state are used at inference time.

The detector also estimates the physical dip amplitude by dividing each matched
response by the selected template's gain. This removes most of the square-root
area advantage of longer/wider sticks and gives stage 2 an objective closer to
"maximum interdot contrast" than image standard deviation.

### Challenge 2 — global search, empirical drift correction, local refinement

`solutions/optimization.py` only calls `experiment.start` and
`experiment.measure`:

1. Take a starting image and six small orthogonal barrier probes to identify the
   local plunger response.
2. Track the shared stick geometry between adjacent scans using a bounded
   cross-correlation of row-corrected edge maps. Ambiguous registrations trigger
   a repeated frame and averaging. A robust linear model is fit first; a
   regularized quadratic model is enabled when the observations span it.
3. Explore the default `[-0.5, 0.5]^3` barrier cube with 48 scrambled Sobol
   points, routed in nearest-neighbour order. Intermediate moves are also scored
   and are limited to 0.08 V per barrier, so the image remains trackable while
   `g2/g4` are recentered from measurements.
4. Refine two spatially separated high-scoring starts by coordinate pattern
   search (±0.08, 0.04, 0.02, then 0.01 V). Local comparisons average two
   frames; the returned point is confirmed with three more frames.

The objective is the largest gain-corrected, detector-derived stick amplitude in
a full-resolution CSD frame. This matches the challenge's maximum-contrast
definition and avoids both scene-wide dilution and the pixel-area bias in a raw
matched-filter response.

### Run the solutions

```bash
# Reproducible held-out detector data. Do not reuse the test seed for tuning.
uv run python starter/stage1_detection/generate_data.py --n 100 --out data/val  --seed 999
uv run python starter/stage1_detection/generate_data.py --n 120 --out data/test --seed 2026
uv run python -m solutions.evaluate_detection --dataset data/test

# One deterministic optimization run; no reveal() call is made by the optimizer.
uv run python - <<'PY'
from csd import new_experiment
from solutions.optimization import ContrastOptimizer

exp = new_experiment(seed=0)
result = ContrastOptimizer(seed=0).optimize(exp)
print("working point:", result.working_point)
print("image-derived score:", result.score)
print("measurements / pixels:", result.n_measurements, result.n_pixels)
PY

# Multi-device post-run self-check (reveal is called only after optimize returns).
uv run python -m solutions.evaluate_optimization \
  --count 10 --start-seed 0 --optimizer-seed 0 \
  --global-samples 48 --csv results/optimization_benchmark.csv
```

Generated `.npy` datasets stay under the ignored `data/` directory. The detector
is analytic, not trained, so the validation split is used only to choose the
segmentation threshold; the seed-2026 test split is held out. The benchmark's
oracle measurements and calls to `reveal()` are separate post-run self-checks,
never inputs to the optimizer, and are not included in its pixel budget.

### Reproducible results

On 120 held-out images generated with seed `2026` (threshold fixed at `4.0` using
100 validation images, seed `999`):

| Metric | Result |
|---|---:|
| Macro pixel precision / recall | 0.640 / 0.918 |
| Macro pixel F1 / IoU | **0.746 / 0.608** |
| Micro pixel precision / recall | 0.644 / 0.921 |
| Micro pixel F1 / IoU | 0.758 / 0.610 |
| 8-connected object precision / recall | 0.718 / 0.931 |

Pixel accuracy is intentionally omitted because the positive class occupies
well under 1% of the pixels. The object metric counts a target stick as found
when any predicted foreground overlaps its 8-connected component.

For stage 2, the checked-in `results/optimization_benchmark.csv` contains 10
fresh devices (seeds 0–9), all using optimizer seed 0 and the default budget.
The metric is the repeated, detector-derived contrast score at the returned
point divided by the same score measured at the revealed optimum **after** the
optimizer has finished. The mean ratio was **0.874**, the median **0.857**, and
the minimum **0.745**. The median search used **334.5 measurements** and
**7,526,250 integrated pixels** (range: 325–397 measurements). The initial
working point had a median score ratio of 0.150. A noisy measured ratio can
occasionally exceed 1; it is a self-check on an image-derived proxy, not a claim
that physical contrast exceeds the hidden maximum.

See `assets/detection_example.png` and `assets/optimization_benchmark.png` for
the held-out detector overlay and ten-device optimization plot. `slides.pptx`
and `slides.md` provide the 10-minute presentation and speaker notes. To
regenerate the editable deck, activate the project environment, install the
optional `python-pptx` generator dependency, and run
`python presentation/create_slides.py`.

### Limitations and next steps

- The detector assumes the known baseline stick scale and near-45-degree
  orientation. A real deployment should estimate these from a small calibration
  set or widen the bank if device geometry changes. Short connector segments and
  exceptionally weak sticks remain its main false-positive/false-negative modes.
- Gain correction reduces, but does not eliminate, shape-dependent selection
  bias in the strongest-stick score. A calibrated local contrast-to-noise
  estimator with uncertainty would make fine tuning more reliable.
- Image registration can become ambiguous if all sticks are near the noise
  floor or move outside the scan. The optimizer detects low-confidence matches
  and repeats them, but it still depends on locally trackable geometry and uses
  a smooth linear/quadratic drift approximation.
- The optimizer is robust but not perfect: a ten-device median score ratio of
  0.857 leaves room for a more sample-efficient Bayesian or trust-region search,
  multi-resolution overviews, and explicit uncertainty-aware stopping.

---

## Reference solution (implemented)

The two required challenges are implemented without changing the organiser's
`csd/config.py` defaults or reading simulator internals.

### Challenge 1 — matched filtering, periodicity, and charge-quantized slopes

`solutions/detection.py` constructs a training-free matched-filter bank from the
public appearance ranges: short dark rectangles, blur, length and width. A robust
row-median subtraction suppresses shared acquisition stripes; each normalized
zero-sum template compares a stick with its local background annulus. The best
response per pixel is peak-suppressed and rasterized using its winning footprint.
The initial baseline bank is intentionally tuned to the organiser's nominal
45-degree simulator geometry.

Two post-processors address false positives and orientation transfer:

1. `PeriodicInterdotDetector` converts detected centres to gate-voltage
   coordinates and RANSAC-fits a two-dimensional integer charge lattice. Pair
   separations near the nominal charging period seed the fit; candidates within
   0.015 V of integer lattice sites are retained (at least four inliers,
   confidence ≥0.10), with a score-18 rescue for isolated or edge-truncated sticks.
   This raises object precision on the default test split.
2. `ChargeConstrainedInterdotDetector` uses a full-angle bank (15-degree samples
   over the unoriented 0–180-degree range), estimates the dominant axial stick
   angle from matched-filter responses, then uses that measured angle as a soft
   tie-breaker between equivalent charge-lattice bases. For a basis matrix `B`,
   charge coordinates are `q = inv(B) (V − V0)`. A charge-transfer boundary
   `q1 − q2 = const` has normal `grad(q1) − grad(q2)`; because charge polarity is
   unlabelled in a grayscale image, both relative-sign hypotheses are evaluated.
   Isolated candidates use a 25-degree tolerance; lattice-supported candidates
   use 35 degrees. A charge slope may replace the morphology angle only when it
   agrees within 15 degrees. No fixed 45-degree rule is imposed.

The electrostatic basis is physically motivated, not universal: constant
chemical-potential slopes depend on gate capacitances and charging energies
(e.g. `dVg2/dVg1 = −Cg1 EC1/(Cg2 ECm)` for one dot's transition). Quantization
alone does not identify the gate-axis labels or determine a unique interdot slope
from an unlabeled image. The implementation therefore uses the dual-lattice
relation as a soft, image-informed prior and reports its limitations. See
van der Wiel et al., *Rev. Mod. Phys.* 75, 1 (2003), DOI
[10.1103/RevModPhys.75.1](https://doi.org/10.1103/RevModPhys.75.1) for the
constant-interaction electrostatic model.

The detector also estimates dip amplitude by dividing a matched response by its
template gain. This reduces the square-root-area advantage of larger templates
and gives Challenge 2 a closer proxy for maximum interdot contrast than image
standard deviation.

### Challenge 2 — barrier-to-plunger model and calibrated zoom search

`solutions/optimization.py` exposes two methods. Both use only participant-facing
`start`, `extent`, and `measure` calls; neither reads simulator fields or calls
`reveal()` during optimization.

- `ContrastOptimizer` is the full-frame reference: it measures six small
  orthogonal barrier probes, tracks full-frame image motion, explores the
  `[-0.5, 0.5]^3` barrier cube with 48 scrambled Sobol points and short
  interpolated moves, then refines two separated high-scoring points.
- `ZoomedContrastOptimizer` takes one full-frame overview, selects up to four
  spatially separated stick windows, and calibrates the map
  `Δ(g2,g4) = A·(g1,g3,g5) + quadratic terms` using full-frame registration.
  Registration is measured relative to the initial overview (to avoid cumulative
  shift error); the predicted displacement recentres each ROI after barrier
  changes. The default ROI is 0.12 V square at 2 mV per pixel (60×60 rather than
  150×150), followed by the same global barrier search and local refinement.
  It spends more scan calls but substantially fewer integrated pixels.

Both score the strongest gain-corrected detected stick, not a frame-wide
statistic. Local comparisons and final confirmation use repeated frames. Under
the challenge's stated 1 ms per pixel, pixel count is the acquisition-time proxy;
fixed per-scan overhead is not included in that estimate.

### Reproduce the evaluations

```bash
# Default Challenge 1 split: the 2026 test seed is held out from tuning.
uv run python starter/stage1_detection/generate_data.py --n 100 --out data/val  --seed 999
uv run python starter/stage1_detection/generate_data.py --n 120 --out data/test --seed 2026
uv run python -m solutions.evaluate_detection \
  --dataset data/test --method all --csv results/detection_comparison.csv

# Optional charge-orientation stress split (does not modify the organiser defaults).
uv run python -m solutions.generate_charge_stress --n 100 --out data/charge_val  --seed 303
uv run python -m solutions.generate_charge_stress --n 120 --out data/charge_test --seed 404
uv run python -m solutions.evaluate_detection \
  --dataset data/charge_test --method all --csv results/charge_stress_comparison.csv

# Paired Challenge 2 comparison on 10 identical fresh-device seeds.
uv run python -m solutions.evaluate_optimization \
  --count 10 --start-seed 0 --optimizer-seed 0 --global-samples 48 \
  --method all --csv results/optimization_comparison.csv
```

`--method all` evaluates `matched-filter`, `periodic`, `wide-periodic`, and
`charge-quantized` on the same images. The stress generator uses broader
charging-line angle parameters, sets stick orientation from its charge-basis
model, and rejects scenes unless the basis crossing sine is at least 0.5 and 6–30
sticks are visible. It is a counterfactual robustness test; it leaves the default
simulator and the default test score unchanged. Generated `.npy` datasets belong
under the ignored `data/` directory.

The detection CLI reports per-image macro pixel precision/recall/F1/IoU, aggregate
micro metrics, connected-component object precision/recall, false-positive and
false-negative pixels, and lattice-fit rate. Pixel accuracy is omitted because
foreground occupies far below 1% of a frame. An object counts as found when any
predicted foreground overlaps its 8-connected target component.

The optimization evaluator creates a fresh independently seeded device per
method, calls `reveal()` only after that method returns, then compares three
repeated image-derived scores at the returned point and the revealed optimum.
Those oracle measurements are evaluation-only and are excluded from the reported
optimizer budget. Pixel time uses the challenge's 1 ms/pixel rate; it excludes
any fixed overhead per scan call.

### Challenge 1 — held-out results and accuracy evolution

Default test: 120 images, seed `2026`; matched-filter threshold `4.0` fixed using
100 validation images, seed `999`. The periodic and charge slope settings were
also selected on validation only. All rows use the same test images.

| Method | Macro P | Macro R | Macro F1 | Macro IoU | Object P/R | FP pixels |
|---|---:|---:|---:|---:|---:|---:|
| Initial matched filter | 0.6400 | **0.9176** | 0.7460 | 0.6078 | 0.7184 / **0.9308** | 4,233 |
| + periodicity | 0.6833 | 0.9015 | 0.7692 | 0.6389 | 0.8784 / 0.9124 | 3,339 |
| Wide-angle + periodicity (negative control) | 0.5881 | 0.8583 | 0.6842 | 0.5345 | 0.8222 / 0.8959 | 5,338 |
| Charge-quantized slope filter | **0.7094** | 0.8730 | **0.7786** | **0.6490** | **0.9947** / 0.9018 | **2,855** |

Periodicity raises macro F1 by 0.0232 over the initial detector and cuts false
positive pixels by 21.1%. The charge-angle method raises F1 by 0.0326 over the
initial detector (0.0094 over periodicity) and cuts false-positive pixels by
32.6%, at the cost of some pixel/object recall. The wide-angle negative control
shows why simply expanding the template bank is not enough: it increases the
multiple-template false-positive burden without a useful slope prior.

The optional charge-consistent stress test uses 120 images, seed `404`, with
broader lattice/angle settings: 46.7% of generated mean stick orientations lie
outside the baseline bank's ±11.46-degree neighbourhood around 45 degrees.
Samples are conditioned to avoid empty scenes (6–30 visible sticks).

| Method | Macro P | Macro R | Macro F1 | Macro IoU | Object P/R | FP pixels |
|---|---:|---:|---:|---:|---:|---:|
| Initial matched filter | 0.6479 | 0.8614 | 0.7307 | 0.5866 | 0.7212 / 0.8931 | 4,476 |
| + periodicity | 0.6849 | 0.8477 | 0.7486 | 0.6106 | 0.8864 / 0.8703 | 3,643 |
| Wide-angle + periodicity | 0.6093 | 0.8637 | 0.7002 | 0.5502 | 0.8168 / **0.9002** | 5,814 |
| Charge-quantized slope filter | **0.7335** | **0.8662** | **0.7870** | **0.6582** | **0.9590** / 0.8926 | **3,005** |

On this stress test, the charge-slope method improves F1 by 0.0563 over the
initial detector and 0.0384 over periodicity alone; false-positive pixels fall
32.9% versus the initial detector. The stress result is deliberately reported
separately from the default benchmark and is not evidence that all real devices
share this exact capacitance model.

### Challenge 2 — measured accuracy versus pixels

`results/optimization_comparison.csv` contains 10 paired seeds (0–9), each method
with optimizer seed 0 and 48 global Sobol samples. The repeated score ratio is
`image-derived score at returned point / score at post-run revealed optimum`.

| Optimizer | Mean ratio | Median ratio | Min–max | Median measurements | Median pixels | Time at 1 ms/pixel |
|---|---:|---:|---:|---:|---:|---:|
| Full-frame global + local | **0.874** | **0.857** | 0.745–1.121 | 334.5 | 7,526,250 | 125.4 min |
| Calibrated ROI zoom | 0.822 | 0.842 | 0.357–1.121 | 792 | 3,626,100 | 60.4 min |

The zoomed strategy uses **51.8% fewer integrated pixels**, reducing the stated
pixel-time estimate by about **65.0 minutes per run**, while its median score ratio
is 0.0155 lower than the full-frame median. It makes roughly 2.4× as many scan
calls because each candidate visits several ROIs; instruments with large fixed
per-scan overhead may realize less wall-clock savings. The zoom method's worst
seed is also weaker, so ROI coverage and drift-model uncertainty remain material
trade-offs—not hidden by the average.

Measured ratios occasionally exceed 1 because the score is estimated from noisy
finite acquisitions. It is a post-run validation proxy, not a claim that the
physical device exceeds its true optimum. The reported pixels and score ratios,
not wall-clock simulator runtime, are the reproducible quantities.

See `assets/detection_example.png`, `assets/detection_comparison.png`, and
`assets/optimization_benchmark.png` for visuals. `slides.pptx` and `slides.md`
provide the 10-minute presentation and speaker notes. Regenerate the editable deck
with `python presentation/create_slides.py` in an environment containing
`python-pptx`.

### Limitations and next steps

- The charge-slope prior is conditional on a constant-interaction model and on
  resolving an unlabeled lattice basis. Quantization alone does not uniquely fix
  the gate-axis labels, capacitances, or interdot slope. Real hardware should
  measure lever arms/capacitances or use known bias-response labels; the code's
  morphology-informed basis selection is a soft estimate, not a universal rule.
- The stress generator is a simulator counterfactual. Its conditioned scenes
  test varied slopes but do not reproduce every experimental effect (curvature,
  tunnel coupling, charge polarity, drift, or nonuniform noise).
- Gain correction reduces, but does not eliminate, shape-dependent selection
  bias in the strongest-stick objective. A calibrated local contrast-to-noise
  estimator with uncertainty would make fine tuning more reliable.
- Full-frame absolute registration is more reliable than tiny-crop registration,
  but it can still fail if the image shifts beyond the search bound or features
  disappear. The drift model extrapolates a quadratic fit; model uncertainty is
  largest near the barrier-space boundaries.
- Zoom trades pixel-time for score reliability: the ten-device median score
  ratio is 0.842 versus 0.857 for full-frame search, and the worst zoomed seed is
  lower. More adaptive ROI discovery, occasional full-frame refreshes, and
  uncertainty-aware sample allocation are natural next improvements.

---

## Repository layout

```
csd/                       # the engine — provided
  generator.py             #   builds a CSD scene and renders it
  simulator.py             #   the tunable device (drift + contrast + panning)
  dataset.py               #   generator -> training folder (challenge 1 data)
  challenge.py             #   optimization harness (challenge 2)
  config.py                #   fixed, organiser-set hyperparameters
starter/                   # illustrative starting points — edit freely
  stage1_detection/        #   generate_data.py, explore_data.py
  stage2_optimization/     #   optimize.py, explore_simulator.py
solutions/                 # reference detector, optimizer, and evaluators
  detection.py             #   matched-filter detector + pixel metrics
  optimization.py          #   API-only drift tracking and contrast search
  evaluate_detection.py    #   held-out mask and object metrics
  evaluate_optimization.py #   post-run, reveal-based self-check
assets/                    # figures used by README and slides
results/                   # reproducible ten-device benchmark CSV
presentation/              # editable-deck generation script
  create_slides.py         #   rebuilds slides.pptx (requires python-pptx)
tests/                     # focused unit and API-boundary tests
data/                      # generated datasets (ignored by git)
slides.pptx                # editable 10-minute presentation
slides.md                  # slide text and speaker notes
pyproject.toml
```

The public API is what `import csd` exposes (`new_experiment`, `load_dataset`,
`generate_dataset`, …). 
