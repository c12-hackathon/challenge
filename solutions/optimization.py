"""Budget-aware contrast search using only the participant-facing experiment API.

The optimizer combines three experimentally motivated pieces:

* empirical plunger recentering from image registration (barrier changes move
  the CSD in the plunger plane);
* a scrambled Sobol design for global exploration of the three barrier gates;
* local coordinate-pattern refinement around distinct high-scoring regions.

No simulator internals, hidden optimum, or ``reveal()`` calls are used here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, log2

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate
from scipy.stats import qmc

from csd import BARRIERS

from .detection import InterdotDetector

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class OptimizationResult:
    """Summary of an optimization run.

    ``working_point`` is ready to pass to ``exp.measure``. ``history`` contains
    all evaluated barrier vectors and their image-derived contrast scores; it is
    useful for plots and reproducibility, and contains no simulator state.
    """

    working_point: dict[str, float]
    score: float
    initial_score: float
    n_measurements: int
    n_pixels: int
    history: FloatArray = field(repr=False)

    @property
    def barriers(self) -> dict[str, float]:
        return {gate: self.working_point[gate] for gate in BARRIERS}


class _DriftTracker:
    """Estimate barrier-to-plunger drift from consecutive measured images."""

    def __init__(
        self,
        experiment,
        *,
        step_h: float,
        step_v: float,
        barrier_scale: float,
    ) -> None:
        self.experiment = experiment
        self.start = dict(experiment.start)
        self.step_h = float(step_h)
        self.step_v = float(step_v)
        self.barrier_scale = float(barrier_scale)

        self._barriers: list[FloatArray] = []
        self._drifts: list[FloatArray] = []
        self._previous_barriers: FloatArray | None = None
        self._previous_pan: FloatArray | None = None
        self._previous_drift: FloatArray | None = None
        self._previous_registration: FloatArray | None = None

    @staticmethod
    def _registration_image(image: npt.ArrayLike) -> FloatArray:
        """Suppress row stripes, then register the stable edge/ridge geometry."""
        frame = np.asarray(image, dtype=np.float64)
        row_centered = frame - np.median(frame, axis=1, keepdims=True)
        smoothed = gaussian_filter(row_centered, sigma=1.0)
        grad_y, grad_x = np.gradient(smoothed)
        edges = np.hypot(grad_x, grad_y)
        edges -= float(np.mean(edges))
        return edges / max(float(np.std(edges)), 1e-12)

    @staticmethod
    def _registration_shift(reference: FloatArray, moving: FloatArray) -> tuple[FloatArray, float]:
        """Estimate a small 2-D translation by bounded, unnormalized correlation.

        Phase-only correlation can lock onto a different bright patch when a
        regional contrast changes sharply. A bounded real-space cross-correlation
        uses the persistent edge geometry and keeps the estimate near the expected
        small scan-to-scan drift.
        """
        corr = correlate(reference, moving, mode="full", method="fft")
        height, width = reference.shape
        shift_limit = min(60, height // 2 - 1, width // 2 - 1)
        rows = np.arange(corr.shape[0]) - (moving.shape[0] - 1)
        cols = np.arange(corr.shape[1]) - (moving.shape[1] - 1)
        allowed = (np.abs(rows[:, None]) <= shift_limit) & (np.abs(cols[None, :]) <= shift_limit)
        bounded = np.where(allowed, corr, -np.inf)
        row, col = np.unravel_index(int(np.argmax(bounded)), bounded.shape)
        peak = float(bounded[row, col])
        runner_up = bounded.copy()
        runner_up[max(0, row - 3) : row + 4, max(0, col - 3) : col + 4] = -np.inf
        second_peak = float(np.max(runner_up))
        confidence = peak / max(second_peak, 1e-12)

        # Subpixel parabolic interpolation around the discrete correlation peak.
        def offset(before: float, center: float, after: float) -> float:
            denominator = before - 2.0 * center + after
            if abs(denominator) < 1e-12:
                return 0.0
            return float(np.clip(0.5 * (before - after) / denominator, -0.5, 0.5))

        dy = float(rows[row])
        dx = float(cols[col])
        if 0 < row < corr.shape[0] - 1:
            dy += offset(corr[row - 1, col], corr[row, col], corr[row + 1, col])
        if 0 < col < corr.shape[1] - 1:
            dx += offset(corr[row, col - 1], corr[row, col], corr[row, col + 1])
        return np.array([dy, dx], dtype=np.float64), confidence

    def _features(self, barriers: npt.ArrayLike, quadratic: bool) -> FloatArray:
        b = np.asarray(barriers, dtype=np.float64) / self.barrier_scale
        if not quadratic:
            return b
        b1, b2, b3 = b
        return np.array(
            [b1, b2, b3, b1 * b1, b2 * b2, b3 * b3, 2 * b1 * b2, 2 * b1 * b3, 2 * b2 * b3],
            dtype=np.float64,
        )

    def _model(self) -> tuple[FloatArray, bool]:
        """Fit a regularized linear/quadratic empirical drift model."""
        if not self._barriers:
            return np.zeros((3, 2), dtype=np.float64), False

        barriers = np.asarray(self._barriers)
        # The quadratic terms are only admitted after the sampled design spans
        # them. This prevents an ill-conditioned early fit from panning the scene
        # off-screen before the three lever arms have been measured.
        candidates = np.column_stack([self._features(b, quadratic=True) for b in barriers]).T
        quadratic = len(barriers) >= 12 and np.linalg.matrix_rank(candidates, tol=1e-5) == 9
        design = candidates if quadratic else barriers / self.barrier_scale
        target = np.asarray(self._drifts)

        # A small ridge term handles correlated samples along a scan path. Huber-
        # style reweighting prevents an occasional bad registration from
        # dominating the lever-arm estimate.
        penalty = 1e-3 if quadratic else 1e-5
        weights = np.ones(len(design), dtype=np.float64)
        coefficients = np.zeros((design.shape[1], 2), dtype=np.float64)
        for _ in range(3):
            weighted_design = design * weights[:, None]
            lhs = design.T @ weighted_design + penalty * np.eye(design.shape[1])
            rhs = design.T @ (target * weights[:, None])
            coefficients = np.linalg.solve(lhs, rhs)
            residual = np.linalg.norm(target - design @ coefficients, axis=1)
            median = float(np.median(residual))
            scale = max(1.4826 * median, 2 * min(self.step_h, self.step_v))
            weights = np.minimum(1.0, 2.5 * scale / np.maximum(residual, 1e-12))
        return coefficients, quadratic

    def predict_drift(self, barriers: npt.ArrayLike) -> FloatArray:
        """Predict the (g2, g4) translation in volts from observed measurements."""
        coefficients, quadratic = self._model()
        if not self._barriers:
            return np.zeros(2, dtype=np.float64)
        prediction = self._features(barriers, quadratic) @ coefficients
        # A loose physical guard protects the scan from a transient regression
        # outlier; the default landscape's drift is generally within this box.
        return np.clip(prediction, -1.0, 1.0)

    def working_point(self, barriers: npt.ArrayLike) -> tuple[dict[str, float], FloatArray]:
        pan = self.predict_drift(barriers)
        point = dict(self.start)
        point.update({gate: float(value) for gate, value in zip(BARRIERS, barriers)})
        point["g2"] = float(self.start["g2"] + pan[0])
        point["g4"] = float(self.start["g4"] + pan[1])
        return point, pan

    def observe(
        self,
        image: npt.ArrayLike,
        barriers: npt.ArrayLike,
        pan: npt.ArrayLike,
        *,
        force: bool = False,
    ) -> bool:
        """Use relative image registration to update the drift model.

        If the image moves by ``s`` between scans while the panner changes by
        ``dp``, the physical barrier-induced drift changed by ``s + dp``. Relative
        registration is used rather than comparing against the initial frame:
        local scans have much more stable contrast patterns. Return ``False``
        without changing tracker state if the peak is ambiguous, so the caller can
        average one more frame at the same working point and try again.
        """
        current_barriers = np.asarray(barriers, dtype=np.float64)
        current_pan = np.asarray(pan, dtype=np.float64)
        registration = self._registration_image(image)

        if self._previous_registration is None:
            current_drift = np.zeros(2, dtype=np.float64)
        else:
            shift_yx, confidence = self._registration_shift(
                self._previous_registration, registration
            )
            if confidence < 1.18 and not force:
                return False
            image_motion = np.array(
                [-shift_yx[1] * self.step_h, -shift_yx[0] * self.step_v],
                dtype=np.float64,
            )
            delta_pan = current_pan - self._previous_pan
            delta_barriers = current_barriers - self._previous_barriers

            # Large, physically implausible jumps are usually a registration
            # ambiguity (e.g. a faint patch disappeared). Ask for an averaged
            # frame before using such a shift; on the forced retry, keep the
            # empirical drift prediction rather than fitting the outlier.
            max_motion = 0.08 + 2.2 * float(np.sum(np.abs(delta_barriers)))
            implausible = (
                not np.isfinite(image_motion).all()
                or np.linalg.norm(image_motion) > max_motion
                or np.max(np.abs(image_motion)) > 0.30
            )
            if implausible and not force:
                return False
            if implausible:
                image_motion = np.zeros(2, dtype=np.float64)

            current_drift = self._previous_drift + image_motion + delta_pan

        self._barriers.append(current_barriers.copy())
        self._drifts.append(np.asarray(current_drift, dtype=np.float64))
        self._previous_barriers = current_barriers.copy()
        self._previous_pan = current_pan.copy()
        self._previous_drift = np.asarray(current_drift, dtype=np.float64)
        self._previous_registration = registration
        return True


class ContrastOptimizer:
    """Global-plus-local optimizer for the default C12 5-gate challenge.

    All objective measurements use a full-resolution CSD frame and the detector's
    strongest normalized interdot matched-filter response. The full frame also supports empirical
    drift tracking, so the g2/g4 panners are recentered while the barriers move.
    """

    def __init__(
        self,
        *,
        barrier_bounds: tuple[float, float] = (-0.5, 0.5),
        scan_span: float = 0.3,
        scan_step: float = 2e-3,
        global_samples: int = 48,
        max_path_step: float = 0.08,
        local_starts: int = 2,
        local_rounds: int = 4,
        seed: int = 0,
    ) -> None:
        low, high = map(float, barrier_bounds)
        if not low < high:
            raise ValueError("barrier_bounds must be an increasing pair")
        if not low <= 0.0 <= high:
            raise ValueError("barrier_bounds must include the zero-barrier start point")
        if scan_span <= 0 or scan_step <= 0:
            raise ValueError("scan_span and scan_step must be positive")
        if global_samples < 1 or max_path_step <= 0:
            raise ValueError("global_samples and max_path_step must be positive")
        if local_starts < 1 or local_rounds < 1:
            raise ValueError("local_starts and local_rounds must be positive")

        self.bounds = (low, high)
        self.scan_span = float(scan_span)
        self.scan_step = float(scan_step)
        self.global_samples = int(global_samples)
        self.max_path_step = float(max_path_step)
        self.local_starts = int(local_starts)
        self.local_rounds = int(local_rounds)
        self.seed = int(seed)
        self.detector = InterdotDetector(
            step_h=self.scan_step,
            step_v=self.scan_step,
            threshold=1.0 if self.scan_step >= 3e-3 else 4.0,
            n_lengths=4,
            n_widths=3,
            angle_offsets=(-0.15, 0.0, 0.15),
        )

    def _sample_barriers(self) -> FloatArray:
        power = ceil(log2(self.global_samples))
        unit = qmc.Sobol(d=3, scramble=True, seed=self.seed).random_base2(power)
        unit = unit[: self.global_samples]
        low, high = self.bounds
        return low + unit * (high - low)

    def _ordered_path(self) -> list[FloatArray]:
        """Greedy nearest-neighbour route with short interpolated tracking steps."""
        remaining = [p for p in self._sample_barriers()]
        current = np.zeros(3, dtype=np.float64)
        path: list[FloatArray] = []
        while remaining:
            index = int(np.argmin([np.linalg.norm(point - current) for point in remaining]))
            target = remaining.pop(index)
            n_step = max(1, ceil(float(np.max(np.abs(target - current))) / self.max_path_step))
            for step in range(1, n_step + 1):
                path.append(current + (target - current) * (step / n_step))
            current = target
        return path

    def _record_score(self, image: npt.ArrayLike) -> float:
        # The challenge defines the objective as the strongest interdot contrast,
        # not the scene-wide mean. Averaging several frame maxima is handled in
        # the local-refinement/final confirmation steps.
        return self.detector.contrast_score(image, top_k=1)

    def optimize(self, experiment, *, verbose: bool = False) -> OptimizationResult:
        """Tune one fresh experiment and return the best measured working point.

        ``experiment`` is used only through its documented ``start`` and
        ``measure`` API. It must use the default 0.3 V / 2 mV scan distribution
        (or matching ``scan_span``/``scan_step`` supplied to this optimizer).
        """
        lo, hi = self.bounds
        barrier_scale = max(abs(lo), abs(hi), 1e-6)
        tracker = _DriftTracker(
            experiment,
            step_h=self.scan_step,
            step_v=self.scan_step,
            barrier_scale=barrier_scale,
        )
        history: list[tuple[float, float, float, float]] = []
        n_measurements = 0
        current_barriers = np.zeros(3, dtype=np.float64)

        def acquire(barriers: npt.ArrayLike) -> tuple[float, dict[str, float]]:
            nonlocal n_measurements, current_barriers
            b = np.asarray(barriers, dtype=np.float64)
            point, pan = tracker.working_point(b)
            image = experiment.measure(
                **point,
                span_h=self.scan_span,
                span_v=self.scan_span,
                step_h=self.scan_step,
                step_v=self.scan_step,
            )
            n_frames = 1
            registered = tracker.observe(image, b, pan)
            if not registered:
                # Ambiguous motion estimates are rare but can occur while the
                # sticks are close to the noise floor. Average a second frame at
                # the same point before attempting to update the drift model.
                second = experiment.measure(
                    **point,
                    span_h=self.scan_span,
                    span_v=self.scan_span,
                    step_h=self.scan_step,
                    step_v=self.scan_step,
                )
                image = (np.asarray(image) + np.asarray(second)) / 2.0
                n_frames += 1
                registered = tracker.observe(image, b, pan)
                if not registered:
                    tracker.observe(image, b, pan, force=True)
            score = self._record_score(image)
            n_measurements += n_frames
            current_barriers = b.copy()
            history.append((float(b[0]), float(b[1]), float(b[2]), score))
            return score, point

        def travel_and_score(target: npt.ArrayLike) -> float:
            """Measure each short move so drift tracking never loses overlap."""
            nonlocal current_barriers
            target = np.asarray(target, dtype=np.float64)
            delta = target - current_barriers
            n_step = max(1, ceil(float(np.max(np.abs(delta))) / self.max_path_step))
            score = float("nan")
            origin = current_barriers.copy()
            for step in range(1, n_step + 1):
                position = origin + delta * (step / n_step)
                score, _ = acquire(position)
            return score

        # Start scan establishes the reference image and the zero-drift origin.
        initial_score, _ = acquire(current_barriers)
        best_barriers = current_barriers.copy()
        best_score = initial_score

        # Small orthogonal probes identify the local lever arms before the global
        # space-filling walk. The returned images are still ordinary observations.
        probe = min(0.04, (hi - lo) / 12)
        calibration = [
            np.array([probe, 0.0, 0.0]),
            np.array([0.0, probe, 0.0]),
            np.array([0.0, 0.0, probe]),
            np.array([-probe, 0.0, 0.0]),
            np.array([0.0, -probe, 0.0]),
            np.array([0.0, 0.0, -probe]),
        ]
        for b in calibration:
            score = travel_and_score(b)
            if score > best_score:
                best_score, best_barriers = score, b.copy()

        # Global exploration. Interpolated points are scored as well: this makes
        # the route informative rather than charging pixels only for navigation.
        for b in self._ordered_path():
            score = travel_and_score(b)
            if score > best_score:
                best_score, best_barriers = score, np.asarray(b).copy()
        global_best = max(history, key=lambda item: item[3])
        if global_best[3] > best_score:
            best_score = global_best[3]
            best_barriers = np.asarray(global_best[:3], dtype=np.float64)
        if verbose:
            print(f"global phase: {len(history)} scored scans, best score {best_score:.2f}")

        # Select distinct high-scoring starts so multiple spatially separated
        # contrast regions can be refined, not just the first local maximum.
        scored = sorted(history, key=lambda item: item[3], reverse=True)
        starts: list[FloatArray] = []
        min_separation = 0.18 * (hi - lo)
        for g1, g3, g5, score in scored:
            candidate = np.array([g1, g3, g5], dtype=np.float64)
            if all(np.linalg.norm(candidate - chosen) >= min_separation for chosen in starts):
                starts.append(candidate)
                if len(starts) >= self.local_starts:
                    break
        if not starts:
            starts = [best_barriers.copy()]

        for start_barriers in starts:
            # Re-measure the seed to reduce winner's-curse bias from the global
            # sweep's single noisy frame.
            seed_score = float(np.mean([travel_and_score(start_barriers) for _ in range(2)]))
            point = start_barriers.copy()
            point_score = seed_score
            step_size = 0.08 * (hi - lo)
            tolerance = 0.10

            for _ in range(self.local_rounds):
                if step_size < 0.01 * (hi - lo):
                    break
                proposals: list[tuple[FloatArray, float]] = []
                for axis in range(3):
                    for direction in (-1.0, 1.0):
                        trial = point.copy()
                        trial[axis] = np.clip(trial[axis] + direction * step_size, lo, hi)
                        if np.allclose(trial, point, atol=1e-10):
                            continue
                        # Two frames stabilize the fine-scale comparison; the
                        # global search is deliberately single-frame to save time.
                        trial_score = float(np.mean([travel_and_score(trial) for _ in range(2)]))
                        proposals.append((trial, trial_score))
                if proposals:
                    candidate, candidate_score = max(proposals, key=lambda item: item[1])
                    if candidate_score > point_score + tolerance:
                        point, point_score = candidate.copy(), candidate_score
                        if point_score > best_score:
                            best_score, best_barriers = point_score, point.copy()
                    else:
                        step_size *= 0.5
            if point_score > best_score:
                best_score, best_barriers = point_score, point.copy()
            if verbose:
                print(
                    "local phase:",
                    {g: round(float(v), 3) for g, v in zip(BARRIERS, point)},
                    f"score={point_score:.2f}",
                )

        # Return a recentered final working point. Two repeat measurements also
        # give a less noisy final score than selecting the largest single frame.
        final_scores: list[float] = []
        for _ in range(3):
            final_scores.append(travel_and_score(best_barriers))
        final_score = float(np.mean(final_scores))
        final_point, _ = tracker.working_point(best_barriers)

        if verbose:
            print(
                f"final: score={final_score:.2f}, "
                f"barriers={{{', '.join(f'{g}: {final_point[g]:.3f}' for g in BARRIERS)}}}, "
                f"measurements={n_measurements}, pixels={n_measurements * round(self.scan_span / self.scan_step) ** 2:,}"
            )

        return OptimizationResult(
            working_point=final_point,
            score=final_score,
            initial_score=float(initial_score),
            n_measurements=n_measurements,
            n_pixels=n_measurements * round(self.scan_span / self.scan_step) ** 2,
            history=np.asarray(history, dtype=np.float64),
        )


def optimize(
    experiment,
    *,
    seed: int = 0,
    global_samples: int = 48,
    verbose: bool = False,
) -> OptimizationResult:
    """Convenience wrapper for the default challenge limits and scan settings."""
    return ContrastOptimizer(seed=seed, global_samples=global_samples).optimize(
        experiment, verbose=verbose
    )
