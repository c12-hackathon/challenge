"""Charge-lattice consistency and charge-quantization slope priors.

The CSD contains a sparse set of interdot candidates arranged at crossings of
repeated charge-transition lines. This module fits that periodic point pattern
without using stick metadata, then rejects detections that do not belong to the
best-supported lattice. It also exposes the dual-lattice construction used to
predict the interdot-line direction from two charge-period vectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from csd.config import GENERATOR, GeneratorConfig

from .detection import DetectionResult, InterdotDetector

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


def _det2(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Two-dimensional determinant, without NumPy's deprecated vector cross."""
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    return float(av[0] * bv[1] - av[1] * bv[0])


def _deduplicate(centers_xy: FloatArray, strengths: FloatArray, radius: float) -> IntArray:
    """Keep the strongest candidate in each sub-stick cluster."""
    order = np.argsort(-strengths, kind="stable")
    representatives: list[int] = []
    for index in order:
        if all(
            np.linalg.norm(centers_xy[index] - centers_xy[kept]) > radius
            for kept in representatives
        ):
            representatives.append(int(index))
    return np.asarray(representatives, dtype=np.int64)


def _dual_charge_angles(basis: npt.ArrayLike) -> FloatArray:
    """Return the two charge-transfer slope hypotheses for a lattice basis."""
    vectors = np.asarray(basis, dtype=np.float64)
    if vectors.shape != (2, 2) or abs(np.linalg.det(vectors)) < 1e-12:
        return np.empty(0, dtype=np.float64)
    dual = np.linalg.inv(vectors)
    angles: list[float] = []
    for relative_sign in (1.0, -1.0):
        normal = dual[0] - relative_sign * dual[1]
        if np.linalg.norm(normal) < 1e-12:
            continue
        tangent = np.array([-normal[1], normal[0]])
        angle = float(np.mod(np.arctan2(tangent[1], tangent[0]), np.pi))
        if all(
            abs(((angle - existing + np.pi / 2) % np.pi) - np.pi / 2) > 1e-3 for existing in angles
        ):
            angles.append(angle)
    return np.asarray(angles, dtype=np.float64)


@dataclass(frozen=True)
class ChargeLatticeFit:
    """Fitted two-dimensional charge-period lattice in physical gate volts.

    ``basis[:, 0]`` and ``basis[:, 1]`` are translations between equivalent
    charge-lattice sites. ``origin`` is one fitted site. The inlier/residual
    arrays correspond to ``representative_indices`` in the detector candidate
    list; candidates omitted from that list were duplicate peaks on one stick.
    """

    basis: FloatArray
    origin: FloatArray
    representative_indices: IntArray
    inliers: BoolArray
    residuals: FloatArray
    confidence: float

    @property
    def periods(self) -> FloatArray:
        """Gate-space lengths of the two fitted charge-period vectors."""
        return np.linalg.norm(self.basis, axis=0)

    @property
    def charge_line_angles(self) -> FloatArray:
        """Unoriented direction angles of the two charge-line families (radians)."""
        angles = np.arctan2(self.basis[1, :], self.basis[0, :])
        return np.mod(angles, np.pi)

    @property
    def normal_periods(self) -> FloatArray:
        """Perpendicular spacing between adjacent lines in each family (volts)."""
        area = abs(_det2(self.basis[:, 0], self.basis[:, 1]))
        return np.array([area / self.periods[0], area / self.periods[1]])

    def interdot_angle_hypotheses(self) -> FloatArray:
        """Return charge-quantization interdot slopes, modulo pi.

        Let ``q = inv(basis) @ (V - origin)`` be the inferred charge coordinates.
        An interdot transition changes the two dot occupations oppositely, so a
        line of constant ``q1 - q2`` has normal ``grad(q1) - grad(q2)``. The
        observed lattice has no charge-label polarity, so both relative signs are
        returned rather than silently selecting one.
        """
        return _dual_charge_angles(self.basis)


def fit_charge_lattice(
    centers_xy: npt.ArrayLike,
    strengths: npt.ArrayLike,
    *,
    config: GeneratorConfig = GENERATOR,
    deduplicate_radius: float = 0.020,
    residual_tolerance: float = 0.015,
    min_inliers: int = 4,
    min_cross_sine: float = 0.25,
    max_pair_vectors: int = 180,
    preferred_interdot_angle: float | None = None,
    angle_prior_sigma: float = 0.30,
) -> ChargeLatticeFit | None:
    """RANSAC-fit a two-family periodic lattice to candidate stick centres.

    Pair differences near one nominal charging period seed candidate basis
    vectors. Each non-collinear pair of vectors is scored by how many measured
    centres fall near integer lattice coordinates. A least-squares affine refit
    removes pixel quantization and centre jitter. The fitted model is rejected
    when the points do not support both independent charge directions. If a
    measured interdot angle is supplied, the dual-basis charge-slope equation
    is a soft tie-breaker for otherwise equivalent lattice bases; it does not
    impose a fixed 45-degree orientation.

    This is an image-only inference routine: ``strengths`` are matched-filter
    responses, not stick labels. Pixel/ground-truth metadata must not be passed.
    """
    points = np.asarray(centers_xy, dtype=np.float64)
    scores = np.asarray(strengths, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("centers_xy must have shape (n, 2)")
    if scores.ndim != 1 or len(scores) != len(points):
        raise ValueError("strengths must have one value per candidate centre")
    if not np.isfinite(points).all() or not np.isfinite(scores).all():
        raise ValueError("candidate centres and strengths must be finite")
    if deduplicate_radius <= 0 or residual_tolerance <= 0 or min_inliers < 3:
        raise ValueError("lattice tolerances must be positive and min_inliers at least three")
    if max_pair_vectors < 2 or not 0 < min_cross_sine < 1 or angle_prior_sigma <= 0:
        raise ValueError("invalid lattice hypothesis limits")
    if preferred_interdot_angle is not None and not np.isfinite(preferred_interdot_angle):
        raise ValueError("preferred_interdot_angle must be finite")

    representatives = _deduplicate(points, scores, deduplicate_radius)
    unique = points[representatives]
    if len(unique) < min_inliers:
        return None

    spacing = float(config.avg_spacing)
    spacing_sd = float(np.sqrt(config.var_spacing))
    min_spacing = max(2.0 * deduplicate_radius, spacing - 3.0 * spacing_sd)
    max_spacing = spacing + 3.0 * spacing_sd
    vectors: list[tuple[FloatArray, int, float]] = []
    for first in range(len(unique)):
        for second in range(first + 1, len(unique)):
            vector = unique[second] - unique[first]
            length = float(np.linalg.norm(vector))
            if min_spacing <= length <= max_spacing:
                vectors.append((vector, first, length))
    if len(vectors) < 2:
        return None

    vectors.sort(key=lambda item: abs(item[2] - spacing))
    if len(vectors) > max_pair_vectors:
        sample = np.linspace(0, len(vectors) - 1, max_pair_vectors).round().astype(int)
        vectors = [vectors[index] for index in np.unique(sample)]

    best: tuple[float, FloatArray, FloatArray, BoolArray, FloatArray] | None = None
    # Iterate vector pairs (not arbitrary labels); support is measured against
    # every candidate and integer coordinates are recomputed for each hypothesis.
    for i, (first_vector, first_anchor, first_length) in enumerate(vectors):
        for second_vector, _, second_length in vectors[i + 1 :]:
            determinant = _det2(first_vector, second_vector)
            cross_sine = abs(determinant) / (first_length * second_length)
            if cross_sine < min_cross_sine:
                continue
            basis = np.column_stack((first_vector, second_vector))
            if abs(determinant) < 1e-10:
                continue
            origin = unique[first_anchor]
            coordinates = np.linalg.solve(basis, (unique - origin).T).T
            integer_coordinates = np.rint(coordinates)
            residuals = np.linalg.norm((coordinates - integer_coordinates) @ basis.T, axis=1)
            inliers = residuals <= residual_tolerance
            if int(np.sum(inliers)) < min_inliers:
                continue
            if (
                np.ptp(integer_coordinates[inliers, 0]) < 1
                or np.ptp(integer_coordinates[inliers, 1]) < 1
            ):
                continue
            median_residual = float(np.median(residuals[inliers]))
            # Consensus count dominates, but unit charge translations should
            # remain near the configured charging period. Penalize hypotheses
            # that gain accidental support from unrealistically short or long
            # basis vectors, while allowing the generator's spacing variation.
            spacing_penalty = 0.12 * (
                ((first_length - spacing) / spacing_sd) ** 2
                + ((second_length - spacing) / spacing_sd) ** 2
            )
            angle_penalty = 0.0
            if preferred_interdot_angle is not None:
                hypotheses = _dual_charge_angles(basis)
                if not len(hypotheses):
                    continue
                angle_errors = np.abs(
                    (hypotheses - preferred_interdot_angle + np.pi / 2) % np.pi - np.pi / 2
                )
                angle_penalty = float(np.min(angle_errors) / angle_prior_sigma) ** 2
            score = (
                float(np.sum(inliers))
                - 0.1 * median_residual / residual_tolerance
                - spacing_penalty
                - angle_penalty
            )
            if best is None or score > best[0]:
                best = (score, basis, origin, inliers, residuals)

    if best is None:
        return None
    _, basis, origin, inliers, residuals = best

    # Refit origin and both primitive vectors to their inferred integer indices.
    for _ in range(3):
        coordinates = np.linalg.solve(basis, (unique - origin).T).T
        integer_coordinates = np.rint(coordinates)
        residuals = np.linalg.norm((coordinates - integer_coordinates) @ basis.T, axis=1)
        inliers = residuals <= residual_tolerance
        if int(np.sum(inliers)) < min_inliers:
            break
        indices = integer_coordinates[inliers]
        design = np.column_stack((np.ones(len(indices)), indices))
        if np.linalg.matrix_rank(design) < 3:
            break
        coefficients = np.linalg.lstsq(design, unique[inliers], rcond=None)[0]
        origin = coefficients[0]
        basis = coefficients[1:].T
        if abs(np.linalg.det(basis)) < 1e-10:
            return None

    coordinates = np.linalg.solve(basis, (unique - origin).T).T
    integer_coordinates = np.rint(coordinates)
    residuals = np.linalg.norm((coordinates - integer_coordinates) @ basis.T, axis=1)
    inliers = residuals <= residual_tolerance
    if int(np.sum(inliers)) < min_inliers:
        return None
    if np.ptp(integer_coordinates[inliers, 0]) < 1 or np.ptp(integer_coordinates[inliers, 1]) < 1:
        return None
    fitted_periods = np.linalg.norm(basis, axis=0)
    if np.any(fitted_periods < min_spacing) or np.any(fitted_periods > max_spacing):
        return None

    # Confidence combines coverage and geometric tightness. The caller decides
    # whether a weak fit should be used; the original detector remains the fallback.
    coverage = float(np.mean(inliers))
    tightness = float(np.exp(-np.median(residuals[inliers]) / residual_tolerance))
    confidence = coverage * tightness
    return ChargeLatticeFit(
        basis=np.asarray(basis, dtype=np.float64),
        origin=np.asarray(origin, dtype=np.float64),
        representative_indices=representatives,
        inliers=np.asarray(inliers, dtype=bool),
        residuals=np.asarray(residuals, dtype=np.float64),
        confidence=confidence,
    )


@dataclass(frozen=True)
class PeriodicDetectionResult:
    """Raw and lattice-refined detector outputs for one CSD frame."""

    mask: BoolArray
    raw: DetectionResult
    lattice: ChargeLatticeFit | None
    kept_peaks: BoolArray
    charge_angle_hypotheses: FloatArray | None = None
    angle_residuals: FloatArray | None = None


class PeriodicInterdotDetector:
    """Matched-filter detector with a conservative charge-lattice false-positive filter.

    A valid lattice model removes low-confidence candidates that do not align
    with repeated charge-state crossings. Strong isolated candidates are retained
    because a real dot can be missing its lattice neighbours at a scan boundary
    or through the generator's appearance probability. If the image does not
    support a two-dimensional lattice, this class falls back to the raw mask.
    """

    def __init__(
        self,
        detector: InterdotDetector | None = None,
        *,
        residual_tolerance: float = 0.015,
        min_inliers: int = 4,
        min_confidence: float = 0.10,
        preserve_score: float = 18.0,
        deduplicate_radius: float = 0.020,
    ) -> None:
        if residual_tolerance <= 0 or min_inliers < 3:
            raise ValueError("invalid lattice filtering thresholds")
        if not 0 <= min_confidence <= 1 or preserve_score < 0:
            raise ValueError("min_confidence must be in [0, 1] and preserve_score non-negative")
        self.detector = detector or InterdotDetector()
        self.residual_tolerance = float(residual_tolerance)
        self.min_inliers = int(min_inliers)
        self.min_confidence = float(min_confidence)
        self.preserve_score = float(preserve_score)
        self.deduplicate_radius = float(deduplicate_radius)

    def analyze(self, image: npt.ArrayLike) -> PeriodicDetectionResult:
        """Detect sticks, then retain lattice-consistent (or strongly isolated) peaks."""
        raw = self.detector.analyze(image)
        centers = np.column_stack(
            (raw.peak_xy[:, 1] * self.detector.step_h, raw.peak_xy[:, 0] * self.detector.step_v)
        )
        lattice = fit_charge_lattice(
            centers,
            raw.peak_scores,
            config=self.detector.config,
            deduplicate_radius=self.deduplicate_radius,
            residual_tolerance=self.residual_tolerance,
            min_inliers=self.min_inliers,
        )
        kept = np.ones(len(raw.peak_xy), dtype=bool)
        if lattice is not None and lattice.confidence >= self.min_confidence:
            kept[:] = False
            representatives = lattice.representative_indices
            kept[representatives[lattice.inliers]] = True
            # Candidate amplitudes are in matched-filter noise units; this rescue
            # avoids discarding isolated real sticks when the lattice is occluded.
            kept |= raw.peak_scores >= self.preserve_score

        mask = self.detector.mask_from_peaks(
            np.asarray(image).shape,
            raw.peak_xy,
            raw.peak_template_indices,
            kept,
        )
        return PeriodicDetectionResult(mask=mask, raw=raw, lattice=lattice, kept_peaks=kept)

    def predict(self, image: npt.ArrayLike) -> BoolArray:
        """Return the lattice-refined binary interdot mask."""
        return self.analyze(image).mask


class ChargeConstrainedInterdotDetector:
    """Full-angle detector with image-informed, charge-quantized slope filtering.

    A wide-angle first pass estimates the dominant stick direction from candidate
    template responses, so the detector does not assume 45 degrees. It then fits
    the charging lattice and uses the dual-basis electrochemical-potential
    relation to select among equivalent lattice bases. Because an unlabeled
    charge diagram cannot reveal charge polarity, both interdot-slope hypotheses
    are retained and the closest one is used only when it agrees with the
    measured stick direction. Periodic inliers tolerate a broader angular error;
    isolated candidates must agree more closely with the inferred direction.

    This is a soft model, not a universal equality: unknown lever arms,
    cross-capacitances, and basis-label ambiguity can weaken the charge prior.
    The independent fixed-angle simulator remains a separate baseline.
    """

    def __init__(
        self,
        *,
        step_h: float = GENERATOR.scene_step,
        step_v: float = GENERATOR.scene_step,
        threshold: float = 4.0,
        wide_angle_offsets: tuple[float, ...] = (
            -0.785398,
            -0.523599,
            -0.261799,
            0.0,
            0.261799,
            0.523599,
            0.785398,
            1.047198,
            1.308997,
            1.570796,
            1.832596,
            2.094395,
        ),
        angular_sigma: float = 0.30,
        angle_tolerance: float = 0.436332,
        lattice_angle_tolerance: float = 0.610865,
        charge_alignment_tolerance: float = 0.261799,
        min_orientation_concentration: float = 0.20,
        residual_tolerance: float = 0.015,
        min_inliers: int = 4,
        min_confidence: float = 0.10,
        preserve_score: float = 18.0,
    ) -> None:
        if (
            angular_sigma <= 0
            or not 0 < angle_tolerance <= lattice_angle_tolerance < np.pi / 2
            or charge_alignment_tolerance <= 0
            or not 0 <= min_orientation_concentration <= 1
            or preserve_score < 0
        ):
            raise ValueError("invalid angular prior or score settings")
        self.wide_detector = InterdotDetector(
            step_h=step_h,
            step_v=step_v,
            threshold=threshold,
            angle_offsets=wide_angle_offsets,
        )
        self.angular_sigma = float(angular_sigma)
        self.angle_tolerance = float(angle_tolerance)
        self.lattice_angle_tolerance = float(lattice_angle_tolerance)
        self.charge_alignment_tolerance = float(charge_alignment_tolerance)
        self.min_orientation_concentration = float(min_orientation_concentration)
        self.residual_tolerance = float(residual_tolerance)
        self.min_inliers = int(min_inliers)
        self.min_confidence = float(min_confidence)
        self.preserve_score = float(preserve_score)

    def analyze(self, image: npt.ArrayLike) -> PeriodicDetectionResult:
        """Infer a dominant slope, resolve the charge basis, and filter peaks."""
        frame = np.asarray(image, dtype=np.float64)
        raw = self.wide_detector.analyze(frame)
        n_peaks = len(raw.peak_xy)
        if n_peaks == 0:
            return PeriodicDetectionResult(
                mask=raw.mask,
                raw=raw,
                lattice=None,
                kept_peaks=np.zeros(0, dtype=bool),
            )

        # The axial circular mean is invariant to a stick's endpoint direction.
        # Matched-filter margin weights make coherent, high-contrast sticks more
        # influential than weak orientation-noise candidates.
        weight_floor = max(1.0, self.wide_detector.threshold - 1.0)
        weights = np.maximum(raw.peak_scores - weight_floor, 0.0) ** 2
        if float(np.sum(weights)) == 0.0:
            weights = np.ones(n_peaks, dtype=np.float64)
        resultant = np.sum(weights * np.exp(2j * raw.peak_angles))
        orientation = float(np.mod(0.5 * np.angle(resultant), np.pi))
        concentration = float(abs(resultant) / np.sum(weights))
        if concentration < self.min_orientation_concentration:
            return PeriodicDetectionResult(
                mask=raw.mask,
                raw=raw,
                lattice=None,
                kept_peaks=np.ones(n_peaks, dtype=bool),
            )

        centers = np.column_stack(
            (
                raw.peak_xy[:, 1] * self.wide_detector.step_h,
                raw.peak_xy[:, 0] * self.wide_detector.step_v,
            )
        )
        lattice = fit_charge_lattice(
            centers,
            raw.peak_scores,
            config=self.wide_detector.config,
            deduplicate_radius=0.020,
            residual_tolerance=self.residual_tolerance,
            min_inliers=self.min_inliers,
            preferred_interdot_angle=orientation,
            angle_prior_sigma=self.angular_sigma,
        )
        predicted_angles = (
            lattice.interdot_angle_hypotheses()
            if lattice is not None and lattice.confidence >= self.min_confidence
            else None
        )
        target_angle = orientation
        lattice_supported = np.zeros(n_peaks, dtype=bool)
        if predicted_angles is not None and len(predicted_angles):
            physical_errors = np.abs(
                (predicted_angles - orientation + np.pi / 2) % np.pi - np.pi / 2
            )
            closest = int(np.argmin(physical_errors))
            if physical_errors[closest] <= self.charge_alignment_tolerance:
                target_angle = float(predicted_angles[closest])
            lattice_supported[lattice.representative_indices[lattice.inliers]] = True

        angular_residuals = np.abs((raw.peak_angles - target_angle + np.pi / 2) % np.pi - np.pi / 2)
        kept = angular_residuals <= self.angle_tolerance
        kept[lattice_supported] = (
            angular_residuals[lattice_supported] <= self.lattice_angle_tolerance
        )
        # Extremely strong evidence survives model mismatch and edge truncation.
        kept |= raw.peak_scores >= 3.0 * self.preserve_score
        mask = self.wide_detector.mask_from_peaks(
            frame.shape, raw.peak_xy, raw.peak_template_indices, kept
        )
        return PeriodicDetectionResult(
            mask=mask,
            raw=raw,
            lattice=lattice,
            kept_peaks=kept,
            charge_angle_hypotheses=predicted_angles,
            angle_residuals=angular_residuals,
        )

    def predict(self, image: npt.ArrayLike) -> BoolArray:
        """Return the charge-slope-consistent binary interdot mask."""
        return self.analyze(image).mask
