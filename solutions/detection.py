"""Physics-guided interdot detection for raw charge-stability diagrams.

The detector is deliberately training-free: it searches for short, dark,
approximately 45-degree rectangular ridges with a normalized matched-filter
bank. A per-row median subtraction removes the shared horizontal acquisition
noise without requiring image normalization or access to simulator metadata.

The defaults are expressed in volts and match the shared generator. Pass the
measurement pixel steps when using a non-default scan grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt
from scipy.ndimage import correlate, gaussian_filter
from skimage.feature import peak_local_max

from csd.config import GENERATOR, GeneratorConfig

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class DetectionResult:
    """Outputs of one detector pass.

    Attributes
    ----------
    mask:
        Predicted interdot pixels, with the same shape as the input image.
    score_map:
        Best normalized dark-template response at each pixel. Larger values
        indicate stronger evidence for an interdot centre.
    peak_xy:
        Candidate centres as ``(row, column)`` integer coordinates.
    peak_scores:
        Matched-filter scores at ``peak_xy``.
    """

    mask: BoolArray
    score_map: FloatArray
    peak_xy: npt.NDArray[np.int64]
    peak_scores: FloatArray
    peak_contrasts: FloatArray


@dataclass(frozen=True)
class _Template:
    kernel: FloatArray
    footprint: BoolArray
    amplitude_gain: float


@lru_cache(maxsize=24)
def _template_bank(
    step_h: float,
    step_v: float,
    config: GeneratorConfig,
    n_lengths: int = 5,
    n_widths: int = 4,
    angle_offsets: tuple[float, ...] = (-0.20, -0.10, 0.0, 0.10, 0.20),
) -> tuple[_Template, ...]:
    """Construct a normalized bank from the generator's physical size ranges."""
    # GeneratorConfig is frozen/hashable, so it is safe in the cache key.
    length_min = config.avg_spacing * config.length_frac[0]
    length_max = config.avg_spacing * config.length_frac[1]
    width_min = config.avg_spacing * config.length_frac[0] * config.width_frac[0]
    width_max = config.avg_spacing * config.length_frac[1] * config.width_frac[1]
    lengths = np.linspace(length_min, length_max, n_lengths)
    widths = np.linspace(width_min, width_max, n_widths)
    angles = config.stick_theta + np.asarray(angle_offsets)

    templates: list[_Template] = []
    for length in lengths:
        for width in widths:
            for theta in angles:
                # A one-pixel grid with a 2-pixel background annulus on all sides.
                half_x = (
                    int(
                        np.ceil(
                            (length * abs(np.cos(theta)) + width * abs(np.sin(theta)))
                            / (2 * step_h)
                        )
                    )
                    + 3
                )
                half_y = (
                    int(
                        np.ceil(
                            (length * abs(np.sin(theta)) + width * abs(np.cos(theta)))
                            / (2 * step_v)
                        )
                    )
                    + 3
                )
                yy, xx = np.mgrid[-half_y : half_y + 1, -half_x : half_x + 1]
                x = xx * step_h
                y = yy * step_v
                along = x * np.cos(theta) + y * np.sin(theta)
                across = -x * np.sin(theta) + y * np.cos(theta)
                stick = (np.abs(along) <= length / 2) & (np.abs(across) <= width / 2)
                if not stick.any():
                    continue

                # Match the label generation's blur in the signal template. The
                # annulus estimates local background and rejects broad features.
                blurred = gaussian_filter(stick.astype(float), config.sigma_blur)
                ring_parallel = max(2 * step_h, 2 * step_v)
                ring_perp = max(2 * step_h, 2 * step_v)
                outer = (np.abs(along) <= length / 2 + ring_parallel) & (
                    np.abs(across) <= width / 2 + ring_perp
                )
                ring = outer & ~stick
                if not ring.any():
                    continue
                kernel = blurred / blurred.sum() - ring.astype(float) / ring.sum()
                norm = float(np.sqrt(np.sum(kernel * kernel)))
                if norm == 0.0:
                    continue
                kernel = kernel / norm
                footprint = blurred > 0.5
                templates.append(
                    _Template(
                        kernel=np.asarray(kernel, dtype=np.float64),
                        footprint=np.asarray(footprint, dtype=bool),
                        amplitude_gain=float(np.sum(blurred * kernel)),
                    )
                )
    return tuple(templates)


class InterdotDetector:
    """Detect interdot pixels in one raw CSD image.

    Parameters
    ----------
    step_h, step_v:
        Physical voltage step per image pixel. The defaults (2 mV) match the
        challenge dataset and default simulator scan.
    threshold:
        Minimum peak response in noise-standard-deviation units. The default
        was selected on a held-out validation split; it trades some recall for
        fewer noise/charging-line false positives.
    config:
        Shared appearance configuration. Kept public for transfer tests; the
        hackathon baseline should use the default.
    min_distance:
        Non-maximum suppression radius in pixels.
    """

    def __init__(
        self,
        *,
        step_h: float = GENERATOR.scene_step,
        step_v: float = GENERATOR.scene_step,
        threshold: float = 4.0,
        config: GeneratorConfig = GENERATOR,
        min_distance: int = 4,
        n_lengths: int = 5,
        n_widths: int = 4,
        angle_offsets: tuple[float, ...] = (-0.20, -0.10, 0.0, 0.10, 0.20),
    ) -> None:
        if step_h <= 0 or step_v <= 0:
            raise ValueError("pixel steps must be positive")
        if min_distance < 1:
            raise ValueError("min_distance must be at least one pixel")
        if n_lengths < 1 or n_widths < 1 or not angle_offsets:
            raise ValueError("the template bank must contain at least one length, width, and angle")
        self.step_h = float(step_h)
        self.step_v = float(step_v)
        self.threshold = float(threshold)
        self.config = config
        self.min_distance = int(min_distance)
        self._bank = _template_bank(
            self.step_h,
            self.step_v,
            config,
            int(n_lengths),
            int(n_widths),
            tuple(float(value) for value in angle_offsets),
        )
        if not self._bank:
            raise ValueError("the scan pixel size is too coarse for the interdot templates")

    def analyze(self, image: npt.ArrayLike) -> DetectionResult:
        """Return a binary mask and centre-evidence map for ``image``."""
        frame = np.asarray(image, dtype=np.float64)
        if frame.ndim != 2 or min(frame.shape) < 3:
            raise ValueError("image must be a 2-D array at least 3 pixels wide and high")
        if not np.isfinite(frame).all():
            raise ValueError("image must contain only finite values")

        # The simulator adds one noise value per row as well as pixel noise.
        # Removing the robust row baseline suppresses that nuisance term while
        # preserving sparse, sub-row interdot features.
        residual = frame - np.median(frame, axis=1, keepdims=True)

        best_score = np.full(frame.shape, -np.inf, dtype=np.float64)
        best_template = np.zeros(frame.shape, dtype=np.int16)
        for index, template in enumerate(self._bank):
            response = -correlate(residual, template.kernel, mode="reflect")
            update = response > best_score
            best_score[update] = response[update]
            best_template[update] = index

        peaks = peak_local_max(
            best_score,
            min_distance=self.min_distance,
            threshold_abs=self.threshold,
            exclude_border=False,
        )
        if len(peaks):
            peak_scores = best_score[peaks[:, 0], peaks[:, 1]]
            # Deterministic ordering makes candidate lists reproducible.
            order = np.argsort(-peak_scores, kind="stable")
            peaks = peaks[order]
            peak_scores = peak_scores[order]
        else:
            peak_scores = np.empty(0, dtype=np.float64)

        peak_contrasts = np.empty(len(peaks), dtype=np.float64)
        for index, ((row, col), score) in enumerate(zip(peaks, peak_scores)):
            template = self._bank[int(best_template[row, col])]
            peak_contrasts[index] = float(score / max(template.amplitude_gain, 1e-12))

        predicted = np.zeros(frame.shape, dtype=bool)
        for (row, col), score in zip(peaks, peak_scores):
            if score < self.threshold:
                continue
            template = self._bank[int(best_template[row, col])]
            footprint = template.footprint
            half_y, half_x = footprint.shape[0] // 2, footprint.shape[1] // 2
            y0, y1 = max(0, row - half_y), min(frame.shape[0], row + half_y + 1)
            x0, x1 = max(0, col - half_x), min(frame.shape[1], col + half_x + 1)
            fy0, fx0 = y0 - (row - half_y), x0 - (col - half_x)
            predicted[y0:y1, x0:x1] |= footprint[fy0 : fy0 + y1 - y0, fx0 : fx0 + x1 - x0]

        return DetectionResult(
            mask=predicted,
            score_map=best_score,
            peak_xy=np.asarray(peaks, dtype=np.int64),
            peak_scores=np.asarray(peak_scores, dtype=np.float64),
            peak_contrasts=peak_contrasts,
        )

    def predict(self, image: npt.ArrayLike) -> BoolArray:
        """Return only the binary interdot-pixel mask."""
        return self.analyze(image).mask

    def contrast_score(self, image: npt.ArrayLike, *, top_k: int = 3) -> float:
        """Return a robust image-level interdot contrast proxy.

        Matched-filter response is divided by its template gain to undo the
        square-root-area advantage of long/wide sticks. The result estimates the
        dip amplitude (the challenge's contrast definition), rather than how
        many pixels a particular stick happens to cover.

        Candidate amplitudes are ordered independently of detection confidence;
        the default ``top_k=3`` is robust for a generic overview, while ``top_k=1``
        matches the challenge's maximum-interdot objective.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least one")
        result = self.analyze(image)
        if not len(result.peak_contrasts):
            return 0.0
        count = min(top_k, len(result.peak_contrasts))
        strongest = np.sort(result.peak_contrasts)[::-1][:count]
        return float(np.mean(strongest))


def pixel_metrics(prediction: npt.ArrayLike, target: npt.ArrayLike) -> dict[str, float]:
    """Compute foreground precision, recall, F1, and IoU for binary masks."""
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError(f"prediction shape {pred.shape} != target shape {truth.shape}")
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}
