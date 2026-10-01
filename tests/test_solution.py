from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, shift

from solutions.detection import InterdotDetector, pixel_metrics
from solutions.optimization import ContrastOptimizer, _DriftTracker


def _synthetic_stick_image(seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shape = (150, 150)
    step = 0.002
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    x = (xx - 75) * step
    y = (yy - 75) * step
    theta = np.pi / 4
    along = x * np.cos(theta) + y * np.sin(theta)
    across = -x * np.sin(theta) + y * np.cos(theta)
    rectangle = (np.abs(along) <= 0.012 / 2) & (np.abs(across) <= 0.002 / 2)
    noise = rng.normal(0.0, 0.9, size=shape) + rng.normal(0.0, 0.7, size=(shape[0], 1))
    image = gaussian_filter(noise - 12.0 * rectangle, sigma=0.5)
    target = gaussian_filter(rectangle.astype(float), sigma=0.5) > 0.5
    return image, target


def test_detector_recovers_a_dark_diagonal_stick() -> None:
    image, target = _synthetic_stick_image()
    prediction = InterdotDetector().predict(image)
    metrics = pixel_metrics(prediction, target)
    assert metrics["recall"] > 0.65
    assert metrics["precision"] > 0.45


def test_pixel_metrics_are_well_defined_for_empty_masks() -> None:
    empty = np.zeros((8, 8), dtype=bool)
    assert pixel_metrics(empty, empty) == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "iou": 0.0,
    }
    target = empty.copy()
    target[2:4, 3:5] = True
    assert pixel_metrics(target, target) == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "iou": 1.0,
    }


def test_bounded_registration_uses_documented_shift_sign() -> None:
    rng = np.random.default_rng(12)
    reference = gaussian_filter(rng.normal(size=(80, 80)), sigma=1.0)
    moving = shift(reference, shift=(5, -8), order=0, mode="wrap")
    shift_yx, confidence = _DriftTracker._registration_shift(reference, moving)
    # The result is the shift to apply to the moving image to align it with the reference.
    assert np.allclose(shift_yx, (-5, 8), atol=0.2)
    assert confidence > 1.1


def test_sobol_candidates_are_reproducible_and_bounded() -> None:
    first = ContrastOptimizer(global_samples=16, seed=123)._sample_barriers()
    second = ContrastOptimizer(global_samples=16, seed=123)._sample_barriers()
    assert np.array_equal(first, second)
    assert np.all(first >= -0.5)
    assert np.all(first <= 0.5)


class _PublicApiOnlyExperiment:
    """Small deterministic device stub with no reveal or simulator attributes."""

    def __init__(self) -> None:
        self.start = {"g1": 0.0, "g2": 0.15, "g3": 0.0, "g4": 0.15, "g5": 0.0}
        self._rng = np.random.default_rng(7)
        self._shape = (150, 150)

    def measure(self, **kwargs: float) -> np.ndarray:
        image = self._rng.normal(0.0, 0.9, size=self._shape)
        image += self._rng.normal(0.0, 0.7, size=(self._shape[0], 1))
        barriers = np.array([kwargs["g1"], kwargs["g3"], kwargs["g5"]])
        center = np.array([0.12, -0.08, 0.15])
        contrast = 3.0 + 20.0 * np.exp(-np.sum(((barriers - center) / 0.08) ** 2))
        yy, xx = np.mgrid[:150, :150]
        along = (xx - 75) / np.sqrt(2) + (yy - 75) / np.sqrt(2)
        across = -(xx - 75) / np.sqrt(2) + (yy - 75) / np.sqrt(2)
        stick = (np.abs(along) <= 3.0) & (np.abs(across) <= 0.7)
        image[stick] -= contrast
        return gaussian_filter(image, sigma=0.5)


def test_optimizer_uses_only_start_and_measure() -> None:
    experiment = _PublicApiOnlyExperiment()
    optimizer = ContrastOptimizer(
        barrier_bounds=(-0.2, 0.2),
        global_samples=2,
        max_path_step=0.5,
        local_starts=1,
        local_rounds=1,
        seed=3,
    )
    result = optimizer.optimize(experiment)
    assert set(result.working_point) == {"g1", "g2", "g3", "g4", "g5"}
    assert np.isfinite(result.score)
    assert result.n_measurements > 0
    assert result.history.shape[1] == 4
