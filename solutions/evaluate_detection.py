"""Evaluate the held-out pixel-mask detector on a generated dataset.

Example:
    python -m solutions.evaluate_detection --dataset data/test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.ndimage import label

from csd import load_dataset
from solutions.detection import InterdotDetector


def _object_counts(prediction: npt.ArrayLike, target: npt.ArrayLike) -> tuple[int, int, int, int]:
    """Return detected/total target and prediction connected-component counts."""
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    structure = np.ones((3, 3), dtype=np.uint8)
    pred_labels, n_pred = label(pred, structure=structure)
    truth_labels, n_truth = label(truth, structure=structure)

    detected_truth = 0
    for component in range(1, n_truth + 1):
        detected_truth += int(np.any(pred[truth_labels == component]))
    matched_prediction = 0
    for component in range(1, n_pred + 1):
        matched_prediction += int(np.any(truth[pred_labels == component]))
    return detected_truth, n_truth, matched_prediction, n_pred


def evaluate(dataset: str | Path, *, threshold: float = 4.0) -> dict[str, float]:
    """Evaluate foreground pixel and connected-component metrics."""
    ds = load_dataset(dataset)
    images, masks = ds["images"], ds["masks"]
    if len(images) != len(masks):
        raise ValueError("images.npy and masks.npy have different sample counts")

    detector = InterdotDetector(threshold=threshold)
    tp = fp = fn = 0
    detected = target_objects = matched = predicted_objects = 0
    per_image: list[tuple[float, float, float, float]] = []

    for image, target in zip(images, masks):
        prediction = detector.predict(image)
        truth = np.asarray(target, dtype=bool)
        true_positive = int(np.count_nonzero(prediction & truth))
        false_positive = int(np.count_nonzero(prediction & ~truth))
        false_negative = int(np.count_nonzero(~prediction & truth))
        tp += true_positive
        fp += false_positive
        fn += false_negative

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        iou = (
            true_positive / (true_positive + false_positive + false_negative)
            if true_positive + false_positive + false_negative
            else 0.0
        )
        per_image.append((precision, recall, f1, iou))
        hit, n_truth, match, n_pred = _object_counts(prediction, truth)
        detected += hit
        target_objects += n_truth
        matched += match
        predicted_objects += n_pred

    macro = np.mean(np.asarray(per_image), axis=0)
    micro_precision = tp / (tp + fp) if tp + fp else 0.0
    micro_recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    micro_iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "n_images": float(len(images)),
        "threshold": float(threshold),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "macro_iou": float(macro[3]),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "micro_iou": micro_iou,
        "object_precision": matched / predicted_objects if predicted_objects else 0.0,
        "object_recall": detected / target_objects if target_objects else 0.0,
        "n_target_objects": float(target_objects),
        "n_predicted_objects": float(predicted_objects),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, required=True, help="folder with images.npy and masks.npy"
    )
    parser.add_argument(
        "--threshold", type=float, default=4.0, help="matched-filter detection threshold"
    )
    args = parser.parse_args()

    metrics = evaluate(args.dataset, threshold=args.threshold)
    print(f"dataset: {args.dataset}  samples: {int(metrics['n_images'])}")
    for key, value in metrics.items():
        if key in {"n_images", "threshold"}:
            continue
        print(f"{key:>20}: {value:.4f}")


if __name__ == "__main__":
    main()
