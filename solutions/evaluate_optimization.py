"""Post-run self-check for Challenge 2 across reproducible fresh devices.

The optimizer itself never calls ``reveal``. This separate benchmark calls it
only after optimization has returned, then compares the image-derived score at
the submitted working point with a repeated measurement at the revealed oracle
point. Those oracle scans are reported separately from the optimizer's budget.

Example:
    python -m solutions.evaluate_optimization --count 10 --start-seed 0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from csd import BARRIERS, new_experiment
from solutions.optimization import ContrastOptimizer


def _repeat_score(
    optimizer: ContrastOptimizer, experiment, point: dict[str, float], repeats: int
) -> float:
    scores = []
    for _ in range(repeats):
        image = experiment.measure(
            **point,
            span_h=optimizer.scan_span,
            span_v=optimizer.scan_span,
            step_h=optimizer.scan_step,
            step_v=optimizer.scan_step,
        )
        scores.append(optimizer.detector.contrast_score(image, top_k=1))
    return float(np.mean(scores))


def evaluate_seeds(
    *,
    count: int = 10,
    start_seed: int = 0,
    optimizer_seed: int = 0,
    global_samples: int = 48,
    oracle_repeats: int = 3,
) -> list[dict[str, float]]:
    """Run deterministic devices and perform a post-hoc reveal-based check."""
    if count < 1 or oracle_repeats < 1:
        raise ValueError("count and oracle_repeats must be positive")

    results: list[dict[str, float]] = []
    for device_seed in range(start_seed, start_seed + count):
        experiment = new_experiment(seed=device_seed)
        optimizer = ContrastOptimizer(seed=optimizer_seed, global_samples=global_samples)
        result = optimizer.optimize(experiment)
        optimizer_measurements = result.n_measurements
        optimizer_pixels = result.n_pixels

        # Explicitly post-run only: this is a self-check baseline, never part of
        # ContrastOptimizer.optimize or its candidate-selection logic.
        truth = experiment.reveal()
        oracle_point = dict(experiment.start)
        oracle_point.update(truth["optimum_barriers"])
        oracle_point.update(truth["optimum_plungers"])

        measured_found = _repeat_score(optimizer, experiment, result.working_point, oracle_repeats)
        measured_oracle = _repeat_score(optimizer, experiment, oracle_point, oracle_repeats)
        ratio = measured_found / measured_oracle if measured_oracle > 0 else 0.0
        distance = float(
            np.linalg.norm(
                [result.working_point[gate] - truth["optimum_barriers"][gate] for gate in BARRIERS]
            )
        )
        results.append(
            {
                "device_seed": float(device_seed),
                "optimizer_seed": float(optimizer_seed),
                "initial_score": result.initial_score,
                "found_score": measured_found,
                "oracle_score": measured_oracle,
                "measured_score_ratio": ratio,
                "barrier_distance_v": distance,
                "n_measurements": float(optimizer_measurements),
                "n_pixels": float(optimizer_pixels),
                "oracle_factor": float(truth["max_contrast_factor"]),
            }
        )
        print(
            f"device {device_seed:>3}: score ratio={ratio:.3f}, "
            f"barrier distance={distance:.3f} V, "
            f"optimizer measurements={optimizer_measurements}, "
            f"pixels={optimizer_pixels:,}"
        )

    ratios = np.asarray([row["measured_score_ratio"] for row in results])
    pixels = np.asarray([row["n_pixels"] for row in results])
    print(
        "summary: "
        f"mean score ratio={ratios.mean():.3f}, "
        f"median={np.median(ratios):.3f}, min={ratios.min():.3f}; "
        f"median optimizer pixels={np.median(pixels):,.0f}"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10, help="number of independent devices")
    parser.add_argument("--start-seed", type=int, default=0, help="first device seed")
    parser.add_argument("--optimizer-seed", type=int, default=0, help="Sobol-design seed")
    parser.add_argument(
        "--global-samples", type=int, default=48, help="Sobol sample count per device"
    )
    parser.add_argument("--oracle-repeats", type=int, default=3, help="post-run score repetitions")
    parser.add_argument("--csv", type=Path, default=None, help="optional CSV output path")
    args = parser.parse_args()

    rows = evaluate_seeds(
        count=args.count,
        start_seed=args.start_seed,
        optimizer_seed=args.optimizer_seed,
        global_samples=args.global_samples,
        oracle_repeats=args.oracle_repeats,
    )
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
