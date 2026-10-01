"""Generate a reproducible charge-quantization orientation stress split.

This is an evaluation-only counterfactual: it changes only the stick-angle rule
so it follows the inferred integer-charge lattice. The organiser's default
``csd.config.GENERATOR`` and baseline results remain untouched.

Example:
    python -m solutions.generate_charge_stress --n 120 --out data/charge_test --seed 404
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from csd.config import GENERATOR
from csd.generator import (
    LineSpec,
    from_interdots_pos_to_interdots,
    generate_label,
    random_interdots_positions,
    render_csd,
    scene_window,
)


def _charge_interdot_angle(basis: np.ndarray) -> float:
    """Direction of a constant (N1 - N2) line in the gate-voltage plane."""
    gradients = np.linalg.inv(basis)
    normal = gradients[0] - gradients[1]
    tangent = np.array([-normal[1], normal[0]])
    return float(np.mod(np.arctan2(tangent[1], tangent[0]), np.pi))


def _make_line_specs(interdots, config, mean_intensity: float) -> list[LineSpec]:
    """Replicate the shared generator's neighbour-line Bernoulli draws."""
    index = {(stick.i, stick.j): k for k, stick in enumerate(interdots)}
    line_intensity = float(mean_intensity) * config.line_intensity_frac
    specs: list[LineSpec] = []
    for (i, j), k in index.items():
        lower = index.get((i + 1, j))
        if lower is not None and np.random.rand() > 1 - config.p_line:
            specs.append(LineSpec("h", k, lower, line_intensity))
    for (i, j), k in index.items():
        right = index.get((i, j + 1))
        if right is not None and np.random.rand() > 1 - config.p_line:
            specs.append(LineSpec("v", k, right, line_intensity))
    return specs


def generate_charge_stress(
    n: int,
    out_dir: str | Path,
    *,
    seed: int = 404,
    config=None,
    overwrite: bool = False,
) -> Path:
    """Generate images whose stick slope follows the charge-lattice model.

    The per-image charge basis is fitted from the generated integer-index grid,
    then the interdot angle is set to the tangent of ``N1-N2 = const``. This is
    deliberately separate from the baseline generator, whose 45-degree stick
    angle is independent of its charge-line lattice.
    """
    if n < 1:
        raise ValueError("n must be positive")
    base = config or replace(GENERATOR, angle_alpha=0.7, angle_beta=0.7)
    window = scene_window(base)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; pass overwrite=True to overwrite")

    images = np.lib.format.open_memmap(
        out / "images.npy", mode="w+", dtype=np.float32, shape=(n, window.n_v, window.n_h)
    )
    masks = np.lib.format.open_memmap(
        out / "masks.npy", mode="w+", dtype=np.uint8, shape=(n, window.n_v, window.n_h)
    )
    np.random.seed(seed)
    angle_records: list[float] = []
    with (out / "sticks.jsonl").open("w") as stick_file:
        accepted_counts: list[int] = []
        for sample in range(n):
            for _ in range(200):
                positions = random_interdots_positions(base, window)
                # The integer (i,j) construction is an affine charge lattice plus
                # small independent centre jitter; averaging its adjacent vectors
                # estimates the two one-electron translations robustly.
                basis = np.column_stack(
                    (
                        np.mean(np.diff(positions, axis=0), axis=(0, 1)),
                        np.mean(np.diff(positions, axis=1), axis=(0, 1)),
                    )
                )
                cross_sine = abs(np.linalg.det(basis)) / np.prod(np.linalg.norm(basis, axis=0))
                if cross_sine < 0.5:
                    continue
                angle = _charge_interdot_angle(basis)
                image_config = replace(base, stick_theta=angle)
                mean_intensity = float(
                    np.random.uniform(
                        image_config.intensity_range[0], image_config.intensity_range[1]
                    )
                )
                intensity = (
                    mean_intensity * (1 - image_config.intensity_jitter),
                    mean_intensity * (1 + image_config.intensity_jitter),
                )
                sticks = from_interdots_pos_to_interdots(
                    positions, window, image_config, intensity=intensity
                )
                if 6 <= len(sticks) <= 30:
                    break
            else:
                raise RuntimeError("could not generate a well-populated stress lattice")
            angle_records.append(angle)
            accepted_counts.append(len(sticks))
            line_specs = _make_line_specs(sticks, image_config, mean_intensity)
            image = render_csd(sticks, line_specs, window, config=image_config, normalize=False)
            label = generate_label(sticks, window, image_config)
            images[sample] = image.astype(np.float32)
            masks[sample] = (label > 0.5).astype(np.uint8)
            stick_file.write(
                json.dumps(
                    {
                        "image_id": sample,
                        "sticks": [
                            {
                                "x": float(s.middle[0]),
                                "y": float(s.middle[1]),
                                "theta": float(s.theta),
                                "length": float(s.length),
                                "width": float(s.width),
                                "i": int(s.i),
                                "j": int(s.j),
                            }
                            for s in sticks
                        ],
                    }
                )
                + "\n"
            )

    images.flush()
    masks.flush()
    meta = {
        "n": n,
        "seed": seed,
        "normalized": False,
        "orientation_stress": "interdot theta from dual charge lattice; no fixed 45-degree prior",
        "stick_angle_degrees": [float(np.rad2deg(v)) for v in angle_records],
        "visible_stick_counts": accepted_counts,
        "selection_note": "stress samples conditioned on at least 6 visible sticks and crossing sine >= 0.5",
        "image_shape": [window.n_v, window.n_h],
        "scan_window": asdict(window),
        "generator_config": asdict(base),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate_charge_stress(args.n, args.out, seed=args.seed, overwrite=args.overwrite)
    print(f"wrote {args.n} charge-consistent stress images to {args.out}")


if __name__ == "__main__":
    main()
