"""Diagonal observables from saved frozen-state temporal evaluation blocks.

Szz is measured directly; no SU(2) or point-group reconstruction is applied.
Errors use the original sweep blocks, never subdivisions of a sweep.
"""

import argparse
import json
from pathlib import Path
import numpy as np


def block_observables(spins, L):
    N = 3 * L**2
    if spins.ndim != 2 or spins.shape[1] != 2 * N:
        raise ValueError("Unexpected configuration shape")
    if not np.all(np.isin(spins, [-1, 1])):
        raise ValueError("Expected +/-1 occupation encoding")
    occupied = (spins.astype(np.float64) + 1) / 2
    up, down = np.split(occupied, 2, axis=1)
    np.testing.assert_array_equal(up.sum(axis=1), N // 2)
    np.testing.assert_array_equal(down.sum(axis=1), N // 2)
    density = up + down
    charge = density - 1
    sz = (up - down) / 2
    charge_k = np.fft.fftn(charge.reshape(-1, 3 * L, L), axes=(1, 2))
    sz_k = np.fft.fftn(sz.reshape(-1, 3 * L, L), axes=(1, 2))
    count = len(spins)
    return dict(
        double_occupancy=np.mean(up * down),
        density_mean=density.mean(axis=0),
        sz_mean=sz.mean(axis=0),
        density_correlation_raw=density.T @ density / count,
        charge_correlation_about_half_filling=charge.T @ charge / count,
        spin_zz_correlation_raw=sz.T @ sz / count,
        charge_structure_factor=np.mean(abs(charge_k) ** 2, axis=0) / N,
        spin_zz_structure_factor=np.mean(abs(sz_k) ** 2, axis=0) / N,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    cfg = json.loads((args.run / "config.json").read_text())
    out = args.run / "observables"
    out.mkdir(exist_ok=True)
    blocks = []
    energies = []
    for index in range(1, cfg["eval_sweeps"] + 1):
        path = args.run / "sampler_diagnostics" / f"evaluation_{index:03d}.npz"
        with np.load(path) as saved:
            if saved["spins"].shape[0] != cfg["samples"]:
                raise ValueError("Evaluation sample count mismatch")
            block = block_observables(saved["spins"], cfg["L"])
            energy = float(saved["energy_locals"].real.mean() / cfg["sites"])
        # Persist each complete estimator block before processing the next one.
        np.savez_compressed(
            out / f"block_{index:03d}.npz", **block, energy_per_site=energy
        )
        blocks.append(block)
        energies.append(energy)

    def estimate(values):
        values = np.asarray(values)
        return dict(
            mean=float(values.mean()),
            sweep_block_sem=float(values.std(ddof=1) / np.sqrt(len(values))),
        )

    summary = dict(
        blocks=len(blocks),
        samples_per_block=cfg["samples"],
        total_samples=len(blocks) * cfg["samples"],
        energy_per_site=estimate(energies),
        double_occupancy=estimate([b["double_occupancy"] for b in blocks]),
        error_policy="original temporal sweep blocks; no chain subdivision",
        spin_observable="SzSz only; no SU(2) reconstruction",
        symmetry_averaging=False,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
