"""Direct S_i.S_j local estimators, independently for every unordered site pair.

Reuse the four original frozen-state sample blocks. No point-group or spin
isotropy reconstruction: Szz plus both transverse exchange terms are measured.
The equality O_ij=O_ji is an exact operator identity, not a state symmetry.
"""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import numpy as np
import quantax as qtx
from quantax.operator import (
    create_u,
    create_d,
    annihilate_u,
    annihilate_d,
    number_u,
    number_d,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from corrected_distance_envelope import configure


def spin_dot(i, j):
    zi = (number_u(i) - number_d(i)) * 0.5
    zj = (number_u(j) - number_d(j)) * 0.5
    sp_i = create_u(i) @ annihilate_d(i)
    sm_i = create_d(i) @ annihilate_u(i)
    sp_j = create_u(j) @ annihilate_d(j)
    sm_j = create_d(j) @ annihilate_u(j)
    return zi @ zj + 0.5 * (sp_i @ sm_j + sm_i @ sp_j)


def validate():
    qtx.sites.Chain(2, particle_type="spinful_fermion", Nparticles=(1, 1))
    pair = spin_dot(0, 1).todense()
    local = spin_dot(0, 0).todense()
    np.testing.assert_allclose(
        np.linalg.eigvalsh(pair), [-0.75, 0, 0, 0.25], atol=1e-12
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(local), [0, 0, 0.75, 0.75], atol=1e-12
    )
    total = sum(spin_dot(i, j) for i in range(2) for j in range(2)).todense()
    _, vectors = np.linalg.eigh(pair)
    np.testing.assert_allclose(vectors[:, 0] @ total @ vectors[:, 0], 0, atol=1e-12)
    np.testing.assert_allclose(
        spin_dot(0, 1).todense(), spin_dot(1, 0).todense(), atol=1e-12
    )
    print(
        "Exact singlet, triplet, onsite-moment and pair-identity checks passed",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path)
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args()
    qtx.set_default_dtype(jnp.float64)
    qtx.set_random_seed(73421)
    if args.validate_only:
        validate()
        return
    if not args.run:
        p.error("--run is required")
    devices = jax.devices()
    if len(devices) != 2 or any(d.platform != "gpu" for d in devices):
        raise RuntimeError("This analysis requests exactly two GPU devices")
    cfg = json.loads((args.run / "config.json").read_text())
    summary = json.loads((args.run / "summary.json").read_text())
    if summary["status"] != "complete" or summary["iterations"] != cfg["iterations"]:
        raise ValueError("Training and frozen-state evaluation must be complete")
    if cfg["complex_features"] or cfg["pairing_dtype"] != "float64":
        raise ValueError("This measurement reconstructs the real CNN-Pfaffian")
    model_class = qtx.model.MultiplicativePfBackflow
    assert (
        hashlib.sha256(Path(inspect.getfile(model_class)).read_bytes()).hexdigest()
        == cfg["model_sha256"]
    )
    lattice = qtx.sites.TriangularB(
        cfg["L"], particle_type="spinful_fermion", Nparticles=tuple(cfg["electrons"])
    )
    model = model_class.resconv(
        d=cfg["pairing_features"],
        channels=cfg["channels"],
        nblocks=cfg["nblocks"],
        kernel_size=cfg["kernel_size"],
        dtype=jnp.float32,
        pairing_dtype=jnp.float64,
    )
    model = configure(model, cfg)
    state = qtx.state.Variational(
        model, param_file=args.run / "latest.eqx", max_parallel=(256, 32, 256)
    )
    N = lattice.Nsites
    out = args.run / "su2_correlations"
    out.mkdir(exist_ok=True)
    paths = [
        args.run / "sampler_diagnostics" / f"evaluation_{b+1:03d}.npz"
        for b in range(cfg["eval_sweeps"])
    ]
    signature = dict(
        model_sha256=hashlib.sha256((args.run / "latest.eqx").read_bytes()).hexdigest(),
        sample_sha256=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
        estimator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sites=N,
        blocks=len(paths),
        samples_per_block=cfg["samples"],
        observable="S_i.S_j=Sz_i Sz_j+(S+_i S-_j+S-_i S+_j)/2",
        point_group_averaging=False,
        spin_symmetry_reconstruction=False,
    )
    provenance = out / "provenance.json"
    if provenance.exists() and json.loads(provenance.read_text()) != signature:
        raise ValueError("Measurement provenance changed; refusing to mix estimators")
    provenance.write_text(json.dumps(signature, indent=2) + "\n")
    started = time.monotonic()
    means = []
    for b, path in enumerate(paths):
        with np.load(path) as data:
            spins = np.asarray(data["spins"])
        assert spins.shape == (cfg["samples"], 2 * N)
        occupied = (spins > 0).astype(float)
        up, down = np.split(occupied, 2, axis=1)
        np.testing.assert_array_equal(up.sum(axis=1), cfg["electrons"][0])
        np.testing.assert_array_equal(down.sum(axis=1), cfg["electrons"][1])
        samples = qtx.sampler.Samples(
            qtx.utils.to_distributed_array(spins), psi=state(spins)
        )
        local_path = out / f"block_{b+1:03d}_locals.npy"
        shape = (len(spins), N, N)
        if local_path.exists():
            local = np.load(local_path, mmap_mode="r+")
            assert local.shape == shape
        else:
            local = np.lib.format.open_memmap(
                local_path, mode="w+", dtype=np.float64, shape=shape
            )
            local[:] = np.nan
            local.flush()
        for i in range(N):
            local[:, i, i] = 0.75 * (up[:, i] + down[:, i] - 2 * up[:, i] * down[:, i])
            for j in range(i + 1, N):
                if (
                    np.isfinite(local[:, i, j]).all()
                    and np.isfinite(local[:, j, i]).all()
                ):
                    continue
                value = np.asarray(spin_dot(i, j).Oloc(state, samples))
                if not np.isfinite(value).all():
                    raise FloatingPointError(
                        f"Nonfinite spin estimator in block {b+1}, pair {i},{j}"
                    )
                local[:, i, j] = value
                local[:, j, i] = value
            # Persist expensive estimators after each row, before advancing progress.
            local.flush()
            progress = dict(
                block=b + 1, completed_row=i, elapsed_seconds=time.monotonic() - started
            )
            temp = out / "progress.json.tmp"
            temp.write_text(json.dumps(progress) + "\n")
            temp.replace(out / "progress.json")
            print(json.dumps(progress), flush=True)
        block = np.mean(local, axis=0)
        np.save(out / f"block_{b+1:03d}_mean.npy", block)
        means.append(block)
    means = np.asarray(means)
    np.savez_compressed(
        out / "correlations.npz",
        block_means=means,
        mean=means.mean(axis=0),
        sem=means.std(axis=0, ddof=1) / np.sqrt(len(means)),
        coordinates=lattice.coord,
        xyz=lattice.xyz_from_index[:, 1:],
        extent=lattice.shape[1:],
        basis_vectors=lattice.basis_vectors,
        reciprocal_vectors=lattice.reciprocal_vectors,
    )
    (out / "summary.json").write_text(
        json.dumps(
            dict(
                status="complete",
                total_samples=len(paths) * cfg["samples"],
                elapsed_seconds=time.monotonic() - started,
            ),
            indent=2,
        )
        + "\n"
    )
    print("Completed direct SU(2) spin correlations", flush=True)


if __name__ == "__main__":
    main()
