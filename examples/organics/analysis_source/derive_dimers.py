"""Derive full spin-dimer Gram tensors from saved direct S_i.S_j local estimators.

No additional samples or wavefunction evaluations are needed. All four original
temporal blocks and all spatial bond origins are retained independently.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from dimer_kernels import DIRECTIONS, minimum_distances, reduce_locals
from dimer_analysis import analyze


def derive(run):
    started = time.monotonic()
    cfg = json.loads((run / "config.json").read_text())
    spin_dir = run / "su2_correlations"
    signature = json.loads((spin_dir / "provenance.json").read_text())
    checkpoint_hash = hashlib.sha256((run / "latest.eqx").read_bytes()).hexdigest()
    if signature["model_sha256"] != checkpoint_hash:
        raise ValueError("Saved spin local estimators do not match the checkpoint")
    with np.load(spin_dir / "correlations.npz") as f:
        geom = {k: f[k] for k in ["coordinates", "xyz", "extent", "basis_vectors"]}
    with np.load(run / "analysis.npz") as f:
        q, q_raw = f["q"], f["q_raw"]
    xyz, extent, basis = geom["xyz"], geom["extent"], geom["basis_vectors"]
    N = cfg["sites"]
    shifts_xyz = np.rint(DIRECTIONS @ np.linalg.inv(basis)).astype(int)
    np.testing.assert_allclose(shifts_xyz @ basis, DIRECTIONS, atol=1e-12)
    neighbors = np.array(
        [
            np.ravel_multi_index(((xyz + delta) % extent).T, tuple(extent))
            for delta in shifts_xyz
        ]
    ).T
    inverse = np.ravel_multi_index(((-xyz) % extent).T, tuple(extent))
    phase = np.exp(-1j * q @ geom["coordinates"].T)
    np.testing.assert_allclose(phase @ phase.conj().T, N * np.eye(N), atol=1e-10)
    out = run.parents[1] / "dimer_correlations"
    out.mkdir(exist_ok=True)
    nblocks = cfg["eval_sweeps"]
    chains = cfg["samples"]
    local_path = out / "dimer_locals.npy"
    if local_path.exists():
        local = np.load(local_path, mmap_mode="r+")
        assert local.shape == (nblocks, chains, N, 3)
    else:
        local = np.lib.format.open_memmap(
            local_path, mode="w+", dtype=np.float64, shape=(nblocks, chains, N, 3)
        )
        local[:] = np.nan
        local.flush()
    means = []
    structures = []
    site_means = []
    full_tensors = []
    residuals = []
    for b in range(nblocks):
        source = spin_dir / f"block_{b+1:03d}_locals.npy"
        pair = np.load(source, mmap_mode="r")
        assert pair.shape == (chains, N, N)
        value = np.asarray(pair[:, np.arange(N)[:, None], neighbors])
        if not np.isfinite(value).all():
            raise ValueError("Saved direct spin estimators are incomplete")
        local[b] = value
        local.flush()
        mean, structure = reduce_locals(value, phase)
        flat = value.reshape(chains, N * 3)
        raw_tensor = (flat.T @ flat / chains).reshape(N, 3, N, 3)
        sample_file = run / "sampler_diagnostics" / f"evaluation_{b+1:03d}.npz"
        if (
            hashlib.sha256(sample_file.read_bytes()).hexdigest()
            != signature["sample_sha256"][b]
        ):
            raise ValueError("Original frozen sample block changed")
        with np.load(sample_file) as f:
            occupied = (f["spins"] > 0).reshape(chains, 2, N)
        single = occupied[:, 0] ^ occupied[:, 1]
        pp = np.mean(single[:, :, None] * single[:, neighbors], axis=(0, 1))
        second = np.mean(value**2, axis=(0, 1))
        residual = second - (3 / 16 * pp - 0.5 * mean.real)
        np.savez_compressed(
            out / f"raw_block_{b+1:03d}.npz",
            bond_mean_by_site=value.mean(axis=0),
            full_raw_bond_tensor=raw_tensor,
            mean_bond=mean,
            raw_structure=structure,
            second_moment_identity_residual=residual,
        )
        means.append(mean)
        structures.append(structure)
        site_means.append(value.mean(axis=0))
        full_tensors.append(raw_tensor)
        residuals.append(residual)
        print(
            f"Persisted original block {b+1}/{nblocks}, elapsed {time.monotonic()-started:.1f}s",
            flush=True,
        )
    np.savez_compressed(
        out / "blocks.npz",
        means=means,
        raw_structure=structures,
        full_raw_bond_tensors=full_tensors,
        bond_means_by_site=site_means,
        phase=phase,
        q=q,
        q_raw=q_raw,
        inverse=inverse,
        distance=minimum_distances(geom["coordinates"], extent, basis),
        coordinates=geom["coordinates"],
        directions=DIRECTIONS,
        selected_blocks=np.arange(nblocks),
        selected_chains=np.arange(chains),
        second_moment_identity_residual=residuals,
        neighbors=neighbors,
    )
    result = analyze(out, rmin=cfg["L"] / 2)
    # Independent Fourier contraction directly from the preserved full bond tensor.
    tensors = np.asarray(full_tensors)
    via_tensor = (
        np.einsum("qi,biajc,qj->bqac", phase.conj(), tensors, phase, optimize=True) / N
    )
    np.testing.assert_allclose(via_tensor, np.asarray(structures), atol=1e-10)
    metadata = dict(
        status="complete",
        samples=nblocks * chains,
        blocks=nblocks,
        samples_per_block=chains,
        U=cfg["U"],
        sites=N,
        checkpoint_sha256=checkpoint_hash,
        reused_from=str(spin_dir),
        operator="B_i,a=S_i.S_(i+delta_a)",
        estimator="Hermitian Gram product of independently measured direct spin local estimators",
        raw_tensor_shape=list(tensors.shape),
        point_group_reconstruction=False,
        disconnected_subtraction="orientation means; recomputed in delete-one-original-block jackknife",
        elapsed_seconds=time.monotonic() - started,
    )
    (out / "measurement.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Completed dimers", out, flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    derive(p.parse_args().run.resolve())
