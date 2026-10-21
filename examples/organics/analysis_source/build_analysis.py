"""Combine raw spin tensors and original charge blocks into notebook data."""

import argparse
from pathlib import Path
import numpy as np


def sem(blocks):
    return np.std(blocks, axis=0, ddof=1) / np.sqrt(len(blocks))


def build(run):
    run = Path(run)
    with np.load(run / "su2_correlations/correlations.npz") as f:
        spin = f["block_means"]
        xyz, extent, coords = f["xyz"], f["extent"], f["coordinates"]
        basis, reciprocal = f["basis_vectors"], f["reciprocal_vectors"]
    N = len(xyz)
    q_raw = (xyz / extent) @ reciprocal
    images = np.array([(a, b) for a in range(-3, 4) for b in range(-3, 4)]) @ reciprocal
    candidates = q_raw[:, None] - images[None]
    q = candidates[np.arange(N), np.argmin(np.linalg.norm(candidates, axis=-1), axis=1)]
    phases = np.exp(-1j * q_raw @ coords.T)
    np.testing.assert_allclose(phases @ phases.conj().T, N * np.eye(N), atol=1e-10)
    zz, charge, energy, double = [], [], [], []
    for b in range(len(spin)):
        with np.load(run / "observables" / f"block_{b+1:03d}.npz") as f:
            zz.append(f["spin_zz_correlation_raw"])
            charge.append(f["charge_structure_factor"].ravel())
            energy.append(f["energy_per_site"])
            double.append(f["double_occupancy"])
    zz, charge = np.asarray(zz), np.asarray(charge)
    sq = np.einsum("qi,bij,qj->bq", phases.conj(), spin, phases).real / N
    szzq = np.einsum("qi,bij,qj->bq", phases.conj(), zz, phases).real / N
    np.testing.assert_allclose(
        sq.sum(axis=1), np.trace(spin, axis1=1, axis2=2), atol=1e-9
    )
    np.testing.assert_allclose(
        charge.sum(axis=1), 2 * N * np.asarray(double), atol=1e-9
    )
    np.testing.assert_allclose(charge[:, 0], 0, atol=1e-12)
    np.testing.assert_allclose(szzq[:, 0], 0, atol=1e-12)
    shifts = np.array(
        [np.ravel_multi_index(((xyz + r) % extent).T, tuple(extent)) for r in xyz]
    )
    # Translation averages are computed after retaining the full independent tensor.
    cr = np.stack([spin[:, np.arange(N), row].mean(axis=1) for row in shifts], axis=1)
    winding = np.array([(a, b) for a in range(-2, 3) for b in range(-2, 3)])
    cart_images = coords[:, None] - coords[0] + (winding * extent) @ basis
    r_cart = cart_images[
        np.arange(N), np.argmin(np.linalg.norm(cart_images, axis=-1), axis=1)
    ]
    result = dict(
        q_raw=q_raw,
        q=q,
        k=np.linalg.norm(q, axis=1),
        coordinates=coords,
        r_cart=r_cart,
        spin_correlation_blocks=spin,
        spin_correlation=spin.mean(axis=0),
        spin_correlation_sem=sem(spin),
        Cdot_r=cr.mean(axis=0),
        Cdot_r_sem=sem(cr),
        S_q=sq.mean(axis=0),
        S_q_sem=sem(sq),
        S_q_blocks=sq,
        Szz_q=szzq.mean(axis=0),
        Szz_q_sem=sem(szzq),
        N_q=charge.mean(axis=0),
        N_q_sem=sem(charge),
        N_q_blocks=charge,
        energy_per_site_blocks=energy,
        double_occupancy_blocks=double,
    )
    out = run / "analysis.npz"
    np.savez_compressed(out, **result)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    print(build(p.parse_args().run))
