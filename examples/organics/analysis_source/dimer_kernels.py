"""Dimer correlations from Gram products of Hermitian local estimators.

For B_i=B_i^dagger, <B_i B_j>=E_[|psi|^2][b_i(s)^* b_j(s)],
where b_i(s)=(B_i psi)(s)/psi(s). This avoids applying each eight-fermion
product separately and gives the full translation average by FFT contraction.
"""

from pathlib import Path
import numpy as np

DIRECTIONS = np.array([[1.0, 0.0], [-0.5, np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 2]])


def minimum_distances(coordinates, extent, basis):
    import itertools

    winding = np.array(list(itertools.product(range(-2, 3), repeat=len(extent))))
    images = (
        coordinates[:, None, :]
        - coordinates[0]
        + (winding * np.asarray(extent)) @ basis
    )
    return np.linalg.norm(images, axis=-1).min(axis=1)


def reduce_locals(local, phase, weights=None):
    """local[s,i,a], phase[q,i]; structure factor normalized by N sites."""
    if weights is None:
        weights = np.full(len(local), 1 / len(local))
    else:
        weights = np.asarray(weights) / np.sum(weights)
    mean = np.einsum("s,sia->a", weights, local) / local.shape[1]
    fourier = np.einsum("qi,sia->sqa", phase, local, optimize=True) / np.sqrt(
        local.shape[1]
    )
    structure = np.einsum(
        "s,sqa,sqb->qab", weights, fourier.conj(), fourier, optimize=True
    )
    return mean, structure


def connected(mean, raw, phase):
    """Subtract <B_a><B_b>, retaining both anchor and bond-center conventions."""
    n = phase.shape[1]
    correlation = np.einsum("qr,qab->rab", phase.conj(), raw, optimize=True) / n
    correlation -= mean.conj()[None, :, None] * mean[None, None, :]
    structure = raw.copy()
    qzero = np.flatnonzero(np.max(abs(phase - 1), axis=1) < 1e-9)
    assert len(qzero) == 1
    structure[qzero[0]] -= n * mean.conj()[:, None] * mean[None, :]
    return correlation, structure


def fold_center(structure, q):
    gauge = np.exp(-0.5j * q @ DIRECTIONS.T)
    return gauge.conj()[:, :, None] * structure * gauge[:, None, :]
