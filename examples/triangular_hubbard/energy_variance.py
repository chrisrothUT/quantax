"""Recover raw frozen-checkpoint local-energy variance from saved estimators."""
from pathlib import Path
import json
import numpy as np


def energy_statistics(local_energy, nsites):
    """Pooled variance and delete-one-block jackknife SEM (blocks may correlate)."""
    energy = np.asarray(local_energy)
    if energy.ndim != 2 or len(energy) < 3 or not np.isfinite(energy).all():
        raise ValueError('Expected at least three finite blocks of local energies')
    mean = energy.mean()
    centered = np.abs(energy - mean)**2
    variance = centered.mean()
    b = len(energy)
    # Center first to avoid cancellation in the jackknife second moments.
    delta = energy - mean
    leave_mean = (delta.sum() - delta.sum(axis=1)) / ((b-1)*energy.shape[1])
    leave_second = (centered.sum() - centered.sum(axis=1)) / ((b-1)*energy.shape[1])
    jack = leave_second - np.abs(leave_mean)**2
    error = np.sqrt((b-1)/b * np.sum((jack-jack.mean())**2))
    return dict(energy_variance=float(variance), energy_variance_sem=float(error),
                energy_variance_per_site=float(variance/nsites),
                energy_variance_per_site_sem=float(error/nsites),
                energy_per_site_variance=float(variance/nsites**2),
                energy_variance_definition='mean |E_loc - mean E_loc|^2 over all frozen-checkpoint measurement samples; raw, unclipped; units t^2',
                energy_variance_uncertainty='Delete-one-measurement-block jackknife SEM; residual block correlation can underestimate uncertainty.')


def compute(directory):
    directory = Path(directory)
    summary = json.loads((directory/'summary.json').read_text())
    config = json.loads((Path(summary['run'])/'config.json').read_text())
    data = np.load(directory/'observables.npz')
    xyz = data['site_cell_coordinates']
    coords = data['site_coordinates']
    extent = xyz.max(axis=0)+1
    basis = np.linalg.lstsq(xyz, coords, rcond=None)[0]
    # All periodic images needed to identify nearest-neighbor displacements.
    translations = np.array([(i,j) for i in (-1,0,1) for j in (-1,0,1)]) * extent
    lengths = np.linalg.norm(coords[:,None,:] + translations[None,:,:] @ basis, axis=-1).min(axis=1)
    nearest = np.isclose(lengths, 1, atol=1e-10)
    assert nearest.sum() == 6
    n = len(coords)
    with np.load(directory/'measurement_locals.npz') as local:
        energy = -n*(local['rho_up_r'][...,nearest].sum(axis=-1) + local['rho_down_r'][...,nearest].sum(axis=-1)) + config['U']*n*local['double_occupancy']
    np.testing.assert_allclose(energy.mean()/n, summary['energy_per_site'], atol=1e-10)
    stats = energy_statistics(energy, n)
    np.save(directory/'energy_locals.npy', energy)
    summary.update(stats)
    (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return stats


if __name__ == '__main__':
    import sys
    for arg in sys.argv[1:]:
        print(arg, json.dumps(compute(arg)))
