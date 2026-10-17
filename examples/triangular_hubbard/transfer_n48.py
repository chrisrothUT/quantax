"""Transfer the CNN and signed-displacement pairing table between lattice sizes."""

import hashlib
import inspect
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import quantax as qtx


def transfer_model(target, source_dir, target_L, interaction):
    source_dir = source_dir.resolve()
    config = json.loads((source_dir / "config.json").read_text())
    assert json.loads((source_dir / "summary.json").read_text())["status"] == "complete"
    assert config["U"] == interaction and config["L"] == 4 and target_L == 6
    assert config["model_sha256"] == hashlib.sha256(
        Path(inspect.getsourcefile(type(target))).read_bytes()
    ).hexdigest()
    qtx.sites.TriangularB(4, particle_type="spinful_fermion", Nparticles=(24, 24))
    source = type(target).resconv(
        d=16, nblocks=8, channels=16, kernel_size=3,
        dtype=jnp.float32, pairing_dtype=jnp.float64,
    )
    checkpoint = source_dir / "latest.eqx"
    source = eqx.tree_deserialise_leaves(checkpoint, source)
    # Restore the target global geometry before constructing any state or sampler.
    qtx.sites.TriangularB(6, particle_type="spinful_fermion", Nparticles=(54, 54))
    coords = np.indices((18, 6)).reshape(2, -1).T
    signed = np.where(coords <= np.array([9, 3]), coords, coords - np.array([18, 6]))
    source_indices = np.ravel_multi_index((signed % np.array([12, 4])).T, (12, 4))
    table = np.asarray(source.pairing_weights)[:, :, source_indices, :]
    reverse = np.ravel_multi_index(((-coords) % np.array([18, 6])).T, (18, 6))
    table = (table - table.swapaxes(0, 1)[:, :, reverse, :]) / 2
    index = np.asarray(target._weight_index)
    sign = np.asarray(target._weight_sign)
    flat = table.reshape(-1, target.d)
    weights = np.empty(target.weight_real.shape)
    for k in range(1, len(weights) + 1):
        row = np.flatnonzero(index == k)[0]
        weights[k - 1] = flat[row] * sign[row]
    target = eqx.tree_at(
        lambda m: (m.net, m.weight_real, m.log_decay), target,
        (source.net, jnp.asarray(weights), source.log_decay),
    )
    np.testing.assert_allclose(np.asarray(target.pairing_weights), table, atol=1e-14)
    return target, dict(
        checkpoint=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_sites=48, target_sites=108,
        method="Copy CNN and log_decay; map shortest signed cell displacements modulo (12,4); antisymmetrize on target torus",
        optimizer="fresh AdamSR state", sampler="fresh target-sector thermalization",
    )
