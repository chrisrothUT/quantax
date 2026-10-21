"""Check bond geometry, hopping signs and an exact noninteracting spectrum."""

import json
from pathlib import Path
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import quantax as qtx
from anisotropic_hubbard import directional_bonds, hubbard, bond_metadata

qtx.set_default_dtype(jnp.float64)
checks = []
for L in [4, 6, 8, 10]:
    N = 3 * L * L
    lat = qtx.sites.TriangularB(
        L, particle_type="spinful_fermion", Nparticles=(N // 2, N // 2)
    )
    groups = directional_bonds(lat)
    for edges in groups:
        degree = np.bincount(edges.ravel(), minlength=N)
        np.testing.assert_array_equal(degree, np.full(N, 2))
    checks.append(dict(L=L, sites=N, **bond_metadata(lat, 0.4)))

# Small fixed-particle sector permits an exact matrix check without a half-filled ED.
lat = qtx.sites.TriangularB(2, particle_type="spinful_fermion", Nparticles=(1, 0))
np.testing.assert_allclose(
    hubbard(lat, 0.0, 1.0).todense(),
    qtx.operator.Hubbard(U=0.0, t=1.0).todense(),
    atol=1e-12,
)
for tp in [0.4, 0.8]:
    hopping = np.zeros((lat.Nsites, lat.Nsites))
    for edges, t in zip(directional_bonds(lat), [tp, 1.0, 1.0]):
        for i, j in edges:
            hopping[i, j] -= t
            hopping[j, i] -= t
    actual = hubbard(lat, 0.0, tp).todense()
    np.testing.assert_allclose(actual, actual.T, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(actual), np.linalg.eigvalsh(hopping), atol=1e-12
    )
result = dict(
    status="passed",
    geometries=checks,
    isotropic_operator_matches=True,
    anisotropic_exact_one_particle_spectra_match=True,
)
out = Path(__file__).resolve().parents[1] / "preflight.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
