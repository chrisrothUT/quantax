"""Check Peierls signs, Hermiticity and twist spectrum in the one-particle sector."""

import numpy as np
import quantax as qtx
import jax.numpy as jnp

qtx.set_default_dtype(jnp.complex128)
from square_hubbard import SquareTwistHubbard

qtx.sites.Square(3, particle_type="spinful_fermion", Nparticles=(1, 0))
angles = np.array([0.0, 0.7, 2 * np.pi])
family = SquareTwistHubbard(angles, angles)
matrices = [op.todense() for op in family.terms]
for coeff, (x, y) in zip(np.asarray(family.coefficients), np.asarray(family.couplings)):
    matrix = sum(c * m for c, m in zip(coeff, matrices))
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-12)
    k = 2 * np.pi * np.arange(3) / 3
    expected = (
        -2 * np.cos(k[:, None] - x / 3) - 2 * np.cos(k[None, :] - y / 3)
    ).ravel()
    np.testing.assert_allclose(
        np.linalg.eigvalsh(matrix), np.sort(expected), atol=1e-12
    )
print(
    "PASS: 9 twist pairs, Hermitian hopping and analytic free-particle spectra, including 2pi periodicity"
)
