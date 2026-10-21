"""Periodic TriangularB Hubbard model with t-prime on Cartesian x bonds."""

import numpy as np
import quantax as qtx
from quantax.operator.common_operators import _hop


def directional_bonds(lattice):
    if lattice.shape[0] != 1 or np.any(lattice.boundary != 1):
        raise ValueError("Expected a one-site-cell periodic lattice")
    extent = np.asarray(lattice.shape[1:])
    xyz = np.asarray(lattice.xyz_from_index[:, 1:])
    shifts = np.array([[1, 0], [-1, 1], [-2, 1]])
    vectors = shifts @ lattice.basis_vectors
    np.testing.assert_allclose(
        vectors, [[1, 0], [0.5, np.sqrt(3) / 2], [-0.5, np.sqrt(3) / 2]], atol=1e-12
    )
    groups = []
    for shift in shifts:
        target = (xyz + shift) % extent
        indices = lattice.index_from_xyz[(np.zeros(len(xyz), dtype=int), *target.T)]
        groups.append(np.column_stack([np.arange(lattice.Nsites), indices]).astype(int))
    keys = [tuple(sorted(edge)) for bonds in groups for edge in bonds]
    if len(set(keys)) != 3 * lattice.Nsites:
        raise ValueError("Unexpected duplicate or missing nearest-neighbor bonds")
    neighbors = {tuple(sorted(edge)) for edge in lattice.get_neighbor()}
    if set(keys) != neighbors:
        raise ValueError("Directional bonds do not match Quantax nearest neighbors")
    return groups


def bond_metadata(lattice, t_prime):
    groups = directional_bonds(lattice)
    return dict(
        bond_counts=[len(g) for g in groups],
        hopping_by_direction=[float(t_prime), 1.0, 1.0],
        bond_vectors=[[1.0, 0.0], [0.5, np.sqrt(3) / 2], [-0.5, np.sqrt(3) / 2]],
    )


def hubbard(lattice, U, t_prime):
    # Reuse the installed operator's hopping signs and onsite interaction.
    H = qtx.operator.Hubbard(U=U, t=1.0)
    horizontal = directional_bonds(lattice)[0]
    for i, j in horizontal:
        H = H - (t_prime - 1.0) * _hop(int(i), int(j))
    return H
