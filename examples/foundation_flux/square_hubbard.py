"""Square Hubbard torus with independent total twists in radians."""

import numpy as np
import quantax as qtx
from quantax.operator import HamiltonianFamily, Operator
from quantax.operator.site_operator import (
    create_u,
    create_d,
    annihilate_u,
    annihilate_d,
    number_u,
    number_d,
)


class SquareTwistHubbard(HamiltonianFamily):
    """Uniform gauge: forward hopping -exp(i theta_axis/L), reverse conjugate.

    Each plaquette has zero magnetic flux; theta_x/y are torus holonomies.
    The uniform gauge preserves translation equivariance of the shared CNN.
    """

    def __init__(self, flux_x, flux_y, *, U=4.0, sample_chunk=4):
        lat = qtx.get_lattice()
        extent = np.asarray(lat.shape[1:])
        xyz = np.asarray(lat.xyz_from_index[:, 1:])
        terms = []
        for axis in range(2):
            shift = np.eye(2, dtype=int)[axis]
            target = (xyz + shift) % extent
            indices = lat.index_from_xyz[(np.zeros(len(xyz), dtype=int), *target.T)]
            cosine, sine = Operator([]), Operator([])
            for i, j in enumerate(indices):
                for create, annihilate in [
                    (create_u, annihilate_u),
                    (create_d, annihilate_d),
                ]:
                    forward = create(int(j)) @ annihilate(i)
                    reverse = create(i) @ annihilate(int(j))
                    cosine += -forward - reverse
                    sine += -1j * forward + 1j * reverse
            terms.extend([cosine, sine])
        terms.append(
            sum((number_u(i) @ number_d(i) for i in range(lat.Nsites)), Operator([]))
        )
        x, y = np.meshgrid(flux_x, flux_y, indexing="ij")
        coeff = np.stack(
            [
                np.cos(x / extent[0]),
                np.sin(x / extent[0]),
                np.cos(y / extent[1]),
                np.sin(y / extent[1]),
                np.full_like(x, U),
            ],
            axis=-1,
        )
        super().__init__(
            terms,
            coeff,
            np.stack([x, y], axis=-1),
            term_names=("Tx_cos", "Tx_sin", "Ty_cos", "Ty_sin", "D"),
            sample_chunk=sample_chunk,
        )
