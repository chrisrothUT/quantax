"""Flavor-blocked fermion lattices, using the existing spinless backend."""

from collections.abc import Sequence
from itertools import product
from numbers import Integral
import numpy as np
import jax
import jax.numpy as jnp
from .lattice import Lattice
from ..global_defs import PARTICLE_TYPE


class FlavorLattice(Lattice):
    """Periodic Bravais lattice with multiple fermion flavors per physical site.

    Mode ``a * Nphysical + i`` represents flavor ``a`` at physical site ``i``.
    Internally these are spinless modes with coincident flavor offsets. Thus
    ``Nsites == Nmodes == nflavors * Nphysical``; use ``Nphysical`` when
    normalizing physical densities or energies per site.

    ``Nparticles`` may be a total integer or one integer per flavor. In the latter
    case ``rand_states`` initializes each population separately, and the usual
    ``ParticleHop`` sampler preserves it because neighbor tables never mix
    flavors. The underlying QuSpin basis fixes the *total* number only: explicitly
    restrict it for fixed-flavor exact diagonalization or exact sampling.

    Only periodic boundaries and one physical site per unit cell are supported.
    Independent occupations of each flavor allow two or more different flavors
    on the same physical site; Pauli exclusion acts separately on each mode.
    """

    def __init__(
        self,
        extent: Sequence[int],
        basis_vectors,
        nflavors: int = 3,
        Nparticles: int | Sequence[int] | None = None,
    ):
        if (
            not isinstance(nflavors, Integral)
            or isinstance(nflavors, bool)
            or nflavors < 2
        ):
            raise ValueError("nflavors must be an integer >= 2")
        if len(extent) == 0 or any(
            not isinstance(n, Integral) or isinstance(n, bool) or n <= 0 for n in extent
        ):
            raise ValueError("extent must contain positive integers")
        basis = np.asarray(basis_vectors, dtype=float)
        if basis.shape != (len(extent), len(extent)) or not np.isfinite(basis).all():
            raise ValueError(
                "basis_vectors must be a finite square matrix matching extent"
            )
        if np.linalg.matrix_rank(basis) != len(extent):
            raise ValueError("basis_vectors must be linearly independent")
        self._nflavors = int(nflavors)
        physical = int(np.prod(extent))
        self._flavor_counts = None
        if Nparticles is not None:
            if isinstance(Nparticles, (int, np.integer)) and not isinstance(
                Nparticles, bool
            ):
                total = int(Nparticles)
                if not 0 <= total <= nflavors * physical:
                    raise ValueError(
                        "Total particle count is outside the Hilbert space"
                    )
            else:
                if not isinstance(Nparticles, Sequence) or len(Nparticles) != nflavors:
                    raise ValueError("Specify one particle count per flavor")
                if any(
                    not isinstance(n, Integral)
                    or isinstance(n, bool)
                    or not 0 <= n <= physical
                    for n in Nparticles
                ):
                    raise ValueError(
                        "Each flavor count must be an integer in [0, Nphysical]"
                    )
                self._flavor_counts = tuple(int(n) for n in Nparticles)
                total = sum(self._flavor_counts)
        else:
            total = None
        super().__init__(
            extent,
            basis,
            np.zeros((nflavors, len(extent))),
            boundary=1,
            particle_type=PARTICLE_TYPE.spinless_fermion,
            Nparticles=total,
        )

    @property
    def nflavors(self) -> int:
        """Number of internal fermion flavors."""
        return self._nflavors

    @property
    def Nphysical(self) -> int:
        """Number of spatial sites, excluding the flavor multiplicity."""
        return self.ncells

    @property
    def flavor_counts(self) -> tuple[int, ...] | None:
        """Fixed populations requested on construction, or None for total only."""
        return self._flavor_counts

    def mode_index(self, site: int, flavor: int) -> int:
        """Convert a zero-based physical site/flavor pair to a fermion mode."""
        if (
            not isinstance(site, Integral)
            or isinstance(site, bool)
            or not 0 <= site < self.Nphysical
        ):
            raise ValueError("Physical site index out of range")
        if (
            not isinstance(flavor, Integral)
            or isinstance(flavor, bool)
            or not 0 <= flavor < self.nflavors
        ):
            raise ValueError("Flavor index out of range")
        return int(flavor * self.Nphysical + site)

    def count_flavors(self, spins: jax.Array) -> jax.Array:
        """Count occupied modes per flavor in one configuration or a batch."""
        spins = jnp.asarray(spins)
        if spins.shape[-1] != self.Nmodes:
            raise ValueError("Configuration length does not match this lattice")
        return (
            spins.reshape(*spins.shape[:-1], self.nflavors, self.Nphysical) > 0
        ).sum(axis=-1)

    def _get_dist_sign(self):
        # Exact finite closest-image search. A candidate image no longer than the
        # centered displacement bounds its integer winding via sigma_min(periods).
        physical = self.coord[: self.Nphysical]
        periods = np.asarray(self.shape[1:])[:, None] * self.basis_vectors
        raw = physical[:, None] - physical[None, :]
        fractional = raw @ np.linalg.inv(periods)
        centered = (fractional - np.rint(fractional)) @ periods
        radius = float(np.linalg.norm(centered, axis=-1).max())
        bound = int(np.ceil(2 * radius / np.linalg.svd(periods, compute_uv=False)[-1]))
        shortest = np.full((self.Nphysical, self.Nphysical), np.inf)
        for winding in product(range(-bound, bound + 1), repeat=len(self.shape) - 1):
            distance = np.linalg.norm(centered - np.asarray(winding) @ periods, axis=-1)
            shortest = np.minimum(shortest, distance)
        # Same-site flavor copies coincide; spatial neighbor shells skip zero.
        distances = np.tile(shortest, (self.nflavors, self.nflavors))
        return distances, np.ones(distances.shape, dtype=np.int64)

    def _compute_neighbor(self, max_neighbor: int = 1) -> None:
        super()._compute_neighbor(max_neighbor)
        self._neighbors = [
            pairs[pairs[:, 0] // self.Nphysical == pairs[:, 1] // self.Nphysical]
            for pairs in self._neighbors
        ]


class TriangularFlavors(FlavorLattice):
    """Periodic triangular lattice with flavor-conserving spatial neighbors.

    Example: ``TriangularFlavors((3, 2), Nparticles=(2, 2, 2))`` has six
    physical sites, 18 fermion modes, and six particles (one per physical site).
    Neighbor bonds are unique unordered physical pairs. On very small tori,
    distinct infinite-lattice directions can coincide and are counted once.
    """

    def __init__(
        self,
        extent: int | Sequence[int],
        nflavors: int = 3,
        Nparticles: int | Sequence[int] | None = None,
    ):
        if isinstance(extent, bool):
            raise ValueError("Spatial extents must be positive integers")
        if isinstance(extent, (int, np.integer)):
            spatial_extent = (int(extent), int(extent))
        else:
            spatial_extent = extent
        if len(spatial_extent) != 2:
            raise ValueError("TriangularFlavors requires two spatial extents")
        super().__init__(
            spatial_extent, [[1.0, 0.0], [0.5, np.sqrt(3) / 2]], nflavors, Nparticles
        )
