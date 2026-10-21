"""Flavor-conserving Hubbard operators on a flavor-blocked fermion lattice."""

from numbers import Integral
import numpy as np
from .operator import Operator
from .site_operator import create, annihilate, number
from ..sites.flavor_lattice import FlavorLattice
from ..global_defs import get_sites, is_default_cpl


def _lattice() -> FlavorLattice:
    sites = get_sites()
    if not isinstance(sites, FlavorLattice):
        raise ValueError(
            "Flavor operators require a FlavorLattice or TriangularFlavors"
        )
    return sites


def create_flavor(site: int, flavor: int) -> Operator:
    """Return c†[site,flavor]; both indices are zero based."""
    return create(_lattice().mode_index(site, flavor))


def annihilate_flavor(site: int, flavor: int) -> Operator:
    """Return c[site,flavor], with the usual fermionic anticommutation signs."""
    return annihilate(_lattice().mode_index(site, flavor))


def number_flavor(site: int, flavor: int) -> Operator:
    """Return n[site,flavor] = c†[site,flavor] c[site,flavor]."""
    return number(_lattice().mode_index(site, flavor))


def hopping_flavor(i: int, j: int, flavor: int, t: complex = 1.0) -> Operator:
    r"""Hermitian bond term -t c†[i,a]c[j,a] - t* c†[j,a]c[i,a].

    A physical bond must be supplied once; this constructor includes its reverse.
    Complex t requires a complex Quantax default dtype.
    """
    value = np.asarray(t)
    if (
        value.ndim
        or not np.issubdtype(value.dtype, np.number)
        or not np.isfinite(value)
    ):
        raise ValueError("t must be a finite scalar")
    if i == j:
        raise ValueError("Hopping requires two different physical sites")
    if np.imag(value) != 0 and not is_default_cpl():
        raise ValueError("Complex hopping requires a complex default dtype")
    forward = create_flavor(i, flavor) @ annihilate_flavor(j, flavor)
    value = value.item()
    return -value * forward - np.conjugate(value) * forward.H


def HubbardFlavors(U: float, t: float = 1.0, n_neighbor: int = 1) -> Operator:
    r"""SU(nflavors) Hubbard Hamiltonian with flavor-conserving hopping.

    H = -t sum_<ij>,a (c†[i,a] c[j,a] + h.c.)
        + U sum_i sum_a<b n[i,a] n[i,b].

    Each unordered bond and each unordered flavor pair is counted once. Three
    particles on a physical site cost 3U for three flavors. Equal t and U for
    all flavors give SU(nflavors) symmetry. No chemical potential is included.
    ``n_neighbor`` selects one spatial distance shell. The class does not
    include a projection forbidding multiple flavors on a physical site.
    """
    lattice = _lattice()
    for name, value in [("U", U), ("t", t)]:
        if not np.isscalar(value) or not np.isreal(value) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite real scalar")
    if (
        not isinstance(n_neighbor, Integral)
        or isinstance(n_neighbor, bool)
        or n_neighbor < 1
    ):
        raise ValueError("n_neighbor must be a positive integer")
    terms = []
    for i, j in lattice.get_neighbor(int(n_neighbor)):
        flavor = int(i // lattice.Nphysical)
        terms.append(
            hopping_flavor(
                int(i % lattice.Nphysical), int(j % lattice.Nphysical), flavor, float(t)
            )
        )
    for i in range(lattice.Nphysical):
        for a in range(lattice.nflavors):
            for b in range(a + 1, lattice.nflavors):
                terms.append(float(U) * (number_flavor(i, a) @ number_flavor(i, b)))
    return sum(terms, Operator([]))
