"""Fixed operator bases with batched coupling-dependent coefficients."""

from typing import TYPE_CHECKING
import math
from operator import index as integer_index
import numpy as np
import jax
import jax.numpy as jnp
from .operator import Operator
from ..sampler.family_samples import FamilySamples
from ..sampler.samples import Samples
from ..utils import to_distributed_array
from ..global_defs import get_lattice, get_sites, PARTICLE_TYPE

if TYPE_CHECKING:
    from ..state import Foundation


class HamiltonianFamily:
    """H(gamma) = sum_a coefficients(gamma,a) * terms[a].

    A fixed small operator basis is applied to *all* configurations and dynamic
    contexts at once. Only the operator-basis loop is in Python; there is no
    loop over Hamiltonians or separate bound-state evaluation per grid point.
    Each fixed basis term uses the ordinary Operator.Oloc path, including valid
    connection compaction and native chunking. Only coefficient combinations
    differ. No Python loop dispatches individual Hamiltonians.

    coefficients has shape (*grid_shape,A), couplings (*grid_shape,C). Public
    couplings/coefficients are flattened to (R,C)/(R,A); expectation results
    recover grid_shape. Samples have shape (R,S,Nmodes). ``sample_chunk`` is
    accepted for compatibility; native forward/ref chunk sizes govern evaluation.
    """

    def __init__(
        self, terms, coefficients, couplings, *, term_names=None, sample_chunk=16
    ):
        terms = tuple(terms)
        coefficient = np.asarray(coefficients)
        context = np.asarray(couplings, dtype=float)
        if not terms or not all(isinstance(op, Operator) for op in terms):
            raise ValueError("Provide a nonempty fixed basis of Quantax Operators")
        if coefficient.ndim < 2 or coefficient.shape[-1] != len(terms):
            raise ValueError("coefficients must have shape (*grid_shape, nterms)")
        if (
            context.ndim < 2
            or context.shape[:-1] != coefficient.shape[:-1]
            or not context.shape[-1]
        ):
            raise ValueError("couplings must have the same grid axes as coefficients")
        if (
            not np.isfinite(coefficient).all()
            or not np.isfinite(context).all()
            or not context.size
        ):
            raise ValueError(
                "Coefficients and coupling grids must be nonempty and finite"
            )
        if not isinstance(sample_chunk, int) or sample_chunk <= 0:
            raise ValueError("sample_chunk must be a positive integer")
        self.terms = terms
        self.grid_shape = coefficient.shape[:-1]
        self.nsystems = math.prod(self.grid_shape)
        self.coefficients = jnp.asarray(coefficient.reshape(self.nsystems, -1))
        self.couplings = jnp.asarray(context.reshape(self.nsystems, -1))
        self.term_names = (
            tuple(term_names)
            if term_names is not None
            else tuple(f"term_{i}" for i in range(len(terms)))
        )
        if len(self.term_names) != len(terms):
            raise ValueError("Provide one name per operator basis term")
        self.sample_chunk = sample_chunk

    def __len__(self):
        return self.nsystems

    def __iter__(self):
        return (self[i] for i in range(self.nsystems))

    def __getitem__(self, index):
        flat = (
            np.ravel_multi_index(index, self.grid_shape)
            if isinstance(index, tuple)
            else index
        )
        flat = integer_index(flat)
        if flat < -self.nsystems or flat >= self.nsystems:
            raise IndexError("Hamiltonian grid index out of range")
        coefficient = np.asarray(self.coefficients[flat])
        if coefficient.ndim != 1:
            raise ValueError("Index one Hamiltonian at a time")
        return sum(
            (
                float(c.real) * term if np.isrealobj(c) else complex(c) * term
                for c, term in zip(coefficient, self.terms)
            ),
            Operator([]),
        )

    def local_terms(self, state: "Foundation", samples, *, couplings=None):
        """Return every basis local energy, shape (R,S,A), from a joint pass."""
        spins = (
            samples.spins
            if isinstance(samples, FamilySamples)
            else jnp.asarray(samples)
        )
        if (
            spins.ndim != 3
            or spins.shape[0] != self.nsystems
            or spins.shape[2] != get_sites().Nmodes
            or not spins.shape[1]
        ):
            raise ValueError("Samples must have shape (R,S,Nmodes) for this grid")
        contexts = self.couplings if couplings is None else jnp.asarray(couplings)
        if contexts.shape != (self.nsystems, state.coupling_size):
            raise ValueError("Wrong context grid shape")
        _, flat, context = state._context_inputs(spins, contexts)
        if flat.shape[0] % jax.device_count():
            raise ValueError("Total samples must divide evenly across devices")
        flat, context = to_distributed_array(flat), to_distributed_array(context)
        # Compute denominators once; every fixed basis uses the same native
        # valid-connection compaction, forward chunking and segment reduction.
        batch = Samples(flat, state.evaluate(flat, context))
        result = jnp.stack(
            [term.Oloc(state, batch, contexts=context) for term in self.terms], axis=-1
        )
        return result.reshape(self.nsystems, spins.shape[1], len(self.terms))

    def combine(self, term_locals):
        """Linear combinations *before* estimating variances preserve term covariance."""
        if (
            term_locals.ndim != 3
            or term_locals.shape[0] != self.nsystems
            or term_locals.shape[-1] != len(self.terms)
        ):
            raise ValueError("Expected basis local energies of shape (R,S,A)")
        return jnp.einsum("rsa,ra->rs", term_locals, self.coefficients)

    def Oloc(self, state, samples, *, couplings=None):
        return self.combine(self.local_terms(state, samples, couplings=couplings))

    def expectation(self, state, samples, *, return_var=False, couplings=None):
        energy = self.Oloc(state, samples, couplings=couplings)
        weight = samples.reweight_factor if isinstance(samples, FamilySamples) else None
        weight = jnp.ones(energy.shape) if weight is None else jnp.asarray(weight)
        if (
            weight.shape != energy.shape
            or jnp.iscomplexobj(weight)
            or not bool(jnp.all(jnp.isfinite(weight) & (weight >= 0)))
        ):
            raise ValueError(
                "Weights must be finite, real and nonnegative, with shape (R,S)"
            )
        norm = weight.sum(axis=1, keepdims=True)
        if not bool(jnp.all(jnp.isfinite(norm) & (norm > 0))):
            raise ValueError("Each Hamiltonian needs positive finite sampling weight")
        weight = weight / norm
        mean = jnp.sum(weight * energy, axis=1)
        if not return_var:
            return mean.reshape(self.grid_shape)
        var = jnp.sum(weight * abs(energy - mean[:, None]) ** 2, axis=1)
        return mean.reshape(self.grid_shape), var.reshape(self.grid_shape)


class Anisohub(HamiltonianFamily):
    """Half-filled anisotropic triangular Hubbard family with distinct x bonds.

    H = t*T_diagonal + t_prime*T_x + U*D. Scalar/1D U and t_prime inputs
    form a Cartesian grid by default: 20 of each produces 400 Hamiltonians.
    Set cartesian=False for explicitly paired broadcastable arrays. Contexts
    are physical (U,t_prime); scale them in the model via coupling_scale.
    This operator also permits other fixed filling sectors for validation.
    """

    def __init__(self, U, t_prime, *, t=1.0, cartesian=True, sample_chunk=16):
        from .common_operators import _hop
        from .site_operator import number_u, number_d

        lattice = get_lattice()
        if (
            lattice.particle_type != PARTICLE_TYPE.spinful_fermion
            or lattice.shape[0] != 1
            or np.any(lattice.boundary != 1)
        ):
            raise ValueError(
                "Anisohub requires a periodic one-site-cell spinful-fermion triangular lattice"
            )
        U, tp = np.asarray(U, dtype=float), np.asarray(t_prime, dtype=float)
        if cartesian:
            if U.ndim > 1 or tp.ndim > 1:
                raise ValueError("Cartesian grids require scalar or 1D axes")
            ug, tpg = np.meshgrid(np.atleast_1d(U), np.atleast_1d(tp), indexing="ij")
        else:
            ug, tpg = np.broadcast_arrays(np.atleast_1d(U), np.atleast_1d(tp))
        if not np.isfinite(t) or t <= 0:
            raise ValueError("t must be finite and positive")
        xyz = lattice.xyz_from_index[:, 1:]
        extent = np.asarray(lattice.shape[1:])
        shifts = np.array(
            [[1.0, 0.0], [0.5, np.sqrt(3) / 2], [-0.5, np.sqrt(3) / 2]]
        ) @ np.linalg.inv(lattice.basis_vectors)
        np.testing.assert_allclose(shifts, np.rint(shifts), atol=1e-12)
        groups = []
        for shift in np.rint(shifts).astype(int):
            target = (xyz + shift) % extent
            indices = lattice.index_from_xyz[(np.zeros(len(xyz), dtype=int), *target.T)]
            groups.append([(i, int(j)) for i, j in enumerate(indices)])
        keys = {tuple(sorted(edge)) for group in groups for edge in group}
        neighbors = {tuple(sorted(edge)) for edge in lattice.get_neighbor()}
        if len(keys) != 3 * lattice.Nsites or keys != neighbors:
            raise ValueError("Degenerate periodic bonds: use extents at least three")
        tx = sum((-_hop(i, j) for i, j in groups[0]), Operator([]))
        diagonal = sum(
            (-_hop(i, j) for group in groups[1:] for i, j in group), Operator([])
        )
        double = sum(
            (number_u(i) @ number_d(i) for i in range(lattice.Nsites)), Operator([])
        )
        coefficients = np.stack([np.full_like(ug, t), tpg, ug], axis=-1)
        super().__init__(
            (diagonal, tx, double),
            coefficients,
            np.stack([ug, tpg], axis=-1),
            term_names=("T_diagonal", "T_x", "D"),
            sample_chunk=sample_chunk,
        )
