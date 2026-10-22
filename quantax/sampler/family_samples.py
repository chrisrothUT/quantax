"""Samples with explicit Hamiltonian and per-Hamiltonian chain axes."""

from dataclasses import dataclass
import jax


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FamilySamples:
    """Configurations (R,S,Nmodes) and optional sampling weights (R,S).

    R is the number of Hamiltonians; S is samples per coupling. The physical
    coupling order is supplied by the HamiltonianFamily or FamilyMetropolis.
    Different coupling points always keep their own configurations/weights.
    """

    spins: jax.Array
    reweight_factor: jax.Array | None = None

    @property
    def nsystems(self):
        return self.spins.shape[0]

    @property
    def samples_per_coupling(self):
        return self.spins.shape[1]

    def tree_flatten(self):
        return (self.spins, self.reweight_factor), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)
