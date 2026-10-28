"""Persistent Monte Carlo chains batched across Hamiltonian coupling points."""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from .family_samples import FamilySamples
from .common_samplers import LocalFlip, ParticleHop, SpinExchange
from ..state import Foundation
from ..global_defs import get_subkeys
from ..utils import to_distributed_array, chunk_map


class FamilyMetropolis:
    """Run all coupling/chain pairs together with dynamically conditioned amplitudes.

    Arrays are (R,S,Nmodes), with S chains per coupling. Metropolis proposals
    run on the flattened R*S axis, so no loop or static binding is needed per
    Hamiltonian. Native Metropolis performs no-op skipping and proposal steps
    inside each forward-sized chain chunk. Proposals use Quantax's existing fixed-sector kernels. The default
    target is |psi(spins|coupling)|^2 at each point, with independent random draws.
    """

    def __init__(
        self,
        state: Foundation,
        couplings,
        *,
        samples_per_coupling: int,
        kind="particle_hop",
        thermal_steps=0,
        sweep_steps=None,
        initial_spins=None,
    ):
        contexts = np.asarray(couplings, dtype=float)
        if (
            contexts.ndim != 2
            or contexts.shape[1] != state.coupling_size
            or not contexts.size
            or not np.isfinite(contexts).all()
        ):
            raise ValueError("couplings must be a finite nonempty (R,C) matrix")
        if not isinstance(samples_per_coupling, int) or samples_per_coupling <= 0:
            raise ValueError("samples_per_coupling must be a positive integer")
        if kind not in ("local_flip", "particle_hop", "spin_exchange"):
            raise ValueError("Unsupported family proposal kind")
        self.state = state
        self.couplings = jnp.asarray(contexts)
        self.nsystems, self.samples_per_coupling = (
            contexts.shape[0],
            samples_per_coupling,
        )
        total = self.nsystems * samples_per_coupling
        if total % jax.device_count():
            raise ValueError("Total chains must divide across devices")
        self.sweep_steps = 2 * state.Nmodes if sweep_steps is None else sweep_steps
        for count in (thermal_steps, self.sweep_steps):
            if not isinstance(count, int) or count < 0:
                raise ValueError("Sweep counts must be nonnegative integers")
        cls = {
            "local_flip": LocalFlip,
            "particle_hop": ParticleHop,
            "spin_exchange": SpinExchange,
        }[kind]
        self._proposal = cls(state, nsamples=total, thermal_steps=0, sweep_steps=0)
        self._engine = self._proposal
        self._contexts = to_distributed_array(
            jnp.repeat(self.couplings, samples_per_coupling, axis=0)
        )
        self._spins = self._proposal.sweep(0).spins
        if initial_spins is not None:
            self.reset(initial_spins)
        self.acceptance = None

        def sweep(network, spins, contexts, key, steps):
            samples, accepted = self._engine._sweep_variational_core(
                key, steps, spins, network, contexts, self._proposal.propose
            )
            return samples.spins, accepted

        self._sweep = eqx.filter_jit(sweep)
        if thermal_steps:
            self.sweep(thermal_steps)

    def reset(self, initial_spins):
        spins = jnp.asarray(initial_spins)
        shape = (self.nsystems, self.samples_per_coupling, self.state.Nmodes)
        if spins.shape != shape:
            raise ValueError(f"Expected initial_spins with shape {shape}")
        if not bool(jnp.all((spins == 1) | (spins == -1))):
            raise ValueError("Configurations must contain only +1 and -1")
        from ..global_defs import get_sites

        sites = get_sites()
        counts = sites.Nparticles
        if counts is not None:
            if isinstance(counts, tuple):
                up, down = jnp.split(spins > 0, 2, axis=-1)
                valid = (up.sum(axis=-1) == counts[0]) & (
                    down.sum(axis=-1) == counts[1]
                )
            else:
                valid = (spins > 0).sum(axis=-1) == counts
            if not bool(jnp.all(valid)):
                raise ValueError(
                    "Configurations must respect the global particle sector"
                )
        self._spins = to_distributed_array(
            spins.reshape(-1, self.state.Nmodes).astype(jnp.int8)
        )

    def sweep(self, steps=None):
        count = self.sweep_steps if steps is None else steps
        if not isinstance(count, int) or count < 0:
            raise ValueError("Sweep count must be a nonnegative integer")
        if count:

            def partial_sweep(count, spins, contexts):
                return self._sweep(
                    self.state.conditional_model,
                    spins,
                    contexts,
                    get_subkeys(),
                    jnp.asarray(count),
                )

            # Same loop order as Metropolis.sweep: finish the entire Markov
            # sweep on each forward-sized chain chunk before advancing the next.
            fn = chunk_map(
                partial_sweep, in_axes=(None, 0, 0), chunk_size=self.state.forward_chunk
            )
            self._spins, accepted = fn(count, self._spins, self._contexts)
            self.acceptance = (
                accepted.reshape(self.nsystems, self.samples_per_coupling).mean(axis=1)
                / count
            )
        return FamilySamples(
            self._spins.reshape(
                self.nsystems, self.samples_per_coupling, self.state.Nmodes
            )
        )
