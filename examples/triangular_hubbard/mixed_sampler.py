"""Symmetric nearest-neighbor site and single-spin exchange mixture."""
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import quantax as qtx


class MixedExchange(qtx.sampler.ParticleHop):
    """70% whole-site swaps, 30% single-spin ParticleHop attempts.

    Mixture probabilities apply to attempted moves, including no-ops/rejections.
    Both components conserve each spin population; whole-site swaps additionally
    preserve doublon count. The regular periodic lattice makes both proposals
    symmetric, so the inherited |psi_new/psi_old|^2 rule applies unchanged.
    """
    @partial(jax.jit, static_argnums=0)
    def propose_with_kind(self, key, old_spins):
        khop, ksite, kneighbor, kmix = jr.split(key, 4)
        single = super().propose(khop, old_spins)
        batch, modes = old_spins.shape
        n = modes // 2
        i = jr.randint(ksite, (batch,), 0, n)
        neighbor_slot = jr.randint(kneighbor, (batch,), 0, self._neighbors.shape[1])
        j = self._neighbors[i, neighbor_slot]
        j = jnp.where(j < 0, i, j)
        def swap(s, a, b):
            idx = jnp.array([a, a+n, b, b+n])
            return s.at[idx].set(s[jnp.array([b, b+n, a, a+n])])
        both = jax.vmap(swap)(old_spins, i, j)
        whole_site = jr.uniform(kmix, (batch,)) < 0.7
        return jnp.where(whole_site[:,None], both, single), whole_site

    @partial(jax.jit, static_argnums=0)
    def propose(self, key, old_spins):
        return self.propose_with_kind(key, old_spins)[0]


def make_sampler(state, *, kind='particle_hop', **kwargs):
    from sampler_diagnostics import DiagnosticMixedExchange
    cls = {'particle_hop': qtx.sampler.ParticleHop, 'mixed_exchange': DiagnosticMixedExchange}[kind]
    return cls(state, **kwargs)
