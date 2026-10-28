"""Foundation adapters use the ordinary Quantax Markov and operator kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx
import quantax as qtx


class Toy(eqx.Module):
    theta: jax.Array

    def __call__(self, s, g):
        return qtx.nn.exp_by_log(
            self.theta @ jnp.array([s[0], s[1] * g[0], s[0] * s[1], g[0] ** 2])
        )


@pytest.mark.parametrize("samples_per_coupling", [6, 12])
def test_sampler_matches_native_rng_and_chunk_order(x64, samples_per_coupling):
    qtx.sites.Chain(4, particle_type="spinful_fermion", Nparticles=(2, 2))
    qtx.set_default_dtype(jnp.float64)
    state = qtx.state.Foundation(
        Toy(jnp.array([0.1, -0.2, 0.3, 0.05])), [0.8], max_parallel=(4, 2, 4)
    )
    total = 2 * samples_per_coupling
    initial = qtx.utils.rand_states(total)
    family = qtx.sampler.FamilyMetropolis(
        state,
        [[0.8], [0.8]],
        samples_per_coupling=samples_per_coupling,
        initial_spins=initial.reshape(2, samples_per_coupling, 8),
        sweep_steps=27,
    )
    native = qtx.sampler.ParticleHop(
        state.detach([0.8]),
        nsamples=total,
        thermal_steps=0,
        sweep_steps=27,
        initial_spins=initial,
    )
    qtx.set_random_seed(1234)
    actual = family.sweep().spins.reshape(total, 8)
    qtx.set_random_seed(1234)
    expected = native.sweep().spins
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert bool(jnp.all((family.acceptance >= 0) & (family.acceptance <= 1)))
    assert not state._views


def test_context_distribution_and_shared_updates(x64):
    qtx.sites.Chain(4)
    qtx.set_default_dtype(jnp.float64)
    state = qtx.state.Foundation(
        Toy(jnp.array([0.1, -0.4, 0.2, 0.05])), [0.2], max_parallel=(32, 4, 32)
    )
    contexts = jnp.array([[0.2], [1.5]])
    sampler = qtx.sampler.FamilyMetropolis(
        state, contexts, samples_per_coupling=1024, kind="local_flip", sweep_steps=64
    )
    batch = sampler.sweep()
    basis = qtx.utils.ints_to_array(np.arange(16, dtype=np.uint64))
    for r, g in enumerate(contexts):
        logabs = state.evaluate(basis, g).logabs
        prob = np.asarray(jax.nn.softmax(2 * logabs))
        exact = prob @ np.asarray(basis[:, 1])
        assert abs(float(batch.spins[r, :, 1].mean()) - exact) < 0.10
    state.update(jnp.array([0.01, 0.02, -0.01, 0.01]))
    assert np.isfinite(
        np.asarray(state.evaluate(sampler.sweep().spins, contexts))
    ).all()
    assert not state._views


def test_native_operator_contexts_with_fused_denominators(x64):
    qtx.sites.Chain(4)
    qtx.set_default_dtype(jnp.float64)
    state = qtx.state.Foundation(
        Toy(jnp.array([0.1, -0.2, 0.3, 0.05])), [0.8], max_parallel=(4, 2, 4)
    )
    spins = qtx.utils.rand_states(16).reshape(2, 8, 4)
    contexts = jnp.array([[0.8], [1.3]])
    op = qtx.operator.Ising(h=0.7)
    flat = qtx.utils.to_distributed_array(spins.reshape(16, 4))
    gamma = qtx.utils.to_distributed_array(jnp.repeat(contexts, 8, axis=0))
    fused = op.Oloc(state, flat, contexts=gamma).reshape(2, 8)
    cached = op.Oloc(
        state, qtx.sampler.Samples(flat, state.evaluate(flat, gamma)), contexts=gamma
    ).reshape(2, 8)
    np.testing.assert_allclose(np.asarray(fused), np.asarray(cached), atol=1e-12)
    for r, g in enumerate(np.asarray(contexts)):
        reference = op.Oloc(state.detach(g), qtx.utils.to_distributed_array(spins[r]))
        np.testing.assert_allclose(
            np.asarray(fused[r]), np.asarray(reference), atol=1e-12
        )


def test_fully_blocked_hops_do_not_evaluate_illegal_connections(x64):
    qtx.sites.Chain(4, particle_type="spinful_fermion", Nparticles=(4, 4))
    qtx.set_default_dtype(jnp.float64)

    class SectorToy(Toy):
        def __call__(self, s, g):
            psi = super().__call__(s, g)
            return qtx.utils.LogArray(
                psi.sign, jnp.where(jnp.all(s == 1), psi.logabs, jnp.nan)
            )

    state = qtx.state.Foundation(
        SectorToy(jnp.array([0.1, -0.2, 0.3, 0.05])), [0.8], max_parallel=(4, 2, 4)
    )
    H = qtx.operator.HamiltonianFamily(
        [qtx.operator.Hubbard(U=0, t=1)], [[1], [1]], [[0.8], [1.3]]
    )
    spins = jnp.ones((2, 4, 8), dtype=jnp.int8)
    np.testing.assert_allclose(np.asarray(H.Oloc(state, spins)), 0, atol=1e-12)


def test_custom_noop_proposer_uses_native_free_steps(x64):
    qtx.sites.Chain(4)
    qtx.set_default_dtype(jnp.float64)
    state = qtx.state.Foundation(
        Toy(jnp.array([0.1, -0.2, 0.3, 0.05])), [0.8], max_parallel=(4, 2, 4)
    )
    sampler = qtx.sampler.FamilyMetropolis(
        state, [[0.8], [1.3]], samples_per_coupling=4, kind="local_flip", sweep_steps=17
    )

    class Noop:
        def propose(self, key, spins):
            return spins

    sampler._proposal = Noop()
    before = np.asarray(sampler.sweep(0).spins)
    after = sampler.sweep()
    np.testing.assert_array_equal(np.asarray(after.spins), before)
    np.testing.assert_allclose(np.asarray(sampler.acceptance), 1.0)
