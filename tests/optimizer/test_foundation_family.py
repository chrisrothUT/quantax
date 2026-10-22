"""Batched family energies, Jacobians and shared SR versus native operators."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import quantax as qtx


class Toy(eqx.Module):
    theta: jax.Array
    complex_output: bool = eqx.field(static=True, default=False)

    def __call__(self, s, g):
        features = jnp.array([s[0], s[1] * g[0], s[0] * s[1], g[0] ** 2])
        logpsi = self.theta @ features
        if self.complex_output:
            logpsi += 1j * self.theta[1] * features[1]
        return qtx.nn.exp_by_log(logpsi)


def family():
    contexts = np.array([[0.8], [1.2]])
    return qtx.operator.HamiltonianFamily(
        [qtx.operator.Ising(h=0), qtx.operator.Ising(J=0, h=1)],
        [[1, 0.8], [1, 1.2]],
        contexts,
        sample_chunk=4,
    )


@pytest.mark.parametrize("complex_output", [False, True])
@pytest.mark.parametrize("clip_mode", ["mean_std", "median_mad"])
def test_batched_matches_native_and_sr(x64, complex_output, clip_mode):
    qtx.sites.Chain(4)
    qtx.set_default_dtype(jnp.complex128 if complex_output else jnp.float64)
    state = qtx.state.Foundation(
        Toy(jnp.array([0.1, -0.2, 0.3, 0.05]), complex_output), [0.8]
    )
    H = family()
    spins = qtx.utils.rand_states(16).reshape(2, 8, 4)
    weights = jnp.arange(1.0, 17.0).reshape(2, 8)
    batch = qtx.sampler.FamilySamples(spins, weights)
    fast = qtx.optimizer.FoundationSR(
        state, H, weights=[0.3, 0.7], clip=1.0, clip_mode=clip_mode
    )
    O, E = fast.get_equation(batch)
    assert not state._views
    dynamic_psi = np.asarray(state.evaluate(spins, H.couplings))
    dynamic_jac = np.asarray(state.jacobian_at(spins, H.couplings))
    terms = np.asarray(H.local_terms(state, batch))
    native = []
    for r, context in enumerate(np.asarray(H.couplings)):
        view = state.at(context)
        s = qtx.utils.to_distributed_array(spins[r])
        native.append(
            qtx.sampler.Samples(
                s, reweight_factor=qtx.utils.to_distributed_array(weights[r])
            )
        )
        np.testing.assert_allclose(dynamic_psi[r], np.asarray(view(s)), atol=1e-12)
        np.testing.assert_allclose(
            dynamic_jac[r], np.asarray(view.jacobian(s)), atol=1e-12
        )
        for a, term in enumerate(H.terms):
            np.testing.assert_allclose(
                terms[r, :, a], np.asarray(term.Oloc(view, native[-1])), atol=1e-12
            )
        np.testing.assert_allclose(
            np.asarray(H.combine(jnp.asarray(terms)))[r],
            np.asarray(H[r].Oloc(view, native[-1])),
            atol=1e-12,
        )
    legacy = qtx.optimizer.FoundationSR(
        state,
        [H[0], H[1]],
        [[0.8], [1.2]],
        weights=[0.3, 0.7],
        clip=1.0,
        clip_mode=clip_mode,
    )
    oldO, oldE = legacy.get_equation(native)
    np.testing.assert_allclose(O, oldO, atol=1e-12)
    np.testing.assert_allclose(E, oldE, atol=1e-12)
    np.testing.assert_allclose(fast.energies, legacy.energies, atol=1e-12)
    np.testing.assert_allclose(fast.variances, legacy.variances, atol=1e-12)
    mean, var = H.expectation(state, batch, return_var=True)
    np.testing.assert_allclose(mean.real, fast.energies, atol=1e-12)
    np.testing.assert_allclose(var, fast.variances, atol=1e-12)
    state.update(fast.get_step(batch))
    assert np.isfinite(np.asarray(state.evaluate(spins, H.couplings))).all()


def test_batched_fermion_sampler(x64):
    qtx.sites.Chain(4, particle_type="spinful_fermion", Nparticles=(2, 2))
    qtx.set_default_dtype(jnp.float64)
    state = qtx.state.Foundation(Toy(jnp.array([0.1, -0.2, 0.3, 0.05])), [0.8])
    # Two chains each: only the combined batch must divide over four devices.
    sampler = qtx.sampler.FamilyMetropolis(
        state, [[0.8], [1.2]], samples_per_coupling=2, sweep_steps=4
    )
    batch = sampler.sweep()
    assert batch.spins.shape == (2, 2, 8)
    for part in np.split(np.asarray(batch.spins) > 0, 2, axis=-1):
        np.testing.assert_array_equal(part.sum(axis=-1), 2)
    state.update(jnp.array([0.01, 0.02, -0.01, 0.01]))
    assert np.isfinite(
        np.asarray(state.evaluate(sampler.sweep().spins, sampler.couplings))
    ).all()
    assert not state._views
    sampler.reset(batch.spins)
    with pytest.raises(ValueError):
        sampler.reset(jnp.ones_like(batch.spins))
    with pytest.raises(ValueError):
        sampler.reset(jnp.zeros_like(batch.spins))


def test_anisohub_400_grid_and_native_fermion_energies(x64):
    qtx.sites.Triangular((3, 3), particle_type="spinful_fermion", Nparticles=(1, 1))
    qtx.set_default_dtype(jnp.float64)
    model = qtx.model.ConditionalTransformer(
        2, dim=4, heads=1, layers=0, dtype=jnp.float64, coupling_scale=(10, 1)
    )
    U, tp = np.linspace(4, 16, 20), np.linspace(0.8, 1, 20)
    H = qtx.operator.Anisohub(U, tp, sample_chunk=4)
    assert H.grid_shape == (20, 20) and len(H) == 400
    np.testing.assert_allclose(
        np.asarray(H.couplings).reshape(20, 20, 2)[7, 9], [U[7], tp[9]]
    )
    state = qtx.state.Foundation(model, H.couplings[0], max_parallel=(128, 16, 128))
    spins = qtx.utils.rand_states(400).reshape(400, 1, 18)
    terms = H.local_terms(state, spins)
    energies = np.asarray(H.combine(terms))
    assert terms.shape == (400, 1, 3) and np.isfinite(energies).all()
    assert not state._views
    # Match all three native basis local energies and the full native Hubbard.
    for r in [0, 173, 399]:
        view = state.at(np.asarray(H.couplings[r]))
        s = qtx.utils.to_distributed_array(jnp.repeat(spins[r], 4, axis=0))
        batch = qtx.sampler.Samples(s)
        for a, term in enumerate(H.terms):
            np.testing.assert_allclose(
                terms[r, 0, a], np.asarray(term.Oloc(view, batch))[0], atol=1e-11
            )
        np.testing.assert_allclose(
            energies[r, 0], np.asarray(H[r].Oloc(view, batch))[0], atol=1e-11
        )
    np.testing.assert_allclose(
        np.asarray(H[-1].todense()),
        np.asarray(qtx.operator.Hubbard(U=16, t=1).todense()),
        atol=1e-12,
    )
    # With one up fermion, horizontal one-particle matrix elements are -t'.
    lattice = qtx.get_sites()
    assert lattice.Nsites == 9
    paired = qtx.operator.Anisohub([4, 8], [0.8, 1], cartesian=False)
    assert paired.grid_shape == (2,)


def test_dynamic_context_reuses_compilation_and_projection(x64):
    qtx.sites.Chain(4)
    qtx.set_default_dtype(jnp.float64)
    traces = []

    class Counting(Toy):
        def __call__(self, s, g):
            traces.append(1)
            return super().__call__(s, g)

    state = qtx.state.Foundation(
        Counting(jnp.array([0.1, -0.2, 0.3, 0.05])),
        [0.8],
        symm=qtx.symmetry.Z2Inversion(),
    )
    spins = qtx.utils.rand_states(8).reshape(2, 4, 4)
    g = jnp.array([[0.8], [1.2]])
    first = np.asarray(state.evaluate(spins, g))
    jac = np.asarray(state.jacobian_at(spins, g))
    count = len(traces)
    second = np.asarray(state.evaluate(spins, g + 0.1))
    state.jacobian_at(spins, g + 0.1).block_until_ready()
    assert len(traces) == count and not np.allclose(first, second)
    assert not state._views
    for r in range(2):
        s = qtx.utils.to_distributed_array(spins[r])
        np.testing.assert_allclose(
            first[r], np.asarray(state.at(np.asarray(g[r]))(s)), atol=1e-12
        )
        np.testing.assert_allclose(
            jac[r], np.asarray(state.at(np.asarray(g[r])).jacobian(s)), atol=1e-12
        )


def test_diagonal_basis_covariance_and_invalid_weights(x64):
    qtx.sites.Chain(4)
    qtx.set_default_dtype(jnp.float64)
    state = qtx.state.Foundation(Toy(jnp.array([0.1, -0.2, 0.3, 0.05])), [0.8])
    z = qtx.operator.sigma_z(0)
    H = qtx.operator.HamiltonianFamily([z, 2 * z], [[1, -0.5]], [[0.8]])
    assert len(list(H)) == 1
    with pytest.raises(IndexError):
        H[1]
    spins = qtx.utils.rand_states(8).reshape(1, 8, 4)
    mean, var = H.expectation(state, spins, return_var=True)
    np.testing.assert_allclose(mean, 0, atol=1e-12)
    np.testing.assert_allclose(var, 0, atol=1e-12)
    for weights in [jnp.zeros((1, 8)), -jnp.ones((1, 8)), jnp.full((1, 8), jnp.nan)]:
        with pytest.raises(ValueError):
            H.expectation(state, qtx.sampler.FamilySamples(spins, weights))


def test_horizontal_hopping_matches_independent_construction(x64):
    from quantax.operator.common_operators import _hop

    lattice = qtx.sites.Triangular(
        (3, 3), particle_type="spinful_fermion", Nparticles=(1, 1)
    )
    H = qtx.operator.Anisohub(6, 0.8)
    xyz = lattice.xyz_from_index[:, 1:]
    shift = np.rint(np.array([1.0, 0.0]) @ np.linalg.inv(lattice.basis_vectors)).astype(
        int
    )
    target = (xyz + shift) % 3
    neighbors = lattice.index_from_xyz[(np.zeros(9, dtype=int), *target.T)]
    expected = qtx.operator.Hubbard(U=6, t=1)
    for i, j in enumerate(neighbors):
        expected = expected - (0.8 - 1) * _hop(i, int(j))
    np.testing.assert_allclose(
        np.asarray(H[0].todense()), np.asarray(expected.todense()), atol=1e-6
    )
