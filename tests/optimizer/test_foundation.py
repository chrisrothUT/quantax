"""Numerical checks of conditional gauge invariance and the shared SR equations."""

import io
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
        if self.complex_output:
            logpsi = (
                self.theta[0] * features[0]
                + self.theta[2] * features[2]
                + 1j * (self.theta[1] * features[1] + self.theta[3] * features[3])
            )
        else:
            logpsi = self.theta @ features
        return qtx.nn.exp_by_log(logpsi)


class StubHamiltonian:
    def __init__(self, energies):
        self.values = jnp.asarray(energies)

    def Oloc(self, state, samples):
        return self.values


def setup(complex_output=False):
    qtx.sites.Chain(4)
    qtx.set_default_dtype(jnp.complex128 if complex_output else jnp.float64)
    return qtx.state.Foundation(
        Toy(jnp.array([0.1, -0.2, 0.3, 0.05]), complex_output), [0.8]
    )


def samples(spins, weight=None):
    return qtx.sampler.Samples(
        qtx.utils.to_distributed_array(jnp.asarray(spins)),
        reweight_factor=(
            None
            if weight is None
            else qtx.utils.to_distributed_array(jnp.asarray(weight))
        ),
    )


@pytest.mark.parametrize("complex_output", [False, True])
def test_joint_covariance_and_conditional_gauge(x64, complex_output):
    state = setup(complex_output)
    conditions = [[0.8], [1.3]]
    spins = np.array([[1, 1, -1, -1], [1, -1, 1, -1], [-1, 1, -1, 1], [-1, -1, 1, 1]])
    batches = [
        samples(spins, [1.0, 2.0, 3.0, 4.0]),
        samples(np.tile(spins, (2, 1)), np.arange(1.0, 9.0)),
    ]
    eloc = [
        np.array([1.0, 3.0, 2.0, 7.0]),
        np.array([11.0, 12.0, 14.0, 13.0, 9.0, 8.0, 10.0, 12.0]),
    ]
    if complex_output:
        eloc = [e + 1j * np.arange(len(e)) * 0.1 for e in eloc]
    p = np.array([0.25, 0.75])
    opt = qtx.optimizer.FoundationSR(
        state, [StubHamiltonian(e) for e in eloc], conditions, weights=p, clip=None
    )
    Obar, Ebar = map(np.asarray, opt.get_equation(batches))
    metric = np.zeros((4, 4), dtype=complex)
    force = np.zeros(4, dtype=complex)
    for weight, view, batch, E in zip(p, opt.states, batches, eloc):
        O = np.asarray(view.jacobian(batch.spins))
        w = np.asarray(batch.reweight_factor)
        w = w / w.sum()
        Oc = O - (w[:, None] * O).sum(axis=0)
        Ec = E - (w * E).sum()
        metric += weight * (Oc.conj().T @ (w[:, None] * Oc))
        force += weight * (Oc.conj().T @ (w * Ec))
    np.testing.assert_allclose(Obar.conj().T @ Obar, metric, atol=1e-12)
    np.testing.assert_allclose(Obar.conj().T @ Ebar, force, atol=1e-12)
    # The g^2 parameter is a different normalization/phase for each Hamiltonian.
    # A globally centered mixture would incorrectly give this direction a metric.
    np.testing.assert_allclose(Obar[:, -1], 0, atol=1e-12)
    opt2 = qtx.optimizer.FoundationSR(
        state,
        [StubHamiltonian(e + c) for e, c in zip(eloc, [100.0, -23.0])],
        conditions,
        weights=p,
        clip=None,
    )
    O2, E2 = map(np.asarray, opt2.get_equation(batches))
    np.testing.assert_allclose(O2, Obar, atol=1e-12)
    np.testing.assert_allclose(E2, Ebar, atol=1e-12)


def test_shared_views_detach_and_serialization(x64):
    state = setup()
    s = qtx.utils.rand_states(8)
    a = state.at([0.8])
    b = state.at([1.3])
    detached = state.detach([1.3])
    assert state.nparams == 4 and a.nparams == 4 and state.at([1.3]) is b
    original = np.asarray(detached(s))
    before = np.asarray(a(s))
    with pytest.raises(RuntimeError):
        a.update(jnp.ones(4) * 0.01)
    state.update(jnp.array([0.02, 0.01, -0.03, 0.01]))
    assert not np.allclose(np.asarray(a(s)), before)
    np.testing.assert_allclose(
        np.asarray(b(s)),
        np.asarray(
            qtx.state.Variational(
                qtx.model.Conditioned(state.conditional_model, [1.3])
            )(s)
        ),
    )
    np.testing.assert_allclose(np.asarray(detached(s)), original)
    stream = io.BytesIO()
    state.save(stream)
    stream.seek(0)
    restored = qtx.state.Foundation(Toy(jnp.zeros(4)), [0.8], param_file=stream)
    np.testing.assert_allclose(np.asarray(restored.at([1.3])(s)), np.asarray(b(s)))


def test_single_system_reduces_to_existing_sr(x64):
    state = setup()
    spins = qtx.utils.rand_states(16)
    H = qtx.operator.Ising(h=0.8)
    old = qtx.optimizer.SR(state.detach([0.8]), H, clip=None)
    new = qtx.optimizer.FoundationSR(state, [H], [[0.8]], clip=None)
    np.testing.assert_allclose(
        np.asarray(new.get_step([spins])),
        np.asarray(old.get_step(spins)),
        rtol=1e-8,
        atol=1e-9,
    )


def test_exact_force_is_energy_gradient(x64):
    state = setup()
    basis = qtx.symmetry.Identity()
    basis.basis_make()
    spins = qtx.utils.to_distributed_array(qtx.utils.ints_to_array(basis.basis.states))
    H = [qtx.operator.Ising(h=h) for h in [0.8, 1.3]]
    conditions = [[0.8], [1.3]]
    prob = np.array([0.3, 0.7])
    batches = []
    for condition in conditions:
        psi = np.asarray(state.at(condition)(spins))
        batches.append(samples(spins, abs(psi) ** 2))
    opt = qtx.optimizer.FoundationSR(state, H, conditions, weights=prob, clip=None)
    O, E = map(np.asarray, opt.get_equation(batches))
    force = 2 * (O.conj().T @ E).real
    dense = [np.asarray(h.todense()) for h in H]
    model = state.conditional_model

    def loss(theta):
        ansatz = eqx.tree_at(lambda x: x.theta, model, jnp.asarray(theta))
        total = 0.0
        for weight, g, h in zip(prob, conditions, dense):
            psi = np.asarray(jax.vmap(lambda s: ansatz(s, jnp.asarray(g)))(spins))
            psi = psi / np.linalg.norm(psi)
            total += weight * np.vdot(psi, h @ psi).real
        return total

    theta = np.asarray(state.get_params_flatten())
    eps = 1e-5
    finite = np.array(
        [
            (loss(theta + eps * np.eye(4)[i]) - loss(theta - eps * np.eye(4)[i]))
            / (2 * eps)
            for i in range(4)
        ]
    )
    np.testing.assert_allclose(force, finite, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("mode", ["mean_std", "median_mad"])
def test_per_system_clipping_and_invalid_inputs(x64, mode):
    state = setup()
    spins = qtx.utils.rand_states(8)
    batches = [samples(spins), samples(spins)]
    opt = qtx.optimizer.FoundationSR(
        state,
        [StubHamiltonian([0, 1, 2, 3, 4, 5, 6, 1000])] * 2,
        [[0.8], [1.3]],
        clip=1.0,
        clip_mode=mode,
        C0=0.02,
        updater=qtx.optimizer.AdamUpdater(),
    )
    O, E = opt.get_equation(batches)
    for start in [0, 8]:
        np.testing.assert_allclose(
            np.asarray(E[start : start + 8]).sum(), 0, atol=1e-12
        )
    step = opt.get_step(batches)
    assert float(jnp.linalg.norm(step)) <= 0.02000001
    stream = io.BytesIO()
    opt.save(stream)
    stream.seek(0)
    other = qtx.optimizer.FoundationSR(
        state,
        [StubHamiltonian(np.arange(8))] * 2,
        [[0.8], [1.3]],
        file=stream,
        updater=qtx.optimizer.AdamUpdater(),
    )
    for a, b in zip(jax.tree.leaves(opt._buffers), jax.tree.leaves(other._buffers)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    with pytest.raises(ValueError):
        opt.get_step(batches[:1])
    with pytest.raises(ValueError):
        opt.get_step([samples(spins, jnp.zeros(8)), batches[1]])
    with pytest.raises(ValueError):
        opt.get_step([samples(spins, -jnp.ones(8)), batches[1]])
    with pytest.raises(ValueError):
        qtx.optimizer.FoundationSR(
            state, [StubHamiltonian(np.arange(8))], [[0.8]], weights=[np.nan]
        )
    bad = qtx.optimizer.FoundationSR(state, [StubHamiltonian([np.nan] * 8)], [[0.8]])
    with pytest.raises(FloatingPointError):
        bad.get_step([spins])


@pytest.mark.parametrize("complex_output", [False, True])
def test_transformer_conditioning_gradients(x64, complex_output):
    qtx.sites.Chain(4)
    qtx.set_default_dtype(jnp.complex128 if complex_output else jnp.float64)
    model = qtx.model.ConditionalTransformer(
        2, dim=8, heads=2, layers=1, dtype=jnp.float64, complex_output=complex_output
    )
    state = qtx.state.Foundation(model, [0.8, 0.2])
    spins = qtx.utils.rand_states(8)
    params = sum(x.size for x in jax.tree.leaves(model) if eqx.is_inexact_array(x))
    assert state.nparams == params
    psi = np.asarray(state.at([0.8, 0.2])(spins))
    other = np.asarray(state.at([1.2, 0.7])(spins))
    assert np.isfinite(psi).all() and not np.allclose(psi, other)
    jac = state.at([0.8, 0.2]).jacobian(spins)
    assert jac.shape == (8, params) and np.isfinite(np.asarray(jac)).all()
    assert np.isfinite(
        np.asarray(
            jax.jacfwd(lambda g: model.log_psi(spins[0], g))(jnp.array([0.8, 0.2]))
        )
    ).all()
    with pytest.raises(ValueError):
        state.at([0.8])
    with pytest.raises(ValueError):
        state.at([np.nan, 0.2])


def test_transformer_validation():
    qtx.sites.Chain(4)
    for kwargs in [
        dict(ncouplings=0),
        dict(ncouplings=1, patch_size=3),
        dict(ncouplings=1, dim=7, heads=2),
        dict(ncouplings=1, layers=-1),
        dict(ncouplings=1, dtype=jnp.complex64),
    ]:
        with pytest.raises(ValueError):
            qtx.model.ConditionalTransformer(**kwargs)


@pytest.mark.parametrize("complex_output", [False, True])
def test_coupling_metric_ignores_conditional_gauge(x64, complex_output):
    state = setup(complex_output)
    spins = jnp.array([[1, 1, -1, -1], [1, -1, 1, -1], [-1, 1, -1, 1], [-1, -1, 1, 1]])
    weight = np.array([1.0, 2.0, 3.0, 4.0])
    weight /= weight.sum()
    mean = (np.asarray(spins[:, 1]) * weight).sum()
    expected = 0.2**2 * ((np.asarray(spins[:, 1]) - mean) ** 2 * weight).sum()
    metric = state.coupling_qgt([0.8], spins, weights=weight)
    np.testing.assert_allclose(np.asarray(metric), [[expected]], atol=1e-12)


def test_fermion_samplers_follow_shared_updates(x64):
    qtx.set_default_dtype(jnp.complex128)
    qtx.sites.Chain(4, particle_type="spinful_fermion", Nparticles=(2, 2))
    model = qtx.model.ConditionalTransformer(
        1, dim=4, heads=1, layers=0, complex_output=True, dtype=jnp.float64
    )
    state = qtx.state.Foundation(model, [0.4])
    opt = qtx.optimizer.FoundationSR(
        state,
        [qtx.operator.Hubbard(U=4), qtx.operator.Hubbard(U=8)],
        [[0.4], [0.8]],
        C0=0.02,
    )
    samplers = [
        qtx.sampler.ParticleHop(view, nsamples=8, thermal_steps=0, sweep_steps=4)
        for view in opt.states
    ]
    batches = [sampler.sweep() for sampler in samplers]
    state.update(opt.get_step(batches))
    for sampler, view in zip(samplers, opt.states):
        batch = sampler.sweep()
        assert sampler.state is view
        np.testing.assert_array_equal(
            np.asarray(view.get_params_flatten()),
            np.asarray(state.get_params_flatten()),
        )
        assert np.isfinite(np.asarray(view(batch.spins))).all()
        for sector in np.split(np.asarray(batch.spins) > 0, 2, axis=1):
            np.testing.assert_array_equal(sector.sum(axis=1), 2)


def test_coupling_metric_uses_projected_wavefunction(x64):
    qtx.sites.Chain(4)
    qtx.set_default_dtype(jnp.float64)
    state = qtx.state.Foundation(
        Toy(jnp.array([0.1, -0.2, 0.3, 0.05])), [0.8], symm=qtx.symmetry.Z2Inversion()
    )
    spins = qtx.utils.ints_to_array(np.arange(16, dtype=np.uint64))

    def normalized(h):
        psi = np.asarray(state.at([h])(spins))
        return psi / np.linalg.norm(psi)

    psi = normalized(0.8)
    eps = 1e-5
    tangent = (normalized(0.8 + eps) - normalized(0.8 - eps)) / (2 * eps)
    expected = np.vdot(tangent, tangent) - abs(np.vdot(psi, tangent)) ** 2
    metric = state.coupling_qgt([0.8], spins, weights=abs(psi) ** 2)
    np.testing.assert_allclose(np.asarray(metric), [[expected]], atol=1e-9, rtol=1e-7)
