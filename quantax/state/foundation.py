"""One shared parameter set with dynamic batched coupling inputs."""

from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO, Callable
import jax
import equinox as eqx
import jax.numpy as jnp
from ..utils import (
    LogArray,
    PsiArray,
    jit_chunk_vmap,
    chunk_map,
    shmap,
    get_distributed_sharding,
    to_distributed_array,
    array_extend,
    filter_tree_map,
)
from .variational import Variational
from ..model.foundation import Conditioned, coupling_tuple


class _FoundationView(Variational):
    def update(self, step):
        raise RuntimeError(
            "Update the owning Foundation state, not an individual view; use detach() to fine-tune"
        )


class Foundation(Variational):
    """A fixed-Hilbert-space family psi(spins | couplings) with shared weights.

    ``at(couplings)`` returns a cached read-only Variational view compatible with
    existing operators and samplers. One ``foundation.update(step)`` updates every
    live view, including held-out couplings. Each view can own an independent
    persistent sampler. Couplings are frozen and excluded from nparams/Jacobians.

    The supplied model accepts two positional arrays: spins and couplings. All
    states share one geometry, particle sector, parameter layout and symmetry.
    ``evaluate`` and ``jacobian_at`` accept dynamic coupling batches and share
    compiled kernels across grid values. ``at`` remains a static compatibility
    interface. Varying system sizes or coupling dimensions require a new state.
    """

    def __init__(
        self,
        model: Callable,
        couplings: Sequence[float],
        param_file: str | Path | BinaryIO | None = None,
        **kwargs,
    ):
        self._views = {}
        self._view_options = dict(kwargs)
        super().__init__(Conditioned(model, couplings), param_file=param_file, **kwargs)
        self._coupling_size = len(coupling_tuple(couplings))
        self._init_context_kernels()

    @property
    def conditional_model(self):
        """The unbound two-input network, suitable for evaluating new couplings."""
        assert isinstance(self.model, Conditioned)
        return self.model.network

    def at(self, couplings: Sequence[float]) -> Variational:
        key = coupling_tuple(couplings)
        if len(key) != self._coupling_size:
            raise ValueError("Coupling dimension cannot change within a Foundation")
        if key not in self._views:
            self._views[key] = _FoundationView(
                Conditioned(self.conditional_model, key), **self._view_options
            )
        return self._views[key]

    def update(self, step):
        super().update(step)
        for key, view in self._views.items():
            # Parameter layout and static structure are unchanged, so compiled
            # forward/Jacobian functions and sampler references remain valid.
            view._model = Conditioned(self.conditional_model, key)

    def detach(self, couplings: Sequence[float]) -> Variational:
        """Make an independent single-Hamiltonian state for ordinary fine-tuning."""
        key = coupling_tuple(couplings)
        if len(key) != self._coupling_size:
            raise ValueError("Coupling dimension cannot change within a Foundation")
        return Variational(
            Conditioned(self.conditional_model, key), **self._view_options
        )

    def coupling_qgt(self, couplings: Sequence[float], spins, weights=None):
        """Estimate the coupling-space QGT from current-state configurations.

        The real part is the fidelity metric. Couplings are differentiated as
        continuous inputs of the *unbound* network, with model weights fixed.
        No finite differences or new optimization are used. Pass Born weights
        for exact enumeration; otherwise spins must be drawn from the current
        state at these couplings. This diagnoses the learned family, so poor
        generalization can yield a misleading physical susceptibility.
        """
        key = coupling_tuple(couplings)
        if len(key) != self._coupling_size:
            raise ValueError("Coupling dimension cannot change within a Foundation")
        context = jnp.asarray(key)

        def parts(spin, gamma):
            transformed = self.symm.get_symm_spins(spin)
            amplitudes = jax.vmap(lambda s: self.conditional_model(s, gamma))(
                transformed
            )
            psi = self.symm.symmetrize(amplitudes, spin)
            if isinstance(psi, LogArray):
                return psi.sign, psi.logabs
            magnitude = jnp.abs(psi)
            return psi / magnitude, jnp.log(magnitude)

        def derivative(spin):
            sign, _ = parts(spin, context)
            dsign, dlogabs = jax.jacfwd(lambda gamma: parts(spin, gamma))(context)
            return dlogabs + jnp.conj(sign) * dsign

        O = jax.vmap(derivative)(jnp.asarray(spins))
        if O.ndim != 2 or O.shape[0] == 0 or not bool(jnp.all(jnp.isfinite(O))):
            raise ValueError(
                "Expected nonempty configurations with finite coupling derivatives"
            )
        w = jnp.ones(O.shape[0]) if weights is None else jnp.asarray(weights)
        if (
            jnp.iscomplexobj(w)
            or w.shape != (O.shape[0],)
            or not bool(jnp.all(jnp.isfinite(w) & (w >= 0)))
            or not bool(jnp.isfinite(w.sum()))
            or not float(w.sum()) > 0
        ):
            raise ValueError(
                "Weights must be finite, nonnegative and have positive sum"
            )
        w = w / w.sum()
        centered = O - jnp.sum(w[:, None] * O, axis=0)
        return centered.conj().T @ (w[:, None] * centered)

    @property
    def coupling_size(self) -> int:
        """Number of dynamic physical conditioning inputs."""
        return self._coupling_size

    def _init_context_kernels(self):
        # Mirror Variational's direct/native-jitted split: connected-state batch
        # lengths vary after compaction, so direct evaluation dispatches only
        # fixed-sized compiled tiles rather than specializing a whole scan per size.
        batch = eqx.filter_jit(
            shmap(
                eqx.filter_vmap(self._forward_one, in_axes=(None, 0, 0)),
                in_axes=(None, 0, 0),
                out_axes=0,
            )
        )

        def forward_batch(model, spins, contexts):
            sharding = get_distributed_sharding()
            return batch(
                model,
                jax.device_put(spins, sharding),
                jax.device_put(contexts, sharding),
            )

        self._context_direct_forward = chunk_map(
            forward_batch,
            in_axes=(None, 0, 0),
            chunk_size=self.forward_chunk,
            atleast_1chunk=True,
        )
        self._context_forward = jit_chunk_vmap(
            self._forward_one,
            in_axes=(None, 0, 0),
            chunk_size=self.forward_chunk,
            shard_batch=True,
        )
        self._context_jacobian = jit_chunk_vmap(
            self._log_derivative_one,
            in_axes=(None, 0, 0),
            chunk_size=self.backward_chunk,
            shard_batch=True,
        )

    def _context_inputs(self, spins, couplings):
        spins, context = jnp.asarray(spins), jnp.asarray(couplings)
        if spins.shape[-1:] != (self.Nmodes,) or context.shape[-1:] != (
            self.coupling_size,
        ):
            raise ValueError(
                "Configuration or coupling dimension does not match the family"
            )
        shape = spins.shape[:-1]
        if context.ndim > spins.ndim:
            raise ValueError("Too many coupling batch dimensions")
        context = context.reshape(
            *context.shape[:-1],
            *([1] * (spins.ndim - context.ndim)),
            self.coupling_size,
        )
        context = jnp.broadcast_to(context, (*shape, self.coupling_size))
        spins, context = spins.reshape(-1, self.Nmodes), context.reshape(
            -1, self.coupling_size
        )
        if not spins.shape[0]:
            raise ValueError("Configuration batches must be nonempty")
        return shape, spins, context

    def evaluate(self, spins, couplings) -> PsiArray:
        """Evaluate all configurations with dynamic contexts in one batched kernel.

        ``spins`` may have shape (R,S,Nmodes), and ``couplings`` (R,C), or
        one context per configuration. Coupling values do not enter compilation
        cache keys. Outputs retain the leading configuration axes.
        """
        shape, spins, context = self._context_inputs(spins, couplings)
        count = spins.shape[0]
        spins = to_distributed_array(array_extend(spins, jax.device_count()))
        context = to_distributed_array(array_extend(context, jax.device_count()))
        out = self._context_direct_forward(self.conditional_model, spins, context)
        return filter_tree_map(lambda x: x[:count].reshape(shape), out)

    def jacobian_at(self, spins, couplings) -> jax.Array:
        """Batched log derivatives at dynamic contexts, shape (..., Nparameters)."""
        shape, spins, context = self._context_inputs(spins, couplings)
        count = spins.shape[0]
        spins = to_distributed_array(array_extend(spins, jax.device_count()))
        context = to_distributed_array(array_extend(context, jax.device_count()))
        out = self._context_jacobian(self.conditional_model, spins, context)
        return out[:count].reshape(*shape, out.shape[-1])
