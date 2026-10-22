"""One trainable parameter set with persistent fixed-Hamiltonian state views."""

from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO, Callable
import jax
import jax.numpy as jnp
from ..utils import LogArray
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
    Arbitrary changing/disordered coupling fields and varying sizes are not yet
    supported efficiently: new coupling tuples get separate compiled views.
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
