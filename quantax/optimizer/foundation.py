"""Ensemble stochastic reconfiguration with one shared update."""

from collections.abc import Sequence, Callable
import math
from pathlib import Path
from typing import BinaryIO
import numpy as np
import jax
import jax.numpy as jnp
from .qngd import QNGD
from .gradient import EnergyGrad, _median_mad_residual
from .updater import Updater
from .sr import _constrain_norm
from ..state import Foundation, Variational
from ..operator import Operator
from ..sampler import Samples
from ..utils import to_distributed_array


class FoundationSR(QNGD):
    r"""Joint SR for a weighted family of Hamiltonians on the same Hilbert space.

    For each Hamiltonian k, center O and E *within that system*, scale its rows
    by sqrt(p_k w_s / sum_s w_s), concatenate all systems, and solve once using
    Quantax's SR/MinSR solver. Thus S = sum_k p_k Cov_k(O,O) and
    F = sum_k p_k Cov_k(O,E). This is not an average of separate SR steps and
    does not include the spurious between-Hamiltonian normalization covariance.

    A sample batch is required per Hamiltonian, in the declared order. Its
    length need not equal the other lengths; system weights remain explicit.
    reweight_factor supports exact Born-weighted enumeration for small tests.
    Nonfinite energies/Jacobians or invalid weights raise instead of silently
    changing the ensemble objective. Clipping affects forces only, per system.
    ``C0`` optionally applies a constant bound to the final shared update.
    ``updater=AdamUpdater(...)`` enables shared Adam moments; save() restores
    its buffers through file= as in QNGD. Apply returned steps to Foundation.
    """

    def __init__(
        self,
        state: Foundation,
        hamiltonians: Sequence[Operator],
        couplings: Sequence[Sequence[float]],
        *,
        weights=None,
        solver: Callable | None = None,
        updater: Updater | None = None,
        file: str | Path | BinaryIO | None = None,
        clip: float | None = 5.0,
        clip_mode: str = "mean_std",
        clip_scale_floor: float = 1e-8,
        C0: float | None = None,
    ):
        if not isinstance(state, Foundation):
            raise TypeError("FoundationSR requires a Foundation state")
        hamiltonians, couplings = tuple(hamiltonians), tuple(couplings)
        if not hamiltonians or len(hamiltonians) != len(couplings):
            raise ValueError("Provide one coupling vector per Hamiltonian")
        if weights is None:
            weights = np.ones(len(hamiltonians))
        weights = np.asarray(weights, dtype=float)
        if (
            weights.shape != (len(hamiltonians),)
            or not np.isfinite(weights).all()
            or np.any(weights <= 0)
        ):
            raise ValueError("System weights must be a finite positive vector")
        if not np.isfinite(weights.sum()):
            raise ValueError("System weights must have a finite sum")
        if C0 is not None and (not math.isfinite(C0) or C0 <= 0):
            raise ValueError("C0 must be finite and positive")
        # Reuse the existing public clipping argument validation.
        grad = EnergyGrad(
            hamiltonians[0],
            clip,
            clip_mode=clip_mode,
            clip_scale_floor=clip_scale_floor,
        )
        super().__init__(state, grad, solver=solver, updater=updater, file=file)
        self._hamiltonians = hamiltonians
        self._views = tuple(state.at(c) for c in couplings)
        self._weights = jnp.asarray(weights / weights.sum())
        self._clip, self._clip_mode = clip, clip_mode
        self._clip_floor, self._C0 = clip_scale_floor, C0
        self._energies = self._variances = None

    @property
    def hamiltonian(self):
        return None

    @property
    def hamiltonians(self):
        return self._hamiltonians

    @property
    def states(self) -> tuple[Variational, ...]:
        """Persistent views in Hamiltonian order; attach one sampler to each."""
        return self._views

    @property
    def energies(self):
        return self._energies

    @property
    def variances(self):
        return self._variances

    @property
    def energy(self):
        return None if self._energies is None else self._weights @ self._energies

    @property
    def VarE(self):
        """Ensemble mean of within-system variances, excluding between-system shifts."""
        return None if self._variances is None else self._weights @ self._variances

    def get_equation(self, samples: Sequence[Samples | jax.Array]):
        """Return centered stacked (Obar, Ebar); update per-system diagnostics."""
        if len(samples) != len(self._views):
            raise ValueError("Provide a sample batch for every Hamiltonian")
        obars, ebars, energies, variances = [], [], [], []
        for index, (state, H, batch) in enumerate(
            zip(self._views, self._hamiltonians, samples)
        ):
            if not isinstance(batch, Samples):
                batch = Samples(to_distributed_array(jnp.asarray(batch)))
            if batch.nsamples == 0:
                raise ValueError("Sample batches must be nonempty")
            O = state.jacobian(batch.spins)
            E = jnp.asarray(H.Oloc(state, batch))
            if E.shape != (batch.nsamples,):
                raise ValueError("Expected one local energy per configuration")
            w = (
                jnp.ones(E.shape, dtype=E.real.dtype)
                if batch.reweight_factor is None
                else jnp.asarray(batch.reweight_factor)
            )
            if w.shape != E.shape or jnp.iscomplexobj(w):
                raise ValueError(
                    "Sample weights must be a real vector matching local energies"
                )
            if not bool(jnp.all(jnp.isfinite(E))) or not bool(jnp.all(jnp.isfinite(O))):
                raise FloatingPointError(f"Nonfinite energy/Jacobian in system {index}")
            if (
                not bool(jnp.all(jnp.isfinite(w) & (w >= 0)))
                or not bool(jnp.isfinite(w.sum()))
                or not float(w.sum()) > 0
            ):
                raise ValueError(
                    "Sample weights must be finite, nonnegative and have positive sum"
                )
            w = w / w.sum()
            Emean = jnp.sum(w * E)
            residual = E - Emean
            energies.append(Emean.real)
            variances.append(jnp.sum(w * abs(residual) ** 2).real)
            if self._clip is not None:
                if self._clip_mode == "median_mad":
                    residual = _median_mad_residual(E, w, self._clip, self._clip_floor)
                else:
                    radius = self._clip * jnp.sqrt(variances[-1])
                    if jnp.iscomplexobj(residual):
                        residual = jnp.clip(
                            residual.real, -radius, radius
                        ) + 1j * jnp.clip(residual.imag, -radius, radius)
                    else:
                        residual = jnp.clip(residual, -radius, radius)
            residual = residual - jnp.sum(w * residual)
            factor = jnp.sqrt(self._weights[index] * w)
            obars.append((O - jnp.sum(w[:, None] * O, axis=0)) * factor[:, None])
            ebars.append(residual * factor)
        self._energies = jnp.stack(energies)
        self._variances = jnp.stack(variances)
        return jnp.concatenate(obars, axis=0), jnp.concatenate(ebars, axis=0)

    def get_step(self, samples: Sequence[Samples | jax.Array]):
        Obar, Ebar = self.get_equation(samples)
        step, self._buffers = self.solve(Obar, Ebar, self._buffers)
        if self._C0 is not None:
            step = _constrain_norm(step, jnp.asarray(self._C0))
        return step
