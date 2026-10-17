from typing import Optional
from jax.typing import ArrayLike
from warnings import warn
import jax
import jax.numpy as jnp

from ..state import State, Variational, DenseState
from ..sampler import Samples
from ..operator import Operator
from ..symmetry import Symmetry
from ..utils import array_extend, to_distributed_array, where, isfinite
from ..global_defs import get_default_dtype


def _median_mad_residual(energies, weight, clip, scale_floor):
    """Weighted median/MAD clipping with clipped-mean recentering.

    MAD is the unscaled median absolute deviation (no Gaussian 1.4826 factor).
    Zero-weight entries, including invalid samples, do not affect statistics.
    Complex components are clipped separately. The floor is in energy units.
    """

    def weighted_median(values):
        order = jnp.argsort(values)
        values, weights = values[order], weight[order]
        index = jnp.searchsorted(jnp.cumsum(weights), jnp.sum(weights) / 2, side="left")
        return values[jnp.minimum(index, values.size - 1)]

    def component(values):
        center = weighted_median(values)
        scale = jnp.maximum(weighted_median(jnp.abs(values - center)), scale_floor)
        residual = jnp.clip(values - center, -clip * scale, clip * scale)
        residual -= jnp.sum(residual * weight) / jnp.sum(weight)
        return jnp.where(weight > 0, residual, 0)

    if jnp.iscomplexobj(energies):
        return component(energies.real) + 1j * component(energies.imag)
    return component(energies)


class EnergyGrad:
    r"""
    Energy gradient source defining :math:`\bar \epsilon` from the local energies
    of a Hamiltonian, used for ground-state search and time evolution.
    """

    def __init__(
        self,
        hamiltonian: Operator,
        clip: Optional[float] = 5.0,
        *,
        clip_mode: str = "mean_std",
        clip_scale_floor: float = 1e-8,
    ):
        r"""
        :param hamiltonian:
            The Hamiltonian for the evolution.

        :param clip_mode:
            "mean_std" preserves the original clipping. "median_mad" clips
            around the weighted median at clip times the unscaled weighted median
            absolute deviation, then subtracts the weighted clipped mean.
        :param clip_scale_floor:
            Positive absolute floor on the median deviation, in energy units.

        :param clip:
            Clipping the local energies to :math:`\bar E \pm c \sigma` to avoid divergence,
            default to 5.0. Clipping is disabled if set to None.

            The clipping only enters :math:`\bar\epsilon` (the gradient); the reported
            `~quantax.optimizer.EnergyGrad.energy` and `~quantax.optimizer.EnergyGrad.VarE`
            remain the unclipped, unbiased estimators. For complex local energies, the
            real and imaginary parts are clipped independently with the shared scale
            :math:`\sigma = \sqrt{\mathrm{VarE}}`, as they enter the stacked SR
            equations as two equivalent real samples.

            Clipping is a biased operation, recommended for ground-state search but
            not for real-time evolution, where the bias is a systematic error in the
            dynamics instead of a stabilizer. `~quantax.optimizer.TimeEvol` disables
            it; pass ``clip=None`` when using `~quantax.optimizer.SR` with
            ``imag_time=False`` directly.
        """
        if clip_mode not in ("mean_std", "median_mad"):
            raise ValueError("clip_mode must be mean_std or median_mad")
        import math

        if not math.isfinite(clip_scale_floor) or clip_scale_floor <= 0:
            raise ValueError("clip_scale_floor must be finite and positive")
        if clip is not None and (not math.isfinite(clip) or clip <= 0):
            raise ValueError("clip must be finite and positive, or None")
        self._clip_mode = clip_mode
        self._clip_scale_floor = clip_scale_floor
        self._hamiltonian = hamiltonian
        self._energy = None
        self._VarE = None
        self._clip = clip

    @property
    def hamiltonian(self) -> Operator:
        """The Hamiltonian for the evolution."""
        return self._hamiltonian

    @property
    def energy(self) -> ArrayLike | None:
        """Energy of the current step."""
        return self._energy

    @property
    def VarE(self) -> ArrayLike | None:
        r"""Energy variance :math:`\left< (H - E)^2 \right>` of the current step."""
        return self._VarE

    def ebar(self, state: Variational, samples: Samples | jax.Array) -> jax.Array:
        r"""
        Compute :math:`\bar \epsilon` for given samples. The local energy is
        :math:`E_{loc, s} = \sum_{s'} \frac{\psi_{s'}}{\psi_s} \left< s|H|s' \right>`,
        and :math:`\bar \epsilon` is defined as
        :math:`\bar \epsilon = \frac{1}{\sqrt{N_s}} (E_{loc, s} - \left<E_{loc, s}\right>)`.

        Samples with non-finite local energy (e.g. from an overflowed or NaN
        wave function) are excluded from all estimators and contribute zero to
        :math:`\bar \epsilon`, so a single corrupted sample can't poison the
        mean energy or the clipping threshold. A warning is emitted when such
        samples are detected.
        """
        if not isinstance(samples, Samples):
            samples = Samples(to_distributed_array(samples))

        if samples.reweight_factor is None:
            reweight_factor = 1
        else:
            reweight_factor = samples.reweight_factor

        Eloc = self._hamiltonian.Oloc(state, samples).astype(get_default_dtype())
        valid = jnp.isfinite(Eloc) & jnp.isfinite(reweight_factor)
        n_invalid = Eloc.size - jnp.count_nonzero(valid)
        if jax.process_index() == 0 and n_invalid > 0:
            warn(f"{n_invalid} sample(s) with non-finite local energy excluded.")

        real_dtype = jnp.finfo(get_default_dtype()).dtype
        weight = jnp.where(valid, reweight_factor, 0).astype(real_dtype)
        norm = jnp.mean(weight)
        Emean = jnp.mean(jnp.where(valid, Eloc, 0) * weight) / norm
        self._energy = Emean.real
        # invalid samples are pinned at Emean so they pass the clip with d = 0
        Eloc = jnp.where(valid, Eloc, Emean)
        Evar = jnp.abs(Eloc - Emean) ** 2
        self._VarE = (jnp.mean(Evar * weight) / norm).real

        if self._clip is not None and self._clip_mode == "median_mad":
            residual = _median_mad_residual(
                Eloc, weight, self._clip, self._clip_scale_floor
            )
            return residual * jnp.sqrt(weight / samples.nsamples)

        if self._clip is not None:
            sigma = jnp.sqrt(self._VarE)
            d = Eloc - Emean
            if jnp.iscomplexobj(Eloc):
                d_real = jnp.clip(d.real, -self._clip * sigma, self._clip * sigma)
                d_imag = jnp.clip(d.imag, -self._clip * sigma, self._clip * sigma)
                d = d_real + 1j * d_imag
            else:
                d = jnp.clip(d, -self._clip * sigma, self._clip * sigma)
            Eloc = Emean + d

        Eloc -= jnp.mean(Eloc)
        Eloc *= jnp.sqrt(weight / samples.nsamples)
        return Eloc

    def ebar_dense(self, psi: jax.Array, symm: Symmetry, Ns: int) -> jax.Array:
        r"""
        Compute :math:`\bar \epsilon = (H - E) \psi` for the normalized wave
        function ``psi`` given in the full Hilbert space of ``symm``, whose first
        ``Ns`` entries are the physical amplitudes.
        """
        dense = DenseState(psi[:Ns], symm)
        H_psi = self._hamiltonian @ dense
        energy = dense @ H_psi
        self._energy = jnp.asarray(energy.real)
        Ebar = H_psi - dense * energy
        Ebar = jnp.asarray(Ebar.psi)
        return array_extend(Ebar, psi.size)


class OverlapGrad:
    r"""
    Overlap gradient source defining :math:`\bar \epsilon` from the amplitude
    ratios with a target state, used for supervised wave function optimization.
    """

    def __init__(self, target_state: State, clip: Optional[float] = None):
        r"""
        :param target_state:
            The target state to be approximated.

        :param clip:
            The clipping value of the centered amplitude ratios, default to no
            clipping.
        """
        self._target_state = target_state
        self._clip = clip
        self._target_psi = None

    @property
    def target_state(self) -> State:
        """The target state to be approximated."""
        return self._target_state

    def ebar(self, state: Variational, samples: Samples | jax.Array) -> jax.Array:
        r"""
        Compute :math:`\bar \epsilon` from the normalized amplitude ratios
        :math:`\phi_s / \psi_s` for given samples.

        Samples with non-finite amplitude ratio (e.g. from an overflowed or NaN
        wave function) are excluded from the mean ratio and contribute zero to
        :math:`\bar \epsilon`, so a single corrupted sample can't poison the
        whole gradient. A warning is emitted when such samples are detected.
        """
        if not isinstance(samples, Samples):
            samples = Samples(to_distributed_array(samples))

        if samples.psi is None:
            psi = state(samples.spins)
        else:
            psi = samples.psi
        if samples.reweight_factor is None:
            reweight = 1
        else:
            reweight = samples.reweight_factor

        phi = self._target_state(samples.spins)
        ratio = phi / psi
        valid = isfinite(ratio) & jnp.isfinite(reweight)
        n_invalid = valid.size - jnp.count_nonzero(valid)
        if jax.process_index() == 0 and n_invalid > 0:
            warn(f"{n_invalid} sample(s) with non-finite amplitude ratio excluded.")

        real_dtype = jnp.finfo(get_default_dtype()).dtype
        weight = jnp.where(valid, reweight, 0).astype(real_dtype)
        norm = jnp.mean(weight)
        ratio = where(valid, ratio, 0)
        ratio_mean = (ratio * weight).mean() / norm
        ratio = jnp.asarray(ratio / ratio_mean) - 1
        ratio = jnp.where(valid, ratio, 0)
        if self._clip is not None:
            if jnp.iscomplexobj(ratio):
                ratio_real = jnp.clip(ratio.real, -self._clip, self._clip)
                ratio_imag = jnp.clip(ratio.imag, -self._clip, self._clip)
                ratio = ratio_real + 1j * ratio_imag
            else:
                ratio = jnp.clip(ratio, -self._clip, self._clip)
        Ebar = -ratio * jnp.sqrt(weight / samples.nsamples)
        return Ebar

    def ebar_dense(
        self, psi: jax.Array, symm: Symmetry, Ns: int | None = None
    ) -> jax.Array:
        r"""
        Compute :math:`\bar \epsilon = \psi - \phi / \left<\psi|\phi\right>` for
        the normalized wave function ``psi`` given in the full Hilbert space of
        ``symm``, whose first ``Ns`` entries are the physical amplitudes.
        """
        if self._target_psi is None:
            self._target_psi = jnp.asarray(self._target_state.todense(symm).psi)
        if Ns is None:
            Ns = psi.size

        target_psi = self._target_psi
        Ebar = psi[:Ns] - target_psi / jnp.vdot(psi[:Ns], target_psi)
        return array_extend(Ebar, psi.size)
