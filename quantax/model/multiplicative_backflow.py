"""Translation-equivariant multiplicative pairing, evaluated as a Pfaffian."""

from collections.abc import Callable, Sequence

import equinox as eqx
import fermix
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike

from .conv_nets import _ConvBlock
from ..global_defs import get_lattice, get_subkeys
from ..nn import Conv, Embedding, Sequential, fermion_idx, fermion_inverse_sign
from ..utils import LogArray


class MultiplicativePfBackflow(eqx.Module):
    r"""Symmetry-preserving multiplicative Pfaffian backflow.

    .. math::

        f^{\sigma\sigma'}_{ij}(n) = p_{ij}
        \sum_\alpha W^{\sigma\sigma'}_{i-j,\alpha}
        y^\sigma_{i\alpha}(n)y^{\sigma'}_{j\alpha}(n),
        \qquad \psi(n) = \operatorname{pf}[n\star f(n)\star n].

    Only independent weights are trained; their reversed partners satisfy
    :math:`W^{\sigma'\sigma}_{-r}=-W^{\sigma\sigma'}_r` identically.
    The envelope is :math:`p_{ij}=\exp(-\lambda d_{ij})/
    \sum_{kl}\exp(-\lambda d_{kl})`, normalized over *spatial* site pairs.
    ``decay`` is positive and trainable via a softplus parametrization.

    This implements Eq. (7) in the supplied Pfaffian formulation, extending
    Eq. (4) of https://arxiv.org/abs/2608.12465. It differs from ``PfBackflow``,
    whose correction is additive in the single-particle orbitals.

    Supports periodic lattices of fixed, positive, even fermion number, including
    multiple sites per unit cell. A custom ``net`` must be equivariant under cell
    translations. The :meth:`resconv` constructor supplies such a network.
    No translation projection is performed. Amplitudes include Quantax's fermion
    ordering sign and transform with the fermionic permutation sign.
    """

    net: Callable
    weight_real: jax.Array
    weight_imag: jax.Array | None
    log_decay: jax.Array
    d: int = eqx.field(static=True)
    dtype: DTypeLike = eqx.field(static=True)
    opposite_spin: bool = eqx.field(static=True)
    output_softmax_gate: bool = eqx.field(static=True)
    _nlabels: int = eqx.field(static=True)
    _nspecies: int = eqx.field(static=True)
    _distances: tuple[float, ...] = eqx.field(static=True)
    _weight_index: jax.Array
    _weight_sign: jax.Array
    _mode_pairs: jax.Array

    def __init__(
        self,
        net: Callable,
        d: int | None = None,
        decay: float = 1.0,
        opposite_spin: bool = False,
        dtype: DTypeLike = jnp.float64,
        *,
        output_softmax_gate: bool = False,
    ):
        """Initialize the pairing head.

        :param net:
            Network mapping a Quantax +/-1 configuration to channels-first
            features of shape ``(nspecies * nsites_per_cell * d, *extent)``
            (replace d by 2*d with output_softmax_gate enabled).
            Channel order is ``(spin, site_in_cell, feature)``; flattening the
            trailing spatial axes uses the original lattice ordering.
        :param d: Number of features per spin-orbital (not total CNN channels).
            Defaults to the full spin-orbital dimension, ``lattice.Nfmodes``
            (``2 * Nsites`` for spinful fermions).
        :param decay: Initial positive inverse decay length in lattice units.
        :param opposite_spin:
            Keep only opposite-spin weights, recovering the BCS determinant
            up to a configuration-independent sign in a balanced spin sector.
            This restriction alone does not enforce spin-singlet symmetry.
        :param output_softmax_gate: Split 2*d output features per spin-orbital into
            logits and signed values. Multiply softmax(logits) over d by values
            before contracting the pairing kernels. The default uses d outputs.
        :param dtype: Real or complex pairing parameter / Pfaffian precision.
        """
        lattice = get_lattice()
        if not lattice.is_fermion:
            raise ValueError("MultiplicativePfBackflow requires fermions.")
        if np.any(lattice.boundary != 1):
            raise ValueError("MultiplicativePfBackflow requires periodic boundaries.")
        n = lattice.Ntotal
        if n is None or n <= 0 or n % 2:
            raise ValueError("A fixed, positive, even fermion number is required.")
        if d is None:
            d = lattice.Nfmodes
        if not isinstance(d, int) or d <= 0:
            raise ValueError("d must be a positive integer.")
        if not np.isfinite(decay) or decay <= 0:
            raise ValueError("decay must be finite and positive.")
        dtype = jnp.dtype(dtype)
        if not jnp.issubdtype(dtype, jnp.inexact):
            raise ValueError("dtype must be floating point or complex.")
        if opposite_spin and not lattice.is_spinful:
            raise ValueError("opposite_spin requires spinful fermions.")

        self.net, self.d, self.dtype = net, d, dtype
        self.opposite_spin = opposite_spin
        self.output_softmax_gate = output_softmax_gate
        ns = 2 if lattice.is_spinful else 1
        nb, nc = lattice.shape[0], lattice.ncells
        q = ns * nb
        self._nlabels, self._nspecies = q, ns
        extent = np.asarray(lattice.shape[1:])
        coords = lattice.xyz_from_index[:nc, 1:]
        reverse = np.asarray(np.ravel_multi_index(((-coords) % extent).T, extent))

        # An oriented key is (spin/sublattice a, spin/sublattice b, i-j).
        # Zero indexes a constant zero row, including self-reversing channels.
        index = np.zeros((q, q, nc), dtype=np.int32)
        sign = np.zeros_like(index)
        nweights = 0
        for a in range(q):
            for b in range(q):
                for r in range(nc):
                    partner = (b, a, int(reverse[r]))
                    if (a, b, r) >= partner:
                        continue
                    if opposite_spin and a // nb == b // nb:
                        continue
                    nweights += 1
                    index[a, b, r] = index[partner] = nweights
                    sign[a, b, r], sign[partner] = 1, -1
        self._weight_index = jnp.asarray(index.ravel())
        self._weight_sign = jnp.asarray(sign.ravel())
        weights = jax.random.normal(get_subkeys(), (nweights, d), dtype=dtype) / d**0.5
        self.weight_real = weights.real
        self.weight_imag = (
            weights.imag if jnp.issubdtype(dtype, jnp.complexfloating) else None
        )
        real_dtype = jnp.empty((), dtype=dtype).real.dtype
        self.log_decay = jnp.asarray(decay + np.log(-np.expm1(-decay)), real_dtype)

        mode_sites = np.tile(np.arange(lattice.Nsites), ns)
        labels = np.arange(lattice.Nfmodes) // nc
        positions = lattice.xyz_from_index[mode_sites, 1:]
        diff = (positions[:, None] - positions[None, :]) % extent
        r = np.ravel_multi_index(np.moveaxis(diff, -1, 0), extent)
        self._mode_pairs = jnp.asarray((labels[:, None] * q + labels[None, :]) * nc + r)
        # Constants are tuples so the variational state cannot train geometry.
        distances = np.empty((q, q, nc))
        for a in range(q):
            for b in range(q):
                distances[a, b] = lattice.dist[
                    (a % nb) * nc + np.arange(nc), (b % nb) * nc
                ]
        distances = (distances + distances.swapaxes(0, 1)[..., reverse]) / 2
        self._distances = tuple(float(v) for v in distances.ravel())

    @classmethod
    def resconv(
        cls,
        d: int | None = None,
        nblocks: int = 4,
        channels: int = 64,
        kernel_size: int | Sequence[int] = 3,
        use_rmsnorm: bool = False,
        decay: float = 1.0,
        opposite_spin: bool = False,
        dtype: DTypeLike = jnp.float32,
        pairing_dtype: DTypeLike = jnp.float64,
        *,
        output_softmax_gate: bool = False,
    ) -> "MultiplicativePfBackflow":
        """Build a complete residual-CNN Pfaffian wavefunction.

        ``dtype`` controls the real CNN and ``pairing_dtype`` the pairing weights
        and Pfaffian. Omitting ``d`` uses the full spin-orbital dimension
        (``2 * Nsites`` for spinful fermions), independently of CNN ``channels``.
        Complex pairing weights are stored as separate real and
        imaginary arrays, allowing Quantax to train them with real CNN features.
        Enable JAX x64 before using float64/complex128.
        Other arguments have the same meaning as in ``ResConv`` / ``__init__``.
        The final 1x1 convolution emits the up/down feature halves without a
        spatial sum or symmetry projection. With ``output_softmax_gate=True``,
        it emits 2*d channels per spin-orbital, split into alpha logits and
        signed values before the pairing contraction.
        """
        if nblocks < 0 or channels <= 0:
            raise ValueError("nblocks must be nonnegative and channels positive.")
        if not jnp.issubdtype(dtype, jnp.floating):
            raise ValueError("The residual CNN requires a real floating dtype.")
        lattice = get_lattice()
        if d is None:
            d = lattice.Nfmodes
        if not isinstance(d, int) or d <= 0:
            raise ValueError("d must be a positive integer.")
        ns = 2 if lattice.is_spinful else 1
        layers: list[Callable] = [
            Embedding(channels, dtype=dtype),
            lattice.to_neighbor_repr,
        ]
        layers.extend(
            _ConvBlock(i, channels, kernel_size, use_rmsnorm, True, dtype)
            for i in range(nblocks)
        )
        layers.extend(
            [
                lambda x: x / (nblocks + 1) ** 0.5,
                Conv(
                    channels,
                    ns * lattice.shape[0] * d * (2 if output_softmax_gate else 1),
                    1,
                    dtype=dtype,
                    key=get_subkeys(),
                ),
                lattice.to_original_repr,
            ]
        )
        return cls(
            Sequential(layers),
            d,
            decay,
            opposite_spin,
            pairing_dtype,
            output_softmax_gate=output_softmax_gate,
        )

    @property
    def weights(self) -> jax.Array:
        """Independent pairing weights; complex weights use two real parameters."""
        if self.weight_imag is None:
            return self.weight_real
        return self.weight_real + 1j * self.weight_imag

    @property
    def decay(self) -> jax.Array:
        """Positive inverse decay length."""
        return jax.nn.softplus(self.log_decay)

    @property
    def pairing_weights(self) -> jax.Array:
        """Constrained W, shape ``(nlabels, nlabels, ncells, d)``.

        Labels combine spin and site-in-cell; the displacement is flattened in
        the original cell ordering. Reversal uses transpose, not conjugation.
        """
        table = jnp.concatenate([jnp.zeros((1, self.d), self.dtype), self.weights])
        w = table[self._weight_index] * self._weight_sign[:, None]
        return w.reshape(self._nlabels, self._nlabels, -1, self.d)

    @property
    def envelope(self) -> jax.Array:
        """Spatial-pair-normalized envelope, in the same order as W without d."""
        distances = jnp.asarray(self._distances, dtype=self.log_decay.dtype)
        p = jax.nn.softmax(-self.decay * distances)
        # Each relative pair occurs ncells times; spin copies are not part of
        # the spatial normalization in Eq. (4).
        p *= self._nspecies**2 / get_lattice().ncells
        return p.reshape(self._nlabels, self._nlabels, -1)

    def pairing_matrix(self, s: jax.Array, occupied_only: bool = True) -> jax.Array:
        """Return the occupied pairing matrix, or full f when requested.

        Only the selected pairs are contracted; the default avoids constructing
        a full ``Nfmodes x Nfmodes x d`` tensor.
        """
        lattice = get_lattice()
        y = jnp.asarray(self.net(s)).astype(self.dtype)
        nfeatures = self.d * (2 if self.output_softmax_gate else 1)
        y = y.reshape(self._nlabels, nfeatures, lattice.ncells)
        y = y.transpose(0, 2, 1).reshape(lattice.Nfmodes, nfeatures)
        if self.output_softmax_gate:
            logits, values = jnp.split(y, 2, axis=-1)
            y = jax.nn.softmax(logits, axis=-1) * values
        idx = (
            jnp.asarray(fermion_idx(s))
            if occupied_only
            else jnp.arange(lattice.Nfmodes)
        )
        pair = self._mode_pairs[idx[:, None], idx[None, :]]
        table = jnp.concatenate([jnp.zeros((1, self.d), self.dtype), self.weights])
        w = table[self._weight_index[pair]] * self._weight_sign[pair, None]
        f = jnp.einsum("ijc,ic,jc->ij", w, y[idx], y[idx])
        f *= self.envelope.ravel()[pair]
        # Remove floating-point contraction asymmetry; no conjugation.
        upper = jnp.triu(f, k=1)
        return upper - upper.T

    def __call__(self, s: jax.Array) -> LogArray:
        """Evaluate a stable log-Pfaffian in Quantax's fermion convention."""
        sign, logabs = fermix.slogpf(self.pairing_matrix(s))
        return LogArray(sign, logabs) * fermion_inverse_sign(s)
