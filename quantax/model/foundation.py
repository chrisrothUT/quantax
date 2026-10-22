"""Hamiltonian-conditioned wavefunctions for a fixed Hilbert space."""

from collections.abc import Callable, Sequence
import math
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from ..global_defs import get_sites, get_subkeys
from ..nn import exp_by_log
from ..utils import PsiArray


def coupling_tuple(couplings: Sequence[float]) -> tuple[float, ...]:
    """Validate concrete physical inputs; these are never variational parameters."""
    values = tuple(float(x) for x in couplings)
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("couplings must be a nonempty finite real vector")
    return values


class Conditioned(eqx.Module):
    """Bind ``model(spins, couplings)`` for use with ordinary Variational states.

    Couplings are static physical inputs, not trainable leaves. Each distinct
    coupling tuple can incur its own JAX compilation. Use this for modest grids,
    not as a claim of the paper's fixed-total-batch scaling to thousands of systems.
    """

    network: Callable
    couplings: tuple[float, ...] = eqx.field(static=True)

    def __init__(self, network: Callable, couplings: Sequence[float]):
        self.network = network
        self.couplings = coupling_tuple(couplings)

    @property
    def holomorphic(self):
        return getattr(self.network, "holomorphic", False)

    def __call__(self, spins: jax.Array) -> PsiArray:
        return self.network(spins, jnp.asarray(self.couplings))


class _Encoder(eqx.Module):
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    dense1: eqx.nn.Linear
    dense2: eqx.nn.Linear

    def __init__(self, dim, heads, dtype):
        self.norm1 = eqx.nn.LayerNorm(dim, dtype=dtype)
        self.norm2 = eqx.nn.LayerNorm(dim, dtype=dtype)
        self.attention = eqx.nn.MultiheadAttention(
            heads, dim, dtype=dtype, key=get_subkeys()
        )
        self.dense1 = eqx.nn.Linear(dim, 4 * dim, dtype=dtype, key=get_subkeys())
        self.dense2 = eqx.nn.Linear(4 * dim, dim, dtype=dtype, key=get_subkeys())

    def __call__(self, x):
        norm = jax.vmap(self.norm1)(x)
        x = x + self.attention(norm, norm, norm)
        norm = jax.vmap(self.norm2)(x)
        return x + jax.vmap(self.dense2)(jax.nn.gelu(jax.vmap(self.dense1)(norm)))


class ConditionalTransformer(eqx.Module):
    """Patched Transformer conditioned on a small vector of global couplings.

    Couplings are concatenated to each configuration patch before embedding,
    following the global-coupling construction of Rende et al. (2025),
    https://doi.org/10.1038/s41467-025-62098-x. This initial implementation uses
    learned absolute positions rather than their translation-invariant attention.
    It does not import their weights or reproduce their trained architecture.

    It returns a scalar LogArray; complex_output uses real parameters for both
    log magnitude and phase. The geometry and Hilbert space are fixed. For
    spinful fermions, patches follow Quantax's up-then-down mode ordering.
    Scale physical inputs explicitly (e.g. U/t divided by 10) in the caller.
    """

    embedding: eqx.nn.Linear
    position: jax.Array
    encoders: tuple[_Encoder, ...]
    norm: eqx.nn.LayerNorm
    readout: eqx.nn.Linear
    ncouplings: int = eqx.field(static=True)
    nmodes: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    complex_output: bool = eqx.field(static=True)
    coupling_scale: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self,
        ncouplings: int,
        *,
        dim: int = 32,
        heads: int = 4,
        layers: int = 2,
        patch_size: int = 2,
        complex_output: bool = False,
        dtype: DTypeLike = jnp.float32,
        coupling_scale: Sequence[float] | None = None,
    ):
        nmodes = get_sites().Nmodes
        if any(
            not isinstance(x, int) or isinstance(x, bool) or x <= 0
            for x in (ncouplings, dim, heads, patch_size)
        ):
            raise ValueError(
                "ncouplings, dim, heads and patch_size must be positive integers"
            )
        if not isinstance(layers, int) or layers < 0:
            raise ValueError("layers must be a nonnegative integer")
        if dim % heads or nmodes % patch_size:
            raise ValueError(
                "dim must divide into heads and patch_size must divide Nmodes"
            )
        if not jnp.issubdtype(dtype, jnp.floating):
            raise ValueError("Transformer parameters must be real floating arrays")
        scale = (
            (1.0,) * ncouplings
            if coupling_scale is None
            else coupling_tuple(coupling_scale)
        )
        if len(scale) != ncouplings or any(x <= 0 for x in scale):
            raise ValueError(
                "coupling_scale must contain one finite positive value per coupling"
            )
        self.coupling_scale = scale
        self.ncouplings, self.nmodes = ncouplings, nmodes
        self.patch_size, self.complex_output = patch_size, complex_output
        self.embedding = eqx.nn.Linear(
            patch_size + ncouplings, dim, dtype=dtype, key=get_subkeys()
        )
        self.position = 0.02 * jax.random.normal(
            get_subkeys(), (nmodes // patch_size, dim), dtype=dtype
        )
        self.encoders = tuple(_Encoder(dim, heads, dtype) for _ in range(layers))
        self.norm = eqx.nn.LayerNorm(dim, dtype=dtype)
        self.readout = eqx.nn.Linear(
            dim, 2 if complex_output else 1, dtype=dtype, key=get_subkeys()
        )

    def log_psi(self, spins: jax.Array, couplings: jax.Array) -> jax.Array:
        if spins.shape != (self.nmodes,) or couplings.shape != (self.ncouplings,):
            raise ValueError(
                "Expected one configuration and the declared coupling vector"
            )
        dtype = self.embedding.weight.dtype
        patches = spins.astype(dtype).reshape(-1, self.patch_size)
        context = jnp.broadcast_to(
            couplings.astype(dtype) / jnp.asarray(self.coupling_scale, dtype=dtype),
            (len(patches), self.ncouplings),
        )
        x = (
            jax.vmap(self.embedding)(jnp.concatenate([patches, context], axis=1))
            + self.position
        )
        for encoder in self.encoders:
            x = encoder(x)
        z = jax.vmap(self.norm)(x).sum(axis=0) / jnp.sqrt(len(patches))
        out = self.readout(z)
        return out[0] + 1j * out[1] if self.complex_output else out[0]

    def __call__(self, spins: jax.Array, couplings: jax.Array) -> PsiArray:
        return exp_by_log(self.log_psi(spins, couplings))
