"""Gated Pfaffian times an independently gated, translation-summed symmetric factor."""

import equinox as eqx
import jax
import jax.numpy as jnp
import quantax as qtx


class ConditionalPfaffian(eqx.Module):
    """Psi(s,c) = exp(sum_{alpha,r} g_sym(c)[alpha] z_sym(s,c)[alpha,r]) Pf(s,c).

    Both components share the context-conditioned CNN backbone. Their output
    projections and affine-to-softmax gates are independent, with the same d
    alpha indices. The positive symmetric factor is computed in log space.
    Spatial sites enumerate the full translation group on the one-site torus.
    """

    head: qtx.model.MultiplicativePfBackflow
    context_projection: eqx.nn.Linear
    gate_projection: eqx.nn.Linear
    symmetric_projection: qtx.nn.Conv
    symmetric_gate_projection: eqx.nn.Linear

    def __init__(self, channels=48, d=192, nblocks=8):
        head = qtx.model.MultiplicativePfBackflow.resconv(
            d=d,
            channels=channels,
            nblocks=nblocks,
            kernel_size=3,
            dtype=jnp.float32,
            pairing_dtype=jnp.complex128,
            output_softmax_gate=False,
        )
        self.head = head
        self.context_projection = eqx.nn.Linear(
            2, channels, use_bias=False, dtype=jnp.float32, key=qtx.get_subkeys()
        )
        self.gate_projection = eqx.nn.Linear(
            2, d, use_bias=True, dtype=jnp.float32, key=qtx.get_subkeys()
        )
        projection = qtx.nn.Conv(
            channels, d, 1, use_bias=False, dtype=jnp.float32, key=qtx.get_subkeys()
        )
        # The old ansatz is an exact initial limit; gradients of this projection
        # are already nonzero at initialization. Gate gradients become active
        # once the symmetric features become nonzero.
        self.symmetric_projection = eqx.tree_at(
            lambda p: p.weight, projection, jnp.zeros_like(projection.weight)
        )
        self.symmetric_gate_projection = eqx.nn.Linear(
            2, d, use_bias=True, dtype=jnp.float32, key=qtx.get_subkeys()
        )

    def gate(self, couplings):
        inputs = couplings.astype(jnp.float32) / jnp.array(
            [jnp.pi, jnp.pi], jnp.float32
        )
        return jax.nn.softmax(self.gate_projection(inputs), axis=0)

    def symmetric_gate(self, couplings):
        inputs = couplings.astype(jnp.float32) / jnp.array(
            [jnp.pi, jnp.pi], jnp.float32
        )
        return jax.nn.softmax(self.symmetric_gate_projection(inputs), axis=0)

    def features(self, spins, couplings):
        inputs = couplings.astype(jnp.float32) / jnp.array(
            [jnp.pi, jnp.pi], jnp.float32
        )
        x = self.head.net.layers[0](spins)
        x = x + self.context_projection(inputs).reshape((-1,) + (1,) * (x.ndim - 1))
        # Share the expensive backbone, including its final normalization.
        for layer in self.head.net.layers[1:-2]:
            x = layer(x)
        antisymmetric = self.head.net.layers[-1](self.head.net.layers[-2](x))
        symmetric = qtx.get_lattice().to_original_repr(self.symmetric_projection(x))
        return antisymmetric, symmetric

    def symmetric_log_amplitude(self, spins, couplings):
        _, symmetric = self.features(spins, couplings)
        values = symmetric.reshape(self.head.d, -1)
        return jnp.sum(
            self.symmetric_gate(couplings)[:, None] * values, dtype=jnp.float64
        )

    def components(self, spins, couplings):
        antisymmetric, symmetric = self.features(spins, couplings)
        gate = self.gate(couplings)
        shape = antisymmetric.shape
        values = antisymmetric.reshape(-1, self.head.d, *shape[1:])
        gated = (
            values * gate.reshape((1, self.head.d) + (1,) * (antisymmetric.ndim - 1))
        ).reshape(shape)
        # Hand the precomputed feature map to the unchanged native Pfaffian head.
        head = eqx.tree_at(lambda h: h.net, self.head, lambda _: gated)
        pf = head(spins)
        J = jnp.sum(
            self.symmetric_gate(couplings)[:, None]
            * symmetric.reshape(self.head.d, -1),
            dtype=jnp.float64,
        )
        return pf, J

    def __call__(self, spins, couplings):
        pf, J = self.components(spins, couplings)
        return qtx.utils.LogArray(pf.sign, pf.logabs + J)
