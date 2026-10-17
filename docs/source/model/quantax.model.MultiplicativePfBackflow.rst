quantax.model.MultiplicativePfBackflow
====================================

.. currentmodule:: quantax.model

.. autoclass:: MultiplicativePfBackflow()
   :members:

   .. automethod:: __init__
   .. automethod:: __call__

Usage
-----

The residual CNN produces one vector of ``d`` features per spin-orbital, with
spin-up and spin-down features in separate channel blocks. The output head
contracts two feature vectors with a displacement-dependent pairing weight and
a normalized distance envelope. A Pfaffian of the occupied submatrix gives the
amplitude, including same-spin pairing. No translation projection is needed.

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import quantax as qtx

    jax.config.update("jax_enable_x64", True)
    qtx.set_default_dtype(jnp.float64)
    qtx.set_random_seed(42)
    lattice = qtx.sites.Triangular(
        (3, 3), particle_type="spinful_fermion", Nparticles=(2, 2)
    )
    model = qtx.model.MultiplicativePfBackflow.resconv(
        d=16, nblocks=4, channels=64, kernel_size=3,
        dtype=jnp.float32, pairing_dtype=jnp.float64,
    )
    state = qtx.state.Variational(model)

For a complex wavefunction, set the Quantax default dtype to ``jnp.complex128``
and ``pairing_dtype=jnp.complex128``. Pairing weights are represented by two real
parameter arrays, so they can be trained together with the real CNN and real
inverse decay length. Complex conjugation is never used in the pairing bilinear
or its antisymmetry constraint.

A custom equivariant network can be supplied as
``MultiplicativePfBackflow(net, d=16)``. It must output channels-first features
with channel order ``(spin, site_in_cell, feature)`` and spatial axes in the
original lattice ordering. An arbitrary network is not automatically made
equivariant by the pairing head.

BCS limit and scope
-------------------

``opposite_spin=True`` structurally removes the same-spin weights. For equal
numbers ``m`` of up and down electrons, the occupied matrix is
``[[0, B], [-B.T, 0]]`` and its Pfaffian is
``(-1)**(m*(m-1)//2) * det(B)``. Quantax's additional fermion ordering sign is
included by both modes. Opposite-spin pairing alone does not impose singlet
symmetry on ``B``.

Only periodic boundary conditions and a fixed, positive, even total fermion
number are currently supported. Both spinful and spinless fermions, and lattices
with multiple sites per cell, are supported. The generic multi-site case is
invariant under cell translations; extra symmetries exchanging basis sites are
not imposed. Odd-particle states need a blocked-orbital extension.

Each weight and its spin/displacement-reversed partner share a single trainable
parameter, with opposite signs. Self-reversing channels therefore vanish. This
includes onsite same-spin pairing and, for one site per cell, same-spin channels
at displacements equal to their negatives modulo the lattice extent. These
zeros are required by the shared-feature form of Eq. (7).

The decay envelope is normalized over spatial site pairs, independently of the
spin multiplicity. Its inverse length is ``softplus(log_decay)``. The physical
lattice distances are constants rather than variational parameters. The head
contracts only occupied pairs in the normal forward pass and uses a stable
``LogArray`` output. It recomputes the Pfaffian; accelerated local low-rank
updates are not implemented for this multiplicative head.

This implements the supplied Pfaffian Eq. (7), extending the symmetry-preserving
BCS pairing in Eq. (4) of `Rende et al., arXiv:2608.12465
<https://arxiv.org/abs/2608.12465>`_. The CNN uses Quantax's existing residual
blocks. It does not reproduce the paper's transformer, seeded d-wave term, or
published training schedule.
