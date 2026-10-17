quantax.optimizer.AdamSR
========================

.. currentmodule:: quantax.optimizer

.. autoclass:: AdamSR()
   :show-inheritance:
   :members:
   :inherited-members:

   .. automethod:: __init__

Norm-constrained AdamSR
-----------------------

The optional ``C0`` argument enables the schedule in the supplied AdamSR
Algorithm S1 and Eq. S19:

.. math::

   c_t = \frac{C_0}{1 + \max(t-t_0,0)/T}, \qquad
   \gamma_t = \min\left(1,\frac{c_t}{\|\delta\theta_t\|_2}\right).

Both the initial MinSR solve and the preconditioned correction use a
pseudoinverse by default in this mode. Adam's bias-corrected first and second
moments, and the fourth-root diagonal preconditioner, are computed before
applying the constraint. The returned direction already includes
:math:`\gamma_t`; the momentum buffers retain their unconstrained values.

.. code-block:: python

   optimizer = quantax.optimizer.AdamSR(
       state, hamiltonian, mu=0.95, beta=0.995,
       C0=0.1, t0=100, T=1000,
   )
   for samples in sample_stream:
       state.update(optimizer.get_step(samples))

The schedule values above are examples; the equations do not prescribe them.
``C0`` bounds the Euclidean norm of the complete parameter update, rather than
acting as a fixed learning rate. Do not multiply the returned step by the old
learning rate when following this algorithm.

The schedule clock starts at one and is stored with the Adam moment buffers.
When restoring ``file=...``, supply the same ``C0``, ``t0`` and ``T``; these
configuration values are not stored in the buffer file. Nonfinite momentum
updates retain the existing buffers and do not advance the clock.

Leaving ``C0=None`` preserves the previous behavior, including its default
``auto_shift_eig`` solver. A custom ``solver=...`` overrides either default.
