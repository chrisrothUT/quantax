"""Small triangular Hubbard VMC example for multiplicative Pfaffian backflow.

Run from an environment with this checkout installed:
    python examples/multiplicative_pfaffian.py --steps 3

Use --steps 0 for a forward-pass check. The small defaults are a smoke example,
not the architecture or convergence schedule used in the reference paper.
"""

import argparse

import jax
import jax.numpy as jnp
import quantax as qtx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--opposite-spin", action="store_true")
    parser.add_argument("--complex", action="store_true", dest="complex_output")
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = jnp.complex128 if args.complex_output else jnp.float64
    qtx.set_default_dtype(dtype)
    qtx.set_random_seed(42)
    lattice = qtx.sites.Triangular(
        (3, 3), particle_type="spinful_fermion", Nparticles=(2, 2)
    )
    model = qtx.model.MultiplicativePfBackflow.resconv(
        d=4,
        nblocks=1,
        channels=8,
        dtype=jnp.float32,
        pairing_dtype=dtype,
        opposite_spin=args.opposite_spin,
    )
    state = qtx.state.Variational(model)
    print(f"Sites: {lattice.Nsites}, parameters: {state.nparams}")
    s = qtx.utils.rand_states(1)[0]
    psi = model(s)
    print(f"Initial sign/phase: {psi.sign}, log|psi|: {psi.logabs}")
    if args.steps <= 0:
        return
    hamiltonian = qtx.operator.Hubbard(U=8.0)
    sampler = qtx.sampler.ParticleHop(state, nsamples=args.samples)
    optimizer = qtx.optimizer.SR(state, hamiltonian)
    sampler.sweep()
    for step in range(args.steps):
        samples = sampler.sweep()
        update = optimizer.get_step(samples)
        state.update(1e-2 * update)
        print(f"Step {step}: E = {optimizer.energy}")


if __name__ == "__main__":
    main()
