"""Build a shared half-filled triangular Hubbard family; never submits jobs."""

from itertools import product
import numpy as np
import quantax as qtx


def anisotropic_hubbard(lattice, U, t_prime):
    """t=1 on diagonals and t_prime on the horizontal physical direction.

    Uses the installed Hubbard operator's fermion signs. For even L>=4 tori,
    horizontal bonds are unique and are a subset of the lattice neighbors.
    """
    from quantax.operator.common_operators import _hop

    if not np.isfinite([U, t_prime]).all():
        raise ValueError("U and t_prime must be finite")
    extent = np.asarray(lattice.shape[1:])
    xyz = lattice.xyz_from_index[:, 1:]
    shift = np.array([1.0, 0.0]) @ np.linalg.inv(lattice.basis_vectors)
    np.testing.assert_allclose(shift, np.rint(shift), atol=1e-12)
    target = (xyz + np.rint(shift).astype(int)) % extent
    indices = lattice.index_from_xyz[(np.zeros(len(xyz), dtype=int), *target.T)]
    edges = [(i, int(j)) for i, j in enumerate(indices)]
    keys = {tuple(sorted(edge)) for edge in edges}
    neighbors = {tuple(sorted(edge)) for edge in lattice.get_neighbor()}
    if len(keys) != lattice.Nsites or not keys <= neighbors:
        raise ValueError("Expected unique periodic x bonds; use L>=4")
    H = qtx.operator.Hubbard(U=float(U), t=1.0)
    for i, j in edges:
        H = H - (float(t_prime) - 1.0) * _hop(i, j)
    return H


def build(
    L=4,
    U_values=(6.0, 9.0, 12.0),
    t_prime_values=(0.8,),
    *,
    dim=32,
    heads=4,
    layers=2,
    samples_per_system=128,
    C0=0.1,
):
    """Create geometry once, a conditional state, joint optimizer and a batched sampler.

    Call in a fresh interpreter (Quantax has one global Hilbert space). Physical
    couplings remain (U,t_prime); model coupling_scale handles normalization. Change these
    grids for an exploratory calculation; this function schedules no work.
    Outputs follow the product order U_values x t_prime_values.
    """
    import jax.numpy as jnp

    if L < 4 or L % 2:
        raise ValueError("Use even L>=4 for balanced half-filled triangular tori")
    physical = tuple(product(U_values, t_prime_values))
    if not physical or not np.isfinite(physical).all():
        raise ValueError("Provide nonempty finite coupling grids")
    qtx.set_default_dtype(jnp.complex128)
    lattice = qtx.sites.Triangular(
        (L, L), particle_type="spinful_fermion", Nparticles=(L * L // 2, L * L // 2)
    )
    hamiltonians = qtx.operator.Anisohub(U_values, t_prime_values)
    contexts = hamiltonians.couplings
    model = qtx.model.ConditionalTransformer(
        2,
        dim=dim,
        heads=heads,
        layers=layers,
        patch_size=2,
        complex_output=True,
        dtype=jnp.float32,
        coupling_scale=(10.0, 1.0),
    )
    state = qtx.state.Foundation(model, contexts[0], max_parallel=(128, 16, 128))
    optimizer = qtx.optimizer.FoundationSR(
        state,
        hamiltonians,
        C0=C0,
        updater=qtx.optimizer.AdamUpdater(mu=0.95, beta=0.995),
        clip_mode="median_mad",
    )
    sampler = qtx.sampler.FamilyMetropolis(
        state,
        contexts,
        samples_per_coupling=samples_per_system,
        kind="particle_hop",
        thermal_steps=0,
        sweep_steps=4 * L * L,
    )
    return lattice, state, optimizer, sampler, hamiltonians


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--U", type=float, nargs="+", default=[6.0, 9.0, 12.0])
    parser.add_argument("--t-prime", type=float, nargs="+", default=[0.8])
    args = parser.parse_args()
    print(
        "Example only. Import build() from a fresh interpreter to initialize a family."
    )
    print("L=", args.L, "couplings=", list(product(args.U, args.t_prime)))
