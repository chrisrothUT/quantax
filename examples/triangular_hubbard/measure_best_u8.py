"""Measure full spin-dot correlations and spin-resolved 1-RDMs at a frozen checkpoint."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import quantax as qtx
from quantax.operator import (
    create_u,
    create_d,
    annihilate_u,
    annihilate_d,
    number_u,
    number_d,
    Operator,
)


def sp(i):
    return create_u(i) @ annihilate_d(i)


def sm(i):
    return create_d(i) @ annihilate_u(i)


def spin_dot(i, j):
    z_i = (number_u(i) - number_d(i)) * 0.5
    z_j = (number_u(j) - number_d(j)) * 0.5
    return z_i @ z_j + (sp(i) @ sm(j) + sm(i) @ sp(j)) * 0.5


def validate():
    # Two electrons on two sites: singlet, triplet Sz=0, and two doublons.
    qtx.sites.Chain(2, particle_type="spinful_fermion", Nparticles=(1, 1))
    pair = np.asarray(spin_dot(0, 1).get_quspin_op().toarray())
    local = np.asarray(spin_dot(0, 0).get_quspin_op().toarray())
    np.testing.assert_allclose(
        np.linalg.eigvalsh(pair), [-0.75, 0, 0, 0.25], atol=1e-13
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(local), [0, 0, 0.75, 0.75], atol=1e-13
    )
    w, v = np.linalg.eigh(pair)
    singlet = v[:, np.argmin(w)]
    total = sum((spin_dot(i, j) for i in range(2) for j in range(2)), Operator([]))
    np.testing.assert_allclose(
        singlet @ total.get_quspin_op().toarray() @ singlet, 0, atol=1e-13
    )
    for create, annihilate in [(create_u, annihilate_u), (create_d, annihilate_d)]:
        rho = np.array(
            [
                [
                    singlet
                    @ (create(i) @ annihilate(j)).get_quspin_op().toarray()
                    @ singlet
                    for j in range(2)
                ]
                for i in range(2)
            ]
        )
        np.testing.assert_allclose(rho, np.eye(2) * 0.5, atol=1e-13)
    print(
        "Exact two-site spin singlet/triplet, onsite moment and spin-resolved 1-RDM checks passed.",
        flush=True,
    )


def sem(blocks):
    return np.std(blocks, axis=0, ddof=1) / np.sqrt(len(blocks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    qtx.set_default_dtype(jnp.float64)
    qtx.set_random_seed(73421)
    if args.validate_only:
        validate()
        return
    if args.blocks < 2:
        raise ValueError("At least two evaluation blocks are required")
    if not all(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("GPU required for this measurement")
    started = time.monotonic()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.run / "config.json").read_text())
    model_source = Path(
        inspect.getfile(qtx.model.MultiplicativePfBackflow)
    ).read_bytes()
    assert (
        hashlib.sha256(model_source).hexdigest() == config["model_sha256"]
    ), "Model source changed since training"
    lattice = qtx.sites.TriangularB(
        2, particle_type="spinful_fermion", Nparticles=(6, 6)
    )
    n = lattice.Nsites
    model = qtx.model.MultiplicativePfBackflow.resconv(
        d=16,
        nblocks=8,
        channels=16,
        kernel_size=3,
        dtype=jnp.float32,
        pairing_dtype=jnp.float64,
    )
    state = qtx.state.Variational(
        model, param_file=args.run / "latest.eqx", max_parallel=(1024, 128, 1024)
    )
    saved = np.load(args.run / "samples.npy")
    assert saved.shape == (1024, 24)
    np.testing.assert_array_equal(
        np.sum(saved.reshape(1024, 2, 12) > 0, axis=2), np.full((1024, 2), 6)
    )
    extent = np.asarray(lattice.shape[1:])
    xyz = lattice.xyz_from_index[:, 1:]
    inverse = np.ravel_multi_index(((-xyz) % extent).T, extent)
    shifts = np.array(
        [np.ravel_multi_index(((xyz + r) % extent).T, extent) for r in xyz]
    )
    displacement = np.ravel_multi_index(
        np.moveaxis((xyz[None, :, :] - xyz[:, None, :]) % extent, -1, 0), extent
    )
    canonical = [r for r in range(n) if r <= inverse[r] and r != 0]
    rho_ops = [{}, {}]
    xy_ops = {}
    for r in canonical:
        xy_op = Operator([])
        for spin, (create, annihilate) in enumerate(
            [(create_u, annihilate_u), (create_d, annihilate_d)]
        ):
            op = Operator([])
            for i, j in enumerate(shifts[r]):
                j = int(j)
                op += ((create(i) @ annihilate(j)) + (create(j) @ annihilate(i))) * (
                    0.5 / n
                )
            rho_ops[spin][r] = op
        for i, j in enumerate(shifts[r]):
            j = int(j)
            xy_op += ((sp(i) @ sm(j)) + (sm(i) @ sp(j))) * (0.5 / n)
        xy_ops[r] = xy_op
    print(
        "Loaded",
        args.run,
        "samples",
        saved.shape,
        "canonical nonzero displacements",
        canonical,
        "devices",
        jax.devices(),
        flush=True,
    )

    def measure(samples):
        occupations = (np.asarray(samples.spins).reshape(-1, 2, n) > 0).astype(float)
        sz = (occupations[:, 0] - occupations[:, 1]) * 0.5
        double = np.mean(occupations[:, 0] * occupations[:, 1], axis=1)
        zz = np.stack([np.mean(sz * sz[:, p], axis=1) for p in shifts], axis=1)
        xy = np.zeros_like(zz)
        xy[:, 0] = (np.mean(occupations, axis=(1, 2)) * 2 - 2 * double) * 0.5
        rho = np.zeros((2, len(occupations), n))
        rho[:, :, 0] = np.mean(occupations, axis=2).T
        for r in canonical:
            for spin in range(2):
                value = np.asarray(rho_ops[spin][r].Oloc(state, samples))
                assert np.isfinite(value).all()
                rho[spin, :, r] = rho[spin, :, inverse[r]] = value
            value = np.asarray(xy_ops[r].Oloc(state, samples))
            assert np.isfinite(value).all()
            xy[:, r] = xy[:, inverse[r]] = value
        return dict(
            rho_up_r=rho[0],
            rho_down_r=rho[1],
            Czz_r=zz,
            Cxy_r=xy,
            Cdot_r=zz + xy,
            double_occupancy=double,
        )

    def samples_from(spins):
        spins = jnp.asarray(spins)
        return qtx.sampler.Samples(spins=spins, psi=state(spins))

    snapshot_samples = samples_from(saved)
    snapshot = measure(snapshot_samples)
    np.savez_compressed(out / "saved_snapshot_locals.npz", **snapshot)
    np.savez(
        out / "saved_snapshot_estimates.npz",
        **{key: val.mean(axis=0) for key, val in snapshot.items()},
    )
    hamiltonian = qtx.operator.Hubbard(U=8.0)
    edges = lattice.get_neighbor(1)

    def reconstructed_energy(values):
        r = displacement[edges[:, 0], edges[:, 1]]
        kinetic = -2 * (
            values["rho_up_r"][:, r].sum(axis=1)
            + values["rho_down_r"][:, r].sum(axis=1)
        )
        return kinetic + 8 * n * values["double_occupancy"]

    e_direct = np.asarray(hamiltonian.Oloc(state, snapshot_samples))
    # Translation averaging commutes with H; equality holds per configuration.
    np.testing.assert_allclose(
        reconstructed_energy(snapshot), e_direct, atol=1e-8, rtol=1e-9
    )
    print(
        "Saved-snapshot operator/energy consistency passed; E/site",
        e_direct.mean() / n,
        "elapsed",
        time.monotonic() - started,
        flush=True,
    )
    sampler = qtx.sampler.ParticleHop(
        state,
        nsamples=len(saved),
        initial_spins=jnp.asarray(saved),
        thermal_steps=480,
        sweep_steps=48,
    )
    blocks = {key: [] for key in snapshot}
    local_blocks = {key: [] for key in snapshot}
    configurations = []
    for b in range(args.blocks):
        values = measure(sampler.sweep())
        for key, value in values.items():
            blocks[key].append(value.mean(axis=0))
            local_blocks[key].append(value)
        configurations.append(np.asarray(sampler.sweep(0).spins))
        print(
            "Measurement block",
            b + 1,
            "E/site",
            float(reconstructed_energy(values).mean() / n),
            "elapsed",
            time.monotonic() - started,
            flush=True,
        )
    blocks = {key: np.asarray(value) for key, value in blocks.items()}
    np.savez_compressed(
        out / "measurement_locals.npz",
        **{key: np.asarray(value) for key, value in local_blocks.items()},
    )
    np.save(out / "measurement_samples.npy", np.asarray(configurations))
    np.savez(out / "block_means.npz", **blocks)
    q_indices = xyz.copy()
    q_raw = (xyz / extent) @ lattice.reciprocal_vectors
    reciprocal_shifts = (
        np.array([(i, j) for i in range(-3, 4) for j in range(-3, 4)])
        @ lattice.reciprocal_vectors
    )
    candidates = q_raw[:, None, :] - reciprocal_shifts[None, :, :]
    q_fold = candidates[
        np.arange(n), np.argmin(np.linalg.norm(candidates, axis=2), axis=1)
    ]
    phase = np.cos(q_raw @ lattice.coord.T)
    sq = blocks["Cdot_r"] @ phase.T
    szzq = blocks["Czz_r"] @ phase.T
    n_up_q = blocks["rho_up_r"] @ phase.T
    n_down_q = blocks["rho_down_r"] @ phase.T
    rho_up = blocks["rho_up_r"][:, displacement]
    rho_down = blocks["rho_down_r"][:, displacement]
    spin_corr = blocks["Cdot_r"][:, displacement]
    np.testing.assert_allclose(rho_up, np.swapaxes(rho_up, 1, 2), atol=1e-12)
    np.testing.assert_allclose(rho_down, np.swapaxes(rho_down, 1, 2), atol=1e-12)
    np.testing.assert_allclose(np.trace(rho_up, axis1=1, axis2=2), 6, atol=1e-12)
    np.testing.assert_allclose(np.trace(rho_down, axis1=1, axis2=2), 6, atol=1e-12)
    np.testing.assert_allclose(sq.sum(axis=1), n * blocks["Cdot_r"][:, 0], atol=1e-11)
    np.testing.assert_allclose(n_up_q.sum(axis=1), 6, atol=1e-11)
    np.testing.assert_allclose(n_down_q.sum(axis=1), 6, atol=1e-11)
    np.testing.assert_allclose(szzq[:, 0], 0, atol=1e-12)
    results = dict(
        q_indices=q_indices,
        q_raw=q_raw,
        q_fold=q_fold,
        site_coordinates=lattice.coord,
        site_cell_coordinates=xyz,
        rho_up=rho_up.mean(axis=0),
        rho_down=rho_down.mean(axis=0),
        rho_up_sem=sem(rho_up),
        rho_down_sem=sem(rho_down),
        spin_correlation=spin_corr.mean(axis=0),
        spin_correlation_sem=sem(spin_corr),
        S_q=sq.mean(axis=0),
        S_q_sem=sem(sq),
        Szz_q=szzq.mean(axis=0),
        Szz_q_sem=sem(szzq),
        n_up_q=n_up_q.mean(axis=0),
        n_down_q=n_down_q.mean(axis=0),
        n_up_q_sem=sem(n_up_q),
        n_down_q_sem=sem(n_down_q),
    )
    np.savez(out / "observables.npz", **results)
    for name in [
        "rho_up",
        "rho_down",
        "rho_up_sem",
        "rho_down_sem",
        "spin_correlation",
        "spin_correlation_sem",
    ]:
        np.savetxt(out / f"{name}.csv", results[name], delimiter=",")
    with (out / "momentum_observables.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "m1",
                "m2",
                "qx",
                "qy",
                "S_q",
                "S_q_block_sem",
                "Szz_q",
                "n_up_q",
                "n_down_q",
            ]
        )
        for k in range(n):
            writer.writerow(
                [
                    *q_indices[k],
                    *q_fold[k],
                    results["S_q"][k],
                    results["S_q_sem"][k],
                    results["Szz_q"][k],
                    results["n_up_q"][k],
                    results["n_down_q"][k],
                ]
            )
    e_blocks = np.array(
        [
            reconstructed_energy(
                {key: np.asarray(local_blocks[key][b]) for key in local_blocks}
            ).mean()
            for b in range(args.blocks)
        ]
    )
    peak = int(np.argmax(results["S_q"]))
    summary = dict(
        run=str(args.run),
        checkpoint="latest.eqx",
        samples_source="samples.npy",
        original_snapshot_samples=len(saved),
        snapshot_note="Snapshot precedes final training update; raw estimates retained, reported estimates re-equilibrated at frozen final checkpoint.",
        thermal_steps=480,
        measurement_blocks=args.blocks,
        samples_per_block=len(saved),
        measurement_samples=args.blocks * len(saved),
        translation_averaged=True,
        rdm_definition="rho_sigma[i,j] = <c^dagger_{i,sigma} c_{j,sigma}>",
        structure_factor_definition="S(q) = (1/N) sum_ij exp(i q.(r_i-r_j)) <S_i.S_j>; spin operators have eigenvalues +/-1/2",
        real_state_note="Wavefunction is real. Hermitian real hopping estimators enforce rho(r)=rho(-r); imaginary 1-RDM parts vanish exactly.",
        uncertainty="SEM of 20 sweep means; approximate if residual inter-sweep correlation remains.",
        energy_per_site=float(e_blocks.mean() / n),
        energy_per_site_sem=float(sem(e_blocks) / n),
        total_spin_squared=float(n * results["S_q"][0]),
        total_spin_squared_sem=float(n * results["S_q_sem"][0]),
        mean_double_occupancy=float(blocks["double_occupancy"].mean()),
        mean_local_moment_squared=float(blocks["Cdot_r"][:, 0].mean()),
        spin_sum_rule_residual=float(
            abs(results["S_q"].sum() - np.trace(results["spin_correlation"]))
        ),
        rho_up_trace=float(np.trace(results["rho_up"])),
        rho_down_trace=float(np.trace(results["rho_down"])),
        rho_up_eigenvalues=np.linalg.eigvalsh(results["rho_up"]).tolist(),
        rho_down_eigenvalues=np.linalg.eigvalsh(results["rho_down"]).tolist(),
        peak_q_index=q_indices[peak].tolist(),
        peak_q_cartesian=q_fold[peak].tolist(),
        peak_S_q=float(results["S_q"][peak]),
        peak_S_q_sem=float(results["S_q_sem"][peak]),
        total_seconds=time.monotonic() - started,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_plots(out, results)
    print(json.dumps(summary), flush=True)


def make_plots(out, data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    # Show all equivalent images on the BZ boundary, without point-group averaging.
    basis = np.linalg.lstsq(data["site_cell_coordinates"], data["site_coordinates"], rcond=None)[0]
    reciprocal = 2 * np.pi * np.linalg.inv(basis).T
    shifts = np.array([(i,j) for i in range(-3,4) for j in range(-3,4)]) @ reciprocal
    plot_points, plot_values = [], []
    for k, raw in enumerate(data["q_raw"]):
        candidates = raw[None,:] - shifts
        norms = np.sum(candidates**2, axis=1)
        for point in candidates[np.isclose(norms, norms.min(), atol=1e-10)]:
            plot_points.append(point)
            plot_values.append(data["S_q"][k])
    q = np.asarray(plot_points)
    angle = np.arange(6) * np.pi / 3
    boundary = np.column_stack([np.cos(angle), np.sin(angle)]) * (4 * np.pi / 3)
    axes[0].add_patch(Polygon(boundary, fill=False, color="gray", linestyle="--"))
    dots = axes[0].scatter(
        q[:, 0], q[:, 1], c=plot_values, s=140, cmap="viridis", edgecolor="black"
    )
    for i, point in enumerate(q):
        axes[0].annotate(
            f'{plot_values[i]:.3f}',
            point,
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=7,
        )
    axes[0].set(
        title=r"SU(2) invariant $S(\mathbf{q})$",
        xlabel=r"$q_x$",
        ylabel=r"$q_y$",
        aspect="equal",
    )
    axes[0].set_xlim(-4.7, 4.7)
    axes[0].set_ylim(-4.5, 4.5)
    fig.colorbar(dots, ax=axes[0], label=r"$S(\mathbf{q})$")
    limit = max(np.max(np.abs(data["rho_up"])), np.max(np.abs(data["rho_down"])))
    for ax, name, title in zip(
        axes[1:],
        ["rho_up", "rho_down"],
        [r"$\rho^\uparrow_{ij}$", r"$\rho^\downarrow_{ij}$"],
    ):
        im = ax.imshow(
            data[name], cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower"
        )
        ax.set(title=title, xlabel="j", ylabel="i")
        fig.colorbar(im, ax=ax)
    fig.savefig(out / "observables.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
