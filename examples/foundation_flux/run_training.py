"""Checkpointed N=48 Pfaffian with independently gated symmetric log amplitude."""

import argparse
import json
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import jax
import jax.numpy as jnp
import quantax as qtx
import quantax.global_defs as global_defs
from conditional_pfaffian import ConditionalPfaffian
from square_hubbard import SquareTwistHubbard

ROOT = Path(os.environ.get("FOUNDATION_RUN_DIR", Path.cwd())).resolve()
STOP = False


def stop(signum, frame):
    global STOP
    STOP = True
    print(
        f"Signal {signum}: checkpoint and stop at next consistent boundary", flush=True
    )


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


class MixedProposals:
    """The established 70% site-exchange / 30% particle-hop proposal mixture."""

    def __init__(self, hop, exchange):
        self.hop, self.exchange = hop, exchange

    def propose(self, key, spins):
        khop, kexchange, kchoose = jax.random.split(key, 3)
        hop = self.hop.propose(khop, spins)
        exchange = self.exchange.propose(kexchange, spins)
        use_exchange = jax.random.bernoulli(kchoose, 0.7, (spins.shape[0], 1))
        return jnp.where(use_exchange, exchange, hop)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    stop_after = args.iterations if args.stop_after is None else args.stop_after
    if not 0 < stop_after <= args.iterations:
        parser.error("Require 0 < stop-after <= iterations")
    out = ROOT / ("smoke" if args.smoke else "run")
    out.mkdir(exist_ok=True)
    qtx.set_default_dtype(jnp.complex128)
    qtx.set_random_seed(3232)
    jax.config.update("jax_default_matmul_precision", "highest")
    L, channels, d, blocks = (2, 4, 8, 1) if args.smoke else (6, 48, 192, 8)
    nsites = L * L
    particles = (1, 1) if args.smoke else (15, 15)
    lat = qtx.sites.Square(L, particle_type="spinful_fermion", Nparticles=particles)
    U = (
        np.array([0.0, np.pi])
        if args.smoke
        else np.linspace(0.0, 2 * np.pi, 20, endpoint=False)
    )
    tp = U.copy()
    H = SquareTwistHubbard(U, tp, sample_chunk=4)
    model = ConditionalPfaffian(channels=channels, d=d, nblocks=blocks)
    state = qtx.state.Foundation(model, H.couplings[0], max_parallel=(1024, 128, 1024))
    S = 4 if args.smoke else 20
    total = len(H) * S
    config = dict(
        geometry="Square",
        L=L,
        N=nsites,
        particles=list(particles),
        U=4.0,
        t=1.0,
        hole_doping=1 / 6,
        flux_x=U.tolist(),
        flux_y=tp.tolist(),
        grid_shape=list(H.grid_shape),
        systems=len(H),
        samples_per_coupling=S,
        total_samples=total,
        cnn_channels=channels,
        pairing_dimension=d,
        blocks=blocks,
        parameters=state.nparams,
        cnn_dtype="float32",
        pairing_dtype="complex128",
        flux_convention="uniform Peierls phase exp(i theta_axis/L); theta in radians, [0,2pi)",
        C0=0.5,
        learning_rate=1.0,
        optimizer="FoundationSR/Adam(mu=.95,beta=.995)",
        solver="auto_pinv_eig",
        clip=5.0,
        clip_mode="median_mad",
        sampler="70% SiteExchange / 30% ParticleHop",
        warmup_steps=4 if args.smoke else 40 * nsites,
        sweep_steps=4 if args.smoke else 4 * nsites,
        max_parallel=[1024, 128, 1024],
        sample_chunk_per_device=4,
        iterations=args.iterations,
        stop_after=stop_after,
        seed=3232,
        devices=len(jax.devices()),
        device_kinds=[x.device_kind for x in jax.devices()],
        execution_backend="native Quantax family kernels",
        quantax_file=qtx.__file__,
        dense_complex_jacobian_GiB=total * state.nparams * 16 / 2**30,
        dense_complex_jacobian_per_gpu_GiB=total
        * state.nparams
        * 16
        / 2**30
        / len(jax.devices()),
    )
    if args.preflight:
        spins = qtx.utils.rand_states(2).reshape(2, 1, lat.Nmodes)
        psi = state.evaluate(spins, H.couplings[jnp.array([0, len(H) - 1])])
        config["forward_finite"] = bool(jnp.all(jnp.isfinite(psi.logabs)))
        if not config["forward_finite"]:
            raise FloatingPointError("Nonfinite initial amplitudes")
        atomic_json(ROOT / "preflight.json", config)
        print(json.dumps(config, indent=2), flush=True)
        return
    if not args.smoke and (
        len(jax.devices()) != 2
        or not all("H100" in x.device_kind for x in jax.devices())
    ):
        raise RuntimeError(
            "Requires exactly two H100 GPUs; no CPU or other-GPU fallback"
        )
    latest = json.loads((out / "latest.json").read_text()) if args.resume else None
    if not args.resume and (out / "latest.json").exists():
        raise RuntimeError("Existing checkpoint: use --resume")
    optimizer = qtx.optimizer.FoundationSR(
        state,
        H,
        C0=0.5,
        solver=qtx.optimizer.auto_pinv_eig(),
        updater=qtx.optimizer.AdamUpdater(mu=0.95, beta=0.995),
        clip=5.0,
        clip_mode="median_mad",
    )
    sampler = qtx.sampler.FamilyMetropolis(
        state,
        H.couplings,
        samples_per_coupling=S,
        kind="particle_hop",
        thermal_steps=0,
        sweep_steps=config["sweep_steps"],
    )
    # Install the fixed mixture before the family sweep kernel is first traced.
    exchange = qtx.sampler.SiteExchange(
        state, nsamples=total, thermal_steps=0, sweep_steps=0
    )
    sampler._proposal = MixedProposals(sampler._proposal, exchange)
    history = []
    iteration = warm = 0
    if latest is not None:
        import equinox as eqx

        directory = out / latest["directory"]
        state._model = qtx.utils.filter_tree_map(
            qtx.utils.to_replicated_array,
            eqx.tree_deserialise_leaves(directory / "model.eqx", state.model),
        )
        optimizer._buffers = qtx.utils.filter_tree_map(
            qtx.utils.to_replicated_array,
            eqx.tree_deserialise_leaves(
                directory / "optimizer.eqx", optimizer._buffers
            ),
        )
        with np.load(directory / "sampler.npz") as z:
            sampler.reset(z["spins"])
            global_defs.KEY = qtx.utils.local_to_replicated(
                jax.random.wrap_key_data(jnp.asarray(z["rng"], dtype=jnp.uint32))
            )
        saved = json.loads((directory / "boundary.json").read_text())
        iteration, warm, history = (
            saved["iteration"],
            saved["warmup_steps"],
            saved["history"],
        )
    atomic_json(out / "config.json", config)
    np.savez_compressed(
        out / "geometry_and_grid.npz",
        flux_x=U,
        flux_y=tp,
        couplings=np.asarray(H.couplings),
        coefficients=np.asarray(H.coefficients),
        coord=lat.coord,
        basis=lat.basis_vectors,
        extent=lat.shape[1:],
    )
    for sig in [signal.SIGUSR1, signal.SIGTERM]:
        signal.signal(sig, stop)

    def checkpoint(reason):
        directory = out / f"checkpoint_i{iteration:05d}_w{warm:05d}_{time.time_ns()}"
        temp = directory.with_name(directory.name + ".tmp")
        temp.mkdir()
        state.save(temp / "model.eqx")
        optimizer.save(temp / "optimizer.eqx")
        np.savez_compressed(
            temp / "sampler.npz",
            spins=np.asarray(sampler.sweep(0).spins),
            rng=np.asarray(jax.random.key_data(global_defs.KEY)),
        )
        atomic_json(
            temp / "boundary.json",
            dict(
                iteration=iteration,
                warmup_steps=warm,
                history=history,
                reason=reason,
                energies_evaluated_before_update=True,
            ),
        )
        temp.rename(directory)
        atomic_json(
            out / "latest.json",
            dict(
                directory=directory.name,
                iteration=iteration,
                warmup_steps=warm,
                reason=reason,
            ),
        )
        print("CHECKPOINT", directory.name, reason, flush=True)

    checkpoint("initial_or_resumed")
    try:
        while warm < config["warmup_steps"] and not STOP:
            count = min(192, config["warmup_steps"] - warm)
            start = time.monotonic()
            batch = sampler.sweep(count)
            batch.spins.block_until_ready()
            warm += count
            print(
                "WARMUP",
                warm,
                "seconds",
                time.monotonic() - start,
                "acceptance",
                float(jnp.mean(sampler.acceptance)),
                flush=True,
            )
            checkpoint("warmup")
        (out / "diagnostics").mkdir(exist_ok=True)
        last_checkpoint = time.monotonic()
        while iteration < stop_after and not STOP:
            start = time.monotonic()
            batch = sampler.sweep()
            batch.spins.block_until_ready()
            sampled = time.monotonic()
            update = optimizer.get_step(batch)
            update.block_until_ready()
            solved = time.monotonic()
            if not bool(jnp.all(jnp.isfinite(update))):
                raise FloatingPointError(
                    "Nonfinite SR update; preserving preceding consistent checkpoint"
                )
            hubbard_energies = optimizer.energies
            hubbard_variances = optimizer.variances
            pinning_energies = jnp.zeros_like(optimizer.energies)
            norm = float(jnp.linalg.norm(update))
            state.update(update)
            iteration += 1
            entry = dict(
                iteration=iteration,
                sampling_seconds=sampled - start,
                energy_jacobian_solve_seconds=solved - sampled,
                total_seconds=solved - start,
                execution_backend=config["execution_backend"],
                ensemble_energy=float(jnp.mean(hubbard_energies)),
                objective_ensemble_energy=float(optimizer.energy),
                mean_pinning_energy=float(jnp.mean(pinning_energies)),
                pinning_h=0.0,
                mean_variance=float(jnp.mean(hubbard_variances)),
                objective_mean_variance=float(optimizer.VarE),
                update_norm=norm,
                mean_acceptance=float(jnp.mean(sampler.acceptance)),
            )
            history.append(entry)
            np.savez_compressed(
                out / "diagnostics" / f"step_{iteration:05d}.npz",
                energies=np.asarray(hubbard_energies),
                objective_energies=np.asarray(optimizer.energies),
                pinning_energies=np.asarray(pinning_energies),
                variances=np.asarray(hubbard_variances),
                objective_variances=np.asarray(optimizer.variances),
                term_energies=np.asarray(optimizer.term_energies),
                spins=np.asarray(batch.spins),
                acceptance=np.asarray(sampler.acceptance),
                **entry,
            )
            atomic_json(out / "history.json", history)
            print("ITERATION", json.dumps(entry), flush=True)
            if iteration % 3 == 0 or time.monotonic() - last_checkpoint > 900:
                checkpoint("training")
                last_checkpoint = time.monotonic()
        checkpoint("signal" if STOP else "complete")
        atomic_json(
            out / "summary.json",
            dict(
                status="interrupted" if STOP else "complete",
                iterations=iteration,
                warmup_steps=warm,
                history=history,
                static_views_created=len(state._views),
            ),
        )
    except Exception:
        atomic_json(
            out / "failure.json",
            dict(
                iteration=iteration, warmup_steps=warm, traceback=traceback.format_exc()
            ),
        )
        # Do not overwrite the last consistent restart with partially updated buffers.
        raise


if __name__ == "__main__":
    main()
