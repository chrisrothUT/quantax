"""Half-filled anisotropic triangular CNN Pfaffian training with durable restart boundaries."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import csv
import hashlib
import importlib.metadata as md
import inspect
import json
from pathlib import Path
import platform
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
import quantax as qtx
from sampler_diagnostics import save_diagnostics

RATES = [0.001, 0.003, 0.01, 0.03, 0.1]


def save_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--C0", type=float, default=1.0)
    parser.add_argument("--t-prime", type=float, required=True)
    parser.add_argument("--complex-features", action="store_true")
    parser.add_argument("--max-parallel", type=int, nargs=3, default=[256, 32, 256])
    parser.add_argument("--t0", type=float, default=1000)
    parser.add_argument("--T", type=float, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--U", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=3232)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--eval-sweeps", type=int, default=20)
    parser.add_argument("--transfer-from", type=Path)
    parser.add_argument(
        "--clip-mode", choices=["mean_std", "median_mad"], default="mean_std"
    )
    parser.add_argument(
        "--sampler", choices=["particle_hop", "mixed_exchange"], default="particle_hop"
    )
    parser.add_argument("--constant-norm", action="store_true")
    parser.add_argument(
        "--pairing-features",
        type=int,
        default=96,
        help="Pairing dimension; fixed at 96 by default for every size",
    )
    parser.add_argument("--cnn-channels", type=int, default=32)
    parser.add_argument("--rank-checkpoint-every", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stop-after", type=int, help="Absolute iteration at this job's boundary"
    )
    args = parser.parse_args()
    if not np.isfinite(args.t_prime) or args.t_prime <= 0:
        parser.error("t-prime must be finite and positive")
    if args.complex_features:
        from complex_pfaffian import MultiplicativePfBackflow
    else:
        from quantax.model import MultiplicativePfBackflow
    pairing_dtype = jnp.complex128 if args.complex_features else jnp.float64
    stop_after = args.iterations if args.stop_after is None else args.stop_after
    if not 0 < stop_after <= args.iterations:
        parser.error("stop-after must be between 1 and iterations")
    if args.resume and args.transfer_from:
        parser.error("resume and transfer-from are mutually exclusive")
    if (
        (args.pairing_features is not None and args.pairing_features <= 0)
        or args.cnn_channels <= 0
        or args.rank_checkpoint_every < 0
    ):
        parser.error(
            "pairing-features and cnn-channels must be positive; rank-checkpoint-every must be nonnegative"
        )
    # A finite t0 beyond the full run disables decay on the optimizer clock.
    constraint_t0 = max(args.t0, args.iterations) if args.constant_norm else args.t0
    lr = 1.0
    if not np.isfinite(args.C0) or args.C0 <= 0:
        parser.error("C0 must be finite and positive")
    if not np.isfinite(lr) or lr <= 0:
        parser.error("Learning rate must be finite and positive")
    nsites = 3 * args.L**2
    if nsites % 2:
        raise ValueError("Half-filled balanced sectors require an even site count")
    nspin = nsites // 2
    thermal_steps, sweep_steps = 40 * nsites, 4 * nsites
    out = args.output / f"C0_{args.C0:g}"
    out.mkdir(parents=True, exist_ok=args.resume)
    resume = json.loads((out / "resume.json").read_text()) if args.resume else None
    start_iteration = resume["iteration"] if resume else 0
    if start_iteration >= stop_after:
        raise ValueError("Checkpoint must precede requested stop-after")
    checkpoint = Path(resume["directory"]) if resume else None
    previous_config = json.loads((out / "config.json").read_text()) if resume else None
    started = time.monotonic() - (resume["elapsed_seconds"] if resume else 0)
    config = dict(
        U=args.U,
        t=1.0,
        t_prime=args.t_prime,
        anisotropic_direction="x",
        point_group_projection=False,
        geometry="TriangularB",
        L=args.L,
        sites=nsites,
        electrons=[nspin, nspin],
        initialization=f"fresh_seed_{args.seed}",
        nblocks=8,
        channels=args.cnn_channels,
        pairing_features=(
            2 * nsites if args.pairing_features is None else args.pairing_features
        ),
        rank_checkpoint_every=args.rank_checkpoint_every,
        kernel_size=3,
        cnn_dtype="float32",
        pairing_dtype="complex128" if args.complex_features else "float64",
        complex_features=args.complex_features,
        optimizer="AdamSR_norm_constrained",
        C0=args.C0,
        constraint_t0=constraint_t0,
        constant_norm=args.constant_norm,
        sampler=args.sampler,
        site_exchange_probability=0.7 if args.sampler == "mixed_exchange" else 0.0,
        constraint_T=args.T,
        solver="auto_pinv_eig",
        mu=0.95,
        beta=0.995,
        energy_clip=5.0,
        energy_clip_mode=args.clip_mode,
        energy_clip_scale_floor=1e-8,
        energy_clip_mad_normalization="unscaled",
        learning_rate=lr,
        lr_schedule=(
            "unit_lr_with_constant_norm_cap"
            if args.constant_norm
            else "unit_lr_with_scheduled_norm_cap"
        ),
        seed=args.seed,
        samples=args.samples,
        iterations=args.iterations,
        thermal_steps=thermal_steps,
        sweep_steps=sweep_steps,
        max_parallel=args.max_parallel,
        eval_sweeps=args.eval_sweeps,
        quantax_file=qtx.__file__,
        job_id=os.environ.get("SLURM_JOB_ID"),
        array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
        hostname=platform.node(),
        versions={
            p: md.version(p) for p in ["quantax", "jax", "jaxlib", "fermix", "equinox"]
        },
    )
    from corrected_distance_envelope import metadata as envelope_metadata

    config.update(envelope_metadata())
    (out / "distance_envelope_helper.py").write_text(
        Path(__file__).with_name("corrected_distance_envelope.py").read_text()
    )
    if (
        previous_config
        and previous_config.get("distance_envelope_policy")
        != config["distance_envelope_policy"]
    ):
        raise ValueError("Cannot change distance policy across a resume")
    if previous_config:
        for key in [
            "t_prime",
            "anisotropic_direction",
            "complex_features",
            "U",
            "L",
            "C0",
            "constraint_t0",
            "constraint_T",
            "constant_norm",
            "samples",
            "iterations",
            "sampler",
            "channels",
            "pairing_features",
            "energy_clip_mode",
            "seed",
            "max_parallel",
        ]:
            if config[key] != previous_config[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        config["initialization"] = previous_config["initialization"]
        config["resume_jobs"] = previous_config.get("resume_jobs", []) + [
            config["job_id"]
        ]
    src = Path(inspect.getfile(MultiplicativePfBackflow)).read_bytes()
    config["model_sha256"] = hashlib.sha256(src).hexdigest()
    (out / "model_source.py").write_bytes(src)
    (out / "training_script.py").write_text(Path(__file__).read_text())
    for name in ["sr.py", "updater.py", "solver.py", "qngd.py", "gradient.py"]:
        source = Path(qtx.__file__).parent / "optimizer" / name
        raw = source.read_bytes()
        (out / f"optimizer_{name}").write_bytes(raw)
        config[f"optimizer_{name}_sha256"] = hashlib.sha256(raw).hexdigest()
    from mixed_sampler import make_sampler

    sampler_source = Path(inspect.getfile(make_sampler)).read_bytes()
    (out / "sampler_source.py").write_bytes(sampler_source)
    config["sampler_sha256"] = hashlib.sha256(sampler_source).hexdigest()
    (out / "sampler_diagnostics_source.py").write_text(
        Path(inspect.getfile(save_diagnostics)).read_text()
    )
    config["sampler_diagnostics"] = args.sampler == "mixed_exchange"
    for name in ["common_samplers.py", "metropolis.py"]:
        raw = (Path(qtx.__file__).parent / "sampler" / name).read_bytes()
        (out / f"sampler_{name}").write_bytes(raw)
        config[f"sampler_{name}_sha256"] = hashlib.sha256(raw).hexdigest()
    if args.rank_checkpoint_every:
        from rank_checkpoints import save_rank_checkpoint

        source = Path(inspect.getfile(save_rank_checkpoint)).read_bytes()
        (out / "rank_checkpoints_source.py").write_bytes(source)
        config["rank_checkpoints_sha256"] = hashlib.sha256(source).hexdigest()
    config["hamiltonian_sha256"] = hashlib.sha256(
        Path(__file__).with_name("anisotropic_hubbard.py").read_bytes()
    ).hexdigest()
    (out / "hamiltonian_source.py").write_bytes(
        Path(__file__).with_name("anisotropic_hubbard.py").read_bytes()
    )
    resume_source = Path(__file__).with_name("training_resume.py").read_bytes()
    (out / "training_resume_source.py").write_bytes(resume_source)
    config["training_resume_sha256"] = hashlib.sha256(resume_source).hexdigest()
    if previous_config:
        for key, value in previous_config.items():
            if key.endswith("_sha256") or key == "versions":
                if config.get(key) != value:
                    raise ValueError(f"Resume source/version mismatch: {key}")
    save_json(out / "config.json", config)
    try:
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")
        qtx.set_default_dtype(pairing_dtype)
        qtx.set_random_seed(config["seed"])
        config["devices"] = [str(x) for x in jax.devices()]
        config["device_kinds"] = [x.device_kind for x in jax.devices()]
        save_json(out / "config.json", config)
        if not all(d.platform == "gpu" for d in jax.devices()):
            raise RuntimeError(
                "This benchmark requires the allocated GPU; CPU fallback is disabled."
            )
        lattice = qtx.sites.TriangularB(
            args.L, particle_type="spinful_fermion", Nparticles=(nspin, nspin)
        )
        assert lattice.Nsites == nsites
        model = MultiplicativePfBackflow.resconv(
            d=args.pairing_features,
            nblocks=8,
            channels=args.cnn_channels,
            kernel_size=3,
            dtype=jnp.float32,
            pairing_dtype=pairing_dtype,
        )
        if args.transfer_from is not None:
            from transfer_n48 import transfer_model

            model, provenance = transfer_model(
                model, args.transfer_from, args.L, args.U
            )
            (out / "transfer_source.py").write_text(
                Path(inspect.getfile(transfer_model)).read_text()
            )
            config["initialization"] = "transferred_checkpoint"
            config["transfer"] = provenance
        from corrected_distance_envelope import configure as configure_envelope

        model = configure_envelope(model, config)
        state = qtx.state.Variational(
            model,
            param_file=checkpoint / "model.eqx" if resume else None,
            max_parallel=tuple(args.max_parallel),
        )
        config["parameters"] = state.nparams
        save_json(out / "config.json", config)
        if not resume:
            state.save(out / "initial.eqx")
        from anisotropic_hubbard import hubbard, bond_metadata

        config.update(bond_metadata(lattice, args.t_prime))
        save_json(out / "config.json", config)
        hamiltonian = hubbard(lattice, U=args.U, t_prime=args.t_prime)
        optimizer = qtx.optimizer.AdamSR(
            state,
            hamiltonian,
            file=checkpoint / "optimizer.eqx" if resume else None,
            mu=0.95,
            beta=0.995,
            clip=5.0,
            clip_mode=args.clip_mode,
            clip_scale_floor=1e-8,
            C0=args.C0,
            t0=constraint_t0,
            T=args.T,
        )
        print(json.dumps(config), flush=True)
        sampler = make_sampler(
            state,
            kind=args.sampler,
            nsamples=args.samples,
            thermal_steps=0,
            initial_spins=(
                jnp.asarray(np.load(checkpoint / "spins.npy")) if resume else None
            ),
            sweep_steps=sweep_steps,
        )
        from training_resume import save_boundary, restore_rng

        if resume:
            restore_rng(checkpoint, resume["rng_impl"])
        warm_completed = resume.get("warmup_completed", thermal_steps) if resume else 0
        for warm in range(warm_completed, thermal_steps, sweep_steps):
            count = min(sweep_steps, thermal_steps - warm)
            sampler.sweep(count)
            save_boundary(
                out,
                state,
                optimizer,
                sampler,
                0,
                time.monotonic() - started,
                warmup_completed=warm + count,
            )
            save_json(
                out / "warmup_progress.json",
                dict(completed=warm + count, total=thermal_steps),
            )
            print(f"Warmup {warm+count}/{thermal_steps}", flush=True)
        if not resume or "warmup_completed" in resume:
            save_diagnostics(sampler, out, "initial_warmup")
        if args.rank_checkpoint_every and not resume:
            save_rank_checkpoint(
                state, optimizer, sampler.sweep(0), hamiltonian, out, 0
            )
        print(f"Thermalized at {time.monotonic()-started:.1f} seconds", flush=True)
        if not resume or "warmup_completed" in resume:
            save_boundary(out, state, optimizer, sampler, 0, time.monotonic() - started)
        last_durable_checkpoint = time.monotonic()
        records = []
        if resume:
            from checkpoint_history import reconcile_history

            records = reconcile_history(out, start_iteration)
        with (out / "history.csv").open(
            "a" if resume else "w", buffering=1, newline=""
        ) as log:
            fields = [
                "sampling_seconds",
                "optimizer_seconds",
                "update_seconds",
                "iteration",
                "energy",
                "energy_per_site",
                "variance",
                "naive_sem",
                "step_norm",
                "applied_step_norm",
                "constraint_bound",
                "decay",
                "seconds",
                "elapsed_seconds",
            ]
            writer = csv.DictWriter(log, fieldnames=fields)
            if log.tell() == 0:
                writer.writeheader()
            for i in range(start_iteration, stop_after):
                tick = time.monotonic()
                samples = sampler.sweep()
                save_diagnostics(sampler, out, f"training_{i+1:05d}")
                sampled_at = time.monotonic()
                update = optimizer.get_step(samples)
                energy = float(optimizer.energy)
                variance = float(optimizer.VarE)
                norm = float(jnp.linalg.norm(update))
                if not all(np.isfinite([energy, variance, norm])):
                    raise FloatingPointError(f"Nonfinite optimization at iteration {i}")
                bound = args.C0 / (1 + max(i + 1 - constraint_t0, 0) / args.T)
                if norm > bound * (1 + 1e-6):
                    raise RuntimeError(f"Norm {norm} exceeded bound {bound}")
                optimized_at = time.monotonic()
                state.update(update)
                jax.block_until_ready(state.get_params_flatten())
                updated_at = time.monotonic()
                row = dict(
                    sampling_seconds=sampled_at - tick,
                    optimizer_seconds=optimized_at - sampled_at,
                    update_seconds=updated_at - optimized_at,
                    iteration=i + 1,
                    energy=energy,
                    energy_per_site=energy / nsites,
                    variance=variance,
                    naive_sem=np.sqrt(variance / args.samples),
                    step_norm=norm,
                    applied_step_norm=norm,
                    constraint_bound=bound,
                    decay=float(state.model.decay),
                    seconds=time.monotonic() - tick,
                    elapsed_seconds=time.monotonic() - started,
                )
                writer.writerow(row)
                records.append(row)
                if i < 5 or (i + 1) % 25 == 0:
                    print(json.dumps(row), flush=True)
                if (i + 1) % 100 == 0 or i + 1 == stop_after:
                    state.save(out / "latest.eqx")
                    optimizer.save(out / "optimizer.eqx")
                    np.save(out / "samples.npy", np.asarray(samples.spins))
                if i + 1 < stop_after and (
                    (i + 1) % 25 == 0
                    or time.monotonic() - last_durable_checkpoint >= 1800
                ):
                    log.flush()
                    os.fsync(log.fileno())
                    save_boundary(
                        out,
                        state,
                        optimizer,
                        sampler,
                        i + 1,
                        time.monotonic() - started,
                    )
                    last_durable_checkpoint = time.monotonic()
                    print("Committed periodic checkpoint", i + 1, flush=True)
                if args.rank_checkpoint_every and (
                    (i + 1) % args.rank_checkpoint_every == 0
                    or i + 1 == args.iterations
                ):
                    save_rank_checkpoint(
                        state, optimizer, samples, hamiltonian, out, i + 1
                    )
        save_boundary(
            out, state, optimizer, sampler, stop_after, time.monotonic() - started
        )
        if stop_after < args.iterations:
            save_json(
                out / "summary.json",
                dict(
                    status="checkpointed",
                    iterations=stop_after,
                    total_seconds=time.monotonic() - started,
                ),
            )
            print(
                f"Committed continuation checkpoint at iteration {stop_after}",
                flush=True,
            )
            return
        # Compare final, frozen parameters rather than minimum noisy training energies.
        evaluation = []
        sampler.sweep(thermal_steps)
        save_diagnostics(sampler, out, "final_warmup")
        with (out / "evaluation.csv").open("w", buffering=1, newline="") as log:
            writer = csv.writer(log)
            writer.writerow(["sweep", "energy", "variance"])
            for i in range(args.eval_sweeps):
                samples = sampler.sweep()
                eloc = np.asarray(hamiltonian.Oloc(state, samples))
                if not np.isfinite(eloc).all():
                    raise FloatingPointError("Nonfinite final evaluation")
                energy = float(eloc.real.mean())
                variance = float(np.mean(np.abs(eloc - energy) ** 2))
                save_diagnostics(
                    sampler, out, f"evaluation_{i+1:03d}", energy=eloc, keep_spins=True
                )
                evaluation.append(energy)
                writer.writerow([i + 1, energy, variance])
        # These samples now match the frozen final checkpoint, unlike the pre-update training snapshots.
        np.save(out / "samples.npy", np.asarray(samples.spins))
        config["samples_match_final_checkpoint"] = True
        save_json(out / "config.json", config)
        tail = np.array([r["energy"] for r in records[-200:]])
        summary = dict(
            status="complete",
            learning_rate=lr,
            C0=args.C0,
            iterations=len(records),
            tail200_energy_mean=float(tail.mean()),
            tail200_energy_std=float(tail.std()),
            final_energy=float(np.mean(evaluation)),
            final_energy_per_site=float(np.mean(evaluation) / nsites),
            final_block_sem=(
                float(np.std(evaluation, ddof=1) / np.sqrt(len(evaluation)))
                if len(evaluation) > 1
                else None
            ),
            final_evaluation_samples=args.eval_sweeps * args.samples,
            median_step_seconds=(
                float(np.median([r["seconds"] for r in records[5:]]))
                if len(records) > 5
                else None
            ),
            total_seconds=time.monotonic() - started,
        )
        save_json(out / "summary.json", summary)
        print(json.dumps(summary), flush=True)
    except Exception:
        save_json(
            out / "summary.json",
            dict(
                status="failed",
                learning_rate=lr,
                C0=args.C0,
                iterations=len(locals().get("records", [])),
                elapsed_seconds=time.monotonic() - started,
                traceback=traceback.format_exc(),
            ),
        )
        raise


if __name__ == "__main__":
    main()
