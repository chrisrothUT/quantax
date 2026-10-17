"""AdamSR training for a half-filled TriangularB lattice."""

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
from quantax.model import MultiplicativePfBackflow

RATES = [0.001, 0.003, 0.01, 0.03, 0.1]


def save_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    rates = parser.add_mutually_exclusive_group(required=True)
    rates.add_argument("--index", type=int, choices=range(len(RATES)))
    rates.add_argument("--learning-rate", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--U", type=float, default=8.0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--eval-sweeps", type=int, default=20)
    args = parser.parse_args()
    lr = args.learning_rate if args.learning_rate is not None else RATES[args.index]
    if not np.isfinite(lr) or lr <= 0:
        parser.error("Learning rate must be finite and positive")
    nsites = 3 * args.L**2
    if nsites % 2:
        raise ValueError("Half-filled balanced sectors require an even site count")
    nspin = nsites // 2
    thermal_steps, sweep_steps = 40 * nsites, 4 * nsites
    out = args.output / f"lr_{lr:g}"
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = dict(
        U=args.U,
        t=1.0,
        geometry="TriangularB",
        L=args.L,
        sites=nsites,
        electrons=[nspin, nspin],
        initialization="fresh_seed_3232",
        nblocks=8,
        channels=16,
        pairing_features=16,
        kernel_size=3,
        cnn_dtype="float32",
        pairing_dtype="float64",
        optimizer="AdamSR",
        mu=0.95,
        beta=0.995,
        energy_clip=5.0,
        learning_rate=lr,
        lr_schedule="constant",
        seed=3232,
        samples=args.samples,
        iterations=args.iterations,
        thermal_steps=thermal_steps,
        sweep_steps=sweep_steps,
        max_parallel=[1024, 128, 1024],
        eval_sweeps=args.eval_sweeps,
        quantax_file=qtx.__file__,
        job_id=os.environ.get("SLURM_JOB_ID"),
        array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
        hostname=platform.node(),
        versions={
            p: md.version(p) for p in ["quantax", "jax", "jaxlib", "fermix", "equinox"]
        },
    )
    src = Path(inspect.getfile(MultiplicativePfBackflow)).read_bytes()
    config["model_sha256"] = hashlib.sha256(src).hexdigest()
    (out / "model_source.py").write_bytes(src)
    (out / "training_script.py").write_text(Path(__file__).read_text())
    save_json(out / "config.json", config)
    try:
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")
        qtx.set_default_dtype(jnp.float64)
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
            d=16,
            nblocks=8,
            channels=16,
            kernel_size=3,
            dtype=jnp.float32,
            pairing_dtype=jnp.float64,
        )
        state = qtx.state.Variational(model, max_parallel=(1024, 128, 1024))
        config["parameters"] = state.nparams
        save_json(out / "config.json", config)
        state.save(out / "initial.eqx")
        hamiltonian = qtx.operator.Hubbard(U=args.U, t=1.0)
        optimizer = qtx.optimizer.AdamSR(
            state, hamiltonian, mu=0.95, beta=0.995, clip=5.0
        )
        print(json.dumps(config), flush=True)
        sampler = qtx.sampler.ParticleHop(
            state,
            nsamples=args.samples,
            thermal_steps=thermal_steps,
            sweep_steps=sweep_steps,
        )
        print(f"Thermalized at {time.monotonic()-started:.1f} seconds", flush=True)
        records = []
        with (out / "history.csv").open("w", buffering=1, newline="") as log:
            fields = [
                "iteration",
                "energy",
                "energy_per_site",
                "variance",
                "naive_sem",
                "step_norm",
                "applied_step_norm",
                "decay",
                "seconds",
                "elapsed_seconds",
            ]
            writer = csv.DictWriter(log, fieldnames=fields)
            writer.writeheader()
            for i in range(args.iterations):
                tick = time.monotonic()
                samples = sampler.sweep()
                update = optimizer.get_step(samples)
                energy = float(optimizer.energy)
                variance = float(optimizer.VarE)
                norm = float(jnp.linalg.norm(update))
                if not all(np.isfinite([energy, variance, norm])):
                    raise FloatingPointError(f"Nonfinite optimization at iteration {i}")
                state.update(lr * update)
                row = dict(
                    iteration=i + 1,
                    energy=energy,
                    energy_per_site=energy / nsites,
                    variance=variance,
                    naive_sem=np.sqrt(variance / args.samples),
                    step_norm=norm,
                    applied_step_norm=lr * norm,
                    decay=float(state.model.decay),
                    seconds=time.monotonic() - tick,
                    elapsed_seconds=time.monotonic() - started,
                )
                writer.writerow(row)
                records.append(row)
                if i < 5 or (i + 1) % 25 == 0:
                    print(json.dumps(row), flush=True)
                if (i + 1) % 100 == 0 or i + 1 == args.iterations:
                    state.save(out / "latest.eqx")
                    optimizer.save(out / "optimizer.eqx")
                    np.save(out / "samples.npy", np.asarray(samples.spins))
        # Compare final, frozen parameters rather than minimum noisy training energies.
        evaluation = []
        sampler.sweep(thermal_steps)
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
                iterations=len(locals().get("records", [])),
                elapsed_seconds=time.monotonic() - started,
                traceback=traceback.format_exc(),
            ),
        )
        raise


if __name__ == "__main__":
    main()
