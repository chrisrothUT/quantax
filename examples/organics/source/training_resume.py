"""Complete, committed boundaries for continuation across batch jobs."""

import json
from pathlib import Path
import jax
import numpy as np
from quantax import global_defs


def save_boundary(
    out, state, optimizer, sampler, iteration, elapsed_seconds, warmup_completed=None
):
    name = (
        f"iteration_{iteration:06d}"
        if warmup_completed is None
        else f"warmup_{warmup_completed:06d}"
    )
    directory = Path(out) / "resume" / name
    directory.mkdir(parents=True, exist_ok=False)
    state.save(directory / "model.eqx")
    optimizer.save(directory / "optimizer.eqx")
    np.save(directory / "spins.npy", np.asarray(sampler.sweep(0).spins))
    np.save(directory / "rng.npy", np.asarray(jax.random.key_data(global_defs.KEY)))
    metadata = dict(
        iteration=iteration,
        elapsed_seconds=elapsed_seconds,
        rng_impl=str(jax.random.key_impl(global_defs.KEY)),
    )
    if warmup_completed is not None:
        metadata["warmup_completed"] = warmup_completed
    (directory / "boundary.json").write_text(json.dumps(metadata) + "\n")
    pointer = Path(out) / "resume.json.tmp"
    pointer.write_text(
        json.dumps(dict(directory=str(directory.resolve()), **metadata)) + "\n"
    )
    pointer.replace(Path(out) / "resume.json")
    return metadata


def restore_rng(directory, impl):
    global_defs.KEY = jax.random.wrap_key_data(
        np.load(Path(directory) / "rng.npy"), impl=impl
    )
