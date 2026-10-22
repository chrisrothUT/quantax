"""Small exact-sum FNQS benchmark; no Monte Carlo noise, cluster job or GPU needed."""

import argparse
import json
from pathlib import Path
import numpy as np


def train(steps=120, sites=6, output=None):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import quantax as qtx

    qtx.set_default_dtype(jnp.float64)
    qtx.set_random_seed(42)
    qtx.sites.Chain(sites)
    model = qtx.model.ConditionalTransformer(
        1, dim=8, heads=2, layers=1, patch_size=2, dtype=jnp.float64
    )
    state = qtx.state.Foundation(model, [0.6], max_parallel=(64, 16, 64))
    fields = np.array([0.6, 1.0, 1.4])
    H = [qtx.operator.Ising(h=float(h)) for h in fields]
    optimizer = qtx.optimizer.FoundationSR(
        state,
        H,
        [[float(h)] for h in fields],
        clip=None,
        C0=1.0,
        solver=qtx.optimizer.auto_shift_eig(ashift=1e-3),
    )
    symm = qtx.symmetry.Identity()
    symm.basis_make()
    spins = qtx.utils.to_distributed_array(qtx.utils.ints_to_array(symm.basis.states))
    reference = np.array([np.linalg.eigvalsh(h.todense())[0] for h in H])
    history = []
    for iteration in range(steps):
        batches = []
        for view in optimizer.states:
            psi = view(spins)
            probability = jax.nn.softmax(2 * psi.logabs)
            batches.append(qtx.sampler.Samples(spins, reweight_factor=probability))
        update = optimizer.get_step(batches)
        history.append(np.asarray(optimizer.energies))
        state.update(0.1 * update)
        if iteration % 25 == 0 or iteration + 1 == steps:
            print(iteration + 1, np.asarray(optimizer.energies), flush=True)
    test_fields = np.array([0.6, 0.8, 1.0, 1.2, 1.4])
    predictions = []
    exact = []
    for h in test_fields:
        op = qtx.operator.Ising(h=float(h))
        matrix = op.todense()
        psi = np.asarray(state.at([float(h)])(spins))
        psi = psi / np.linalg.norm(psi)
        predictions.append(float(np.vdot(psi, matrix @ psi).real))
        exact.append(float(np.linalg.eigvalsh(matrix)[0]))
    report = dict(
        sites=sites,
        steps=steps,
        parameters=state.nparams,
        training_fields=fields.tolist(),
        evaluated_fields=test_fields.tolist(),
        variational_energies=predictions,
        exact_energies=exact,
        relative_errors=(
            (np.array(predictions) - exact) / abs(np.array(exact))
        ).tolist(),
        initial_training_energies=history[0].tolist(),
        last_training_energies=history[-1].tolist(),
        note="Held-out fields 0.8 and 1.2 receive no optimization. Exact sums replace MCMC in this small benchmark.",
    )
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output / "training.npz",
            fields=fields,
            history=np.array(history),
            exact=reference,
            test_fields=test_fields,
            predictions=np.array(predictions),
            test_exact=np.array(exact),
        )
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        state.save(output / "model.eqx")
        optimizer.save(output / "optimizer.eqx")
    return state, optimizer, np.array(history), reference, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--sites", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    *_, report = train(args.steps, args.sites, args.output)
    print(json.dumps(report, indent=2))
