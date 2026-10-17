"""Preserve training checkpoints and singular spectra with energy-matched probes."""
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from quantax.nn import fermion_idx


def singular_summary(s, shape):
    s=np.asarray(s)
    relative=s/s[..., :1]
    tol=max(shape)*np.finfo(s.dtype).eps
    rank=(relative>tol).sum(axis=-1)
    condition=np.divide(s[...,0],s[...,-1],out=np.full_like(s[...,0],np.inf),where=s[...,-1]>0)
    return dict(shape=list(shape),rank_min=int(rank.min()),rank_max=int(rank.max()),
                rank_at_relative_1e8_min=int((relative>1e-8).sum(axis=-1).min()),
                rank_at_relative_1e12_min=int((relative>1e-12).sum(axis=-1).min()),
                condition_median=float(np.median(condition)),condition_max=float(condition.max()),
                relative_smin_min=float(relative[...,-1].min()))


@eqx.filter_jit
def probe_matrices(model, spins):
    full=jax.vmap(lambda x:model.pairing_matrix(x,occupied_only=False))(spins)
    occupied=jax.vmap(model.pairing_matrix)(spins)
    return full,occupied


def save_rank_checkpoint(state, optimizer, samples, hamiltonian, out, iteration, nprobe=128):
    """Inspect post-update parameters; probe samples retain their sampling provenance."""
    directory=Path(out)/'checkpoints'/f'iteration_{iteration:06d}'
    directory.mkdir(parents=True,exist_ok=False)
    state.save(directory/'model.eqx')
    optimizer.save(directory/'optimizer.eqx')
    spins=np.asarray(samples.spins)
    np.save(directory/'samples.npy',spins)
    # Recompute after the update, so energy and matrix probes use identical params.
    # Fresh Samples avoids reusing any amplitudes from the preceding parameters.
    energy=np.asarray(hamiltonian.Oloc(state,jnp.asarray(spins)))
    np.save(directory/'energy_locals.npy',energy)
    half=min(nprobe//2,len(spins)//2)
    regular=np.linspace(0,len(spins)-1,half,dtype=int)
    extreme=np.argsort(abs(energy-energy.mean()))[-half:]
    chosen=np.unique(np.concatenate([regular,extreme]))
    m=state.model
    full,occupied=map(np.asarray,probe_matrices(m,jnp.asarray(spins[chosen])))
    # Verify before/after slicing use the same orbital ordering.
    indices=np.asarray(jax.vmap(fermion_idx)(jnp.asarray(spins[chosen])))
    sliced=np.stack([f[np.ix_(i,i)] for f,i in zip(full,indices)])
    np.testing.assert_allclose(sliced,occupied,atol=1e-13,rtol=1e-10)
    independent=np.asarray(m.weights)
    table=jnp.concatenate([jnp.zeros((1,m.d),m.dtype),m.weights])
    expanded=np.asarray((table[m._weight_index]*m._weight_sign[:,None])[m._mode_pairs]).transpose(2,0,1)
    envelope=np.asarray(m.envelope).ravel()[np.asarray(m._mode_pairs)]
    matrices={'independent_W':independent,'expanded_W':expanded,'enveloped_W':expanded*envelope[None],
              'full_F':full,'occupied_F':occupied}
    arrays=dict(probe_indices=chosen,regular_indices=regular,extreme_indices=extreme,
                probe_spins=spins[chosen],probe_energy_locals=energy[chosen])
    report=dict(iteration=iteration,parameter_updates_completed=iteration,probe_count=len(chosen),
                pairing_features=m.d,
                sample_note='Iteration 0 samples follow initialization warmup. Later checkpoint probes were sampled before the final update; energies and matrices are recomputed with the saved post-update parameters. These are diagnostics, not re-equilibrated estimates.',
                energy_mean=float(energy.real.mean()),energy_variance=float(np.mean(abs(energy-energy.mean())**2)),
                numerical_rank_tolerance='max(matrix dimensions) * float64 epsilon * largest singular value')
    for key,a in matrices.items():
        s=np.linalg.svd(a,compute_uv=False)
        arrays[key+'_singular_values']=s
        report[key]=singular_summary(s,a.shape[-2:])
    # Keep the actual probe matrices to inspect near-singular events independently.
    arrays['full_F']=full;arrays['occupied_F']=occupied
    np.savez_compressed(directory/'rank_diagnostics.npz',**arrays)
    (directory/'rank_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Rank checkpoint:',json.dumps(report),flush=True)
    return report
