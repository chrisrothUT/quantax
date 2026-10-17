"""Exact per-chain attempt/change/accept counters for the mixed Metropolis kernel.

Preserves candidate keys, free no-op steps, stopping times and acceptance draws.
Counts columns are [site_attempt, site_changed, site_accepted,
                   single_attempt, single_changed, single_accepted].
"""
import csv
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from quantax.sampler import Samples
from quantax.utils import chunk_map
from mixed_sampler import MixedExchange


class DiagnosticMixedExchange(MixedExchange):
    @eqx.filter_jit
    def _sweep_variational(self, key, nsweeps, spins, model):
        assert not self.use_ref
        nsweeps=jnp.asarray(nsweeps,jnp.int32)
        samples=Samples(spins,self._state._fulljit_forward(model,spins))
        batch=spins.shape[0]
        K=int(np.ceil(np.log2(batch)))+3
        steps=jnp.zeros(batch,jnp.int32)
        counts=jnp.zeros((batch,6),jnp.int32)
        def condition(carry):
            return ~jnp.all(carry[2]>=nsweeps)
        def body(carry):
            key,samples,steps,counts=carry
            key,kpropose=jr.split(key)
            candidates,kinds=jax.vmap(lambda k:self.propose_with_kind(k,samples.spins))(jr.split(kpropose,K))
            changed=jnp.any(candidates!=samples.spins[None],axis=2)
            any_changed=jnp.any(changed,axis=0)
            first=jnp.argmax(changed,axis=0).astype(steps.dtype)
            remaining=nsweeps-steps
            free=jnp.minimum(jnp.where(any_changed,first,K),remaining)
            take=any_changed & (first<remaining)
            selected=jnp.take_along_axis(candidates,first[None,:,None],axis=0)[0]
            new_spins=jnp.where(take[:,None],selected,samples.spins)
            selected_kind=jnp.take_along_axis(kinds,first[None],axis=0)[0]
            used=jnp.arange(K)[:,None] < (free+take)[None]
            site_attempt=jnp.sum(used & kinds,axis=0,dtype=jnp.int32)
            site_changed=take & selected_kind
            single_changed=take & ~selected_kind
            new_samples=Samples(new_spins,self._state._fulljit_forward(model,new_spins))
            key,updated=self._update(key,None,samples,new_samples)
            accepted=jnp.any(updated.spins!=samples.spins,axis=1)
            increment=jnp.stack([site_attempt,site_changed,accepted & selected_kind,free+take-site_attempt,single_changed,accepted & ~selected_kind],axis=1)
            return key,updated,steps+free+take,counts+increment
        _,samples,_,counts=jax.lax.while_loop(condition,body,(key,samples,steps,counts))
        return Samples(samples.spins,samples.psi,counts)

    def sweep(self, nsweeps=None):
        if nsweeps is None:
            nsweeps=self._sweep_steps
        if nsweeps<=0:
            return Samples(self._spins,None,None,self._reweight_factor)
        chunk_size=self._state.forward_chunk
        if chunk_size is not None and chunk_size < self.nsamples//jax.device_count():
            fn=chunk_map(self._partial_sweep,in_axes=(None,0),chunk_size=chunk_size)
            result=fn(nsweeps,self._spins)
        else:
            result=self._partial_sweep(nsweeps,self._spins)
        self._spins=result.spins
        self._psi=result.psi
        self._reweight_factor=self._get_reweight_factor(result.psi)
        self.last_counts=np.asarray(result.state_internal)
        assert np.all(self.last_counts[:,[0,3]].sum(axis=1)==nsweeps)
        return Samples(self._spins,None,None,self._reweight_factor)


def save_diagnostics(sampler, directory, label, energy=None, keep_spins=False):
    if not hasattr(sampler,'last_counts'):
        return
    directory=Path(directory)/'sampler_diagnostics'
    directory.mkdir(exist_ok=True)
    counts=sampler.last_counts
    payload=dict(counts=counts)
    if keep_spins:
        payload['spins']=np.asarray(sampler._spins)
    if energy is not None:
        payload['energy_locals']=np.asarray(energy)
    # Keep chain identity and endpoint state to trace rare large-energy events.
    np.savez_compressed(directory/f'{label}.npz',**payload)
    total=counts.sum(axis=0)
    row={'label':label}
    for offset,kind in [(0,'site'),(3,'single')]:
        attempts,changed,accepted=map(int,total[offset:offset+3])
        row.update({f'{kind}_attempts':attempts,f'{kind}_noops':attempts-changed,
                    f'{kind}_changed_proposals':changed,f'{kind}_accepted_changes':accepted,
                    f'{kind}_acceptance_given_change':accepted/changed if changed else float('nan'),
                    f'{kind}_accepted_changes_per_attempt':accepted/attempts if attempts else float('nan')})
    row['chains_with_no_accepted_change']=int(np.sum(counts[:,[2,5]].sum(axis=1)==0))
    if energy is not None:
        e=np.asarray(energy)
        row.update(energy_mean=float(e.real.mean()),energy_variance=float(np.mean(abs(e-e.mean())**2)),energy_min=float(e.real.min()),energy_max=float(e.real.max()))
    path=directory/'sweeps.csv'
    # Separate files for energy-bearing records keep the CSV schema stable.
    if energy is not None:
        path=directory/'energy_sweeps.csv'
    exists=path.exists()
    with path.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if not exists:writer.writeheader()
        writer.writerow(row)
    print('Sampler diagnostics:',row,flush=True)
