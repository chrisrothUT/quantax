"""GPU audit of frozen-state sampler versus observable forward paths."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse, hashlib, inspect, json, time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import quantax as q
from quantax.utils import to_distributed_array


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    jax.config.update('jax_enable_x64',True)
    jax.config.update('jax_default_matmul_precision','highest')
    q.set_default_dtype(jnp.float64)
    q.set_random_seed(73421)
    assert all(d.platform=='gpu' for d in jax.devices())
    run=args.run_root/'train/C0_1'
    c=json.loads((run/'config.json').read_text())
    assert hashlib.sha256(Path(inspect.getfile(q.model.MultiplicativePfBackflow)).read_bytes()).hexdigest()==c['model_sha256']
    q.sites.TriangularB(c['L'],particle_type='spinful_fermion',Nparticles=tuple(c['electrons']))
    m=q.model.MultiplicativePfBackflow.resconv(d=c['pairing_features'],nblocks=c['nblocks'],channels=c['channels'],kernel_size=c['kernel_size'],dtype=jnp.float32,pairing_dtype=jnp.float64)
    state=q.state.Variational(m,param_file=run/'latest.eqx',max_parallel=(1024,128,1024))
    assert not state.use_ref
    report=dict(run=str(run),devices=[d.device_kind for d in jax.devices()],use_ref=state.use_ref,checks=[])
    started=time.monotonic()
    def compare(label,a,b):
        sa,la,sb,lb=map(np.asarray,(a.sign,a.logabs,b.sign,b.logabs))
        finite=np.isfinite(la)&np.isfinite(lb)
        delta=la[finite]-lb[finite]
        result=dict(label=label,count=len(la),sign_mismatches=int(np.sum(sa!=sb)),nonfinite_a=int(np.sum(~np.isfinite(la))),nonfinite_b=int(np.sum(~np.isfinite(lb))),max_abs_log_difference=float(np.max(abs(delta))) if len(delta) else None,rms_log_difference=float(np.sqrt(np.mean(delta**2))) if len(delta) else None)
        report['checks'].append(result)
        print(json.dumps(result),flush=True)
        return result
    @eqx.filter_jit
    def fused(model,old,new):
        a=state._fulljit_forward(model,old)
        b=state._fulljit_forward(model,new)
        probability=jnp.minimum(1.,jnp.abs(jnp.asarray(b/a))**2)
        return a,b,probability
    def proposals(label,old,seed):
        old=to_distributed_array(jnp.asarray(old))
        new=sampler.propose(jr.PRNGKey(seed),old)
        a,b,p=fused(state.model,old,new)
        da,db=state(old),state(new)
        compare(label+'_old',a,da);compare(label+'_proposed',b,db)
        direct_p=jnp.minimum(1.,jnp.abs(jnp.asarray(db/da))**2)
        p,pd=map(np.asarray,(p,direct_p))
        u=np.asarray(jr.uniform(jr.PRNGKey(seed+1),p.shape,dtype=jnp.float64))
        result=dict(label=label+'_acceptance',count=len(p),changed_proposals=int(np.sum(np.any(np.asarray(old)!=np.asarray(new),axis=1))),nonfinite_probabilities=int(np.sum(~np.isfinite(p)|~np.isfinite(pd))),max_abs_probability_difference=float(np.max(abs(p-pd))),decision_disagreements=int(np.sum((u<p)!=(u<pd))))
        report['checks'].append(result);print(json.dumps(result),flush=True)
        np.savez_compressed(args.output/(label+'.npz'),old=np.asarray(old),new=np.asarray(new),sampler_old_log=np.asarray(a.logabs),sampler_new_log=np.asarray(b.logabs),direct_old_log=np.asarray(da.logabs),direct_new_log=np.asarray(db.logabs),sampler_probability=p,direct_probability=pd,uniform=u)
    saved=np.load(run/'samples.npy')
    # No warmup: examine saved configurations first; then exercise actual sweep.
    sampler=q.sampler.ParticleHop(state,nsamples=len(saved),initial_spins=jnp.asarray(saved),thermal_steps=0,sweep_steps=c['sweep_steps'])
    for i in range(4):
        proposals('saved_proposals_'+str(i),saved,9100+10*i)
    sampler.sweep()
    compare('retained_after_432_attempts',sampler._psi,state(sampler._spins))
    np.savez_compressed(args.output/'retained.npz',spins=np.asarray(sampler._spins),retained_sign=np.asarray(sampler._psi.sign),retained_log=np.asarray(sampler._psi.logabs))
    proposals('post_sweep_proposals',sampler._spins,9300)
    # Explicitly stress the configurations with the largest saved energy deviations.
    directory=args.run_root/'observables'
    energy=np.load(directory/'energy_locals.npy').ravel()
    configurations=np.load(directory/'measurement_samples.npy',mmap_mode='r').reshape(-1,2*c['sites'])
    idx=np.argsort(abs(energy-energy.mean()))[-1024:]
    proposals('largest_energy_deviations',configurations[idx],9400)
    report['seconds']=time.monotonic()-started
    report['source_sha256']={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in [('model',inspect.getfile(type(state.model))),('sampler',inspect.getfile(q.sampler.ParticleHop)),('metropolis',inspect.getfile(q.sampler.Metropolis)),('state',inspect.getfile(type(state)))]}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('COMPLETE',json.dumps(report),flush=True)

if __name__=='__main__':main()
