"""Singular spectra of independent W, expanded W_alpha and occupied F(x)."""
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
jax.config.update('jax_default_matmul_precision','highest')
import jax.numpy as jnp
import equinox as eqx
import quantax as q


def spectrum(a):
    s=np.linalg.svd(a,compute_uv=False)
    tol=s[...,0]*max(a.shape[-2:])*np.finfo(s.dtype).eps
    return s,dict(shape=list(a.shape),rank=np.sum(s>tol[...,None],axis=-1).tolist(),condition=(s[...,0]/s[...,-1]).tolist(),smax=s[...,0].tolist(),smin=s[...,-1].tolist(),relative_smin=(s[...,-1]/s[...,0]).tolist(),stable_rank=(np.sum(s*s,axis=-1)/s[...,0]**2).tolist(),rank_relative_1e8=np.sum(s>s[...,0,None]*1e-8,axis=-1).tolist(),rank_relative_1e12=np.sum(s>s[...,0,None]*1e-12,axis=-1).tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    c=json.loads((args.run/'config.json').read_text());n=c['sites'];q.set_default_dtype(jnp.float64);q.set_random_seed(3232)
    q.sites.TriangularB(c['L'],particle_type='spinful_fermion',Nparticles=tuple(c['electrons']))
    model=q.model.MultiplicativePfBackflow.resconv(d=c['pairing_features'],nblocks=c['nblocks'],channels=c['channels'],kernel_size=c['kernel_size'],dtype=jnp.float32,pairing_dtype=jnp.float64)
    out=args.output/args.label;out.mkdir(parents=True,exist_ok=False)
    summaries={}
    for checkpoint in ['initial','latest']:
        path=args.run/f'{checkpoint}.eqx'
        raw=path.read_bytes();(out/f'{checkpoint}.eqx').write_bytes(raw)
        state=q.state.Variational(model,param_file=out/f'{checkpoint}.eqx',max_parallel=(128,128,128));m=state.model
        arrays={};summary={'checkpoint_sha256':hashlib.sha256(raw).hexdigest(),'n':n,'U':c['U']}
        # The independent trainable coefficients are a tall matrix, not the
        # occupied skew matrix whose Pfaffian is evaluated.
        arrays['independent_s'],summary['independent']=spectrum(np.asarray(m.weights))
        table=jnp.concatenate([jnp.zeros((1,m.d),m.dtype),m.weights])
        expanded=np.asarray((table[m._weight_index]*m._weight_sign[:,None])[m._mode_pairs]).transpose(2,0,1)
        arrays['expanded_s'],summary['expanded']=spectrum(expanded)
        pfull=np.asarray(m.envelope).ravel()[np.asarray(m._mode_pairs)]
        arrays['enveloped_s'],summary['enveloped']=spectrum(expanded*pfull[None])
        initial_probe=np.load(f'initialization_audit_7254404/n{n}/initial_measurements.npz')['spins'].reshape(-1,2*n)
        latest_samples=np.load(args.run/'samples.npy')
        for name,samples in [('initial_probe',initial_probe),('checkpoint_samples',latest_samples)]:
            samples=samples[np.linspace(0,len(samples)-1,128,dtype=int)]
            matrices=np.asarray(eqx.filter_jit(jax.vmap(m.pairing_matrix))(jnp.asarray(samples)))
            arrays[name+'_s'],summary[name]=spectrum(matrices)
            arrays[name+'_spins']=samples
        np.savez_compressed(out/f'{checkpoint}_spectra.npz',**arrays)
        summaries[checkpoint]=summary
        print(args.label,checkpoint,'independent',summary['independent'],'expanded_rank_range',np.min(summary['expanded']['rank']),np.max(summary['expanded']['rank']),'F_median_cond',np.median(summary['checkpoint_samples']['condition']),'F_max_cond',np.max(summary['checkpoint_samples']['condition']),flush=True)
    (out/'report.json').write_text(json.dumps(summaries,indent=2)+'\n')

if __name__=='__main__':main()
