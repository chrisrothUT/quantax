"""Frozen initial-checkpoint variance and Pfaffian scale/conditioning audit."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse,hashlib,inspect,json,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import fermix
import quantax as q
from quantax.nn import fermion_idx


def quantiles(x):
    return dict(zip(['min','p01','p50','p99','max'],map(float,np.quantile(np.asarray(x),[0,.01,.5,.99,1]))))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sites',type=int,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    jax.config.update('jax_enable_x64',True);jax.config.update('jax_default_matmul_precision','highest');q.set_default_dtype(jnp.float64)
    assert all(d.platform=='gpu' for d in jax.devices())
    n=args.sites
    if n==48:run=Path('n48_u7_u9_C0_1_s4096_20261015_225047/u7_n48_C0_1_s4096/train/C0_1')
    else:run=Path('n108_u7_u9_median_mad_20261016_013200/u7_n108_medianMAD_C0_1_s4096/train/C0_1')
    c=json.loads((run/'config.json').read_text())
    assert hashlib.sha256(Path(inspect.getfile(q.model.MultiplicativePfBackflow)).read_bytes()).hexdigest()==c['model_sha256']
    out=args.output/f'n{n}';out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();q.set_random_seed(73421)
    lattice=q.sites.TriangularB(c['L'],particle_type='spinful_fermion',Nparticles=tuple(c['electrons']))
    m=q.model.MultiplicativePfBackflow.resconv(d=16,nblocks=8,channels=16,kernel_size=3,dtype=jnp.float32,pairing_dtype=jnp.float64)
    state=q.state.Variational(m,param_file=run/'initial.eqx',max_parallel=(1024,128,1024));m=state.model
    H=q.operator.Hubbard(U=7.)
    sampler=q.sampler.ParticleHop(state,nsamples=4096,thermal_steps=40*n,sweep_steps=4*n)
    energies=[];logs=[];configs=[];doublons=[]
    for b in range(12):
        samples=sampler.sweep()
        e=np.asarray(H.Oloc(state,samples));spins=np.asarray(samples.spins)
        d=((spins[:,:n]>0)&(spins[:,n:]>0)).sum(axis=1)
        energies.append(e);doublons.append(d);logs.append(np.asarray(state(samples.spins).logabs));configs.append(spins)
        print(json.dumps(dict(n=n,block=b+1,energy_per_site=float(e.mean()/n),variance=float(np.var(e)),seconds=time.monotonic()-started)),flush=True)
    energies=np.asarray(energies);doublons=np.asarray(doublons);logs=np.asarray(logs);configs=np.asarray(configs)
    np.savez_compressed(out/'initial_measurements.npz',energy_U7=energies,energy_U9=energies+2*doublons,logabs=logs,spins=configs,doublons=doublons)
    from energy_variance import energy_statistics
    report=dict(n=n,checkpoint=str(run/'initial.eqx'),devices=[d.device_kind for d in jax.devices()],blocks=12,samples_per_block=4096)
    for u,e in [(7,energies),(9,energies+2*doublons)]:
        report[f'U{u}']=dict(energy_per_site=float(e.mean()/n),energy_per_site_sem=float(e.mean(axis=1).std(ddof=1)/np.sqrt(len(e))/n),**energy_statistics(e,n))
    # Evenly sample the full measurement history; also inspect uniform sector states.
    flat=configs.reshape(-1,2*n);chosen=jnp.asarray(flat[np.linspace(0,len(flat)-1,128,dtype=int)])
    random=q.utils.rand_states(128)
    @eqx.filter_jit
    def matrix_stats(model,spins):
        f=jax.vmap(model.pairing_matrix)(spins)
        y=jax.vmap(model.net)(spins).reshape(len(spins),model._nlabels,model.d,lattice.ncells).transpose(0,1,3,2).reshape(len(spins),2*n,model.d)
        return f,y
    for label,spins in [('sampled',chosen),('uniform_sector',random)]:
        f,y=map(np.asarray,matrix_stats(m,spins))
        singular=np.linalg.svd(f,compute_uv=False)
        psi=state(spins)
        sign,logdet=np.linalg.slogdet(f)
        condition=singular[:,0]/singular[:,-1]
        report[label]=dict(feature_norm=quantiles(np.linalg.norm(y,axis=-1)),feature_rms=float(np.sqrt(np.mean(y*y))),matrix_abs=quantiles(abs(f).ravel()),matrix_max_abs=quantiles(abs(f).max(axis=(1,2))),condition_number=quantiles(condition),logabs=quantiles(psi.logabs),max_abs_state_2logabs_vs_separately_compiled_matrix_logdet=float(np.max(abs(2*np.asarray(psi.logabs)-logdet))),max_abs_2logpf_minus_logdet=float(np.max(abs(2*np.asarray(fermix.slogpf(jnp.asarray(f))[1])-logdet))),negative_det_count=int((sign<0).sum()))
        np.savez(out/f'{label}_matrices.npz',matrices=f,singular_values=singular,features=y)
        if label=='sampled':
            # Global and row scaling must cancel after correcting log-Pfaffian.
            fdev=jnp.asarray(f)
            reference_sign,reference_log=fermix.slogpf(fdev)
            tests=[]
            for scale in [1e-8,1e8]:
                ss,ll=fermix.slogpf(fdev*scale)
                tests.append(dict(scale=scale,sign_mismatches=int(np.sum(np.asarray(ss)!=np.asarray(reference_sign))),max_corrected_log_error=float(np.max(abs(np.asarray(ll)-n/2*np.log(scale)-np.asarray(reference_log))))))
            row_scale=jnp.sqrt(jnp.sum(fdev*fdev,axis=-1))
            normalized=fdev/row_scale[:,:,None]/row_scale[:,None,:]
            ss,ll=fermix.slogpf(normalized)
            report['row_scale_correction']=dict(sign_mismatches=int(np.sum(np.asarray(ss)!=np.asarray(reference_sign))),max_corrected_log_error=float(np.max(abs(np.asarray(ll)+np.log(np.asarray(row_scale)).sum(axis=1)-np.asarray(reference_log)))))
            report['global_scale_tests']=tests
    envelope=np.asarray(m.envelope)
    report['envelope']=dict(spatial_sum=float(envelope.sum()*lattice.ncells/4),minimum=float(envelope.min()),maximum=float(envelope.max()))
    report['sampled_logabs']=quantiles(logs)
    # Proposal wavefunction ratios expose physically relevant dynamic range.
    proposal=sampler.propose(jax.random.PRNGKey(9301),jnp.asarray(configs[-1]))
    oldpsi=state(jnp.asarray(configs[-1]));newpsi=state(proposal)
    delta=np.asarray(newpsi.logabs-oldpsi.logabs)
    report['single_hop_logabs_ratio']=quantiles(delta)
    report['single_hop_nonfinite_logratio']=int((~np.isfinite(delta)).sum())
    report['seconds']=time.monotonic()-started
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('COMPLETE',json.dumps(report),flush=True)

if __name__=='__main__':main()
