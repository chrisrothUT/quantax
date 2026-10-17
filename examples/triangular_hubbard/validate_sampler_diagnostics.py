"""Ensure counters preserve the original GPU trajectory and exact step counts."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import quantax as q
from quantax.utils import to_distributed_array
from mixed_sampler import MixedExchange
from sampler_diagnostics import DiagnosticMixedExchange
jax.config.update('jax_enable_x64',True)
jax.config.update('jax_default_matmul_precision','highest')
q.set_default_dtype(jnp.float64)
q.set_random_seed(73421)
run=Path('n108_u7_u9_median_mad_20261016_013200/u7_n108_medianMAD_C0_1_s4096/train/C0_1')
c=json.loads((run/'config.json').read_text())
q.sites.TriangularB(6,particle_type='spinful_fermion',Nparticles=(54,54))
m=q.model.MultiplicativePfBackflow.resconv(d=16,nblocks=8,channels=16,kernel_size=3,dtype=jnp.float32,pairing_dtype=jnp.float64)
s=q.state.Variational(m,param_file=run/'latest.eqx',max_parallel=(1024,128,1024))
x=to_distributed_array(jnp.asarray(np.load(run/'samples.npy')))
a=MixedExchange(s,nsamples=len(x),initial_spins=x,thermal_steps=0)
b=DiagnosticMixedExchange(s,nsamples=len(x),initial_spins=x,thermal_steps=0)
key=jax.random.PRNGKey(8133)
plain=a._sweep_variational(key,np.asarray(7,np.int32),x,s.model)
counted=b._sweep_variational(key,np.asarray(7,np.int32),x,s.model)
np.testing.assert_array_equal(plain.spins,counted.spins)
np.testing.assert_array_equal(plain.psi.sign,counted.psi.sign)
np.testing.assert_array_equal(plain.psi.logabs,counted.psi.logabs)
counts=np.asarray(counted.state_internal)
np.testing.assert_array_equal(counts[:,0]+counts[:,3],np.full(len(x),7))
for i in [0,3]:
 assert np.all(counts[:,i]>=counts[:,i+1])
 assert np.all(counts[:,i+1]>=counts[:,i+2])
print('PASS: diagnostic kernel exactly preserves configurations and amplitudes; every chain has exactly 7 attempts and consistent per-type counts on',jax.devices(),flush=True)
