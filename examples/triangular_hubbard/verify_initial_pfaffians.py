"""Isolate Pfaffian arithmetic from CNN batch/compiler rounding."""
from pathlib import Path
import json
import jax
jax.config.update('jax_enable_x64',True)
jax.config.update('jax_default_matmul_precision','highest')
import jax.numpy as jnp
import numpy as np
import fermix
assert all(d.platform=='gpu' for d in jax.devices())
for p in Path('initialization_audit_7254404').glob('n*/report.json'):
 r=json.loads(p.read_text())
 for label in ['sampled','uniform_sector']:
  f=np.load(p.parent/f'{label}_matrices.npz')['matrices']
  sign,logpf=fermix.slogpf(jnp.asarray(f));_,ld=np.linalg.slogdet(f)
  key='max_abs_2logpf_minus_logdet'
  r[label]['max_abs_state_2logabs_vs_separately_compiled_matrix_logdet']=r[label].pop(key)
  r[label][key]=float(np.max(abs(2*np.asarray(logpf)-ld)))
  print(p.parent.name,label,'matrix-only discrepancy',r[label][key],flush=True)
 r['matrix_comparison_note']='CNN features in matrix_stats were evaluated as a batch of 128, whereas state evaluates fixed chunks of 1024. State-versus-matrix differences include CNN/compiler rounding. Matrix-only check evaluates Fermix and NumPy slogdet on exactly identical saved float64 matrices.'
 p.write_text(json.dumps(r,indent=2)+'\n')
