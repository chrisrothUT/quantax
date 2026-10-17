"""Create an executed notebook after momentum measurement, for arbitrary size and U."""
from pathlib import Path
import argparse
import json
import nbformat as nbf
from nbclient import NotebookClient
from charge_structure_factor import compute, plot_charge


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--learning-rate',type=float,default=0.03)
    parser.add_argument('--training-dir',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    run=args.run_root.resolve()
    directory=run/'observables'
    training_dir=args.training_dir.resolve() if args.training_dir else run/'train'/f'lr_{args.learning_rate:g}'
    config=json.loads((training_dir/'config.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    from energy_variance import compute as compute_variance
    compute_variance(directory)
    charge=compute(directory)
    n,U=config['sites'],config['U']
    title=f'N={n}, U={U:g}: connected charge structure factor'
    fig=plot_charge(charge,title=title)
    fig.savefig(directory/'charge_structure_factor.png',dpi=180)
    import matplotlib.pyplot as plt
    plt.close(fig)
    nb=nbf.v4.new_notebook()
    nb.metadata.kernelspec=dict(display_name='Python (quantax_new)',language='python',name='quantax_new')
    md=nbf.v4.new_markdown_cell
    code=nbf.v4.new_code_cell
    optimizer_description=(f"norm-constrained AdamSR, C0={config['C0']}, t0={config['constraint_t0']}, T={config['constraint_T']}; unit learning rate, auto_pinv_eig" if 'C0' in config else f"AdamSR lr={config['learning_rate']}")
    optimizer_description += '; constant norm cap' if config.get('constant_norm') else ''
    optimizer_description += '; 70% whole-site / 30% single-spin exchange attempts' if config.get('sampler') == 'mixed_exchange' else ''
    nb.cells=[md(f'''# N={n}, U={U:g}: training, spin, occupations and charge

TriangularB({config['L']}), half filling ({config['electrons']}), t=1; {optimizer_description}, {config['iterations']} iterations, {config['samples']} samples, {config['nblocks']} blocks and {config['channels']} CNN channels, {config['pairing_features']} pairing features. Fresh seed {config['seed']}.

Frozen-state measurements: {summary['measurement_blocks']} blocks of {summary['samples_per_block']} samples. Error bars are block SEMs; residual correlation can make them too small. A single optimization run does not establish ground-state convergence. Blackwell nodes are excluded. All figures below are rendered, and rerunning this notebook uses saved data on CPU.'''),
    code(f'''%matplotlib inline
from pathlib import Path
import sys, json
import numpy as np
import matplotlib.pyplot as plt
root=Path({str(root)!r})
sys.path.insert(0,str(root))
from momentum_panels import plot_momentum
from charge_structure_factor import plot_charge
run=Path({str(run)!r})
training_dir=Path({str(training_dir)!r})
data_dir=run/'observables'
data=dict(np.load(data_dir/'observables.npz'))
charge=dict(np.load(data_dir/'charge_structure_factor.npz'))
summary=json.loads((data_dir/'summary.json').read_text())
print('E/site:',summary['energy_per_site'],'±',summary['energy_per_site_sem'])
print('Var(H):',summary['energy_variance'],'±',summary['energy_variance_sem'], 't²')
print('Var(H)/N:',summary['energy_variance_per_site'],'±',summary['energy_variance_per_site_sem'], 't²')
print('Var(H/N):',summary['energy_per_site_variance'], 't²')
print('Total spin squared:',summary['total_spin_squared'],'±',summary['total_spin_squared_sem'])'''),
    md(r'''Energy variance is $\mathrm{Var}(H)=\langle |E_{\rm loc}-\langle E_{\rm loc}\rangle|^2\rangle$, computed from all raw, unclipped local energies at the frozen checkpoint. It vanishes for an exact eigenstate. We report both $\mathrm{Var}(H)/N$ and $\mathrm{Var}(H/N)=\mathrm{Var}(H)/N^2$ to distinguish their normalizations. Variance errors use a delete-one-block jackknife; residual block correlation can underestimate errors. A small error on the mean energy does not imply a small energy variance.'''),
    md('## Training history\nThe running mean uses 25 iterations. A flat curve alone does not establish convergence.'),
    code('''history=np.genfromtxt(training_dir/'history.csv',delimiter=',',names=True)
x,y=history['iteration'],history['energy_per_site']
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for ax in axes:
    ax.plot(x,y,alpha=.25,lw=.6)
    ax.plot(x[24:],np.convolve(y,np.ones(25)/25,mode='valid'))
    ax.set(xlabel='Iteration',ylabel='Energy per site')
    ax.grid(alpha=.2)
axes[1].set_xlim(max(1,x[-1]-199),x[-1])
plt.show()
print('Last 200 iterations mean E/site:',y[-200:].mean())'''),
    md(r'''## Spin structure factor and momentum occupations
$S(\mathbf q)=N^{-1}\sum_{ij}e^{i\mathbf q\cdot(\mathbf r_i-\mathbf r_j)}\langle\mathbf S_i\cdot\mathbf S_j\rangle$ includes transverse terms explicitly.
$n_\sigma(\mathbf k)=\langle c^\dagger_{\mathbf k\sigma}c_{\mathbf k\sigma}\rangle$, with $c_{\mathbf k\sigma}=N^{-1/2}\sum_j e^{-i\mathbf k\cdot\mathbf r_j}c_{j\sigma}$. Occupations are per spin, with physical range 0–1. The panels share the Brillouin zone and up/down color scale. Equivalent boundary points are repeated for display; no point-group averaging is imposed.'''),
    code('''fig=plot_momentum(data)
plt.show()
np.testing.assert_allclose(data['n_up_q'].sum(),summary['sites']/2,atol=1e-9)
np.testing.assert_allclose(data['n_down_q'].sum(),summary['sites']/2,atol=1e-9)
np.testing.assert_allclose(data['S_q'].sum(),summary['sites']*summary['mean_local_moment_squared'],atol=1e-9)
print('Occupation sums:', data['n_up_q'].sum(),data['n_down_q'].sum())
print('Peak S(q):',summary['peak_S_q'],'±',summary['peak_S_q_sem'])
print('Peak momentum:',summary['peak_q_cartesian'])'''),
    md(r'''## Charge structure factor versus momentum magnitude
$N(\mathbf k)=N^{-1}\langle|\sum_j e^{-i\mathbf k\cdot\mathbf r_j}(n_{j\uparrow}+n_{j\downarrow}-1)|^2\rangle$ uses the exact uniform density of the translation-invariant half-filled state. It is computed directly from saved configurations. Fixed particle number enforces $N(0)=0$ in both phases.

The usual low-momentum diagnostic is linear $N(k)\propto |k|$ for a conventional metal and quadratic $N(k)\propto k^2$ for a charge-gapped insulator; see [Rüger et al., section 3.1](https://arxiv.org/html/1311.6504). A single finite cluster cannot establish asymptotic scaling, and a gapless charge response alone need not identify a conventional metal. Compare the smallest momenta across system sizes and assess optimization convergence.

Gray points retain directions. Each blue point averages the per-configuration squared Fourier charge amplitude over all saved samples and all momenta in one equal-length shell. Equal-size blocks make this exactly the pooled sample mean. Shell sizes vary; no smoothing or fit is applied. Connecting shells can look bumpy because nearby radii can represent different directions on this finite lattice. Radial means average equal-length momenta inside each block before estimating SEM, preserving covariance. Directional spread is not included in SEM. Linear and quadratic guides pass through the first shell; they are not fits. Ratios omit k=0. The charge Parseval sum rule is checked for every configuration.'''),
    code(f'''fig=plot_charge(charge,title={title!r})
plt.show()
np.testing.assert_allclose(charge['N_q'].sum(),2*summary['sites']*summary['mean_double_occupancy'],atol=1e-9)
print('Charge sum rule:',charge['N_q'].sum())
print('Smallest nonzero |k|:',charge['shell_k'][1])
print(' k_abs, N(k), SEM, N(k)/k, N(k)/k²')
for k,y,e in zip(charge['shell_k'][1:],charge['shell_N'][1:],charge['shell_sem'][1:]):
    print(f'{{k:.6f}}  {{y:.7f}}  {{e:.7f}}  {{y/k:.7f}}  {{y/k**2:.7f}}')'''),
    md('Numerical data and SEMs: `observables/momentum_observables.csv`, `observables/charge_structure_factor.csv`, and `observables/charge_radial_shells.csv` under this run directory. Checkpoints, samples and history are retained. Charge and momentum PNGs are saved alongside the CSVs.')]
    path=run/'observables.ipynb'
    nbf.validate(nb)
    nbf.write(nb,path)
    try:
        NotebookClient(nb,timeout=180,kernel_name='quantax_new',resources={'metadata':{'path':str(root)}}).execute()
    finally:
        nbf.write(nb,path)
    errors=[o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
    images=sum('image/png' in o.get('data',{}) for c in nb.cells if c.cell_type=='code' for o in c.outputs)
    assert not errors,errors
    assert images==3,images
    nbf.validate(nb)
    print('Saved executed notebook:',path,flush=True)

if __name__=='__main__': main()
