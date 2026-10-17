"""Prepare a rerunnable notebook for the lowest-energy completed U=8, N=48 run."""
from pathlib import Path
import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

root = Path(__file__).resolve().parent
output = root / 'n48_u8_C0_sweep_20261015_202601' / 'best_n48_correlations.ipynb'
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = dict(display_name='Python (quantax_new)', language='python', name='quantax_new')
nb.cells = [
md(r'''# U=8, N=48: best-run correlation functions

Select the completed optimization with the lowest final evaluation energy per site, then measure its **final checkpoint**. Lowest reported energy is a selection criterion; the small differences between some candidates do not establish a statistically significant ranking.

**Measurement status:** GPU measurements have not run. This session has no GPU, and Slurm submission failed with `Unable to contact slurm controller (connect failure)`. No correlation values from a different run are substituted. The training comparison below is available now. Rerun all cells on a GPU to measure and display all three observables, or submit `run_best_n48_correlations.sbatch` from the project directory once Slurm is accessible. That job also writes a fully executed `observables.ipynb` alongside this notebook.

The measurement uses 20 fresh sampling blocks of 4096 configurations (81,920 total), initialized from the final checkpoint's saved configurations with re-equilibration. Error bars are SEMs across blocks; residual autocorrelation and optimization uncertainty are not included.'''),
code(f'''%matplotlib inline
from pathlib import Path
import json, sys, subprocess, os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
root = Path({str(root)!r})
sys.path.insert(0, str(root))
run_root = root / 'n48_u8_C0_sweep_20261015_202601'
data_dir = run_root / 'observables'
training_dir = run_root / 'runs/C0_1'
'''),
md('## Selection from completed optimizations\nFinal evaluation means are compared, rather than the minimum noisy training step. Pilot and verification runs are listed when available.'),
code('''candidates = []
for config_path in root.glob('*/**/config.json'):
    summary_path = config_path.with_name('summary.json')
    if not summary_path.exists():
        continue
    config = json.loads(config_path.read_text())
    summary = json.loads(summary_path.read_text())
    if config.get('sites') != 48 or config.get('U') != 8 or summary.get('status') != 'complete':
        continue
    energy = summary.get('final_energy_per_site')
    if energy is None or not np.isfinite(energy):
        continue
    sem = summary.get('final_block_sem', float('nan')) / 48
    candidates.append((energy, sem, config_path.parent))
candidates.sort(key=lambda item: item[0])
assert candidates and candidates[0][2] == training_dir, candidates
print('Final E/site      block SEM       completed run')
for energy, sem, directory in candidates:
    print(f'{energy: .9f}      {sem:.9f}      {directory.relative_to(root)}')
config = json.loads((training_dir / 'config.json').read_text())
assert config['samples_match_final_checkpoint']
assert (training_dir / 'latest.eqx').exists()
assert (training_dir / 'samples.npy').exists()
print('Selected:', training_dir.relative_to(root))
print('Checkpoint: latest.eqx; C0 =', config['C0'])'''),
code('''history = np.genfromtxt(training_dir / 'history.csv', delimiter=',', names=True)
x, y = history['iteration'], history['energy_per_site']
fig, ax = plt.subplots(figsize=(9, 3.5), layout='constrained')
ax.plot(x, y, alpha=.3, linewidth=.7)
ax.plot(x[24:], np.convolve(y, np.ones(25)/25, mode='valid'), label='25-step mean')
ax.set(xlabel='Iteration', ylabel='Energy per site', title='Selected optimization: C0=1')
ax.legend(); ax.grid(alpha=.2)
plt.show()'''),
md(r'''## Run frozen-checkpoint measurements

The existing estimator checks the installed model source against the training hash, checks saved particle counts, and checks the energy reconstructed from the 1-RDM and double occupancy against the direct Hamiltonian estimator. Sampling never changes model parameters. The full SU(2) spin operator has also been checked in the exact two-site singlet/triplet basis.

The next cell runs the measurement only if a GPU is available and results are absent. It reuses completed data on subsequent runs. On a CPU host, submit the provided batch script when Slurm is reachable. Partial output directories from a failed measurement must be inspected before retrying.'''),
code('''measurement_complete = (data_dir / 'summary.json').exists()
if not measurement_complete:
    import jax
    has_gpu = any(device.platform == 'gpu' for device in jax.devices())
    if has_gpu:
        subprocess.run([sys.executable, '-u', str(root/'measure_momentum.py'),
                        '--run', str(training_dir), '--output', str(data_dir),
                        '--blocks', '20'], cwd=root, check=True)
        measurement_complete = True
    else:
        print('PENDING: GPU measurements have not run. This kernel has no GPU.')
        print('From the project directory, submit: sbatch run_best_n48_correlations.sbatch')
if measurement_complete:
    from charge_structure_factor import compute
    charge = compute(data_dir)
    data = dict(np.load(data_dir/'observables.npz'))
    summary = json.loads((data_dir/'summary.json').read_text())
    assert Path(summary['run']).name == training_dir.name
    print('Measurement E/site:', summary['energy_per_site'], '+/-', summary['energy_per_site_sem'])
    print('Configurations:', summary['measurement_samples'])'''),
md(r'''## Momentum-space 1-RDM and SU(2) spin structure factor

With $c_{k\sigma}=N^{-1/2}\sum_j e^{-ik\cdot r_j}c_{j\sigma}$, the translation-invariant 1-RDM is diagonal in momentum: $\rho_\sigma(k,k')=\delta_{kk'}n_\sigma(k)$. The estimator averages $\langle c^\dagger_{i\sigma}c_{j\sigma}\rangle$ over translations before Fourier transformation. Fixed spin populations eliminate the spin-off-diagonal 1-RDM. The real wavefunction has vanishing imaginary real-space matrix elements.

$$S(q)=\frac1N\sum_{ij}e^{iq\cdot(r_i-r_j)}\langle\mathbf S_i\cdot\mathbf S_j\rangle,$$
$$\mathbf S_i\cdot\mathbf S_j=S_i^zS_j^z+\tfrac12(S_i^+S_j^-+S_i^-S_j^+).$$

Transverse terms are measured explicitly, without assuming $S(q)=3S^{zz}(q)$. Occupations are per spin, with $\sum_k n_\sigma(k)=24$. No point-group averaging is imposed; equivalent boundary points are repeated only for display.'''),
code('''if measurement_complete:
    from momentum_panels import plot_momentum
    fig = plot_momentum(data)
    plt.show()
    blocks = dict(np.load(data_dir/'momentum_block_means.npz'))
    np.testing.assert_allclose(data['n_up_q'].sum(), 24, atol=1e-9)
    np.testing.assert_allclose(data['n_down_q'].sum(), 24, atol=1e-9)
    np.testing.assert_allclose(data['S_q'].sum(), 48*summary['mean_local_moment_squared'], atol=1e-9)
    print('Peak S(q):', summary['peak_S_q'], '+/-', summary['peak_S_q_sem'])
    print('Peak q:', summary['peak_q_cartesian'])
    print('<S_total²>:', summary['total_spin_squared'], '+/-', summary['total_spin_squared_sem'])
    print('kx, ky, n_up, SEM_up, n_down, SEM_down, S(q), SEM_S')
    display(np.column_stack([data['q_fold'], data['n_up_q'], data['n_up_q_sem'],
                             data['n_down_q'], data['n_down_q_sem'], data['S_q'], data['S_q_sem']]))
else:
    print('PENDING: momentum 1-RDM and SU(2) spin measurements.')'''),
md(r'''## Connected charge structure factor

For this translation-invariant half-filled state, $\langle n_j\rangle=1$:
$$N(q)=\frac1N\left\langle\left|\sum_j e^{-iq\cdot r_j}(n_{j\uparrow}+n_{j\downarrow}-1)\right|^2\right\rangle.$$

Compute this diagonal observable from the same saved configurations. Fixed particle number gives $N(0)=0$. The Parseval sum rule is checked for every configuration, and $\sum_qN(q)=2N\langle n_\uparrow n_\downarrow\rangle$ is checked below. Shell error bars preserve covariance between momenta by averaging within each block first. The small-momentum ratios are descriptive on this finite cluster; the linear and quadratic guides are not fits.'''),
code('''if measurement_complete:
    from charge_structure_factor import plot_charge
    fig = plot_charge(charge, title='U=8, N=48, C0=1: connected charge structure factor')
    plt.show()
    np.testing.assert_allclose(charge['N_q'].sum(), 96*summary['mean_double_occupancy'], atol=1e-9)
    np.testing.assert_allclose(charge['N_q'][0], 0, atol=1e-20)
    print('|q|, shell N(q), block SEM')
    display(np.column_stack([charge['shell_k'], charge['shell_N'], charge['shell_sem']]))
    print('Data:', data_dir)
    print('Momentum: momentum_observables.csv; charge: charge_structure_factor.csv')
else:
    print('PENDING: charge structure factor from fresh measurement configurations.')''')]
nbf.validate(nb)
# Execute in-process: this managed session does not permit kernel sockets.
shell = InteractiveShell.instance()
shell.enable_gui = lambda gui=None: None
execution_count = 0
for cell in nb.cells:
    if cell.cell_type != 'code':
        continue
    execution_count += 1
    with capture_output() as captured:
        result = shell.run_cell(cell.source)
    if result.error_before_exec or result.error_in_exec:
        raise RuntimeError(captured.stdout + captured.stderr) from (result.error_before_exec or result.error_in_exec)
    cell.execution_count = execution_count
    cell.outputs = []
    for name, content in [('stdout', captured.stdout), ('stderr', captured.stderr)]:
        if content:
            cell.outputs.append(nbf.v4.new_output('stream', name=name, text=content))
    for rich in captured.outputs:
        cell.outputs.append(nbf.v4.new_output('display_data', data=rich.data, metadata=rich.metadata))
nbf.validate(nb)
nbf.write(nb, output)
print(output)
