"""Create and execute the completed 48-site momentum-observable notebook."""

from pathlib import Path
import json
import textwrap
import nbformat as nbf
from nbclient import NotebookClient

root = Path("/mnt/ceph/users/croth/HubbardTriangularNewQuantax")
run_root = root / "u8_n48_lr003_20261015"
measurement = run_root / "observables"
summary = json.loads((measurement / "summary.json").read_text())
training = json.loads((run_root / "train/lr_0.03/summary.json").read_text())
config = json.loads((run_root / "train/lr_0.03/config.json").read_text())
path = root / "n48_momentum_observables.ipynb"
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = dict(
    display_name="Python (quantax_new)", language="python", name="quantax_new"
)
nb.metadata.language_info = dict(name="python", version="3.12.4")
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))


md(r"""
# N=48: spin structure factor and momentum occupations

**TriangularB(4), U=8, t=1, 24 up + 24 down electrons.**
AdamSR learning rate 0.03; 1024 samples per iteration; 1000 iterations;
8 residual blocks; 16 CNN channels; 16 pairing features per spin-orbital;
real float32 CNN and float64 pairing. Fresh initialization with seed 3232.

All plots are already rendered. Select **Python (quantax_new)** and **Run All Cells** to regenerate them from saved results. This notebook runs on CPU and does not retrain or submit jobs.

After training, the final state was frozen and the saved chains were re-equilibrated. Measurements use 20 sweeps of 1024 samples (20,480 total). Error bars are SEMs of the 20 sweep means and can underestimate uncertainty if sweeps remain correlated.
""")
code(r"""
%matplotlib inline
from pathlib import Path
import csv, json, sys
import numpy as np
import matplotlib.pyplot as plt

root = Path.cwd()
if not (root / 'u8_n48_lr003_20261015').is_dir():
    root = Path('/mnt/ceph/users/croth/HubbardTriangularNewQuantax')
sys.path.insert(0,str(root))
from momentum_panels import plot_momentum
run_root = root / 'u8_n48_lr003_20261015'
data_dir = run_root / 'observables'
data = dict(np.load(data_dir / 'observables.npz'))
summary = json.loads((data_dir / 'summary.json').read_text())
training = json.loads((run_root / 'train/lr_0.03/summary.json').read_text())
config = json.loads((run_root / 'train/lr_0.03/config.json').read_text())
print(f'Training: {training["iterations"]} iterations, {config["sites"]} sites, {config["parameters"]:,} parameters')
print(f'Frozen-checkpoint E/site = {summary["energy_per_site"]:.7f} ± {summary["energy_per_site_sem"]:.7f}')
print(f'Measurement samples: {summary["measurement_samples"]:,}; allowed momenta: {len(data["S_q"])}')
""")
md("""
## Training history

The pale curve shows individual stochastic energy estimates; the solid curve is a 25-iteration running mean. A constant learning rate was used throughout. A flattening curve alone does not establish ground-state convergence.
""")
code(r"""
with (run_root / 'train/lr_0.03/history.csv').open() as f:
    history = list(csv.DictReader(f))
x = np.array([int(r['iteration']) for r in history])
y = np.array([float(r['energy_per_site']) for r in history])
window = 25
fig,axes = plt.subplots(1,2,figsize=(12,4),layout='constrained')
for ax in axes:
    ax.plot(x,y,alpha=0.25,color='C0',linewidth=0.6)
    ax.plot(x[window-1:],np.convolve(y,np.ones(window)/window,mode='valid'),color='C0')
    ax.set(xlabel='Iteration',ylabel='Energy per site')
    ax.grid(alpha=0.2)
axes[0].set_title('N=48, U=8, AdamSR lr=0.03')
axes[1].set(title='Last 200 iterations',xlim=(801,1000))
low,high=y[-200:].min(),y[-200:].max()
axes[1].set_ylim(low-0.02,high+0.02)
plt.show()
print(f'Mean E/site, iterations 601–800: {y[600:800].mean():.7f}')
print(f'Mean E/site, iterations 801–1000: {y[800:1000].mean():.7f}')
""")
md(r"""
## Momentum-space observables

\[
S(\mathbf q)=\frac1N\sum_{ij}e^{i\mathbf q\cdot(\mathbf r_i-\mathbf r_j)}
\langle\mathbf S_i\cdot\mathbf S_j\rangle,
\qquad
n_\sigma(\mathbf k)=\langle c^\dagger_{\mathbf k\sigma}c_{\mathbf k\sigma}\rangle,
\]
\[
c_{\mathbf k\sigma}=\frac1{\sqrt N}\sum_j e^{-i\mathbf k\cdot\mathbf r_j}c_{j\sigma}.
\]

Translation invariance lets us measure only the relative displacements and Fourier-transform them. The transverse spin terms are included explicitly, rather than approximating the dot product by three times its z component. Occupations are **per spin**, with physical range 0–1; they are not divided by N. Finite-sample estimates can fluctuate outside that range.

All three panels use the same momenta and first Brillouin zone. Equivalent boundary points are repeated for display. No point-group or spin-up/down averaging is imposed. Both occupation panels use the same 0–1 color scale.
""")
code(r"""
fig = plot_momentum(data, annotate=False)
plt.show()
peak = int(np.argmax(data['S_q']))
print(f'Largest S(q): {data["S_q"][peak]:.6f} ± {data["S_q_sem"][peak]:.6f}')
print(f'q = {data["q_fold"][peak]}, momentum indices = {data["q_indices"][peak]}')
print(f'⟨S_total²⟩ = {summary["total_spin_squared"]:.6f} ± {summary["total_spin_squared_sem"]:.6f}')
""")
md("""
## Normalizations and saved numerical values

The occupation sum rules are sum_k n_up(k)=24 and sum_k n_down(k)=24.
The spin sum rule is sum_q S(q)=sum_i <S_i²>. These checks and the per-configuration energy reconstructed from hopping correlations and double occupancy were verified during measurement.
""")
code(r"""
np.testing.assert_allclose(data['n_up_q'].sum(),24,atol=1e-10)
np.testing.assert_allclose(data['n_down_q'].sum(),24,atol=1e-10)
np.testing.assert_allclose(data['S_q'].sum(),48*summary['mean_local_moment_squared'],atol=1e-10)
print(f'Σk n_up(k)   = {data["n_up_q"].sum():.8f}')
print(f'Σk n_down(k) = {data["n_down_q"].sum():.8f}')
print(f'Σq S(q)     = {data["S_q"].sum():.8f}')
print(f'Up occupation range:   {data["n_up_q"].min():.6f} to {data["n_up_q"].max():.6f}')
print(f'Down occupation range: {data["n_down_q"].min():.6f} to {data["n_down_q"].max():.6f}')
print(f'CSV: {data_dir / "momentum_observables.csv"}')
# This named-column array contains all 48 momenta and their SEMs.
momentum_table = np.genfromtxt(data_dir / 'momentum_observables.csv',delimiter=',',names=True)
with np.printoptions(precision=5,suppress=True,linewidth=160):
    print(momentum_table)
""")
md("""
Results live under `u8_n48_lr003_20261015/observables/`:

- `momentum_observables.png`: the three Brillouin-zone panels.
- `momentum_observables.csv`: all distinct momenta, values and SEMs.
- `observables.npz`: arrays used by this notebook.
- `momentum_block_means.npz`, `measurement_locals.npz`, `measurement_samples.npy`: evaluation data for reanalysis.
- `summary.json`: measurement settings and diagnostics.

The final checkpoint, optimizer buffers, training history and matched final-checkpoint samples are under `train/lr_0.03/`. Blackwell nodes were excluded.
""")
if (measurement / 'charge_structure_factor.npz').exists():
    from charge_structure_factor import notebook_cells
    cells.extend(notebook_cells())
nb.cells = cells
nbf.validate(nb)
nbf.write(nb, path)
try:
    NotebookClient(
        nb,
        timeout=180,
        kernel_name="quantax_new",
        resources={"metadata": {"path": str(root)}},
    ).execute()
finally:
    nbf.write(nb, path)
errors = [
    o
    for c in nb.cells
    if c.cell_type == "code"
    for o in c.outputs
    if o.output_type == "error"
]
images = sum(
    "image/png" in o.get("data", {})
    for c in nb.cells
    if c.cell_type == "code"
    for o in c.outputs
)
assert not errors, [(o.ename, o.evalue) for o in errors]
assert images == (3 if (measurement / 'charge_structure_factor.npz').exists() else 2), images
nbf.validate(nb)
nbf.write(nb, path)
(run_root / "README.md").write_text(f"""# N=48, U=8, AdamSR lr=0.03

Completed {training['iterations']} iterations with the same 1024 samples, 8 blocks, 16 CNN channels and 16 pairing features as the 12-site comparison. TriangularB(4), periodic boundaries, 24 up and 24 down electrons. Fresh seed 3232; no transfer initialization. Sampler thermalization 1920 proposals, 192 proposals per optimization sweep. Final saved samples match the frozen checkpoint.

Final measured E/site: {summary['energy_per_site']:.8f} +/- {summary['energy_per_site_sem']:.8f} (block SEM).
Peak S(q): {summary['peak_S_q']:.8f} +/- {summary['peak_S_q_sem']:.8f}, at {summary['peak_q_cartesian']}.

Open ../n48_momentum_observables.ipynb in Jupyter for executed training and momentum-space plots. The 1-RDM is represented by n_sigma(k)=<c^dagger_(k,sigma)c_(k,sigma)>; sum_k n_sigma(k)=24. No dense site-by-site 1-RDMs were needed.

Twenty frozen-state evaluation sweeps provide 20480 observable samples. Error bars are approximate block SEMs. This single run does not establish optimization convergence or ground-state accuracy.

Training job 7250101; measurement job 7250105. Blackwell nodes workergpu[171-193] excluded.
""")
print(f"Saved executed notebook: {path}; {images} figures; no errors.")
