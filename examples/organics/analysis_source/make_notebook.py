"""Generate executed Organics notebooks with common periodic and low-k plots."""

import argparse
import json
from pathlib import Path
import nbformat as nbf


def make(root, interaction=None, execute=False):
    root = root.resolve()
    experiment = json.loads((root / "experiment.json").read_text())
    chosen = sorted(experiment["U_order"]) if interaction is None else [interaction]
    targets = [
        root / f"u{u}_n48_tp0.8_cnn32_d96_seed3232_C0_1_s4096/train/C0_1"
        for u in chosen
    ]
    notebook = nbf.v4.new_notebook()
    notebook.metadata.kernelspec = dict(
        display_name="Quantax new", language="python", name="quantax_new"
    )
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            r"""# N=48, $t'/t=0.8$: spin correlations and charge localization

32 CNN features, $d=96$, $C_0=1$, periodic TriangularB, half filling. The weaker hopping lies along x. Couplings and plots are ordered by increasing $U$.

Each analysis reuses the **four original temporal blocks of 4,096 frozen-state configurations** (16,384 total). Spin correlations directly include $S_i^zS_j^z+(S_i^+S_j^-+S_i^-S_j^+)/2$; $3\langle S_i^zS_j^z\rangle$ is not substituted. Full raw correlation tensors are retained. Translation averaging is applied afterward for displacement plots, with no rotational averaging.

Real-space and Brillouin-zone plots include **all equivalent periodic images tied at the boundary**. These are display copies of the same estimates, not new measurements, and are excluded from statistical averages and fits. Dashed outlines delimit the cluster Wigner-Seitz cell and first Brillouin zone; no rotation or reflection of the measured values is applied.

The charge diagnostic is $N(\mathbf{k})=N^{-1}\langle|\sum_j e^{-i\mathbf{k}\cdot\mathbf{r}_j}(n_j-1)|^2\rangle$. Fixed particle number forces $N(0)=0$ in both phases. Errors use the four original sweep blocks, with no subdivision of chains. Residual autocorrelation, optimization bias and finite-size effects are not included."""
        )
    ]
    setup = """from pathlib import Path
import sys, json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
%matplotlib inline
plt.rcParams.update({'figure.dpi': 110})
"""
    setup += "ROOT = Path(" + repr(str(root)) + ")\n"
    setup += "sys.path.insert(0, str(ROOT / 'analysis_source'))\n"
    setup += "from plot_helpers import (plot_training, plot_correlations, plot_low_k, plot_energy, plot_dimer_real_space, plot_dimer_momentum, plot_dimer_distance, save_plot)\n"
    setup += "targets = [Path(p) for p in " + repr([str(p) for p in targets]) + "]\n"
    setup += """ready = []
for run in targets:
    if (run / 'analysis.npz').exists():
        cfg = json.loads((run / 'config.json').read_text())
        with np.load(run / 'analysis.npz') as f:
            data = {key: f[key] for key in f.files}
        ready.append((run, cfg, data))
    else:
        print('Analysis pending:', run.parents[1].name)
ready.sort(key=lambda item: item[1]['U'])
print(f'{len(ready)}/{len(targets)} analyses available; U/t =', [cfg['U'] for _, cfg, _ in ready])
"""
    notebook.cells.append(nbf.v4.new_code_cell(setup))
    notebook.cells.append(nbf.v4.new_code_cell("""def estimate(x):
    x = np.asarray(x)
    return f'{x.mean():.7f} ± {x.std(ddof=1)/np.sqrt(len(x)):.7f}'
print('U/t     E/N (sweep-block SEM)          double occupancy (sweep-block SEM)')
for run, cfg, d in ready:
    print(f"{cfg['U']:g}       {estimate(d['energy_per_site_blocks'])}     {estimate(d['double_occupancy_blocks'])}")
"""))
    notebook.cells.append(
        nbf.v4.new_markdown_cell(
            r"""## Training convergence

The left panel retains the full training history. The right panel shows the final 25% of steps (500 for these runs), with vertical limits set from **all** estimates in that interval plus padding. Gray lines show individual training estimates; blue lines show a 25-step trailing mean. No tail outliers are discarded."""
        )
    )
    notebook.cells.append(nbf.v4.new_code_cell("""for run, cfg, d in ready:
    fig = plot_training(run, cfg)
    plt.show()
    plt.close(fig)
"""))
    notebook.cells.append(
        nbf.v4.new_markdown_cell(
            r"""## Direct SU(2) spin correlations

Full site-pair tensor, translation-averaged displacement correlations with periodic edge copies, and $S(\mathbf{k})$ over the complete first BZ. The onsite displacement is omitted from the spatial scatter to resolve the offsite correlations."""
        )
    )
    notebook.cells.append(nbf.v4.new_code_cell("""for run, cfg, d in ready:
    fig = plot_correlations(run, cfg, d, 'spin')
    plt.show()
    plt.close(fig)
    print('U/t =', cfg['U'], ': spin sum Σk S(k) =', d['S_q'].sum(), '; onsite trace =', np.trace(d['spin_correlation']))
    print('S(k=0), without a singlet assumption =', d['S_q'][0], '±', d['S_q_sem'][0])
"""))
    notebook.cells.append(
        nbf.v4.new_markdown_cell(
            r"""## Charge correlations and structure factor

The real-space tensor is $\langle(n_i-1)(n_j-1)\rangle$. As for the spin plots, only equivalent periodic boundary images are repeated. The momentum-space panel shows the measured $N(\mathbf{k})$ without angular averaging."""
        )
    )
    notebook.cells.append(nbf.v4.new_code_cell("""for run, cfg, d in ready:
    fig = plot_correlations(run, cfg, d, 'charge')
    plt.show()
    plt.close(fig)
"""))
    notebook.cells.append(
        nbf.v4.new_markdown_cell(
            r"""## Low-k charge structure factor — meeting notebook style

Black points and error bars are **radial-shell means ± original-block SEM** for the three smallest nonzero momentum magnitudes. Directions are averaged inside each original block before computing errors and covariance; faint gray points retain the directional variation. Periodic display duplicates do not enter this calculation.

As in `HubbardTriangularNewQuantax/meeting_20261020.ipynb`, blue solid and orange dashed lines show separate zero-intercept fits $N(k)=ak$ and $N(k)=bk^2$. Each fit now minimizes the **ordinary, unweighted squared residuals of exactly the first three black points**. The coefficient errors are the SEM of the same linear fit estimator over the four original blocks. No covariance or inverse-variance weighting is applied. The selected points, predictions and residuals are printed below and exported to `charge_low_k_fit_points.csv`.

These are finite-momentum comparisons, not an extrapolation establishing a metallic or insulating phase. The first three shells here are $|k|=\pi/3$, $\pi/\sqrt{3}$ and $2\pi/3$ (maximum $2.0944$)."""
        )
    )
    output = (
        root / "analysis_plots"
        if interaction is None
        else targets[0].parents[1] / "plots"
    )
    notebook.cells.append(
        nbf.v4.new_code_cell(
            """if ready:
    fig, fits = plot_low_k(ready, Path("""
            + repr(str(output))
            + """))
    plt.show()
    plt.close(fig)
    print('U/t    linear a ± block SEM       quadratic b ± block SEM')
    for U, fit in fits.items():
        a, ae = fit['coefficients'][1]
        b, be = fit['coefficients'][2]
        print(f"{U:g}      {a:.7f} ± {ae:.7f}      {b:.7f} ± {be:.7f}")
"""
        )
    )
    notebook.cells.append(nbf.v4.new_code_cell("""if ready:
    print('Exactly these three black points enter each fit:')
    print('U/t    |k|         N(k)        linear prediction    quadratic prediction')
    for U, fit in fits.items():
        for k, y in zip(fit['k'], fit['mean']):
            print(f"{U:g}     {k:.7f}    {y:.7f}    {fit['coefficients'][1][0]*k:.7f}            {fit['coefficients'][2][0]*k*k:.7f}")
"""))
    if interaction is None:
        notebook.cells.append(
            nbf.v4.new_markdown_cell(
                r"""## Energy versus U and comparison of saved trial states

The black points are the **final frozen-state energy per site**, with SEMs from the original four evaluation blocks. They are not minima of noisy training estimates.

For a fixed saved state, its expectation at another interaction is exactly linear: $E_{\psi}(U)/N=E_{\psi}(U_0)/N+(U-U_0)D_{\psi}$, where $D=\langle n_\uparrow n_\downarrow\rangle$. The right panel re-evaluates the saved U=9,10,12 states without any reoptimization; uncertainties preserve covariance between E and D within each original block. A lower cross-evaluated energy indicates that the independently trained state has not found the lowest energy already available within this set of trial states. Correlation-pattern differences alone should therefore not be assigned to distinct thermodynamic phases."""
            )
        )
        notebook.cells.append(nbf.v4.new_code_cell("""if ready:
    fig, cross_energies = plot_energy(ready, ROOT / 'analysis_plots')
    plt.show()
    plt.close(fig)
    print('Saved-state U    evaluated U    E/N ± original-block SEM')
    for row in cross_energies:
        print(f"{row['source_U']:g}                {row['evaluated_U']:g}              {row['energy_per_site']:.7f} ± {row['block_sem']:.7f}")
"""))
    notebook.cells.append(
        nbf.v4.new_markdown_cell(
            r"""## Connected dimer–dimer correlations: U=9,10,12

For $B_{i,a}=\mathbf{S}_i\cdot\mathbf{S}_{i+\delta_a}$, the three orientations are $(1,0)$, $(-1/2,\sqrt{3}/2)$ and $(-1/2,-\sqrt{3}/2)$. The full tensor is $C_{ab}(R)=N^{-1}\sum_i\langle B_{i,a}B_{i+R,b}\rangle-\mu_a\mu_b$. It is obtained from the **same 16,384 configurations** using the Hermitian Gram identity $\langle B_iB_j\rangle=\mathbb{E}[b_i^*b_j]$, with each bond local estimator extracted from the previously measured independent spin-dot-spin pair tensor. No new samples, symmetry reconstruction or wavefunction evaluations are used. Full raw site/bond tensors and all original blocks are saved before translation averaging.

Real-space panels match the meeting notebook: one black reference bond per orientation, all three target orientations colored with a shared symmetric-log scale and linear interval $\pm10^{-3}$, retaining all tied periodic edge copies. Bond-center momentum matrices differ from the anchor convention by a unitary phase transformation, so their eigenvalues agree, including across reciprocal boundary copies. The dominant dimer eigenvalue is shown with the same BZ style as before. Distance-shell tensors are averaged before diagonalization. Disconnected subtraction and eigenvalues are recomputed inside a delete-one-original-block jackknife. The distance-averaged mode uses $|R|\ge L/2=2$; nearly degenerate modes have no unique signed vector."""
        )
    )
    notebook.cells.append(nbf.v4.new_code_cell("""dimer_items = []
for run, cfg, d in ready:
    if cfg['U'] not in [9,10,12]:
        continue
    path = run.parents[1] / 'dimer_correlations' / 'dimer_correlations.npz'
    if not path.exists() or not (path.parent / 'measurement.json').exists():
        print('Dimer analysis pending for U =', cfg['U'])
        continue
    with np.load(path) as f:
        dimers = {key:f[key] for key in f.files}
    dimer_items.append((run,cfg,dimers))
    fig = plot_dimer_real_space(run,cfg,dimers)
    plt.show()
    plt.close(fig)
    fig = plot_dimer_distance(cfg,dimers)
    save_plot(fig,run.parents[1]/'plots','dimer_distance_modes')
    plt.show()
    plt.close(fig)
    print('U/t =',cfg['U'],'; mean bond energies:',dimers['mean_bond'])
    print('Distance-averaged eigenvalues:',dimers['long_eigenvalues'])
    print('Jackknife SEM:',dimers['long_eigenvalues_sem'])
if dimer_items:
    fig = plot_dimer_momentum(dimer_items,ROOT/'analysis_plots')
    plt.show()
    plt.close(fig)
"""))
    notebook.cells.append(
        nbf.v4.new_markdown_cell(
            "## Comparison across U at the smallest nonzero momenta"
        )
    )
    notebook.cells.append(nbf.v4.new_code_cell("""if ready:
    smallest = np.min(ready[0][2]['k'][ready[0][2]['k']>1e-9])
    indices = np.flatnonzero(np.isclose(ready[0][2]['k'], smallest))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for i in indices:
        q = ready[0][2]['q'][i]
        axes[0].errorbar([c['U'] for _, c, _ in ready], [d['N_q'][i] for _, _, d in ready],
                         yerr=[d['N_q_sem'][i] for _, _, d in ready], fmt='o-', capsize=3,
                         label=f'({q[0]:.2f}, {q[1]:.2f})')
    axes[0].set(xlabel='U/t', ylabel='N(k) at smallest nonzero momenta')
    axes[0].legend(fontsize=8)
    axes[1].errorbar([c['U'] for _, c, _ in ready],
                    [d['double_occupancy_blocks'].mean() for _, _, d in ready],
                    yerr=[d['double_occupancy_blocks'].std(ddof=1)/np.sqrt(len(d['double_occupancy_blocks'])) for _, _, d in ready], fmt='o-', capsize=3)
    axes[1].set(xlabel='U/t', ylabel='Double occupancy')
    plt.show()
    plt.close(fig)
"""))
    destination = (
        root / "spin_charge_analysis.ipynb"
        if interaction is None
        else targets[0].parents[1] / "spin_charge_analysis.ipynb"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if execute:
        from nbclient import NotebookClient

        notebook = NotebookClient(
            notebook,
            timeout=600,
            kernel_name="quantax_new",
            resources={"metadata": {"path": str(root)}},
        ).execute()
        if any(
            o.get("output_type") == "error"
            for c in notebook.cells
            if c.cell_type == "code"
            for o in c.get("outputs", [])
        ):
            raise RuntimeError("Notebook contains an execution or rendering error")
    temporary = destination.with_suffix(".ipynb.tmp")
    nbf.write(notebook, temporary)
    temporary.replace(destination)
    return destination


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--U", type=int)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    print(make(args.root, args.U, args.execute))
