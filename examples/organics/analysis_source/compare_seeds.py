"""Reproducible four-seed comparison; incomplete runs are listed, never imputed."""

from pathlib import Path
import json, csv, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from plot_helpers import geometry, periodic_display_points, low_k_fits, save_plot

ROOT = Path(__file__).resolve().parents[1]


def load(root=ROOT):
    manifest = json.loads((root / "experiment.json").read_text())
    ready = []
    pending = []
    for item in manifest["runs"]:
        if item.get("existing"):
            continue
        run = Path(item["run"])
        dp = run.parents[1] / "dimer_correlations/dimer_correlations.npz"
        marker = dp.parent / "measurement.json"
        if (
            not (run / "analysis.npz").exists()
            or not dp.exists()
            or not marker.exists()
        ):
            pending.append(item["seed"])
            continue
        cfg = json.loads((run / "config.json").read_text())
        assert (
            cfg["seed"] == item["seed"]
            and cfg["U"] == manifest["U"]
            and cfg["C0"] == 0.5
        )
        with np.load(run / "analysis.npz") as f:
            data = {k: f[k] for k in f.files}
        with np.load(dp) as f:
            dimers = {k: f[k] for k in f.files}
        ready.append((run, cfg, data, dimers))
    return ready, pending


def sem(x):
    x = np.asarray(x)
    return x.std(axis=0, ddof=1) / np.sqrt(len(x))


def table(ready, root=ROOT):
    rows = []
    for run, c, d, b in ready:
        rows.append(
            dict(
                seed=c["seed"],
                energy_per_site=np.mean(d["energy_per_site_blocks"]),
                energy_sem=sem(d["energy_per_site_blocks"]),
                double_occupancy=np.mean(d["double_occupancy_blocks"]),
                double_sem=sem(d["double_occupancy_blocks"]),
                spin_peak_index=int(np.argmax(d["S_q"])),
                dimer_peak_index=int(np.argmax(b["S_eigenvalues"][:, -1])),
                run=str(run),
            )
        )
    if rows:
        out = root / "comparison_plots"
        out.mkdir(exist_ok=True)
        with (out / "seed_summary.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def energy_plot(ready, root=ROOT):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    seeds = [str(c["seed"]) for _, c, _, _ in ready]
    for ax, key, label in zip(
        axes,
        ["energy_per_site_blocks", "double_occupancy_blocks"],
        ["Frozen-state E/N", "Double occupancy"],
    ):
        blocks = [d[key] for _, _, d, _ in ready]
        ax.errorbar(
            range(len(seeds)),
            [np.mean(x) for x in blocks],
            yerr=[sem(x) for x in blocks],
            fmt="o",
            capsize=4,
        )
        ax.set_xticks(range(len(seeds)), seeds)
        ax.set(xlabel="Initialization seed", ylabel=label)
        ax.grid(alpha=0.2)
    save_plot(fig, root / "comparison_plots", "energy_by_seed", pdf=True)
    return fig


def maps(ready, root=ROOT):
    fig, axes = plt.subplots(
        3, len(ready), figsize=(4 * len(ready), 11), squeeze=False, layout="constrained"
    )
    for row, field in enumerate(["S_q", "N_q", "dimer"]):
        values = [
            b["S_eigenvalues"][:, -1] if field == "dimer" else d[field]
            for _, _, d, b in ready
        ]
        low = min(0.0, min(float(x.min()) for x in values))
        high = max(float(x.max()) for x in values)
        for col, ((run, c, d, b), v) in enumerate(zip(ready, values)):
            g = geometry(run)
            points, source, boundary = periodic_display_points(
                d["q_raw"], g["reciprocal_vectors"]
            )
            ax = axes[row, col]
            ax.add_patch(Polygon(boundary, fill=False, color="gray", linestyle="--"))
            im = ax.scatter(
                *points.T,
                c=v[source],
                vmin=low,
                vmax=high,
                cmap="viridis",
                s=70,
                edgecolor="black",
                linewidth=0.4,
            )
            ax.set(
                aspect="equal",
                xlabel="$k_x$",
                ylabel="$k_y$",
                title=f"Seed {c['seed']}",
            )
        fig.colorbar(
            im,
            ax=list(axes[row]),
            label={
                "S_q": "Spin S(k)",
                "N_q": "Charge N(k)",
                "dimer": "Largest dimer eigenvalue",
            }[field],
            shrink=0.8,
        )
    save_plot(fig, root / "comparison_plots", "structure_factors_by_seed", pdf=True)
    return fig


def charge_plot(ready, root=ROOT):
    fig, axes = plt.subplots(
        1,
        len(ready),
        figsize=(4 * len(ready), 3.8),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for ax, (_, c, d, _) in zip(axes.flat, ready):
        f = low_k_fits(d)
        k = f["k"]
        sel = (d["k"] > 1e-9) & (d["k"] <= k[-1] + 1e-8)
        ax.scatter(d["k"][sel], d["N_q"][sel], s=16, color=".7", alpha=0.5)
        ax.errorbar(k, f["mean"], yerr=f["sem"], fmt="ko", capsize=3)
        x = np.linspace(0, k[-1] * 1.06, 200)
        ax.plot(x, f["coefficients"][1][0] * x, color="C0", label="$ak$")
        ax.plot(x, f["coefficients"][2][0] * x * x, "--", color="C1", label="$bk^2$")
        ax.set(title=f"Seed {c['seed']}", xlabel="$|k|$", ylabel="$N(k)$")
        ax.legend(frameon=False)
    save_plot(fig, root / "comparison_plots", "low_k_by_seed", pdf=True)
    return fig


def distances(ready, root=ROOT):
    # Absolute RMS of independent, nonduplicated momentum classes; no inferred phase labels.
    fields = {
        "spin": np.array([d["S_q"] for _, _, d, _ in ready]),
        "charge": np.array([d["N_q"] for _, _, d, _ in ready]),
        "long_range_dimer": np.array(
            [
                b["C"][b["distance"] >= float(b["long_rmin"]) - 1e-9].real.ravel()
                for _, _, _, b in ready
            ]
        ),
    }
    seeds = [c["seed"] for _, c, _, _ in ready]
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, (field, features) in zip(axes, fields.items()):
        rms = np.sqrt(np.mean((features[:, None] - features[None, :]) ** 2, axis=-1))
        im = ax.imshow(rms, cmap="viridis", vmin=0)
        fig.colorbar(im, ax=ax, label="Absolute RMS difference")
        ax.set_xticks(range(len(seeds)), seeds, rotation=45)
        ax.set_yticks(range(len(seeds)), seeds)
        ax.set_title(field.replace("_", " "))
        for i, s in enumerate(seeds):
            for j, t in enumerate(seeds):
                rows.append(
                    dict(observable=field, seed_i=s, seed_j=t, rms_difference=rms[i, j])
                )
    with (root / "comparison_plots/pairwise_distances.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    save_plot(fig, root / "comparison_plots", "pairwise_pattern_distances", pdf=True)
    return fig


def make(root=ROOT, execute=False):
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = dict(
        display_name="Quantax new", language="python", name="quantax_new"
    )
    nb.cells = [
        nbf.v4.new_markdown_cell(
            """# U=10: four independent initialization seeds

N=48, half filling, horizontal t′/t=0.8, 32 CNN features, d=96, C0=0.5. Seeds 3233–3236 repeat the original 3232 protocol from scratch: 2,000 training steps, 4,096 samples per step, two GPUs per job. Each final evaluation retains four original temporal blocks (16,384 samples).

The table and plots include only completed analyses. Error bars are SEMs of the four original blocks, excluding optimization bias and residual autocorrelation. Structure-factor rows share color scales across seeds; periodic boundary copies are for display only. Pairwise absolute RMS differences retain amplitude and use unique momenta or the full long-range dimer tensor (|R|≥2). They are descriptive comparisons, with no significance threshold or automatic phase assignment. A few patterns could indicate competing optimization basins; four seeds on one cluster cannot establish thermodynamic phases."""
        )
    ]
    interaction = json.loads((root / "experiment.json").read_text())["U"]
    nb.cells[0].source = nb.cells[0].source.replace("U=10:", f"U={interaction:g}:")
    setup = (
        "from pathlib import Path\nimport sys\nROOT=Path("
        + repr(str(root))
        + ")\nsys.path.insert(0,str(ROOT/'analysis_source'))\nimport matplotlib.pyplot as plt\nfrom IPython.display import display\nfrom compare_seeds import load, table, energy_plot, maps, charge_plot, distances\nfrom plot_helpers import plot_training, plot_correlations, plot_dimer_real_space, plot_dimer_distance, save_plot\n%matplotlib inline\nready,pending=load(ROOT)\nprint('Completed seeds:',[c['seed'] for _,c,_,_ in ready])\nprint('Pending or incomplete:',pending)\nprint('Seed       E/N ± SEM                  double occupancy ± SEM')\nfor row in table(ready,ROOT):\n    print(f\"{row['seed']}     {row['energy_per_site']:.8f} ± {row['energy_sem']:.8f}    {row['double_occupancy']:.8f} ± {row['double_sem']:.8f}\")"
    )
    nb.cells.append(nbf.v4.new_code_cell(setup))
    for title, call in [
        ("Frozen-state energy and occupancy", "energy_plot"),
        ("Spin, charge and dimer structure factors", "maps"),
        (
            "First three nonzero charge shells: unweighted fits through zero",
            "charge_plot",
        ),
        ("Pairwise pattern differences", "distances"),
    ]:
        nb.cells.extend(
            [
                nbf.v4.new_markdown_cell("## " + title),
                nbf.v4.new_code_cell(
                    f"if ready:\n    fig={call}(ready,ROOT)\n    plt.show()\n    plt.close(fig)"
                ),
            ]
        )
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## Training convergence and full correlations per seed\n\nDirect SU(2) correlations include both transverse exchange terms. Connected dimer tensors reuse the same spin local estimators via the Hermitian Gram identity, with original-block jackknife errors and no new samples. No rotational averaging is applied."
        )
    )
    nb.cells.append(nbf.v4.new_code_cell("""for run,cfg,d,b in ready:
    print('Seed',cfg['seed'])
    figs=[plot_training(run,cfg),plot_correlations(run,cfg,d,'spin'),plot_correlations(run,cfg,d,'charge'),plot_dimer_real_space(run,cfg,b),plot_dimer_distance(cfg,b)]
    for i,fig in enumerate(figs):
        fig.suptitle(f"U={cfg['U']:g}, seed {cfg['seed']}")
        save_plot(fig,run.parents[1]/'plots',f'seed_{cfg["seed"]}_detail_{i}')
        plt.show()
        plt.close(fig)
"""))
    if execute:
        from nbclient import NotebookClient

        nb = NotebookClient(
            nb,
            timeout=900,
            kernel_name="quantax_new",
            resources={"metadata": {"path": str(root)}},
        ).execute()
    p = root / "seed_comparison.ipynb"
    tmp = p.with_suffix(".ipynb.tmp")
    nbf.write(nb, tmp)
    tmp.replace(p)
    print(p)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    make(root=a.root, execute=a.execute)
