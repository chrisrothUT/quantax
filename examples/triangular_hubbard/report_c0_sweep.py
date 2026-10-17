"""Report finished and failed C0 trials, without selecting noisy training minima."""

from pathlib import Path
import argparse, csv, json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
from nbclient import NotebookClient

CAPS = [0.1, 0.3, 1, 2, 3, 5]


def plot(root):
    root = Path(root)
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    for cap in CAPS:
        folder = root / "runs" / f"C0_{cap:g}"
        summary = (
            json.loads((folder / "summary.json").read_text())
            if (folder / "summary.json").exists()
            else {}
        )
        row = dict(
            C0=cap,
            status=summary.get("status", "incomplete"),
            iterations=summary.get("iterations", 0),
            energy_per_site=summary.get("final_energy_per_site"),
            block_sem_per_site=summary.get("final_block_sem"),
        )
        if row["block_sem_per_site"] is not None:
            row["block_sem_per_site"] /= 48
        if (folder / "history.csv").exists():
            h = np.atleast_1d(
                np.genfromtxt(folder / "history.csv", delimiter=",", names=True)
            )
            if len(h):
                row["iterations"] = len(h)
                x, y = h["iteration"], h["energy_per_site"]
                (line,) = axes[0].plot(x, y, alpha=0.18, lw=0.6)
                w = min(25, len(y))
                axes[0].plot(
                    x[w - 1 :],
                    np.convolve(y, np.ones(w) / w, mode="valid"),
                    label=f"C0={cap:g}",
                    color=line.get_color(),
                )
                axes[1].plot(x, h["applied_step_norm"], label=f"C0={cap:g}")
        rows.append(row)
        if row["status"] == "complete":
            axes[2].errorbar(
                cap,
                row["energy_per_site"],
                yerr=row["block_sem_per_site"],
                fmt="o",
                capsize=4,
                color="C0",
            )
            axes[2].annotate(
                f"{cap:g}",
                (cap, row["energy_per_site"]),
                xytext=(4, 5),
                textcoords="offset points",
            )
    axes[0].set(
        xlabel="Iteration",
        ylabel="Energy per site",
        title="Training: raw and 25-step mean",
    )
    axes[1].set(
        xlabel="Iteration",
        ylabel="Applied parameter-step norm",
        title="Constant caps for this sweep",
        yscale="log",
    )
    axes[2].set(
        xlabel="C0",
        ylabel="Frozen-checkpoint energy per site",
        title="20 evaluation blocks ± block SEM",
        xscale="log",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("N=48, U=8, norm-constrained AdamSR: 4096 samples, 1000 iterations")
    return fig, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    fig, rows = plot(root)
    fig.savefig(root / "comparison.png", dpi=180)
    plt.close(fig)
    with (root / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    valid = [r for r in rows if r["status"] == "complete" and r["iterations"] == 1000]
    result = dict(
        trials=rows,
        best_completed_C0=(
            min(valid, key=lambda r: r["energy_per_site"])["C0"] if valid else None
        ),
        caveat="One seed per cap: block SEM excludes optimization and seed uncertainty. Repeat promising candidates. Caps are constant; decay schedule not tuned.",
    )
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = dict(
        name="quantax_new", language="python", display_name="Python (quantax_new)"
    )
    nb.cells = [
        nbf.v4.new_markdown_cell(
            """# N=48, U=8: norm-constrained AdamSR C0 sweep

C0 = 0.1, 0.3, 1, 2, 3, 5. Each run starts from seed 3232 with 4096 samples, 1000 iterations, 8 blocks, 16 CNN channels and pairing features. Adam factors mu=0.95, beta=0.995. Both solves use auto_pinv_eig. The norm cap is applied internally with unit learning rate. t0=1000 and T=1000 keep the cap constant throughout this comparison; decay is not tuned here.

Final energies use re-equilibrated frozen checkpoints and 20 evaluation sweeps of 4096 samples each, rather than the minimum stochastic training energy. Error bars are SEMs of sweep means; residual correlation may make them too small. They do not include optimization or random-seed uncertainty. Failed and incomplete trials remain visible. Lower final energy ranks the completed runs at this training budget, but does not establish convergence. Repeat promising caps with other seeds before choosing a production setting.

The earlier N=48 lr=0.03 result used a shifted solver and 1024 samples, so it is a contextual baseline, not a controlled cap-only comparison."""
        ),
        nbf.v4.new_code_cell(f"""%matplotlib inline
from pathlib import Path
import sys,json
import matplotlib.pyplot as plt
sys.path.insert(0,{str(Path(__file__).resolve().parent)!r})
from report_c0_sweep import plot
root=Path({str(root)!r})
fig,rows=plot(root)
plt.show()
print('C0    status      iterations       E/site          block SEM')
for r in rows:
    print(r)
print('Best completed cap:',json.loads((root/'comparison.json').read_text())['best_completed_C0'])
print('Comparison CSV:',root/'comparison.csv')"""),
    ]
    path = root / "comparison.ipynb"
    try:
        NotebookClient(
            nb,
            timeout=180,
            kernel_name="quantax_new",
            resources={"metadata": {"path": str(root)}},
        ).execute()
    finally:
        nbf.write(nb, path)
    nbf.validate(nb)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
