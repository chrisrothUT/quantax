"""Compare finished AdamSR trials and plot their training curves."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    tail_values = []
    verification_used = []
    for index, lr in enumerate([0.001, 0.003, 0.01, 0.03, 0.1]):
        color = f"C{index}"
        folder = args.root / "runs" / f"lr_{lr:g}"
        verified = args.root / "h100_verification" / f"lr_{lr:g}"
        checked = verified / "summary.json"
        if (
            checked.exists()
            and json.loads(checked.read_text()).get("status") == "complete"
        ):
            folder = verified
            verification_used.append(lr)
        summary = folder / "summary.json"
        data = json.loads(summary.read_text()) if summary.exists() else {}
        row = dict(
            learning_rate=lr,
            status=data.get("status", "missing"),
            iterations=data.get("iterations", 0),
            final_energy_per_site=data.get("final_energy_per_site"),
            final_block_sem_per_site=data.get("final_block_sem", None),
            tail200_energy_per_site=None,
            median_step_seconds=data.get("median_step_seconds"),
            source_directory=str(folder.relative_to(args.root)),
            node=(
                json.loads((folder / "config.json").read_text()).get("hostname")
                if (folder / "config.json").exists()
                else None
            ),
        )
        if row["final_block_sem_per_site"] is not None:
            row["final_block_sem_per_site"] /= 12
        if "tail200_energy_mean" in data:
            row["tail200_energy_per_site"] = data["tail200_energy_mean"] / 12
        rows.append(row)
        history = folder / "history.csv"
        if history.exists():
            with history.open() as f:
                history_rows = list(csv.DictReader(f))
            if history_rows:
                x = np.array([int(r["iteration"]) for r in history_rows])
                y = np.array([float(r["energy_per_site"]) for r in history_rows])
                tail_values.extend(y[x >= 800].tolist())
                for ax in axes:
                    ax.plot(x, y, color=color, alpha=0.18, linewidth=0.6)
                    window = min(25, len(y))
                    ax.plot(
                        x[window - 1 :],
                        np.convolve(y, np.ones(window) / window, mode="valid"),
                        label=f"lr={lr:g}",
                        color=color,
                    )
    axes[0].set(
        title="U=8, TriangularB(2), AdamSR",
        xlabel="Iteration",
        ylabel="Energy per site",
    )
    axes[1].set(
        title="Last 200 iterations",
        xlabel="Iteration",
        ylabel="Energy per site",
        xlim=(800, 1000),
    )
    if tail_values:
        low, high = min(tail_values), max(tail_values)
        pad = max(0.02, 0.05 * (high - low))
        axes[1].set_ylim(low - pad, high + pad)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(args.root / "comparison.png", dpi=180)
    plt.close(fig)
    with (args.root / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    valid = [
        r
        for r in rows
        if r["status"] == "complete"
        and r["iterations"] == 1000
        and r["final_energy_per_site"] is not None
    ]
    valid.sort(key=lambda r: r["final_energy_per_site"])
    lines = [
        "# U=8 AdamSR learning-rate sweep",
        "",
        "12-site TriangularB(2), (6 up, 6 down), 1024 samples, 1000 iterations, 8 residual blocks, 16 CNN channels and 16 pairing features. Constant learning rates; identical seed 3232. Each final frozen state is evaluated over 20 additional sweeps (20480 samples).",
        "",
        "| Learning rate | Status | Final E/site | Block SEM/site | Last-200 mean E/site |",
        "|---|---|---:|---:|---:|",
    ]
    fmt = lambda v: "—" if v is None else f"{v:.8f}"
    for r in rows:
        lines.append(
            f'| {r["learning_rate"]:g} | {r["status"]} | {fmt(r["final_energy_per_site"])} | {fmt(r["final_block_sem_per_site"])} | {fmt(r["tail200_energy_per_site"])} |'
        )
    if verification_used:
        lines.extend(
            [
                "",
                "The 0.01 result uses the H100 verification run. The original Blackwell trial is preserved under runs/lr_0.01; future submissions exclude workergpu[171-193].",
            ]
        )
    if valid:
        best = valid[0]
        lines.extend(
            [
                "",
                f'Lowest final measured energy: learning rate **{best["learning_rate"]:g}**.',
                "",
                "This is a single-seed screening result. Block standard errors are approximate and may underestimate residual autocorrelation. Close candidates should be compared across more seeds and with a decaying learning rate.",
            ]
        )
        (args.root / "best.json").write_text(json.dumps(best, indent=2) + "\n")
    else:
        lines.extend(
            [
                "",
                "No complete 1000-iteration trial is available for ranking. Check the job logs and per-trial summary.json files.",
            ]
        )
    (args.root / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
