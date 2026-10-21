"""Plot saved Organics estimators; periodic display copies never enter statistics."""

import csv
import itertools
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import Voronoi

STYLE = {
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
}


def periodic_display_points(raw, periods):
    """Include every minimum-image representative, including tied boundary images."""
    raw, periods = np.asarray(raw), np.asarray(periods)
    shifts = np.array(list(itertools.product(range(-4, 5), repeat=2))) @ periods
    candidates = raw[:, None, :] - shifts[None, :, :]
    distances2 = np.sum(candidates**2, axis=-1)
    keep = np.isclose(
        distances2, distances2.min(axis=1, keepdims=True), atol=1e-9, rtol=1e-10
    )
    source, image = np.nonzero(keep)
    points = candidates[source, image]
    # The origin's Voronoi region is the Wigner-Seitz domain of this period lattice.
    origin = int(np.flatnonzero(np.all(shifts == 0, axis=1))[0])
    voronoi = Voronoi(shifts)
    region = voronoi.regions[voronoi.point_region[origin]]
    if -1 in region:
        raise ValueError("Periodic domain unexpectedly unbounded")
    boundary = voronoi.vertices[region]
    boundary = boundary[np.argsort(np.arctan2(boundary[:, 1], boundary[:, 0]))]
    return points, source, boundary


def geometry(run):
    with np.load(run / "su2_correlations/correlations.npz") as f:
        return {
            k: f[k]
            for k in [
                "coordinates",
                "xyz",
                "extent",
                "basis_vectors",
                "reciprocal_vectors",
            ]
        }


def translation_average(tensors, xyz, extent):
    indices = np.arange(len(xyz))
    shifts = [np.ravel_multi_index(((xyz + r) % extent).T, tuple(extent)) for r in xyz]
    return np.stack(
        [tensors[:, indices, shifted].mean(axis=1) for shifted in shifts], axis=1
    )


def domain_scatter(
    ax, points, source, boundary, values, *, title, labels, signed=False
):
    ax.add_patch(
        Polygon(
            boundary, fill=False, color="gray", linestyle="--", linewidth=1, zorder=1
        )
    )
    colors = values[source]
    if signed:
        maximum = max(float(np.max(abs(colors))), 1e-12)
        kwargs = dict(cmap="coolwarm", vmin=-maximum, vmax=maximum)
    else:
        kwargs = dict(
            cmap="viridis",
            vmin=min(0.0, float(np.min(colors))),
            vmax=max(float(np.max(colors)), 1e-12),
        )
    dots = ax.scatter(
        *points.T,
        c=colors,
        s=100,
        edgecolor="black",
        linewidth=0.45,
        zorder=3,
        **kwargs,
    )
    limits = np.max(abs(boundary), axis=0) * 1.10
    ax.set(
        title=title,
        xlabel=labels[0],
        ylabel=labels[1],
        xlim=(-limits[0], limits[0]),
        ylim=(-limits[1], limits[1]),
        aspect="equal",
    )
    return dots


def save_plot(fig, folder, name, pdf=False):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=180)
    if pdf:
        fig.savefig(folder / f"{name}.pdf")


def plot_training(run, cfg, *, tail_fraction=0.25, smooth_window=25):
    with (run / "history.csv").open() as f:
        rows = list(csv.DictReader(f))
    iteration = np.array([int(r["iteration"]) for r in rows])
    energy = np.array([float(r["energy_per_site"]) for r in rows])
    window = min(smooth_window, len(energy))
    smoothed = np.convolve(energy, np.ones(window) / window, mode="valid")
    count = max(1, int(np.ceil(tail_fraction * len(iteration))))
    tail = slice(len(iteration) - count, None)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), layout="constrained")
    for ax in axes:
        ax.plot(
            iteration,
            energy,
            color="0.65",
            lw=0.7,
            alpha=0.8,
            label="Training estimates",
        )
        ax.plot(
            iteration[window - 1 :],
            smoothed,
            color="#2864a5",
            lw=1.4,
            label=f"{window}-step trailing mean",
        )
        ax.set(xlabel="Training iteration", ylabel="Energy per site")
        ax.grid(alpha=0.18)
    axes[0].set_title("Full optimization")
    lo, hi = np.min(energy[tail]), np.max(energy[tail])
    padding = max((hi - lo) * 0.12, abs((lo + hi) / 2) * 1e-5, 1e-6)
    axes[1].set(
        xlim=(iteration[tail][0], iteration[-1]),
        ylim=(lo - padding, hi + padding),
        title=f"Final {count} steps",
    )
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"U/t = {cfg['U']:g}")
    save_plot(fig, run.parents[1] / "plots", "training_convergence")
    return fig


def plot_correlations(run, cfg, data, field):
    geom = geometry(run)
    real_periods = geom["extent"][:, None] * geom["basis_vectors"]
    real_points, real_source, real_boundary = periodic_display_points(
        geom["coordinates"] - geom["coordinates"][0], real_periods
    )
    q_points, q_source, q_boundary = periodic_display_points(
        data["q_raw"], geom["reciprocal_vectors"]
    )
    if field == "spin":
        tensors = data["spin_correlation_blocks"]
        values_q = data["S_q"]
        matrix_title = r"$\langle\mathbf{S}_i\cdot\mathbf{S}_j\rangle$"
        real_title = r"$C_S(\mathbf{r})$"
        momentum_title = r"$S(\mathbf{k})$"
    elif field == "charge":
        blocks = []
        for b in range(len(data["N_q_blocks"])):
            with np.load(run / "observables" / f"block_{b+1:03d}.npz") as f:
                blocks.append(f["charge_correlation_about_half_filling"])
        tensors = np.asarray(blocks)
        values_q = data["N_q"]
        matrix_title = r"$\langle(n_i-1)(n_j-1)\rangle$"
        real_title = r"$C_n(\mathbf{r})$"
        momentum_title = r"$N(\mathbf{k})$"
    else:
        raise ValueError(field)
    real_values = translation_average(tensors, geom["xyz"], geom["extent"]).mean(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), layout="constrained")
    im = axes[0].imshow(tensors.mean(axis=0), cmap="coolwarm", origin="lower")
    axes[0].set(xlabel="Site j", ylabel="Site i", title=matrix_title)
    fig.colorbar(im, ax=axes[0])
    keep = np.linalg.norm(real_points, axis=1) > 1e-9
    im = domain_scatter(
        axes[1],
        real_points[keep],
        real_source[keep],
        real_boundary,
        real_values,
        title=real_title,
        labels=("Displacement x", "Displacement y"),
        signed=True,
    )
    fig.colorbar(im, ax=axes[1])
    im = domain_scatter(
        axes[2],
        q_points,
        q_source,
        q_boundary,
        values_q,
        title=momentum_title,
        labels=(r"$k_x$", r"$k_y$"),
    )
    fig.colorbar(im, ax=axes[2])
    fig.suptitle(f"U/t = {cfg['U']:g}")
    save_plot(fig, run.parents[1] / "plots", f"{field}_correlations")
    return fig


def low_k_fits(data, n_shells=3):
    # Work exclusively with the original unique momentum classes, not display copies.
    radii, inverse = np.unique(np.round(data["k"], 10), return_inverse=True)
    selected = np.flatnonzero(radii > 1e-9)[:n_shells]
    k = radii[selected]
    blocks = np.stack(
        [data["N_q_blocks"][:, inverse == index].mean(axis=1) for index in selected],
        axis=1,
    )
    mean = blocks.mean(axis=0)
    sem = blocks.std(axis=0, ddof=1) / np.sqrt(len(blocks))
    covariance = np.cov(blocks, rowvar=False, ddof=1) / len(blocks)
    rank = np.linalg.matrix_rank(covariance)
    coefficients = {}
    residuals = {}
    for power in [1, 2]:
        x = k**power
        denominator = x @ x
        coefficient = x @ mean / denominator
        coefficient_blocks = blocks @ x / denominator
        error = coefficient_blocks.std(ddof=1) / np.sqrt(len(blocks))
        coefficients[power] = (float(coefficient), float(error))
        residuals[power] = mean - coefficient * x
    return dict(
        k=k,
        mean=mean,
        sem=sem,
        blocks=blocks,
        covariance=covariance,
        rank=rank,
        condition_number=float(np.linalg.cond(covariance)),
        method="ordinary least squares, zero intercept, first three nonzero shell means",
        residuals=residuals,
        coefficients=coefficients,
    )


def plot_low_k(ready, output):
    columns = min(3, len(ready))
    rows = int(np.ceil(len(ready) / columns))
    fits, records, fit_points = {}, [], []
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(4 * columns, 3.5 * rows),
            sharex=True,
            layout="constrained",
            squeeze=False,
        )
        for ax, (_, cfg, data) in zip(axes.flat, ready):
            fit = fits[cfg["U"]] = low_k_fits(data)
            k, mean, sem = fit["k"], fit["mean"], fit["sem"]
            selected = (data["k"] > 1e-9) & (data["k"] <= k[-1] + 1e-8)
            ax.scatter(
                data["k"][selected],
                data["N_q"][selected],
                color="0.7",
                s=17,
                alpha=0.5,
                zorder=1,
                label="Individual directions",
            )
            x = np.linspace(0, 1.08 * k[-1], 250)
            ax.errorbar(
                k,
                mean,
                yerr=sem,
                fmt="o",
                color="black",
                markersize=5,
                capsize=3,
                label="Radial mean ± SEM",
                zorder=3,
            )
            ax.plot(
                x,
                fit["coefficients"][1][0] * x,
                color="#2864a5",
                linewidth=2,
                label=r"Linear fit: $ak$",
            )
            ax.plot(
                x,
                fit["coefficients"][2][0] * x**2,
                color="#cf6829",
                linewidth=2,
                linestyle="--",
                label=r"Quadratic fit: $bk^2$",
            )
            ax.set_title(f"U = {cfg['U']:g}")
            ax.set_xlim(0, x[-1])
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.18)
            for n, (radius, value, error) in enumerate(zip(k, mean, sem), 1):
                fit_points.append(
                    dict(
                        U=cfg["U"],
                        point=n,
                        k=radius,
                        N_k=value,
                        block_sem=error,
                        linear_prediction=fit["coefficients"][1][0] * radius,
                        linear_residual=fit["residuals"][1][n - 1],
                        quadratic_prediction=fit["coefficients"][2][0] * radius**2,
                        quadratic_residual=fit["residuals"][2][n - 1],
                    )
                )
            for power, (coefficient, error) in fit["coefficients"].items():
                records.append(
                    dict(
                        U=cfg["U"],
                        power=power,
                        coefficient=coefficient,
                        coefficient_sem=error,
                        shells=len(k),
                        k_max=float(k[-1]),
                        blocks=len(fit["blocks"]),
                        method="unweighted_OLS_zero_intercept",
                        squared_residual_sum=float(
                            fit["residuals"][power] @ fit["residuals"][power]
                        ),
                    )
                )
        for ax in list(axes.flat)[len(ready) :]:
            ax.set_visible(False)
        axes.flat[0].legend(frameon=False, fontsize=8, loc="upper left")
        fig.supxlabel(r"$|k|$")
        fig.supylabel(r"$N(k)$")
        output = Path(output)
        save_plot(fig, output, "charge_low_k_fits", pdf=True)
        with (output / "charge_low_k_fit_coefficients.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        with (output / "charge_low_k_fit_points.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fit_points[0]))
            writer.writeheader()
            writer.writerows(fit_points)
    return fig, fits


def plot_energy(ready, output):
    """Frozen E/N, and exact-in-U re-evaluation of the same saved trial states."""
    ready = sorted(ready, key=lambda item: item[1]["U"])
    U = np.array([cfg["U"] for _, cfg, _ in ready])
    energy = np.array([d["energy_per_site_blocks"].mean() for _, _, d in ready])
    error = np.array(
        [
            d["energy_per_site_blocks"].std(ddof=1)
            / np.sqrt(len(d["energy_per_site_blocks"]))
            for _, _, d in ready
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    axes[0].errorbar(U, energy, yerr=error, fmt="o-", color="black", capsize=3)
    axes[0].set(
        xlabel="U/t",
        ylabel="Frozen-state energy per site",
        title="Saved optimized states",
    )
    rows = []
    for _, cfg, d in ready:
        if cfg["U"] not in [9, 10, 12]:
            continue
        grid = np.linspace(9, 12, 200)
        # E(U) and D are from the same original samples: preserve their covariance.
        blocks = (
            d["energy_per_site_blocks"][:, None]
            + (grid - cfg["U"]) * d["double_occupancy_blocks"][:, None]
        )
        mean = blocks.mean(axis=0)
        sem = blocks.std(axis=0, ddof=1) / np.sqrt(len(blocks))
        axes[1].plot(grid, mean, label=f"Saved U={cfg['U']:g} state")
        axes[1].fill_between(grid, mean - sem, mean + sem, alpha=0.12)
        for target in [9, 10, 12]:
            value = (
                d["energy_per_site_blocks"]
                + (target - cfg["U"]) * d["double_occupancy_blocks"]
            )
            rows.append(
                dict(
                    source_U=cfg["U"],
                    evaluated_U=target,
                    energy_per_site=float(value.mean()),
                    block_sem=float(value.std(ddof=1) / np.sqrt(len(value))),
                )
            )
    selected = np.isin(U, [9, 10, 12])
    axes[1].errorbar(
        U[selected],
        energy[selected],
        yerr=error[selected],
        fmt="ko",
        capsize=3,
        label="Independently trained energies",
    )
    axes[1].set(
        xlabel="Hamiltonian U/t",
        ylabel="Energy per site",
        title="Same trial states; no reoptimization",
    )
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.18)
    output = Path(output)
    save_plot(fig, output, "energy_vs_U", pdf=True)
    with (output / "energy_vs_U.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["U", "energy_per_site", "block_sem"])
        writer.writerows(zip(U, energy, error))
    with (output / "cross_evaluated_energies.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return fig, rows


def plot_dimer_real_space(run, cfg, dimers):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import SymLogNorm

    geom = geometry(run)
    directions = dimers["directions"]
    C = dimers["C"].real
    periods = geom["extent"][:, None] * geom["basis_vectors"]
    _, _, boundary = periodic_display_points(geom["coordinates"], periods)
    maximum = max(float(np.max(abs(C))), 1e-12)
    norm = SymLogNorm(linthresh=1e-3, linscale=1, vmin=-maximum, vmax=maximum, base=10)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), layout="constrained")
    for a, ax in enumerate(axes):
        raw_centers = (
            geom["coordinates"][:, None]
            - geom["coordinates"][0]
            + (directions[None] - directions[a]) / 2
        )
        centers, source, _ = periodic_display_points(
            raw_centers.reshape(-1, 2), periods
        )
        target_direction = directions[source % 3]
        segments = np.stack(
            [centers - target_direction / 2, centers + target_direction / 2], axis=1
        )
        colors = C[:, a, :].reshape(-1)[source]
        reference = np.array([-directions[a] / 2, directions[a] / 2])
        ax.add_patch(Polygon(boundary, fill=False, color="0.6", linewidth=0.8))
        ax.plot(
            *reference.T, color="black", linewidth=7, solid_capstyle="round", zorder=2
        )
        collection = LineCollection(
            segments, array=colors, cmap="RdBu_r", norm=norm, linewidths=2.7, zorder=3
        )
        ax.add_collection(collection)
        ax.scatter(*reference.T, color="black", s=14, zorder=4)
        span = np.max(abs(boundary), axis=0) + 0.75
        ax.set(
            xlim=(-span[0], span[0]),
            ylim=(-span[1], span[1]),
            aspect="equal",
            xlabel=r"$r_x$",
            ylabel=r"$r_y$",
            title=f"Reference orientation {a+1}",
        )
    fig.colorbar(
        collection,
        ax=axes,
        label=r"$C_{ab}(R)$ (symmetric log scale)",
        shrink=0.85,
        pad=0.02,
    )
    fig.suptitle(f"N={cfg['sites']}, U={cfg['U']:g}: connected dimer correlations")
    save_plot(fig, run.parents[1] / "plots", "dimer_real_space", pdf=True)
    return fig


def plot_dimer_momentum(items, output):
    fig, axes = plt.subplots(
        1,
        len(items),
        figsize=(5 * len(items), 4.8),
        layout="constrained",
        squeeze=False,
    )
    for ax, (run, cfg, dimers) in zip(axes.flat, items):
        geom = geometry(run)
        points, source, boundary = periodic_display_points(
            dimers["q"], geom["reciprocal_vectors"]
        )
        values = dimers["S_eigenvalues"][:, -1]
        im = domain_scatter(
            ax,
            points,
            source,
            boundary,
            values,
            title=f"U = {cfg['U']:g}",
            labels=(r"$q_x$", r"$q_y$"),
        )
        fig.colorbar(
            im,
            ax=ax,
            label="Dominant dimer eigenvalue",
            shrink=0.85,
            pad=0.025,
            fraction=0.045,
        )
    save_plot(fig, Path(output), "dimer_dominant_eigenvalue", pdf=True)
    return fig


def plot_dimer_distance(cfg, dimers):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    axes[0].scatter(
        dimers["distance"], dimers["individual_even_leading"], alpha=0.3, s=12
    )
    axes[0].errorbar(
        dimers["shell_distance"],
        dimers["shell_eigenvalues"][:, -1],
        yerr=dimers["shell_eigenvalues_sem"][:, -1],
        fmt="o-",
        capsize=3,
    )
    axes[0].set(
        xlabel="Bond-anchor separation |R|", ylabel="Leading connected dimer eigenvalue"
    )
    v = dimers["long_dominant_vector"].real
    axes[1].bar(range(3), v, color=["C0" if x >= 0 else "C3" for x in v])
    axes[1].set_xticks(range(3), ["(1,0)", "(-1/2,√3/2)", "(-1/2,-√3/2)"])
    axes[1].set(
        ylabel="Signed normalized component",
        title=f"Distance-averaged mode, |R| ≥ {float(dimers['long_rmin']):g}",
    )
    for ax in axes:
        ax.axhline(0, color="gray", lw=0.7)
    fig.suptitle(f"U/t = {cfg['U']:g}")
    return fig
