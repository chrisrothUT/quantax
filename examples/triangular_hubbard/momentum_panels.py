"""Common Brillouin-zone plotting for S(q) and spin-resolved momentum occupations."""

import numpy as np


def plot_momentum(data, output=None, annotate=False):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    basis = np.linalg.lstsq(
        data["site_cell_coordinates"], data["site_coordinates"], rcond=None
    )[0]
    reciprocal = 2 * np.pi * np.linalg.inv(basis).T
    shifts = np.array([(i, j) for i in range(-3, 4) for j in range(-3, 4)]) @ reciprocal
    plot_points, source = [], []
    for k, raw in enumerate(data["q_raw"]):
        candidates = raw[None, :] - shifts
        lengths = np.sum(candidates**2, axis=1)
        for point in candidates[
            np.isclose(lengths, lengths.min(), atol=1e-10, rtol=1e-10)
        ]:
            plot_points.append(point)
            source.append(k)
    points = np.asarray(plot_points)
    source = np.asarray(source)
    angle = np.arange(6) * np.pi / 3
    boundary = np.column_stack([np.cos(angle), np.sin(angle)]) * (4 * np.pi / 3)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    fields = ["S_q", "n_up_q", "n_down_q"]
    titles = [
        r"SU(2)-invariant $S(\mathbf{q})$",
        r"$n_\uparrow(\mathbf{k})$",
        r"$n_\downarrow(\mathbf{k})$",
    ]
    for ax, field, title in zip(axes, fields, titles):
        ax.add_patch(Polygon(boundary, fill=False, color="gray", linestyle="--"))
        colors = data[field][source]
        kwargs = (
            dict(cmap="viridis", vmin=0, vmax=max(float(np.max(data[field])), 1e-12))
            if field == "S_q"
            else dict(cmap="magma", vmin=0, vmax=1)
        )
        dots = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=colors,
            s=105,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
            **kwargs,
        )
        if annotate:
            for point, value in zip(points, colors):
                ax.annotate(
                    f"{value:.2f}",
                    point,
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=6,
                )
        label = "q" if field == "S_q" else "k"
        ax.set(
            title=title,
            xlabel=f"${label}_x$",
            ylabel=f"${label}_y$",
            xlim=(-4.65, 4.65),
            ylim=(-4.5, 4.5),
            aspect="equal",
        )
        ax.annotate(
            r"$\Gamma$", (0, 0), xytext=(5, 6), textcoords="offset points", fontsize=10
        )
        fig.colorbar(
            dots,
            ax=ax,
            label=r"$S(\mathbf{q})$" if field == "S_q" else "Occupation per spin",
        )
    if output is not None:
        fig.savefig(output, dpi=180)
    return fig
