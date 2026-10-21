"""Connected spin-dimer tensor, distance modes and bond-center structure factors."""

from pathlib import Path
import json
import numpy as np
from dimer_kernels import connected, fold_center


def hermitian(matrix):
    return (matrix + matrix.conj().swapaxes(-1, -2)) / 2


def analyze(directory, rmin=2.0):
    directory = Path(directory)
    if not (directory / "blocks.npz").exists():
        raise RuntimeError("Dimer measurement has not finished")
    f = np.load(directory / "blocks.npz")
    means = f["means"]
    raw = f["raw_structure"]
    phase = f["phase"]
    n = len(raw)
    distance = f["distance"]
    radii, which = np.unique(np.round(distance, 10), return_inverse=True)
    keep = distance >= rmin - 1e-10

    def statistics(mean, s):
        c, s = connected(mean, s, phase)
        center = hermitian(fold_center(s, f["q"]))
        shell = np.array(
            [hermitian(c[which == i].mean(axis=0)) for i in range(len(radii))]
        )
        long = hermitian(c[keep].mean(axis=0))
        return dict(
            C=c,
            S_anchor=s,
            S_center=center,
            shell_C=shell,
            long_C=long,
            shell_eigenvalues=np.linalg.eigvalsh(shell),
            long_eigenvalues=np.linalg.eigvalsh(long),
            S_eigenvalues=np.linalg.eigvalsh(center),
            S_total=center.sum(axis=(-1, -2)).real / 3,
            S_orientation_average=np.trace(center, axis1=-1, axis2=-2).real / 3,
        )

    mean = means.mean(axis=0)
    s = raw.mean(axis=0)
    d = statistics(mean, s)
    jk = [
        statistics(
            (means.sum(axis=0) - means[i]) / (n - 1),
            (raw.sum(axis=0) - raw[i]) / (n - 1),
        )
        for i in range(n)
    ]
    for key in list(d):
        v = np.array([x[key] for x in jk])
        d[key + "_sem"] = np.sqrt(
            (n - 1) / n * np.sum(abs(v - v.mean(axis=0)) ** 2, axis=0)
        )
    d.update(
        mean_bond=mean,
        C_raw=np.einsum("qr,qab->rab", phase.conj(), s) / phase.shape[1],
        S_raw_anchor=s,
        S_raw_center=fold_center(s, f["q"]),
        shell_distance=radii,
        distance=distance,
        q=f["q"],
        directions=f["directions"],
        long_rmin=np.array(rmin),
    )
    _, vec = np.linalg.eigh(d["long_C"])
    v = vec[:, -1]
    v = v * np.exp(-1j * np.angle(v[np.argmax(abs(v))]))
    d["long_dominant_vector"] = v
    _, vec = np.linalg.eigh(d["S_center"])
    v = vec[:, :, -1]
    v *= np.exp(-1j * np.angle(v[np.arange(len(v)), np.argmax(abs(v), axis=1)]))[
        :, None
    ]
    d["S_dominant_vectors"] = v
    d["individual_even_leading"] = np.linalg.eigvalsh(hermitian(d["C"]))[:, -1]
    np.testing.assert_allclose(
        d["C"][f["inverse"]], d["C"].swapaxes(-1, -2).conj(), atol=1e-10
    )
    np.testing.assert_allclose(d["S_anchor"].mean(axis=0), d["C"][0], atol=1e-10)
    if np.linalg.eigvalsh(hermitian(d["S_anchor"])).min() < -1e-10:
        raise ValueError("Structure factor is not positive semidefinite")
    residual = f["second_moment_identity_residual"]
    peak = int(np.argmax(d["S_eigenvalues"][:, -1]))
    np.savez(
        directory / "dimer_correlations.npz",
        **d,
        mean_blocks=means,
        S_raw_blocks=raw,
        selected_blocks=f["selected_blocks"],
        selected_chains=f["selected_chains"],
    )
    np.savetxt(
        directory / "dimer_distance_eigenvalues.csv",
        np.column_stack([radii, d["shell_eigenvalues"], d["shell_eigenvalues_sem"]]),
        delimiter=",",
        header="distance,lambda1,lambda2,lambda3,sem1,sem2,sem3",
        comments="",
    )
    np.savetxt(
        directory / "dimer_structure_factor.csv",
        np.column_stack(
            [
                f["q"],
                d["S_total"],
                d["S_total_sem"],
                d["S_orientation_average"],
                d["S_eigenvalues"][:, -1],
                d["S_eigenvalues_sem"][:, -1],
            ]
        ),
        delimiter=",",
        header="qx,qy,S_total,sem_total,S_orientation_average,lambda_max,sem_lambda_max",
        comments="",
    )
    summary = dict(
        samples=16384,
        tensor_shape=list(d["C"].shape),
        operator="B_i,a = S_i dot S_(i+delta_a)",
        connected=True,
        structure_normalization="S_anchor(q)=sum_R exp(-iq.R) C(R); S_total=sum_ab S_center,ab/3",
        mean_bond=mean.real.tolist(),
        long_eigenvalues=d["long_eigenvalues"].tolist(),
        long_eigenvalue_sem=d["long_eigenvalues_sem"].tolist(),
        long_vector_real=d["long_dominant_vector"].real.tolist(),
        long_vector_imag=d["long_dominant_vector"].imag.tolist(),
        peak_q=f["q"][peak].tolist(),
        peak_eigenvalue=float(d["S_eigenvalues"][peak, -1]),
        peak_sem=float(d["S_eigenvalues_sem"][peak, -1]),
        spin_identity_residual=residual.mean(axis=0).tolist(),
        spin_identity_residual_sem=(residual.std(axis=0, ddof=1) / np.sqrt(n)).tolist(),
    )
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return d
