"""Diagonal charge measurements from frozen-state samples; no retraining needed."""
from pathlib import Path
import csv
import numpy as np


def compute(directory):
    directory = Path(directory)
    data = np.load(directory / 'observables.npz')
    samples = np.load(directory / 'measurement_samples.npy')
    nsites = samples.shape[-1] // 2
    occupation = (samples > 0).astype(np.float64)
    charge = occupation[..., :nsites] + occupation[..., nsites:] - 1
    np.testing.assert_allclose(charge.sum(axis=-1), 0, atol=0)
    phase = np.exp(-1j * data['q_raw'] @ data['site_coordinates'].T)
    np.testing.assert_allclose(phase @ phase.conj().T, nsites*np.eye(nsites), atol=1e-10)
    intensity = abs(charge @ phase.T)**2 / nsites
    blocks = intensity.mean(axis=1)
    mean = blocks.mean(axis=0)
    sem = blocks.std(axis=0, ddof=1)/np.sqrt(len(blocks))
    k = np.linalg.norm(data['q_fold'], axis=1)
    shells, inverse = np.unique(np.round(k, 10), return_inverse=True)
    shell_blocks = np.stack([blocks[:,inverse == i].mean(axis=1) for i in range(len(shells))], axis=1)
    shell_mean = shell_blocks.mean(axis=0)
    shell_sem = shell_blocks.std(axis=0,ddof=1)/np.sqrt(len(blocks))
    counts = np.bincount(inverse)
    # Parseval sum rule checked on every configuration, not only on averages.
    np.testing.assert_allclose(intensity.sum(axis=-1), (charge**2).sum(axis=-1), atol=1e-10)
    np.testing.assert_allclose(mean[k < 1e-10],0,atol=1e-20)
    result = dict(q_raw=data['q_raw'], q_fold=data['q_fold'], k=k, N_q=mean, N_q_sem=sem,
                  block_means=blocks, shell_k=shells, shell_N=shell_mean,
                  shell_sem=shell_sem, shell_count=counts, shell_block_means=shell_blocks)
    np.savez(directory / 'charge_structure_factor.npz', **result)
    for name, header, rows in [
        ('charge_structure_factor.csv', ['qx','qy','k_abs','N_q','N_q_sem'],
         np.column_stack([data['q_fold'],k,mean,sem])),
        ('charge_radial_shells.csv', ['k_abs','N_q','N_q_sem','momentum_count'],
         np.column_stack([shells,shell_mean,shell_sem,counts]))]:
        with (directory/name).open('w',newline='') as f:
            writer=csv.writer(f); writer.writerow(header); writer.writerows(rows)
    return result


def plot_charge(data, title=None):
    import matplotlib.pyplot as plt
    k, y, err = data['shell_k'], data['shell_N'], data['shell_sem']
    keep = k > 1e-9
    k,y,err = k[keep],y[keep],err[keep]
    fig, axes = plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    for ax,power in zip(axes,[0,1,2]):
        mask=data['k']>1e-9
        ax.scatter(data['k'][mask],data['N_q'][mask]/data['k'][mask]**power,
                   s=20,alpha=.45,color='gray',label='Individual momenta')
        ax.errorbar(k,y/k**power,yerr=err/k**power,fmt='o-',capsize=3,
                    label='Radial mean ± block SEM',color='C0')
        ax.set(xlabel=r'$|\mathbf{k}|$ (nearest-neighbor spacing = 1)',
               ylabel=[r'$N(\mathbf{k})$',r'$N(\mathbf{k})/|\mathbf{k}|$',r'$N(\mathbf{k})/|\mathbf{k}|^2$'][power])
        ax.grid(alpha=.2)
    x=np.linspace(0,k[2],100)
    axes[0].plot(x,y[0]*x/k[0],'--',label='Linear guide (first shell)',color='C1')
    axes[0].plot(x,y[0]*(x/k[0])**2,':',label='Quadratic guide (first shell)',color='C2')
    axes[0].legend(fontsize=8)
    axes[1].set_title('Finite small-k limit: metallic scaling')
    axes[2].set_title('Finite small-k limit: insulating scaling')
    fig.suptitle(title or 'N=48, U=8: connected charge structure factor (20,480 saved samples)')
    return fig


def notebook_cells():
    import nbformat as nbf
    text = r'''## Charge structure factor and the metal–insulator diagnostic

For this translation-invariant, half-filled state, $\langle n_j\rangle=1$. We measure
$$N(\mathbf{k})=\frac{1}{48}\left\langle\left|\sum_j e^{-i\mathbf{k}\cdot\mathbf r_j}(n_{j\uparrow}+n_{j\downarrow}-1)\right|^2\right\rangle.$$
This diagonal observable is computed directly from the 20,480 saved measurement configurations. No new training or sampling is involved. The uniform density is subtracted exactly; a noisy sample estimate of the Fourier mean is not subtracted. Fixed total particle number forces $N(0)=0$ for both a metal and an insulator.

The usual zero-temperature diagnostic is $N(k)\propto |k|$ for a conventional metal and $N(k)\propto |k|^2$ for a charge-gapped insulator as $k\to0$ (see [Rüger et al., section 3.1](https://arxiv.org/abs/1311.6504)). The guides are normalized to the first nonzero shell; they are not fits. The smallest accessible momentum here is $\pi/3\simeq1.047$, so this cluster does not resolve the asymptotic limit. Optimization error also remains unquantified; these plots characterize the current variational state.

For this run, the first three nonzero shell values are approximately 0.0580, 0.1172 and 0.1372 at $|k|=1.047$, 1.814 and 2.094. Their growth is closer to linear than quadratic: a descriptive exponent between the first and third shells is about 1.24. This leans toward metallic scaling over the accessible range, but does not establish a metal; a crossover to quadratic behavior below the available momenta remains possible. Larger clusters and better-converged states are needed. A charge structure factor alone also does not distinguish a conventional metal from every possible gapless charge phase.

Gray points retain individual directions. Blue points average equal-$|k|$ momenta within each of the 20 measurement blocks before estimating the SEM, preserving correlations between momenta. Directional variation is not included in that SEM. Blocks may still be correlated. Ratios omit $k=0$.'''
    source = '''from charge_structure_factor import plot_charge
charge = dict(np.load(data_dir / 'charge_structure_factor.npz'))
fig = plot_charge(charge)
plt.show()
print('Charge sum rule Σk N(k) =', charge['N_q'].sum())
print('Smallest nonzero |k| =', charge['shell_k'][1])
print('   |k|           N(k)           SEM          N(k)/k²')
for k, value, sem in zip(charge['shell_k'][1:],charge['shell_N'][1:],charge['shell_sem'][1:]):
    print(f'{k:10.6f}   {value:12.7f}   {sem:10.7f}   {value/k**2:12.7f}')
print('CSV files:', data_dir / 'charge_structure_factor.csv', data_dir / 'charge_radial_shells.csv')'''
    cells=[nbf.v4.new_markdown_cell(text),nbf.v4.new_code_cell(source)]
    for cell in cells: cell.metadata['tags']=['charge-structure-factor']
    return cells


if __name__ == '__main__':
    root=Path(__file__).resolve().parent
    directory=root/'u8_n48_lr003_20261015/observables'
    result=compute(directory)
    fig=plot_charge(result)
    fig.savefig(directory/'charge_structure_factor.png',dpi=180)
    for k,y,e in zip(result['shell_k'],result['shell_N'],result['shell_sem']):
        print(f'{k:.6f} {y:.8f} +/- {e:.8f}')
    print('Sum rule:',result['N_q'].sum())
