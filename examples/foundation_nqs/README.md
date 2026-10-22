# Foundation neural-network quantum states in Quantax

Prototype of a shared wavefunction `psi(spins, couplings)` for a family of
Hamiltonians on one fixed geometry and particle sector. The numerical framework
is inspired by [Rende et al., Nature Communications 16, 7213 (2025)](https://doi.org/10.1038/s41467-025-62098-x).
It is not a port of the authors' pretrained weights or their exact architecture.

From this `foundation-nqs` checkout, install into a separate environment:

```bash
python -m pip install -e '.[full]'
```

Open `examples/foundation_nqs.ipynb` for a small executable training/interpolation
example. A CPU version can be run from the repository root:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 python examples/foundation_nqs/benchmark.py --steps 120 --output /path/to/results
```

The example uses a six-site Ising chain, trains at h/J=0.6,1.0,1.4, and evaluates
0.8 and 1.2 without optimizing those points. Exact Born-weighted enumeration
replaces sampling to separate algorithm correctness from Monte Carlo noise.
Exact diagonalization supplies reference energies; it is not training data.
Quantax uses Pauli matrices in its Ising operator. Restart the notebook kernel
before constructing a different lattice or rerunning the geometry initialization.

## Public interface

```python
model = qtx.model.ConditionalTransformer(
    ncouplings=1, dim=32, heads=4, layers=2, patch_size=2,
)
family = qtx.state.Foundation(model, couplings=[0.8])
conditions = [[0.8], [1.0], [1.2]]
operators = [qtx.operator.Ising(h=g[0]) for g in conditions]
optimizer = qtx.optimizer.FoundationSR(
    family, operators, conditions, C0=0.1,
    updater=qtx.optimizer.AdamUpdater(mu=0.95, beta=0.995),
    clip_mode="median_mad",
)
samplers = [qtx.sampler.LocalFlip(view, nsamples=128)
            for view in optimizer.states]
for iteration in range(100):
    samples = [sampler.sweep() for sampler in samplers]
    family.update(optimizer.get_step(samples))
    print(optimizer.energies, optimizer.variances)
```

Initialize a geometry and dtype before this code, as in any Quantax workflow.
The default model is positive real; choose `complex_output=True` and a complex
Quantax default dtype for sign/phase-structured problems such as Hubbard systems.
Every trainable array is real; the model represents log magnitude and phase with
two real output channels. Inputs follow Quantax's +/-1 occupation convention.

- `family.at(gamma)` returns a persistent **read-only** Variational view. Use one
  sampler per view, and update the family once. Existing samplers see new weights.
- `family.detach(gamma)` produces an independent Variational state for ordinary
  single-Hamiltonian fine-tuning, without modifying the family.
- `qtx.model.Conditioned(your_model, gamma)` also binds any two-input model for
  standalone Variational use. The family API is architecture-agnostic.
- Physical couplings are **not trainable leaves**. Scale them explicitly, e.g.
  encode a triangular Hubbard point as `(U/10, t_prime)`.
- `weights=` in FoundationSR selects the probability of each Hamiltonian. Sample
  counts can differ without changing those probabilities. Positive finite system
  weights are normalized to sum to one.
- `family.coupling_qgt(gamma, spins, weights=None)` returns the coupling-space
  quantum geometric tensor. Its real part is the fidelity metric. Configurations
  must come from the current state at gamma, or use exact Born weights. This
  measures the learned family and can mislead if interpolation is inaccurate.

## The joint SR equation

For each Hamiltonian independently, subtract its weighted means from the
log-derivative matrix O and local energy E. Multiply that system's rows by
`sqrt(p_k * sample_weight / sum(sample_weight))` and concatenate them. One solve
then gives a single update for all model weights:

```
S = sum_k p_k Cov_k(O, O)
F = sum_k p_k Cov_k(O, E)
S * step = F
```

This retains Quantax's imaginary-time convention: the ordinary real energy
gradient is `2*Re(F)`, and `family.update(step)` subtracts the returned step.
It differs both from globally centering a mixture and from averaging separately
solved SR updates. Conditional amplitude/phase gauges must have zero metric.
The implementation uses the existing SR/MinSR solver and updater interfaces.
Optional energy clipping happens independently within each Hamiltonian and
changes the force, not the reported energies/variances. Invalid energies,
Jacobians or sample weights raise an error. `VarE` is the ensemble mean of the
within-system variances, without adding variance between different mean energies.
`C0` bounds the final shared step and is constant in this first implementation.

## Checkpoints

`family.save(path)` writes shared model weights; `optimizer.save(path)` writes
shared updater buffers. Reconstruct the same model and family using `param_file=`
and the same optimizer using `file=`. Recreate the physical coupling order,
Hamiltonians, updater settings and normalizations explicitly: static context
values are not stored as trainable parameters in the Equinox checkpoint.
For an exact MCMC restart also preserve every sampler endpoint, proposal settings,
Quantax's PRNG key, accepted iteration and training history. Restore endpoints
with `sampler.reset(initial_spins=..., nsweeps=0)` and restore the saved RNG after
constructing states/samplers. The exact-sum benchmark does not require Markov
chain checkpoints. Existing Organics checkpoints cannot be loaded directly into
this new architecture.

## Triangular Hubbard starting point

`examples/foundation_nqs/triangular_hubbard.py` provides `build()` for a standard
even L×L triangular torus at balanced half filling. It retains t_prime along x,
uses inputs `(U/10, t_prime)`, creates complex-output states and one ParticleHop
sampler per point. Run it in a fresh interpreter, enable JAX x64 first, and choose
the physical grid deliberately. Thermalize every sampler before optimization.
The helper does not launch training or submit Slurm jobs. Its positive example
values are a template, not a recommendation to restart the canceled campaign.

## Limits of this prototype

- One geometry and Hilbert space; no transfer between N=64 and N=100.
- Global coupling vectors; no efficient site-dependent disorder inputs yet.
- Learned absolute positional embeddings and standard attention, rather than the
  paper's specialized translation-invariant attention. Symmetry is not imposed
  unless a compatible common symmetry is passed to the family explicitly.
- Each distinct coupling tuple has a separate cached compiled state view.
  Per-system sampling and Jacobian evaluation are sequential in Python. This is
  suitable for a modest fixed grid, not the paper's thousand-system throughput.
  Dynamic-context batched evaluation is a further optimization.
- No pretrained triangular Hubbard model is delivered. Generalization across
  a phase boundary and retention of competing minima need scientific validation.
- A complex Transformer in the fermion Fock basis is supported, but it has not
  been benchmarked against the existing Pfaffian/backflow ansatz on large tori.

Tests compare the joint metric/force against explicit weighted covariances,
check exact energy finite differences, conditional gauge cancellation, ordinary
single-system SR equivalence, complex outputs, shared views and optimizer
serialization, and analytic coupling metrics. The small benchmark reports its
actual variational and held-out errors rather than claiming production accuracy.
