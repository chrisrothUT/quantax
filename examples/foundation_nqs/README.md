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
import numpy as np
# Initialize a periodic spinful triangular lattice at balanced half filling first.
H = qtx.operator.Anisohub(
    U=np.linspace(4, 16, 20), t_prime=np.linspace(0.8, 1, 20),
)
model = qtx.model.ConditionalTransformer(
    2, dim=32, heads=4, layers=2, complex_output=True,
    coupling_scale=(10, 1),
)
family = qtx.state.Foundation(model, H.couplings[0])
optimizer = qtx.optimizer.FoundationSR(family, H, C0=0.1)
sampler = qtx.sampler.FamilyMetropolis(
    family, H.couplings, samples_per_coupling=128,
    kind="particle_hop", thermal_steps=100,
)
for iteration in range(100):
    samples = sampler.sweep()  # (400, 128, Nmodes)
    family.update(optimizer.get_step(samples))
    print(optimizer.energies.reshape(H.grid_shape))
```

Initialize a geometry and dtype before this code, as in any Quantax workflow.
The default model is positive real; choose `complex_output=True` and a complex
Quantax default dtype for sign/phase-structured problems such as Hubbard systems.
Every trainable array is real; the model represents log magnitude and phase with
two real output channels. Inputs follow Quantax's +/-1 occupation convention.

- `family.at(gamma)` returns a persistent **read-only** Variational view. Use one
  sampler per view for legacy workflows. The batched family path needs no views.
- `family.detach(gamma)` produces an independent Variational state for ordinary
  single-Hamiltonian fine-tuning, without modifying the family.
- `qtx.model.Conditioned(your_model, gamma)` also binds any two-input model for
  standalone Variational use. The family API is architecture-agnostic.
- Physical couplings are **not trainable leaves**. Keep physical `(U, t_prime)` inputs and
  set `coupling_scale=(10, 1)` in ConditionalTransformer.
- `weights=` in FoundationSR selects the probability of each Hamiltonian. Sample
  counts can differ in the legacy list interface; the batched path uses equal
  counts per point. Positive finite system
  weights are normalized to sum to one.
- `family.coupling_qgt(gamma, spins, weights=None)` returns the coupling-space
  quantum geometric tensor. Its real part is the fidelity metric. Configurations
  must come from the current state at gamma, or use exact Born weights. This
  measures the learned family and can mislead if interpolation is inaccurate.


## Batched operator basis

`Anisohub` builds `H = t*T_diagonal + t_prime*T_x + U*D`. Two one-dimensional
axes produce their Cartesian product (U first, t_prime second); use
`cartesian=False` for paired broadcastable inputs. The distinct bond is along
physical x. `H.local_terms(family, samples)` returns `(R,S,3)` local energies
from one batched application of the fixed operator basis. `H.combine(terms)`
forms `(R,S)` Hamiltonian local energies using each row's coefficients. The
optimizer reports per-point `energies`, `variances` and `term_energies`.

Each ratio uses the wavefunction conditioned on that sample's own coupling.
The operator connectivity and compiled kernels are shared; values for different
wavefunctions are evaluated separately along batch axes. Variances use the
combined local energies, preserving covariance between hopping and occupancy.
`HamiltonianFamily(terms, coefficients, couplings)` provides the same mechanism
for other linearly parameterized models. `sample_chunk` bounds configurations
per device in each connection batch; Foundation's `max_parallel` bounds neural forward/Jacobian
chunks. On multiple devices total `R*S` must divide evenly over devices.

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
with `sampler.reset(initial_spins=...)` for FamilyMetropolis and restore the saved RNG after
constructing states/samplers. The exact-sum benchmark does not require Markov
chain checkpoints. Existing Organics checkpoints cannot be loaded directly into
this new architecture.

## Triangular Hubbard starting point

`examples/foundation_nqs/triangular_hubbard.py` provides `build()` for a standard
even L×L triangular torus at balanced half filling. It retains t_prime along x,
uses physical inputs `(U, t_prime)` with model normalization, creates complex-output
states and one FamilyMetropolis sampler across the entire grid. Run it in a fresh interpreter, enable JAX x64 first, and choose
the physical grid deliberately. Thermalize every sampler before optimization.
The helper does not launch training or submit Slurm jobs. Its positive example
values are a template, not a recommendation to restart the canceled campaign.

## Limits of this prototype

- One geometry and Hilbert space; no transfer between N=64 and N=100.
- Global coupling vectors; no efficient site-dependent disorder inputs yet.
- Learned absolute positional embeddings and standard attention, rather than the
  paper's specialized translation-invariant attention. Symmetry is not imposed
  unless a compatible common symmetry is passed to the family explicitly.
- Work and Jacobian storage still scale with total samples across the grid.
  Couplings are dynamic arrays: sampling, forward passes and Jacobians share
  compiled kernels. Physical configurations and connected amplitudes are chunked
  on device; there is no Python evaluation loop per Hamiltonian.
- No pretrained triangular Hubbard model is delivered. Generalization across
  a phase boundary and retention of competing minima need scientific validation.
- A complex Transformer in the fermion Fock basis is supported, but it has not
  been benchmarked against the existing Pfaffian/backflow ansatz on large tori.

Tests compare the joint metric/force against explicit weighted covariances,
check exact energy finite differences, conditional gauge cancellation, ordinary
single-system SR equivalence, complex outputs, shared views and optimizer
serialization, and analytic coupling metrics. The small benchmark reports its
actual variational and held-out errors rather than claiming production accuracy.
