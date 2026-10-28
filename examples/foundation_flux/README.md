# Square Hubbard foundation flux test

6x6 periodic square lattice, t=1, U=4, 30 electrons (15 up, 15 down),
1/6 hole doping relative to half filling. Two conditioning axes are total
boundary twists theta_x and theta_y, in radians. Each uses 20 equally spaced
values in [0, 2pi), for 400 Hamiltonians and 20 samples each (8,000 total).
Uniform Peierls phases exp(i theta_axis/6) preserve translation symmetry;
there is no magnetic flux through individual plaquettes.

Matches the unpinned October 26 organics fermion configuration: 48 CNN channels,
8 residual blocks, pairing dimension 192, independently gated Pfaffian and
symmetric amplitude projections, FoundationSR C0=0.5, learning rate 1,
Adam mu=0.95/beta=0.995, auto_pinv_eig, median-MAD clipping at 5,
70% site exchange / 30% particle hop, 40*N warmup and 4*N sweeps.
CNN parameters are float32; pairing is complex128 to represent twist phases,
with real and imaginary pairing weights stored as real trainable arrays.
No point-group projection or pinning. Seed 3232; target 20,000 iterations.

Run on **two H100 GPUs**, in 8-hour allocations with checkpointed continuation using the batch launcher in the FoundationTwist
calculation directory. It checks the full-size forward pass, then reduced-size
training/restart before production. Set PYTHONPATH to this checkout and this
example directory; FOUNDATION_RUN_DIR defaults to the working directory.
Checkpoints save model, optimizer, sampler configurations, RNG and history
atomically every three iterations or 15 minutes, and at a signal boundary.
Raw sample blocks and per-coupling energies are saved each iteration.

CPU validation: run validate_hamiltonian.py to compare the hopping spectra to
analytic twisted one-particle energies. run_training.py --smoke --iterations 2
checks complex-amplitude training; --smoke --resume --iterations 3 checks restart.
