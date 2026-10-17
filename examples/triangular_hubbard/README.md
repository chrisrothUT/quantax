# Triangular Hubbard multiplicative Pfaffian calculations

Research workflow for half-filled TriangularB lattices, N=3*L**2. The current
production runs use U=7/9, CNN width 32 with eight residual blocks, a full pairing
basis d=2*N, 4096 chains, constant C0=1 AdamSR updates, median/MAD energy clipping,
and a mixed nearest-neighbor sampler with 70% site exchange / 30% spin exchange.

`train_c0_sweep.py` writes per-iteration energy/variance to train/C0_1/history.csv.
`--iterations` specifies the full optimization clock; `--stop-after` is an absolute
iteration boundary. `--resume` restores model parameters, optimizer buffers and
clock, sampler configurations, and Quantax PRNG state from a committed resume.json.
Complete continuation states are separate from rank diagnostic checkpoints.
Final frozen evaluation occurs only at the full iteration target. Histories must
match the committed boundary; interruption mid-chunk requires manual recovery
of the history to that boundary before restarting. Resume enforces configuration
and library-source hashes. Preserve the same code/environment for a run's lifetime.

`measure_momentum.py` performs frozen-parameter momentum and correlation
measurements; `make_run_notebook.py` renders the report. Supporting scripts
provide initialization, rank, sampler, and precision diagnostics and earlier sweeps.

## Flatiron Slurm submissions

These batch files preserve the paths used for the experiments. Update the Python
environment path, working directory and RUN_BASE for another checkout/site.
Install this branch of Quantax in the selected environment. The trainer requires
GPUs and disables CPU fallback. One Python process uses all GPUs on its node.

- N=108 (L=6): two H100 PCIe GPUs, two-hour 200-iteration chunks.
- N=192 (L=8): four H100/H200 GPUs on one node; pending training chunks were
  increased to six hours. `submit_n192_fullbasis_chunks.py` submits per-U smoke
  tests, ten dependent 200-iteration arrays, and a final measurement array.
- N=300 (L=10): one gpuxl node, eight H200 GPUs and 72 hours per U. The batch
  script runs ten checkpointed segments within the allocation, then measures.

Array task 0 is U=7; task 1 is U=9. aftercorr dependencies preserve each U's own
chain. A failed or timed-out stage prevents downstream measurements. N=300 is
submitted with `sbatch --export=ALL,RUN_BASE=/absolute/run/path
run_n300_fullbasis_cnn32.sbatch`; first create RUN_BASE/source and copy these
Python modules there (as done for the original runs). Submission helpers create
job manifests; generated data, logs, notebooks and checkpoints are not versioned.

Validation was performed with external model regression checks, optimizer tests,
and a CPU split-versus-uninterrupted training comparison. Production smoke tests
gate the N=192 chains. Machine-specific runtime flags live in the batch scripts.
