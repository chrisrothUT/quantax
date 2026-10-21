# Anisotropic triangular Hubbard workflows

These scripts preserve the fixes used for the Organics calculations: horizontal t′ hopping, a true minimum-image distance envelope on skew periodic cells, complete restart states, original-block correlation estimators, symmetric periodic display boundaries, and unweighted charge fits to exactly the first three nonzero shell means.

## Training and restart

Install this branch and its full dependencies. Run on two GPUs with JAX x64 enabled by the driver. For example, from a persistent experiment directory:

```sh
python /path/to/quantax/examples/organics/source/train.py   --L 4 --U 10 --t-prime 0.8 --cnn-channels 32 --pairing-features 96   --C0 0.5 --constant-norm --sampler mixed_exchange --clip-mode median_mad   --seed 3233 --samples 4096 --iterations 2000 --eval-sweeps 4   --max-parallel 256 32 256 --output /shared/experiment/seed3233/train
```

The resulting run directory is `/shared/experiment/seed3233/train/C0_0.5`. Use `--resume` with identical arguments to recover the last committed boundary. Full model, optimizer, sampler configurations, RNG state, iteration and history are committed every 25 steps or 30 minutes. `checkpoint_history.py` archives and truncates uncommitted history before continuation. Final evaluation retains four independent temporal blocks of 4,096 configurations and their raw local energies. Do not treat the training chain endpoints as a new frozen-state evaluation.

`corrected_distance_envelope.py` changes only static distances on a copy of the model. Its explicit policy and source hash are checked when reconstructing a saved model. This prevents silently using a different envelope for an old checkpoint. Keep each experiment's source snapshot and package versions with its outputs; formatting changes also change these provenance hashes. `preflight.py` checks hopping geometry and exact one-particle spectra. Historical size-transfer and rank-probe options in the driver retain their original specialized helpers; the command above uses fresh initialization.

## Correlations

Use the scripts from this checkout or a matching frozen snapshot. Each `RUN` argument below denotes the final `train/C0_...` directory:

```sh
python source/summarize_observables.py "$RUN"
# This direct SU(2) measurement requires exactly two allocated GPUs.
python analysis_source/measure_su2.py --run "$RUN"
python analysis_source/build_analysis.py "$RUN"
python analysis_source/derive_dimers.py --run "$RUN"
```

Run these commands from `examples/organics`. SU(2) measurements include both transverse terms and retain independent full site-pair local estimators incrementally. Dimer analysis uses their Hermitian Gram products on exactly the same original blocks, preserving full site/bond tensors before translation averaging. Disconnected subtraction and nonlinear eigenvalue errors use a delete-one-original-block jackknife. No rotational or SU(2) reconstruction is substituted for direct measurement.

## Reports

`plot_helpers.py` provides training curves with a final-quarter zoom, full spin/charge tensors and structure factors, periodic boundary display copies, connected dimer plots, E(U), and cross-evaluated saved-state energies. Low-k fits use ordinary least squares through zero on exactly three radial-shell means. Fit coordinates, predictions and residuals are exported; display copies never enter statistics.

`make_notebook.py --root EXPERIMENT --execute` reproduces the original coupling scan. Its `experiment.json` needs `U_order`, and it expects the original `u{U}_n48_tp0.8_cnn32_d96_seed3232_C0_1_s4096/train/C0_1` layout. `refresh_reports.py --root EXPERIMENT` renders that combined report and all per-coupling reports.

For four-seed C0=0.5 experiments, run `compare_seeds.py --root EXPERIMENT --execute`. Its `experiment.json` needs `U` and `runs`, with each run containing `seed`, an absolute `run` path and `existing=false`. Entries marked `existing=true` are excluded. The report shows energies, common-scale structure factors and descriptive pairwise pattern differences without assigning thermodynamic phases.

Report execution additionally requires matplotlib, scipy, nbformat, nbclient, IPython and a `quantax_new` Jupyter kernel pointing to the intended environment. The notebooks contain no embedded experiment data until executed. This directory deliberately excludes job IDs, cluster-specific batch scripts, model weights, sample arrays, and rendered notebook outputs. Submit training and measurement jobs separately; choose resource limits for the local cluster.
