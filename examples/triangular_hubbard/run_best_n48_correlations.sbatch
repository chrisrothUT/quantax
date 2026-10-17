#!/bin/bash
#SBATCH --job-name=qtx-n48-correlations
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100_pcie:1
#SBATCH --exclude=workergpu[171-193]
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=n48_u8_C0_sweep_20261015_202601/correlations-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export MPLBACKEND=Agg
cd /mnt/ceph/users/croth/HubbardTriangularNewQuantax
PYTHON=/mnt/home/croth/miniconda3/envs/quantax_new/bin/python
RUN_ROOT=n48_u8_C0_sweep_20261015_202601
TRAINING_DIR="$RUN_ROOT/runs/C0_1"
srun "$PYTHON" -u measure_momentum.py --run "$TRAINING_DIR" --output "$RUN_ROOT/observables" --blocks 20
export JAX_PLATFORMS=cpu
srun "$PYTHON" -u make_run_notebook.py --run-root "$RUN_ROOT" --training-dir "$TRAINING_DIR"
