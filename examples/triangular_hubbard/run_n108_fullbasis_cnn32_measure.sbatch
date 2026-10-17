#!/bin/bash
#SBATCH --job-name=qtx-full32-measure
#SBATCH --partition=gpu
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100_pcie:2
#SBATCH --exclude=workergpu[171-193]
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/qtx-matplotlib-${SLURM_JOB_ID}"
cd /mnt/ceph/users/croth/HubbardTriangularNewQuantax
PYTHON=/mnt/home/croth/miniconda3/envs/quantax_new/bin/python
interactions=(7 9)
interaction="${interactions[$SLURM_ARRAY_TASK_ID]}"
run_root="${RUN_BASE:?RUN_BASE required}/u${interaction}_n108_d216_cnn32_mixed70_constantC0_1_s4096"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
srun "$PYTHON" -u measure_momentum.py --run "$run_root/train/C0_1" \
  --output "$run_root/observables" --blocks 20
export JAX_PLATFORMS=cpu
srun "$PYTHON" -u make_run_notebook.py --run-root "$run_root" --training-dir "$run_root/train/C0_1"
