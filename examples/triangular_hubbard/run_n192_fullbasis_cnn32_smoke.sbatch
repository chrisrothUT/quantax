#!/bin/bash
#SBATCH --job-name=qtx-n192-smoke
#SBATCH --partition=gpu,gpuxl
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=h100|h200
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:30:00
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
run_root="${RUN_BASE:?RUN_BASE required}/u${interaction}_n192_d384_cnn32_mixed70_constantC0_1_s4096"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
smoke_root="${RUN_BASE}/smoke/u${interaction}"
common_args=(--C0 1 --constant-norm --sampler mixed_exchange --cnn-channels 32
  --rank-checkpoint-every 2 --L 8 --U "$interaction" --samples 4096
  --iterations 4 --clip-mode median_mad --eval-sweeps 2 --output "$smoke_root/train")
srun "$PYTHON" -u train_c0_sweep.py "${common_args[@]}" --stop-after 2
srun "$PYTHON" -u train_c0_sweep.py "${common_args[@]}" --resume --stop-after 4
