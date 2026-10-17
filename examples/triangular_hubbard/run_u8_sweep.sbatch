#!/bin/bash
#SBATCH --job-name=qtx-u8-adamsr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=workergpu[171-193]
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
cd /mnt/ceph/users/croth/HubbardTriangularNewQuantax
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
srun /mnt/home/croth/miniconda3/envs/quantax_new/bin/python -u sweep_u8.py \
  --index "${SLURM_ARRAY_TASK_ID}" --output "${RUN_ROOT:?RUN_ROOT required}" \
  --iterations "${ITERATIONS:-1000}" --eval-sweeps "${EVAL_SWEEPS:-20}"
