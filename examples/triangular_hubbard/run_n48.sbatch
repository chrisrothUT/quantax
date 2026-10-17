#!/bin/bash
#SBATCH --job-name=qtx-u8-n48
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100_pcie:1
#SBATCH --exclude=workergpu[171-193]
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
cd /mnt/ceph/users/croth/HubbardTriangularNewQuantax
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
srun /mnt/home/croth/miniconda3/envs/quantax_new/bin/python -u train_triangular_b.py \
  --index 3 --L 4 --samples 1024 --output "${OUTPUT_ROOT:?OUTPUT_ROOT required}" \
  --iterations "${ITERATIONS:-1000}" --eval-sweeps "${EVAL_SWEEPS:-20}"
