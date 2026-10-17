#!/bin/bash
#SBATCH --job-name=qtx-d32-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100_pcie:2
#SBATCH --exclude=workergpu[171-193]
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:20:00
set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
cd /mnt/ceph/users/croth/HubbardTriangularNewQuantax
srun /mnt/home/croth/miniconda3/envs/quantax_new/bin/python -u train_c0_sweep.py \
  --C0 1 --constant-norm --sampler mixed_exchange --pairing-features 32 --rank-checkpoint-every 2 --L 6 --U 7 --samples 4096 \
  --iterations 3 --clip-mode median_mad --eval-sweeps 2 --output "${RUN_BASE:?}/smoke/train"
