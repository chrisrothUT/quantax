#!/bin/bash
#SBATCH --job-name=qtx-n48-C0
#SBATCH --partition=gpu
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100_pcie:1
#SBATCH --exclude=workergpu[171-193]
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
cd /mnt/ceph/users/croth/HubbardTriangularNewQuantax
caps=(0.1 0.3 1 2 3 5)
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
srun /mnt/home/croth/miniconda3/envs/quantax_new/bin/python -u train_c0_sweep.py \
  --C0 "${caps[$SLURM_ARRAY_TASK_ID]}" --L 4 --U 8 --samples 4096 \
  --iterations 1000 --eval-sweeps 20 --t0 1000 --T 1000 --output "${RUN_BASE:?}/runs"
