#!/bin/bash
#SBATCH --job-name=qtx-init-audit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100_pcie:1
#SBATCH --exclude=workergpu[171-193]
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=initialization_audit-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
cd /mnt/ceph/users/croth/HubbardTriangularNewQuantax
for sites in 48 108; do
srun /mnt/home/croth/miniconda3/envs/quantax_new/bin/python -u audit_initialization.py --sites "$sites" --output "initialization_audit_${SLURM_JOB_ID}"
done
