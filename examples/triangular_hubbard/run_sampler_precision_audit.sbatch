#!/bin/bash
#SBATCH --job-name=qtx-sampler-audit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100_pcie:1
#SBATCH --exclude=workergpu[171-193]
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=sampler_precision_audit-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
cd /mnt/ceph/users/croth/HubbardTriangularNewQuantax
for interaction in 7 9; do
srun /mnt/home/croth/miniconda3/envs/quantax_new/bin/python -u audit_sampler_precision.py --run-root "n108_u7_u9_median_mad_20261016_013200/u${interaction}_n108_medianMAD_C0_1_s4096" --output "sampler_precision_audit_${SLURM_JOB_ID}/u${interaction}"
done
