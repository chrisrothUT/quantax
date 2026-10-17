#!/bin/bash
#SBATCH --job-name=qtx-n300-full32
#SBATCH --partition=gpuxl
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=3-00:00:00
set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/qtx-matplotlib-${SLURM_JOB_ID}"
cd "${RUN_BASE:?RUN_BASE required}/source"
PYTHON=/mnt/home/croth/miniconda3/envs/quantax_new/bin/python
interactions=(7 9)
interaction="${interactions[$SLURM_ARRAY_TASK_ID]}"
run_root="${RUN_BASE}/u${interaction}_n300_d600_cnn32_mixed70_constantC0_1_s4096"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
# Keep all chunks within the same allocation, preserving complete restart states.
for end in {200..2000..200}; do
  resume_args=()
  if (( end > 200 )); then
    resume_args+=(--resume)
  fi
  srun "$PYTHON" -u train_c0_sweep.py --C0 1 --constant-norm \
    --sampler mixed_exchange --cnn-channels 32 --rank-checkpoint-every 500 \
    --L 10 --U "$interaction" --samples 4096 --iterations 2000 \
    --stop-after "$end" "${resume_args[@]}" --clip-mode median_mad \
    --eval-sweeps 20 --output "$run_root/train"
done
srun "$PYTHON" -u measure_momentum.py --run "$run_root/train/C0_1" \
  --output "$run_root/observables" --blocks 20
export JAX_PLATFORMS=cpu
srun "$PYTHON" -u make_run_notebook.py --run-root "$run_root" --training-dir "$run_root/train/C0_1"
