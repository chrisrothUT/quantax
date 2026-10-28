#!/bin/bash
#SBATCH --job-name=foundation-square-twist
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --signal=USR1@900
#SBATCH --chdir=/mnt/ceph/users/croth/FoundationTwist/calculation
#SBATCH --output=/mnt/ceph/users/croth/FoundationTwist/calculation/logs/train-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 JAX_ENABLE_X64=true
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export PYTHONPATH=/mnt/ceph/users/croth/FoundationTwist/quantax:/mnt/ceph/users/croth/FoundationTwist/quantax/examples/foundation_flux
export JAX_COMPILATION_CACHE_DIR="$PWD/jax_cache"
PYTHON=/mnt/home/croth/miniconda3/envs/quantax_new/bin/python
SCRIPT=../quantax/examples/foundation_flux/run_training.py
srun "$PYTHON" -u "$SCRIPT" --preflight
if [[ ! -f smoke/summary.json ]]; then
  srun "$PYTHON" -u "$SCRIPT" --smoke --iterations 2
  srun "$PYTHON" -u "$SCRIPT" --smoke --resume --iterations 3
fi
resume=()
if [[ -s run/latest.json ]]; then resume=(--resume); fi
srun "$PYTHON" -u "$SCRIPT" "${resume[@]}" --iterations 20000
"$PYTHON" - <<'PY'
import json, subprocess, os
from pathlib import Path
saved=json.loads(Path('run/latest.json').read_text())
if saved['iteration'] < 20000:
    if saved['reason'] != 'signal':
        raise SystemExit(f'Unexpected early stop: {saved}')
    job=subprocess.check_output(['sbatch','--parsable',f"--dependency=afterok:{os.environ['SLURM_JOB_ID']}",'train.sbatch'],text=True).strip()
    with open('continuations.jsonl','a') as f:
        f.write(json.dumps(dict(job=job,checkpoint=saved))+'\n')
    print('Queued checkpoint continuation', job, flush=True)
PY
