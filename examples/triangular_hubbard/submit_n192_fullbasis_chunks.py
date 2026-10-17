"""Submit N=192 U=7/U=9 smoke tests and checkpointed production chains."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-base', type=Path)
    args = parser.parse_args()
    base = (args.run_base or Path(f'n192_u7_u9_fullbasis_cnn32_{datetime.now():%Y%m%d_%H%M%S}')).resolve()
    base.mkdir(exist_ok=False)
    jobs = []
    manifest = base / 'chunk_jobs.json'

    def record(status):
        manifest.write_text(json.dumps(dict(run_base=str(base),status=status,jobs=jobs),indent=2)+'\n')

    def submit(stage, script, dependency=None, end=None, hold=False):
        cmd = ['sbatch','--parsable','--kill-on-invalid-dep=yes']
        if dependency:
            cmd.append(f'--dependency=aftercorr:{dependency}')
        if hold:
            cmd.append('--hold')
        exports = f'ALL,RUN_BASE={base}' + (f',CHUNK_END={end}' if end else '')
        label = f'chunk-{end:04d}' if end else stage
        cmd += [f'--export={exports}',f'--output={base}/{label}-%A_%a.log',script]
        job = subprocess.check_output(cmd,text=True).strip().split(';')[0]
        jobs.append(dict(job=job,stage=stage,**(dict(end_iteration=end) if end else {})))
        record('submitting')
        return job

    try:
        previous = submit('smoke','run_n192_fullbasis_cnn32_smoke.sbatch',hold=True)
        for end in range(200,2001,200):
            previous = submit('train','run_n192_fullbasis_cnn32_chunk.sbatch',previous,end)
        submit('measure','run_n192_fullbasis_cnn32_measure.sbatch',previous)
        for name in ['run_n192_fullbasis_cnn32_smoke.sbatch','run_n192_fullbasis_cnn32_chunk.sbatch',
                     'run_n192_fullbasis_cnn32_measure.sbatch','train_c0_sweep.py','training_resume.py',
                     'rank_checkpoints.py','submit_n192_fullbasis_chunks.py']:
            (base/name).write_bytes(Path(name).read_bytes())
        (base/'README.md').write_text('''# N=192 U=7 and U=9 full-basis runs

L=8 TriangularB, half filling (96 up, 96 down), full pairing dimension 384,
CNN width 32, eight residual blocks, 4096 chains, mixed 70% site / 30%
single-spin exchange sampler, constant C0=1, median/MAD clipping, seed 3232.
Fresh initialization, 2000 total iterations, ten 200-iteration chunks.

Each array task uses four H100/H200 GPUs on one node, 16 CPUs, 256 GB RAM.
Both gpu and gpuxl partitions are eligible. Array task 0 is U=7; task 1 is U=9.
A 30-minute smoke test trains two steps, saves a complete continuation boundary,
then resumes for two more steps and performs a two-sweep frozen evaluation.
Production is gated by the matching smoke task using aftercorr dependencies.
Production chunks request four hours; final measurements request six hours.
Actual N=192 performance is unmeasured until the smoke runs.

Each chunk restores model, AdamSR buffers/clock, sampler configurations and
Quantax PRNG state. Frozen energy evaluation (20 sweeps) occurs only after
iteration 2000. Momentum/correlation measurement (20 blocks) and notebook
creation follow in a separate array. See chunk_jobs.json for submitted IDs.
Energy histories are in u{7,9}_n192_d384_cnn32_mixed70_constantC0_1_s4096/train/C0_1/history.csv.

Validation before submission: bash syntax checks and scheduler resource dry run.
Continuation machinery previously passed a fresh-process CPU equivalence test;
the N=192 four-GPU smoke tests additionally gate production for this geometry.
''')
    except Exception:
        for entry in reversed(jobs):
            subprocess.run(['scancel',entry['job']],check=False)
        record('submission_failed')
        raise
    record('held')
    subprocess.run(['scontrol','release',jobs[0]['job']],check=True)
    record('submitted')
    print(manifest.read_text())


if __name__ == '__main__':
    main()
