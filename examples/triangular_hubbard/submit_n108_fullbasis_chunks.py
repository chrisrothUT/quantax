"""Submit ten held/dependent arrays, then replace the old pending array safely."""
import argparse
import json
from pathlib import Path
import subprocess


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-base', type=Path, required=True)
    parser.add_argument('--replace-job', required=True)
    args = parser.parse_args()
    base = args.run_base.resolve()
    old = command('scontrol','show','job',args.replace_job)
    if 'JobState=PENDING' not in old or 'JobName=qtx-n108-full32' not in old:
        raise RuntimeError('Replacement target must still be the pending full32 run')
    if list(base.glob('u*/train/C0_1')):
        raise RuntimeError('Production output already exists; refusing duplicate writers')
    manifest = base / 'chunk_jobs.json'
    if manifest.exists():
        raise RuntimeError('Chunk submission manifest already exists')
    jobs = []
    try:
        previous = None
        for end in range(200, 2001, 200):
            options = ['--hold'] if previous is None else [f'--dependency=aftercorr:{previous}']
            job = command('sbatch','--parsable',*options,'--kill-on-invalid-dep=yes',
                          f'--export=ALL,RUN_BASE={base},CHUNK_END={end}',
                          f'--output={base}/chunk-{end:04d}-%A_%a.log',
                          'run_n108_fullbasis_cnn32_chunk.sbatch').split(';')[0]
            jobs.append(dict(job=job, end_iteration=end))
            previous = job
            manifest.write_text(json.dumps(dict(replaces=args.replace_job, jobs=jobs,
                                               status='submitting'),indent=2)+'\n')
        job = command('sbatch','--parsable',f'--dependency=aftercorr:{previous}',
                      '--kill-on-invalid-dep=yes',f'--export=ALL,RUN_BASE={base}',
                      f'--output={base}/measure-%A_%a.log',
                      'run_n108_fullbasis_cnn32_measure.sbatch').split(';')[0]
        jobs.append(dict(job=job, stage='measure'))
        # Check again before cancellation: the held replacement cannot write yet.
        if 'JobState=PENDING' not in command('scontrol','show','job',args.replace_job):
            raise RuntimeError('Original started during replacement; keeping original')
        command('scancel',args.replace_job)
    except Exception:
        for entry in reversed(jobs):
            subprocess.run(['scancel',entry['job']],check=False)
        manifest.write_text(json.dumps(dict(jobs=jobs,status='submission_failed'),indent=2)+'\n')
        raise
    # If release fails, leave the complete replacement held for explicit recovery.
    manifest.write_text(json.dumps(dict(replaces=args.replace_job,jobs=jobs,status='held'),indent=2)+'\n')
    command('scontrol','release',jobs[0]['job'])
    manifest.write_text(json.dumps(dict(replaces=args.replace_job,jobs=jobs,status='submitted'),indent=2)+'\n')
    print(manifest.read_text())


if __name__ == '__main__':
    main()
