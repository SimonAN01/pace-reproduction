#!/usr/bin/env python3
"""Run a named historical protocol without changing its frozen trainer."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
TARGETS = json.loads((ROOT/'configs/targets.json').read_text())

def validate_code():
    for rel, expected in json.loads((ROOT/'configs/frozen_code.json').read_text()).items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen code changed: '+rel)

def prepare():
    # Relative symlink gives COVID the exact same arrays. No .git is copied.
    link = ROOT/'covid_source/datasets'
    if not link.exists(): link.symlink_to('../pace_repo/datasets', target_is_directory=True)
    if link.resolve() != (ROOT/'pace_repo/datasets').resolve(): raise RuntimeError('Unexpected COVID data link')
    for source in ['pace_repo','covid_source']:
        for name in ['logs','checkpoints']: (ROOT/source/name).mkdir(parents=True,exist_ok=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','evaluate'])
    p.add_argument('target',choices=list(TARGETS))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--base-checkpoint',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--epochs',type=int,help='Sanity override; no final test and not a reference protocol')
    p.add_argument('--input-pool',type=int,choices=[256,1024],default=1024)
    p.add_argument('--allow-environment-drift',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(); t=TARGETS[a.target];validate_code()
    try:
        execution_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        p.error('Use git clone of the published repository; a Git HEAD is required by the frozen runners')
    if a.output.exists(): p.error('Output exists; use a fresh directory')
    expected=json.loads((ROOT/'configs/environments.json').read_text())[t['env']]
    actual={'python':platform.python_version(),**{k:importlib.metadata.version(k) for k in expected if k!='python'}}
    if actual!=expected and not a.allow_environment_drift: p.error(f'Environment mismatch: expected {expected}; got {actual}')
    if a.epochs is not None and a.epochs<=0:p.error('--epochs must be positive')
    if a.target=='bacteria_pretrain' and a.epochs is not None:p.error('Upstream Bacteria pretraining forces 50 epochs; sanity override is unsupported')
    if a.mode=='evaluate' and a.target=='bacteria_pretrain':p.error('Pretraining has no registered final-test score')
    if a.mode=='evaluate' and a.epochs is not None:p.error('--epochs is only for training sanity')
    if a.mode=='train':
        runner='train_cov_gap.py' if a.target=='covid_high' else 'train_public.py'
        command=[sys.executable,str(ROOT/'scripts'/runner)]
        if a.target=='covid_high':command += ['--source-root',str(ROOT/'covid_source')]
        command += ['--dataset',t['dataset'],'--epochs',str(a.epochs or t['epochs']),'--seed',str(t['seed']),
                    '--train-size',str(t['train_size']),'--device',a.device,'--output',str(a.output.resolve())]
        if t['linear_resize']:command += ['--linear-resize',str(t['linear_resize'])]
        if a.target=='bacteria':
            if not a.base_checkpoint:p.error('bacteria requires --base-checkpoint from bacteria_pretrain')
            command += ['--tune','--base-checkpoint',str(a.base_checkpoint.resolve())]
        if a.target!='bacteria_pretrain' and a.epochs is None:command += ['--evaluate-test']
    else:
        checkpoint=a.checkpoint or ROOT/'reference_models'/(a.target+'.pth')
        if not checkpoint.is_file():p.error('Missing checkpoint; run bootstrap.py models or pass --checkpoint')
        command=[sys.executable,str(ROOT/'scripts/evaluate.py'),'--dataset',t['dataset'],'--checkpoint',str(checkpoint.resolve()),
                 '--output',str(a.output.resolve()),'--device',a.device,'--input-pool',str(a.input_pool)]
        if t['linear_resize']:command += ['--linear-resize',str(t['linear_resize'])]
    record=dict(target=a.target,mode=a.mode,command=command,environment=actual,expected_environment=expected,
                environment_drift=actual!=expected,sanity=a.epochs is not None,output=str(a.output.resolve()),
                created_utc=datetime.now(timezone.utc).isoformat(),execution_repository_commit=execution_commit,upstream_commit='4e1f541b5776e7fd1df14fce271b11a5cbfe3f87')
    print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
    if a.dry_run:return
    if a.mode=='train':
        needed=t['dataset'];subprocess.run([sys.executable,str(ROOT/'scripts/bootstrap.py'),'data','--datasets',needed,'--verify-only'],check=True)
        if a.base_checkpoint and not a.base_checkpoint.is_file():p.error('Base checkpoint missing')
    else:
        rows=json.loads((ROOT/'configs/data_manifest.json').read_text())['files']
        from bootstrap import valid
        needed={f"pace_repo/datasets/{t['dataset']}/"+s for s in ['test_x.npy','test_y.npy']}
        if t['dataset']!='FunctionalGroups':needed.add(f"pace_repo/datasets/{t['dataset']}/label.json")
        for row in rows:
            if row['path'] in needed and not valid(ROOT/row['path'],row):raise RuntimeError('Evaluation data missing or differs: '+row['path'])
    prepare();(ROOT/'.runtime').mkdir(exist_ok=True)
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'scripts')+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    # Keep the historical thread/backend defaults; do not add RNG calls here.
    with (ROOT/'.runtime/gpu.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if a.output.exists():raise RuntimeError('Output created while waiting for lock')
        record_path=ROOT/'.runtime'/('launch_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')
        record_path.write_text(json.dumps(record,indent=2)+'\n')
        result=subprocess.run(command,cwd=ROOT,env=env)
        record['exit_code']=result.returncode;record_path.write_text(json.dumps(record,indent=2)+'\n')
        result.check_returncode()
        if a.output.is_dir():
            # Frozen historical runners discover this publication Git HEAD; retain it
            # separately and give the upstream field its explicit contracted meaning.
            for rel in ['run.json','test/metrics.json','metrics.json']:
                metadata_path=a.output/rel
                if metadata_path.is_file():
                    metadata=json.loads(metadata_path.read_text())
                    metadata['execution_repository_commit']=metadata.get('upstream_commit',execution_commit)
                    metadata['upstream_commit']=record['upstream_commit']
                    metadata['publication_metadata_normalized']=True
                    metadata_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
            (a.output/'publication_launch.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
