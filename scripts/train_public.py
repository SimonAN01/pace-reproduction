#!/usr/bin/env python3
"""Run the upstream trainer with durable provenance and final test evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

from experiment_contract import identity, environment

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'pace_repo'


def sha256(path):
    with open(path, 'rb') as file:
        digest = hashlib.sha256()
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
        return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=['COV', 'Marine', 'RRUFF', 'FunctionalGroups', 'Bacteria'])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--train-size', type=float, default=.8)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--linear-resize', type=int)
    p.add_argument('--output', type=Path)
    p.add_argument('--tune', action='store_true')
    p.add_argument('--base-checkpoint', type=Path)
    p.add_argument('--queue-job-id')
    p.add_argument('--queue-job-sha256')
    p.add_argument('--queue-contract-sha256')
    p.add_argument('--environment-role', choices=['modern', 'legacy'])
    p.add_argument('--evaluate-test', action='store_true',
                   help='Opt in to one final test after selection; omit for sanity/pretraining')
    a = p.parse_args()
    if not 0 < a.train_size < 1 or a.epochs <= 0:
        p.error('--train-size must be between 0 and 1; --epochs must be positive')
    if a.tune and not a.base_checkpoint:
        p.error('--tune requires --base-checkpoint')
    output = a.output or ROOT / 'runs' / (a.dataset + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    mode = 'tune' if a.tune else 'train'
    cmd = [sys.executable, 'main.py', '--' + mode, '--ds', a.dataset,
           '--device', a.device, '--epoch', str(a.epochs), '--seed', str(a.seed),
           '--train_size', str(a.train_size)]
    if a.tune:
        cmd += ['--base_checkpoint', str(a.base_checkpoint.resolve())]
    if a.linear_resize:
        cmd += ['--linear_resize', str(a.linear_resize)]
    source_files = [REPO / 'main.py', REPO / 'config.py', REPO / 'models/PACE.py',
                    REPO / 'utils/utils.py', REPO / 'utils/dataloader.py', REPO / 'utils/transform.py']
    data_files = sorted((REPO / 'datasets' / a.dataset).glob('*.npy'))
    invocation = vars(a).copy()
    invocation = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in invocation.items()}
    metadata = dict(invocation=invocation, environment=environment(), command=cmd, cwd=str(REPO), status='running', final_test_requested=a.evaluate_test,
                    python=platform.python_version(),
                    dependencies={n: importlib.metadata.version(n) for n in ['torch', 'numpy', 'scipy', 'scikit-learn', 'torchvision']},
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    upstream_commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                    code_sha256={str(f.relative_to(REPO)): sha256(f) for f in source_files},
                    data_sha256={f.name: sha256(f) for f in data_files})
    if a.base_checkpoint:
        metadata['base_checkpoint_path'] = str(a.base_checkpoint.resolve())
        metadata['base_checkpoint_sha256'] = sha256(a.base_checkpoint)
    for f in source_files:
        dst = output / 'source' / f.relative_to(REPO)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
    for name in ['train_public.py', 'evaluate.py', 'experiment_contract.py']:
        src = ROOT / 'scripts' / name
        dst = output / 'runner' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        metadata['code_sha256']['scripts/' + name] = sha256(src)
    labels = np.load(REPO / 'datasets' / a.dataset / ('tune_y.npy' if a.tune else 'train_y.npy'))
    train_ids, valid_ids = train_test_split(np.arange(len(labels)), train_size=a.train_size,
                                           random_state=42, stratify=labels if labels.ndim == 1 else None)
    np.savez_compressed(output / 'split.npz', train=train_ids, valid=valid_ids)
    metadata['split'] = dict(random_state=42, train_size=a.train_size,
                            n_train=len(train_ids), n_valid=len(valid_ids), sha256=sha256(output / 'split.npz'))
    try:
        metadata['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                                   '--format=csv,noheader'], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        metadata['gpu'] = None
    metadata_path = output / 'run.json'
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logs = REPO / 'logs' / a.dataset / 'PACE'
    before = set(logs.glob('*_' + mode + '.log'))
    try:
        print('Run directory:', output, flush=True)
        with (output / 'stdout.log').open('w') as stdout:
            child = subprocess.Popen(cmd, cwd=REPO, stdout=stdout, stderr=subprocess.STDOUT)
            metadata['training_pid'] = child.pid
            metadata['training_start_ticks'] = identity(child.pid)
            metadata_path.write_text(json.dumps(metadata, indent=2))
            code = child.wait()
            if code != 0:
                raise subprocess.CalledProcessError(code, cmd)
        new_logs = set(logs.glob('*_' + mode + '.log')) - before
        if len(new_logs) != 1:
            raise RuntimeError(f'Expected one new log, found {len(new_logs)}; avoid concurrent runs of the same dataset')
        log = new_logs.pop()
        shutil.copy2(log, output / 'training.log')
        history = []
        text = log.read_text()
        losses = {int(e): float(l) for e, l in re.findall(r'Epoch\s*(\d+), train loss:([\d.eE+-]+)', text)}
        for epoch, loss, accuracy in re.findall(r'Epoch\s*(\d+), valid loss:([\d.eE+-]+), valid accuracy:([\d.eE+-]+)', text):
            history.append(dict(epoch=int(epoch), train_loss=losses[int(epoch)], val_loss=float(loss), val_accuracy=float(accuracy)))
        if not history:
            raise RuntimeError('No completed validation epochs; inspect stdout.log')
        (output / 'history.json').write_text(json.dumps(history, indent=2))
        if a.tune:
            checkpoint = a.base_checkpoint.resolve().with_suffix('') / 'tune.pth'
        else:
            checkpoint = REPO / 'checkpoints' / a.dataset / 'PACE' / log.name.removesuffix('_train.log')
            checkpoint = checkpoint.with_suffix('.pth')
        shutil.copy2(checkpoint, output / 'checkpoint.pth')
        metadata['checkpoint_sha256'] = sha256(output / 'checkpoint.pth')
        metadata['completed_epochs'] = len(history)
        metadata['best_val_accuracy'] = max(h['val_accuracy'] for h in history)
        selected = None
        for record in history:
            if selected is None or record['val_accuracy'] >= selected['val_accuracy'] + 0.0001:
                selected = record
        metadata['selected_epoch'] = selected['epoch']
        # A single final evaluation after validation selection; fixed upstream test path.
        if a.evaluate_test:
            evaluate_command = [sys.executable, str(ROOT / 'scripts/evaluate.py'), '--dataset', a.dataset,
                            '--checkpoint', str(output / 'checkpoint.pth'), '--output', str(output / 'test'),
                            '--device', a.device]
            if a.linear_resize:
                evaluate_command += ['--linear-resize', str(a.linear_resize)]
            with (output / 'evaluation.stdout.log').open('w') as evaluation_log:
                subprocess.run(evaluate_command, check=True, stdout=evaluation_log, stderr=subprocess.STDOUT)
        metadata['status'] = 'complete'
    except BaseException:
        metadata['status'] = 'failed'
        metadata['error'] = traceback.format_exc()
        raise
    finally:
        metadata['finished_utc'] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
