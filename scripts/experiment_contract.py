"""Shared local experiment identity helpers; no training behavior."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE = ['main.py', 'config.py', 'models/PACE.py', 'utils/utils.py', 'utils/dataloader.py', 'utils/transform.py']
PACKAGES = ['torch', 'numpy', 'scipy', 'scikit-learn', 'torchvision']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def environment():
    return dict(executable=os.path.abspath(sys.executable), prefix=sys.prefix, python=platform.python_version(),
                dependencies={name: importlib.metadata.version(name) for name in PACKAGES})


if __name__ == '__main__':
    print(json.dumps(environment()))
