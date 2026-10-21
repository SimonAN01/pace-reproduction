#!/usr/bin/env python3
"""Strict evaluation of the upstream PACE model; never selects by test score."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from sklearn import metrics

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'pace_repo'
sys.path.insert(0, str(REPO))
from models.PACE import PACE
from utils.dataloader import myDataset, valid_transform, bacteria_valid_transform
from torch.utils.data import DataLoader


def sha256(path):
    with open(path, 'rb') as file:
        digest = hashlib.sha256()
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
        return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=['COV', 'Bacteria', 'Marine', 'RRUFF', 'FunctionalGroups'])
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--input-pool', type=int, default=1024,
                   help='1024 matches upstream test_model, 256 matches upstream train-time test')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--linear-resize', type=int,
                   help='Opt-in paper protocol: linear interpolation before pooling, align_corners=False')
    p.add_argument('--dim', type=int, default=1024)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--patch-size', type=int, default=16)
    p.add_argument('--pool-dim', type=int, default=256)
    p.add_argument('--kernel-size', type=int, default=9)
    p.add_argument('--allow-unused-wave-emb', action='store_true',
                   help='Explicitly discard only the legacy wave_emb keys absent from the published model')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    data = REPO / 'datasets' / a.dataset
    xfile, yfile = data / 'test_x.npy', data / 'test_y.npy'
    x, y = np.load(xfile, allow_pickle=False), np.load(yfile, allow_pickle=False)
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError('Invalid spectra or label cardinality')
    multilabel = a.dataset == 'FunctionalGroups'
    classes = 17 if multilabel else len(json.loads((data / 'label.json').read_text()))
    if multilabel:
        if y.shape != (len(x), classes) or not np.isin(y, [0, 1]).all():
            raise ValueError('Expected binary multilabel ground truth')
    elif not np.equal(y, y.astype('int64')).all() or y.min() < 0 or y.max() >= classes:
        raise ValueError('Invalid class indices')
    config = dict(dim=a.dim, depth=a.depth, kernel_size=a.kernel_size,
                  patch_size=a.patch_size, pool_dim=a.pool_dim, n_classes=classes)
    model = PACE(**config).to(a.device)
    # Strict loading catches absent/unexpected keys unlike upstream load_net_state.
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    ignored = {}
    if a.allow_unused_wave_emb:
        legacy_keys = {'wave_emb.0.weight', 'wave_emb.0.bias', 'wave_emb.1.weight',
                       'wave_emb.1.bias', 'wave_emb.1.running_mean',
                       'wave_emb.1.running_var', 'wave_emb.1.num_batches_tracked'}
        ignored = {k: list(state[k].shape) for k in legacy_keys if k in state}
        state = {k: v for k, v in state.items() if k not in legacy_keys}
    model.load_state_dict(state, strict=True)
    transform = bacteria_valid_transform if a.dataset == 'Bacteria' else valid_transform
    dataset = myDataset(x, y, transform=transform, pool_dim=a.input_pool or None, resize_dim=a.linear_resize)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=False, num_workers=0)
    model.eval()
    outputs = []
    with torch.inference_mode():
        for inputs, _ in loader:
            outputs.append(model(inputs.to(a.device)).cpu().numpy())
    logits = np.concatenate(outputs)
    if not np.isfinite(logits).all():
        raise ValueError('Nonfinite model outputs')
    predictions = (logits >= 0).astype('int64') if multilabel else logits.argmax(1)
    result = {
        'dataset': a.dataset, 'n_test': len(y), 'accuracy': metrics.accuracy_score(y, predictions),
        'correct': int(np.all(predictions == y, axis=1).sum()) if multilabel else int((predictions == y).sum()),
        'metric': 'subset_accuracy' if multilabel else 'accuracy',
        'macro_f1': metrics.f1_score(y, predictions, average='macro', zero_division=0),
        'micro_f1': metrics.f1_score(y, predictions, average='micro', zero_division=0),
        'model': config, 'input_pool': a.input_pool,
        'linear_resize': a.linear_resize, 'interpolation_align_corners': False if a.linear_resize else None,
        'explicitly_ignored_legacy_keys': ignored,
        'checkpoint': str(a.checkpoint.resolve()), 'checkpoint_sha256': sha256(a.checkpoint),
        'data_sha256': {f.name: sha256(f) for f in [xfile, yfile]},
        'upstream_commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        'python': platform.python_version(), 'device': a.device,
        'dependencies': {name: importlib.metadata.version(name) for name in ['torch', 'torchvision', 'numpy', 'scikit-learn', 'scipy']},
        'classification_report': metrics.classification_report(y, predictions, output_dict=True, zero_division=0),
    }
    (a.output / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    np.savez_compressed(a.output / 'predictions.npz', logits=logits, predictions=predictions, targets=y)
    if not multilabel:
        np.savetxt(a.output / 'confusion_matrix.csv', metrics.confusion_matrix(y, predictions, labels=np.arange(classes)), fmt='%d', delimiter=',')
    print(json.dumps({k: result[k] for k in ['dataset', 'correct', 'n_test', 'accuracy', 'macro_f1', 'input_pool']}))


if __name__ == '__main__':
    main()
