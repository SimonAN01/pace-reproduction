#!/usr/bin/env python3
"""Fetch pinned public data or release weights, and verify every byte."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def valid(path, row):
    return path.is_file() and path.stat().st_size == row['bytes'] and digest(path) == row['sha256']

def obtain(row, local=None, verify_only=False):
    dest = ROOT / row['path']
    if dest.exists():
        if not valid(dest, row): raise RuntimeError(f'Existing file differs, refusing overwrite: {dest}')
        print('VERIFIED', row['path'], flush=True); return
    if verify_only: raise FileNotFoundError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    if local:
        src = local / row['path']
        if not valid(src, row): raise RuntimeError(f'Local source does not match: {src}')
        shutil.copyfile(src, part)
    else:
        offset = part.stat().st_size if part.exists() else 0
        if offset > row['bytes']: raise RuntimeError(f'Partial file too large: {part}')
        if offset < row['bytes']:
            headers = {'User-Agent': 'PACE-reproduction/1.0'}
            if offset: headers['Range'] = f'bytes={offset}-'
            req = urllib.request.Request(row['url'], headers=headers)
            with urllib.request.urlopen(req, timeout=120) as response:
                if response.status == 206:
                    content_range = response.headers.get('Content-Range', '')
                    if not content_range.startswith(f'bytes {offset}-') or not content_range.endswith('/' + str(row['bytes'])):
                        raise RuntimeError(f'Unexpected Content-Range: {content_range}')
                    mode = 'ab' if offset else 'wb'
                elif response.status == 200:
                    mode = 'wb'  # Server ignored Range; restart this file safely.
                else: raise RuntimeError(f'Unexpected HTTP status {response.status}')
                with part.open(mode) as output: shutil.copyfileobj(response, output, 1024 * 1024)
    if not valid(part, row): raise RuntimeError(f'Size/SHA256 mismatch; partial file kept: {part}')
    part.replace(dest)
    print('DOWNLOADED', row['path'], flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind', choices=['data', 'models'])
    p.add_argument('--datasets', nargs='+', choices=['COV','Marine','RRUFF','FunctionalGroups','Bacteria'])
    p.add_argument('--from-local', type=Path, help='Root containing pace_repo/ or reference_models/')
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--repo', default='SimonAN01/pace-reproduction')
    a = p.parse_args()
    if a.kind == 'data':
        rows = json.loads((ROOT/'configs/data_manifest.json').read_text())['files']
        if a.datasets: rows = [r for r in rows if r['path'].split('/')[2] in a.datasets]
    else:
        manifest = json.loads((ROOT/'configs/model_manifest.json').read_text())
        rows = manifest['files']
        for row in rows:
            row['url'] = f"https://github.com/{a.repo}/releases/download/{manifest['tag']}/{Path(row['path']).name}"
    for row in rows: obtain(row, a.from_local, a.verify_only)
    print('PASS', a.kind, len(rows), 'files; all sizes and SHA256 verified')

if __name__ == '__main__': main()
