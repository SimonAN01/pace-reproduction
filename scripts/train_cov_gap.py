#!/usr/bin/env python3
"""Reuse the frozen trainer on one explicitly declared, isolated source tree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import train_public


if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a,remaining=p.parse_known_args()
    train_public.REPO=a.source_root.resolve()
    os.environ['PACE_COV_GAP_DIR']=str(a.output.resolve())
    os.environ['PACE_COV_REFERENCE']=str(train_public.ROOT/'pace_repo/checkpoints/COV/PACE/924.pth')
    sys.argv=[sys.argv[0],*remaining,'--output',str(a.output)]
    try:
        train_public.main()
    finally:
        path=a.output.resolve()/'run.json'
        if path.exists():
            meta=json.loads(path.read_text())
            meta['cov_gap_adapter_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            meta['cov_gap_source_root']=str(a.source_root.resolve())
            trace=a.output.resolve()/'forensics/trajectory.json'
            if trace.exists():
                observation=json.loads(trace.read_text());observation['status']=meta['status']
                trace.write_text(json.dumps(observation,indent=2)+'\n')
                meta['cov_gap_trajectory_sha256']=hashlib.sha256(trace.read_bytes()).hexdigest()
            path.write_text(json.dumps(meta,indent=2)+'\n')
