#!/usr/bin/env python3
"""Sequential execution of either the reported mixed table or unified linear baseline."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','evaluate'])
    p.add_argument('--env220',type=Path,required=True)
    p.add_argument('--env201',type=Path)
    p.add_argument('--protocol',choices=['reported','linear-baseline'],default='reported')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.output.exists():p.error('Output root already exists; choose a fresh directory')
    if a.protocol=='reported' and not a.env201:p.error('reported requires --env201')
    a.output.mkdir(parents=True)
    jobs=[('covid_high' if a.protocol=='reported' else 'covid_linear'), 'marine','rruff','functional_groups']
    if a.mode=='train':jobs.append('bacteria_pretrain')
    jobs.append('bacteria')
    for target in jobs:
        interpreter=a.env201 if target=='covid_high' else a.env220
        command=[str(interpreter.resolve()),str(ROOT/'scripts/run_target.py'),a.mode,target,
                 '--output',str((a.output/target).resolve()),'--device',a.device]
        if target=='bacteria' and a.mode=='train':command += ['--base-checkpoint',str((a.output/'bacteria_pretrain/checkpoint.pth').resolve())]
        print('RUN',target,flush=True);subprocess.run(command,cwd=ROOT,check=True)
    print('PASS: all requested jobs complete; inspect each test/metrics.json (training) or metrics.json (evaluation).')
if __name__=='__main__':main()
