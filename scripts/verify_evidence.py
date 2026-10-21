#!/usr/bin/env python3
"""Recompute archived scores from ground-truth labels and audit split/source pins."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parents[1]
def require(condition, message):
    if not condition: raise RuntimeError(message)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--with-data',action='store_true');a=p.parse_args()
    from run_target import validate_code
    validate_code();rows=[]
    for key,t in json.loads((ROOT/'configs/targets.json').read_text()).items():
        base=ROOT/'evidence'/key;meta=json.loads((base/'run.json').read_text())
        require(meta['checkpoint_sha256']==t['checkpoint_sha256'],key+': checkpoint metadata mismatch')
        for rel,expected in t['core_sha256'].items():require(hashlib.sha256((ROOT/t['source']/rel).read_bytes()).hexdigest()==expected,key+': core changed '+rel)
        if a.with_data:
            name='tune_y.npy' if key=='bacteria' else 'train_y.npy'
            labels=np.load(ROOT/'pace_repo/datasets'/t['dataset']/name,allow_pickle=False)
            train,val=train_test_split(np.arange(len(labels)),train_size=t['train_size'],random_state=42,
                                       stratify=labels if labels.ndim==1 else None)
            split=np.load(base/'split.npz',allow_pickle=False)
            require(np.array_equal(train,split['train']) and np.array_equal(val,split['valid']),key+': split differs')
        if t['correct'] is None:continue
        prediction=np.load(base/'test/predictions.npz',allow_pickle=False)
        pred,y=prediction['predictions'],prediction['targets'];logits=prediction['logits']
        require(np.isfinite(logits).all(),key+': nonfinite logits')
        require(np.array_equal(pred,(logits>=0).astype('int64') if y.ndim==2 else logits.argmax(1)),key+': predictions differ from logits')
        if a.with_data:require(np.array_equal(y,np.load(ROOT/'pace_repo/datasets'/t['dataset']/'test_y.npy',allow_pickle=False)),key+': targets differ from public labels')
        correct=int(np.all(pred==y,axis=1).sum()) if y.ndim==2 else int((pred==y).sum())
        require(correct==t['correct'] and len(y)==t['n_test'],key+': score differs')
        metrics=json.loads((base/'test/metrics.json').read_text());require(abs(metrics['accuracy']-correct/len(y))<1e-12,key+': metrics differ')
        rows.append(dict(target=key,correct=correct,n_test=len(y),accuracy=correct/len(y)))
    print(json.dumps(dict(status='PASS',truth=a.with_data and 'public dataset labels' or 'archived targets; add --with-data to verify public labels',rows=rows),indent=2))
if __name__=='__main__':main()
