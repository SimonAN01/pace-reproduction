"""Read-only training-state observations; no forward/backward/optimizer changes."""
from datetime import datetime, timezone
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch


def rng():
    return (torch.get_rng_state().clone(),
            [v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [])


def same_rng(a, b):
    return torch.equal(a[0], b[0]) and len(a[1]) == len(b[1]) and all(torch.equal(x,y) for x,y in zip(a[1],b[1]))


def install():
    from utils.utils import Engine
    target = os.environ.get('PACE_COV_GAP_DIR')
    if not target:
        raise RuntimeError('PACE_COV_GAP_DIR is required for the observation hook')
    out = Path(target)/'forensics'
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__,out/'cov_gap_hook.py')
    before = rng()
    reference_path = Path(os.environ['PACE_COV_REFERENCE'])
    reference = torch.load(reference_path,map_location='cpu',weights_only=True)
    record = dict(status='observing', created_utc=datetime.now(timezone.utc).isoformat(),
        reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_arrays_or_labels_read=False,optimizer_step_modified=False,
        snapshots={},epochs=[],rng_unchanged_by_observations=True)
    original = Engine.train_epoch
    start = {}
    def write():
        temp = out/'trajectory.tmp'
        temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(out/'trajectory.json')
    def snapshot(engine, counter):
        state={k:v.detach().cpu().clone() for k,v in engine.model.state_dict().items()}
        path=out/f'step_{counter}.pth';torch.save(state,path)
        main_rng=rng()
        rng_files={}
        rng_path=out/f'step_{counter}_cpu_rng.bin'
        rng_path.write_bytes(bytes(main_rng[0].tolist()));rng_files[rng_path.name]=hashlib.sha256(rng_path.read_bytes()).hexdigest()
        for i,value in enumerate(main_rng[1]):
            rng_path=out/f'step_{counter}_cuda{i}_rng.bin'
            rng_path.write_bytes(bytes(value.tolist()));rng_files[rng_path.name]=hashlib.sha256(rng_path.read_bytes()).hexdigest()
        record['snapshots'][str(counter)]=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            active_bn_counter=counter,optimizer_step_counter=engine._cov_gap_steps,
            process_torch_rng_only=True,worker_and_sampler_state_not_saved=True,rng_sha256=rng_files)
    @functools.wraps(original)
    def observed(engine,epoch,*args,**kwargs):
        if not start:
            mark=rng()
            start.update({k:v.detach().cpu().double().clone() for k,v in engine.model.state_dict().items()
                          if k.endswith('weight') and v.ndim>=2})
            engine._cov_gap_steps=0
            record['initial_active_bn_counter']=int(engine.model.state_dict()['spec_emb.1.num_batches_tracked'])
            def after_step(opt, step_args, step_kwargs):
                guard=rng();engine._cov_gap_steps+=1
                counter=int(engine.model.state_dict()['spec_emb.1.num_batches_tracked'])
                if counter in [500,505,506]: snapshot(engine,counter)
                if not same_rng(guard,rng()): raise RuntimeError('Observation changed torch RNG')
            engine._cov_gap_handle=engine.optimizer.register_step_post_hook(after_step)
            if not same_rng(mark,rng()):raise RuntimeError('Hook registration changed torch RNG')
        loss=original(engine,epoch,*args,**kwargs)
        guard=rng(); per_layer={}; total_error=total_ref=0.
        for key,initial in start.items():
            current=engine.model.state_dict()[key].detach().cpu().double()
            ref=reference[key].double()
            error=float((current-ref).square().sum()); total_error+=error;total_ref+=float(ref.square().sum())
            per_layer[key]=dict(delta_rms=float((current-initial).square().mean().sqrt()),
                author_rmse=(error/current.numel())**.5)
        row=dict(epoch=int(epoch),optimizer_steps=engine._cov_gap_steps,
            active_bn_counter=int(engine.model.state_dict()['spec_emb.1.num_batches_tracked']),
            learning_rate=float(engine.optimizer.param_groups[0]['lr']),
            weight_decay=float(engine.optimizer.param_groups[0]['weight_decay']),
            betas=list(engine.optimizer.param_groups[0]['betas']),eps=float(engine.optimizer.param_groups[0]['eps']),
            relative_weight_error_to_author=total_error/total_ref,per_layer=per_layer)
        record['epochs'].append(row);write()
        if not same_rng(guard,rng()):raise RuntimeError('Trajectory observation changed torch RNG')
        return loss
    Engine.train_epoch=observed
    if not same_rng(before,rng()):raise RuntimeError('Hook installation changed torch RNG')
    write()
