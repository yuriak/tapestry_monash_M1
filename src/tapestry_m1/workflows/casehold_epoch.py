"""One exact CaseHOLD epoch with DDP, followed by sharded dev/test evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

from tapestry_m1.core.io import ContractError,atomic_json,dumps,fingerprint
from tapestry_m1.data.casehold import FILES,load_split
from tapestry_m1.reranker.choice import aggregate,group_ce,metrics,select
from tapestry_m1.reranker.distributed_groups import accumulate_groups,checkpoint_steps,epoch_order,local_batch,lr_factor
from tapestry_m1.reranker.peft_backend import build_backend
from tapestry_m1.runtime.devices import gpu_runtime
from tapestry_m1.runtime.provenance import code_fingerprint
from tapestry_m1.workflows.casehold_pilot import evaluate,independent_final_audit,log,read,scores_for,seed_all,weight_hash


def rank_rng():
    import torch
    import numpy as np
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}


def restore_rank_rng(state):
    import torch
    import numpy as np
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu());torch.cuda.set_rng_state(state['cuda'].cpu())


def save_distributed(path,model,optimizer,scheduler,step,identity,rank,world,seen):
    import torch
    import torch.distributed as dist
    states=[None]*world if rank==0 else None
    dist.gather_object(rank_rng(),states,dst=0)
    if rank==0:
        stage=Path(tempfile.mkdtemp(prefix='.checkpoint-',dir=path.parent))
        model.save_pretrained(stage,safe_serialization=True,save_embedding_layers=False)
        torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'rank_rng':states,'step':step,'samples_seen':seen},stage/'continuation.pt')
        names=['adapter_config.json','adapter_model.safetensors','continuation.pt']
        atomic_json(stage/'manifest.json',{'format':'m1.peft.ddp.resume.v1','identity':identity,
            'step':step,'world_size':world,'samples_seen':seen,'adapter_hash':weight_hash(model,True),
            'files':{n:fingerprint(stage/n) for n in names}})
        for n in names+['manifest.json']:
            with (stage/n).open('rb') as handle:os.fsync(handle.fileno())
        if path.exists():raise ContractError('checkpoint exists; never overwrite')
        os.rename(stage,path)
        log('checkpoint_saved',step=step,samples_seen=seen)
    dist.barrier()


def load_state(path,identity,world):
    import torch
    meta=read(path/'manifest.json')
    if meta['identity']!=identity or meta['world_size']!=world:
        raise ContractError('checkpoint identity/world-size mismatch')
    for name,sha in meta['files'].items():
        if fingerprint(path/name)!=sha:raise ContractError('checkpoint checksum mismatch')
    return torch.load(path/'continuation.pt',map_location='cpu',weights_only=False)


def read_evaluation(directory):
    summary=read(directory/'summary.json');rows=[]
    for name,sha in sorted(summary['chunks_sha256'].items()):
        if fingerprint(directory/name)!=sha:raise ContractError('evaluation chunk checksum mismatch')
        rows.extend(read(directory/name))
    if len(rows)!=summary['count']:raise ContractError('evaluation coverage mismatch')
    return summary,rows


def merged_evaluation(directory,groups,world):
    rows=[]
    for rank in range(world):
        _,part=read_evaluation(directory/f'rank-{rank}');rows.extend(part)
    by_id={r['query_id']:r for r in rows}
    if len(by_id)!=len(rows) or set(by_id)!={g['query_id'] for g in groups}:
        raise ContractError('duplicate or missing distributed evaluation queries')
    rows=[by_id[g['query_id']] for g in groups]
    if any(r['gold']!=g['gold'] for r,g in zip(rows,groups)):
        raise ContractError('label mismatch in merged evaluation')
    summary=aggregate(rows);summary['truncated_pairs']=sum(r['truncated_pairs'] for r in rows)
    atomic_json(directory/'merged-summary.json',summary)
    return summary,rows


def verify_baseline(baseline,config,splits):
    old=read(baseline/'identity.json')['inputs'];done=read(baseline/'completed.json')
    if fingerprint(baseline/'results.json')!=done['results_sha256']:
        raise ContractError('baseline completed result checksum mismatch')
    for key in ['instruction','max_tokens','dtype','attention_backend','reranker_batch']:
        if config[key]!=old['config'][key]:raise ContractError('baseline scoring protocol differs')
    results={}
    for split in ['validation','test']:
        summary,rows=read_evaluation(baseline/('base-'+split))
        if [(r['query_id'],r['gold']) for r in rows]!=[(g['query_id'],g['gold']) for g in splits[split]]:
            raise ContractError('baseline split identity mismatch')
        results[split]=(summary,rows)
    return results


def train(backend,config,train_groups,output,identity,rank,world,latest,total,checkpoints):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    model=backend.model
    params=[p for p in model.parameters() if p.requires_grad]
    if any(p.dtype!=torch.float32 for p in params):raise ContractError('FP32 adapters required')
    optimizer=torch.optim.AdamW(params,**config['optimizer'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:lr_factor(n+1,total,config['warmup_steps']))
    state=load_state(output/f'checkpoint-{latest}',identity,world) if latest else None
    ddp=DistributedDataParallel(model,device_ids=[int(os.environ['LOCAL_RANK'])],broadcast_buffers=False)
    backend.model=ddp
    if state:
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        restore_rank_rng(state['rank_rng'][rank]);del state
    else:
        seed_all(config['seed']+rank)
    order=epoch_order(len(train_groups),config['seed'])
    started=time.monotonic();ddp.train()
    for step in range(latest+1,total+1):
        ids,count=local_batch(order,step,config['effective_groups'],rank,world)
        optimizer.zero_grad(set_to_none=True);items=[]
        def loss_fn(index):
            group=train_groups[index];scores,_=scores_for(backend,[group])
            items.append(metrics(scores.detach()[0].cpu().tolist(),group['gold']))
            return group_ce(scores,[group['gold']])
        loss=accumulate_groups(ddp,ids,count,world,loss_fn)
        norm=torch.nn.utils.clip_grad_norm_(params,config['max_grad_norm'],error_if_nonfinite=True)
        lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
        keys=['accuracy','mrr','ndcg','margin','dispersion']
        values=torch.tensor([loss]+[sum(r[k] for r in items) for k in keys],dtype=torch.float64,device='cuda')
        dist.all_reduce(values);values=(values/count).cpu().tolist()
        if rank==0:
            seen=min(step*config['effective_groups'],len(train_groups))
            record={'time':datetime.now(timezone.utc).isoformat(),'step':step,'total_steps':total,
                'samples_seen':seen,'epoch':seen/len(train_groups),'groups_in_update':count,
                'loss':values[0],**dict(zip(keys,values[1:])),'lr':lr,'gradient_norm':norm.item(),
                'seconds_per_step':(time.monotonic()-started)/(step-latest),
                'estimated_training_seconds_remaining':(total-step)*(time.monotonic()-started)/(step-latest)}
            with (output/'training.jsonl').open('a') as handle:handle.write(dumps(record)+'\n')
            if step==latest+1 or step%config['log_every']==0 or step in checkpoints:
                log('train_update',**{k:v for k,v in record.items() if k!='time'})
        if step in checkpoints:
            save_distributed(output/f'checkpoint-{step}',model,optimizer,scheduler,step,identity,rank,world,
                             min(step*config['effective_groups'],len(train_groups)))
    backend.model=model
    del ddp,optimizer,scheduler,params
    torch.cuda.empty_cache();dist.barrier()
    if rank==0:
        atomic_json(output/'training-completed.json',{'identity':identity,'updates':total,
            'unique_samples':len(train_groups),'checkpoint_steps':checkpoints,'epochs':1})
        log('training_complete',updates=total,samples=len(train_groups))


def run(args):
    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    config=read(args.config);output=args.output;output.mkdir(parents=True,exist_ok=True)
    hardware=gpu_runtime();torch.set_num_threads(2)
    if world!=config['world_size'] or config['epochs']!=1:raise ContractError('unexpected epoch/world size')
    if torch.cuda.get_device_capability()[0]<8:raise ContractError('native BF16 GPU required')
    splits={s:load_split(args.raw,s) for s in FILES}
    total=math.ceil(len(splits['train'])/config['effective_groups'])
    checkpoints=checkpoint_steps(len(splits['train']),config['effective_groups'],config['checkpoint_fractions'])
    spec=read(args.config.parent/config['model'])
    inputs={'config':config,'code':code_fingerprint(),'raw':{s:v[2] for s,v in FILES.items()},
            'model_spec':spec,'world_size':world,'gpu':hardware['gpu'],
            'environment_lock_sha256':fingerprint(os.environ['M1_ENVIRONMENT_LOCK']),
            'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft','accelerate','tokenizers','numpy','safetensors']},
            'baseline_results_sha256':fingerprint(args.baseline/'results.json')}
    identity=hashlib.sha256(dumps(inputs).encode()).hexdigest()
    baseline=None
    if rank==0:
        baseline=verify_baseline(args.baseline,config,splits)
        atomic_json(output/'identity.json',{'identity':identity,'inputs':inputs})
        log('epoch_run_start',identity=identity,hardware=hardware,world_size=world,total_steps=total,checkpoint_steps=checkpoints)
    dist.barrier()
    existing=[s for s in checkpoints if (output/f'checkpoint-{s}/manifest.json').exists()]
    latest=max(existing,default=0)
    seed_all(config['seed'])
    backend,_,_=build_backend(config,spec,args.data_root,adapter=output/f'checkpoint-{latest}' if latest else None)
    if latest<total:
        train(backend,config,splits['train'],output,identity,rank,world,latest,total,checkpoints)
    if rank==0:
        atomic_json(output/'training-completed.json',{'identity':identity,'updates':total,
            'unique_samples':len(splits['train']),'checkpoint_steps':checkpoints,'epochs':1})
    backend.model.eval()
    for p in backend.model.parameters():p.requires_grad_(False)
    history=[]
    for step in checkpoints:
        cp=output/f'checkpoint-{step}';meta=read(cp/'manifest.json')
        if fingerprint(cp/'adapter_model.safetensors')!=meta['files']['adapter_model.safetensors']:
            raise ContractError('adapter checksum mismatch')
        set_peft_model_state_dict(backend.model,load_file(str(cp/'adapter_model.safetensors')))
        evaluate(backend,splits['validation'][rank::world],output/f'validation-{step}/rank-{rank}',
                 identity,meta['adapter_hash'],config['reranker_batch'])
        dist.barrier()
        if rank==0:
            summary,_=merged_evaluation(output/f'validation-{step}',splits['validation'],world)
            history.append(summary|{'step':step,'adapter_hash':meta['adapter_hash']})
            log('validation_checkpoint_complete',**history[-1])
    if rank==0:
        chosen=select(history)
        atomic_json(output/'selection.json',{'identity':identity,'selected':chosen,'validation_history':history,
                    'rule':'full-validation accuracy, then MRR, then earlier step','test_not_used_for_selection':True})
        if not (output/'test-start.json').exists():
            atomic_json(output/'test-start.json',{'time':datetime.now(timezone.utc).isoformat(),
                        'selection_sha256':fingerprint(output/'selection.json')})
        if read(output/'test-start.json')['selection_sha256']!=fingerprint(output/'selection.json'):
            raise ContractError('selection changed after test opening')
        log('selection_frozen',step=chosen['step'])
    dist.barrier()
    results=[]
    for step in checkpoints:
        cp=output/f'checkpoint-{step}';meta=read(cp/'manifest.json')
        set_peft_model_state_dict(backend.model,load_file(str(cp/'adapter_model.safetensors')))
        evaluate(backend,splits['test'][rank::world],output/f'test-{step}/rank-{rank}',identity,meta['adapter_hash'],config['reranker_batch'])
        dist.barrier()
        if rank==0:
            summary,rows=merged_evaluation(output/f'test-{step}',splits['test'],world)
            audit=independent_final_audit(baseline['test'][1],rows,config)
            result={'step':step,'metrics':summary,'audit':audit}
            atomic_json(output/f'test-{step}/audit.json',result);results.append(result)
            log('test_checkpoint_complete',step=step,**summary)
        dist.barrier()
    backend.close()
    if rank==0:
        result={'identity':identity,'selection':read(output/'selection.json'),'test_checkpoints':results,
                'base_validation':baseline['validation'][0],'base_test':baseline['test'][0],
                'training':{'epochs':1,'samples':len(splits['train']),'updates':total,'world_size':world}}
        atomic_json(output/'results.json',result)
        atomic_json(output/'completed.json',{'status':'complete_and_metric_audited','identity':identity,
                    'results_sha256':fingerprint(output/'results.json')})
        log('epoch_pilot_complete',selected_step=chosen['step'],test_checkpoints=len(results))
    dist.barrier();dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','raw','data-root','output','baseline']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    # One writer process group per output; only global rank0 holds this lock.
    import fcntl
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('a') as lock:
        if int(os.environ['RANK'])==0:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)


if __name__=='__main__':main()
