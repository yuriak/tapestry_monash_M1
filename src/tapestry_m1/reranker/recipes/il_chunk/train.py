"""IL chunk-selected single-seed BCE LoRA run, throughput probes and full quarter-checkpoint evaluation."""
import argparse,gc,math,os,random,time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import get_peft_model_state_dict,set_peft_model_state_dict
from transformers import get_linear_schedule_with_warmup
from .common import load,dump,views,metrics
from .model import Ranker,Tokens,pack
from .sampling import mixed_negatives

from ..distributed import broadcast, clean, gather, restore, rng, seed_all, sync

def probe(a,tokens,data):
    path=Path(a.output)/'profile.json'
    cached=broadcast(load(path) if a.rank==0 and path.exists() else None)
    if cached:return cached
    # A longest legal pair is a conservative peak-memory probe; actual batches are length-bucketed.
    longest=tokens.longest_pair();total=torch.cuda.get_device_properties(a.local_rank).total_memory
    records=[]
    for checkpointing in [False,True]:
        model=Ranker(a.model,train=True,checkpointing=checkpointing).cuda().train()
        for count in [1,2,4,8,16,32]:
            ids,lengths=tokens.tensors([longest]*count,a.local_rank)
            try:
                model.zero_grad(set_to_none=True);model(ids,lengths).square().mean().backward();torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats();start=time.monotonic()
                model.zero_grad(set_to_none=True);model(ids,lengths).square().mean().backward();torch.cuda.synchronize()
                record={'pairs':count,'tokens':count*len(longest),'checkpointing':checkpointing,'pairs_per_second':count/(time.monotonic()-start),'peak_bytes':torch.cuda.max_memory_allocated()}
                records.append(record);print('TRAIN PROBE',a.rank,record,flush=True)
                if record['peak_bytes']>total*.78:break
            except torch.OutOfMemoryError:
                print('TRAIN PROBE OOM',a.rank,checkpointing,count,flush=True);model.zero_grad(set_to_none=True);clean();break
        del model,ids,lengths;clean()
    choices=gather(records);mode_best=[]
    for ck in [False,True]:
        opts=[[r for r in rs if r['checkpointing']==ck and r['peak_bytes']<total*.78] for rs in choices]
        if all(opts):
            chosen=[max(rs,key=lambda r:r['pairs_per_second']) for rs in opts]
            mode_best.append({'checkpointing':ck,'train_tokens':min(r['tokens'] for r in chosen),'throughput':min(r['pairs_per_second'] for r in chosen)})
    selected=max(mode_best,key=lambda r:r['throughput'])
    # Independent inference probe: backward memory should not constrain inference batches.
    model=Ranker(a.model).cuda().eval();ev=[]
    # Compare fast last-token scoring with the stock model on padded mixed lengths.
    q=next(iter(tokens.cache['queries']));ds=data['document_ids'][:2]
    test_rows=[tokens.pair(q,d,a.max_length) for d in ds]
    with torch.inference_mode():
        ids,lengths=tokens.tensors(test_rows,a.local_rank)
        fast=model(ids,lengths)
        mask=torch.arange(ids.shape[1],device='cuda')[None,:]<lengths[:,None]
        official=model.base(input_ids=ids,attention_mask=mask,use_cache=False).logits
        last=official[torch.arange(len(ids),device='cuda'),lengths-1]
        reference=last[:,model.yes].float()-last[:,model.no].float()
        torch.testing.assert_close(fast,reference,atol=.15,rtol=.02)
        print('SCORER CHECK PASSED',a.rank,'max_diff',float((fast-reference).abs().max()),flush=True)
        del official,last,reference,fast,mask
    with torch.inference_mode():
        for count in [4,8,16,32,64]:
            ids,lengths=tokens.tensors([longest]*count,a.local_rank)
            try:
                model(ids,lengths);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic();model(ids,lengths);torch.cuda.synchronize()
                rec={'pairs':count,'tokens':count*len(longest),'pairs_per_second':count/(time.monotonic()-t),'peak_bytes':torch.cuda.max_memory_allocated()};ev.append(rec);print('EVAL PROBE',a.rank,rec,flush=True)
                if rec['peak_bytes']>total*.82:break
            except torch.OutOfMemoryError:clean();break
    del model,ids,lengths;clean()
    best=max([r for r in ev if r['peak_bytes']<total*.82],key=lambda r:r['pairs_per_second'])
    selected.update(eval_tokens=min(r['tokens'] for r in gather(best)),train_probes=choices,eval_probes=gather(ev),world=a.world,gpu=torch.cuda.get_device_name(),max_length=a.max_length)
    if a.rank==0:dump(path,selected)
    sync();return selected

def evaluate(a,model,tokens,frozen,directory,budget):
    directory=Path(directory);summary=directory/'metrics.json'
    done=broadcast(load(summary) if a.rank==0 and summary.exists() else None)
    if done is not None:return done
    model.eval();candidates=frozen['candidates'];gold=frozen['gold'];qs=list(candidates)[a.rank::a.world];shard=directory/f'scores-rank-{a.rank}.json';start=time.monotonic()
    if shard.exists():scores=load(shard)
    else:
        rows=[(q,i,tokens.pair(q,d,a.max_length)) for q in qs for i,d in enumerate(candidates[q])];rows.sort(key=lambda r:len(r[-1]));scores={q:[None]*len(candidates[q]) for q in qs};count=0;last=time.monotonic()
        def score(chunk):
            ids,lengths=tokens.tensors([r[-1] for r in chunk],a.local_rank)
            return model(ids,lengths).cpu().tolist()
        def safe(chunk):
            try:return score(chunk)
            except torch.OutOfMemoryError:
                clean()
                if len(chunk)==1:raise
                mid=len(chunk)//2;return safe(chunk[:mid])+safe(chunk[mid:])
        with torch.inference_mode():
            for chunk in pack(rows,budget,512):
                values=safe(chunk)
                if not np.isfinite(values).all():raise RuntimeError('Non-finite evaluation scores')
                for (q,i,_),v in zip(chunk,values):scores[q][i]=v
                count+=len(chunk)
                if time.monotonic()-last>30 or count==len(rows):
                    elapsed=time.monotonic()-start;print('EVAL',directory.name,'rank',a.rank,f'{count}/{len(rows)}',f'pairs/s={count/elapsed:.1f}',f'ETA={(len(rows)-count)/(count/elapsed)/60:.1f}min',flush=True);last=time.monotonic()
        dump(shard,scores)
    sync()
    if a.rank==0:
        merged={q:v for rank in range(a.world) for q,v in load(directory/f'scores-rank-{rank}.json').items()}
        results=views(candidates,merged,gold);_,per=metrics(candidates,merged,gold)
        if directory.name=='test':
            dev=load(directory.parent/'dev/metrics.json')
            baseline=load(Path(a.output)/'zero-shot/dev/metrics.json')
            for k,v in results.items():
                v['exploratory_max_f1']=max(v['f1_by_k'])
                v['k']=dev[k]['k'];v['macro_f1']=v['f1_by_k'][v['k']-1]
                v['baseline_dev_k']=baseline[k]['k'];v['macro_f1_baseline_dev_k']=v['f1_by_k'][v['baseline_dev_k']-1]
        dump(directory/'per-query.json',per);dump(summary,results)
        print('EVAL COMPLETE',str(directory),results,'seconds',time.monotonic()-start,flush=True)
    else:results=None
    return broadcast(results)

def main(a):
    torch.cuda.set_device(a.local_rank);dist.init_process_group('nccl',timeout=timedelta(minutes=60),device_id=torch.device('cuda',a.local_rank))
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True);data=load(Path(a.data)/'data.json');manifest=load(Path(a.data)/'manifest.json');a.max_length=manifest['max_length']
    tokens=Tokens(a.model,Path(a.data));prof=probe(a,tokens,data)
    if a.profile_only:dist.destroy_process_group();return
    frozen=load(Path(a.retrieval)/'frozen.json')
    wb=None
    if a.rank==0:
        import wandb
        wb=wandb.init(entity=a.entity,project=a.project,group=a.group,name='il-chunk-'+Path(a.data).name+'-bce-seed42',id=a.group+'-'+Path(a.data).name+'-bce-seed42',resume='allow',config={'dataset':manifest,'recipe':'pointwise BCE all positives; 1 hard + 2 broad dense negatives; chunk-selected input','seed':42,'epochs':1,'task':Path(a.data).name,'global_groups':a.batch,'profile':prof,'lr':1e-5,'lora_r':16,'lora_alpha':32})
    zero={};model=Ranker(a.model).cuda()
    for split in ['dev','test']:zero[split]=evaluate(a,model,tokens,frozen[split],root/'zero-shot'/split,prof['eval_tokens'])
    if a.rank==0:
        dump(root/'zero-shot/completed.json',zero)
        wb.log({f'zero/{s}/{k}/{m}':v for s,r in zero.items() for k,ms in r.items() for m,v in ms.items()},step=0)
    del model;clean();sync();seed_all()
    model=Ranker(a.model,train=True,checkpointing=prof['checkpointing']).cuda();ddp=DDP(model,device_ids=[a.local_rank],broadcast_buffers=False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=.01)
    pairs=data['train_pairs'];order=list(range(len(pairs)));random.Random(42).shuffle(order);total=math.ceil(len(pairs)/a.batch)
    scheduler=get_linear_schedule_with_warmup(optimizer,math.ceil(total*.03),total);checkpoints={math.ceil(total*f) for f in [.25,.5,.75,1.]}
    negatives=data['negative']
    assert all(len(ds)>=3 for ds in negatives.values())
    state_path=root/'latest.pt';start_step=0;train_seconds=0.;pending=False
    # Shared read avoids broadcasting optimizer tensors as a massive Python object.
    if state_path.exists():
        state=torch.load(state_path,map_location='cpu',weights_only=False);set_peft_model_state_dict(model.base,state['adapter']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);start_step=state['step'];train_seconds=state['train_seconds'];pending=state['pending'];restore(state['rng'][a.rank]);del state
    def save(step,pending=False):
        states=gather(rng())
        if a.rank==0:
            obj={'step':step,'adapter':{k:v.cpu() for k,v in get_peft_model_state_dict(model.base).items()},'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':states,'train_seconds':train_seconds,'pending':pending}
            torch.save(obj,root/'latest.tmp');(root/'latest.tmp').replace(state_path)
        sync()
    def checkpoint(step):
        out=root/f'checkpoint-step-{step:05d}'
        if a.rank==0:model.base.save_pretrained(out/'adapter')
        sync();results={}
        for split in ['dev','test']:results[split]=evaluate(a,model,tokens,frozen[split],out/split,prof['eval_tokens'])
        if a.rank==0:
            dump(out/'completed.json',{'step':step,'epoch':min(1,step*a.batch/len(pairs)),'metrics':results,'train_seconds':train_seconds})
            wb.log({f'{s}/{k}/{m}':v for s,r in results.items() for k,ms in r.items() for m,v in ms.items()},step=step)
        save(step);model.train()
    if pending:checkpoint(start_step)
    model.train()
    for step in range(start_step,total):
        t=time.monotonic();selected=order[step*a.batch:(step+1)*a.batch];valid=len(selected);selected += order[:a.batch-valid];local=[]
        for j in range(a.rank,a.batch,a.world):
            q,d=pairs[selected[j]];ns=mixed_negatives(negatives[q],selected[j])
            rows=[tokens.pair(q,x,a.max_length) for x in [d]+ns];local.append((rows,float(j<valid)*a.world/valid))
        # BCE factorizes over pairs: microbatch individual pairs, even when an 8K
        # group cannot fit together. Preserve exact group weighting/global batch.
        flat=[(row,float(i==0),weight/4.) for rows,weight in local for i,row in enumerate(rows)]
        flat.sort(key=lambda x:len(x[0]));chunks=[];chunk=[];longest=0
        for row in flat:
            new=max(longest,len(row[0]))
            if chunk and new*(len(chunk)+1)>prof['train_tokens']:chunks.append(chunk);chunk=[];longest=0
            chunk.append(row);longest=max(longest,len(row[0]))
        if chunk:chunks.append(chunk)
        optimizer.zero_grad(set_to_none=True);value=0.
        for ci,chunk in enumerate(chunks):
            with ddp.no_sync() if ci<len(chunks)-1 else nullcontext():
                ids,lengths=tokens.tensors([row for row,_,_ in chunk],a.local_rank);scores=ddp(ids,lengths)
                labels=torch.tensor([label for _,label,_ in chunk],device='cuda')
                weights=torch.tensor([weight for _,_,weight in chunk],device='cuda')
                loss=(torch.nn.functional.binary_cross_entropy_with_logits(scores,labels,reduction='none')*weights).sum()
                loss.backward();value+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not math.isfinite(value) or not torch.isfinite(norm):raise RuntimeError('Non-finite train loss or gradient')
        optimizer.step();scheduler.step();torch.cuda.synchronize();train_seconds+=time.monotonic()-t
        loss_global=torch.tensor(value,device='cuda');dist.all_reduce(loss_global);loss_global/=a.world
        if a.rank==0:
            row={'step':step+1,'steps':total,'loss':float(loss_global),'epoch':min(1,(step+1)*a.batch/len(pairs)),'train_seconds':train_seconds,'lr':scheduler.get_last_lr()[0]}
            import json
            with open(root/'loss.jsonl','a') as f:f.write(json.dumps(row)+'\n')
            wb.log({'train/'+k:v for k,v in row.items()},step=step+1)
            print('TRAIN',row,flush=True)
        if (step+1)%25==0 or step+1 in checkpoints:save(step+1,pending=step+1 in checkpoints)
        if step+1 in checkpoints:checkpoint(step+1)
    if a.rank==0:
        result={'complete':True,'train_seconds':train_seconds,'steps':total,'groups':len(pairs),'checkpoints':sorted(checkpoints),'zero_shot':zero,'embedding':load(Path(a.retrieval)/'embedding-metrics.json'),'runs':{str(s):load(root/f'checkpoint-step-{s:05d}/completed.json') for s in sorted(checkpoints)}}
        dump(root/'completed.json',result);wb.finish();print('IL CHUNK ALL COMPLETE training_seconds',train_seconds,flush=True)
    sync();dist.destroy_process_group()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--retrieval',required=True);p.add_argument('--output',required=True);p.add_argument('--model',required=True);p.add_argument('--batch',type=int,default=64);p.add_argument('--profile-only',action='store_true');p.add_argument('--entity');p.add_argument('--project',default='reranker-reproduction');p.add_argument('--group',required=True)
    a=p.parse_args();a.rank=int(os.environ.get('RANK',0));a.local_rank=int(os.environ.get('LOCAL_RANK',0));a.world=int(os.environ.get('WORLD_SIZE',1));assert a.batch%a.world==0;main(a)
