"""Fresh one-epoch screening: both splits at every quarter; no checkpoint selection."""
import argparse,gc,math,os,random,time,json
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import get_peft_model_state_dict,set_peft_model_state_dict
from transformers import get_linear_schedule_with_warmup
from .data import load,dump
from .model import Ranker,Tokens,pack
from .metrics import evaluate,paired
from .topk import summarize


def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}
def restore(s):random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state(s['cuda'])
def sync():dist.barrier(device_ids=[torch.cuda.current_device()])
def broadcast(value):
    objects=[value if dist.get_rank()==0 else None];dist.broadcast_object_list(objects,src=0);return objects[0]
def root_exists(path):return broadcast(Path(path).exists() if dist.get_rank()==0 else None)
def gather(value):
    values=[None]*dist.get_world_size();dist.all_gather_object(values,value);return values

def batch_indices(order,step,size):
    part=order[step*size:(step+1)*size];valid=len(part)
    return part+[order[i%len(order)] for i in range(size-valid)],valid

def sample_negatives(row,pool,data,seed):
    forbidden=set(data['known_gold'][row['qh']]);candidates=[]
    for other in pool:
        d=other['dh']
        if d not in forbidden and other['docgroup']!=row['docgroup'] and row['docgroup'] not in data.get('metadata',{}).get(d,{}).get('groups',[]) and d not in candidates:candidates.append(d)
    generator=random.Random(seed);generator.shuffle(candidates);fallback=False
    if len(candidates)<3:
        fallback=True;others=list(data['train']);generator.shuffle(others)
        for other in others:
            d=other['dh']
            if d not in forbidden and other['docgroup']!=row['docgroup'] and row['docgroup'] not in data.get('metadata',{}).get(d,{}).get('groups',[]) and d not in candidates:candidates.append(d)
            if len(candidates)>=3:break
    assert len(candidates)>=3
    return candidates[:3],fallback

def mixed_negatives(row,pool,data,seed):
    # Keep one premined hard negative and two unique random in-batch negatives.
    hard=data['hard_negatives'][row['id']][seed%3]
    ns,fallback=sample_negatives(row,pool,data,seed)
    randoms=[d for d in ns if d!=hard]
    assert len(randoms)>=2
    return [hard]+randoms[:2],fallback

def eval_model(model,tokens,frozen,directory,rank,world,budget,limit):
    model.eval();candidates=frozen['candidates'];gold=frozen['gold']
    flat=[(q,i,tokens.pair(q,d,limit)) for q in candidates for i,d in enumerate(candidates[q])];flat.sort(key=lambda r:len(r[-1]));flat=flat[rank::world]
    scores=[];start=time.monotonic();done=0
    with torch.inference_mode():
        for chunk in pack(flat,budget,maximum=1024):
            ids,lengths=tokens.tensors([r[-1] for r in chunk],torch.cuda.current_device());values=model(ids,lengths).cpu().tolist()
            if not np.isfinite(values).all():raise RuntimeError('Non-finite evaluation scores')
            scores.extend((q,i,v) for (q,i,_),v in zip(chunk,values))
            done+=len(chunk)
            if done%2000<len(chunk):print('EVAL',directory.name,'rank',rank,done,len(flat),'seconds',round(time.monotonic()-start,1),flush=True)
    merged={q:[None]*len(ds) for q,ds in candidates.items()}
    for shard in gather(scores):
        for q,i,value in shard:merged[q][i]=value
    assert all(v is not None for values in merged.values() for v in values)
    summary,per=evaluate(candidates,merged,gold)
    if rank==0:
        dump(directory/'scores.json',merged);dump(directory/'per-query.json',per);dump(directory/'metrics.json',{**summary,'seconds':time.monotonic()-start})
        print('EVAL COMPLETE',directory,summary,flush=True)
    sync();return summary

def profile(a,tokens,data):
    path=Path(a.output)/'profile.json';cached=broadcast(load(path) if a.rank==0 and path.exists() else None)
    if cached is not None:return cached
    sample=[]
    for r in data['train']:
        for d in [r['dh']]+data['hard_negatives'][r['id']]:sample.append(tokens.pair(r['qh'],d,a.max_length))
    longest=max(sample,key=len);del sample
    total=torch.cuda.get_device_properties(a.local_rank).total_memory;records=[]
    for cp in [False,True]:
        model=Ranker(a.model,train=True,checkpointing=cp).cuda().train()
        for count in [16,32,64]:
            try:
                ids,lengths=tokens.tensors([longest]*count,a.local_rank)
                model.zero_grad(set_to_none=True);v=model(ids,lengths).reshape(-1,4);torch.nn.functional.softplus(v[:,1:]-v[:,:1]).mean().backward();del v
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic()
                for _ in range(2):
                    model.zero_grad(set_to_none=True);v=model(ids,lengths).reshape(-1,4);torch.nn.functional.softplus(v[:,1:]-v[:,:1]).mean().backward();del v
                torch.cuda.synchronize();r={'gc':cp,'pairs':count,'tokens':count*len(longest),'pair_s':2*count/(time.monotonic()-t),'peak_bytes':torch.cuda.max_memory_allocated()};records.append(r)
                if a.rank==0:print('TRAIN PROFILE',r,flush=True)
                if r['peak_bytes']>total*.88:break
            except torch.OutOfMemoryError:
                model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();break
        del model;gc.collect();torch.cuda.empty_cache()
    safe=[r for r in records if r['peak_bytes']<total*.9];fast=max(safe,key=lambda r:r['pair_s']);off=[r for r in safe if not r['gc']]
    if off and max(r['pair_s'] for r in off)>=fast['pair_s']*.95:fast=max(off,key=lambda r:r['pair_s'])
    choice=broadcast(fast if a.rank==0 else None)
    model=Ranker(a.model,train=True,checkpointing=choice['gc']).cuda().eval();eval_budgets={};eval_records={}
    for split in ['dev','test','tax']:
        qs=list(data['tax']['queries']) if split=='tax' else list(data[split]);ds=list(data['tax']['documents']) if split=='tax' else list({d for r in data[split].values() for d in r['gold']})
        q=max(qs,key=lambda x:len(tokens.cache['queries'][x]));d=max(ds,key=lambda x:len(tokens.cache['documents'][x]));row=tokens.pair(q,d,16384 if split=='tax' else a.max_length)
        tested=[]
        with torch.inference_mode():
            for count in ([8,16,32,64,128] if split=='tax' else [64,128,256,512]):
                try:
                    ids,lengths=tokens.tensors([row]*count,a.local_rank);model(ids,lengths);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic();model(ids,lengths);torch.cuda.synchronize()
                    r={'pairs':count,'tokens':count*len(row),'pair_s':count/(time.monotonic()-t),'peak_bytes':torch.cuda.max_memory_allocated()};tested.append(r)
                    if a.rank==0:print('EVAL PROFILE',split,r,flush=True)
                    if r['peak_bytes']>total*.88:break
                except torch.OutOfMemoryError:gc.collect();torch.cuda.empty_cache();break
        safe_eval=[r for r in tested if r['peak_bytes']<total*.9];eval_budgets[split]=max(safe_eval,key=lambda r:r['pair_s'])['tokens'];eval_records[split]=tested
    obj={'train_tokens':choice['tokens'],'checkpointing':choice['gc'],'eval_tokens':eval_budgets,'train_probes':records,'eval_probes':eval_records,'gpu':torch.cuda.get_device_name()}
    obj=broadcast(obj if a.rank==0 else None)
    if a.rank==0:dump(path,obj)
    del model;gc.collect();torch.cuda.empty_cache();sync();return obj

def main(a):
    torch.cuda.set_device(a.local_rank);dist.init_process_group('nccl',timeout=timedelta(minutes=30),device_id=torch.device('cuda',a.local_rank))
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    if a.decision_file:
        decision=broadcast(load(a.decision_file) if a.rank==0 else None)
        if decision['skip_training']:
            if a.rank==0:dump(root/'skipped.json',decision);print('TRAINING SKIPPED',decision,flush=True)
            dist.destroy_process_group();return
    data=load(Path(a.data)/'data.json');tokens=Tokens(a.model,Path(a.data)/'tokens.pkl')
    if a.train_country:
        data['train']=[r for r in data['train'] if r['country']==a.train_country]
        if not data['train']:raise ValueError('No training rows for requested country')
    if a.rank==0:print('TRAIN CONTRACT',{'country':a.train_country,'rows':len(data['train']),'arms':a.arms,'epochs':a.epochs,'checkpoint_cadence':a.checkpoint_cadence},flush=True)
    prof=profile(a,tokens,data)
    if a.profile_only:dist.destroy_process_group();return
    frozen=load(Path(a.retrieval)/'frozen.json')
    if not root_exists(root/'zero-shot/completed.json'):
        model=Ranker(a.model).cuda()
        for split in ['dev','test','tax']:eval_model(model,tokens,frozen[split],root/'zero-shot'/split,a.rank,a.world,prof['eval_tokens'][split],16384 if split=='tax' else a.max_length)
        if a.rank==0:dump(root/'zero-shot/completed.json',{'complete':True})
        del model;gc.collect();torch.cuda.empty_cache();sync()
    assert a.batch%a.world==0
    for seed in [42]:
      for arm in a.arms:
        name=f'{arm}-seed{seed}';out=root/name;out.mkdir(parents=True,exist_ok=True)
        if root_exists(out/'completed.json'):continue
        seed_all(seed);model=Ranker(a.model,train=True,checkpointing=prof['checkpointing']).cuda()
        ddp=DDP(model,device_ids=[a.local_rank],broadcast_buffers=False)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=.01)
        per_epoch=math.ceil(len(data['train'])/a.batch);total_steps=min(per_epoch*a.epochs,a.max_steps) if a.max_steps else per_epoch*a.epochs
        scheduler=get_linear_schedule_with_warmup(optimizer,math.ceil(.03*total_steps),total_steps)
        checkpoints=({per_epoch*e for e in range(1,a.epochs+1)} if a.checkpoint_cadence=='epoch' else {max(1,math.ceil(total_steps*f)) for f in [.25,.5,.75,1.]})
        if a.max_steps and a.checkpoint_cadence=='quarter':checkpoints={max(1,round(total_steps*f)) for f in [.25,.5,.75,1.]}
        if a.rank==0:dump(out/'training_contract.json',{'country':a.train_country,'train_rows':len(data['train']),'epochs':a.epochs,'steps_per_epoch':per_epoch,'total_steps':total_steps,'checkpoint_steps':sorted(checkpoints),'batch':a.batch,'seed':seed})
        latest=out/'latest.pt';start_step=0;train_seconds=0.;fallbacks=0;pending_dev=False
        state=broadcast(torch.load(latest,map_location='cpu',weights_only=False) if a.rank==0 and latest.exists() else None)
        if state is not None:
            set_peft_model_state_dict(model.base,state['adapter']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);start_step=state['step'];train_seconds=state['train_seconds'];fallbacks=state.get('fallbacks',[0]*a.world)[a.rank];pending_dev=state.get('pending_dev',False);restore(state['rng'][a.rank]);del state
        if a.rank==0 and (out/'loss.jsonl').exists():
            lines=[x for x in (out/'loss.jsonl').read_text().splitlines() if json.loads(x)['step']<=start_step]
            (out/'loss.jsonl').write_text('\n'.join(lines)+('\n' if lines else ''))
        wb=None
        if a.rank==0:
            import wandb
            wb=wandb.init(entity=a.entity,project=a.project,group=a.group,name=name,id=f'{a.group}-{name}',resume='allow',config={'seed':seed,'arm':arm,'epochs':a.epochs,'global_query_batch':a.batch,'train_rows':len(data['train']),'train_country':a.train_country,'checkpoint_cadence':a.checkpoint_cadence,'max_length':a.max_length,'lr':1e-5,'warmup':.03,'lora_r':16,'lora_alpha':32,'profile':prof,'objective':'pairwise logistic, one positive and three negatives'})
        def save(step,pending_dev=False):
            states=gather(rng());fallback_counts=gather(fallbacks)
            if a.rank==0:
                payload={'step':step,'adapter':{k:v.cpu() for k,v in get_peft_model_state_dict(model.base).items()},'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':states,'train_seconds':train_seconds,'fallbacks':fallback_counts,'pending_dev':pending_dev}
                torch.save(payload,out/'latest.tmp');(out/'latest.tmp').replace(latest)
            sync()
        def assess(step):
            percent=int(round(step/total_steps*100));directory=out/f'checkpoint-step-{step:05d}'
            if a.rank==0:model.base.save_pretrained(directory/'adapter')
            results={}
            for split in ['dev','test','tax']:
                marker=directory/split/'metrics.json'
                if root_exists(marker):results[split]=broadcast(load(marker) if a.rank==0 else None)
                else:results[split]=eval_model(model,tokens,frozen[split],directory/split,a.rank,a.world,prof['eval_tokens'][split],16384 if split=='tax' else a.max_length)
            if a.rank==0:
                country_views = {}
                for split in ('dev', 'test'):
                    ranking = frozen[split]
                    sizes = [k for k in (100, 200, 500) if all(len(ds) >= k for ds in ranking['candidates'].values())]
                    countries = {q: row['country'] for q, row in data[split].items()}
                    country_views[split] = summarize(ranking, load(directory/split/'scores.json'), countries,
                                                     sizes, load(root/'zero-shot'/split/'scores.json'))
                dump(directory/'country-metrics.json', country_views)
                dump(directory/'completed.json',{'step':step,'epoch':step/per_epoch,'percentage':percent,'metrics':results})
                wb.log({f'{split}/{k}':v for split,m in results.items() for k,v in m.items()},step=step)
            save(step);model.train()
        if pending_dev:assess(start_step)
        order=None;model.train()
        for step in range(start_step,total_steps):
            t0=time.monotonic();epoch=step//per_epoch;within=step%per_epoch
            if order is None or within==0:
                order=list(range(len(data['train'])));random.Random(seed+epoch).shuffle(order)
            indices,valid=batch_indices(order,within,a.batch);global_rows=[data['train'][i] for i in indices];local=[]
            for j in range(a.rank,a.batch,a.world):
                r=global_rows[j]
                if arm=='hard':negative=data['hard_negatives'][r['id']]
                else:
                    sampler=mixed_negatives if arm=='mixed' else sample_negatives
                    negative,fallback=sampler(r,global_rows,data,seed*10000000+step*a.batch+j);fallbacks+=int(fallback and j<valid)
                rows=[tokens.pair(r['qh'],d,a.max_length) for d in [r['dh']]+negative]
                local.append((rows,float(j<valid)*a.world/valid))
            local.sort(key=lambda x:max(map(len,x[0])))
            chunks=[];chunk=[];max_length=0
            for group in local:
                length=max(map(len,group[0]));new=max(max_length,length)
                if chunk and new*4*(len(chunk)+1)>prof['train_tokens']:chunks.append(chunk);chunk=[];max_length=0
                chunk.append(group);max_length=max(max_length,length)
            if chunk:chunks.append(chunk)
            optimizer.zero_grad(set_to_none=True);value=0.
            for ci,chunk in enumerate(chunks):
                with ddp.no_sync() if ci<len(chunks)-1 else nullcontext():
                    rows=[r for group,_ in chunk for r in group];ids,lengths=tokens.tensors(rows,a.local_rank);scores=ddp(ids,lengths).reshape(-1,4)
                    losses=torch.nn.functional.softplus(scores[:,1:]-scores[:,:1]).mean(1)
                    loss=(losses*torch.tensor([w for _,w in chunk],device='cuda')).sum();loss.backward();value+=float(loss.detach())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not math.isfinite(value) or not torch.isfinite(norm):raise RuntimeError('Non-finite loss or gradient')
            optimizer.step();scheduler.step();torch.cuda.synchronize();train_seconds+=time.monotonic()-t0
            loss_global=torch.tensor(value,device='cuda');dist.all_reduce(loss_global);loss_global/=a.world
            if a.rank==0:
                print(f'TRAIN {name} step={step+1}/{total_steps} loss={float(loss_global):.6f} train_seconds={train_seconds:.1f} query/s={((step+1)*a.batch)/train_seconds:.2f}',flush=True)
                with (out/'loss.jsonl').open('a') as f:f.write(json.dumps({'step':step+1,'loss':float(loss_global),'train_seconds':train_seconds,'lr':scheduler.get_last_lr()[0]})+'\n')
                wb.log({'train/loss':float(loss_global),'train/seconds':train_seconds,'epoch':(step+1)/per_epoch},step=step+1)
            if (step+1)%25==0 or step+1 in checkpoints:save(step+1,pending_dev=step+1 in checkpoints)
            if step+1 in checkpoints:assess(step+1)
        all_fallbacks=sum(gather(fallbacks))
        if a.rank==0:
            wb.finish();dump(out/'completed.json',{'name':name,'arm':arm,'seed':seed,'train_seconds':train_seconds,'inbatch_fallback_groups':all_fallbacks,
                'checkpoints':[load(out/f'checkpoint-step-{step:05d}/completed.json') for step in sorted(checkpoints)]})
        del ddp,model,optimizer,scheduler;gc.collect();torch.cuda.empty_cache();sync()
    if a.rank==0:dump(root/'completed.json',{arm:load(root/f'{arm}-seed42/completed.json') for arm in a.arms})
    sync();dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--retrieval',required=True);p.add_argument('--output',required=True);p.add_argument('--model',required=True)
    p.add_argument('--rank',type=int,default=int(os.environ.get('RANK',0)));p.add_argument('--local-rank',type=int,default=int(os.environ.get('LOCAL_RANK',0)));p.add_argument('--world',type=int,default=int(os.environ.get('WORLD_SIZE',1)))
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--batch',type=int,default=64);p.add_argument('--max-length',type=int,default=2048);p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44]);p.add_argument('--profile-only',action='store_true')
    p.add_argument('--train-country',choices=['au','in']);p.add_argument('--arms',nargs='+',choices=['inbatch','hard','mixed'],default=['inbatch','hard']);p.add_argument('--checkpoint-cadence',choices=['quarter','epoch'],default='quarter')
    p.add_argument('--max-steps',type=int);p.add_argument('--decision-file')
    p.add_argument('--entity');p.add_argument('--project',default='reranker-reproduction');p.add_argument('--group',default='judicialmind');main(p.parse_args())
