"""Single-seed, one-query-pass IL-PCSR screening; all quarter checkpoints evaluated."""
import argparse,gc,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
from peft import get_peft_model_state_dict,set_peft_model_state_dict
from transformers import get_linear_schedule_with_warmup
from .model import Scorer,seed_all
from .common import load,dump,batches,metrics

class FastScorer(Scorer):
    def score(self,model,rows):
        lengths=np.array([len(r) for r in rows],dtype=np.int64)
        array=np.full((len(rows),int(lengths.max())),self.tok.pad_token_id,dtype=np.int64)
        for i,r in enumerate(rows):array[i,:len(r)]=r
        ids=torch.from_numpy(array).to('cuda');ends=torch.from_numpy(lengths-1).to('cuda')
        base=model.get_base_model() if hasattr(model,'get_base_model') else model
        hidden=base.model(input_ids=ids,use_cache=False).last_hidden_state
        logits=base.lm_head(hidden[torch.arange(len(rows),device='cuda'),ends])
        return logits[:,self.yes].float()-logits[:,self.no].float()

def objective(scores,kind):
    s=scores.reshape(-1,4)
    if kind=='pair':return torch.nn.functional.softplus(s[:,1:]-s[:,:1]).mean()
    target=torch.zeros_like(s);target[:,0]=1
    return torch.nn.functional.binary_cross_entropy_with_logits(s,target)

def tune(scorer,kind,longest,out):
    if out.exists():return load(out)
    records=[];capacity=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    for checkpointing in [False,True]:
        scorer.gradient_checkpointing=checkpointing;m=scorer.model(train=True).train()
        for count in [4,8,16,32,64]:
            try:
                rows=[longest]*count;m.zero_grad(set_to_none=True)
                objective(scorer.score(m,rows),kind).backward();torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats();t=time.monotonic()
                for _ in range(2):
                    m.zero_grad(set_to_none=True);objective(scorer.score(m,rows),kind).backward()
                torch.cuda.synchronize();elapsed=time.monotonic()-t
                r={'gc':checkpointing,'pairs':count,'tokens':count*len(longest),'pair_s':2*count/elapsed,'peak_bytes':torch.cuda.max_memory_allocated()}
                records.append(r);print('TRAIN PROFILE',r,flush=True)
                if r['peak_bytes']>capacity*.88:break
            except torch.OutOfMemoryError:
                print('TRAIN PROFILE OOM',checkpointing,count,flush=True);m.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();break
        del m;gc.collect();torch.cuda.empty_cache()
    safe=[r for r in records if r['peak_bytes']<capacity*.9]
    fastest=max(safe,key=lambda r:r['pair_s'])
    off=[r for r in safe if not r['gc']]
    if off:
        best_off=max(off,key=lambda r:r['pair_s'])
        if best_off['pair_s']>=fastest['pair_s']*.95:fastest=best_off
    scorer.gradient_checkpointing=fastest['gc'];m=scorer.model(train=True).eval()
    eval_records=[]
    with torch.inference_mode():
        for count in [4,8,16,32,64,128]:
            try:
                rows=[longest]*count;scorer.score(m,rows);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic()
                scorer.score(m,rows);torch.cuda.synchronize()
                r={'pairs':count,'tokens':count*len(longest),'pair_s':count/(time.monotonic()-t),'peak_bytes':torch.cuda.max_memory_allocated()};eval_records.append(r);print('EVAL PROFILE',r,flush=True)
                if r['peak_bytes']>capacity*.88:break
            except torch.OutOfMemoryError:
                print('EVAL PROFILE OOM',count,flush=True);gc.collect();torch.cuda.empty_cache();break
    safe_eval=[r for r in eval_records if r['peak_bytes']<capacity*.9]
    best_eval=max(safe_eval,key=lambda r:r['pair_s'])
    result={'train_tokens':fastest['tokens'],'gc':fastest['gc'],'eval_tokens':best_eval['tokens'],'train_probes':records,'eval_probes':eval_records,'gpu':torch.cuda.get_device_name()}
    dump(out,result);del m;gc.collect();torch.cuda.empty_cache();return result

def evaluation(m,scorer,frozen,gold,out,budget):
    if (out/'metrics.json').exists():return load(out/'metrics.json')
    m.eval();candidates=frozen['candidates'];rows=[(q,i,scorer.ids(q,d)) for q,ds in candidates.items() for i,d in enumerate(ds)]
    rows.sort(key=lambda x:len(x[2]));scores={q:[None]*len(ds) for q,ds in candidates.items()};done=0;start=time.monotonic()
    with torch.inference_mode():
        for chunk in batches(rows,lambda r:len(r[2]),budget,256):
            vals=scorer.score(m,[r[2] for r in chunk]).cpu().tolist()
            if not np.isfinite(vals).all():raise RuntimeError('Non-finite scores')
            for (q,i,_),v in zip(chunk,vals):scores[q][i]=v
            done+=len(chunk)
            if done%1000<len(chunk):print('EVAL',out,done,len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
    summary,per=metrics(candidates,scores,gold);summary.update(seconds=time.monotonic()-start,k_policy='all F1@k reported; maximum for each split is exploratory, no checkpoint selection')
    dump(out/'scores.json',scores);dump(out/'per-query.json',per);dump(out/'metrics.json',summary)
    print('EVAL COMPLETE',out,summary,flush=True);return summary

def run(a):
    torch.cuda.set_device(a.local_rank);task,kind=[('lsr','bce'),('lsr','pair'),('pcr','bce'),('pcr','pair')][a.rank]
    root=Path(a.output);out=root/f'{task}-{kind}-seed42';out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():return
    data=load(Path(a.data)/'data.json');frozen=load(Path(a.retrieval)/'frozen.json')[task];scorer=FastScorer(a,task)
    eligible=sorted(data['negative'][task]);generator=random.Random(42)
    positives={q:generator.choice(data['gold'][task][q]) for q in eligible}
    groups={q:[scorer.ids(q,d) for d in [positives[q]]+data['negative'][task][q]] for q in eligible}
    longest=max((r for g in groups.values() for r in g),key=len)
    profile=tune(scorer,kind,longest,out/'profile.json');scorer.gradient_checkpointing=profile['gc']
    # Profiling never updates weights; reset seed and instantiate the pretrained base afresh.
    seed_all(42);model=scorer.model(train=True);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=.01)
    order=eligible.copy();random.Random(42).shuffle(order)
    order=[q for i in range(0,len(order),256) for q in sorted(order[i:i+256],key=lambda q:max(map(len,groups[q])))]
    total=math.ceil(len(order)/32);points={math.ceil(total*f):int(f*100) for f in [.25,.5,.75,1.]}
    scheduler=get_linear_schedule_with_warmup(optimizer,math.ceil(.03*total),total);start=0;pending=False;train_seconds=0.
    latest=out/'latest.pt'
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False);set_peft_model_state_dict(model,state['adapter']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start=state['step'];pending=state['pending'];train_seconds=state['train_seconds'];torch.set_rng_state(state['rng']);torch.cuda.set_rng_state(state['cuda_rng']);del state
    dump(out/'training_contract.json',{'seed':42,'steps':total,'queries':len(order),'query_batch':32,'one_positive_per_query':True,'positives':positives,'checkpoints':points,'max_length':8192,'fresh_from_pretrained':True})
    if (out/'loss.jsonl').exists():
        lines=[x for x in (out/'loss.jsonl').read_text().splitlines() if json.loads(x)['step']<=start]
        (out/'loss.jsonl').write_text('\n'.join(lines)+('\n' if lines else ''))
    import wandb
    wb=wandb.init(entity=a.entity,project=a.project,group=a.group,name=f'il-{task}-{kind}',id=f'{a.group}-il-{task}-{kind}',resume='allow',config={'queries':len(order),'steps':total,'seed':42,'max_length':8192,'profile':profile,'one_positive_per_query':True})
    def save(step,pending):
        state={'step':step,'pending':pending,'train_seconds':train_seconds,'adapter':{k:v.cpu() for k,v in get_peft_model_state_dict(model).items()},'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
        torch.save(state,out/'latest.tmp');(out/'latest.tmp').replace(latest)
    def checkpoint(step):
        directory=out/f'checkpoint-{points[step]:03d}';model.save_pretrained(directory/'adapter')
        result={s:evaluation(model,scorer,frozen[s],data['gold'][task],directory/s,profile['eval_tokens']) for s in ['dev','test']}
        dump(directory/'completed.json',{'step':step,'percentage':points[step],'metrics':result})
        wb.log({f'{s}/{k}':v for s,m in result.items() for k,v in m.items() if isinstance(v,(float,int))},step=step)
        save(step,False);model.train()
    if pending:checkpoint(start)
    model.train()
    for step in range(start,total):
        before=time.monotonic();qs=order[step*32:(step+1)*32];optimizer.zero_grad(set_to_none=True);loss_value=0.
        if kind=='bce':
            flat=[(r,float(i==0)) for q in qs for i,r in enumerate(groups[q])];flat.sort(key=lambda x:len(x[0]))
            for chunk in batches(flat,lambda x:len(x[0]),profile['train_tokens'],256):
                scores=scorer.score(model,[r for r,_ in chunk]);target=torch.tensor([y for _,y in chunk],device='cuda')
                loss=torch.nn.functional.binary_cross_entropy_with_logits(scores,target,reduction='sum')/(4*len(qs));loss.backward();loss_value+=float(loss.detach())
        else:
            values=[groups[q] for q in qs]
            for chunk in batches(values,lambda g:4*max(map(len,g)),profile['train_tokens'],64):
                rows=[r for g in chunk for r in g];scores=scorer.score(model,rows).reshape(-1,4)
                loss=torch.nn.functional.softplus(scores[:,1:]-scores[:,:1]).mean(1).sum()/len(qs);loss.backward();loss_value+=float(loss.detach())
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not math.isfinite(loss_value) or not torch.isfinite(grad):raise RuntimeError('Non-finite loss/gradient')
        optimizer.step();scheduler.step();torch.cuda.synchronize();elapsed=time.monotonic()-before;train_seconds+=elapsed
        event={'step':step+1,'total_steps':total,'loss':loss_value,'lr':scheduler.get_last_lr()[0],'grad_norm':float(grad),'queries':len(qs),'step_seconds':elapsed,'train_seconds':train_seconds}
        with (out/'loss.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        wb.log({f'train/{k}':v for k,v in event.items()},step=step+1);print('TRAIN',task,kind,json.dumps(event),flush=True)
        if (step+1)%10==0 or step+1 in points:save(step+1,step+1 in points)
        if step+1 in points:checkpoint(step+1)
    dump(out/'completed.json',{'seed':42,'steps':total,'train_seconds':train_seconds,'checkpoints':[load(out/f'checkpoint-{p:03d}/completed.json') for p in points.values()]});wb.finish()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data',required=True);p.add_argument('--retrieval',required=True);p.add_argument('--output',required=True)
    p.add_argument('--rank',type=int,default=int(os.environ.get('RANK',0)));p.add_argument('--local-rank',type=int,default=int(os.environ.get('LOCAL_RANK',0)))
    p.add_argument('--entity');p.add_argument('--project',default='reranker-reproduction');p.add_argument('--group',default='screen');run(p.parse_args())
