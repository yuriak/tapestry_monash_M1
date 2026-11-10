"""Frozen dense-candidate NFCorpus PEFT trajectories with checkpoint-wide evaluation.

This workflow intentionally evaluates every predeclared checkpoint on dev and test
only after both training trajectories finish. Test curves are descriptive, never
an input to training, stopping, or checkpoint selection.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from datetime import datetime, timezone

from tapestry_m1.core.io import atomic_json, fingerprint, records, text_key, dumps, ContractError
from tapestry_m1.data.au_qa_training import near_pairs
from tapestry_m1.workflows.nfcorpus_pilot import source, qrels, write_rows
from tapestry_m1.reranker.losses import group_loss
from tapestry_m1.reranker.peft_backend import build_backend
from tapestry_m1.reranker.peft_training import scores_for
from tapestry_m1.runtime.training_checkpoint import save_peft_checkpoint, load_peft_state, restore_rng
from tapestry_m1.runtime.devices import gpu_runtime
from tapestry_m1.runtime.provenance import code_fingerprint


def log(event, **values):
    print(dumps({'time':datetime.now(timezone.utc).isoformat(), 'event':event, **values}), flush=True)


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def objective_loss(scores, objective):
    import torch
    if objective == 'L0':
        return group_loss(scores, 'L0')[0]
    if objective != 'L2' or scores.ndim != 2 or scores.shape[1] != 4 or not len(scores) or not torch.isfinite(scores).all():
        raise ContractError('L2 requires finite [groups,4] scores')
    return (-torch.log_softmax(scores.float(),dim=1)[:,0]).mean()


def ranking_metrics(rankings, judgments):
    if len(rankings)!=len(judgments) or {r['query_id'] for r in rankings}!=set(judgments):
        raise ContractError('incomplete or duplicated evaluation queries')
    output=[]
    for row in rankings:
        gold=judgments[row['query_id']];positive={d for d,g in gold.items() if g>0}
        ids=[x['doc_id'] for x in row['candidates']]
        if len(set(ids))!=len(ids) or not positive:raise ContractError('invalid candidates or positives')
        ideal=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(sorted(gold.values(),reverse=True)[:10]))
        dcg=sum((2**gold.get(d,0)-1)/math.log2(i+2) for i,d in enumerate(ids[:10]))
        hits=[i+1 for i,d in enumerate(ids) if d in positive]
        item={'query_id':row['query_id'],'ndcg@10':dcg/ideal,'mrr@10':1/hits[0] if hits and hits[0]<=10 else 0.}
        for k in (10,50,100):
            n=sum(r<=k for r in hits);item[f'recall@{k}']=n/len(positive);item[f'hit@{k}']=int(n>0)
        output.append(item)
    return {'queries':len(output),'gain':'2^rel-1','means':{k:sum(v[k] for v in output)/len(output) for k in output[0] if k!='query_id'},'per_query':output}


class FrozenSchedule:
    def __init__(self,size):self.size=size;self.cursor=0
    def take(self,n):
        if self.cursor+n>self.size:raise ContractError('frozen schedule exhausted; no cycling')
        start=self.cursor;self.cursor+=n;return list(range(start,self.cursor))
    def state_dict(self):return {'size':self.size,'cursor':self.cursor}
    def load_state_dict(self,state):
        if state['size']!=self.size or not 0<=state['cursor']<=self.size:raise ContractError('schedule mismatch')
        self.cursor=state['cursor']


def make_groups(docs, queries, judgments, rankings, seed, count):
    """Train-only supervision; aliases use corpus text, never held-out qrels."""
    names=sorted(docs);near={d:set() for d in names};by_text={}
    for d in names:
        by_text.setdefault(text_key(docs[d]),[]).append(d)
    for aliases in by_text.values():
        for d in aliases:near[d].update(aliases)
    for a,b,_ in near_pairs([docs[d] for d in names]):near[names[a]].add(names[b]);near[names[b]].add(names[a])
    ranked={r['query_id']:r['candidates'] for r in rankings}
    if set(ranked)!=set(judgments):raise ContractError('train retrieval/query mismatch')
    order=sorted(judgments);random.Random(seed).shuffle(order);groups=[];excluded=[]
    for q in order:
        positive=sorted(d for d,g in judgments[q].items() if g>0)
        if not positive:raise ContractError('no train positive')
        rng=random.Random(seed ^ int(hashlib.sha256(q.encode()).hexdigest()[:16],16))
        pos=rng.choice(positive);banned=set(positive)
        for p in positive:banned.update(near[p])
        hard=[]
        for row in ranked[q]:
            d=row['doc_id']
            if d not in banned and all(d not in near[x] and d!=x for x in hard):hard.append(d)
            if len(hard)==2:break
        remaining=[d for d in names if d not in banned and d not in hard and all(d not in near[x] for x in hard)]
        if len(hard)!=2 or not remaining:
            excluded.append({'query_id':q,'reason':'insufficient distinct assumed negatives'});continue
        negative=[rng.choice(remaining),*hard];ids=[pos,*negative]
        if len(set(ids))!=4 or set(negative)&set(positive):raise ContractError('invalid group')
        groups.append({'schema_version':'m1.nfcorpus.groups.v1','query_id':q,'query_text':queries[q],'split':'train',
                       'known_positive_ids':positive,'positive_outside_top100':pos not in {x['doc_id'] for x in ranked[q]},
                       'candidates':[{'doc_id':d,'text':docs[d],'label':int(i==0),'grade':judgments[q].get(d),
                                      'negative_kind':['positive','random','embedding-hard','embedding-hard'][i],
                                      'assumed_negative':i>0 and d not in judgments[q]} for i,d in enumerate(ids)]})
    if len(groups)<count:raise ContractError('insufficient unique eligible training queries')
    return groups[:count],{'eligible':len(groups),'excluded':excluded,'policy':'seeded shuffled queries, one seeded positive; random then two retrieval-hard negatives; exact and existing SimHash/Jaccard alias exclusion'}


def prepare(config,raw,retrieval,out):
    out.mkdir(parents=True,exist_ok=True)
    completion=json.loads((retrieval/'completed.json').read_text())
    for name,sha in completion['files'].items():
        if fingerprint(retrieval/name)!=sha:raise ContractError('retrieval artifact checksum mismatch')
    original=json.loads((retrieval/'inputs.json').read_text())
    for name,sha in original['source'].items():
        if fingerprint(raw/name)!=sha:raise ContractError('source checksum mismatch')
    docs,qs=source(raw);judgments=qrels(raw/'qrels/train.tsv',qs,docs)
    identity={'config':config,'source':original['source'],'retrieval_manifest':fingerprint(retrieval/'completed.json'),
              'code':code_fingerprint(),'environment':fingerprint(os.environ['M1_ENVIRONMENT_LOCK'])}
    atomic_json(out/'inputs.json',identity)
    if not (out/'groups-manifest.json').exists():
        log('GROUPS_BUILD_START',queries=len(judgments))
        groups,audit=make_groups(docs,qs,judgments,list(records(retrieval/'train-ranking.jsonl')),config['seed'],config['steps']*config['effective_groups'])
        write_rows(out/'groups.jsonl',groups)
        atomic_json(out/'groups-manifest.json',{'identity':digest(identity),'sha256':fingerprint(out/'groups.jsonl'),'groups':len(groups),'audit':audit,
                                              'positive_outside_top100':sum(g['positive_outside_top100'] for g in groups)})
    manifest=json.loads((out/'groups-manifest.json').read_text())
    if manifest['identity']!=digest(identity) or fingerprint(out/'groups.jsonl')!=manifest['sha256']:raise ContractError('groups changed')
    groups=list(records(out/'groups.jsonl'))
    log('GROUPS_READY',groups=len(groups),sha256=manifest['sha256'])
    return identity,docs,qs,groups


def seed_all(seed):
    import torch
    random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def adapter_hash(model):
    from peft import get_peft_model_state_dict
    h=hashlib.sha256()
    for name,v in sorted(get_peft_model_state_dict(model).items()):
        h.update(name.encode());h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def optimizer_for(backend,c):
    import torch
    parameters=[p for p in backend.model.parameters() if p.requires_grad]
    if any(p.dtype!=torch.float32 for p in parameters):raise ContractError('expected FP32 adapters')
    opt=torch.optim.AdamW(parameters,lr=c['lr'],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    return parameters,opt,scheduler


def update(backend,groups,objective,parameters,opt,scheduler):
    import torch
    backend.model.train();opt.zero_grad(set_to_none=True);total=0.;truncated=0
    for g in groups:
        scores,audit=scores_for(backend,[g]);loss=objective_loss(scores,objective)
        (loss/len(groups)).backward();total+=float(loss.detach())/len(groups);truncated+=audit['truncated']
    gradients=[p.grad for p in parameters if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients) or not any(g.count_nonzero() for g in gradients):raise ContractError('invalid adapter gradients')
    if any(p.grad is not None for p in backend.model.parameters() if not p.requires_grad):raise ContractError('base gradient ownership violation')
    norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True));opt.step();scheduler.step()
    return {'loss':total,'gradient_norm':norm,'truncated_pairs':truncated}


def preflight(c,spec,data,out,groups,identity):
    import torch
    destination=out/'preflight'
    if (destination/'passed.json').exists():
        if json.loads((destination/'passed.json').read_text())['identity']!=digest(identity):raise ContractError('preflight identity differs')
        log('PREFLIGHT_REUSED');return
    destination.mkdir(parents=True,exist_ok=True);checks={}
    for objective in ('L0','L2'):
        log('PREFLIGHT_START',objective=objective);seed_all(c['seed'])
        backend,_,_=build_backend(c,spec,data);params,opt,sched=optimizer_for(backend,c);sampler=FrozenSchedule(len(groups))
        backend.model.eval()
        with torch.no_grad():
            initial,_=scores_for(backend,[groups[0]])
            with backend.model.disable_adapter():base,_=scores_for(backend,[groups[0]])
        if not torch.equal(initial,base):raise ContractError('initial adapter differs from base')
        pairs=[x['text'] for x in groups[0]['candidates']]
        single,_=backend.score_logits([groups[0]['query_text']]*4,pairs,1)
        if max(abs(a-b) for a,b in zip(single,initial.flatten().tolist()))>.5:raise ContractError('training/inference batching discrepancy')
        sampler.take(1);first=update(backend,[groups[0]],objective,params,opt,sched)
        cp=destination/objective/'step-000001'
        save_peft_checkpoint(cp,backend.model,opt,sched,sampler,{'identity':digest(identity),'step':1})
        backend.model.eval()
        with torch.no_grad():saved,_=scores_for(backend,[groups[1]])
        state,_=load_peft_state(cp,digest(identity));restore_rng(state['rng'])
        second=update(backend,[groups[1]],objective,params,opt,sched);expected=adapter_hash(backend.model)
        backend.close();del params,opt,sched
        backend,_,_=build_backend(c,spec,data,adapter=cp);params,opt,sched=optimizer_for(backend,c)
        backend.model.eval()
        with torch.no_grad():loaded,_=scores_for(backend,[groups[1]])
        error=float((saved-loaded).abs().max())
        if error>1e-5:raise ContractError('checkpoint scoring reload mismatch')
        opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler']);sampler.load_state_dict(state['sampler']);restore_rng(state['rng'])
        observed=update(backend,[groups[1]],objective,params,opt,sched)
        if expected!=adapter_hash(backend.model):raise ContractError('checkpoint optimizer/RNG continuation differs')
        if {int(v['step']) for v in opt.state.values()}!={2}:raise ContractError('optimizer step mismatch')
        checks[objective]={'first':first,'second':second,'resumed_second':observed,'reload_error':error,'exact_continuation':True}
        backend.close();del params,opt,sched
    atomic_json(destination/'passed.json',{'identity':digest(identity),'checks':checks})
    log('PREFLIGHT_PASSED',checks=checks)


def train(c,spec,data,out,groups,identity,objective):
    import torch
    run=out/objective;run.mkdir(exist_ok=True);key=digest({'inputs':identity,'objective':objective,'groups':fingerprint(out/'groups.jsonl')})
    if (run/'completed.json').exists():
        if json.loads((run/'completed.json').read_text())['identity']!=key:raise ContractError('training identity differs')
        for cp in sorted((run/'checkpoints').glob('step-*')):load_peft_state(cp,key)
        log('TRAIN_ALREADY_COMPLETE',objective=objective);return
    checkpoints=sorted((run/'checkpoints').glob('step-*')) if (run/'checkpoints').exists() else []
    resume=checkpoints[-1] if checkpoints else None
    seed_all(c['seed']);backend,model,inventory=build_backend(c,spec,data,adapter=resume)
    params,opt,sched=optimizer_for(backend,c);sampler=FrozenSchedule(len(groups));step=0
    if resume:
        state,meta=load_peft_state(resume,key);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler']);sampler.load_state_dict(state['sampler']);restore_rng(state['rng']);step=meta['step']
    else:
        initial=adapter_hash(backend.model)
        atomic_json(run/'initial.json',{'hash':initial,'seed':c['seed'],'lora':inventory,'model':model})
        other=out/('L2' if objective=='L0' else 'L0')/'initial.json'
        if other.exists() and json.loads(other.read_text())['hash']!=initial:raise ContractError('unmatched initial adapters')
    start=time.monotonic();start_step=step;torch.cuda.reset_peak_memory_stats()
    log('TRAIN_START',objective=objective,resume_step=step,total_steps=c['steps'])
    with (run/'steps.jsonl').open('a') as handle:
        while step<c['steps']:
            tick=time.monotonic();ids=sampler.take(c['effective_groups']);result=update(backend,[groups[i] for i in ids],objective,params,opt,sched);step+=1
            entry={'objective':objective,'step':step,'total_steps':c['steps'],'groups_seen':sampler.cursor,**result,'seconds':time.monotonic()-tick,
                   'eta_seconds':(time.monotonic()-start)/(step-start_step)*(c['steps']-step),'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
            handle.write(dumps(entry)+'\n');handle.flush();log('TRAIN_STEP',**entry)
            if step%c['checkpoint_every']==0:
                cp=run/'checkpoints'/f'step-{step:06d}'
                save_peft_checkpoint(cp,backend.model,opt,sched,sampler,{'identity':key,'step':step,'objective':objective,'group_cursor':sampler.cursor})
                log('CHECKPOINT_SAVED',objective=objective,step=step,path=str(cp))
    if adapter_hash(backend.model)==json.loads((run/'initial.json').read_text())['hash']:raise ContractError('adapter unchanged')
    atomic_json(run/'completed.json',{'identity':key,'step':step,'groups_seen':sampler.cursor,'checkpoint_manifests':{p.parent.name:fingerprint(p) for p in (run/'checkpoints').glob('*/manifest.json')}})
    backend.close();log('TRAIN_COMPLETE',objective=objective,steps=step)


def test_retrieval(raw,retrieval,out,docs,qs,data):
    from tapestry_m1.runtime.assets import verify_model
    from tapestry_m1.retrieval.qwen3 import Qwen3Encoder
    from tapestry_m1.retrieval.encoding import encode_cached
    from tapestry_m1.retrieval.vector_cache import VectorCache
    from tapestry_m1.retrieval.search import exact_topk
    target=out/'test-retrieval.json'
    if target.exists():
        saved=json.loads(target.read_text())
        if fingerprint(out/'test-ranking.jsonl')!=saved['sha256']:raise ContractError('test ranking changed')
        return
    log('TEST_EMBEDDING_START')
    original=json.loads((retrieval/'inputs.json').read_text());c=original['config'];ids=sorted(docs)
    dv,_,_=VectorCache(retrieval/'vectors/documents',original,ids,c['dimension'],c['embedding_shard']).finish()
    judgments=qrels(raw/'qrels/test.tsv',qs,docs);rows=[{'query_id':q,'text':qs[q]} for q in sorted(judgments)]
    model,_=verify_model(c['encoder'],data);encoder=Qwen3Encoder(model,c)
    qv,audit,_=encode_cached(encoder,rows,'text',True,out/'test-vectors',{'original':original,'query_ids':[r['query_id'] for r in rows]},c)
    ranks=exact_topk(ids,dv,qv,100);write_rows(out/'test-ranking.jsonl',[{'query_id':q['query_id'],'candidates':r} for q,r in zip(rows,ranks)])
    atomic_json(target,{'sha256':fingerprint(out/'test-ranking.jsonl'),'queries':len(rows),'lengths':audit,'original_inputs':fingerprint(retrieval/'inputs.json')})
    encoder.close();log('TEST_EMBEDDING_COMPLETE',queries=len(rows))


def evaluate(c,spec,data,out,docs,qs,rankings,judgments,label,adapter=None,*,inference_batch=None):
    import torch
    batch=c['reranker_batch'] if inference_batch is None else inference_batch
    if type(batch) is not int or batch<1:raise ContractError('invalid inference batch')
    directory=out/'evaluation'/label;directory.mkdir(parents=True,exist_ok=True)
    identity={'inputs':fingerprint(out/'inputs.json'),'adapter':fingerprint(adapter/'manifest.json') if adapter else 'base','config':c}
    if adapter:load_peft_state(adapter)
    backend=None
    for split in ('dev','test'):
        dest=directory/split;dest.mkdir(exist_ok=True);frozen=rankings[split]
        score_id=digest({'identity':identity,'rankings':frozen,'split':split})
        if (dest/'completed.json').exists():
            complete=json.loads((dest/'completed.json').read_text())
            if complete['identity']!=score_id:raise ContractError('evaluation identity changed')
            for name,sha in complete['files'].items():
                if fingerprint(dest/name)!=sha:raise ContractError('evaluation checksum mismatch')
            log('EVAL_REUSED',model=label,split=split);continue
        if backend is None:
            if adapter:backend,_,_=build_backend(c,spec,data,adapter=adapter,trainable=False)
            else:
                from tapestry_m1.runtime.assets import verify_model
                from tapestry_m1.reranker.qwen3 import Qwen3Reranker
                root,_=verify_model(spec,data);backend=Qwen3Reranker(root,c)
            backend.model.eval()
        log('EVAL_START',model=label,split=split,queries=len(frozen),pairs=len(frozen)*100,inference_batch=batch)
        cache=dest/'queries';cache.mkdir(exist_ok=True);results=[];truncated=0;start=time.monotonic();new_queries=0;batch_counts={}
        for i,row in enumerate(frozen):
            q=row['query_id'];ids=[v['doc_id'] for v in row['candidates']]
            if len(ids)!=100 or len(set(ids))!=100:raise ContractError('Top100 contract violated')
            path=cache/(hashlib.sha256(q.encode()).hexdigest()+'.json')
            if path.exists():
                saved=json.loads(path.read_text())
                if saved['identity']!=score_id or saved['query_id']!=q:raise ContractError('score cache mismatch')
            else:
                scores,lengths=backend.score_logits([qs[q]]*100,[docs[d] for d in ids],batch)
                if len(scores)!=100 or not all(math.isfinite(s) for s in scores):raise ContractError('invalid inference scores')
                candidates=sorted([{'doc_id':d,'score':s} for d,s in zip(ids,scores)],key=lambda r:(-r['score'],r['doc_id']))
                saved={'identity':score_id,'inference_batch':batch,'query_id':q,'candidates':[dict(v,rank=j+1) for j,v in enumerate(candidates)],'truncated':sum(v['truncated'] for v in lengths),'lengths':lengths}
                atomic_json(path,saved);new_queries+=1
            if {x['doc_id'] for x in saved['candidates']}!=set(ids):raise ContractError('reranker changed candidate membership')
            used=str(saved.get('inference_batch',c['reranker_batch']));batch_counts[used]=batch_counts.get(used,0)+1
            results.append({'query_id':q,'candidates':saved['candidates']});truncated+=saved['truncated']
            if (i+1)%5==0 or i==0 or i+1==len(frozen):
                elapsed=time.monotonic()-start
                log('EVAL_PROGRESS',model=label,split=split,queries_done=i+1,queries_total=len(frozen),pairs_done=(i+1)*100,
                    elapsed_seconds=elapsed,new_queries=new_queries,inference_batch=batch,new_pairs_per_second=new_queries*100/max(elapsed,1e-9),eta_seconds=elapsed/max(1,new_queries)*(len(frozen)-i-1),truncated_pairs=truncated)
        m=ranking_metrics(results,judgments[split]);base=ranking_metrics(frozen,judgments[split])
        for a,b in zip(m['per_query'],base['per_query']):
            if a['query_id']!=b['query_id'] or a['recall@100']!=b['recall@100']:raise ContractError('candidate recall invariance failed')
        write_rows(dest/'ranking.jsonl',results);atomic_json(dest/'metrics.json',m)
        atomic_json(dest/'completed.json',{'identity':score_id,'truncated_pairs':truncated,'queries':len(results),'inference_batch_query_counts':batch_counts,'files':{n:fingerprint(dest/n) for n in ('ranking.jsonl','metrics.json')}})
        log('EVAL_COMPLETE',model=label,split=split,metrics=m['means'])
    if backend:backend.close()


def summarize(c,out):
    import numpy as np
    systems=['E0','E1']+[f'{o}-{s:06d}' for o in ('L0','L2') for s in range(c['checkpoint_every'],c['steps']+1,c['checkpoint_every'])]
    metrics={label:{split:json.loads((out/'evaluation'/label/split/'metrics.json').read_text()) for split in ('dev','test')} for label in systems}
    selection={o:max([f'{o}-{s:06d}' for s in range(c['checkpoint_every'],c['steps']+1,c['checkpoint_every'])],key=lambda name:(metrics[name]['dev']['means']['ndcg@10'],-int(name.split('-')[1]))) for o in ('L0','L2')}
    atomic_json(out/'selection.json',{'criterion':'dev graded nDCG@10; earlier checkpoint tie break','selected':selection,'test_role':'all checkpoints descriptive only; never used for selection'})
    comparisons={}
    for split in ('dev','test'):
        pairs=[('E1','E0')]+[(name,'E1') for name in systems if name.startswith(('L0-','L2-'))]+[(f'L2-{s:06d}',f'L0-{s:06d}') for s in range(c['checkpoint_every'],c['steps']+1,c['checkpoint_every'])]+[(selection['L2'],selection['L0'])]
        comparisons[split]={}
        for a,b in pairs:
            aa=metrics[a][split]['per_query'];bb=metrics[b][split]['per_query']
            if [x['query_id'] for x in aa]!=[x['query_id'] for x in bb]:raise ContractError('paired query order differs')
            delta=np.array([x['ndcg@10']-y['ndcg@10'] for x,y in zip(aa,bb)])
            rng=np.random.default_rng(c['seed']);means=[]
            for _ in range(100):means.extend(delta[rng.integers(0,len(delta),size=(100,len(delta)))].mean(axis=1).tolist())
            comparisons[split][a+' minus '+b]={'delta':float(delta.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'replicates':10000,'seed':c['seed'],'multiple_comparisons_corrected':False}
    atomic_json(out/'curves.json',{'systems':{label:{s:metrics[label][s]['means'] for s in ('dev','test')} for label in systems},'paired_ndcg10':comparisons,'selection':selection,'limitation':'one seed; test curves descriptive; no test-driven tuning'})
    atomic_json(out/'completed.json',{'status':'complete','systems':len(systems),'checkpoints_per_arm':c['steps']//c['checkpoint_every'],'files':{n:fingerprint(out/n) for n in ('inputs.json','groups-manifest.json','selection.json','curves.json')}})
    log('PIPELINE_COMPLETE',selected_by_dev=selection,curves=str(out/'curves.json'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);p.add_argument('--retrieval',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--stage',choices=['preflight','run'],required=True);a=p.parse_args()
    c=json.loads(a.config.read_text());spec=json.loads((a.config.parent/c['model']).read_text())
    if c['top_k']!=100 or c['steps']!=200 or c['checkpoint_every']!=25:raise ContractError('unexpected frozen experiment budget')
    log('PIPELINE_START',stage=a.stage,config=c)
    hardware=gpu_runtime();identity,docs,qs,groups=prepare(c,a.raw,a.retrieval,a.output)
    atomic_json(a.output/'hardware.json',hardware)
    preflight(c,spec,a.data_root,a.output,groups,identity)
    if a.stage=='preflight':return
    for objective in ('L0','L2'):train(c,spec,a.data_root,a.output,groups,identity,objective)
    atomic_json(a.output/'training-frozen.json',{'arms':{o:fingerprint(a.output/o/'completed.json') for o in ('L0','L2')},'checkpoints':list(range(25,201,25)),'test_has_not_influenced_training':True})
    test_retrieval(a.raw,a.retrieval,a.output,docs,qs,a.data_root)
    rankings={'dev':list(records(a.retrieval/'dev-ranking.jsonl')),'test':list(records(a.output/'test-ranking.jsonl'))}
    judgments={s:qrels(a.raw/'qrels'/f'{s}.tsv',qs,docs) for s in ('dev','test')}
    for s in ('dev','test'):
        m=ranking_metrics(rankings[s],judgments[s]);atomic_json(a.output/'evaluation/E0'/s/'metrics.json',m);log('EMBEDDING_BASELINE',split=s,metrics=m['means'])
    evaluate(c,spec,a.data_root,a.output,docs,qs,rankings,judgments,'E1')
    for objective in ('L0','L2'):
        for step in range(25,201,25):
            cp=a.output/objective/'checkpoints'/f'step-{step:06d}'
            evaluate(c,spec,a.data_root,a.output,docs,qs,rankings,judgments,f'{objective}-{step:06d}',cp)
    summarize(c,a.output)

if __name__=='__main__':main()
