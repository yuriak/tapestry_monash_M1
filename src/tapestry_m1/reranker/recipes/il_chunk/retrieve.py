"""Exact chunk-aware Qwen retrieval, with label-independent pair evidence selection."""
import argparse,pickle,os,time,hashlib
from array import array
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
from .common import load,dump,views
from .il_metrics import INSTRUCTIONS
from .model import pack

def encode(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/f'embed-{a.rank}.json').exists():return
    torch.cuda.set_device(a.local_rank)
    with open(Path(a.data)/'chunks.pkl','rb') as f:obj=pickle.load(f)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    reference=AutoTokenizer.from_pretrained(a.reranker,local_files_only=True)
    rv=reference.get_vocab();ev=tok.get_vocab()
    incompatible={v for k,v in rv.items() if ev.get(k)!=v}
    used=set()
    for texts in [obj['tokens']['queries'],obj['tokens']['documents']['lsr'],obj['tokens']['documents']['pcr']]:
        for ids in texts.values():used.update(ids)
    assert not used.intersection(incompatible),'Used token IDs must agree across tokenizers'
    rows=[];number=0
    for task in ['lsr','pcr']:
        for kind,label,texts in [('query','queries',obj['tokens']['queries']),('document',task,obj['tokens']['documents'][task])]:
            prefix=tok.encode(f'Instruct: {INSTRUCTIONS[task]}\nQuery: ',add_special_tokens=False) if kind=='query' else []
            for key,ids in texts.items():
                for ci,(start,end) in enumerate(obj['spans'][label][key]):
                    if number%a.world==a.rank:rows.append(((task,kind,key,ci),array('I',prefix+list(ids[start:end]))))
                    number+=1
    del obj;rows.sort(key=lambda x:len(x[1]));model=AutoModel.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    vectors=[];keys=[];start=time.monotonic();last=start
    def compute(chunk):
        lengths=np.array([len(r[1]) for r in chunk]);ids=np.full((len(chunk),int(lengths.max())),tok.pad_token_id,dtype=np.int64)
        for i,(_,v) in enumerate(chunk):ids[i,:len(v)]=v
        ids=torch.from_numpy(ids).cuda();lens=torch.from_numpy(lengths).cuda()
        h=model(input_ids=ids,use_cache=False).last_hidden_state[torch.arange(len(chunk),device='cuda'),lens-1].float()
        return torch.nn.functional.normalize(h,dim=1).cpu().numpy()
    def safe(chunk):
        try:return compute(chunk)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(chunk)==1:raise
            mid=len(chunk)//2;return np.concatenate([safe(chunk[:mid]),safe(chunk[mid:])])
    with torch.inference_mode():
        for chunk in pack(rows,32768,64):
            vectors.append(safe(chunk));keys.extend(r[0] for r in chunk)
            if time.monotonic()-last>20 or len(keys)==len(rows):
                elapsed=time.monotonic()-start;print('EMBED rank',a.rank,len(keys),'/',len(rows),'seconds',round(elapsed,1),'ETAmin',round((len(rows)-len(keys))/(len(keys)/elapsed)/60,1),flush=True);last=time.monotonic()
    np.save(out/f'embed-{a.rank}.npy',np.concatenate(vectors));dump(out/f'embed-{a.rank}.json',keys)

def aggregate(sim,owner,ndocs):
    """Mean of the best two query-chunk max affinities; return selected chunk indices."""
    # sim [query chunks, all document chunks]. All choices depend only on embeddings.
    affinity=torch.full((sim.shape[0],ndocs),-torch.inf,device=sim.device)
    affinity.scatter_reduce_(1,owner[None,:].expand(sim.shape[0],-1),sim,reduce='amax',include_self=True)
    values,qi=torch.topk(affinity,min(2,sim.shape[0]),dim=0)
    score=values.mean(0)
    if qi.shape[0]==1:qi=qi.expand(2,-1)
    ci=torch.arange(sim.shape[1],device=sim.device)
    relevance=sim[qi[:,owner],ci[None,:]].amax(0)
    choices=[];current=relevance.clone()
    for _ in range(2):
        best=torch.full((ndocs,),-torch.inf,device=sim.device);best.scatter_reduce_(0,owner,current,reduce='amax',include_self=True)
        eligible=torch.where(current==best[owner],ci,sim.shape[1])
        index=torch.full((ndocs,),sim.shape[1],dtype=torch.long,device=sim.device);index.scatter_reduce_(0,owner,eligible,reduce='amin',include_self=True)
        # One-chunk documents reuse their sole chunk; model later removes duplicates.
        if choices:index=torch.where(torch.isneginf(best),choices[0],index)
        choices.append(index);current[index]=-torch.inf
    return score,qi,torch.stack(choices)

def freeze(a):
    torch.cuda.set_device(a.local_rank);out=Path(a.output)
    if (out/f'freeze-{a.rank}.json').exists():return
    vectors={}
    for rank in range(a.world):
        keys=load(out/f'embed-{rank}.json');vals=np.load(out/f'embed-{rank}.npy',mmap_mode='r')
        vectors.update({tuple(k):v for k,v in zip(keys,vals)})
    with open(Path(a.data)/'chunks.pkl','rb') as f:spans=pickle.load(f)['spans']
    manifest={};start=time.monotonic()
    for task in ['lsr','pcr']:
        data=load(Path(a.data)/task/'data.json');ds=data['document_ids'];qs=data['query_ids'][a.rank::a.world];keys=[(task,'document',d,j) for d in ds for j in range(len(spans[task][d]))]
        matrix=torch.from_numpy(np.stack([vectors[k] for k in keys])).cuda()
        counts=[len(spans[task][d]) for d in ds];owner=torch.tensor(np.repeat(np.arange(len(ds)),counts),device='cuda');offsets=torch.tensor(np.cumsum([0]+counts[:-1]),device='cuda')
        selections=np.lib.format.open_memmap(out/f'{task}-selection-{a.rank}.npy',mode='w+',dtype=np.int32,shape=(len(qs),len(ds),4))
        result={}
        for i,q in enumerate(qs):
            query=torch.from_numpy(np.stack([vectors[(task,'query',q,j)] for j in range(len(spans['queries'][q]))])).cuda()
            score,qi,di=aggregate(query@matrix.T,owner,len(ds));local_di=di-offsets[None,:]
            selections[i]=torch.cat([qi,local_di],dim=0).T.cpu().numpy()
            # Freeze all document rankings for train mining; eval uses only its Top200 prefix.
            order=torch.argsort(score,descending=True,stable=True);ix=order.cpu().tolist()
            result[q]={'candidates':[ds[j] for j in ix],'scores':score[order].cpu().tolist()}
            if i%50==0:print('RETRIEVE',task,'rank',a.rank,i,'/',len(qs),'seconds',round(time.monotonic()-start,1),flush=True)
        selections.flush();dump(out/f'{task}-rank-{a.rank}.json',result);manifest[task]={'queries':qs}
        del matrix,query,score,qi,di,selections;torch.cuda.empty_cache()
    dump(out/f'freeze-{a.rank}.json',manifest)

def assemble(a):
    root=Path(a.data);out=Path(a.output);original=load(Path(a.source)/'data.json');summaries={}
    for task in ['lsr','pcr']:
        data=load(root/task/'data.json');qindex={q:i for i,q in enumerate(data['query_ids'])};ds=data['document_ids'];rankings={}
        dest=np.lib.format.open_memmap(root/task/'selection.npy',mode='w+',dtype=np.int32,shape=(len(qindex),len(ds),4))
        for rank in range(a.world):
            meta=load(out/f'freeze-{rank}.json');src=np.load(out/f'{task}-selection-{rank}.npy',mmap_mode='r')
            for i,q in enumerate(meta[task]['queries']):dest[qindex[q]]=src[i]
            rankings.update(load(out/f'{task}-rank-{rank}.json'))
        dest.flush();frozen={};summary={}
        for split,qs in data['splits'].items():
            cs={q:rankings[q]['candidates'][:200] for q in qs};scores={q:rankings[q]['scores'][:200] for q in qs};gold={q:data['gold'][q] for q in qs}
            frozen[split]={'candidates':cs,'gold':gold};summary[split]=views(cs,scores,gold)
            if split=='test':
                for k,v in summary[split].items():v['k']=summary['dev'][k]['k'];v['macro_f1']=v['f1_by_k'][v['k']-1]
        aliases={d:hashlib.sha256(' '.join(t.lower().split()).encode()).hexdigest() for d,t in original['documents'][task].items()}
        negatives={}
        for q in data['splits']['train']:
            if not data['gold'][q]:continue
            seen={aliases[d] for d in data['gold'][q]};pool=[]
            for d in rankings[q]['candidates'][:1000]:
                if aliases[d] not in seen:pool.append(d);seen.add(aliases[d])
            assert len(pool)>=3;negatives[q]=pool
        data['negative']=negatives;dump(root/task/'data.json',data)
        (out/task).mkdir(exist_ok=True);dump(out/task/'frozen.json',frozen);dump(out/task/'embedding-metrics.json',summary);summaries[task]=summary
        print('DENSE COMPLETE',task,summary['test'],flush=True)
    dump(out/'completed.json',summaries)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['encode','freeze','assemble']);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--source');p.add_argument('--model');p.add_argument('--reranker');p.add_argument('--rank',type=int,default=int(os.environ.get('RANK',0)));p.add_argument('--local-rank',type=int,default=int(os.environ.get('LOCAL_RANK',0)));p.add_argument('--world',type=int,default=int(os.environ.get('WORLD_SIZE',4)));a=p.parse_args();globals()[a.stage](a)
