import os
"""Qwen dense retrieval for held-out JudicialMind and the complete native AU Tax pool."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
from .data import load,dump
from .model import INSTRUCTION,trim
from .metrics import evaluate

def encode(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/f'rank-{a.rank}.json').exists():return
    torch.cuda.set_device(a.local_rank);data=load(Path(a.data)/'data.json');flat=[]
    for split in ['dev','test','tax']:
        queries=data['tax']['queries'] if split=='tax' else {q:r['query'] for q,r in data[split].items()}
        docids=list(data['tax']['documents']) if split=='tax' else sorted({d for r in data[split].values() for d in r['gold']})
        documents=data['tax']['documents'] if split=='tax' else data['documents']
        flat.extend(((split,'query',q),f'Instruct: {INSTRUCTION}\nQuery: '+t) for q,t in queries.items())
        flat.extend(((split,'document',d),documents[d]) for d in docids)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    rows=[(key,trim(tok.encode(text,add_special_tokens=False),16384 if key[0]=='tax' else 8192)) for key,text in flat[a.rank::a.world]]
    rows.sort(key=lambda r:len(r[1]));model=AutoModel.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    from .model import pack
    vectors=[];keys=[];start=time.monotonic()
    with torch.inference_mode():
        for chunk in pack(rows,32768,64):
            lengths=torch.tensor([len(r[1]) for r in chunk],device='cuda');ids=torch.full((len(chunk),int(lengths.max())),tok.pad_token_id,dtype=torch.long,device='cuda')
            for i,(_,r) in enumerate(chunk):ids[i,:len(r)]=torch.tensor(r,device='cuda')
            h=model(input_ids=ids,use_cache=False).last_hidden_state[torch.arange(len(chunk),device='cuda'),lengths-1].float()
            vectors.append(torch.nn.functional.normalize(h,dim=1).cpu().numpy());keys.extend(k for k,_ in chunk)
            print('EMBED',a.rank,len(keys),len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
    np.save(out/f'rank-{a.rank}.npy',np.concatenate(vectors));dump(out/f'rank-{a.rank}.json',keys)
def freeze(a):
    out=Path(a.output);data=load(Path(a.data)/'data.json');vectors={}
    for rank in range(a.world):
        keys=load(out/f'rank-{rank}.json');values=np.load(out/f'rank-{rank}.npy');vectors.update({tuple(k):v for k,v in zip(keys,values)})
    result={};summaries={}
    for split in ['dev','test','tax']:
        qs=list(data['tax']['queries']) if split=='tax' else list(data[split]);ds=sorted(k[2] for k in vectors if k[:2]==(split,'document'))
        scores=np.stack([vectors[(split,'query',q)] for q in qs])@np.stack([vectors[(split,'document',d)] for d in ds]).T
        top=len(ds) if split=='tax' else min(a.top,len(ds));candidates={};rank_scores={}
        for q,v in zip(qs,scores):
            indices=np.argsort(-v,kind='stable')[:top];candidates[q]=[ds[i] for i in indices];rank_scores[q]=v[indices].tolist()
        gold=data['tax']['gold'] if split=='tax' else {q:{d:1 for d in r['gold']} for q,r in data[split].items()}
        result[split]={'candidates':candidates,'gold':gold};summary,per=evaluate(candidates,rank_scores,gold);summaries[split]=summary
        dump(out/f'{split}-embedding-scores.json',rank_scores);dump(out/f'{split}-embedding-per-query.json',per)
    dump(out/'frozen.json',result);dump(out/'embedding-metrics.json',summaries)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--top',type=int,default=100);p.add_argument('stage',choices=['encode','freeze']);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--model');p.add_argument('--rank',type=int,default=int(os.environ.get('RANK',0)));p.add_argument('--local-rank',type=int,default=int(os.environ.get('LOCAL_RANK',0)));p.add_argument('--world',type=int,default=int(os.environ.get('WORLD_SIZE',1)));a=p.parse_args();globals()[a.stage](a)
