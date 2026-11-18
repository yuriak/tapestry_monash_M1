import os
"""Four independent shards encode frozen Qwen dense candidates, then CPU merge."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from .common import INSTRUCTIONS, batches, dump, load, metrics, trim

def embed(a):
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    marker=out/f'embed-{a.rank}.json'
    if marker.exists(): return
    torch.cuda.set_device(a.rank if torch.cuda.device_count()>1 else 0)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,padding_side='left')
    model=AutoModel.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    data=load(Path(a.data)/'data.json'); flat=[]
    for task in ['lsr','pcr']:
        for q,text in data['queries'].items(): flat.append((task,'query',q,text))
        for d,text in data['documents'][task].items(): flat.append((task,'document',d,text))
    subset=flat[a.rank::a.world]; rows=[]; truncated=0
    for task,kind,key,text in subset:
        prefix=tok.encode(f'Instruct: {INSTRUCTIONS[task]}\nQuery: ',add_special_tokens=False) if kind=='query' else []
        ids=tok.encode(text,add_special_tokens=False); truncated+=len(prefix)+len(ids)>8192
        rows.append(((task,kind,key),prefix+trim(ids,8192-len(prefix))))
    rows.sort(key=lambda r:len(r[1])); keys=[]; vectors=[]; start=time.monotonic(); budget=32768
    offset=0
    with torch.inference_mode():
        while offset<len(rows):
            chunk=next(batches(rows[offset:],lambda r:len(r[1]),budget,32))
            try:
                lengths=torch.tensor([len(r[1]) for r in chunk],device='cuda')
                ids=torch.full((len(chunk),int(lengths.max())),tok.pad_token_id,dtype=torch.long,device='cuda')
                for i,(_,row) in enumerate(chunk): ids[i,:len(row)]=torch.tensor(row,dtype=torch.long,device='cuda')
                hidden=model(input_ids=ids,use_cache=False).last_hidden_state[torch.arange(len(chunk),device='cuda'),lengths-1].float()
                value=torch.nn.functional.normalize(hidden,dim=1).cpu().numpy()
            except torch.OutOfMemoryError:
                if budget<=8192: raise
                budget=max(8192,budget//2); torch.cuda.empty_cache(); print(f'EMBED OOM reduce token budget={budget}',flush=True); continue
            vectors.append(value); keys.extend(r[0] for r in chunk); offset+=len(chunk)
            if offset%100<len(chunk): print(f'EMBED rank={a.rank} documents={offset}/{len(rows)} tokens={budget} elapsed={time.monotonic()-start:.1f}s',flush=True)
    np.save(out/f'embed-{a.rank}.npy',np.concatenate(vectors))
    dump(marker,{'keys':keys,'truncated':truncated,'seconds':time.monotonic()-start,'token_budget':budget,'gpu':torch.cuda.get_device_name()})

def freeze(a):
    out=Path(a.output)
    if (out/'frozen.json').exists(): return
    data=load(Path(a.data)/'data.json'); vectors={}
    for r in range(a.world):
        meta=load(out/f'embed-{r}.json'); values=np.load(out/f'embed-{r}.npy')
        for key,value in zip(meta['keys'],values): vectors[tuple(key)]=value
    result={}; summaries={}
    for task,top in [('lsr',20),('pcr',10)]:
        ids=list(data['documents'][task]); matrix=np.stack([vectors[(task,'document',d)] for d in ids])
        result[task]={}; summaries[task]={}
        for split,qs in data['splits'].items():
            candidates={}; scores={}
            for start in range(0,len(qs),128):
                part=qs[start:start+128]; sim=np.stack([vectors[(task,'query',q)] for q in part])@matrix.T
                for q,values in zip(part,sim):
                    indices=np.argsort(-values,kind='stable')[:top]
                    candidates[q]=[ids[i] for i in indices]; scores[q]=values[indices].tolist()
            result[task][split]={'candidates':candidates,'scores':scores}
            summary,per=metrics(candidates,scores,data['gold'][task],None if split!='test' else summaries[task]['dev']['k'])
            summaries[task][split]=summary; dump(out/f'{task}-{split}-embedding-per-query.json',per)
    dump(out/'embedding-metrics.json',summaries); dump(out/'frozen.json',result)
    print('DENSE RETRIEVAL FROZEN',summaries,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['embed','freeze']); p.add_argument('--data',required=True); p.add_argument('--output',required=True); p.add_argument('--model'); p.add_argument('--rank',type=int,default=int(os.environ.get('RANK',0))); p.add_argument('--world',type=int,default=int(os.environ.get('WORLD_SIZE',1))); a=p.parse_args(); globals()[a.stage](a)
