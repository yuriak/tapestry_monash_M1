"""Sharded Qwen embeddings and exact dense Top1000 over the whole mini corpus."""
import argparse,os,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
from .common import load,dump,views
from .model import INSTRUCTION,trim,pack

def encode(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/f'rank-{a.rank}.json').exists(): return
    torch.cuda.set_device(a.local_rank);data=load(Path(a.data)/'data.json')
    flat=[(('document',d),t) for d,t in data['documents'].items()]
    instruction = load(Path(a.data)/'manifest.json').get('instruction', INSTRUCTION)
    flat += [(('query',q),f'Instruct: {instruction}\nQuery: {t}') for q,t in data['queries'].items()]
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    rows=[(key,trim(tok.encode(text,add_special_tokens=False),8192)) for key,text in flat[a.rank::a.world]]
    rows.sort(key=lambda r:len(r[1]));model=AutoModel.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    keys=[];vectors=[];start=time.monotonic()
    def compute(chunk):
        lengths=torch.tensor([len(r[1]) for r in chunk],device='cuda');ids=torch.full((len(chunk),int(lengths.max())),tok.pad_token_id,dtype=torch.long,device='cuda')
        for i,(_,r) in enumerate(chunk):ids[i,:len(r)]=torch.tensor(r,device='cuda')
        h=model(input_ids=ids,use_cache=False).last_hidden_state[torch.arange(len(chunk),device='cuda'),lengths-1].float()
        return torch.nn.functional.normalize(h,dim=1).cpu().numpy()
    def safe(chunk):
        try:return compute(chunk)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(chunk)==1:raise
            mid=len(chunk)//2;return np.concatenate([safe(chunk[:mid]),safe(chunk[mid:])])
    with torch.inference_mode():
        for chunk in pack(rows,32768,128):
            vectors.append(safe(chunk));keys.extend(k for k,_ in chunk)
            print('EMBED rank',a.rank,'done',len(keys),'/',len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
    np.save(out/f'rank-{a.rank}.npy',np.concatenate(vectors));dump(out/f'rank-{a.rank}.json',keys)

def freeze(a):
    out=Path(a.output)
    if (out/'frozen.json').exists():return
    torch.cuda.set_device(a.local_rank);data=load(Path(a.data)/'data.json');vectors={}
    for rank in range(a.world):
        keys=load(out/f'rank-{rank}.json');values=np.load(out/f'rank-{rank}.npy');vectors.update({tuple(k):v for k,v in zip(keys,values)})
    ds=sorted(data['documents']);matrix=torch.from_numpy(np.stack([vectors[('document',d)] for d in ds])).cuda();result={};summary={}
    for split,qs in data['splits'].items():
        candidates={};scores={}
        for start in range(0,len(qs),128):
            batch=qs[start:start+128];query=torch.from_numpy(np.stack([vectors[('query',q)] for q in batch])).cuda()
            values,indices=torch.topk(query@matrix.T,min(1000,len(ds)),dim=1,sorted=True)
            for q,v,ix in zip(batch,values.cpu().tolist(),indices.cpu().tolist()):
                candidates[q]=[ds[i] for i in ix];scores[q]=v
        gold={q:data['gold'][q] for q in qs};result[split]={'candidates':candidates,'gold':gold}
        summary[split]=views(candidates,scores,gold);dump(out/f'{split}-embedding-scores.json',scores)
    dump(out/'embedding-metrics.json',summary);dump(out/'frozen.json',result)
    print('RETRIEVAL COMPLETE',summary,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['encode','freeze']);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--model');p.add_argument('--rank',type=int,default=int(os.environ.get('RANK',0)));p.add_argument('--local-rank',type=int,default=int(os.environ.get('LOCAL_RANK',0)));p.add_argument('--world',type=int,default=int(os.environ.get('WORLD_SIZE',4)));a=p.parse_args();globals()[a.stage](a)
