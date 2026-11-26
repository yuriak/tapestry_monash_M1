"""CPU token cache and actual train/evaluation length inventory."""
import argparse,pickle
from array import array
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from .data import load,dump

def main(a):
    data=load(Path(a.data)/'data.json');queries={r['qh']:r['query'] for r in data['train']}
    for s in ['dev','test']:queries.update({q:r['query'] for q,r in data[s].items()})
    queries.update(data['tax']['queries']);docs={**data['documents'],**data['tax']['documents']}
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);cache={};audit={}
    for label,texts in [('queries',queries),('documents',docs)]:
        cache[label]={};lens=[]
        for i,(key,t) in enumerate(texts.items()):
            ids=tok.encode(t,add_special_tokens=False);cache[label][key]=array('I',ids);lens.append(len(ids))
            if i%2000==0:print('TOKENIZE',label,i,len(texts),flush=True)
        audit[label]={'count':len(lens),'p50_p90_p99_max':np.percentile(lens,[50,90,99,100]).tolist()}
    for split,qs,ds in [('train',[r['qh'] for r in data['train']],[r['dh'] for r in data['train']]),('tax',list(data['tax']['queries']),list(data['tax']['documents']))]:
        audit[split]={'query_max':max(len(cache['queries'][q]) for q in qs),'document_max':max(len(cache['documents'][d]) for d in ds)}
    with (Path(a.data)/'tokens.pkl').open('wb') as f:pickle.dump(cache,f,protocol=5)
    dump(Path(a.data)/'lengths.json',audit);print(audit,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--model',required=True);main(p.parse_args())
