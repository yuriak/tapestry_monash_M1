"""Build overlapping, paragraph-ending chunk spans over the existing full token cache."""
import argparse,hashlib,pickle
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from .il_metrics import load,dump

def spans(ids,size=2048,overlap=256,newlines=(198,271)):
    result=[];start=0
    while start<len(ids):
        end=min(start+size,len(ids))
        if end<len(ids):
            boundaries=[i+1 for i in range(start+size*3//4,end) if ids[i] in newlines]
            if boundaries:end=boundaries[-1]
        result.append((start,end))
        if end==len(ids):break
        start=end-overlap
    return result or [(0,0)]

def main(a):
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    data=load(Path(a.source)/'data.json')
    with open(Path(a.source)/'tokens.pkl','rb') as f:cache=pickle.load(f)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    newline_ids=tuple(t for t in set(tok.encode('\n\n',add_special_tokens=False)+tok.encode('\n',add_special_tokens=False)) if '\n' in tok.decode([t]))
    chunks={};summary={}
    for label,texts in [('queries',cache['queries']),('lsr',cache['documents']['lsr']),('pcr',cache['documents']['pcr'])]:
        chunks[label]={k:spans(v,newlines=newline_ids) for k,v in texts.items()}
        summary[label]={'items':len(texts),'chunks':sum(map(len,chunks[label].values())),'max_chunks':max(map(len,chunks[label].values()))}
        for k,ss in chunks[label].items():
            assert ss[0][0]==0 and ss[-1][1]==len(texts[k])
            assert all(0<=s<=e<=len(texts[k]) and e-s<=2048 for s,e in ss)
            assert all(ss[i+1][0]<=ss[i][1] for i in range(len(ss)-1))
    with open(root/'chunks.pkl','wb') as f:pickle.dump({'tokens':cache,'spans':chunks},f,protocol=5)
    for task in ['lsr','pcr']:
        out=root/task;out.mkdir(exist_ok=True)
        # Preserve native IDs and splits, including empty-gold held-out queries.
        pairs=[[q,d] for q in data['splits']['train'] for d in data['gold'][task][q]]
        obj={'splits':data['splits'],'gold':data['gold'][task],'train_pairs':pairs,'query_ids':list(data['queries']),'document_ids':list(data['documents'][task])}
        dump(out/'data.json',obj)
        dump(out/'manifest.json',{'task':task,'train_pairs':len(pairs),'splits':{s:len(qs) for s,qs in data['splits'].items()},'max_length':8192,'chunk_size':2048,'overlap':256,'selection':'top2 query chunks by document affinity, top2 document chunks by selected-query affinity; restore order, merge overlaps','full_positive_pass':True,'seed':42,'global_batch':64})
    dump(root/'manifest.json',{'chunks':summary,'source_sha256':hashlib.sha256((Path(a.source)/'data.json').read_bytes()).hexdigest(),'token_cache_sha256':hashlib.sha256((Path(a.source)/'tokens.pkl').read_bytes()).hexdigest(),'newline_token_ids':newline_ids,'no_full_text_truncation_during_chunking':True})
    print('CHUNKS READY',summary,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--model',required=True);main(p.parse_args())
