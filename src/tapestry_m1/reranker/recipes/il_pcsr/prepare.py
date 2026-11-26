"""Normalize the gated native split, cache tokens, and mine train-only BM25 negatives."""
import argparse
from array import array
import hashlib
import pickle
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer
from .common import dump, INSTRUCTIONS

def digest(text): return hashlib.sha256(' '.join(text.lower().split()).encode()).hexdigest()

def main(a):
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    if (out/'ready.json').exists(): print('PREPARE already complete',flush=True); return
    raw=Path(a.raw)
    records={p.stem:pq.read_table(p).to_pylist() for p in raw.glob('*.parquet')}
    data={'queries':{},'splits':{},'documents':{},'gold':{},'negative':{}}
    audit={'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.glob('*.parquet')},'counts':{},'overlap':{},'tasks':{}}
    for split in ['train','dev','test']:
        rows=records[f'{split}_queries']; ids=[str(r['id']) for r in rows]
        assert len(ids)==len(set(ids)); data['splits'][split]=ids; audit['counts'][split]=len(ids)
        for r in rows:
            q=str(r['id']); assert q not in data['queries']
            data['queries'][q]='\n'.join(r['text'])
    for task,kind in [('lsr','statute'),('pcr','precedent')]:
        rows=records[f'{kind}_candidates']; docs={str(r['id']):'\n'.join(r['text']) for r in rows}
        assert len(docs)==len(rows); data['documents'][task]=docs
        gold={str(r['id']):sorted(set(str(x) for x in r[f'relevant_{kind}_ids'])) for s in ['train','dev','test'] for r in records[f'{s}_queries']}
        unknown={q:sorted(set(g)-docs.keys()) for q,g in gold.items() if set(g)-docs.keys()}
        assert not unknown, f'{task}: unknown gold IDs ({len(unknown)})'
        data['gold'][task]=gold; aliases={d:digest(t) for d,t in docs.items()}
        docids=list(docs)
        # Same smoothed IDF, b=.7, k1=1.6 as the reference BM25; bigrams fixed a priori.
        vectorizer=CountVectorizer(ngram_range=(2,2),max_df=.65,min_df=1,dtype=np.float32)
        counts=vectorizer.fit_transform(list(docs.values())).tocsr()
        dl=np.asarray(counts.sum(1)).ravel(); avg=dl.mean(); df=np.asarray((counts>0).sum(0)).ravel()
        weighted=counts.copy(); row=np.repeat(np.arange(len(docs)),np.diff(weighted.indptr))
        weighted.data=weighted.data*2.6/(weighted.data+1.6*(.3+.7*dl[row]/avg))
        weighted=weighted.multiply(np.log((len(docs)+1)/(df+1))).tocsr()
        negative={}; eligible=[q for q in data['splits']['train'] if gold[q]]
        for start in range(0,len(eligible),32):
            qs=eligible[start:start+32]; query=vectorizer.transform([data['queries'][q] for q in qs]); query.data[:]=1
            scores=(query@weighted.T).toarray()
            for q,score in zip(qs,scores):
                excluded={aliases[g] for g in gold[q]}; seen=set(excluded); chosen=[]
                for index in np.argsort(-score,kind='stable'):
                    d=docids[index]
                    if aliases[d] in seen: continue
                    chosen.append(d); seen.add(aliases[d])
                    if len(chosen)==3: break
                assert len(chosen)==3 and not set(chosen)&set(gold[q]); negative[q]=chosen
            if start%512==0: print(f'BM25 {task} {start}/{len(eligible)}',flush=True)
        data['negative'][task]=negative
        audit['tasks'][task]={'corpus':len(docs),'eligible_train':len(eligible),'empty_gold':{s:sum(not gold[q] for q in data['splits'][s]) for s in data['splits']},'duplicate_document_texts':len(docs)-len(set(aliases.values())),'bm25':{'ngrams':2,'max_df':.65,'b':.7,'k1':1.6,'min_df':1}}
    hashes={s:{digest(data['queries'][q]) for q in ids} for s,ids in data['splits'].items()}
    for s,t in [('train','dev'),('train','test'),('dev','test')]: audit['overlap'][f'{s}-{t}-normalized-text']=len(hashes[s]&hashes[t])
    dump(out/'data.json',data)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    cache={'queries':{},'documents':{}}; lengths={}
    for label,texts in [('queries',data['queries']),*[(t,data['documents'][t]) for t in ['lsr','pcr']]]:
        result={}; lens=[]
        for i,(key,text) in enumerate(texts.items()):
            tokens=tok.encode(text,add_special_tokens=False); lens.append(len(tokens))
            # Keep full tokenization on disk; per-pair budget allocation is deterministic.
            result[key]=array('I',tokens)
            if i%500==0: print(f'TOKENIZE {label} {i}/{len(texts)}',flush=True)
        if label=='queries': cache['queries']=result
        else: cache['documents'][label]=result
        lengths[label]={'p50_p90_p99_max':np.percentile(lens,[50,90,99,100]).tolist(),'over8192':sum(n>8192 for n in lens)}
    with (out/'tokens.pkl').open('wb') as f: pickle.dump(cache,f,protocol=5)
    audit['token_lengths']=lengths; audit['instructions']=INSTRUCTIONS
    dump(out/'ready.json',audit); print('PREPARE COMPLETE',audit['counts'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--raw',required=True); p.add_argument('--output',required=True); p.add_argument('--model',required=True); main(p.parse_args())
