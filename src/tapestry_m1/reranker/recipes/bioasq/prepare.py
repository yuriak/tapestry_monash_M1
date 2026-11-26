"""Prepare a pinned community subset, preserving its published test questions."""
import argparse, hashlib, pickle, random, re
from array import array
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from .common import load,dump

def norm(s): return ' '.join(s.casefold().split())
def main(a):
    root=Path(a.data); source={'dataset':'enelpol/rag-mini-bioasq','revision':'8a845907dc1cff31d42fa6f7bb9c6eef5f3ae6f6'};raw=root/'raw'
    documents={}; duplicate_documents=0
    # The dataset card selects text-corpus/test-*; any train file is inspected separately.
    df=pd.read_parquet(raw/'text-corpus/test-00000-of-00001.parquet')
    for r in df.to_dict('records'):
        key=str(int(r['id']));text=r['passage']
        if not isinstance(text,str) or not text.strip() or text.strip().lower()=='nan': continue
        if key in documents:
            assert documents[key]==text;duplicate_documents+=1
        documents[key]=text
    frames={s:pd.read_parquet(raw/f'question-answer-passages/{s}-00000-of-00001.parquet') for s in ['train','test']}
    test_texts={norm(r['question']) for r in frames['test'].to_dict('records')}
    data={'documents':documents,'queries':{},'gold':{},'splits':{}}; discarded=[];missing={};seen={};query_ids={}
    for split in ['test','train']:
        ids=[]
        for row in frames[split].to_dict('records'):
            q=f"q{int(row['id'])}";text=row['question'];key=norm(text)
            if split=='train' and (key in test_texts or key in seen):
                reason='overlap-with-test' if key in test_texts else 'merged-training-duplicate'
                if key not in test_texts:
                    canonical=seen[key]
                    data['gold'][canonical]=list(dict.fromkeys(data['gold'][canonical]+[str(int(d)) for d in row['relevant_passage_ids']]))
                    missing[canonical]=[d for d in data['gold'][canonical] if d not in documents]
                discarded.append({'id':q,'reason':reason});continue
            # Test labels stay unchanged; duplicate training questions union their labels.
            if q in data['queries']: raise ValueError('Duplicate question ID across splits')
            gold=list(dict.fromkeys(str(int(d)) for d in row['relevant_passage_ids']))
            missing[q]=[d for d in gold if d not in documents]
            # Keep all qrels in denominator, even absent documents; never change test.
            data['queries'][q]=text;data['gold'][q]=gold;ids.append(q);seen[key]=q
        data['splits'][split]=ids
    order=list(data['splits']['train']);random.Random(42).shuffle(order)
    data['splits']['dev']=sorted(order[:400]);data['splits']['train']=sorted(order[400:])
    data['train_pairs']=[[q,d] for q in data['splits']['train'] for d in data['gold'][q] if d in documents]
    assert not set(data['splits']['train'])&set(data['splits']['test'])
    write_dataset(root, data, a.model, source, discarded, missing,
                  {str(p.relative_to(raw)):hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.rglob('*.parquet')},
                  corpus_file='text-corpus/test-00000-of-00001.parquet')


def write_dataset(root, data, model, source, discarded, missing, files, corpus_file, instruction=None):
    root.mkdir(parents=True, exist_ok=True)
    documents = data['documents']
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True);cache={};lengths={}
    for label,texts in [('queries',data['queries']),('documents',documents)]:
        keys=list(texts);cache[label]={}
        for i in range(0,len(keys),512):
            chunk=keys[i:i+512];encoded=tok([texts[k] for k in chunk],add_special_tokens=False)['input_ids']
            cache[label].update({k:array('I',ids) for k,ids in zip(chunk,encoded)})
        lens=[len(v) for v in cache[label].values()]
        lengths[label]={'count':len(lens),'p50_p90_p95_p99_max':np.percentile(lens,[50,90,95,99,100]).tolist()}
    # 2048 minimum, expand only if p99 query + document + prompt exceeds it.
    p99=sum(np.percentile([len(v) for v in cache[label].values()],99) for label in cache)+100
    limit=next((n for n in [2048,4096,8192] if n>=p99),8192)
    pair_lengths=[len(cache['queries'][q])+len(cache['documents'][d])+100 for q,d in data['train_pairs']]
    audit={'source':source,'corpus_file':corpus_file,'documents':len(documents),'splits':{s:len(qs) for s,qs in data['splits'].items()},'train_pairs':len(data['train_pairs']),'discarded_train':discarded,'missing_gold_counts':{s:sum(len(missing[q]) for q in qs) for s,qs in data['splits'].items()},'zero_available_gold_queries':{s:sum(not(set(data['gold'][q])&documents.keys()) for q in qs) for s,qs in data['splits'].items()},'lengths':lengths,'max_length':limit,'train_pair_fraction_over_limit':float(np.mean(np.array(pair_lengths)>limit)),'map10_denominator':'min(number of labeled relevant documents,10); empty qrels score 0','seed':42,'files':files}
    if instruction:
        audit['instruction'] = instruction
    dump(root/'data.json',data);dump(root/'manifest.json',audit)
    with open(root/'tokens.pkl','wb') as f:pickle.dump(cache,f,protocol=5)
    print(audit,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--model',required=True);main(p.parse_args())
