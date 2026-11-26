"""NFCorpus v1 preparation and sealed-dev baseline. No training or test evaluation."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import time
import zipfile
import numpy as np
from tapestry_m1.core.io import atomic_json, fingerprint, records, ContractError


def load(path): return json.loads(Path(path).read_text())
def tokens(text): return re.findall(r'[a-z0-9]+', text.lower())
def representation(row): return (row.get('title','').strip()+'\n'+row['text'].strip()).strip()
def write_rows(path, values):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        for row in values: f.write(json.dumps(row,sort_keys=True)+'\n')
    temp.replace(path)


class BM25:
    """Exact full-corpus Okapi with log(1+(N-df+.5)/(df+.5)) IDF."""
    def __init__(self, documents, k1=1.2, b=.75):
        self.ids=sorted(documents);self.k1=k1;self.b=b;self.postings=defaultdict(list)
        self.lengths=np.array([len(tokens(documents[d])) for d in self.ids],float)
        self.avg=float(self.lengths.mean())
        if self.avg<=0: raise ContractError('empty token corpus')
        for i,d in enumerate(self.ids):
            for term,tf in Counter(tokens(documents[d])).items():self.postings[term].append((i,tf))
    def rank(self,query,k=50):
        scores=np.zeros(len(self.ids),float);n=len(self.ids)
        for term in sorted(set(tokens(query))):
            posting=self.postings.get(term,[]);idf=math.log1p((n-len(posting)+.5)/(len(posting)+.5))
            for i,tf in posting:
                scores[i]+=idf*tf*(self.k1+1)/(tf+self.k1*(1-self.b+self.b*self.lengths[i]/self.avg))
        order=sorted(range(n),key=lambda i:(-scores[i],self.ids[i]))[:k]
        return [{'doc_id':self.ids[i],'score':float(scores[i]),'rank':r+1} for r,i in enumerate(order)]


def qrels(path,queries,docs):
    result=defaultdict(dict)
    with path.open(newline='') as f:
        reader=csv.reader(f,delimiter='\t');header=next(reader)
        if header!=['query-id','corpus-id','score']:raise ContractError('unexpected qrel header')
        for q,d,v in reader:
            grade=int(v)
            if q not in queries or d not in docs or grade<0 or d in result[q]:raise ContractError('invalid qrel join, grade or duplicate')
            result[q][d]=grade
    return dict(result)


def metrics(rankings,judgments):
    if {r['query_id'] for r in rankings}!=set(judgments):raise ContractError('query coverage mismatch')
    output=[]
    for row in rankings:
        gold=judgments[row['query_id']];positive={d for d,g in gold.items() if g>0};cs=row['candidates']
        if len({c['doc_id'] for c in cs})!=len(cs):raise ContractError('duplicate candidate')
        ideal=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(sorted(gold.values(),reverse=True)[:10]))
        dcg=sum((2**gold.get(c['doc_id'],0)-1)/math.log2(i+2) for i,c in enumerate(cs[:10]))
        hit=[i+1 for i,c in enumerate(cs) if c['doc_id'] in positive]
        output.append({'query_id':row['query_id'],'ndcg@10':dcg/ideal if ideal else 0.,'recall@10':sum(i<=10 for i in hit)/len(positive) if positive else 0.,'recall@50':len(hit)/len(positive) if positive else 0.,'mrr@50':1/min(hit) if hit else 0.,'hit@50':int(bool(hit))})
    return {'queries':len(output),'gain':'2^rel-1','means':{k:sum(r[k] for r in output)/len(output) for k in output[0] if k!='query_id'},'per_query':output}


def source(root):
    documents=list(records(root/'corpus.jsonl'));queries=list(records(root/'queries.jsonl'))
    if len({r['_id'] for r in documents})!=len(documents) or len({r['_id'] for r in queries})!=len(queries):raise ContractError('duplicate source ID')
    docs={r['_id']:representation(r) for r in documents};qs={r['_id']:r['text'] for r in queries}
    if not all(x.strip() for x in [*docs.values(),*qs.values()]):raise ContractError('empty text')
    return docs,qs


def prepare(config,archive,out):
    out.mkdir(parents=True,exist_ok=True);c=load(config)
    if hashlib.md5(archive.read_bytes()).hexdigest()!=c['archive_md5']:raise ContractError('archive checksum mismatch')
    raw=out/'raw'
    if not raw.exists():
        raw.mkdir()
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                p=Path(member.filename)
                if p.is_absolute() or '..' in p.parts:raise ContractError('unsafe zip path')
            z.extractall(raw)
    root=raw/'nfcorpus';docs,qs=source(root)
    # Audit all split joins/grades/duplicate-text counts here only. No test metrics or labels exported to training.
    splits={s:qrels(root/'qrels'/f'{s}.tsv',qs,docs) for s in ['train','dev','test']}
    counts={s:{'queries':len(j),'judgments':sum(map(len,j.values())),'grades':dict(Counter(g for v in j.values() for g in v.values()))} for s,j in splits.items()}
    overlap={};sets={q:set(tokens(t)) for q,t in qs.items()};norm={q:' '.join(tokens(t)) for q,t in qs.items()}
    for left,right in [('train','dev'),('train','test'),('dev','test')]:
        same=set(splits[left])&set(splits[right])
        exact=sum(norm[q]==norm[r] for q in splits[left] for r in splits[right]);near=0
        for q in splits[left]:
            a=sets[q]
            for r in splits[right]:
                b=sets[r]
                if min(len(a),len(b))<.9*max(len(a),len(b)):continue
                if len(a&b)/max(1,len(a|b))>=.9:near+=1
        overlap[left+'-'+right]={'same_query_ids':len(same),'normalized_exact_pairs':exact,'token_set_jaccard_ge09_pairs_including_exact':near}
        if same:raise ContractError('query IDs shared across native splits')
    identity={'config':fingerprint(config),'code':fingerprint(__file__),'archive_sha256':fingerprint(archive),'archive_md5':c['archive_md5'],'url':c['archive_url'],'terms':c['terms'],'terms_url':c['terms_url'],'source_files':{str(p.relative_to(root)):fingerprint(p) for p in sorted(root.rglob('*')) if p.is_file()},'corpus':len(docs),'all_queries':len(qs),'splits':counts,'overlap_audit_only':overlap,'shared_corpus':True,'test_policy':'test labels opened only by isolated source auditor; not exported to retrieval/training/selection; no test evaluation before selection.json'}
    if (out/'source.json').exists() and load(out/'source.json')!=identity:raise ContractError('source/config/code changed; use a fresh namespace')
    atomic_json(out/'source.json',identity)
    engine=BM25(docs,c['bm25']['k1'],c['bm25']['b']);summaries={};out.joinpath('bm25').mkdir(exist_ok=True)
    for split in ['train','dev']:
        rankings=[{'query_id':q,'candidates':engine.rank(qs[q],50)} for q in sorted(splits[split])]
        p=out/'bm25'/f'{split}.jsonl';write_rows(p,rankings);m=metrics(rankings,splits[split]);atomic_json(out/'bm25'/f'{split}-metrics.json',m);summaries[split]={'metrics':m['means'],'rankings_sha256':fingerprint(p),'queries':len(rankings)}
    gate=all(s['metrics']['hit@50']>=c['minimum_query_hit_rate'] for s in summaries.values())
    atomic_json(out/'prepared.json',{'status':'ready' if gate else 'blocked-low-coverage','identity':fingerprint(out/'source.json'),'splits':summaries,'minimum_query_hit_rate':c['minimum_query_hit_rate']})
    if not gate:raise ContractError('predeclared candidate coverage gate failed')
    print(json.dumps(load(out/'prepared.json')),flush=True)


def verify_inputs(config,out):
    identity=load(out/'source.json');prepared=load(out/'prepared.json')
    if prepared['status']!='ready' or identity['config']!=fingerprint(config) or prepared['identity']!=fingerprint(out/'source.json'):raise ContractError('prepared identity changed')
    for rel,digest in identity['source_files'].items():
        if fingerprint(out/'raw/nfcorpus'/rel)!=digest:raise ContractError('raw source changed')
    for split,v in prepared['splits'].items():
        if fingerprint(out/'bm25'/f'{split}.jsonl')!=v['rankings_sha256']:raise ContractError('candidate rankings changed')
    return identity


def baseline(config,out,data_root,preflight_only=False):
    from tapestry_m1.runtime.assets import verify_model
    from tapestry_m1.runtime.devices import gpu_runtime
    from tapestry_m1.reranker.qwen3 import Qwen3Reranker
    c=load(config);identity=verify_inputs(config,out);hardware=gpu_runtime();spec=load(config.parent/c['model']);modeldir,model_identity=verify_model(spec,data_root)
    docs,qs=source(out/'raw/nfcorpus');frozen=list(records(out/'bm25/dev.jsonl'));backend=Qwen3Reranker(modeldir,c)
    if (backend.yes,backend.no)!=(9693,2152):raise ContractError('token IDs changed')
    q=frozen[0]['query_id'];ids=[x['doc_id'] for x in frozen[0]['candidates'][:8]]
    one,audit=backend.score_logits([qs[q]]*8,[docs[d] for d in ids],1);two,_=backend.score_logits([qs[q]]*8,[docs[d] for d in ids],2)
    dif=max(abs(x-y) for x,y in zip(one,two));disagree=sum((one[i]>one[j])!=(two[i]>two[j]) for i in range(8) for j in range(i))
    long=' '.join([docs[ids[0]]]*20);long_scores,long_audit=backend.score_logits([qs[q]],[long],1)
    if not long_audit[0]['truncated'] or not all(math.isfinite(s) for s in [*one,*two,*long_scores]):raise ContractError('long-input/finite preflight failed')
    gate=c['preflight'];pf={'max_abs_batch_difference':dif,'order_disagreements':disagree,'hardware':hardware,'audit':audit,'long_input':long_audit,'config_sha256':fingerprint(config),'code_sha256':fingerprint(__file__),'model':model_identity,'batch':1,'token_ids':[9693,2152]}
    atomic_json(out/'preflight.json',pf)
    if dif>gate['max_abs_logit_difference'] or disagree>gate['max_pairwise_order_disagreements']:raise ContractError('batch consistency gate failed')
    if preflight_only:backend.close();print('GPU preflight passed',flush=True);return
    cache=out/'zero-shot/dev';cache.mkdir(parents=True,exist_ok=True);start=time.time();results=[];truncated=0
    scoring_identity={'config':fingerprint(config),'code':fingerprint(__file__),'source':fingerprint(out/'source.json'),'ranking':fingerprint(out/'bm25/dev.jsonl'),'model':spec['revision'],'environment_lock':fingerprint(__import__('os').environ['M1_ENVIRONMENT_LOCK'])}
    for i,row in enumerate(frozen):
        q=row['query_id'];p=cache/(hashlib.sha256(q.encode()).hexdigest()+'.json');ids=[x['doc_id'] for x in row['candidates']]
        if p.exists():
            saved=load(p)
            if saved['identity']!=scoring_identity or saved['query_id']!=q:raise ContractError('cache identity mismatch')
        else:
            scores,lengths=backend.score_logits([qs[q]]*50,[docs[d] for d in ids],1)
            if not all(math.isfinite(s) for s in scores):raise ContractError('nonfinite score')
            candidates=sorted([dict(x,score=s,retrieval_score=x['score'],retrieval_rank=x['rank']) for x,s in zip(row['candidates'],scores)],key=lambda x:(-x['score'],x['doc_id']))
            for rank,x in enumerate(candidates,1):x['rank']=rank
            saved={'identity':scoring_identity,'query_id':q,'candidates':candidates,'lengths':lengths};atomic_json(p,saved)
        if set(ids)!={x['doc_id'] for x in saved['candidates']}:raise ContractError('candidate membership changed')
        results.append({'query_id':q,'candidates':saved['candidates']});truncated+=sum(x['truncated'] for x in saved['lengths']);print(f'dev {i+1}/{len(frozen)}',flush=True)
    pred=out/'zero-shot/dev.jsonl';write_rows(pred,results)
    judgments=qrels(out/'raw/nfcorpus/qrels/dev.tsv',qs,docs);m=metrics(results,judgments);b0=load(out/'bm25/dev-metrics.json')
    for a,b in zip(m['per_query'],b0['per_query']):
        if a['query_id']!=b['query_id'] or a['recall@50']!=b['recall@50']:raise ContractError('recall invariance failed')
    atomic_json(out/'zero-shot/dev-metrics.json',m);backend.close()
    atomic_json(out/'stage-a-complete.json',{'status':'verified-stage-a-complete','identity':scoring_identity,'queries':len(results),'pairs':50*len(results),'truncated_pairs':truncated,'seconds':time.time()-start,'B0':b0['means'],'B1':m['means'],'predictions_sha256':fingerprint(pred),'metrics_sha256':fingerprint(out/'zero-shot/dev-metrics.json'),'test_evaluated':False,'training_started':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','preflight','baseline']);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path);p.add_argument('--data-root',type=Path);a=p.parse_args()
    if a.stage=='prepare':prepare(a.config,a.archive,a.output)
    else:baseline(a.config,a.output,a.data_root,a.stage=='preflight')
if __name__=='__main__':main()
