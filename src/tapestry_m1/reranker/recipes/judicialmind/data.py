"""Bounded JudicialMind AU/IN preparation; no private site paths or redistributed data."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer


def norm(t): return ' '.join(re.findall(r'\w+',t.lower()))
def digest(t): return hashlib.sha256(t.encode()).hexdigest()
def dump(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def load(p):return json.loads(Path(p).read_text())
def jsonl(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def shingles(t,n=5):
    words=norm(t).split();return {' '.join(words[i:i+n]) for i in range(max(0,len(words)-n+1))} or {norm(t)}
def near_overlap(a,b,threshold=.8):
    return bool(a and b and len(a&b)/len(a|b)>=threshold)

def balanced(rows,cap):
    """Deterministic round-robin strata, keeping domains and documents represented."""
    strata=defaultdict(list)
    for r in rows:strata[(r['country'],r['domain'])].append(r)
    for values in strata.values():values.sort(key=lambda r:digest(r['id']+'selection42'))
    picked=[]; index=0
    while len(picked)<cap:
        changed=False
        for key in sorted(strata):
            if index<len(strata[key]):picked.append(strata[key][index]);changed=True
            if len(picked)==cap:break
        if not changed:break
        index+=1
    return picked

def prepare(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);raw=Path(a.raw)
    tax_source=Path(a.tax)
    tax_docs={x['doc_id']:x['text'] for x in jsonl(tax_source/'documents.jsonl')}
    tax_queries={x['query_id']:x['query_text'] for x in jsonl(tax_source/'queries.jsonl')}
    tax_gold=defaultdict(dict)
    for r in jsonl(tax_source/'qrels.jsonl'):tax_gold[r['query_id']][r['doc_id']]=r['relevance']
    assert len(tax_docs)==105 and len(tax_queries)==112
    benchmark_texts=list(tax_docs.values())+list(tax_queries.values())
    tax_hash={digest(norm(t)) for t in benchmark_texts}
    tax_shingles=[shingles(t) for t in benchmark_texts]
    # Exact 5-word Jaccard >= .8: length bound and inverted shingles prune comparisons.
    inverted=defaultdict(set)
    for i,ss in enumerate(tax_shingles):
        for sh in ss:inverted[sh].add(i)
    def contaminated(text):
        if digest(norm(text)) in tax_hash:return True
        ss=shingles(text);hits=Counter(i for sh in ss for i in inverted.get(sh,()))
        for i,n in hits.items():
            other=tax_shingles[i]
            if min(len(ss),len(other)) < .8*max(len(ss),len(other)):continue
            if n/(len(ss)+len(other)-n)>=.8:return True
        return False
    rows=[];counts=Counter();files={};seen=set();removed=Counter()
    for p in sorted(raw.rglob('*.parquet')):
        source=p.stem.rsplit('_',1)[0];files[str(p.relative_to(raw))]={'rows':pq.ParquetFile(p).metadata.num_rows,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        for batch in pq.ParquetFile(p).iter_batches(batch_size=4096):
            for x in batch.to_pylist():
                country=x['country'].lower();bucket=x['bucket'];counts[(country,bucket)]+=1
                q=x['query'].strip();d=x['positive'].strip()
                if not q or not d:removed['empty']+=1;continue
                qh=digest(norm(q));dh=digest(norm(d));identifier=digest(qh+dh)
                # Same pair repeated in different buckets is handled by held-out priority below.
                key=(bucket,identifier)
                if key in seen:removed['within_bucket_duplicate_pair']+=1;continue
                seen.add(key)
                rows.append({'id':identifier,'query':q,'positive':d,'qh':qh,'dh':dh,'docgroup':country+':'+source+':'+str(x['doc_id']),
                    'country':country,'domain':x['legal_domain'],'jurisdiction':x['jurisdiction'],'bucket':bucket})
        print('READ',p.name,files[str(p.relative_to(raw))]['rows'],flush=True)
    pools={b:[r for r in rows if r['bucket']==b] for b in 'ABC'}
    signatures={b:{k:{r[k] for r in pools[b]} for k in ['qh','dh','docgroup']} for b in 'ABC'}
    clean={};overlap={}
    for b,later in [('A','BC'),('B','C'),('C','')]:
        blocked={k:set().union(*(signatures[t][k] for t in later)) for k in ['qh','dh','docgroup']}
        clean[b]=[];overlap[b]=Counter()
        for r in pools[b]:
            reasons=[k for k in blocked if r[k] in blocked[k]]
            if reasons:
                overlap[b].update(reasons);continue
            clean[b].append(r)
    # Keep all available AU training pairs and up to 25k balanced Indian pairs.
    selected={'train':balanced([r for r in clean['A'] if r['country']=='au'],a.train_cap)+balanced([r for r in clean['A'] if r['country']=='in'],a.train_cap)}
    for split,b in [('dev','B'),('test','C')]:
        # Overselect rows, then deduplicate query IDs to obtain the requested native-query count.
        candidates=balanced(clean[b],len(clean[b]));qs=set(); chosen=[]
        for r in candidates:
            if r['qh'] in qs:continue
            if contaminated(r['query']) or contaminated(r['positive']):removed[f'{split}_tax_overlap']+=1;continue
            chosen.append(r);qs.add(r['qh'])
            if len(qs)>=a.eval_queries:break
        selected[split]=chosen
    # Audit ALL texts for selected query groups, including known additional positives.
    wanted={r['qh'] for rs in selected.values() for r in rs}
    all_gold=defaultdict(dict); docmeta={}
    for r in rows:
        if r['qh'] in wanted:all_gold[r['qh']][r['dh']]=r['positive']
        meta=docmeta.setdefault(r['dh'],{'groups':set(),'memberships':set()})
        meta['groups'].add(r['docgroup']);meta['memberships'].add((r['country'],r['jurisdiction'],r['domain']))
    docmeta={d:{'groups':sorted(m['groups']),'memberships':sorted(m['memberships'])} for d,m in docmeta.items()}
    bad_query=set()
    for i,q in enumerate(sorted(wanted)):
        if any(contaminated(t) for t in all_gold[q].values()):bad_query.add(q)
        if i%2000==0:print('DECONTAMINATE',i,len(wanted),flush=True)
    for split,rs in selected.items():
        filtered=[]
        for r in rs:
            if r['qh'] in bad_query or contaminated(r['query']):removed[f'{split}_tax_overlap']+=1
            else:filtered.append(r)
        selected[split]=filtered
    data={'train':[],'documents':{},'metadata':{},'dev':{},'test':{},'tax':{'queries':tax_queries,'documents':tax_docs,'gold':dict(tax_gold)},'known_gold':{}}
    for split,rs in selected.items():
        for r in rs:
            data['known_gold'][r['qh']]=list(all_gold[r['qh']])
            for d,t in all_gold[r['qh']].items():data['documents'][d]=t;data['metadata'][d]=docmeta[d]
            if split=='train':data['train'].append({k:r[k] for k in ['id','query','qh','dh','docgroup','country','domain','jurisdiction']})
            else:data[split][r['qh']]={'query':r['query'],'gold':list(all_gold[r['qh']]),'country':r['country'],'domain':r['domain']}
    heldout_documents={d for split in ['dev','test'] for r in data[split].values() for d in r['gold']}
    heldout_groups={g for d in heldout_documents for g in data['metadata'][d]['groups']}
    clean_train=[r for r in data['train'] if r['dh'] not in heldout_documents and r['docgroup'] not in heldout_groups]
    removed['additional_positive_parent_overlap']=len(data['train'])-len(clean_train);data['train']=clean_train
    # Mine lexical hard negatives strictly within same jurisdiction and legal domain.
    # No dates exist in the released schema; temporal matching is not fabricated.
    trainids={r['dh'] for r in data['train']};groups=defaultdict(set)
    for r in data['train']:groups[(r['country'],r['jurisdiction'],r['domain'])].add(r['dh'])
    negatives={};fallback=0
    for key,ds in sorted(groups.items()):
        ids=sorted(ds); qs=[r for r in data['train'] if (r['country'],r['jurisdiction'],r['domain'])==key]
        if len(ids)<4:continue
        vec=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=100000,dtype=np.float32)
        matrix=vec.fit_transform([data['documents'][d] for d in ids])
        for offset in range(0,len(qs),32):
            part=qs[offset:offset+32];scores=(vec.transform([r['query'] for r in part])@matrix.T).toarray()
            for r,values in zip(part,scores):
                forbidden=set(data['known_gold'][r['qh']]);rank=np.argsort(-values,kind='stable');chosen=[]
                for index in rank:
                    d=ids[index]
                    if d in forbidden or r['docgroup'] in data['metadata'][d]['groups']:continue
                    chosen.append(d)
                    if len(chosen)==3:break
                if len(chosen)==3:negatives[r['id']]=chosen
        print('MINE',key,'queries',len(qs),'documents',len(ids),flush=True)
    # Matched arms use only examples with three jurisdiction/domain-matched negatives.
    before=len(data['train']);data['train']=[r for r in data['train'] if r['id'] in negatives]
    data['hard_negatives']=negatives
    audit={'source_files':files,'raw_counts':{f'{c}/{b}':n for (c,b),n in counts.items()},
        'removed':dict(removed),'heldout_priority_removals':{b:dict(v) for b,v in overlap.items()},
        'train_cap_india':a.train_cap,'train_rows':len(data['train']),'without_3_matched_negatives':before-len(data['train']),
        'train_countries':dict(Counter(r['country'] for r in data['train'])),'train_domains':dict(Counter(r['domain'] for r in data['train'])),
        'dev_queries':len(data['dev']),'test_queries':len(data['test']),'tax_queries':len(tax_queries),'tax_documents':len(tax_docs),
        'decontamination':'normalized exact and exact 5-word Jaccard >=0.8 against AU Tax query and passage texts',
        'temporal_matching':'not available: released rows contain no date field',
        'negative_mining':'TF-IDF unigram/bigram, same country/jurisdiction/domain; excludes all known positives and parent document',
        'no_full_paper_claim':True}
    dump(out/'data.json',data);dump(out/'audit.json',audit);print('PREPARED',audit['train_rows'],audit['train_countries'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--tax',required=True);p.add_argument('--output',required=True);p.add_argument('--train-cap',type=int,default=25000);p.add_argument('--eval-queries',type=int,default=512);prepare(p.parse_args())
