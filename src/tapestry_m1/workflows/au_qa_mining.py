"""Freeze same-split QA retrieval and assumed negatives; no native test retrieval."""
import json
import io
from pathlib import Path
import random
from tapestry_m1.core.io import ContractError,atomic_json,fingerprint,records,text_key
from tapestry_m1.data.au_qa_training import near_pairs,write_rows
from tapestry_m1.contracts.reranker_training import text_hash
from tapestry_m1.contracts.reranker_training_v2 import SCHEMA,validate_groups
from tapestry_m1.runtime.assets import verify_model
from tapestry_m1.runtime.devices import gpu_runtime
from tapestry_m1.runtime.provenance import code_fingerprint
from tapestry_m1.retrieval.qwen3 import Qwen3Encoder
from tapestry_m1.retrieval.search import exact_topk
from tapestry_m1.retrieval.encoding import encode_cached
from tapestry_m1.core.artifacts import atomic_bytes


def choose_negatives(query, passages, ranking, near, seed):
    by_id={p['doc_id']:p for p in passages};known=set(query['known_relevant_ids'])
    sources=set().union(*(set(by_id[d]['source_doc_ids']) for d in known))
    positive_texts={text_key(by_id[d]['text']) for d in known}
    excluded=set(known)
    for d in known:excluded.update(near.get(d,()))
    eligible={p['doc_id'] for p in passages if p['doc_id'] not in excluded
              and p.get('split')==query.get('split') and not sources.intersection(p['source_doc_ids']) and text_key(p['text']) not in positive_texts}
    def compatible(doc,chosen):
        return all(doc not in near.get(other,()) and text_key(by_id[doc]['text'])!=text_key(by_id[other]['text'])
                   and not set(by_id[doc]['source_doc_ids']).intersection(by_id[other]['source_doc_ids']) for other in chosen)
    hard=[]
    for row in ranking:
        d=row['doc_id']
        if d in eligible and d not in hard and compatible(d,hard):hard.append(d)
        if len(hard)==2:break
    random_pool=sorted(d for d in eligible-set(hard) if compatible(d,hard))
    if len(hard)!=2 or not random_pool:raise ContractError('insufficient distinct same-split negatives')
    local_seed=seed ^ int(text_hash(query['query_id'])[:16],16)
    return [random.Random(local_seed).choice(random_pool),*hard],sorted(sources)


def mine(collection,output,config,data_root):
    import numpy as np
    collection,output=Path(collection),Path(output)
    source=json.loads((collection/'manifest.json').read_text())
    for name,sha in source['files'].items():
        if fingerprint(collection/name)!=sha:raise ContractError('collection input checksum mismatch')
    identity={'source':fingerprint(collection/'manifest.json'),'config':config,'code':code_fingerprint()}
    if (output/'manifest.json').exists():
        saved=json.loads((output/'manifest.json').read_text())
        if saved['inputs']!=identity:raise ContractError('mining identity differs; use separate output')
        for name,sha in saved['files'].items():
            if fingerprint(output/name)!=sha:raise ContractError('mined groups/ranking checksum differs')
        return saved
    output.mkdir(parents=True,exist_ok=True)
    hardware=gpu_runtime();model_dir,model_identity=verify_model(config['encoder'],data_root)
    encoder=Qwen3Encoder(model_dir,config)
    queries=list(records(collection/'queries.jsonl'));passages=list(records(collection/'passages.jsonl'))
    near={p['doc_id']:set() for p in passages}
    for a,b,_ in near_pairs([p['text'] for p in passages]):
        near[passages[a]['doc_id']].add(passages[b]['doc_id']);near[passages[b]['doc_id']].add(passages[a]['doc_id'])
    groups=[];rankfiles=[];exclusions=[];audits={}
    for split in ('train','dev'):
        qs=[q for q in queries if q['split']==split];docs=[p for p in passages if p['split']==split]
        path=output/(split+'-vectors.npz')
        vector_meta=output/(split+'-vectors.json')
        vector_identity={'inputs':identity,'split':split,'query_ids':[q['query_id'] for q in qs],'doc_ids':[p['doc_id'] for p in docs]}
        if path.exists() and vector_meta.exists():
            info=json.loads(vector_meta.read_text())
            if info['identity']!=vector_identity or info['sha256']!=fingerprint(path):raise ContractError('embedding cache differs')
            with np.load(path) as archive:dv,qv=archive['documents'],archive['queries']
            audits[split]=info['audit']
        else:
            dv,da,_=encode_cached(encoder,docs,'text',False,output/'vector-shards'/split/'passages',vector_identity,config)
            qv,qa,_=encode_cached(encoder,qs,'query_text',True,output/'vector-shards'/split/'queries',vector_identity,config)
            buffer=io.BytesIO();np.savez(buffer,documents=dv,queries=qv);atomic_bytes(path,buffer.getvalue())
            audits[split]={'passages':da,'queries':qa}
            atomic_json(vector_meta,{'identity':vector_identity,'sha256':fingerprint(path),'audit':audits[split]})
        ranks=exact_topk([p['doc_id'] for p in docs],dv,qv,100)
        rank_path=output/(split+'-ranking.jsonl')
        write_rows(rank_path,[{'query_id':q['query_id'],'split':split,'candidates':r} for q,r in zip(qs,ranks)])
        rankfiles.append(rank_path.name);lookup={p['doc_id']:p for p in docs};indices={p['doc_id']:i for i,p in enumerate(docs)}
        for qi,(q,rank) in enumerate(zip(qs,ranks)):
            try:
                selected,sources=choose_negatives(q,docs,rank,near,config['seed'])
            except ContractError:
                rank=exact_topk([p['doc_id'] for p in docs],dv,qv[qi:qi+1],len(docs))[0]
                try:selected,sources=choose_negatives(q,docs,rank,near,config['seed'])
                except ContractError:
                    exclusions.append({'query_id':q['query_id'],'reason':'insufficient distinct same-split negatives after full-pool expansion'});continue
            rank_lookup={r['doc_id']:r for r in rank}
            candidates=[]
            for n,doc_id in enumerate([q['positive_id']]+selected):
                p=lookup[doc_id]
                candidates.append({'doc_id':doc_id,'source_doc_id':p['source_doc_ids'][0],'source_doc_ids':p['source_doc_ids'],
                  'text':p['text'],'source_hash':p['source_hash'],'split':p['split'],'label':int(n==0),'negative_kind':['positive','random','hard','hard'][n],
                  'ambiguous':n>0,'supervision':'source_positive' if n==0 else 'assumed',
                  'retrieval_rank':rank_lookup.get(doc_id,{}).get('rank'),
                  'similarity':float(np.dot(qv[qi],dv[indices[doc_id]]))})
            groups.append({'schema_version':SCHEMA,'synthetic':False,'collection_id':q['collection_id'],'query_id':q['query_id'],
                'source_query_id':q['source_query_id'],'query_text':q['query_text'],'source_hash':q['source_hash'],'split':split,
                'query_source_doc_ids':sources,'known_relevant_ids':q['known_relevant_ids'],'seed':config['seed'],
                'provenance':'pinned QA source.text excerpt; original row '+str(q['source_row']),
                'original_provenance':q['provenance'],'source_metadata':q['source_metadata'],'candidates':candidates})
    encoder.close()
    validate_groups(groups)
    write_rows(output/'groups.jsonl',groups)
    result={'schema_version':'m1.auqa.groups.v2','inputs':identity,'model':model_identity,'hardware':hardware,
      'counts':{s:sum(g['split']==s for g in groups) for s in ['train','dev']},'exclusions':exclusions,
      'false_negative_caveat':'Unjudged non-positive same-split retrieval candidates are assumed negatives, not certified irrelevance.',
      'files':{name:fingerprint(output/name) for name in rankfiles+['groups.jsonl']},
      'embedding_cache_files':{name:fingerprint(output/name) for name in ['train-vectors.npz','dev-vectors.npz']}}
    atomic_json(output/'manifest.json',result);return result
