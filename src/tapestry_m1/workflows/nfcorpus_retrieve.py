"""Build the NFCorpus dense training and validation rankings."""
import json
import argparse
from pathlib import Path
import time
import numpy as np
from tapestry_m1.core.io import atomic_json, fingerprint
from tapestry_m1.runtime.assets import verify_assets, verify_model
from tapestry_m1.runtime.devices import gpu_runtime
from tapestry_m1.retrieval.qwen3 import Qwen3Encoder
from tapestry_m1.retrieval.encoding import encode_cached
from tapestry_m1.retrieval.search import exact_topk
from tapestry_m1.workflows.nfcorpus_pilot import source, qrels, write_rows


def coverage(rankings, judgments, ks=(10, 50, 100)):
    values=[]
    assert len(rankings)==len(judgments)
    assert {r['query_id'] for r in rankings}==set(judgments)
    for row in rankings:
        positives={d for d,g in judgments[row['query_id']].items() if g>0}
        assert positives
        ids=[c['doc_id'] for c in row['candidates']]
        assert len(ids)==len(set(ids))
        item={'query_id':row['query_id'],'positives':len(positives)}
        for k in ks:
            hits=len(set(ids[:k]) & positives)
            item[f'hit@{k}']=int(hits>0)
            item[f'recall@{k}']=hits/len(positives)
        values.append(item)
    return {'queries':len(values),'means':{key:sum(v[key] for v in values)/len(values) for k in ks for key in (f'hit@{k}',f'recall@{k}')},'per_query':values}


def main():
    started=time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'output', 'config', 'data-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    raw, out = args.raw, args.output
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    config['encoder'] = json.loads((args.config.parent / config['encoder']).read_text())
    model, model_identity = verify_model(config['encoder'], args.data_root)
    source_files = {name: fingerprint(raw/name) for name in
                    ('corpus.jsonl','queries.jsonl','qrels/train.tsv','qrels/dev.tsv','qrels/test.tsv')}
    identity = {'config': config, 'source': source_files, 'model': model_identity, 'test_evaluated': False}
    if (out/'inputs.json').exists():
        assert json.loads((out/'inputs.json').read_text())==identity
    atomic_json(out/'inputs.json',identity)
    hardware=gpu_runtime()
    docs,queries=source(raw)
    judgments={s:qrels(raw/'qrels'/f'{s}.tsv',queries,docs) for s in ('train','dev')}
    encoder=Qwen3Encoder(model,config)
    rows=[{'doc_id':d,'text':docs[d]} for d in sorted(docs)]
    dv,da,_=encode_cached(encoder,rows,'text',False,out/'vectors/documents',identity,config)
    atomic_json(out/'document-lengths.json',da)
    print('Corpus encoding complete',flush=True)
    summary={}
    for split in ('dev','train'):
        qr=[{'query_id':q,'text':queries[q]} for q in sorted(judgments[split])]
        print(f'Encoding {split}: {len(qr)} queries',flush=True)
        qv,qa,_=encode_cached(encoder,qr,'text',True,out/'vectors'/split,identity,config)
        ranks=exact_topk([r['doc_id'] for r in rows],dv,qv,100)
        result=[{'query_id':q['query_id'],'candidates':r} for q,r in zip(qr,ranks)]
        write_rows(out/f'{split}-ranking.jsonl',result)
        metrics=coverage(result,judgments[split]);atomic_json(out/f'{split}-metrics.json',metrics)
        atomic_json(out/f'{split}-lengths.json',qa)
        summary[split]={'queries':metrics['queries'],'means':metrics['means']}
        print(json.dumps({split:summary[split]}),flush=True)
    encoder.close()
    atomic_json(out/'completed.json',{'summary':summary,'elapsed_seconds':time.time()-started,'test_evaluated':False,'files':{p.name:fingerprint(p) for p in out.glob('*.json*') if p.name!='completed.json'}})
    print('COMPLETE',flush=True)

if __name__=='__main__':
    main()
