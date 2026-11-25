"""Check frozen membership, native evidence and the actual training input adapters."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tapestry_m1.data.casehold import FILES, load_split
from .partition import read


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verify(dataset, root, config, model):
    reference = read(config/'reference.json')[dataset]
    target = root/'v1'
    groups = {}
    union = defaultdict(set)
    for site in 'AB':
        out = target/site
        rows = [json.loads(line) for line in (out/'index.jsonl').read_text().splitlines()]
        membership = [[r['split'],r['query_id'],r['group_id']] for r in rows]
        require(fingerprint(sorted(membership))==reference[site]['membership'],'Site membership drift')
        by_split = defaultdict(set)
        for r in rows:
            q,s,g = r['query_id'],r['split'],r['group_id']
            require(r['site']==site and q not in union[s] and groups.setdefault(g,site)==site,
                    'Overlapping group or query')
            union[s].add(q)
            by_split[s].add(q)
        if dataset=='casehold':
            from transformers import AutoTokenizer
            from tapestry_m1.reranker.qwen3 import PREFIX,SUFFIX
            from tapestry_m1.reranker.peft_training import prepare_inputs
            tok = AutoTokenizer.from_pretrained(model,padding_side='left',local_files_only=True)
            recipe = read(config.parent/'training/casehold-epoch.json')
            backend = SimpleNamespace(tokenizer=tok,limit=recipe['max_tokens'],config=recipe,
                prefix=tok.encode(PREFIX,add_special_tokens=False),suffix=tok.encode(SUFFIX,add_special_tokens=False))
            for s in FILES:
                records = load_split(out,s)
                source = [r for r in load_split(root/'raw',s) if r['query_id'] in by_split[s]]
                require(records==source,'Changed CaseHOLD options or labels')
                batch,_ = prepare_inputs(backend,records[:2],device='cpu')
                require(batch.input_ids.shape[0]==10,'Expected two five-option groups')
        elif dataset=='evidence_inference':
            from ..recipes.evidence_inference.data import sample_indices,length_batches
            from ..recipes.evidence_inference.run import Scorer
            scorer = Scorer(model,512)
            docs = read(out/'documents.json')
            original = read(root/'prepared/documents.json')
            require(all(original[q]==v for q,v in docs.items()),'Changed sentence or offset')
            for s,keys in by_split.items():
                records = read(out/f'{s}.json')
                require(records==[r for r in read(root/'prepared'/f'{s}.json') if r['id'] in keys],
                        'Changed EI prompt or evidence')
                row = next(r for r in records if r['positive'])
                doc = docs[row['docid']]
                for ratio in [2,4]:
                    indices = sample_indices(row,len(doc),ratio,42,1)
                    tokens = scorer.tokenized([(row['query'],doc[i]['text']) for i,_ in indices])
                    batches = list(length_batches(list(zip(indices,tokens)),4,2048))
                    require(sum(map(len,batches))==len(indices),'Missing sampled sentence')
        else:
            for task in (['lsr','pcr'] if dataset=='il_pcsr' else ['']):
                check_ranking(root,out,task,by_split,reference[site],model)
    if dataset=='casehold':
        source = {s:{r['query_id'] for r in load_split(root/'raw',s)} for s in FILES}
    elif dataset=='evidence_inference':
        source = {s:{r['id'] for r in read(root/'prepared'/f'{s}.json')} for s in union}
    else:
        native_dir = root/'prepared'/'lsr' if dataset=='il_pcsr' else root/'prepared'
        source = {s:set(qs) for s,qs in read(native_dir/'data.json')['splits'].items()}
    require(dict(union)==source,'Site union differs from the original split IDs')
    counts = {s:len(qs) for s,qs in union.items()}
    require(counts==reference['counts'],'Incomplete split union')
    print(dataset,'verified native inputs for A and B',counts,flush=True)


def check_ranking(root,out,task,by_split,reference,model):
    from ..recipes.bioasq.sampling import sample_negatives
    from ..recipes.il_chunk.sampling import mixed_negatives
    data_dir = out/'data'/task
    native_dir = root/'prepared'/task
    data, original = read(data_dir/'data.json'), read(native_dir/'data.json')
    frozen = read(out/'retrieval'/task/'frozen.json')
    parent = read(root/'retrieval'/task/'frozen.json')
    keys = set.union(*by_split.values())
    require(data['train_pairs']==[p for p in original['train_pairs'] if p[0] in keys],
            'Changed positive relations')
    require(set(data['gold'])==keys and all(data['gold'][q]==original['gold'][q] for q in keys),
            'Changed native qrels')
    require(fingerprint(frozen)==reference[task or 'bioasq'],'Frozen candidates or gold drift')
    for s,qs in by_split.items():
        require(set(data['splits'][s])==qs==set(frozen[s]['candidates']),'Query coverage mismatch')
        require(all(frozen[s]['candidates'][q]==parent[s]['candidates'][q] for q in qs),
                'Candidate prefix changed')
    q,d = data['train_pairs'][0]
    if task:
        from ..recipes.il_chunk.model import Tokens
        original_indices = {q:i for i,q in enumerate(original['query_ids'])}
        selection = np.load(data_dir/'selection.npy',mmap_mode='r')
        source = np.load(native_dir/'selection.npy',mmap_mode='r')
        require(data['document_ids']==original['document_ids'],'Changed IL candidate corpus')
        require(all(np.array_equal(selection[i],source[original_indices[q]])
                    for i,q in enumerate(data['query_ids'])),'Misaligned chunk selection')
        require(all(v==original['negative'][q] for q,v in data['negative'].items()),'Changed negatives')
        tokens = Tokens(model,data_dir)
        negatives = mixed_negatives(data['negative'][q],0)
    else:
        from ..recipes.bioasq.model import Tokens
        require(data['documents']==original['documents'],'Changed BioASQ corpus')
        tokens = Tokens(model,data_dir/'tokens.pkl')
        pool = [c for c in frozen['train']['candidates'][q] if c not in data['gold'][q]]
        negatives = sample_negatives(pool,'basic',0,1,0)
    require(set(tokens.cache['queries'])==keys,'Unexpected query tokens')
    require(not set(negatives).intersection(data['gold'][q]),'False negative')
    limit = read(data_dir/'manifest.json')['max_length']
    batch,lengths = tokens.tensors([tokens.pair(q,c,limit) for c in [d,*negatives]],'cpu')
    require(len(batch)==len(negatives)+1 and max(lengths)<=limit,'Invalid training pair batch')


if __name__=='__main__':
    from huggingface_hub import snapshot_download
    from .acquire import MODEL
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset')
    for name in ['root','config']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    model=os.environ.get('RERANKER_TOKENIZER') or snapshot_download('Qwen/Qwen3-Reranker-0.6B',
             revision=MODEL,allow_patterns=['*.json','*.txt','*.model'])
    verify(a.dataset,a.root,a.config,model)
