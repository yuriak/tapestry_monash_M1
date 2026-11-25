"""Keep AU intact; assign connected Indian query/parent components to two sites."""
from collections import Counter, defaultdict
import json
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .partition import digest, read, write


def assignments(data, rules):
    parents = {}

    def find(key):
        parents.setdefault(key,key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    queries = {}
    for split in ('train','dev','test'):
        items = ((r['qh'],r) for r in data['train']) if split=='train' else data[split].items()
        for q,row in items:
            queries[q] = (split,row)
            for doc in data['known_gold'][q]:
                for group in data['metadata'][doc]['groups']:
                    a,b = sorted([find('q:'+q),find('g:'+group)])
                    parents[b] = a
    components = defaultdict(list)
    for q in queries:
        components[find('q:'+q)].append(q)
    result = {}
    public = set(rules['indian_public_domains'])
    assigned = defaultdict(lambda: Counter())
    ordered = sorted(components.items(),key=lambda item:(
        -sum(queries[q][0]=='train' for q in item[1]), digest(item[0])))
    for group,qs in ordered:
        countries = {queries[q][1]['country'] for q in qs}
        if len(countries)!=1:
            raise ValueError('A connected parent group spans countries')
        if countries=={'au'}:
            site = 'monash'
        elif countries=={'in'}:
            priority = next(s for s in ('train','dev','test') if any(queries[q][0]==s for q in qs))
            domains = [queries[q][1]['domain'] for q in qs if queries[q][0]==priority]
            votes = sum(d in public for d in domains)
            probability = .7 if votes*2>len(domains) else .3 if votes*2<len(domains) else .5
            draw = int(digest('42:judicialmind:'+group)[:16],16)/2**64
            if priority=='train':
                allocated = assigned[probability]
                target = probability*(sum(allocated.values())+len(domains))
                go_first = abs(allocated['bharatgen']+len(domains)-target) < abs(allocated['bharatgen']-target)
                site = 'bharatgen' if go_first else 'iitd'
                allocated[site] += len(domains)
            else:
                site = 'bharatgen' if draw<probability else 'iitd'
        else:
            raise ValueError('Unknown country')
        for q in qs:
            split,row = queries[q]
            result[q] = dict(dataset='judicialmind',version='v2',split=split,query_id=q,
                             group_id=group,site=site,country=row['country'],category=row['domain'])
    return result


def local_negatives(data, rows):
    """Preserve already-local triplets; re-mine missing slots from the local stratum."""
    strata = defaultdict(list)
    for r in rows:
        strata[r['jurisdiction'],r['domain']].append(r)
    negatives = {}
    fallback = []
    all_docs = sorted({r['dh'] for r in rows})
    for subset in strata.values():
        ids = sorted({r['dh'] for r in subset})
        allowed = set(ids)
        pending = []
        for r in subset:
            kept = [d for d in data['hard_negatives'][r['id']] if d in allowed]
            negatives[r['id']] = kept
            if len(kept)<3:
                pending.append(r)
        if not pending:
            continue
        vec = TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=100000,dtype=np.float32)
        matrix = vec.fit_transform(data['documents'][d] for d in ids)
        for i in range(0,len(pending),32):
            part = pending[i:i+32]
            scores = (vec.transform([r['query'] for r in part]) @ matrix.T).toarray()
            for r,values in zip(part,scores):
                kept = negatives[r['id']]
                forbidden = set(data['known_gold'][r['qh']]) | set(kept)
                for j in np.argsort(-values,kind='stable'):
                    d = ids[j]
                    if d not in forbidden and r['docgroup'] not in data['metadata'][d]['groups']:
                        kept.append(d)
                        if len(kept)==3:
                            break
                if len(kept)!=3:
                    fallback.append(r['id'])
                    # Small strata can have fewer than three eligible local documents.
                    start = int(digest(r['id'])[:16],16) % len(all_docs)
                    for j in range(len(all_docs)):
                        d = all_docs[(start+j)%len(all_docs)]
                        if d not in forbidden and d not in kept and r['docgroup'] not in data['metadata'][d]['groups']:
                            kept.append(d)
                            if len(kept)==3:
                                break
                    if len(kept)!=3:
                        raise ValueError('Insufficient local negative corpus')
    return negatives, fallback


def materialize(root, stage, rules):
    data = read(root/'prepared/data.json')
    frozen = read(root/'retrieval/frozen.json')
    with (root/'prepared/tokens.pkl').open('rb') as handle:
        tokens = pickle.load(handle)
    assignment = assignments(data,rules)
    summary = {}
    for site in rules['sites']:
        out = stage/site
        out.mkdir(parents=True)
        index = [r for r in assignment.values() if r['site']==site]
        by_split = {s:[r['query_id'] for r in index if r['split']==s] for s in ('train','dev','test')}
        queries = {r['query_id'] for r in index}
        local = {'train':[r for r in data['train'] if r['qh'] in queries],'tax':data['tax'],
                 'known_gold':{q:data['known_gold'][q] for q in sorted(queries)}}
        local['hard_negatives'],fallback = local_negatives(data,local['train'])
        rankings = {'tax':frozen['tax']}
        docs = {d for ds in local['known_gold'].values() for d in ds}
        for split in ['dev','test']:
            qs = by_split[split]
            local[split] = {q:data[split][q] for q in qs}
            rankings[split] = {field:{q:values[q] for q in qs} for field,values in frozen[split].items()}
            docs.update(d for ds in rankings[split]['candidates'].values() for d in ds)
        for field in ['documents','metadata']:
            local[field] = {d:data[field][d] for d in sorted(docs)}
        write(out/'data/data.json',local)
        write(out/'retrieval/frozen.json',rankings)
        qkeys = queries | data['tax']['queries'].keys()
        dkeys = docs | data['tax']['documents'].keys()
        cache = {k:{i:tokens[k][i] for i in sorted(ids)} for k,ids in [('queries',qkeys),('documents',dkeys)]}
        with (out/'data/tokens.pkl').open('wb') as f:
            pickle.dump(cache,f,protocol=5)
        with (out/'index.jsonl').open('w') as f:
            for row in index:
                f.write(json.dumps(row,sort_keys=True)+'\n')
        summary[site] = {s:{'queries':len(qs),'categories':dict(Counter(assignment[q]['category'] for q in qs))}
                         for s,qs in by_split.items()}
        summary[site]['train'].update(relations=len(local['train']), broad_fallback_rows=len(fallback))
    return summary


def verify_site(root, out, by_split, reference, model, parent_sites):
    from ..recipes.judicialmind.model import Tokens
    from ..recipes.judicialmind.screen import mixed_negatives
    from .verify import require, fingerprint
    local = read(out/'data/data.json')
    original = read(root/'prepared/data.json')
    frozen = read(out/'retrieval/frozen.json')
    parent = read(root/'retrieval/frozen.json')
    require(local['train']==[r for r in original['train'] if r['qh'] in by_split['train']],
            'Missing or altered training relation')
    require(local['tax']==original['tax'] and frozen['tax']==parent['tax'],'Changed external evaluation')
    require(fingerprint(frozen)==reference['judicialmind'],'Changed frozen JudicialMind ranking')
    country = 'au' if out.name=='monash' else 'in'
    require(all(row['country']==country for row in local['train']) and
            all(row['country']==country for s in ('dev','test') for row in local[s].values()),
            'Country routing mismatch')
    require(set(local['known_gold']) == set.union(*by_split.values()),'Unexpected supervised query')
    for q,ds in local['known_gold'].items():
        require(ds==original['known_gold'][q],'Changed known positives')
        for d in ds:
            for group in original['metadata'][d]['groups']:
                require(parent_sites.setdefault(group,out.name)==out.name,'Parent spans sites')
    docs = {r['dh'] for r in local['train']}
    for row in local['train']:
        pool = local['hard_negatives'][row['id']]
        require(len(set(pool))==3 and set(pool)<=docs and
                not set(pool).intersection(original['known_gold'][row['qh']]) and
                all(row['docgroup'] not in local['metadata'][d]['groups'] for d in pool),
                'Invalid local negatives')
    tokens = Tokens(model,out/'data/tokens.pkl')
    row = local['train'][0]
    pairs = [tokens.pair(row['qh'],d,2048) for d in [row['dh'],*local['hard_negatives'][row['id']]]]
    require(tokens.tensors(pairs,'cpu')[0].shape[0]==4,'Invalid JudicialMind batch')
    mixed,_ = mixed_negatives(row,local['train'][:64],local,42)
    require(len(set(mixed))==3 and set(mixed)<=docs,'Invalid local in-batch sample')
    for s in ('dev','test'):
        require(set(local[s])==by_split[s] and all(local[s][q]==original[s][q] for q in local[s]),
                'Changed held-out query')
        for key,values in frozen[s].items():
            require(values=={q:parent[s][key][q] for q in by_split[s]},'Changed candidates or qrels')
