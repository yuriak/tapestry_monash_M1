"""Deterministic group routing. Only source/query features enter the routing decision."""
import hashlib
import json
import re
from collections import Counter, defaultdict

import numpy as np
import pyarrow.parquet as pq

from tapestry_m1.data.casehold import FILES, load_split


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def group_text(text):
    return digest(' '.join(text.casefold().split()))


def category(text, rules):
    scores = {label: len(re.findall(pattern, text.casefold())) for label, pattern in rules.items()}
    best = max(scores.values())
    winners = [label for label, score in scores.items() if score == best]
    return winners[0] if best and len(winners) == 1 else ('mixed' if best else 'unknown')


def records(dataset, root, rules):
    raw, prepared = root/'raw', root/'prepared'
    rows = []
    texts = defaultdict(list)
    def append(split, q, group, topic, preference, text, positives, count):
        if dataset != "il_pcsr":
            texts[group].append(text)
        rows.append(dict(split=split, query_id=q, group_id=group, category=topic,
                         preference=preference, words=len(text.split()), positives=positives,
                         candidates=count))
    if dataset == 'il_pcsr':
        # A single text identity also binds verbatim duplicate case records.
        policy = rules[dataset]
        for split in ('train', 'dev', 'test'):
            for r in pq.read_table(raw/f'{split}_queries.parquet').to_pylist():
                court = (r['jurisdiction'] or '').strip()
                court = policy['aliases'].get(court, court) or 'unknown'
                pref = 'shared' if court in [*policy['shared'], 'unknown'] else (
                    'A' if court in policy['north'] else 'B')
                if rules['version']=='v2' and pref!='shared':
                    pref = 'delhi' if court in rules['delhi'] else 'west' if court in rules['west'] else 'regional'
                text = '\n'.join(r['text'])
                append(split, str(r['id']), group_text(text), court, pref, text,
                       {k:len(set(r[f'relevant_{field}_ids'])) for k,field in
                        [('lsr','statute'),('pcr','precedent')]}, 200)
    elif dataset == 'evidence_inference':
        docs = read(prepared/'documents.json')
        votes = defaultdict(Counter)
        for split in ('train','validation','test'):
            for r in read(prepared/f'{split}.json'):
                intervention = r['query'].split('\nComparator:',1)[0]
                votes[r['docid']][category(intervention,rules[dataset])] += 1
                append(split, r['id'], r['docid'], '', '', r['query'],
                       len(r['positive']),len(docs[r['docid']]))
        for r in rows:
            v = votes[r['group_id']]
            a,b = v['drug'],v['procedure_behavior']
            r['category'] = ('drug' if a>b else 'procedure_behavior' if b>a else
                             'mixed' if a or v['mixed'] else 'unknown')
            r['preference'] = {'drug':'A','procedure_behavior':'B'}.get(r['category'],'shared')
    elif dataset == 'casehold':
        for split in FILES:
            for r in load_split(raw,split):
                text = r['query_text']
                topic = category(text,rules[dataset])
                append(split,r['query_id'],group_text(text),topic,
                       {'criminal':'A','civil':'B'}.get(topic,'shared'),text,1,5)
    else:
        data = read(prepared/'data.json')
        aliases = defaultdict(list)
        canonical = {group_text(t):q for q,t in data['queries'].items()}
        parquet = prepared/'raw/question-answer-passages/train-00000-of-00001.parquet'
        for original in pq.read_table(parquet,columns=['id','question']).to_pylist():
            q = 'q'+str(original['id'])
            owner = canonical[group_text(original['question'])]
            if q != owner:
                aliases[owner].append(q)
        for split, ids in data['splits'].items():
            for q in ids:
                text = data['queries'][q]
                topic = category(text,rules[dataset])
                append(split,q,group_text(text),topic,{'molecular':'A','clinical':'B'}.get(
                    topic,'shared'),text,len(data['gold'][q]),1000)
                rows[-1]['source_aliases'] = sorted(aliases[q])
    if dataset != 'il_pcsr':
        infer_topics(rows, texts, list(rules[dataset]))
    return rows


def infer_topics(rows, texts, labels):
    """Learn from unambiguous train groups, then label only confident unknowns."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    train = {r['group_id']:r['category'] for r in rows
             if r['split']=='train' and r['category'] in labels}
    groups = sorted(train)
    documents = {g:' '.join(sorted(set(parts))) for g,parts in texts.items()}
    vectorizer = TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True)
    matrix = vectorizer.fit_transform(documents[g] for g in groups)
    model = LogisticRegression(class_weight='balanced',solver='liblinear',random_state=42)
    model.fit(matrix,[train[g] for g in groups])
    unknown = sorted({r['group_id'] for r in rows if r['category']=='unknown'})
    predicted = {}
    if unknown:
        probabilities = model.predict_proba(vectorizer.transform(documents[g] for g in unknown))
        for g,prob in zip(unknown, probabilities):
            if max(prob)>=0.65:
                predicted[g] = model.classes_[prob.argmax()]
    for r in rows:
        r['method'] = 'lexical'
        if r['category']=='unknown' and r['group_id'] in predicted:
            topic = predicted[r['group_id']]
            r.update(category=topic,preference='A' if topic==labels[0] else 'B',
                     method='train-text-classifier')


def assign(dataset, rows, rules):
    # A source group has one draw even if present in multiple original splits.
    by_group = defaultdict(set)
    for r in rows:
        by_group[r['group_id']].add((r['split'],r['preference']))
    for r in rows:
        group = r['group_id']
        observations = by_group[group]
        for split in ('train','dev','validation','test'):
            prefs = {value for name,value in observations if name==split}
            if prefs:
                break
        pref = next(iter(prefs)) if len(prefs)==1 else 'shared'
        key = f"{rules['seed']}:{dataset}:{group}"
        draw = int(digest(key)[:16],16) / 2**64
        if rules['version']=='v2':
            first,second,_ = rules.get(dataset+'_weights',{}).get(pref,rules['weights'][pref])
            site = rules['sites'][0 if draw<first else 1 if draw<first+second else 2]
        else:
            probability = {'A':rules['preferred_probability'],'B':1-rules['preferred_probability'],
                           'shared':0.5}[pref]
            site = 'A' if draw<probability else 'B'
        r.update(site=site, dataset=dataset,
                 version=rules['version'], metadata_status=('unclassified' if r['category']
                 in ('unknown','mixed') else 'classified'), reference=r['query_id'])
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['site'],r['split']].append(r)
    summary = {}
    for (site,split), subset in sorted(groups.items()):
        pos = Counter()
        for r in subset:
            pos.update(r['positives'] if isinstance(r['positives'],dict) else {'native':r['positives']})
        summary.setdefault(site,{})[split] = dict(queries=len(subset),
            groups=len({r['group_id'] for r in subset}), positive_relations=dict(pos),
            categories=dict(Counter(r['category'] for r in subset)),
            preference=dict(Counter(r['preference'] for r in subset)),
            words_p50_p90_max=np.percentile([r['words'] for r in subset],[50,90,100]).tolist(),
            candidate_rows=sum(r['candidates'] for r in subset))
    return summary
