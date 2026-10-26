"""One-factor same-split hard-negative replacement, with the random ID fixed."""
import copy
from tapestry_m1.core.io import ContractError, text_key
from tapestry_m1.data.au_qa_training import near_pairs
from tapestry_m1.contracts.reranker_training_v2 import validate_groups


def near_index(passages):
    result = {p['doc_id']: set() for p in passages}
    for a, b, _ in near_pairs([p['text'] for p in passages]):
        x, y = passages[a]['doc_id'], passages[b]['doc_id']
        result[x].add(y); result[y].add(x)
    return result


def select_hard(group, passages, ranking, scores, near):
    """Greedy descending teacher score, ascending ID; never expand the pool."""
    pool = {p['doc_id']: p for p in passages}
    positive, fixed_random = group['candidates'][:2]
    known = set(group['known_relevant_ids'])
    source = set(group['query_source_doc_ids'])
    excluded = known | {fixed_random['doc_id']}
    for d in known:
        excluded.update(near.get(d, ()))
    known_texts = {text_key(pool[d]['text']) for d in known}
    chosen = [fixed_random['doc_id']]
    eligible = []
    for row in sorted(ranking, key=lambda r: (-scores[r['doc_id']], r['doc_id'])):
        d = row['doc_id']; p = pool[d]
        if d in excluded or p['split'] != group['split'] or source.intersection(p['source_doc_ids']):
            continue
        if text_key(p['text']) in known_texts:
            continue
        if any(d == x or d in near.get(x, ()) or text_key(p['text']) == text_key(pool[x]['text'])
               or set(p['source_doc_ids']).intersection(pool[x]['source_doc_ids']) for x in chosen):
            continue
        eligible.append(row); chosen.append(d)
        if len(eligible) == 2:
            break
    if len(eligible) != 2:
        raise ContractError('Top-100 cannot supply two compatible teacher-hard negatives with fixed random')
    result = copy.deepcopy(group)
    result['candidates'] = copy.deepcopy(group['candidates'][:2])
    for row in eligible:
        p = pool[row['doc_id']]
        result['candidates'].append({'doc_id':p['doc_id'],'source_doc_id':p['source_doc_ids'][0],
            'source_doc_ids':p['source_doc_ids'],'text':p['text'],'source_hash':p['source_hash'],
            'split':p['split'],'label':0,'negative_kind':'hard','ambiguous':True,'supervision':'assumed',
            'retrieval_rank':row['rank'],'similarity':row['score'],
            'teacher_logit':scores[p['doc_id']]})
    return result


def assert_comparable(original, replacement):
    if len(original) != len(replacement):
        raise ContractError('controlled group count changed')
    for old, new in zip(original, replacement):
        if {k:v for k,v in old.items() if k != 'candidates'} != {k:v for k,v in new.items() if k != 'candidates'}:
            raise ContractError('query, split, order or source identity changed')
        if old['candidates'][:2] != new['candidates'][:2]:
            raise ContractError('positive or original random changed')
        if len(new['candidates']) != 4 or any(c['negative_kind'] != 'hard' for c in new['candidates'][2:]):
            raise ContractError('only two hard candidates may change')
    validate_groups(replacement)
    return {'passed':True,'counts':{s:sum(g['split']==s for g in replacement) for s in ['train','dev']},
        'same_query_order_positive_random_splits':True,
        'hard_sets_changed':{s:sum({c['doc_id'] for c in x['candidates'][2:]} != {c['doc_id'] for c in y['candidates'][2:]}
            for x,y in zip(original,replacement) if x['split']==s) for s in ['train','dev']}}
