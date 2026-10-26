"""Portable query-group contract and framework-independent numerical references."""
import hashlib
import math
import random
import re
from tapestry_m1.core.io import ContractError, text_key

SCHEMA = 'm1.reranker.training.v1'


def text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def validate_groups(groups, *, synthetic_only=False):
    queries, texts, documents = {}, {}, {}
    for group in groups:
        if group.get('schema_version') != SCHEMA:
            raise ContractError('unsupported training group schema')
        if group.get('split') not in ('train', 'dev'):
            raise ContractError('training input must be train/dev, never benchmark test')
        if synthetic_only and group.get('synthetic') is not True:
            raise ContractError('synthetic-only gate rejects real input')
        for name in ('collection_id', 'query_id', 'source_query_id', 'query_text', 'provenance'):
            if not isinstance(group.get(name), str) or not group[name].strip():
                raise ContractError('missing group ' + name)
        if type(group.get('seed')) is not int:
            raise ContractError('sampling seed required')
        if group.get('source_hash') != text_hash(group['query_text']):
            raise ContractError('query text hash mismatch')
        key = (group['collection_id'], group['query_id'])
        if key in queries:
            raise ContractError('duplicate query identity')
        queries[key] = group['split']
        normalized = text_key(group['query_text'])
        if normalized in texts and texts[normalized] != group['split']:
            raise ContractError('query text leaks across splits')
        texts[normalized] = group['split']
        candidates = group.get('candidates', [])
        if len(candidates) != 4:
            raise ContractError('one complete four-candidate group required')
        if [c.get('label') for c in candidates] != [1, 0, 0, 0] or any(type(c.get('label')) is not int for c in candidates):
            raise ContractError('candidate order must be positive/random/hard/hard')
        if [c.get('negative_kind') for c in candidates] != ['positive', 'random', 'hard', 'hard']:
            raise ContractError('negative kinds must be positive/random/hard/hard')
        ids = set()
        for candidate in candidates:
            for name in ('doc_id', 'source_doc_id', 'text', 'supervision'):
                if not isinstance(candidate.get(name), str) or not candidate[name].strip():
                    raise ContractError('missing candidate ' + name)
            if candidate['doc_id'] in ids:
                raise ContractError('duplicate candidate')
            ids.add(candidate['doc_id'])
            if candidate.get('source_hash') != text_hash(candidate['text']):
                raise ContractError('candidate text hash mismatch')
            if candidate.get('ambiguous') is not False:
                raise ContractError('unjudged/ambiguous candidates are not certified training negatives')
            if candidate['label'] == 0 and candidate['doc_id'] in group.get('known_relevant_ids', []):
                raise ContractError('known relevant candidate sampled as negative')
            for key in [('source_id', candidate['source_doc_id']), ('text', text_key(candidate['text']))]:
                if key in documents and documents[key] != group['split']:
                    raise ContractError('document/source leaks across splits')
                documents[key] = group['split']
    if not groups or not any(g['split'] == 'train' for g in groups):
        raise ContractError('nonempty train groups required')
    return groups


def select_negatives(pool, ranking, known_relevant, seed):
    """Select only explicitly judged train negatives; unresolved pools fail closed."""
    if ranking.get('split') != 'train' or not re.fullmatch('[0-9a-f]{64}', ranking.get('source_sha256', '')):
        raise ContractError('frozen training-query ranking identity required')
    ids = [c['doc_id'] for c in pool]
    if len(ids) != len(set(ids)):
        raise ContractError('duplicate candidate pool IDs')
    permitted = {c['doc_id']: c for c in pool if c.get('split') == 'train' and c.get('relevance') == 0
                 and c.get('judged') is True and c['doc_id'] not in known_relevant}
    hard = list(dict.fromkeys(d for d in ranking['doc_ids'] if d in permitted))[:2]
    remaining = sorted(set(permitted) - set(hard))
    if len(hard) != 2 or not remaining:
        raise ContractError('insufficient certified train negatives; do not replace with test/unjudged data')
    return [random.Random(seed).choice(remaining), *hard]


def reference_loss(scores, objective='L0', margin=1., weight=.5):
    if objective not in ('L0', 'L1') or not scores or any(len(g) != 4 for g in scores):
        raise ContractError('L0/L1 complete query groups required')
    if not all(math.isfinite(v) for row in scores for v in row):
        raise ContractError('finite logit differences required')
    bce, hinge, gradients = [], [], []
    for values in scores:
        labels = [1, 0, 0, 0]
        bce.append(sum(max(s, 0) - s*y + math.log1p(math.exp(-abs(s))) for s, y in zip(values, labels))/4)
        active = [margin - values[0] + s for s in values[1:]]
        hinge.append(sum(max(0., s) for s in active)/3)
        grad = [((1/(1+math.exp(-s))) if s >= 0 else math.exp(s)/(1+math.exp(s))) - y for s, y in zip(values, labels)]
        grad = [v/(4*len(scores)) for v in grad]
        if objective == 'L1':
            grad[0] -= weight*sum(s > 0 for s in active)/(3*len(scores))
            for i, s in enumerate(active, 1):
                grad[i] += weight*(s > 0)/(3*len(scores))
        gradients.append(grad)
    bce = sum(bce)/len(scores); hinge = sum(hinge)/len(scores)
    return {'bce': bce, 'margin': hinge, 'total': bce + (weight*hinge if objective == 'L1' else 0.),
            'gradients_wrt_score': gradients}
