"""Official CaseHOLD all CSV contract, without answer-derived prompts.

Mapping follows casehold/casehold's Apache-2.0 dataset loader (v1.1.0).
Dataset: Zheng, Guha et al., ICAIL 2021, "When Does Pretraining Help?"
"""
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from tapestry_m1.core.io import ContractError, fingerprint, text_key

REVISION = 'acc4532a67e0966e9ed7ec4ca543e8532f983b0c'
FILES = {
    'train': ('train.csv', 42509, '515aa9c21911d644caeef4a7b962fe079c476e693a2901a43fc09b9a998fbbd5'),
    'validation': ('val.csv', 5314, '22e696e4e5e68fefe28fcce86cca430a03bcddfddb4d464a79b25c60c4d56653'),
    'test': ('test.csv', 5314, '1467ea7600fef2c287aa171219ab907346e022c36f70477ff5d52b0767956e62'),
}


def parse_row(row, split):
    if len(row) != 13 or row[12] not in ['0', '1', '2', '3', '4']:
        raise ContractError('CaseHOLD requires 13 columns and a zero-based label')
    if not all(t.strip() for t in row[1:7]) or not row[0].isdigit():
        raise ContractError('empty CaseHOLD text or invalid source identity')
    qid = f'{split}:{row[0]}'
    return {'query_id': qid, 'source_id': row[0], 'split': split,
            'query_text': row[1], 'gold': int(row[12]),
            'candidates': [{'doc_id': f'{qid}:{i}', 'text': row[i+2]} for i in range(5)]}


def load_split(root, split):
    name, count, sha = FILES[split]
    path = Path(root)/name
    if fingerprint(path) != sha:
        raise ContractError('CaseHOLD raw checksum mismatch: '+split)
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle)
        if next(reader) != ['Unnamed: 0']+[str(i) for i in range(12)]:
            raise ContractError('unexpected CaseHOLD CSV header')
        groups = [parse_row(row, split) for row in reader]
    if len(groups) != count or len({g['query_id'] for g in groups}) != count:
        raise ContractError('unexpected split count or repeated source ID')
    return groups


def audit(root, comparison_limit=250000):
    """Exact normalized tuples + bounded bottom-hash blocking/Jaccard screen.

    Approximate near-duplicate screen is explicitly not an exhaustive proof.
    No official rows are removed, including any detected overlaps.
    """
    splits = {s: load_split(root, s) for s in FILES}
    exact, blocks, seen = defaultdict(list), defaultdict(list), []
    overlaps, near, comparisons, capped = Counter(), Counter(), 0, False
    for split, groups in splits.items():
        for group in groups:
            parts = [text_key(group['query_text'])]+[text_key(c['text']) for c in group['candidates']]
            key = hashlib.sha256('\0'.join(parts).encode()).hexdigest()
            for other in exact[key]:
                overlaps['|'.join(sorted([split, other]))] += 1
            exact[key].append(split)
            words = ' '.join(parts).split()
            shingles = {hashlib.blake2b(' '.join(words[i:i+3]).encode(), digest_size=8).digest()
                        for i in range(len(words)-2)}
            # Eight bottom hashes generate candidates; bounded buckets prevent quadratic work.
            keys = sorted(shingles)[:8]
            candidates = set(j for k in keys for j in blocks[k])
            cross = [j for j in sorted(candidates) if seen[j][0] != split and seen[j][1] != key]
            if len(cross) > 16:
                capped = True
            for j in cross[:16]:
                other_split, other_key, other_shingles = seen[j]
                if other_split == split or other_key == key:
                    continue
                if comparisons >= comparison_limit:
                    capped = True
                    break
                comparisons += 1
                similarity = len(shingles & other_shingles)/len(shingles | other_shingles)
                if similarity >= .90:
                    near['|'.join(sorted([split, other_split]))] += 1
            index = len(seen)
            seen.append((split, key, shingles))
            for k in keys:
                if len(blocks[k]) < 32:
                    blocks[k].append(index)
                else:
                    capped = True
    return {'dataset': 'casehold/casehold', 'config': 'all', 'revision': REVISION,
            'mapping': {'query': 1, 'choices': [2,3,4,5,6], 'label': 12, 'label_base': 0},
            'files': {s: {'file': v[0], 'count': v[1], 'sha256': v[2]} for s,v in FILES.items()},
            'label_counts': {s: dict(Counter(g['gold'] for g in gs)) for s,gs in splits.items()},
            'exact_normalized_tuple_pairs': dict(overlaps),
            'near_duplicate_screen': {'method': '8 bottom 64-bit trigram hashes; bucket cap32; max16 cross-split comparisons per query; Jaccard>=0.90',
                'comparisons': comparisons, 'comparison_limit': comparison_limit, 'bounded_or_capped': capped,
                'exhaustive': False, 'cross_split_pairs': dict(near)}, 'official_splits_unchanged': True}
