"""Source-grouped QA excerpt collection for local research; never benchmark qrels."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
from tapestry_m1.core.io import ContractError, atomic_json, dumps, fingerprint, records, text_key
from tapestry_m1.contracts.reranker_training import text_hash
from tapestry_m1.core.artifacts import atomic_bytes

QA_REVISION = 'a2178afc2cff8af50fe4365c7d6f9cecf91dbe4b'
COLLECTION = 'au-qa-excerpts-v2'


def shingles(text):
    words = re.findall(r'\w+', text_key(text))
    return {' '.join(words[i:i+3]) for i in range(max(1, len(words)-2))}


def near_pairs(texts, threshold=.85):
    """Deterministic SimHash blocking + shingle Jaccard; not semantic deduplication."""
    buckets, signatures, grams = defaultdict(list), [], []
    for i, text in enumerate(texts):
        parts = shingles(text); grams.append(parts)
        bits = [0]*64
        for part in parts:
            h = int.from_bytes(hashlib.blake2b(part.encode(), digest_size=8).digest(), 'big')
            for b in range(64): bits[b] += 1 if h & (1<<b) else -1
        signature = sum((1<<b) for b in range(64) if bits[b] >= 0); signatures.append(signature)
        candidates = set()
        for band in range(4): candidates.update(buckets[(band,(signature>>(band*16))&65535)])
        for j in sorted(candidates):
            if (signature ^ signatures[j]).bit_count() <= 3:
                similarity = len(parts & grams[j])/max(1,len(parts | grams[j]))
                if similarity >= threshold: yield j, i, similarity
        for band in range(4): buckets[(band,(signature>>(band*16))&65535)].append(i)


def write_rows(path, rows):
    atomic_bytes(path, ''.join(dumps(row)+'\n' for row in rows).encode())


def construct(pairs, benchmark_queries, benchmark_texts, seed):
    """Group source identities, exact duplicates and detected near duplicates before split."""
    excluded, clean = Counter(), []
    test_queries = {text_key(t) for t in benchmark_queries}
    test_texts = {text_key(t) for t in benchmark_texts}
    # Exact containment uses a first-shingle inverted filter, avoiding a full corpus index.
    anchors = defaultdict(set)
    normalized_tests = sorted(test_texts)
    for i,t in enumerate(normalized_tests):
        for gram in shingles(t): anchors[gram].add(i)
    for row in pairs:
        if not row.get('source_doc_id') or not row.get('query_text','').strip() or not row.get('positive_text','').strip():
            excluded['missing_identity_or_text'] += 1; continue
        q, p = text_key(row['query_text']), text_key(row['positive_text'])
        if q in test_queries: excluded['test_query_exact'] += 1; continue
        if p in test_texts: excluded['test_passage_exact'] += 1; continue
        candidates = set()
        for gram in list(sorted(shingles(p)))[:3]: candidates.update(anchors.get(gram,()))
        if len(p)>=80 and any(p in normalized_tests[i] for i in candidates):
            excluded['positive_contained_in_test'] += 1; continue
        clean.append(row)
    # Approximate near-test passage/query checks, with limitations recorded in the manifest.
    bad = set()
    for a,b,_ in near_pairs(normalized_tests+[r['positive_text'] for r in clean]):
        if a<len(normalized_tests)<=b: bad.add(b-len(normalized_tests))
    testq_list=sorted(test_queries)
    for a,b,_ in near_pairs(testq_list+[r['query_text'] for r in clean],.9):
        if a<len(testq_list)<=b:bad.add(b-len(testq_list))
    excluded['detected_test_near_duplicate'] += len(bad)
    clean=[r for i,r in enumerate(clean) if i not in bad]
    parent=list(range(len(clean)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):parent[find(b)]=find(a)
    keys={}
    for i,row in enumerate(clean):
        for key in [('source',str(row['source_doc_id'])),('query',text_key(row['query_text'])),('positive',text_key(row['positive_text']))]:
            if key in keys:union(keys[key],i)
            else:keys[key]=i
    near_count=0
    for field,threshold in [('positive_text',.85),('query_text',.9)]:
        for a,b,_ in near_pairs([r[field] for r in clean],threshold):union(a,b);near_count+=1
    components=defaultdict(list)
    for i in range(len(clean)):components[find(i)].append(i)
    components=sorted(components.values(),key=lambda ids:min(clean[i]['pair_id'] for i in ids))
    random.Random(seed).shuffle(components)
    target=round(.15*len(clean));dev=set();count=0
    for component in components:
        if abs(count+len(component)-target)<abs(count-target):dev.update(component);count+=len(component)
    if not dev or len(dev)==len(clean):raise ContractError('cannot construct nonempty source-grouped train/dev')
    documents={};queries=[]
    for i,row in enumerate(clean):
        split='dev' if i in dev else 'train';doc_id=COLLECTION+'::'+text_hash(row['positive_text'])
        doc=documents.setdefault(doc_id,{'doc_id':doc_id,'collection_id':COLLECTION,'text':row['positive_text'],
            'source_hash':text_hash(row['positive_text']),'source_doc_ids':[],'split':split,'source_rows':[]})
        if doc['split']!=split:raise ContractError('passage crosses source-grouped split')
        doc['source_doc_ids']=sorted(set(doc['source_doc_ids'])|{str(row['source_doc_id'])})
        doc['source_rows'].append(row['provenance']['source_row'])
        queries.append({'query_id':COLLECTION+'::query-'+str(row['provenance']['source_row']),
            'source_query_id':text_hash(text_key(row['query_text'])),'query_text':row['query_text'],'source_hash':text_hash(row['query_text']),
            'source_doc_id':str(row['source_doc_id']),'positive_id':doc_id,'split':split,'collection_id':COLLECTION,
            'source_row':row['provenance']['source_row'],'source_metadata':row['source_metadata'],'provenance':row['provenance']})
    known=defaultdict(set)
    for row in queries:known[row['source_query_id']].add(row['positive_id'])
    for row in queries:row['known_relevant_ids']=sorted(known[row['source_query_id']])
    stats={'initial_pairs':len(pairs),'retained_pairs':len(queries),'exclusions':dict(excluded),
           'source_versions':len({r['source_doc_id'] for r in clean}),'unique_queries':len(known),
           'passages':len(documents),'duplicate_components':len(components),'near_duplicate_links':near_count,
           'splits':dict(Counter(q['split'] for q in queries)),
           'near_duplicate_policy':'normalized exact source/query/text components plus SimHash64 <=3 and word-trigram Jaccard >=.85 passages / .90 queries; no semantic deduplication guarantee',
           'containment_policy':'normalized positive in held-out text (>=80 chars), first-three-shingle blocking; not all fragment or paraphrase matches'}
    return queries,sorted(documents.values(),key=lambda d:d['doc_id']),stats


def prepare(data_root, output, benchmark_roots, seed=20261026):
    data_root,output=Path(data_root),Path(output)
    source=data_root/'normalized/au-open-australian-legal-qa'/QA_REVISION
    manifest=json.loads((source/'manifest.json').read_text())
    if manifest['source']['revision']!=QA_REVISION or manifest['source']['repo_id']!='isaacus/open-australian-legal-qa':raise ContractError('QA source identity mismatch')
    raw=data_root/'raw/au-open-australian-legal-qa'/QA_REVISION/'qa.jsonl'
    raw_sha=fingerprint(raw)
    if raw_sha!=manifest['input_sha256']['qa.jsonl'] or raw_sha!=manifest['source']['file_sha256']['qa.jsonl']:raise ContractError('pinned QA raw checksum mismatch')
    path=source/'training_pairs.jsonl';sha=fingerprint(path)
    if sha!=manifest['output_sha256']['training_pairs.jsonl']:raise ContractError('normalized QA checksum mismatch')
    spec={'source_revision':QA_REVISION,'raw_sha256':raw_sha,'normalized_sha256':sha,'seed':seed,
          'benchmarks':[{name:fingerprint(Path(root)/name) for name in ('documents.jsonl','queries.jsonl','qrels.jsonl')} for root in benchmark_roots]}
    if (output/'manifest.json').exists():
        old=json.loads((output/'manifest.json').read_text())
        if old['inputs']!=spec:raise ContractError('prepared inputs differ; use a new root')
        for name,h in old['files'].items():
            if fingerprint(output/name)!=h:raise ContractError('prepared artifact checksum mismatch')
        return old
    pairs=list(records(path))
    if len(pairs)!=2124:raise ContractError('expected the full pinned 2124-row source')
    queries=[q['query_text'] for root in benchmark_roots for q in records(Path(root)/'queries.jsonl')]
    texts=[d['text'] for root in benchmark_roots for d in records(Path(root)/'documents.jsonl')]
    q,docs,stats=construct(pairs,queries,texts,seed)
    output.mkdir(parents=True,exist_ok=True)
    write_rows(output/'queries.jsonl',q);write_rows(output/'passages.jsonl',docs)
    result={'schema_version':'m1.auqa.collection.v2','inputs':spec,'counts':stats,
        'files':{name:fingerprint(output/name) for name in ('queries.jsonl','passages.jsonl')},
        'terms':'Local private research only; source-specific terms/attribution retained; no redistribution clearance inferred.',
        'source_terms_url':manifest['source']['terms_url'],'positive_identity':'source-native excerpt, not full-corpus qrel',
        'unmatched_full_corpus_versions_in_original_source':114}
    atomic_json(output/'manifest.json',result);return result
