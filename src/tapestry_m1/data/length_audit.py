"""Uniform document sample plus character-longest tail; no relevance labels."""
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import random
import shutil
import time
from tapestry_m1.core.io import ContractError,atomic_json,fingerprint


def length_audit(corpus_root,tokenizer,output,tokenizer_revision,sample_size=512,seed=20261018):
    import numpy as np
    started=time.monotonic();root=Path(corpus_root);rng=random.Random(seed);sample=[];longest=[];sources=Counter();count=0;sha=hashlib.sha256()
    manifest=json.loads((root/'manifest.json').read_text())
    with (root/'documents.jsonl').open('rb') as handle:
        for line in handle:
            sha.update(line)
            if not line.strip():continue
            doc=json.loads(line);count+=1;sources[doc.get('metadata',{}).get('source','unknown')]+=1
            if len(sample)<sample_size:sample.append(doc)
            else:
                i=rng.randrange(count)
                if i<sample_size:sample[i]=doc
            item=(len(doc['text']),doc['doc_id'],doc)
            if len(longest)<5:heapq.heappush(longest,item)
            elif item[:2]>longest[0][:2]:heapq.heapreplace(longest,item)
    if sha.hexdigest()!=manifest['output_sha256']['documents.jsonl']:raise ContractError('corpus source checksum differs')
    def measure(docs):
        return [{'doc_id':d['doc_id'],'source':d.get('metadata',{}).get('source','unknown'),'tokens':len(tokenizer.encode(d['text'],add_special_tokens=False))} for d in docs]
    lengths=measure(sample);tail=measure([x[2] for x in sorted(longest,reverse=True)]);values=np.asarray([r['tokens'] for r in lengths]);policies=[]
    for window,overlap in [(512,64),(1024,128),(2048,256)]:
        chunks=np.maximum(1,np.ceil(np.maximum(0,values-window)/(window-overlap)).astype(int)+1)
        estimate=float(chunks.mean()*count)
        policies.append({'window':window,'overlap':overlap,'estimated_chunks':estimate,'estimated_fp32_vector_bytes':estimate*2560*4})
    result={'source_sha256':sha.hexdigest(),'source_document_count':count,'source_counts':dict(sources),'tokenizer_revision':tokenizer_revision,'sampling':'fixed-seed uniform reservoir; separate five character-longest documents','sample_size':len(lengths),'seed':seed,'sample':lengths,'character_longest_tail':tail,'sample_token_percentiles':{str(q):float(np.percentile(values,q)) for q in [0,50,90,95,99,100]},'policy_estimates':policies,'disk_free_bytes_observed':shutil.disk_usage(root).free,'wall_seconds':time.monotonic()-started,'limitations':['Token/chunk estimates derive from sample, not an exhaustive token census.','Character-longest is a tail check, not guaranteed token-longest.','GPU-hours require measured encoding throughput; no quality or semantic-overlap conclusion.']}
    atomic_json(output,result);return result
