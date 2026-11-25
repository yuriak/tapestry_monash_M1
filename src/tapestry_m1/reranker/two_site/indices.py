"""Regenerate the two portable coordinate indexes from frozen native preparations."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .partition import read


def pack_evidence(prepared, target):
    docs = read(prepared/'documents.json')
    offsets = {k:[[r['start'],r['end']] for r in rows] for k,rows in docs.items()}
    target.write_bytes(gzip.compress(json.dumps(offsets,separators=(',',':')).encode(),mtime=0))


def pack_bioasq(prepared, retrieval, target):
    data = read(prepared/'data.json')
    docs, queries = sorted(data['documents']), sorted(data['queries'])
    if len(docs)>65536:
        raise ValueError('This index format uses 16-bit document positions')
    positions = {key:i for i,key in enumerate(docs)}
    frozen = read(retrieval/'frozen.json')
    candidates = {q:ds for split in frozen.values() for q,ds in split['candidates'].items()}
    order = np.array([[positions[d] for d in candidates[q]] for q in queries],dtype=np.uint16)
    dictionary = hashlib.sha256(json.dumps([docs,queries]).encode()).hexdigest()
    np.savez_compressed(target,dictionary=dictionary,order=order)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',choices=['evidence_inference','bioasq'])
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--retrieval',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.dataset=='evidence_inference':
        pack_evidence(args.prepared,args.output)
    elif args.retrieval:
        pack_bioasq(args.prepared,args.retrieval,args.output)
    else:
        p.error('--retrieval is required for bioasq')
