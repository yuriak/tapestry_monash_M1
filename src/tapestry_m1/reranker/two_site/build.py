"""Materialize native, self-contained site inputs from one frozen benchmark."""
import argparse
import csv
import hashlib
import json
import pickle
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from tapestry_m1.data.casehold import FILES
from .partition import assign, read, records, summarize, write


def subset(mapping, ids):
    return {key: mapping[key] for key in ids}


def write_pickle(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump(obj, f, protocol=5)


def casehold(root, out, splits):
    for split, (name, _, _) in FILES.items():
        ids = set(splits[split])
        with (root/'raw'/name).open(newline='') as src, (out/name).open('w',newline='') as dest:
            reader, writer = csv.reader(src), csv.writer(dest)
            writer.writerow(next(reader))
            for row in reader:
                if f'{split}:{row[0]}' in ids:
                    writer.writerow(row)


def evidence(root, out, splits):
    prepared = root/'prepared'
    docs = read(prepared/'documents.json')
    articles = set()
    manifest = read(prepared/'manifest.json')
    manifest = {k:manifest[k] for k in ('gold_policy','offset_policy')}
    manifest['sentence_segmentation'] = 'reference en_core_sci_sm 0.5.4, spaCy 3.7.5'
    manifest['splits'] = {}
    for split, ids in splits.items():
        keys = set(ids)
        values = [r for r in read(prepared/f'{split}.json') if r['id'] in keys]
        articles.update(r['docid'] for r in values)
        manifest['splits'][split] = {'queries':len(values),
            'eligible_queries':sum(bool(r['positive']) for r in values)}
        write(out/f'{split}.json',values)
    write(out/'documents.json',subset(docs,sorted(articles)))
    write(out/'manifest.json',manifest)


def ranking(root, out, splits, task=None):
    data_dir = root/'prepared'
    target = out/'data'
    retrieval = root/'retrieval'
    route = out/'retrieval'
    ids = [q for group in splits.values() for q in group]
    keys = set(ids)
    if task:
        data_dir, target = data_dir/task, target/task
        retrieval, route = retrieval/task, route/task
    data = read(data_dir/'data.json')
    for split, qs in splits.items():
        if not set(qs).issubset(data['splits'][split]):
            raise ValueError(f'Unexpected query in {split}')
    data['splits'] = splits
    data['gold'] = subset(data['gold'],ids)
    data['train_pairs'] = [p for p in data['train_pairs'] if p[0] in keys]
    manifest = read(data_dir/'manifest.json')
    manifest.update(splits={s:len(qs) for s,qs in splits.items()}, train_pairs=len(data['train_pairs']))
    if task:
        order = [i for i,q in enumerate(data['query_ids']) if q in keys]
        data['query_ids'] = [data['query_ids'][i] for i in order]
        data['negative'] = {q:v for q,v in data['negative'].items() if q in keys}
        target.mkdir(parents=True, exist_ok=True)
        selection = np.load(data_dir/'selection.npy',mmap_mode='r')
        np.save(target/'selection.npy',selection[order])
    else:
        data['queries'] = subset(data['queries'],ids)
        # Parent-wide audit fields do not describe a site.
        manifest = {k:manifest[k] for k in ('source','corpus_file','max_length','seed','splits','train_pairs')}
        from .acquire import BIO
        manifest['source'] = {'dataset':'enelpol/rag-mini-bioasq','revision':BIO}
        with (data_dir/'tokens.pkl').open('rb') as f:
            cache = pickle.load(f)
        cache['queries'] = subset(cache['queries'],ids)
        write_pickle(target/'tokens.pkl', cache)
    write(target/'data.json',data)
    write(target/'manifest.json',manifest)
    frozen = read(retrieval/'frozen.json')
    views = {}
    for split,qs in splits.items():
        views[split] = {field:subset(frozen[split][field],qs) for field in ('candidates','gold')}
        if any(views[split]['gold'][q] != data['gold'][q] for q in qs):
            raise ValueError('Frozen and native qrels disagree')
    write(route/'frozen.json',views)
    if task:
        from ..recipes.il_chunk.common import views as metric_views
    else:
        from ..recipes.bioasq.common import views as metric_views
    metrics = {}
    for split in ('train','dev','test'):
        candidates = views[split]['candidates']
        # Metrics depend only on rank. Preserve the frozen order, including ties.
        scores = {q:range(0,-len(ds),-1) for q,ds in candidates.items()}
        metrics[split] = metric_views(candidates,scores,views[split]['gold'])
        if task and split=='test':
            for depth, values in metrics[split].items():
                cutoff = metrics['dev'][depth]['k']
                values.update(k=cutoff,macro_f1=values['f1_by_k'][cutoff-1])
    write(route/'embedding-metrics.json',metrics)


def check_inputs(root, expected):
    for name,checksum in expected.items():
        with (root/name).open('rb') as handle:
            actual = hashlib.file_digest(handle,'sha256').hexdigest()
        if actual != checksum:
            raise ValueError(f'Frozen source mismatch: {name}')


def build(dataset, root, rules):
    rows = assign(dataset,records(dataset,root,rules),rules)
    keys = [(r['split'],r['query_id']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Repeated query identity')
    assignment = {r['group_id']:r['site'] for r in rows}
    if any(assignment[r['group_id']] != r['site'] for r in rows):
        raise ValueError('Overlapping site group')
    output = root/rules['version']
    summary = summarize(rows)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        stage = Path(tmp)/rules['version']
        for site in ['A','B']:
            out = stage/site
            out.mkdir(parents=True)
            kept = [r for r in rows if r['site']==site]
            splits = {s:[r['query_id'] for r in kept if r['split']==s] for s in summary[site]}
            if len(splits)!=3 or any(not qs for qs in splits.values()):
                raise ValueError('Missing site split')
            with (out/'index.jsonl').open('w') as f:
                for r in kept:
                    f.write(json.dumps(r,sort_keys=True)+'\n')
            if dataset=='casehold':
                casehold(root,out,splits)
            elif dataset=='evidence_inference':
                evidence(root,out,splits)
            elif dataset=='bioasq':
                ranking(root,out,splits)
            else:
                ids = [q for qs in splits.values() for q in qs]
                with (root/'prepared/chunks.pkl').open('rb') as f:
                    chunks = pickle.load(f)
                for key in ['tokens','spans']:
                    chunks[key]['queries'] = subset(chunks[key]['queries'],ids)
                write_pickle(out/'data/chunks.pkl',chunks)
                for task in ['lsr','pcr']:
                    ranking(root,out,splits,task)
        write(stage/'summary.json',summary)
        hashes = {}
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                with path.open('rb') as f:
                    hashes[str(path.relative_to(stage))] = hashlib.file_digest(f,'sha256').hexdigest()
        write(stage/'checksums.json',hashes)
        if output.exists():
            if read(output/'checksums.json')!=hashes:
                raise ValueError(f'Existing {output} differs. Use a fresh root.')
            for name,expected in hashes.items():
                with (output/name).open('rb') as f:
                    if hashlib.file_digest(f,'sha256').hexdigest()!=expected:
                        raise ValueError(f'Damaged output: {name}')
        else:
            stage.rename(output)
    print(dataset, 'site train queries', *(summary[s]['train']['queries'] for s in ['A','B']), flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=['casehold','evidence_inference','il_pcsr','bioasq'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--rules',type=Path,required=True)
    args=parser.parse_args()
    reference = read(args.rules.with_name('reference.json'))[args.dataset]
    check_inputs(args.root,reference['inputs'])
    build(args.dataset,args.root,read(args.rules))
