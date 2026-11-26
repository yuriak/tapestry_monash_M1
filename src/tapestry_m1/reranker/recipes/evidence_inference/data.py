"""Evidence Inference 2.0 data adapter; original source texts stay outside Git."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False) + '\n')
    tmp.replace(path)


def sentence_labels(sentences, spans, upstream=False):
    result = []
    for i, sent in enumerate(sentences):
        a, b = sent['start'], sent['end']
        for start, end in spans:
            overlap = max(0, min(b, end) - max(a, start))
            if overlap and (not upstream or overlap >= .2 * (b-a) or overlap >= .9 * (end-start)):
                result.append(i)
                break
    return result


def sample_indices(record, count, ratio, seed, epoch):
    positives = set(record['positive'])
    # Any sentence touching a known span is excluded from negatives.
    negatives = [i for i in range(count) if i not in positives]
    digest = hashlib.sha256(f'{seed}:{epoch}:{record["id"]}'.encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], 'big'))
    negatives = rng.sample(negatives, min(len(negatives), ratio * len(positives)))
    values = [(i, 1) for i in sorted(positives)] + [(i, 0) for i in negatives]
    rng.shuffle(values)
    return values


def length_batches(items, max_batch, token_budget):
    """Items end with token IDs. Preserve every item; bound padded tokens."""
    ordered=sorted(items,key=lambda item:len(item[-1]))
    batch=[]
    for item in ordered:
        if batch and (len(batch)>=max_batch or (len(batch)+1)*len(item[-1])>token_budget):
            yield batch
            batch=[]
        batch.append(item)
    if batch:yield batch


def prepare(source, output, workers=1, sentence_index=None):
    source, output = Path(source), Path(output)
    prompts = list(csv.DictReader((source/'prompts_merged.csv').open()))
    annotations = list(csv.DictReader((source/'annotations_merged.csv').open()))
    by_prompt = {}
    for row in annotations:
        if (row['Valid Label'] == 'True' and row['Valid Reasoning'] == 'True'
                and row['Label'] != 'invalid prompt' and row['Annotations']):
            by_prompt.setdefault(row['PromptID'], []).append(row)
    splits = {s: set((source/'splits'/f'{s}_article_ids.txt').read_text().split())
              for s in ['train', 'validation', 'test']}
    for s in splits:
        for t in splits:
            if s != t and splits[s] & splits[t]:
                raise ValueError('Official article splits overlap')
    prompts = [p for p in prompts if p['PromptID'] in by_prompt
               and all(p[k].strip() for k in ['Intervention', 'Comparator', 'Outcome'])
               and any(p['PMCID'] in ids for ids in splits.values())]
    article_ids = sorted({p['PMCID'] for p in prompts})
    # Keep published text byte/character coordinate system, without whitespace normalization.
    def texts():
        for docid in article_ids:
            path = source/'txt_files'/f'PMC{docid}.txt'
            if path.exists():
                yield path.read_text(encoding='utf-8'),docid
    docs = {}
    if sentence_index:
        import gzip
        with gzip.open(sentence_index,'rt') as f:
            boundaries = json.load(f)
        for text, docid in texts():
            docs[docid] = [{'text':text[start:end], 'start':start, 'end':end}
                           for start,end in boundaries[docid]]
        versions = {'sentence_index_sha256':hashlib.sha256(Path(sentence_index).read_bytes()).hexdigest()}
    else:
        import spacy
        nlp = spacy.load('en_core_sci_sm', disable=['ner'])
        nlp.max_length = 5_000_000
        for i, (doc, docid) in enumerate(nlp.pipe(texts(),as_tuples=True,batch_size=8,n_process=workers)):
            docs[docid] = [{'text':s.text, 'start':s.start_char, 'end':s.end_char} for s in doc.sents]
            if i % 100 == 0:
                print(f'PREP articles={i+1}/{len(article_ids)}', flush=True)
        versions = {'spacy':spacy.__version__, 'sentence_model':nlp.meta['version']}
    dump(output/'documents.json', docs)
    stats = {**versions, 'gold_policy':'all nonempty half-open character intersections',
             'offset_policy':'published pipeline half-open; no heuristic text repair', 'splits':{}}
    for split, ids in splits.items():
        rows = []
        st = {'missing_document':0, 'invalid_spans':0, 'out_of_bounds_spans':0,
              'without_positive':0, 'positive_sentences':0, 'sentences':0, 'upstream_gold_diff_queries':0}
        for p in prompts:
            if p['PMCID'] not in ids:
                continue
            if p['PMCID'] not in docs:
                st['missing_document'] += 1
                continue
            sents = docs[p['PMCID']]
            spans = set()
            for a in by_prompt[p['PromptID']]:
                start, end = int(float(a['Evidence Start'])), int(float(a['Evidence End']))
                if start < 0 or end <= start:
                    st['invalid_spans'] += 1
                elif not sents or end > sents[-1]['end']:
                    st['out_of_bounds_spans'] += 1
                else:
                    spans.add((start, end))
            pos = sentence_labels(sents, spans)
            old = sentence_labels(sents, spans, upstream=True)
            st['without_positive'] += not bool(pos)
            st['upstream_gold_diff_queries'] += pos != old
            st['positive_sentences'] += len(pos)
            st['sentences'] += len(sents)
            rows.append({'id':p['PromptID'], 'docid':p['PMCID'],
                         'query':f"Intervention: {p['Intervention']}\nComparator: {p['Comparator']}\nOutcome: {p['Outcome']}",
                         'positive':pos, 'upstream_positive':old, 'spans':sorted(spans)})
        st['queries'] = len(rows)
        st['eligible_queries'] = sum(bool(r['positive']) for r in rows)
        stats['splits'][split] = st
        dump(output/f'{split}.json', rows)
    dump(output/'manifest.json', stats)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--sentence-index', type=Path)
    a = p.parse_args()
    prepare(a.source, a.output, a.workers, a.sentence_index)
