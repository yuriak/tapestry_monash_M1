"""Native identifier metrics: macro query Top1 and pooled sentence AUROC."""
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def evaluate(records, predictions, gold_key='positive'):
    labels, scores, hit, reciprocal, margins, dispersions, query_bce = [], [], [], [], [], [], []
    for r in records:
        gold = set(r[gold_key])
        if not gold:
            continue
        values = predictions[r['id']]
        order = sorted(range(len(values)), key=lambda i: (-values[i], i))
        hit.append(int(order[0] in gold))
        reciprocal.append(next(1/(k+1) for k, i in enumerate(order) if i in gold))
        labels.extend(int(i in gold) for i in range(len(values)))
        scores.extend(values)
        v = np.asarray(values)
        y = np.asarray([int(i in gold) for i in range(len(v))])
        query_bce.append(float(np.mean(np.logaddexp(0,v)-y*v)))
        dispersions.append(float(v.std()))
        negative = [v[i] for i in range(len(v)) if i not in gold]
        if negative:
            margins.append(float(np.mean([v[i] for i in gold])-max(negative)))
    return {'top1':float(np.mean(hit)) if hit else None,
            'auroc':float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
            'sentence_macro_f1':float(f1_score(labels, np.array(scores)>0, average='macro', zero_division=0)) if labels else None,
            'mrr':float(np.mean(reciprocal)) if reciprocal else None,
            'query_bce':float(np.mean(query_bce)) if query_bce else None,
            'mean_logit_std':float(np.mean(dispersions)) if dispersions else None,
            'positive_hardest_negative_margin':float(np.mean(margins)) if margins else None,
            'eligible_queries':len(hit), 'total_queries':len(records),
            'conservative_top1':sum(hit)/len(records) if records else None}


def paired_top1_ci(records, baseline, trained, seed=0, replicates=10000):
    differences = []
    for r in records:
        if not r['positive']:
            continue
        def hit(p):
            scores = p[r['id']]
            return int(max(range(len(scores)), key=lambda i:(scores[i], -i)) in r['positive'])
        differences.append(hit(trained)-hit(baseline))
    d = np.asarray(differences)
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(d, len(d), replace=True))) for _ in range(replicates)]
    return {'difference':float(d.mean()), 'ci95':np.quantile(means,[.025,.975]).tolist(),
            'replicates':replicates, 'resampling_unit':'query, within seed'}
