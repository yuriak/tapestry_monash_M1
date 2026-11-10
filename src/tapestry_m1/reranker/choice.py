"""Native-choice softmax objective and stable original-index ranking metrics."""
import math
from tapestry_m1.core.io import ContractError


def group_ce(scores, labels):
    import torch
    if scores.ndim != 2 or scores.shape[1] != 5 or len(labels) != len(scores):
        raise ContractError('expected five candidates and one label per group')
    labels = torch.as_tensor(labels, device=scores.device, dtype=torch.long)
    if ((labels < 0) | (labels >= 5)).any() or not torch.isfinite(scores).all():
        raise ContractError('invalid labels or nonfinite scores')
    return -torch.log_softmax(scores.float(), dim=1).gather(1, labels[:, None]).mean()


def metrics(scores, gold):
    if len(scores) != 5 or type(gold) is not int or gold not in range(5) or not all(math.isfinite(x) for x in scores):
        raise ContractError('invalid choice scores or label')
    order = sorted(range(5), key=lambda i: (-scores[i], i))
    rank = order.index(gold)+1
    mean = sum(scores)/5
    return {'rank': rank, 'accuracy': float(rank == 1), 'mrr': 1/rank,
            'ndcg': 1/math.log2(rank+1), 'margin': scores[gold]-max(s for i,s in enumerate(scores) if i != gold),
            'dispersion': math.sqrt(sum((s-mean)**2 for s in scores)/5)}


def aggregate(rows):
    if not rows:
        raise ContractError('no scored rows')
    return {'count': len(rows), **{k: sum(r[k] for r in rows)/len(rows)
             for k in ['accuracy','mrr','ndcg','margin','dispersion']}}


def stop_reason(base, history):
    step, current = history[-1]['step'], history[-1]
    if current['dispersion'] < max(1e-6, .05*base['dispersion']):
        return 'severe_discrimination_collapse_below_5_percent_of_base'
    if step == 20 and all(base['accuracy']-h['accuracy'] >= .05-1e-12 for h in history if h['step'] in [5,20]):
        return 'step5_and20_accuracy_drop_at_least_5pp'
    if step > 20:
        best = max(history[:-1], key=lambda h: (h['accuracy'],h['mrr'],-h['step']))
        if current['train_loss'] < best['train_loss'] and best['accuracy']-current['accuracy'] >= .01-1e-12 and current['mrr'] <= best['mrr']:
            return 'train_loss_improves_but_dev_degrades'
    return 'maximum_120_updates' if step == 120 else None


def select(history):
    return max(history, key=lambda h: (h['accuracy'], h['mrr'], -h['step']))


def lr_multiplier(update):
    """LR used for the numbered update: warm up 1..10, decay 11..120."""
    return min(update/10, max(0., (120-update)/110))
