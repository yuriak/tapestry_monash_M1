"""Differentiable yes/no score and query-group reductions; imported only by training."""
from tapestry_m1.core.io import ContractError


def relevance_logits(logits, no_id, yes_id):
    if no_id == yes_id or min(no_id, yes_id) < 0 or max(no_id, yes_id) >= logits.shape[-1]:
        raise ContractError('invalid yes/no token IDs')
    return logits[..., yes_id].float() - logits[..., no_id].float()


def group_loss(scores, objective, margin=1., weight=.5):
    import torch
    if scores.ndim != 2 or scores.shape[1] != 4 or scores.shape[0] == 0 or objective not in ('L0', 'L1'):
        raise ContractError('L0/L1 requires [groups, 4] logit differences')
    if not torch.isfinite(scores).all():
        raise ContractError('nonfinite training scores')
    labels = torch.zeros_like(scores); labels[:, 0] = 1
    bce = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels, reduction='none').mean(dim=1).mean()
    hinge = torch.relu(margin - scores[:, :1] + scores[:, 1:]).mean(dim=1).mean()
    return bce + (weight*hinge if objective == 'L1' else 0), {'bce': bce, 'margin': hinge}
