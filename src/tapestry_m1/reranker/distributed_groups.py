"""Exactly-once global group batches and sample-weighted DDP accumulation."""
from contextlib import nullcontext
import math
import random
from tapestry_m1.core.io import ContractError


def epoch_order(size, seed):
    order=list(range(size));random.Random(seed).shuffle(order)
    return order


def local_batch(order, step, global_batch, rank, world_size):
    batch=order[(step-1)*global_batch:step*global_batch]
    local=batch[rank::world_size]
    if not local:
        raise ContractError('this group schedule requires at least one real sample per rank')
    return local,len(batch)


def checkpoint_steps(size, batch, fractions):
    total=math.ceil(size/batch)
    return [math.ceil(total*f) for f in fractions]


def lr_factor(update, total, warmup):
    return min(update/warmup,max(0.,(total-update)/(total-warmup)))


def accumulate_groups(model, items, global_count, world_size, loss_fn):
    """DDP averages gradients across ranks; scale each sample by world/N.

    Synchronize only on each rank's last real group. Unequal local lengths are
    allowed, with one synchronized backward per rank and no padded samples.
    Forward must be inside no_sync too. Caller zeroes/clips/steps the optimizer.
    """
    if not items or global_count < len(items):
        raise ContractError('invalid distributed group batch')
    total_loss=0.
    for i,item in enumerate(items):
        context=model.no_sync() if i<len(items)-1 else nullcontext()
        with context:
            loss=loss_fn(item)
            if not loss.isfinite():
                raise ContractError('nonfinite group loss')
            (loss*(world_size/global_count)).backward()
            total_loss+=loss.detach().item()
    return total_loss
