"""Shared distributed helpers and random state for resumable LoRA recipes."""
import gc
import random

import numpy as np
import torch
import torch.distributed as dist

def sync():dist.barrier(device_ids=[torch.cuda.current_device()])
def gather(x):
    xs=[None]*dist.get_world_size();dist.all_gather_object(xs,x);return xs

def broadcast(x):
    xs=[x if dist.get_rank()==0 else None];dist.broadcast_object_list(xs,src=0);return xs[0]

def clean():gc.collect();torch.cuda.empty_cache()
def seed_all():random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}
def restore(r):random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state(r['cuda'])

