"""Matched Basic/Shifter dense-negative sampling for a one-pass screening run."""
import random

def window_size(n, sampler, step, total_steps):
    if sampler=='basic':return n
    if sampler!='shifter':raise ValueError(sampler)
    # Adapt upstream epoch-based schedule to four stages of ONE positive-pair pass.
    phase=min(3,step*4//total_steps)
    return max(min(10,n),n-(n//5)*phase)

def sample_negatives(negatives,sampler,step,total_steps,pair_index,count=4):
    size=window_size(len(negatives),sampler,step,total_steps)
    return random.Random(42*100000000+pair_index).sample(negatives[:size],count)
