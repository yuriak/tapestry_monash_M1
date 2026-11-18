"""One hard plus two broad retrieved negatives, excluding known-positive aliases."""
import random

def mixed_negatives(pool,index):
    rng=random.Random(42*100000000+index)
    hard=rng.choice(pool[:min(20,len(pool))])
    return [hard]+rng.sample([d for d in pool if d!=hard],2)
