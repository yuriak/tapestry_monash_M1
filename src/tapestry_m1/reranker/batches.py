"""Restartable deterministic query-group order, independent of model frameworks."""
import random
from tapestry_m1.core.io import ContractError


class GroupSampler:
    def __init__(self, size, seed):
        if size < 1:
            raise ContractError('empty training sampler')
        self.size = size
        self.generator = random.Random(seed)
        self.epoch = 0
        self.cursor = 0
        self.order = list(range(size))
        self.generator.shuffle(self.order)

    def take(self, count):
        result = []
        for _ in range(count):
            if self.cursor == self.size:
                self.epoch += 1
                self.cursor = 0
                self.order = list(range(self.size))
                self.generator.shuffle(self.order)
            result.append(self.order[self.cursor])
            self.cursor += 1
        return result

    def state_dict(self):
        return {'size': self.size, 'epoch': self.epoch, 'cursor': self.cursor, 'order': self.order,
                'generator': self.generator.getstate()}

    def load_state_dict(self, state):
        if state['size'] != self.size or sorted(state['order']) != list(range(self.size)) or not 0 <= state['cursor'] <= self.size:
            raise ContractError('sampler checkpoint identity mismatch')
        self.epoch, self.cursor, self.order = state['epoch'], state['cursor'], list(state['order'])
        self.generator.setstate(state['generator'])


class EpochSampler(GroupSampler):
    """Complete groups, no tail oversampling, checkpoint at optimizer-step boundaries."""
    def take_batch(self,count):
        if self.cursor==self.size:
            self.epoch+=1;self.cursor=0;self.order=list(range(self.size));self.generator.shuffle(self.order)
        actual=min(count,self.size-self.cursor)
        result=self.order[self.cursor:self.cursor+actual];self.cursor+=actual
        return result
