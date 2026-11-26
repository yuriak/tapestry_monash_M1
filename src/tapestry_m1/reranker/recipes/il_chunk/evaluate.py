"""Evaluate a base model or LoRA adapter on the prepared dev and test splits."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .train import evaluate
from .model import Ranker, Tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--retrieval', type=Path, required=True)
    parser.add_argument('--budget', type=int, default=65536)
    a = parser.parse_args()
    a.local_rank = int(os.getenv('LOCAL_RANK', 0))
    a.rank = int(os.getenv('RANK', 0))
    a.world = int(os.getenv('WORLD_SIZE', 1))
    torch.cuda.set_device(a.local_rank)
    dist.init_process_group('nccl')
    frozen = json.loads((a.retrieval / 'frozen.json').read_text())
    a.max_length = 8192
    tokens = Tokens(a.model, a.data)
    for label, adapter in [('zero-shot', None)] + ([('adapted', a.adapter)] if a.adapter else []):
        model = Ranker(a.model, adapter=adapter).cuda()
        for split in ('dev', 'test'):
            evaluate(a, model, tokens, frozen[split], a.output / label / split, a.budget)
        del model
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
