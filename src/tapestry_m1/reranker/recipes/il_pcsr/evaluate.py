"""Evaluate a base model or LoRA adapter on the prepared dev and test splits."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .screen import FastScorer, evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--retrieval', type=Path, required=True)
    parser.add_argument('--budget', type=int, default=65536)
    parser.add_argument('--task', choices=('lsr','pcr'), required=True)
    a = parser.parse_args()
    a.local_rank = int(os.getenv('LOCAL_RANK', 0))
    a.rank = int(os.getenv('RANK', 0))
    a.world = int(os.getenv('WORLD_SIZE', 1))
    torch.cuda.set_device(a.local_rank)
    frozen = json.loads((a.retrieval / 'frozen.json').read_text())
    data = json.loads((a.data / 'data.json').read_text())
    scorer = FastScorer(a, a.task)
    model = scorer.model(adapter=a.adapter)
    for split in ('dev', 'test'):
        evaluation(model, scorer, frozen[a.task][split], data['gold'][a.task], a.output / split, a.budget)


if __name__ == "__main__":
    main()
