"""Evaluate a base model or LoRA adapter on the prepared dev and test splits."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .run import evaluation, Scorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    a = parser.parse_args()
    a.local_rank = int(os.getenv('LOCAL_RANK', 0))
    a.rank = int(os.getenv('RANK', 0))
    a.world = int(os.getenv('WORLD_SIZE', 1))
    torch.cuda.set_device(a.local_rank)
    dist.init_process_group('nccl')
    scorer = Scorer(a.model, 512)
    model = scorer.load(adapter=a.adapter)
    docs = json.loads((a.data / 'documents.json').read_text())
    for split in ('validation', 'test'):
        rows = json.loads((a.data / (split + '.json')).read_text())
        evaluation(model, scorer, rows, docs, a.output / split, a.rank, a.world, 512)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
