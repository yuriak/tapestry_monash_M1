"""Evaluate a base model or LoRA adapter on the prepared dev and test splits."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .train import evaluate
from .model import Ranker, Tokens, INSTRUCTION


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
    a.max_length = json.loads((a.data / 'manifest.json').read_text())['max_length']
    model = Ranker(a.model, adapter=a.adapter).cuda()
    tokens = Tokens(a.model, a.data / 'tokens.pkl', instruction=json.loads((a.data / 'manifest.json').read_text()).get('instruction', INSTRUCTION))
    for split in ('dev', 'test'):
        evaluate(a, model, tokens, frozen[split], a.output / split, a.budget)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
