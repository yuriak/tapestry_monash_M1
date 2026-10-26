"""Immutable local checkpoints with checksums and full RNG/optimizer continuation."""
import json
import os
from pathlib import Path
import random
import tempfile
from tapestry_m1.core.io import atomic_json, ContractError, fingerprint


def rng_state():
    import torch
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import torch
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def save_checkpoint(destination, adapter, optimizer, scheduler, sampler, metadata):
    import torch
    destination = Path(destination)
    if destination.exists():
        raise ContractError('checkpoint already exists; never overwrite')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + destination.name, dir=destination.parent))
    torch.save(adapter, stage / 'adapter.pt')
    torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'sampler': sampler.state_dict(), 'rng': rng_state()}, stage / 'continuation.pt')
    atomic_json(stage / 'manifest.json', metadata | {'files': {name: fingerprint(stage/name)
                for name in ('adapter.pt', 'continuation.pt')}})
    for name in ('adapter.pt', 'continuation.pt', 'manifest.json'):
        with (stage/name).open('rb') as handle:
            os.fsync(handle.fileno())
    os.rename(stage, destination)
    return json.loads((destination/'manifest.json').read_text())


def load_checkpoint(path, expected_identity):
    import torch
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    if manifest['identity'] != expected_identity:
        raise ContractError('resume config/input/model/code/lock identity mismatch')
    for name in ('adapter.pt', 'continuation.pt'):
        if fingerprint(path/name) != manifest['files'][name]:
            raise ContractError('checkpoint checksum mismatch')
    # Only consume our own explicitly supplied, locally generated and checksummed checkpoint.
    adapter = torch.load(path/'adapter.pt', map_location='cpu', weights_only=True)
    continuation = torch.load(path/'continuation.pt', map_location='cpu', weights_only=False)
    return adapter, continuation, manifest
