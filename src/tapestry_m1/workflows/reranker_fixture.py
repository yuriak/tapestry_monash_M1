"""Inspect the synthetic reranker loss fixture."""
import argparse
import json
from pathlib import Path
from tapestry_m1.core.io import ContractError, records
from tapestry_m1.contracts.reranker_training import reference_loss, validate_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('groups', type=Path)
    parser.add_argument('expected', type=Path)
    args = parser.parse_args()
    groups = validate_groups(list(records(args.groups)), synthetic_only=True)
    expected = json.loads(args.expected.read_text())
    output = {'groups': len(groups), 'status': 'contract-ready'}
    for objective in ['L0','L1']:
        observed = reference_loss(expected['score_groups'], objective)
        for key in ['bce','margin','total']:
            if abs(observed[key]-expected[objective][key]) > 1e-12:
                raise ContractError('fixture scalar differs: '+objective+'/'+key)
        if any(abs(a-b)>1e-12 for ra,rb in zip(observed['gradients_wrt_score'],expected[objective]['gradients_wrt_score']) for a,b in zip(ra,rb)):
            raise ContractError('fixture gradients differ')
        output[objective] = observed
    print(json.dumps(output,indent=2)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
