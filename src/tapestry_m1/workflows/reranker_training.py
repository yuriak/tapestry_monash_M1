"""Portable loss-fixture validation and allocated local reranker training CLI."""
import argparse
import json
from pathlib import Path
from tapestry_m1.core.io import ContractError, records
from tapestry_m1.contracts.reranker_training import reference_loss, validate_groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fixture = sub.add_parser('fixture')
    fixture.add_argument('groups', type=Path)
    fixture.add_argument('expected', type=Path)
    train = sub.add_parser('train')
    train.add_argument('config', type=Path)
    train.add_argument('--data-root', type=Path, required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--groups-root', type=Path)
    train.add_argument('--objective', choices=['L0', 'L1'])
    train.add_argument('--resume', type=Path)
    train.add_argument('--stop-after', type=int)
    train.add_argument('--preflight', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'fixture':
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
        from tapestry_m1.reranker.peft_training import run
        run(args.config,args.data_root,args.output,groups_root=args.groups_root,objective=args.objective,resume=args.resume,
            stop_after=args.stop_after,preflight=args.preflight)
        return 0
    except (ContractError, OSError, ValueError) as exc:
        print('Reranker training stopped: '+str(exc))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
