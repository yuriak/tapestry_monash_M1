"""Train a local reranker on prepared groups."""
import argparse
from pathlib import Path
from tapestry_m1.core.io import ContractError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--groups-root', type=Path)
    parser.add_argument('--objective', choices=['L0', 'L1'])
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(argv)
    try:
        from tapestry_m1.reranker.peft_training import run
        run(args.config,args.data_root,args.output,groups_root=args.groups_root,objective=args.objective,resume=args.resume,
            stop_after=args.stop_after,preflight=args.preflight)
        return 0
    except (ContractError, OSError, ValueError) as exc:
        print('Reranker training stopped: '+str(exc))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
