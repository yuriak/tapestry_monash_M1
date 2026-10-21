"""CPU evaluation and paired comparison commands."""
import argparse
import json
from pathlib import Path
import sys
from tapestry_m1.core.io import ContractError, atomic_json, dumps
from tapestry_m1.evaluation.retrieval.evaluator import evaluate
from tapestry_m1.evaluation.statistics.bootstrap import paired_compare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("root", type=Path)
    p.add_argument("rankings", type=Path)
    p.add_argument("--gain", choices=["exponential", "linear"], default="exponential")
    p.add_argument("--mrr-depth", type=int)
    p.add_argument("--split")
    p.add_argument("--output", type=Path)
    p = sub.add_parser("compare")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            result = evaluate(args.root, args.rankings, args.gain, args.mrr_depth, args.split)
        else:
            result = paired_compare(json.loads(args.a.read_text()), json.loads(args.b.read_text()), args.seed, args.resamples)
        if args.output:
            atomic_json(args.output, result)
        print(dumps(result))
        return 0
    except (ContractError, OSError, ValueError, KeyError, TypeError) as exc:
        print(dumps({"status": "error", "command": args.command, "error_type": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
