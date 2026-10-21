"""CPU-only data and evaluation commands; roots are explicit or environment-configured."""
import argparse
import json
import os
from pathlib import Path
import sys
from tapestry_m1.data.acquisition import acquire, pinned_info, utc_now
from tapestry_m1.data.adapters import normalize
from tapestry_m1.data.audit import audit_overlap
from tapestry_m1.data.validation import validate
from tapestry_m1.core.io import ContractError, atomic_json, dumps


def load(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["inventory", "acquire", "normalize"]:
        p = sub.add_parser(name)
        p.add_argument("config", type=Path)
        p.add_argument("--data-root", type=Path, default=Path(os.environ.get("M1_DATA_ROOT", "data")))
        p.add_argument("--output", type=Path)
        if name == "acquire":p.add_argument("--max-bytes", type=int, required=True, help="Per-file byte limit")
        if name == "normalize":p.add_argument("--max-rows", type=int)
    p = sub.add_parser("validate");p.add_argument("root", type=Path);p.add_argument("--output", type=Path)
    p = sub.add_parser("audit");p.add_argument("benchmark", type=Path);p.add_argument("corpus", type=Path)
    p.add_argument("--max-documents", type=int);p.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command in ["inventory", "acquire", "normalize"]:
            spec = load(args.config)
            if args.command == "inventory":
                info = pinned_info(spec)
                result = {"source": spec, "verified_at": utc_now(), "resolved_revision": info["sha"],
                    "access": {"gated": info.get("gated"), "private": info.get("private")},
                    "card_metadata": info.get("cardData", {}), "observed_counts": None,
                    "status": "metadata_verified" if spec["acquisition_allowed"] else "blocked_pending_license_review"}
            elif args.command == "acquire":
                result = {"source": spec, "status": "downloaded", "files": acquire(spec, args.data_root, args.max_bytes)}
            else:
                raw = args.data_root / "raw" / spec["dataset_id"] / spec["revision"]
                destination = args.output or args.data_root / "normalized" / spec["dataset_id"] / spec["revision"]
                result = normalize(spec, raw, destination, args.max_rows)
                print(dumps(result))
                return 0
        elif args.command == "validate":result = validate(args.root)
        else:result = audit_overlap(args.benchmark, args.corpus, args.max_documents)
        if args.output:atomic_json(args.output, result)
        print(dumps(result))
        return 0
    except (ContractError, OSError, ValueError, KeyError, TypeError) as exc:
        print(dumps({"status": "error", "command": args.command, "error_type": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
