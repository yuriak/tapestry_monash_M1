"""Bounded local JSONL gateway transport; no network or model execution."""
import argparse
import json
from pathlib import Path
from tapestry_m1.core.io import ContractError
from tapestry_m1.core.artifacts import atomic_bytes
from tapestry_m1.gateway.batch import process_batch


def _rows(path, max_line_bytes):
    with Path(path).open("rb") as handle:
        while True:
            line = handle.readline(max_line_bytes + 1)
            if not line:
                return
            if len(line) > max_line_bytes:
                raise ContractError("JSONL line byte budget exceeded")
            if line.strip():
                try:
                    yield json.loads(line)
                except (ValueError, UnicodeError) as exc:
                    raise ContractError("invalid JSONL record") from exc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--collection", action="append", required=True)
    p.add_argument("--max-records", type=int, default=1000)
    p.add_argument("--max-line-bytes", type=int, default=1048576)
    a = p.parse_args()
    try:
        if not 1 <= a.max_records <= 1000 or not 1 <= a.max_line_bytes <= 1048576:
            raise ContractError("budgets must be positive and cannot exceed the documented hard limits")
        rows = process_batch(_rows(a.input, a.max_line_bytes), set(a.collection), a.max_records)
        payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in rows)
        atomic_bytes(a.output, payload.encode("utf-8"))
    except (ContractError, OSError, ValueError) as exc:
        p.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()
