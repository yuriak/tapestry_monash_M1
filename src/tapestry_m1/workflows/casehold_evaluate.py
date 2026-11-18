"""Five-class Macro-F1 from one CaseHOLD prediction directory."""
import argparse
import hashlib
import json
from pathlib import Path

from tapestry_m1.reranker.choice import class_f1
from tapestry_m1.core.io import atomic_json
from tapestry_m1.data.casehold import load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("raw", "predictions", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    args = parser.parse_args()
    gold = {r["query_id"]: r["gold"] for r in load_split(args.raw, args.split)}
    seen = set()
    matrix = [[0] * 5 for _ in range(5)]
    folders = sorted(args.predictions.glob("rank-*")) or [args.predictions]
    for folder in folders:
        manifest = json.loads((folder / "summary.json").read_text())
        for name, checksum in manifest["chunks_sha256"].items():
            raw = (folder / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != checksum:
                raise ValueError(f"Invalid prediction chunk {folder / name}")
            for row in json.loads(raw):
                q = row["query_id"]
                if q in seen or row["gold"] != gold[q] or len(row["scores"]) != 5:
                    raise ValueError(f"Invalid prediction for {q}")
                seen.add(q)
                pred = max(range(5), key=row["scores"].__getitem__)
                matrix[gold[q]][pred] += 1
    if seen != gold.keys():
        raise ValueError("Incomplete prediction set")
    f1 = class_f1(matrix)
    atomic_json(args.output, {"split": args.split, "queries": len(seen),
                "choice_macro_f1": sum(f1) / 5, "class_f1": f1, "confusion_matrix": matrix})


if __name__ == "__main__":
    main()
