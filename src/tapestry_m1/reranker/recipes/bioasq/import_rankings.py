"""Import ordered candidate IDs and their retrieval scores for a binary qrels dataset."""
import argparse
from pathlib import Path

from .common import dump, load, views
from .prepare_jsonl import read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "rankings", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    data = load(args.data / "data.json")
    lookup = {}
    for row in read_rows(args.rankings):
        q = str(row["id"])
        docs = list(map(str, row["candidates"]))
        scores = list(map(float, row["scores"]))
        if q in lookup or len(docs) != len(scores) or len(docs) != len(set(docs)):
            raise ValueError(f"Invalid candidate list for {q}")
        if any(d not in data["documents"] for d in docs) or scores != sorted(scores, reverse=True):
            raise ValueError(f"Unknown document or unordered retrieval scores for {q}")
        lookup[q] = (docs, scores)
    if set(lookup) != set(data["queries"]):
        raise ValueError("Expected one candidate list per prepared query")
    frozen, metrics = {}, {}
    for split, queries in data["splits"].items():
        candidates = {q: lookup[q][0] for q in queries}
        scores = {q: lookup[q][1] for q in queries}
        gold = {q: data["gold"][q] for q in queries}
        frozen[split] = dict(candidates=candidates, gold=gold)
        metrics[split] = views(candidates, scores, gold)
    dump(args.output / "embedding-metrics.json", metrics)
    dump(args.output / "frozen.json", frozen)


if __name__ == "__main__":
    main()
