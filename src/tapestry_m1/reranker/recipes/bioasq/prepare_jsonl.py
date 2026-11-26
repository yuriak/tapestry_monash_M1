"""Prepare binary-relevance JSONL collections for the dense retrieval recipe."""
import argparse
import json
from pathlib import Path

from .prepare import write_dataset


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "data", "model"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--instruction", required=True)
    args = parser.parse_args()
    documents = {}
    for row in read_rows(args.input / "documents.jsonl"):
        doc = str(row["id"])
        if doc in documents:
            raise ValueError(f"Duplicate document ID {doc}")
        documents[doc] = row["text"]
    queries, gold = {}, {}
    splits = {name: [] for name in ("train", "dev", "test")}
    for row in read_rows(args.input / "queries.jsonl"):
        q = str(row["id"])
        if q in queries:
            raise ValueError(f"Duplicate query ID {q}")
        queries[q] = row["text"]
        gold[q] = list(dict.fromkeys(map(str, row["positive_ids"])))
        if any(doc not in documents for doc in gold[q]):
            raise ValueError(f"Unknown positive document for {q}")
        splits[row["split"]].append(q)
    pairs = [[q, d] for q in splits["train"] for d in gold[q]]
    if not pairs or not all(splits.values()):
        raise ValueError("Expected training positives and non-empty train/dev/test splits")
    data = dict(documents=documents, queries=queries, gold=gold, splits=splits, train_pairs=pairs)
    write_dataset(args.data, data, args.model, {"format": "jsonl"}, [],
                  {q: [] for q in queries}, {}, "documents.jsonl", args.instruction)


if __name__ == "__main__":
    main()
