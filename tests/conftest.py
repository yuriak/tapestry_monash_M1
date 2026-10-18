import json
from pathlib import Path
import pytest
from tapestry_m1.io import dumps


def write_rows(path, rows):
    path.write_text("".join(dumps(row)+"\n" for row in rows))


@pytest.fixture
def benchmark(tmp_path):
    root=tmp_path/"benchmark";root.mkdir()
    for name in ["documents", "queries", "qrels"]:
        (root/(name+".jsonl")).write_bytes((Path(__file__).parent/"fixtures"/(name+".jsonl")).read_bytes())
    return root


@pytest.fixture
def rankings(tmp_path):
    path=tmp_path/"rankings.jsonl"
    path.write_bytes((Path(__file__).parent/"fixtures/rankings.jsonl").read_bytes())
    return path
