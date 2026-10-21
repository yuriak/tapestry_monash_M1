import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
from tapestry_m1.gateway.batch import process_batch, passthrough, SCHEMA
from tapestry_m1.core.io import ContractError

FIXTURES = Path(__file__).parents[1] / "fixtures/gateway"


@pytest.fixture
def payload():
    return json.loads((FIXTURES / "request.jsonl").read_text())


def test_identity_and_no_aliasing(payload):
    original = copy.deepcopy(payload)
    result = process_batch([payload], {"demo"})[0]
    assert result == json.loads((FIXTURES / "response.jsonl").read_text())
    result["evidence"][0]["text"] = "changed"
    assert payload == original


def test_empty_evidence(payload):
    payload["evidence"] = []
    payload["ranking"]["candidates"] = []
    assert passthrough(payload, {"demo"})["status"] == "no_evidence"


@pytest.mark.parametrize("mutate", [
    lambda r: r["query"].pop("collection_id"),
    lambda r: r["ranking"].update(collection_id="foreign"),
    lambda r: r["evidence"][0].pop("source_reference"),
    lambda r: r["evidence"][0]["source_reference"].pop("revision"),
    lambda r: r["evidence"][0].update(source_doc_id="foreign::d"),
    lambda r: r["evidence"].append(copy.deepcopy(r["evidence"][0])),
    lambda r: r.update(evidence=[]),
    lambda r: r["ranking"]["candidates"].append(copy.deepcopy(r["ranking"]["candidates"][0])),
    lambda r: r["ranking"]["candidates"][0].update(original_rank=True),
    lambda r: r["ranking"]["candidates"][0].update(original_rank=2),
    lambda r: r["ranking"]["candidates"][0].update(reranked_rank=2),
    lambda r: r["ranking"]["candidates"][0].update(score=float("nan")),
    lambda r: r.update(abstraction_level="L1"),
    lambda r: r.update(abstraction_level="L2"),
    lambda r: r.update(abstraction_level="L3"),
    lambda r: r.update(schema_version="v0"),
    lambda r: r["evidence"][0].update(source_class="invented"),
    lambda r: r["query"].update(query_id="foreign::q"),
    lambda r: r["evidence"][0].update(text=""),
])
def test_contract_errors_are_sanitized(payload, mutate):
    mutate(payload)
    result = passthrough(payload, {"demo"})
    assert result["status"] == "error" and result["errors"]
    assert "query" not in result and "evidence" not in result


def test_batch_duplicates_limits_and_allowlist(payload):
    with pytest.raises(ContractError, match="duplicate"):
        process_batch([payload, payload], {"demo"})
    with pytest.raises(ContractError, match="budget"):
        process_batch([payload, dict(payload, request_id="second")], {"demo"}, 1)
    with pytest.raises(ContractError):
        process_batch([], set())
    assert passthrough(payload, {"foreign"})["status"] == "error"
    assert passthrough([], {"demo"})["status"] == "error"


def test_cli_is_atomic_and_bounded(tmp_path):
    out = tmp_path / "response.jsonl"
    cmd = [sys.executable, "-m", "tapestry_m1.gateway", str(FIXTURES / "request.jsonl"), str(out), "--collection", "demo"]
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    assert out.read_bytes() == (FIXTURES / "response.jsonl").read_bytes()
    assert subprocess.run(cmd + ["--max-line-bytes", "8"], capture_output=True).returncode == 2
    assert out.read_bytes() == (FIXTURES / "response.jsonl").read_bytes()
    assert subprocess.run(cmd + ["--max-records", "0"], capture_output=True).returncode == 2
