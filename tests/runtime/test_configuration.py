from pathlib import Path
import json
from tapestry_m1.runtime.configuration import load_config
from tapestry_m1.core.io import fingerprint

ROOT = Path(__file__).resolve().parents[2]


def test_component_configs_resolve_sources_and_selected_lock(monkeypatch, tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text("synthetic-independent-lock")
    monkeypatch.setenv("M1_ENVIRONMENT_LOCK", str(lock))
    monkeypatch.setenv("M1_ENVIRONMENT", "synthetic-stack")
    c = load_config(ROOT / "configs/experiments/zero_shot_native/native-qwen3.json")
    assert c["environment_lock_sha256"] == fingerprint(lock)
    assert c["environment_profile"] == "synthetic-stack"
    assert [d["expected_counts"]["queries"] for d in c["datasets"]] == [112,100]
    corpus = load_config(ROOT / "configs/retrieval/au-corpus.json")
    assert "reranker" not in corpus and "datasets" not in corpus
    assert corpus["encoder"]["model_id"] == "Qwen/Qwen3-Embedding-4B"
