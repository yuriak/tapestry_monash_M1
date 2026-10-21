"""Architectural dependency checks for independent component evolution."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/tapestry_m1"


def test_component_dependency_direction():
    forbidden = {
        "core": {"data","runtime","retrieval","reranker","gateway","abstractor","generator","federation","workflows","evaluation"},
        "contracts": {"data","runtime","retrieval","reranker","gateway","abstractor","generator","federation","workflows","evaluation"},
        "data": {"retrieval","reranker","gateway","abstractor","generator","federation","workflows","evaluation"},
        "retrieval": {"reranker","gateway","abstractor","generator","federation","workflows","evaluation"},
        "reranker": {"retrieval","gateway","abstractor","generator","federation","workflows","evaluation"},
        "gateway": {"runtime","retrieval","reranker","abstractor","generator","federation","workflows","evaluation"},
        "evaluation": {"runtime","retrieval","reranker","gateway","abstractor","generator","federation","workflows"},
    }
    for owner, banned in forbidden.items():
        for path in (SOURCE / owner).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                modules = [node.module] if isinstance(node,ast.ImportFrom) else [n.name for n in node.names] if isinstance(node,ast.Import) else []
                for mod in modules:
                    parts = (mod or "").split(".")
                    if len(parts)>1 and parts[0]=="tapestry_m1":
                        assert parts[1] not in banned, (path,mod)


def test_source_checkout_is_not_a_distribution():
    import tomllib
    config = tomllib.loads((ROOT/"pyproject.toml").read_text())
    assert config["tool"]["uv"]["package"] is False
    assert "build-system" not in config
    assert "scripts" not in config["project"]
    assert config["project"]["dependencies"] == []
