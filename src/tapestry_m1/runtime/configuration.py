"""Resolve explicit model/source references and record the selected environment lock."""
import json
import os
from pathlib import Path
from tapestry_m1.core.io import ContractError, fingerprint
from tapestry_m1.runtime.assets import validate_pin


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ContractError("configuration must be an object")
    declared = config.pop("environment_lock")
    selected = os.environ.get("M1_ENVIRONMENT_LOCK")
    lock = Path(selected) if selected else path.parent / declared
    config["environment_lock_sha256"] = fingerprint(lock)
    config["environment_profile"] = os.environ.get("M1_ENVIRONMENT", "shared")
    for role in ("encoder", "reranker"):
        if role not in config:
            continue
        spec = config[role]
        if isinstance(spec, str):
            spec = json.loads((path.parent / spec).read_text())
        validate_pin(spec)
        if spec.get("trust_remote_code") is not False:
            raise ContractError("remote model code is not enabled")
        config[role] = spec
    if "datasets" in config:
        config["datasets"] = [json.loads((path.parent / source).read_text()) if isinstance(source, str) else source for source in config["datasets"]]
    return config
