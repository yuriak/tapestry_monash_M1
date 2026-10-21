"""Verify normalized source identity before downstream consumption."""
import json
from pathlib import Path
from tapestry_m1.core.io import ContractError, fingerprint


def input_root(data_root,spec):
    root=Path(data_root)/"normalized"/spec["dataset_id"]/spec["revision"]
    manifest=json.loads((root/"manifest.json").read_text())
    if manifest["source"]!=spec or manifest["status"]!="validated" or manifest["truncated"]:raise ContractError("source is partial or revision/config differs")
    for name,sha in manifest["output_sha256"].items():
        if fingerprint(root/name)!=sha:raise ContractError("normalized source hash mismatch")
    return root,manifest
