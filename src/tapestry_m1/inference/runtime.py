"""Verified inputs, pinned artifacts and scheduler-backed GPU execution."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from tapestry_m1.io import ContractError, fingerprint
from tapestry_m1.inference.artifacts import identity


def code_fingerprint():
    root=Path(__file__).parent
    return identity({p.name:fingerprint(p) for p in sorted(root.glob("*.py"))})


def load_config(path):
    path=Path(path);config=json.loads(path.read_text())
    config["environment_lock_sha256"]=fingerprint(path.parent/config.pop("environment_lock"))
    for role in ["encoder","reranker"]:
        config[role]=json.loads((path.parent/config[role]).read_text())
        spec=config[role]
        if len(spec["revision"])!=40 or spec["tokenizer_revision"]!=spec["revision"] or spec["trust_remote_code"]:raise ContractError("models/tokenizers must pin a reviewed commit")
    if config["top_k"]!=100:raise ContractError("native protocol requires Top-100")
    return config


def verify_model(spec,data_root):
    root=Path(data_root)/"models"/spec["model_id"].split("/")[-1]/spec["revision"]
    saved=json.loads((root/"verified.json").read_text())
    if saved["model_id"]!=spec["model_id"] or saved["revision"]!=spec["revision"]:raise ContractError("model identity mismatch")
    for name,meta in spec["files"].items():
        actual=fingerprint(root/name)
        if actual!=saved["files"][name]["sha256"] or meta["sha256"] and actual!=meta["sha256"]:raise ContractError("model checksum mismatch")
    return root,saved


def input_root(data_root,spec):
    root=Path(data_root)/"normalized"/spec["dataset_id"]/spec["revision"]
    manifest=json.loads((root/"manifest.json").read_text())
    if manifest["source"]!=spec or manifest["status"]!="validated" or manifest["truncated"]:raise ContractError("source is partial or revision/config differs")
    for name,sha in manifest["output_sha256"].items():
        if fingerprint(root/name)!=sha:raise ContractError("normalized source hash mismatch")
    return root,manifest


def gpu_runtime():
    job=os.environ.get("SLURM_JOB_ID")
    if not job:raise ContractError("GPU inference requires a scheduler allocation")
    text=subprocess.check_output(["scontrol","show","job","-o",job],text=True)
    fields=dict(x.split("=",1) for x in text.split() if "=" in x)
    if fields.get("JobState")!="RUNNING" or "gres/gpu=" not in fields.get("AllocTRES",""):raise ContractError("scheduler does not show an allocated running GPU job")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import torch
    if not torch.cuda.is_available():raise ContractError("allocated GPU unavailable to torch")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    return {"job_id":job,"gpu":torch.cuda.get_device_name(0),"vram_bytes":torch.cuda.get_device_properties(0).total_memory,
            "driver":subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True).splitlines()[0],"compute_capability":list(torch.cuda.get_device_capability(0)),"torch":torch.__version__,"cuda":torch.version.cuda,"packages":{name:importlib.metadata.version(name) for name in ["transformers","tokenizers","numpy","huggingface-hub"]}}
