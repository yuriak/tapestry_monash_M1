"""Scheduler-backed GPU execution guard; no site-specific allocation policy."""
import importlib.metadata
import os
import subprocess
from tapestry_m1.core.io import ContractError


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
