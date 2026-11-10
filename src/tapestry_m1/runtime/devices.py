"""Explicit scheduler or exclusive-host GPU authorization, without site policy."""
import importlib.metadata
import os
import subprocess
from tapestry_m1.core.io import ContractError


_exclusive_lock = None


def execution_context():
    """Default to Slurm; exclusive hosts require explicit opt-in and a shared lock."""
    global _exclusive_lock
    mode = os.environ.get("M1_GPU_EXECUTION", "slurm")
    if mode == "exclusive":
        import fcntl
        from pathlib import Path
        path = Path(os.environ.get("M1_GPU_LOCK_FILE", ""))
        if not path.is_absolute():
            raise ContractError("exclusive GPU execution requires an absolute shared lock file")
        if _exclusive_lock is None:
            handle = path.open("a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                raise ContractError("exclusive GPU is already held by another process")
            _exclusive_lock = handle  # Held for this process lifetime, including repeated checks.
        elif Path(_exclusive_lock.name) != path:
            raise ContractError("cannot change exclusive GPU lock during execution")
        return {"execution_mode": mode, "job_id": None}
    if mode != "slurm":
        raise ContractError("unknown GPU execution mode")
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise ContractError("GPU inference requires a scheduler allocation")
    text = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True)
    fields = dict(x.split("=", 1) for x in text.split() if "=" in x)
    # Older Slurm reports allocated running-job resources as TRES.
    allocated = fields.get("AllocTRES", fields.get("TRES", ""))
    gpu_count = next((v.split("=", 1)[1] for v in allocated.split(",") if v.startswith("gres/gpu=")), "0")
    if fields.get("JobState") != "RUNNING" or not gpu_count.isdigit() or int(gpu_count) < 1:
        raise ContractError("scheduler does not show an allocated running GPU job")
    return {"execution_mode": mode, "job_id": job}


def release_execution_lock():
    """Release after closing all GPU backends, before a sequential child stage."""
    global _exclusive_lock
    if _exclusive_lock is not None:
        _exclusive_lock.close()
        _exclusive_lock = None


def gpu_runtime():
    context = execution_context()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import torch
    if not torch.cuda.is_available():raise ContractError("allocated GPU unavailable to torch")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    return {**context,"gpu":torch.cuda.get_device_name(0),"vram_bytes":torch.cuda.get_device_properties(0).total_memory,
            "driver":subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True).splitlines()[0],"compute_capability":list(torch.cuda.get_device_capability(0)),"torch":torch.__version__,"cuda":torch.version.cuda,"packages":{name:importlib.metadata.version(name) for name in ["transformers","tokenizers","numpy","huggingface-hub"]}}
