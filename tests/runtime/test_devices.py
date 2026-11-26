import pytest
from tapestry_m1.core.io import ContractError
from tapestry_m1.runtime.devices import gpu_runtime


def test_gpu_guard_no_allocation(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    monkeypatch.delenv('M1_GPU_EXECUTION',raising=False)
    with pytest.raises(ContractError,match='scheduler'):gpu_runtime()


def test_exclusive_lock_rejects_competing_process(monkeypatch, tmp_path):
    import os
    import subprocess
    import sys
    import tapestry_m1.runtime.devices as devices
    monkeypatch.setenv("M1_GPU_EXECUTION", "exclusive")
    monkeypatch.setenv("M1_GPU_LOCK_FILE", str(tmp_path / "gpu.lock"))
    monkeypatch.setattr(devices, "_exclusive_lock", None)
    try:
        assert devices.execution_context() == {"execution_mode": "exclusive", "job_id": None}
        assert devices.execution_context()["job_id"] is None
        result = subprocess.run([sys.executable, "-c", "from tapestry_m1.runtime.devices import execution_context; execution_context()"], capture_output=True, text=True, env=os.environ.copy())
        assert result.returncode != 0
        assert "already held" in result.stderr
    finally:
        devices._exclusive_lock.close()
        devices._exclusive_lock = None
    # The kernel releases the lock; a subsequent independent process can acquire it.
    subprocess.run([sys.executable, "-c", "from tapestry_m1.runtime.devices import execution_context; execution_context()"], check=True)


def test_exclusive_requires_explicit_lock(monkeypatch):
    from tapestry_m1.runtime.devices import execution_context
    monkeypatch.setenv("M1_GPU_EXECUTION", "exclusive")
    monkeypatch.delenv("M1_GPU_LOCK_FILE", raising=False)
    with pytest.raises(ContractError, match="absolute shared lock"):
        execution_context()


@pytest.mark.parametrize("resources,allowed", [
    ("JobState=RUNNING AllocTRES=cpu=2,gres/gpu=1", True),
    ("JobState=RUNNING TRES=cpu=2,gres/gpu=1", True),
    ("JobState=PENDING TRES=cpu=2,gres/gpu=1", False),
    ("JobState=RUNNING AllocTRES=cpu=2 TRES=gres/gpu=1", False),
    ("JobState=RUNNING TRES=cpu=2,gres/gpu=0", False),
])
def test_slurm_resource_field_versions(monkeypatch, resources, allowed):
    import tapestry_m1.runtime.devices as devices
    monkeypatch.setenv("M1_GPU_EXECUTION", "slurm")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(devices.subprocess, "check_output", lambda *a, **kw: resources)
    if allowed:
        assert devices.execution_context()["job_id"] == "123"
    else:
        with pytest.raises(ContractError, match="allocated running GPU"):
            devices.execution_context()


def test_array_guard_queries_exact_task(monkeypatch):
    import tapestry_m1.runtime.devices as devices
    monkeypatch.setenv("M1_GPU_EXECUTION", "slurm")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "123")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "8")
    def lookup(command, **kwargs):
        assert command[-1] == "123_8"
        return "JobState=RUNNING TRES=cpu=2,gres/gpu=1"
    monkeypatch.setattr(devices.subprocess, "check_output", lookup)
    assert devices.execution_context()["job_id"] == "123"
