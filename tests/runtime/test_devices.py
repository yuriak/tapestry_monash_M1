import pytest
from tapestry_m1.core.io import ContractError
from tapestry_m1.runtime.devices import gpu_runtime


def test_gpu_guard_no_allocation(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(ContractError,match='scheduler'):gpu_runtime()
