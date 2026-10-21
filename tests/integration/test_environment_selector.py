"""Check profile routing before uv runs or creates any environment."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/run", root / "scripts/run")
    uv = tmp_path / "fake-uv"
    uv.write_text("#!" + sys.executable + "\nimport json,os,sys\nif sys.argv[1:]==['--version']: print('uv 0.13.1')\nelse: print(json.dumps({'args':sys.argv[1:],'environment':{k:os.environ.get(k) for k in ['UV_PROJECT_ENVIRONMENT','M1_ENVIRONMENT_LOCK','M1_ENVIRONMENT','PYTHONPATH','CONDA_PREFIX','VIRTUAL_ENV']}}))\n")
    uv.chmod(0o755)
    env = os.environ | {"M1_UV_BIN": str(uv), "M1_ENVIRONMENT": "shared", "CONDA_PREFIX": "inherited-invalid", "VIRTUAL_ENV": "inherited-invalid", "PYTHONPATH": "inherited-invalid"}
    return root, env


@pytest.mark.parametrize("profile,folder", [("shared", ".venv"), ("cpu", ".venvs/cpu")])
def test_shared_cpu_source_execution(checkout, profile, folder):
    root, env = checkout
    p = subprocess.run([str(root / "scripts/run"), "--environment", profile, "data", "--help"], env=env, capture_output=True, text=True, check=True)
    result = json.loads(p.stdout)
    assert result["environment"]["UV_PROJECT_ENVIRONMENT"] == str(root / folder)
    assert result["environment"]["M1_ENVIRONMENT_LOCK"] == str(root / "uv.lock")
    assert result["environment"]["PYTHONPATH"] == str(root / "src")
    assert result["environment"]["CONDA_PREFIX"] is None
    assert result["environment"]["VIRTUAL_ENV"] is None
    assert result["args"][-5:] == ["python", "-m", "tapestry_m1", "data", "--help"]
    assert ("--no-default-groups" in result["args"]) == (profile == "cpu")


def test_independent_profile_requires_own_lock(checkout):
    root, env = checkout
    command = [str(root / "scripts/run"), "--environment", "synthetic-stack", "--help"]
    assert subprocess.run(command, env=env, capture_output=True).returncode == 2
    project = root / "environments/synthetic-stack"
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text("# synthetic config")
    assert subprocess.run(command, env=env, capture_output=True).returncode == 2
    (project / "uv.lock").write_text("# synthetic lock")
    p = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    result = json.loads(p.stdout)
    assert result["environment"]["M1_ENVIRONMENT_LOCK"] == str(project / "uv.lock")
    assert result["args"][result["args"].index("--project") + 1] == str(project)


@pytest.mark.parametrize("args", [["--environment"], ["--environment", "../outside"], ["--environment", "/absolute"], ["--environment", ""]])
def test_invalid_profile_fails_before_sync(checkout, args):
    root, env = checkout
    assert subprocess.run([str(root / "scripts/run"), *args], env=env, capture_output=True).returncode == 2
