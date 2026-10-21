from pathlib import Path
from tapestry_m1.runtime import provenance


def test_implementation_identity_excludes_editor_state(monkeypatch, tmp_path):
    root = tmp_path / "source"
    runtime = root / "runtime"
    runtime.mkdir(parents=True)
    path = runtime / "provenance.py"
    path.write_text("# synthetic runtime source\n")
    code = root / "component.py"
    code.write_text("value = 1\n")
    monkeypatch.setattr(provenance, "__file__", str(path))
    first = provenance.code_fingerprint()
    editor = root / ".ipynb_checkpoints"
    editor.mkdir()
    (editor / "component-checkpoint.py").write_text("value = 99\n")
    assert provenance.code_fingerprint() == first
    code.write_text("value = 2\n")
    assert provenance.code_fingerprint() != first
