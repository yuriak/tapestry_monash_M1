"""Source-tree implementation identity for new runs, independent of installation."""
from pathlib import Path
from tapestry_m1.core.io import fingerprint
from tapestry_m1.core.artifacts import identity


def code_fingerprint():
    root = Path(__file__).resolve().parents[1]
    return identity({str(p.relative_to(root)): fingerprint(p) for p in sorted(root.rglob("*.py")) if not any(part.startswith(".") for part in p.relative_to(root).parts)})
