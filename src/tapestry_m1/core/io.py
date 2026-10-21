"""Streaming records and atomic, content-checked artifacts."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unicodedata


class ContractError(ValueError):
    pass


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def records(path):
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("record must be an object")
            except (ValueError, TypeError) as exc:
                raise ContractError(f"{Path(path).name}:{lineno}: {exc}") from exc
            yield row


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (dumps(value) + "\n").encode()
    if path.exists():
        if path.read_bytes() == content:
            return
        raise FileExistsError(f"refusing to overwrite different artifact: {path.name}")
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(content)
            out.flush()
            os.fsync(out.fileno())
        # Hard-link finalization is exclusive, unlike replace (concurrent writers cannot overwrite).
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def text_key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())
