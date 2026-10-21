"""Content-checked immutable cache shards, with explicit completion markers."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from tapestry_m1.core.io import ContractError, atomic_json, dumps, fingerprint


def identity(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def atomic_bytes(path, content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()==content:return
        raise ContractError("refusing to overwrite a different completed artifact")
    fd,tmp=tempfile.mkstemp(prefix="."+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as out:out.write(content);out.flush();os.fsync(out.fileno())
        os.link(tmp,path)
    finally:os.unlink(tmp)
