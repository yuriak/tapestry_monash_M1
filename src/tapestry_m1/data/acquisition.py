"""Pinned, bounded HTTP acquisition with reusable verified files and Range resume."""
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
import urllib.request
from tapestry_m1.io import ContractError, atomic_json, fingerprint


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "tapestry-m1/0.1"})
    with urllib.request.urlopen(req, timeout=60) as response:
        body = response.read(8 * 1024 * 1024 + 1)
    if len(body) > 8 * 1024 * 1024:
        raise ContractError("metadata response too large")
    return json.loads(body)


def download(url, destination, max_bytes, expected_sha256=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = destination.with_name(destination.name + ".download.json")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ContractError("download byte budget must be positive")
    if destination.exists():
        actual = fingerprint(destination)
        if metadata.exists():
            saved = json.loads(metadata.read_text())
            if saved["url"] != url or saved["sha256"] != actual:
                raise ContractError("cached source identity/checksum differs; do not overwrite")
            if expected_sha256 and actual != expected_sha256:
                raise ContractError("cached source differs from official checksum")
            return saved | {"cache_reused": True}
        if expected_sha256 and actual == expected_sha256:
            saved = {"url": url, "sha256": actual, "bytes": destination.stat().st_size,
                     "downloaded_at": utc_now(), "recovered_verified_source": True}
            atomic_json(metadata, saved)
            return saved
        raise ContractError("existing source has no verified metadata; refusing to overwrite")
    partial = destination.with_name(destination.name + ".part")
    sidecar = partial.with_name(partial.name + ".json")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > max_bytes:
        raise ContractError("partial input already exceeds current byte budget")
    if sidecar.exists():
        if json.loads(sidecar.read_text())["url"] != url:
            raise ContractError("partial input belongs to another source")
    else:
        if offset:
            raise ContractError("partial input missing identity metadata")
        atomic_json(sidecar, {"url": url})
    headers = {"User-Agent": "tapestry-m1/0.1", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as response:
        status = response.status
        if offset and (status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
            raise ContractError("server did not honor Range; retained partial source, no silent restart")
        if not offset and status != 200:
            raise ContractError("unexpected download response")
        total = None
        if offset:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match or int(match[1]) != offset or int(match[2]) + 1 != int(match[3]):
                raise ContractError("Range response does not cover the complete remaining source")
            total = int(match[3])
            if total > max_bytes:
                raise ContractError("source exceeds declared byte budget")
        remaining = response.headers.get("Content-Length")
        if remaining is not None and offset + int(remaining) > max_bytes:
            raise ContractError("source exceeds declared byte budget")
        written = offset
        with partial.open("ab" if offset else "wb") as out:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                if written + len(block) > max_bytes:
                    raise ContractError("download stopped at byte budget; partial source retained")
                out.write(block)
                written += len(block)
            out.flush()
            os.fsync(out.fileno())
        if (remaining is not None and written - offset != int(remaining)) or (total is not None and written != total):
            raise ContractError("truncated source response; partial input retained")
    actual = fingerprint(partial)
    if expected_sha256 and actual != expected_sha256:
        raise ContractError("download differs from official SHA-256; no final artifact published")
    saved = {"url": url, "sha256": actual, "bytes": written, "downloaded_at": utc_now(),
             "resumed_bytes": offset, "official_sha256_verified": bool(expected_sha256)}
    os.link(partial, destination)
    partial.unlink()
    atomic_json(metadata, saved)
    sidecar.unlink()
    return saved


def pinned_info(spec):
    info = fetch_json(f"https://huggingface.co/api/datasets/{spec['repo_id']}/revision/{spec['revision']}")
    if info.get("sha") != spec["revision"]:
        raise ContractError("upstream did not resolve the pinned dataset revision")
    return info


def acquire(spec, data_root, max_bytes):
    if not spec["acquisition_allowed"]:
        raise ContractError("blocked_pending_license_review: source is metadata-only")
    info = pinned_info(spec)
    if info.get("gated"):
        raise ContractError("gated source; obtain access explicitly, no bypass")
    root = Path(data_root) / "raw" / spec["dataset_id"] / spec["revision"]
    output = {}
    for name in spec["files"]:
        url = f"https://huggingface.co/datasets/{spec['repo_id']}/resolve/{spec['revision']}/{name}"
        output[name] = download(url, root / name, max_bytes, spec.get("file_sha256", {}).get(name))
    return output
