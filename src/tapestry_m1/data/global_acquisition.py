"""Source-native global acquisition. Stdlib only; transfers require explicit CLI invocation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import gzip
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import tarfile
import time
import tempfile
import threading
import urllib.error
import urllib.request
import zipfile
import zlib

from tapestry_m1.core.io import ContractError
from tapestry_m1.data.acquisition import utc_now


def write_json(path, value):
    """Replace mutable acquisition state atomically; core artifact writers are immutable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name + '.', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def relative(value):
    p = PurePosixPath(value)
    if not value or p.is_absolute() or '..' in p.parts or '\\' in value:
        raise ContractError('unsafe relative acquisition path')
    return p


def target(root, spec, item):
    if item['area'] not in ('raw', 'evaluation'):
        raise ContractError('unknown storage area')
    p = root / item['area'] / relative(spec['dataset_id']) / relative(spec['revision']) / relative(item['path'])
    if not p.resolve().is_relative_to(root.resolve()):
        raise ContractError('destination escapes acquisition root')
    return p


def hashes(path):
    size = path.stat().st_size
    sha = hashlib.sha256()
    blob = hashlib.sha1(b'blob ' + str(size).encode() + b'\0')
    rows = 0
    last = b''
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            sha.update(block)
            blob.update(block)
            rows += block.count(b'\n')
            last = block[-1:]
    return {'bytes': size, 'sha256': sha.hexdigest(), 'git_blob_sha1': blob.hexdigest(),
            'line_count': rows + int(bool(last) and last != b'\n')}


def check_hash(item, actual):
    for key in ('bytes', 'sha256', 'git_blob_sha1'):
        if item.get(key) is not None and item[key] != actual[key]:
            raise ContractError('source ' + key + ' mismatch')


def validate_format(path, kind):
    if kind == 'zip':
        with zipfile.ZipFile(path) as archive:
            if archive.testzip():
                raise ContractError('corrupt zip member')
    elif kind == 'pqa-a':
        with path.open() as handle:
            prefix = handle.read(256 * 1024)
        match = re.match(r'\s*\{\s*"(\d+)"\s*:\s*', prefix)
        if not match:
            raise ContractError('PQA-A must be a nonempty PMID keyed object')
        try:
            value, _ = json.JSONDecoder().raw_decode(prefix[match.end():])
        except ValueError:
            raise ContractError('PQA-A first record invalid or exceeds bounded schema check') from None
        if not isinstance(value, dict) or not {'QUESTION', 'CONTEXTS', 'final_decision'} <= value.keys():
            raise ContractError('PQA-A source schema differs')
        return {'first_pmid': match[1], 'records': None, 'split': 'PQA-A-artificial',
                'validation_scope': 'bounded first-record schema + whole-file transport hash; no complete JSON parse',
                'split_basis': 'official Drive ID in pinned README'}
    elif kind == 'mirage':
        with path.open() as handle:
            value = json.load(handle)
        if not isinstance(value, dict) or not {'mmlu', 'medqa', 'medmcqa', 'pubmedqa', 'bioasq'} <= value.keys():
            raise ContractError('unexpected medical MIRAGE benchmark families')
        return {'family_counts': {k: len(v) for k, v in value.items()}}
    elif kind == 'evidence-inference-2':
        # Inspect without extraction: all bundled labels/text stay in evaluation quarantine.
        with tarfile.open(path) as archive:
            names = [m.name for m in archive if m.isfile()]
        if not all(any(required in p.lower() for p in names)
                   for required in ('annotations_merged.csv', 'prompts_merged.csv', 'train', 'test')):
            raise ContractError('official Evidence Inference 2.0 archive is incomplete')
        if not any(re.search(r'(?:^|[_./ -])(?:dev|val|validation)(?:[_./ -]|$)', p.lower()) for p in names):
            raise ContractError('official Evidence Inference 2.0 validation split not found')
        # Read through the gzip trailer as tar iteration can stop before CRC validation.
        with gzip.open(path, 'rb') as handle:
            for _ in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                pass
        return {'archive_members': names, 'split_policy': 'native paths preserved; mixed splits quarantined; not training-ready'}
    return {}


@contextmanager
def root_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.acquisition.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ContractError('another acquisition owns this data root') from None
        yield


def transfer(item, destination, identity, attempts=5, retry_delay=2, stop_event=None):
    """Range resume with pinned hashes or saved HTTP validators, atomic verified publication."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = destination.with_name(destination.name + '.download.json')
    partial = destination.with_name(destination.name + '.part')
    sidecar = partial.with_name(partial.name + '.json')
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if destination.exists():
        if not receipt.exists():
            raise ContractError('existing destination has no receipt; preserve for manual review')
        saved = json.loads(receipt.read_text())
        actual = hashes(destination)
        if saved['identity'] != signature or saved['sha256'] != actual['sha256']:
            raise ContractError('existing destination identity/checksum changed')
        check_hash(item, actual)
        return saved | {'cache_reused': True}
    if partial.exists() and not sidecar.exists():
        raise ContractError('partial source has no identity; preserve for manual review')
    state = json.loads(sidecar.read_text()) if sidecar.exists() else {'identity': signature}
    if state['identity'] != signature:
        raise ContractError('partial source identity changed')
    write_json(sidecar, state)
    pinned = bool(item.get('sha256') or item.get('git_blob_sha1'))
    limit = item.get('bytes') if item.get('bytes') is not None else item.get('max_bytes')
    if type(limit) is not int or limit < 0 or (limit == 0 and not pinned):
        raise ContractError('file size or explicit byte ceiling required')
    for attempt in range(attempts):
        if stop_event is not None and stop_event.is_set():
            raise ContractError('transfer interrupted; partial retained')
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > limit:
                raise ContractError('partial exceeds source size ceiling')
            # A pinned empty blob is exactly known; no network body is needed.
            if limit == 0 and not partial.exists():
                with partial.open('xb'):
                    pass
            # A full interrupted part can be verified without issuing a Range at EOF.
            if limit != 0 and not (offset and offset == (item.get('bytes') or state.get('total'))):
                headers = {'User-Agent': 'tapestry-m1-acquisition/1', 'Accept-Encoding': 'identity'}
                validator = state.get('etag') or state.get('last_modified')
                if offset:
                    if not pinned and not validator:
                        raise ContractError('unversioned partial lacks HTTP validator; manual review required')
                    headers['Range'] = f'bytes={offset}-'
                    if validator:
                        headers['If-Range'] = validator
                request = urllib.request.Request(item['url'], headers=headers)
                with urllib.request.urlopen(request, timeout=120) as response:
                    if 'text/html' in response.headers.get('Content-Type', '').lower():
                        raise ContractError('blocked_access: HTML/login/Drive confirmation; no dataset accepted')
                    etag = response.headers.get('ETag')
                    if etag and etag.startswith('W/'):
                        etag = None
                    modified = response.headers.get('Last-Modified')
                    if offset and response.status == 200:
                        # Server ignored Range or If-Range detected a changed release.
                        if not pinned and validator and validator not in (etag, modified):
                            raise ContractError('mutable upstream changed during resume; retained partial')
                        offset = 0
                    elif offset:
                        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                        if response.status != 206 or not match or int(match[1]) != offset or int(match[2]) + 1 != int(match[3]):
                            raise ContractError('invalid Range response')
                        if not pinned and validator not in (etag, modified):
                            raise ContractError('resume validator differs from original response')
                    elif response.status != 200:
                        raise ContractError('unexpected HTTP response')
                    length = response.headers.get('Content-Length')
                    total = offset + int(length) if length is not None else None
                    if total and (total > limit or (item.get('bytes') is not None and total != item['bytes'])):
                        raise ContractError('HTTP size differs from selected source')
                    state.update(etag=etag, last_modified=modified, total=total)
                    write_json(sidecar, state)
                    written = offset
                    with partial.open('ab' if offset else 'wb') as handle:
                        for block in iter(lambda: response.read(1024 * 1024), b''):
                            if stop_event is not None and stop_event.is_set():
                                raise ContractError('transfer interrupted; partial retained')
                            written += len(block)
                            if written > limit:
                                raise ContractError('response exceeds source size ceiling')
                            handle.write(block)
                        handle.flush()
                        os.fsync(handle.fileno())
                    if total is not None and written != total:
                        raise http.client.IncompleteRead(b'')
            actual = hashes(partial)
            check_hash(item, actual)
            detail = validate_format(partial, item.get('format'))
            saved = actual | {'identity': signature, 'verified_at': utc_now(), 'validation': detail,
                              'source_url': item['url'], 'etag': state.get('etag'),
                              'last_modified': state.get('last_modified'),
                              'upstream_checksum_verified': pinned, 'resumed_bytes': offset}
            if limit == 0:
                saved['materialization'] = 'verified_pinned_empty_blob_no_network'
            # Store receipt first so a crash after atomic rename is recoverable.
            write_json(receipt, saved)
            os.replace(partial, destination)
            sidecar.unlink(missing_ok=True)
            return saved
        except ContractError as exc:
            if str(exc).startswith('source ') and str(exc).endswith(' mismatch') and partial.exists():
                # Preserve corrupt bytes as evidence, then permit a clean bounded retry.
                partial.rename(partial.with_name(partial.name + '.corrupt.' + str(time.time_ns())))
                state = {'identity': signature}
                write_json(sidecar, state)
                if attempt + 1 < attempts:
                    continue
            raise
        except (urllib.error.URLError, http.client.HTTPException, socket.timeout, ConnectionError, OSError) as exc:
            # Do not log redirected signed URLs or credentials from exception text.
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404):
                raise ContractError(f'blocked_access: HTTP {exc.code}') from None
            if attempt + 1 == attempts:
                raise ContractError('transfer failed after retries: ' + type(exc).__name__) from None
            print(f'retrying {destination.name}: {attempt + 1}/{attempts}', flush=True)
            time.sleep(min(retry_delay * 2**attempt, 30))
    raise AssertionError('unreachable')


def acquire_dataset(spec, root, workers, attempts):
    relative(spec['dataset_id']); relative(spec['revision'])
    manifest_dir = root / 'manifests' / spec['dataset_id'] / spec['revision']
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_dir / 'source.json', spec)
    result = {k: spec[k] for k in ('dataset_id', 'canonical_url', 'revision', 'rights')}
    result.update(planned_files=spec['files'], state='partial', files=[], next_action='Run the same batch command to resume.')
    if spec.get('state') in ('blocked_rights', 'blocked_access', 'not_found'):
        result.update(state=spec['state'], next_action=spec['next_action'])
        write_json(manifest_dir / 'status.json', result)
        return result
    stop_event = threading.Event()
    def one(item):
        path = target(root, spec, item)
        if item.get('state') == 'blocked_rights':
            return {k: item[k] for k in ('path', 'role', 'split')} | {
                'relative_path': path.relative_to(root).as_posix(), 'state': 'blocked_rights',
                'error': item['next_action']}
        identity = {'dataset': spec['dataset_id'], 'revision': spec['revision'], 'file': item}
        try:
            receipt = transfer(item, path, identity, attempts=attempts, stop_event=stop_event)
            return {k: item[k] for k in ('path', 'role', 'split')} | {
                'relative_path': path.relative_to(root).as_posix(), 'state': 'downloaded_verified', **receipt}
        except (ContractError, ValueError, OSError, EOFError, zlib.error, tarfile.TarError, zipfile.BadZipFile) as exc:
            message = str(exc) if isinstance(exc, ContractError) else type(exc).__name__
            return {k: item[k] for k in ('path', 'role', 'split')} | {
                'relative_path': path.relative_to(root).as_posix(), 'state': 'blocked_access' if message.startswith('blocked_access') else 'partial',
                'error': message}
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(one, item) for item in spec['files']]
        for future in as_completed(futures):
            record = future.result()
            result['files'].append(record)
            print(f"{spec['dataset_id']} {len(result['files'])}/{len(futures)} {record['path']}: {record['state']}", flush=True)
            write_json(manifest_dir / 'status.json', result)
    except BaseException:
        stop_event.set()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    result['files'].sort(key=lambda f: f['path'])
    result['verified_files'] = sum(f['state'] == 'downloaded_verified' for f in result['files'])
    result['verified_bytes'] = sum(f.get('bytes', 0) for f in result['files'] if f['state'] == 'downloaded_verified')
    if result['verified_files'] == len(spec['files']):
        result.update(state='downloaded_verified', next_action='Retain original files and attribution; no training/normalization implied.')
    elif all(f['state'] == 'blocked_access' for f in result['files']):
        result['state'] = 'blocked_access'
    if any(f['state'] == 'blocked_rights' for f in result['files']):
        result['next_action'] = 'Review blocked component rights; rerunning will resume other files but will not bypass the hold.'
    write_json(manifest_dir / 'status.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--attempts', type=int, default=5)
    parser.add_argument('--plan', action='store_true', help='Offline file selection only; no writes or requests')
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16 or not 1 <= args.attempts <= 10:
        parser.error('workers must be 1..16 and attempts 1..10')
    manifest = json.loads(args.manifest.read_text())
    datasets = manifest['datasets']
    root = args.data_root.resolve()
    seen = set()
    for spec in datasets:
        for item in spec['files']:
            path = target(root, spec, item)
            if path in seen:
                raise ContractError('duplicate destination in manifest')
            seen.add(path)
    if args.plan:
        print(json.dumps({'batch': manifest['batch'], 'transfer_started': False,
                          'datasets': [{'dataset_id': s['dataset_id'], 'revision': s['revision'],
                           'state': s.get('state', 'awaiting_manual_download'), 'files': len(s['files']),
                           'known_bytes': sum(f.get('bytes') or 0 for f in s['files'])} for s in datasets]}, indent=2))
        return 0
    with root_lock(root):
        for directory in ('raw', 'evaluation', 'manifests', 'logs', 'prepared'):
            (root / directory).mkdir(exist_ok=True)
        catalog_path = root / 'manifests' / 'catalog.json'
        catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else {'schema_version': 1, 'datasets': {}}
        for spec in datasets:
            result = acquire_dataset(spec, root, args.workers, args.attempts)
            catalog['datasets'][spec['dataset_id']] = result
            catalog['updated_at'] = utc_now()
            write_json(catalog_path, catalog)
        states = [catalog['datasets'][s['dataset_id']]['state'] for s in datasets]
        # Rights/access blocks are explicit accepted exceptions, not verified copies.
        return 2 if 'partial' in states else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Acquisition interrupted; rerun the same command to resume.', flush=True)
        raise SystemExit(130)
    except (ContractError, OSError, ValueError) as exc:
        print('Acquisition stopped: ' + (str(exc) if isinstance(exc, ContractError) else type(exc).__name__), flush=True)
        raise SystemExit(2)
