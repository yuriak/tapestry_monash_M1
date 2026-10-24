"""Offline transport checks: all HTTP payloads below are synthetic localhost fixtures."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import threading

import pytest
from tapestry_m1.core.io import ContractError
from tapestry_m1.data.global_acquisition import hashes, main, root_lock, target, transfer


@pytest.fixture
def source():
    payload = b'first synthetic row\nsecond synthetic row\n'
    state = {'requests': [], 'cut': False, 'ignore_range': False, 'html': False, 'corrupt': False}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            offset = int(self.headers.get('Range', 'bytes=0-').split('=')[1].split('-')[0])
            state['requests'].append(offset)
            body = b'x' * len(payload) if state['corrupt'] else payload
            if state['ignore_range']:
                offset = 0
            self.send_response(206 if offset else 200)
            self.send_header('ETag', '"synthetic-v1"')
            self.send_header('Content-Type', 'text/html' if state['html'] else 'application/octet-stream')
            self.send_header('Content-Length', str(len(body) - offset))
            if offset:
                self.send_header('Content-Range', f'bytes {offset}-{len(body)-1}/{len(body)}')
            self.end_headers()
            if state['cut']:
                state['cut'] = False
                self.wfile.write(body[:10])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
            else:
                self.wfile.write(body[offset:])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    item = {'url': f'http://127.0.0.1:{server.server_port}/fixture', 'bytes': len(payload),
            'sha256': hashlib.sha256(payload).hexdigest()}
    yield item, payload, state
    server.shutdown()
    thread.join()
    server.server_close()


def test_resume_after_interruption_and_cache_verification(source, tmp_path):
    item, payload, state = source
    state['cut'] = True
    dest = tmp_path / 'source.jsonl'
    transfer(item, dest, item, retry_delay=0)
    assert dest.read_bytes() == payload
    assert state['requests'] == [0, 10]
    assert transfer(item, dest, item)['cache_reused']
    dest.write_bytes(b'changed')
    with pytest.raises(ContractError, match='checksum changed'):
        transfer(item, dest, item)


def test_server_ignoring_range_restarts_without_append(source, tmp_path):
    item, payload, state = source
    state.update(cut=True, ignore_range=True)
    dest = tmp_path / 'source'
    transfer(item, dest, item, retry_delay=0)
    assert dest.read_bytes() == payload
    assert state['requests'] == [0, 10]


def test_complete_part_after_crash_needs_no_http(source, tmp_path):
    item, payload, state = source
    dest = tmp_path / 'source'
    dest.with_name('source.part').write_bytes(payload)
    signature = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
    dest.with_name('source.part.json').write_text(json.dumps({'identity': signature}))
    transfer(item, dest, item)
    assert dest.read_bytes() == payload
    assert not state['requests']


def test_git_blob_pin_and_checksum_rejection(source, tmp_path):
    item, payload, state = source
    item.pop('sha256')
    item['git_blob_sha1'] = hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()
    transfer(item, tmp_path / 'valid', item)
    state['corrupt'] = True
    with pytest.raises(ContractError, match='git_blob_sha1 mismatch'):
        transfer(item, tmp_path / 'invalid', item, attempts=2)
    assert not (tmp_path / 'invalid').exists()
    assert len(list(tmp_path.glob('invalid.part.corrupt.*'))) == 2


def test_html_is_never_published(source, tmp_path):
    item, _, state = source
    state['html'] = True
    with pytest.raises(ContractError, match='blocked_access'):
        transfer(item, tmp_path / 'not-a-dataset', item)
    assert not (tmp_path / 'not-a-dataset').exists()


def test_unversioned_archive_uses_saved_validator(source, tmp_path):
    item, payload, state = source
    item.pop('sha256')
    state['cut'] = True
    dest = tmp_path / 'archive'
    transfer(item, dest, item, retry_delay=0)
    assert dest.read_bytes() == payload
    assert state['requests'] == [0, 10]


def test_changed_identity_preserves_partial(source, tmp_path):
    item, _, _ = source
    dest = tmp_path / 'source'
    (tmp_path / 'source.part').write_bytes(b'old')
    (tmp_path / 'source.part.json').write_text('{"identity":"different"}')
    with pytest.raises(ContractError, match='identity changed'):
        transfer(item, dest, item)
    assert (tmp_path / 'source.part').read_bytes() == b'old'


def test_missing_receipt_and_duplicate_lock(tmp_path):
    dest = tmp_path / 'old'
    dest.write_text('user file')
    with pytest.raises(ContractError, match='no receipt'):
        transfer({}, dest, {})
    with root_lock(tmp_path):
        with pytest.raises(ContractError, match='another acquisition'):
            with root_lock(tmp_path):
                pass


def test_target_rejects_traversal_and_symlink_escape(tmp_path):
    spec = {'dataset_id': 'dataset', 'revision': 'abc'}
    item = {'area': 'raw', 'path': '../outside'}
    with pytest.raises(ContractError):
        target(tmp_path / 'root', spec, item)
    (tmp_path / 'root').mkdir()
    (tmp_path / 'root/raw').symlink_to(tmp_path)
    item['path'] = 'file'
    with pytest.raises(ContractError, match='escapes'):
        target(tmp_path / 'root', spec, item)


def test_plan_is_offline_and_does_not_create_storage(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'batch': 1, 'datasets': [{'dataset_id': 'fixture', 'revision': 'abc',
        'files': [{'area': 'raw', 'path': 'file', 'bytes': 1}]}]}))
    root = tmp_path / 'absent'
    assert main(['--manifest', str(manifest), '--data-root', str(root), '--plan']) == 0
    assert not root.exists()


def test_blocked_catalog_and_partial_never_report_complete(source, tmp_path):
    item, _, state = source
    state['html'] = True
    item.update(area='evaluation', path='test.json', role='evaluation-only', split='test')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'batch': 1, 'datasets': [
        {'dataset_id': 'blocked', 'canonical_url': 'https://example.org', 'revision': 'abc',
         'rights': 'unresolved', 'files': [], 'state': 'blocked_rights', 'next_action': 'review terms'},
        {'dataset_id': 'fixture', 'canonical_url': 'https://example.org', 'revision': 'abc',
         'rights': 'synthetic', 'files': [item]},
    ]}))
    root = tmp_path / 'root'
    assert main(['--manifest', str(manifest), '--data-root', str(root)]) == 0
    catalog = json.loads((root / 'manifests/catalog.json').read_text())['datasets']
    assert catalog['blocked']['state'] == 'blocked_rights'
    assert catalog['fixture']['state'] == 'blocked_access'
    assert not (root / 'evaluation/fixture/abc/test.json').exists()


def test_pubmed_manifest_complete_and_excludes_indices():
    project = Path(__file__).resolve().parents[2]
    manifest = json.loads((project / 'configs/data/acquisition/global_batch2.json').read_text())
    spec = manifest['datasets'][0]
    shards = [f for f in spec['files'] if f['role'] == 'index-only']
    assert len(shards) == spec['selection']['shards'] == 1166
    assert sum(f['bytes'] for f in shards) == spec['selection']['bytes'] == 70007692549
    assert all(f['path'].startswith('chunk/') and f['path'].endswith('.jsonl') for f in shards)
    assert all(f['sha256'] or f['git_blob_sha1'] for f in shards)


def test_mixed_file_results_force_partial_exit(source, tmp_path):
    item, _, _ = source
    broken = item | {'area': 'evaluation', 'path': 'bad.jsonl', 'role': 'evaluation-only',
                     'split': 'test', 'sha256': '0' * 64}
    good = item | {'area': 'raw', 'path': 'good.jsonl', 'role': 'index-only', 'split': 'corpus'}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'batch': 2, 'datasets': [{'dataset_id': 'fixture',
        'canonical_url': 'https://example.org', 'revision': 'abc', 'rights': 'synthetic',
        'files': [good, broken]}]}))
    root = tmp_path / 'root'
    assert main(['--manifest', str(manifest), '--data-root', str(root), '--attempts', '1']) == 2
    status = json.loads((root / 'manifests/fixture/abc/status.json').read_text())
    assert status['state'] == 'partial'
    assert status['verified_files'] == 1
    assert not (root / 'evaluation/fixture/abc/bad.jsonl').exists()


def test_native_format_checks_are_bounded_and_do_not_extract(tmp_path):
    import io
    import tarfile
    from tapestry_m1.data.global_acquisition import validate_format
    pqa = tmp_path / 'pqa.json'
    pqa.write_text(json.dumps({'123': {'QUESTION': 'synthetic?', 'CONTEXTS': ['synthetic'], 'final_decision': 'yes'}}))
    assert validate_format(pqa, 'pqa-a')['first_pmid'] == '123'
    mirage = tmp_path / 'mirage.json'
    mirage.write_text(json.dumps({name: [] for name in ['mmlu', 'medqa', 'medmcqa', 'pubmedqa', 'bioasq']}))
    assert len(validate_format(mirage, 'mirage')['family_counts']) == 5
    archive = tmp_path / 'ei.tar.gz'
    with tarfile.open(archive, 'w:gz') as handle:
        for name in ['annotations_merged.csv', 'prompts_merged.csv', 'splits/train.txt', 'splits/validation.txt', 'splits/test.txt']:
            member = tarfile.TarInfo(name)
            member.size = 3
            handle.addfile(member, io.BytesIO(b'id\n'))
    assert len(validate_format(archive, 'evidence-inference-2')['archive_members']) == 5
    assert not (tmp_path / 'splits').exists()


def test_cancelled_transfer_never_starts_http(source, tmp_path):
    item, _, state = source
    stop = threading.Event()
    stop.set()
    with pytest.raises(ContractError, match='interrupted'):
        transfer(item, tmp_path / 'cancelled', item, stop_event=stop)
    assert not state['requests']
