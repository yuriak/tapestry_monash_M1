"""Acquire exactly pinned Hub files with hf CLI and verify upstream Git/LFS identity."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
from tapestry_m1.core.io import ContractError,atomic_json,dumps,fingerprint


def validate_pin(spec):
    if not isinstance(spec.get('revision'),str) or not re.fullmatch(r'[0-9a-f]{40}',spec['revision']):raise ContractError('model revision must be a commit SHA')
    if spec.get('tokenizer_revision',spec['revision'])!=spec['revision']:raise ContractError('model/tokenizer revision mismatch')


def verify_assets(spec,root):
    root=Path(root);files={}
    validate_pin(spec)
    for name,meta in spec['files'].items():
        file=root/name
        if file.parent!=root or not file.is_file():raise ContractError('missing or unsafe root model file: '+name)
        sha=hashlib.sha256();git=hashlib.sha1();git.update(b'blob '+str(file.stat().st_size).encode()+b'\0')
        with file.open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024),b''):sha.update(block);git.update(block)
        if meta['sha256']:
            if sha.hexdigest()!=meta['sha256']:raise ContractError('model upstream LFS checksum mismatch: '+name)
        elif not meta.get('git_blob') or git.hexdigest()!=meta['git_blob']:raise ContractError('model upstream Git blob mismatch: '+name)
        files[name]={'sha256':sha.hexdigest(),'bytes':file.stat().st_size,'upstream_verified':True}
    result={'model_id':spec['model_id'],'revision':spec['revision'],'files':files}
    atomic_json(root/'verified.json',result);return result


def acquire_model(spec,data_root,verify_only=False):
    validate_pin(spec)
    root=Path(data_root)/'models'/spec['model_id'].split('/')[-1]/spec['revision']
    if not verify_only:
        hf=Path(sys.executable).with_name('hf')
        subprocess.run([str(hf),'download',spec['model_id'],'--revision',spec['revision'],'--local-dir',str(root),'--include',*spec['files']],check=True)
    return verify_assets(spec,root)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path);p.add_argument('--data-root',type=Path,default=Path(os.environ.get('M1_DATA_ROOT','data')));p.add_argument('--verify-only',action='store_true')
    a=p.parse_args();result=acquire_model(json.loads(a.config.read_text()),a.data_root,a.verify_only);print(dumps(result))


if __name__=='__main__':main()


def verify_model(spec,data_root):
    root=Path(data_root)/"models"/spec["model_id"].split("/")[-1]/spec["revision"]
    saved=json.loads((root/"verified.json").read_text())
    if saved["model_id"]!=spec["model_id"] or saved["revision"]!=spec["revision"]:raise ContractError("model identity mismatch")
    for name,meta in spec["files"].items():
        actual=fingerprint(root/name)
        if actual!=saved["files"][name]["sha256"] or meta["sha256"] and actual!=meta["sha256"]:raise ContractError("model checksum mismatch")
    return root,saved
