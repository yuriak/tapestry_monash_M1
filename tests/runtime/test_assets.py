import json
import hashlib
import pytest
from tapestry_m1.runtime.assets import verify_assets
from tapestry_m1.core.io import ContractError


def test_upstream_blob_and_lfs_verification(tmp_path):
    data=b'synthetic model configuration';(tmp_path/'config.json').write_bytes(data)
    git=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    spec={'model_id':'synthetic/model','revision':'a'*40,'files':{'config.json':{'sha256':None,'git_blob':git}}}
    result=verify_assets(spec,tmp_path)
    assert result==verify_assets(spec,tmp_path) and result['files']['config.json']['upstream_verified']
    (tmp_path/'config.json').write_bytes(b'changed')
    with pytest.raises(ContractError,match='Git blob'):verify_assets(spec,tmp_path)
    spec['files']['config.json']['sha256']='0'*64
    with pytest.raises(ContractError,match='LFS checksum'):verify_assets(spec,tmp_path)


def test_floating_revision_is_rejected_before_network(tmp_path,monkeypatch):
    from tapestry_m1.runtime.assets import acquire_model
    monkeypatch.setattr('subprocess.run',lambda *a,**k:pytest.fail('must not download floating main'))
    with pytest.raises(ContractError,match='commit SHA'):acquire_model({'model_id':'synthetic/model','revision':'main'},tmp_path)


def test_runtime_verifier_uses_asset_owner_and_checks_local_hash(tmp_path):
    from tapestry_m1.runtime.assets import verify_model
    from tapestry_m1.core.io import fingerprint
    model = tmp_path / "models" / "synthetic" / "revision"
    model.mkdir(parents=True)
    file = model / "weights.bin"
    file.write_bytes(b"synthetic-not-a-model")
    spec = {"model_id":"test/synthetic", "revision":"revision", "files":{"weights.bin":{"sha256":None}}}
    saved = {"model_id":spec["model_id"], "revision":spec["revision"], "files":{"weights.bin":{"sha256":fingerprint(file)}}}
    (model/"verified.json").write_text(json.dumps(saved))
    assert verify_model(spec,tmp_path) == (model,saved)
    file.write_bytes(b"changed")
    with pytest.raises(ContractError, match="checksum"):
        verify_model(spec,tmp_path)
