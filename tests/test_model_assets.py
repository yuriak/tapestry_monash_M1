import hashlib
import pytest
from tapestry_m1.model_assets import verify_assets
from tapestry_m1.io import ContractError


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
    from tapestry_m1.model_assets import acquire_model
    monkeypatch.setattr('subprocess.run',lambda *a,**k:pytest.fail('must not download floating main'))
    with pytest.raises(ContractError,match='commit SHA'):acquire_model({'model_id':'synthetic/model','revision':'main'},tmp_path)
