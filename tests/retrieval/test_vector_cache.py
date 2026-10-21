import json
import pytest
pytest.importorskip("numpy")
from tapestry_m1.core.io import ContractError
from tapestry_m1.retrieval.vector_cache import VectorCache


def test_cache_resume_identity_and_tampering(tmp_path):
    recipe={'model':'sha','prompt':'frozen','input':'source-sha'}
    cache=VectorCache(tmp_path,recipe,['a','b','c'],2,2)
    assert cache.read(0) is None
    cache.write(0,[[1,0],[0,1]],[{'tokens':1},{'tokens':2}])
    with pytest.raises(ContractError,match='incomplete'):cache.finish()
    resumed=VectorCache(tmp_path,recipe,['a','b','c'],2,2)
    assert resumed.read(0)[0].shape==(2,2)
    resumed.write(2,[[1,1]],[{'tokens':3}]);array,lengths,manifest=resumed.finish()
    assert array.shape==(3,2) and manifest['complete']
    assert VectorCache(tmp_path,recipe|{'prompt':'changed'},['a','b','c'],2,2).key!=cache.key
    assert VectorCache(tmp_path,recipe,['b','a','c'],2,2).key!=cache.key
    entry=json.loads((cache.root/'batch-00000000.json').read_text())
    (cache.root/entry['array']).write_bytes(b'tampered')
    with pytest.raises(ContractError,match='checksum'):cache.read(0)


def test_orphan_array_is_not_completed_cache(tmp_path):
    cache=VectorCache(tmp_path,{'x':1},['a'],2,1)
    (cache.root/'batch-00000000-orphan.npy').write_bytes(b'interrupted')
    assert cache.read(0) is None
    cache.write(0,[[1,0]],[{'tokens':1}]);assert cache.finish()[2]['complete']
    with pytest.raises(ContractError):cache.write(0,[[0,0]],[{}])
    with pytest.raises(ContractError):VectorCache(tmp_path,{},['a','a'],2,1)
