import copy
import json
import numpy as np
import pytest
import torch
from tapestry_m1.io import ContractError,records
from tapestry_m1.inference.artifacts import VectorCache,identity,atomic_bytes
from tapestry_m1.inference.retrieval import exact_topk,reorder,check_candidates
from tapestry_m1.inference.qwen import last_token_pool,yes_probability,embedding_query,reranker_text
from tapestry_m1.inference.runtime import gpu_runtime


def test_exact_top100_and_ties():
    names=[f'c::{i:03d}' for i in range(105)][::-1]
    vectors=np.ones((105,2));result=exact_topk(names,vectors,[[1,1]])[0]
    assert len(result)==100 and len({r['doc_id'] for r in result})==100
    assert [r['doc_id'] for r in result]==sorted(names)[:100]
    assert result[0]['score']==pytest.approx(1,abs=1e-6)
    assert exact_topk(['c::x'],[[2,0]],[[1,0]])[0][0]['doc_id']=='c::x'
    assert exact_topk(['c::a','c::b'],[[1,0],[0,1]],[[0,2]])[0][0]['doc_id']=='c::b'


@pytest.mark.parametrize('names,docs,queries,k', [(['x','x'],[[1],[2]],[[1]],100),(['x'],[[0,0]],[[1,1]],100),(['x'],[[float('nan')]],[[1]],100),(['x'],[[1]],[[float('inf')]],100),(['x'],[[1,2]],[[1]],100),(['x'],[[1]],[[0]],100),(['x'],[[1]],[[1]],0),(['x'],[[1],[2]],[[1]],100)])
def test_bad_vectors(names,docs,queries,k):
    with pytest.raises(ContractError):exact_topk(names,docs,queries,k)


def test_rerank_preserves_set_and_retrieval_provenance(rankings):
    a=list(records(rankings));b=copy.deepcopy(a)
    for r in b:r['candidates']=reorder(r['candidates'],[.1,.9])
    assert check_candidates(a,b)['candidate_sets_identical']
    assert b[0]['candidates'][0]['retrieval_rank']==2
    b[0]['candidates'][0]['doc_id']='c::absent'
    with pytest.raises(ContractError):check_candidates(a,b)
    with pytest.raises(ContractError):check_candidates(a,b[:1])
    with pytest.raises(ContractError):reorder(a[0]['candidates'],[.1])
    with pytest.raises(ContractError):reorder(a[0]['candidates'],[float('nan'),1])


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


def test_atomic_outputs(tmp_path):
    f=tmp_path/'out';atomic_bytes(f,b'one');atomic_bytes(f,b'one')
    with pytest.raises(ContractError):atomic_bytes(f,b'two')
    assert identity({'a':1,'b':2})==identity({'b':2,'a':1})


def test_official_pool_and_logits_math_cpu():
    hidden=torch.arange(24).reshape(2,3,4)
    right=torch.tensor([[1,1,0],[1,0,0]])
    assert torch.equal(last_token_pool(hidden,right),torch.stack([hidden[0,1],hidden[1,0]]))
    left=torch.tensor([[0,1,1],[1,1,1]])
    assert torch.equal(last_token_pool(hidden,left),hidden[:,-1])
    logits=torch.tensor([[2.,4.,100.]])
    assert yes_probability(logits,0,1).item()==pytest.approx(1/(1+np.exp(-2)),abs=1e-6)
    with pytest.raises(ContractError):yes_probability(logits,1,1)
    with pytest.raises(ContractError):yes_probability(torch.tensor([[float("nan"),0.]]),0,1)
    with pytest.raises(ContractError):last_token_pool(hidden,torch.zeros((2,3)))
    assert embedding_query('TASK','Q')=='Instruct: TASK\nQuery:Q'
    assert reranker_text('TASK','Q','D')=='<Instruct>: TASK\n<Query>: Q\n<Document>: D'


def test_gpu_guard_no_allocation(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(ContractError,match='scheduler'):gpu_runtime()


def test_compatibility_ties_are_a_separate_view(benchmark,rankings,tmp_path):
    from tapestry_m1.evaluation.compatibility import compatibility_view
    original=rankings.read_bytes()
    result=compatibility_view(benchmark,rankings,tmp_path/'compatibility')
    assert result['definition']['gain']=='linear' and result['definition']['mrr_depth']==100
    assert rankings.read_bytes()==original
    assert next(records(tmp_path/'compatibility/rankings.jsonl'))['candidates'][0]['doc_id']=='c::b'
