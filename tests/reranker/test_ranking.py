import copy
import pytest
from tapestry_m1.core.io import ContractError, records
from tapestry_m1.reranker.ranking import reorder, check_candidates


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
