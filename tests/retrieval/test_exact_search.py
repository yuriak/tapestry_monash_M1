import pytest
np = pytest.importorskip("numpy")
from tapestry_m1.core.io import ContractError
from tapestry_m1.retrieval.search import exact_topk


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
