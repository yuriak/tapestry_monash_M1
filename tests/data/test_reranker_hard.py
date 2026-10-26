import copy
import pytest
from tapestry_m1.core.io import ContractError
from tapestry_m1.data.reranker_hard import select_hard, assert_comparable


def fixture():
    docs=[{'doc_id':str(i),'split':'train','source_doc_ids':['source-'+str(i)],
           'text':'fictional passage '+str(i),'source_hash':'unused'} for i in range(10)]
    group={'split':'train','query_source_doc_ids':['source-0'],'known_relevant_ids':['0','2'],
        'candidates':[{'doc_id':str(i),'negative_kind':['positive','random','hard','hard'][i]} for i in range(4)]}
    rank=[{'doc_id':str(i),'rank':i+1,'score':1./(i+1)} for i in range(10)]
    return docs,group,rank


def test_fixed_random_known_alias_near_and_pairwise_constraints():
    docs,g,ranking=fixture();docs[4]['source_doc_ids']=['source-0']
    docs[7]['source_doc_ids']=['source-6']
    scores={str(i):float(20-i) for i in range(10)}
    result=select_hard(g,docs,ranking,scores,{'0':{'3'},'1':{'5'}})
    assert result['candidates'][:2]==g['candidates'][:2]
    assert [c['doc_id'] for c in result['candidates'][2:]]==['6','8']
    assert g['candidates'][2]['doc_id']=='2' # original is immutable


def test_tie_break_and_no_pool_expansion():
    docs,g,rank=fixture();scores={str(i):1. for i in range(10)}
    assert [c['doc_id'] for c in select_hard(g,docs,rank,scores,{})['candidates'][2:]]==['3','4']
    with pytest.raises(ContractError,match='Top-100'):
        select_hard(g,docs,rank[:3],scores,{})


def test_comparability_rejects_random_query_count_and_split_changes():
    _,g,_=fixture()
    for mutate in [lambda x:x['candidates'][1].update(doc_id='other'),lambda x:x.update(split='dev')]:
        changed=copy.deepcopy(g);mutate(changed)
        with pytest.raises(ContractError):assert_comparable([g],[changed])
    with pytest.raises(ContractError):assert_comparable([g],[])
