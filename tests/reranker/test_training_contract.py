import copy
import json
import math
from pathlib import Path
import pytest
from tapestry_m1.core.io import ContractError, records
from tapestry_m1.contracts.reranker_training import validate_groups, reference_loss, select_negatives
from tapestry_m1.reranker.batches import GroupSampler

FIXTURE = Path(__file__).parents[1]/'fixtures/training'


def test_reference_values_and_finite_differences():
    expected = json.loads((FIXTURE/'expected.json').read_text())
    for objective in ['L0', 'L1']:
        observed = reference_loss(expected['score_groups'],objective)
        assert observed == expected[objective]
        for i in range(2):
            for j in range(4):
                left, right = copy.deepcopy(expected['score_groups']), copy.deepcopy(expected['score_groups'])
                left[i][j] -= 1e-6; right[i][j] += 1e-6
                derivative=(reference_loss(right,objective)['total']-reference_loss(left,objective)['total'])/2e-6
                assert derivative == pytest.approx(observed['gradients_wrt_score'][i][j],abs=1e-8)
    assert max(0,1-1.2+.4) == pytest.approx(.2)
    assert math.isfinite(reference_loss([[1000,-1000,1000,-1000]])['total'])


def test_group_shape_identity_supervision_and_test_exclusion():
    groups=list(records(FIXTURE/'groups.jsonl'))
    validate_groups(groups,synthetic_only=True)
    for mutate in [lambda g:g[0].update(split='test'), lambda g:g[0]['candidates'].pop(),
                   lambda g:g[0]['candidates'][1].update(ambiguous=True),
                   lambda g:g[0]['candidates'][1].update(doc_id=g[0]['candidates'][0]['doc_id']),
                   lambda g:g[0].update(synthetic=False),
                   lambda g:g[0]['known_relevant_ids'].append(g[0]['candidates'][1]['doc_id'])]:
        invalid=copy.deepcopy(groups);mutate(invalid)
        with pytest.raises(ContractError):validate_groups(invalid,synthetic_only=True)
    invalid=copy.deepcopy(groups);invalid[-1]['candidates'][0].update(source_doc_id=invalid[0]['candidates'][0]['source_doc_id'])
    with pytest.raises(ContractError,match='leaks'):validate_groups(invalid)


def test_negative_selection_is_train_only_judged_and_repeatable():
    pool=[{'doc_id':str(i),'split':'train','relevance':0,'judged':True} for i in range(6)]
    ranking={'split':'train','source_sha256':'a'*64,'doc_ids':['1','2','3','4']}
    a=select_negatives(pool,ranking,{'1'},7)
    assert a[1:]==['2','3'] and len(set(a))==3
    assert a==select_negatives(pool,ranking,{'1'},7)
    with pytest.raises(ContractError):select_negatives(pool,ranking|{'split':'test'},{'1'},7)
    for p in pool:p['judged']=False
    with pytest.raises(ContractError):select_negatives(pool,ranking,set(),7)


def test_sampler_resume_does_not_repeat_or_skip():
    a=GroupSampler(7,12);prefix=a.take(13);state=a.state_dict();continuation=a.take(37)
    b=GroupSampler(7,12);b.load_state_dict(state)
    assert b.take(37)==continuation
    c=GroupSampler(7,12)
    assert c.take(50)==prefix+continuation


def test_training_serialization_and_suffix_preserving_truncation_without_gpu():
    from types import SimpleNamespace
    from tapestry_m1.reranker.training import prepare_inputs
    from tapestry_m1.reranker.qwen3 import PREFIX,SUFFIX,reranker_text
    groups=list(records(FIXTURE/'groups.jsonl'))
    serialization=json.loads((FIXTURE/'serialization.json').read_text())
    class Inputs(dict):
        def to(self,device):
            assert device=='cuda';return self
    class Tokenizer:
        def __call__(self,pairs,**kwargs):
            assert kwargs=={'padding':False,'truncation':False,'add_special_tokens':True}
            assert pairs[0]==reranker_text('instruction',groups[0]['query_text'],groups[0]['candidates'][0]['text'])
            return {'input_ids':[list(range(200)) for _ in pairs]}
        def pad(self,payload,**kwargs):return Inputs(payload)
    backend=SimpleNamespace(tokenizer=Tokenizer(),prefix=[10,11],suffix=[12,13,14],limit=32,config={'instruction':'instruction'})
    inputs,audit=prepare_inputs(backend,[groups[0]])
    assert audit['truncated']==4
    assert all(len(ids)==32 and ids[:2]==[10,11] and ids[-3:]==[12,13,14] for ids in inputs['input_ids'])
    first=serialization['pairs'][0]
    assert first['serialized_text']==PREFIX+reranker_text('Given a web search query, retrieve relevant passages that answer the query',groups[0]['query_text'],groups[0]['candidates'][0]['text'])+SUFFIX
    assert first['input_ids'][:len(serialization['prefix_ids'])]==serialization['prefix_ids']
    assert first['input_ids'][-len(serialization['suffix_ids']):]==serialization['suffix_ids']



def test_original_query_identity_and_candidate_content_cannot_drift():
    groups=list(records(FIXTURE/'groups.jsonl'))
    groups[-1]['source_query_id']=groups[0]['source_query_id']
    with pytest.raises(ContractError,match='original query identity'):validate_groups(groups)
    groups=list(records(FIXTURE/'groups.jsonl'))
    groups[1]['candidates'][0]['doc_id']=groups[0]['candidates'][0]['doc_id']
    groups[1]['known_relevant_ids']=[groups[1]['candidates'][0]['doc_id']]
    with pytest.raises(ContractError,match='inconsistent source/text'):validate_groups(groups)
    groups=list(records(FIXTURE/'groups.jsonl'));groups[0]['known_relevant_ids']=[]
    with pytest.raises(ContractError,match='known-positive'):validate_groups(groups)
