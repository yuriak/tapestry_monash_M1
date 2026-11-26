import math
import pytest
from tapestry_m1.core.io import ContractError
from tapestry_m1.workflows.nfcorpus_training import objective_loss, ranking_metrics, FrozenSchedule, make_groups


def test_objectives_and_gradients():
    import torch
    from tapestry_m1.reranker.losses import group_loss
    x=torch.tensor([[0.,0.,0.,0.]],requires_grad=True)
    l=objective_loss(x,'L2');assert float(l)==pytest.approx(math.log(4))
    l.backward();assert x.grad[0,0]<0 and (x.grad[0,1:]>0).all()
    assert float(objective_loss(torch.tensor([[5.,0.,0.,0.]]),'L2'))<float(l)
    assert float(objective_loss(torch.tensor([[-5.,0.,0.,0.]]),'L2'))>float(l)
    a=torch.tensor([[.2,-1.,2.,3.],[-.2,1.,-2.,-3.]],requires_grad=True)
    assert torch.equal(objective_loss(a,'L0'),group_loss(a,'L0')[0])
    assert torch.allclose(objective_loss(a,'L2'),objective_loss(a[:,[0,3,1,2]],'L2'))
    assert torch.allclose(objective_loss(a,'L2'),objective_loss(a+100,'L2'))
    assert torch.isfinite(objective_loss(torch.tensor([[1000.,-1000.,-999.,-998.]]),'L2'))
    with pytest.raises(ContractError):objective_loss(torch.tensor([[float('nan'),0,0,0]]),'L2')


def test_full_query_metrics_and_native_grades():
    ranks=[{'query_id':'q1','candidates':[{'doc_id':'b'},{'doc_id':'a'}]},
           {'query_id':'q2','candidates':[{'doc_id':'x'}]}]
    judgments={'q1':{'a':2,'b':1},'q2':{'z':1}}
    m=ranking_metrics(ranks,judgments)
    expected=(1+3/math.log2(3))/(3+1/math.log2(3))
    assert m['means']['ndcg@10']==pytest.approx(expected/2)
    assert m['means']['recall@100']==.5
    assert m['means']['hit@100']==.5
    assert m['means']['mrr@10']==.5
    with pytest.raises(ContractError):ranking_metrics(ranks[:1],judgments)
    with pytest.raises(ContractError):ranking_metrics([ranks[0],ranks[0]],judgments)


def test_schedule_resume_and_exhaustion():
    s=FrozenSchedule(16);assert s.take(8)==list(range(8))
    t=FrozenSchedule(16);t.load_state_dict(s.state_dict())
    assert s.take(8)==t.take(8)==list(range(8,16))
    with pytest.raises(ContractError):s.take(1)


def test_train_group_multi_positive_and_alias_exclusion():
    docs={'a':'positive title text','alias':'positive title text','b':'another known relevant document',
          'c':'distant document one','d':'different second text','e':'third completely separate words','f':'last random example'}
    qs={'q':'query'};judgments={'q':{'a':1,'b':2}}
    ranks=[{'query_id':'q','candidates':[{'doc_id':d} for d in ['alias','b','c','d','a','e','f']]}]
    groups,audit=make_groups(docs,qs,judgments,ranks,7,1)
    assert groups==make_groups(docs,qs,judgments,ranks,7,1)[0]
    cs=groups[0]['candidates'];assert cs[0]['doc_id'] in ('a','b')
    assert not {c['doc_id'] for c in cs[1:]} & {'a','b','alias'}
    assert [c['doc_id'] for c in cs[2:]]==['c','d']
    assert all(c['assumed_negative'] for c in cs[1:])
    with pytest.raises(ContractError):make_groups(docs,qs,judgments,ranks,7,2)
