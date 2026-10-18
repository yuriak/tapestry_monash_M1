import math
import pytest
from tapestry_m1.evaluation.metrics import metrics
from tapestry_m1.io import ContractError


def test_single_and_graded_analytic():
    assert metrics(["a"], {"a":1}) == {"ndcg@10":1,"recall@10":1,"recall@100":1,"mrr":1}
    rel={"a":3,"b":1}
    expected=(1+7/math.log2(3))/(7+1/math.log2(3))
    assert metrics(["b","a"], rel)["ndcg@10"] == pytest.approx(expected)
    assert metrics(["a","b"], rel)["ndcg@10"] == 1
    assert metrics(["b","a"], rel, gain="linear")["ndcg@10"] == pytest.approx((1+3/math.log2(3))/(3+1/math.log2(3)))


def test_boundaries_and_depth():
    docs=[f"d{i}" for i in range(1,102)]
    result=metrics(docs,{"d10":1,"d11":1,"d100":1,"d101":1})
    assert result["recall@10"] == .25
    assert result["recall@100"] == .75
    assert result["mrr"] == .1
    assert metrics(docs,{"d11":1},mrr_depth=10)["mrr@10"] == 0
    assert metrics(docs,{"d11":1})["mrr"] == 1/11
    assert metrics(docs,{"d101":1})["ndcg@10"] == 0


def test_zero_hits():
    assert set(metrics([], {"gold":1}).values()) == {0}
    assert set(metrics(["negative"], {"gold":1,"negative":0}).values()) == {0}


@pytest.mark.parametrize("docs,rels,kwargs", [(["a","a"],{"a":1},{}),([],{},{}),([],{"a":float("nan")},{}),([],{"a":-1},{}),([],{"a":True},{}),([],{"a":1},{"mrr_depth":0}),([],{"a":1},{"gain":"unknown"})])
def test_metric_errors(docs,rels,kwargs):
    with pytest.raises(ContractError):metrics(docs,rels,**kwargs)
