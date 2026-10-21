import copy
import pytest
from tapestry_m1.evaluation.statistics.bootstrap import paired_compare
from tapestry_m1.evaluation.retrieval.evaluator import evaluate
from tapestry_m1.core.io import ContractError


def test_paired_direction_alignment_and_determinism(benchmark,rankings):
    a=evaluate(benchmark,rankings);b=copy.deepcopy(a)
    for row in b["per_query"]:row["metrics"]={m:1. for m in row["metrics"]}
    b["per_query"].reverse();b["run"]["run_id"]="B"
    result=paired_compare(a,b,seed=42,resamples=101)
    assert result==paired_compare(a,b,seed=42,resamples=101)
    m=result["groups"][0]["metrics"]["mrr"]
    assert m=={"mean_delta":.5,"ci95":[.5,.5]}
    reverse=paired_compare(b,a,seed=42,resamples=101)
    assert reverse["groups"][0]["metrics"]["mrr"]["ci95"]==[-.5,-.5]


def test_nonconstant_paired_analytic(benchmark,rankings):
    a=evaluate(benchmark,rankings);b=copy.deepcopy(a)
    a["per_query"][0]["metrics"]["mrr"]=0
    a["per_query"][1]["metrics"]["mrr"]=1
    b["per_query"][0]["metrics"]["mrr"]=1
    b["per_query"][1]["metrics"]["mrr"]=0
    m=paired_compare(a,b,seed=4,resamples=1000)["groups"][0]["metrics"]["mrr"]
    assert m=={"mean_delta":0,"ci95":[-1,1]}


@pytest.mark.parametrize("case",["missing_query","split","fingerprint","definition","duplicate","nonfinite","missing_hash","metrics"])
def test_pair_rejection(benchmark,rankings,case):
    a=evaluate(benchmark,rankings);b=copy.deepcopy(a)
    if case=="missing_query":b["per_query"].pop()
    elif case=="split":b["per_query"][0]["split"]="train"
    elif case=="fingerprint":b["input_sha256"]["qrels.jsonl"]="other"
    elif case=="definition":b["definition"]["mrr_depth"]=10
    elif case=="duplicate":b["per_query"].append(b["per_query"][0])
    elif case=="nonfinite":b["per_query"][0]["metrics"]["mrr"]=float("nan")
    elif case=="missing_hash":del a["input_sha256"];del b["input_sha256"]
    else:del b["per_query"][0]["metrics"]["mrr"]
    with pytest.raises(ContractError):paired_compare(a,b,resamples=20)
