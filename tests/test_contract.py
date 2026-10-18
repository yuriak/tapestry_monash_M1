import copy
import json
import pytest
from conftest import write_rows
from tapestry_m1.io import records, ContractError
from tapestry_m1.data.validation import validate
from tapestry_m1.data.schema import identifier, ranking
from tapestry_m1.evaluation.evaluator import evaluate


def mutate(root, filename, operation):
    path=root/(filename+".jsonl");rows=list(records(path));operation(rows);write_rows(path,rows)


def test_valid_and_namespaced_collision(benchmark,rankings):
    assert identifier("c","a/b") == "c::a%2Fb"
    assert identifier("c","a/b") != identifier("other","a/b")
    assert validate(benchmark)["counts"] == {"documents":3,"queries":2,"qrels":2}
    result=evaluate(benchmark,rankings)
    assert result["counts"] == {"evaluated":2,"skipped":0,"errors":0}
    assert result["groups"][0]["macro"]["mrr"] == .5
    assert result["per_query"][0]["unjudged_retrieved_count"] == 1
    assert result == evaluate(benchmark,rankings)


@pytest.mark.parametrize("case",["missing_gold","wrong_collection","wrong_split","duplicate_doc","duplicate_query","duplicate_qrel","empty_text","nonfinite_label","missing_split","no_positive","contamination"])
def test_invalid_dataset(benchmark,case):
    if case in ["missing_gold","wrong_collection","wrong_split","duplicate_qrel","nonfinite_label","no_positive"]:
        path=benchmark/"qrels.jsonl";rows=list(records(path))
        if case=="missing_gold":rows[0]["doc_id"]="c::absent"
        elif case=="wrong_collection":rows[0]["doc_id"]="other::a"
        elif case=="wrong_split":rows[0]["split"]="train"
        elif case=="duplicate_qrel":rows.append(rows[0])
        elif case=="nonfinite_label":rows[0]["relevance"]=-1
        else:rows[0]["relevance"]=0
    else:
        name="documents" if case in ["duplicate_doc","empty_text"] else "queries"
        path=benchmark/(name+".jsonl");rows=list(records(path))
        if case in ["duplicate_doc","duplicate_query"]:rows.append(rows[0])
        elif case=="empty_text":rows[0]["text"]="  "
        elif case=="missing_split":del rows[0]["split"]
        else:
            row=copy.deepcopy(rows[0]);row["split"]="train";row["query_text"]=" SYNTHETIC   QUESTION ONE? ";rows.append(row)
    write_rows(path,rows)
    with pytest.raises(ContractError):validate(benchmark)


@pytest.mark.parametrize("case",["missing","extra","duplicate_query","duplicate_candidate","wrong_collection","foreign_candidate","rank_order","score","mixed_seed","mixed_run","no_seed"])
def test_invalid_rankings(benchmark,rankings,case):
    rows=list(records(rankings))
    if case=="missing":rows.pop()
    elif case=="extra":rows[0]["query_id"]="bench::unknown"
    elif case=="duplicate_query":rows.append(rows[0])
    elif case=="duplicate_candidate":rows[0]["candidates"][1]["doc_id"]="c::b"
    elif case=="wrong_collection":rows[0]["collection_id"]="other"
    elif case=="foreign_candidate":rows[0]["candidates"][0]["doc_id"]="other::a"
    elif case=="rank_order":rows[0]["candidates"][0]["rank"]=2
    elif case=="score":rows[0]["candidates"][0]["score"]="nan"
    elif case=="mixed_seed":rows[0]["seed"]=1
    elif case=="mixed_run":rows[0]["run_id"]="other-run"
    else:del rows[0]["seed"]
    write_rows(rankings,rows)
    with pytest.raises(ContractError):evaluate(benchmark,rankings)


def test_explicit_empty_predictions(benchmark,rankings):
    rows=list(records(rankings));rows[0]["candidates"]=[];write_rows(rankings,rows)
    result=evaluate(benchmark,rankings)
    assert result["per_query"][0]["metrics"]["recall@100"] == 0
    assert result["groups"][0]["macro"]["mrr"] == .25


def test_nonfinite_score(rankings):
    row=next(records(rankings));row["candidates"][0]["score"]=float("inf")
    with pytest.raises(ContractError):ranking(row)


def test_same_query_id_across_splits_even_if_text_differs(benchmark):
    rows=list(records(benchmark/'queries.jsonl'))
    row=copy.deepcopy(rows[0]);row['split']='train';row['query_text']='Different wording but same identity.';rows.append(row)
    write_rows(benchmark/'queries.jsonl',rows)
    with pytest.raises(ContractError,match='contamination'):validate(benchmark)


def test_training_pair_validation(benchmark):
    row={'dataset_id':'training','split':'train','pair_id':'training::1','query_text':'A synthetic training query?','answer':'Answer.','positive_text':'Source snippet.'}
    write_rows(benchmark/'training_pairs.jsonl',[row])
    assert validate(benchmark)['training_pairs_validated']==1
    duplicate=copy.deepcopy(row);duplicate['split']='validation';duplicate['pair_id']='training::2'
    write_rows(benchmark/'training_pairs.jsonl',[row,duplicate])
    with pytest.raises(ContractError,match='contamination'):validate(benchmark)
    write_rows(benchmark/'training_pairs.jsonl',[row,row])
    with pytest.raises(ContractError,match='duplicate'):validate(benchmark)


def test_malformed_json_record(benchmark):
    (benchmark/'documents.jsonl').write_text('[1, 2]\n')
    with pytest.raises(ContractError,match='documents.jsonl:1'):validate(benchmark)
