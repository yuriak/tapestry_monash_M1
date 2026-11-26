import math
import pytest
from tapestry_m1.workflows.nfcorpus_pilot import BM25,metrics,representation

def test_bm25_full_corpus_and_stable_ties():
    b=BM25({'b':'alpha beta','a':'alpha beta','c':'gamma'})
    r=b.rank('alpha',3)
    assert [x['doc_id'] for x in r]==['a','b','c']
    expected=math.log1p(1.5/2.5)*2.2/(1+1.2*(.25+.75*2/(5/3)))
    assert r[0]['score']==pytest.approx(expected)
    assert [x['doc_id'] for x in b.rank('missing',3)]==['a','b','c']

def test_graded_multi_positive_missing_gold_and_recall():
    r=[{'query_id':'q','candidates':[{'doc_id':'u'},{'doc_id':'b'}]}]
    m=metrics(r,{'q':{'a':2,'b':1}})['means']
    assert m['ndcg@10']==pytest.approx((1/math.log2(3))/(3+1/math.log2(3)))
    assert m['recall@50']==.5 and m['mrr@50']==.5
    assert metrics([{'query_id':'q','candidates':[{'doc_id':'u'}]}],{'q':{'a':1}})['means']['ndcg@10']==0

def test_representation():
    assert representation({'title':' title ','text':' body '})=='title\nbody'
