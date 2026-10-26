import copy
from collections import defaultdict
import pytest
from tapestry_m1.data.au_qa_training import construct,near_pairs
from tapestry_m1.workflows.au_qa_mining import choose_negatives
from tapestry_m1.contracts.reranker_training import text_hash
from tapestry_m1.core.io import ContractError


def pairs():
    rows=[]
    for i,word in enumerate(['amber','zircon','violet','pigeon','coral','cedar','saffron','otter','indigo','maple','ruby','peach']):
        rows.append({'pair_id':str(i),'source_doc_id':'source-'+str(i),'query_text':f'{word} unique hypothetical query {i}',
          'positive_text':f'{word} authored example has {i} singular colorful objects and fictional details '+word*10,
          'source_metadata':{'source':'synthetic'},'provenance':{'source_row':i+1}})
    return rows


def test_split_reproducible_grouped_and_test_excluded():
    rows=pairs();duplicate=copy.deepcopy(rows[0]);duplicate['pair_id']='other';duplicate['provenance']['source_row']=99;rows.append(duplicate)
    q,d,s=construct(rows,[rows[2]['query_text']],[rows[3]['positive_text']],44)
    assert s['exclusions']['test_query_exact']==1 and s['exclusions']['test_passage_exact']==1
    assert (q,d,s)==construct(rows,[rows[2]['query_text']],[rows[3]['positive_text']],44)
    seen=defaultdict(set)
    for r in q:seen[r['source_doc_id']].add(r['split'])
    assert all(len(v)==1 for v in seen.values()) and {r['split'] for r in q}=={'train','dev'}
    assert all(r['text'] in {p['positive_text'] for p in rows} for r in d)


def test_assumed_sampler_excludes_sources_known_positives_near_and_split():
    docs=[{'doc_id':str(i),'source_doc_ids':[str(i)],'text':'invented passage '+str(i)} for i in range(9)]
    docs[1]['source_doc_ids']=['0']
    q={'query_id':'q','known_relevant_ids':['0','2']}
    rank=[{'doc_id':str(i)} for i in range(9)]
    selected,sources=choose_negatives(q,docs,rank,{'0':{'3'}},19)
    assert selected[1:]==['4','5'] and len(set(selected))==3
    assert not set(selected)&{'0','1','2','3'}
    assert selected==choose_negatives(q,docs,rank,{'0':{'3'}},19)[0]
    with pytest.raises(ContractError):choose_negatives(q,docs[:3],rank,{},19)


def test_near_duplicate_detection_is_deterministic():
    text=' '.join('synthetic word'+str(i) for i in range(200))
    assert list(near_pairs([text,text]))==[(0,1,1.)]
