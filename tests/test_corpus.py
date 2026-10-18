import json
from pathlib import Path
import numpy as np
import pytest
from conftest import write_rows
from tapestry_m1.io import atomic_json,fingerprint,ContractError,records
from tapestry_m1.corpus.chunks import chunk_document,stage_chunks
from tapestry_m1.corpus.audit import length_audit


class CharacterTokenizer:
    def __call__(self,text,**kwargs):return {'offset_mapping':[(i,i+1) for i in range(len(text))]}
    def encode(self,text,**kwargs):return list(range(len(text)))


@pytest.fixture
def corpus(tmp_path):
    root=tmp_path/'source';root.mkdir()
    docs=[{'collection_id':'source','doc_id':f'source::{i}','text':'abcdefghij','metadata':{'source':'synthetic'}} for i in range(5)]
    write_rows(root/'documents.jsonl',docs)
    atomic_json(root/'manifest.json',{'status':'validated','truncated':False,'observed_counts':{'documents':5},'output_sha256':{'documents.jsonl':fingerprint(root/'documents.jsonl')}})
    return root


def policy():return {'window':4,'overlap':1,'tokenizer_revision':'synthetic-tokenizer'}


def test_chunk_lineage_boundaries():
    doc={'collection_id':'c','doc_id':'c::a','text':'abcdefghij'}
    rows=list(chunk_document(doc,CharacterTokenizer(),policy()))
    assert [r['text'] for r in rows]==['abcd','defg','ghij']
    assert len({r['doc_id'] for r in rows})==3
    for r in rows:
        s=r['source_span'];assert r['text']==doc['text'][s['character_start']:s['character_end']]
        assert r['source_doc_id']=='c::a'
    assert rows==list(chunk_document(doc,CharacterTokenizer(),policy()))
    assert len(list(chunk_document(doc|{'text':'abcd'},CharacterTokenizer(),policy())))==1
    with pytest.raises(ContractError):list(chunk_document(doc,CharacterTokenizer(),policy()|{'overlap':4}))


def test_stage_resume_offsets_and_checksum(corpus,tmp_path):
    root,a=stage_chunks(corpus,tmp_path/'chunks',CharacterTokenizer(),policy(),3,2)
    assert a['status']=='partial' and a['source_documents_staged']==3
    root,b=stage_chunks(corpus,tmp_path/'chunks',CharacterTokenizer(),policy(),5,2)
    assert b['status']=='complete' and b['chunks_staged']==15
    ids=[r['doc_id'] for m in b['shards'] for r in records(root/m['data_file'])]
    assert len(ids)==len(set(ids))==15
    assert stage_chunks(corpus,tmp_path/'chunks',CharacterTokenizer(),policy(),5,2)[1]==b
    with pytest.raises(ContractError):stage_chunks(corpus,tmp_path/'chunks',CharacterTokenizer(),policy(),2,2)
    (root/b['shards'][0]['data_file']).write_text('tampered')
    with pytest.raises(ContractError,match='checksum'):stage_chunks(corpus,tmp_path/'chunks',CharacterTokenizer(),policy(),5,2)


def test_uniform_audit_reproducibility(corpus,tmp_path):
    a=length_audit(corpus,CharacterTokenizer(),tmp_path/'a.json','synthetic',sample_size=3)
    b=length_audit(corpus,CharacterTokenizer(),tmp_path/'b.json','synthetic',sample_size=3)
    assert a['sample']==b['sample'] and a['source_document_count']==5
    assert a['sample_token_percentiles']['50']==10
    assert len(a['character_longest_tail'])==5
    (corpus/'documents.jsonl').write_text('')
    with pytest.raises(ContractError,match='checksum'):length_audit(corpus,CharacterTokenizer(),tmp_path/'bad.json','synthetic')


def test_budgeted_index_and_search(corpus,tmp_path,monkeypatch):
    import torch
    import tapestry_m1.corpus.index as module
    root,coverage=stage_chunks(corpus,tmp_path/'chunks',CharacterTokenizer(),policy(),5,2)
    class StubModel:
        def __init__(self,*args):pass
        def encode(self,texts):return np.asarray([[1.,0.] for _ in texts]),[{'tokens':len(t),'truncated':False} for t in texts]
        def close(self):pass
    monkeypatch.setattr(module,'gpu_runtime',lambda:{'job_id':'synthetic','gpu':'CPU-stub-NOT-model'})
    monkeypatch.setattr(module,'verify_model',lambda *a:(tmp_path,{'synthetic':True}))
    monkeypatch.setattr(module,'QwenAdapter',StubModel)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    cfg={'encoder':{'model_id':'synthetic'},'dimension':2,'embedding_shard':2}
    index,a=module.build_index(root,coverage,cfg,tmp_path,max_chunks=4,max_seconds=60)
    assert a['status']=='partial' and a['chunks_encoded']==4
    index,b=module.build_index(root,coverage,cfg,tmp_path,max_chunks=100,max_seconds=60)
    assert b['status']=='corpus-index-complete' and b['chunks_encoded']==15 and b['source_documents_fully_indexed']==5
    result=module.search_index(index,b,[[1,0]],k=10)
    assert len(result[0])==10 and [r['doc_id'] for r in result[0]]==sorted(r['doc_id'] for m in coverage['shards'] for r in records(root/m['data_file']))[:10]
