import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tapestry_m1.core.io import ContractError
from tapestry_m1.workflows import reranker_hard as workflow


def test_teacher_query_resume_and_finished_cache_checksum(tmp_path,monkeypatch):
    from tapestry_m1.reranker import qwen3
    from tapestry_m1.runtime import assets,devices
    config={'model':'model.json','instruction':'fictional task','dtype':'bfloat16','attention_backend':'sdpa',
        'reference_sha256':{},'negative_selection':{'teacher_context':8192,'teacher_batch':8}}
    path=tmp_path/'recipe.json';path.write_text(json.dumps(config));(tmp_path/'model.json').write_text('{}')
    lock=tmp_path/'uv.lock';lock.write_text('synthetic lock');monkeypatch.setenv('M1_ENVIRONMENT_LOCK',str(lock))
    docs=[{'doc_id':str(i),'text':'fiction '+str(i),'split':'train'} for i in range(100)]
    groups=[{'query_id':'q'+str(i),'query_text':'synthetic '+str(i),'split':'train',
        'candidates':[{'doc_id':str(j)} for j in range(4)]} for i in range(2)]
    rankings={g['query_id']:{'split':'train','candidates':[{'doc_id':str(i)} for i in range(100)]} for g in groups}
    monkeypatch.setattr(workflow,'reference',lambda *args:(groups,docs,rankings))
    monkeypatch.setattr(assets,'verify_model',lambda *args:(tmp_path,{'model':'synthetic'}))
    monkeypatch.setattr(devices,'gpu_runtime',lambda:{'gpu':'synthetic-test'})
    class Backend:
        calls=0
        fail=True
        def __init__(self,*args):
            self.model=SimpleNamespace(training=False,modules=lambda:[])
            self.tokenizer=SimpleNamespace(padding_side='left')
        def score_logits(self,qs,ds):
            Backend.calls+=1
            if Backend.fail and Backend.calls==2:raise RuntimeError('simulated interruption')
            return [float(i) for i in range(len(ds))],[{'truncated':False} for _ in ds]
        def close(self):pass
    monkeypatch.setattr(qwen3,'Qwen3Reranker',Backend)
    root=tmp_path/'new';root.mkdir()
    with pytest.raises(RuntimeError,match='interruption'):workflow.score_teacher(path,tmp_path,tmp_path,root)
    assert len(list((root/'teacher').glob('query-*.json')))==1
    Backend.calls=0;Backend.fail=False
    manifest=workflow.score_teacher(path,tmp_path,tmp_path,root)
    assert Backend.calls==1 and manifest['queries']==2 and manifest['pairs']==200
    assert workflow.score_teacher(path,tmp_path,tmp_path,root)==manifest and Backend.calls==1
    cached=next((root/'teacher').glob('query-*.json'));cached.write_text('{}')
    with pytest.raises(ContractError,match='checksum'):workflow.score_teacher(path,tmp_path,tmp_path,root)
