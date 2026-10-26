"""Synthetic native Top-100 exercises real evaluator/CI/cache without a GPU."""
import json
from pathlib import Path
from tapestry_m1.core.io import fingerprint
from tapestry_m1.data.au_qa_training import write_rows
from tapestry_m1.workflows import peft_evaluation as workflow


def test_frozen_top100_evaluation_and_reuse(monkeypatch,tmp_path):
    data=tmp_path/'data';root=data/'benchmark';root.mkdir(parents=True)
    docs=[{'collection_id':'synthetic','doc_id':'synthetic::'+str(i),'text':'synthetic native passage '+str(i)} for i in range(100)]
    queries=[{'dataset_id':'synthetic-benchmark','split':'test','query_id':'synthetic-benchmark::q','query_text':'synthetic question','collection_id':'synthetic'}]
    qrels=[{'dataset_id':'synthetic-benchmark','split':'test','query_id':queries[0]['query_id'],'doc_id':docs[0]['doc_id'],'relevance':1}]
    for name,rows in [('documents.jsonl',docs),('queries.jsonl',queries),('qrels.jsonl',qrels)]:write_rows(root/name,rows)
    row={k:queries[0][k] for k in ['dataset_id','split','query_id','collection_id']}
    row.update(run_id='frozen',seed=None,fingerprints={'model':'base','config':'base','index':'base'},candidates=[{'doc_id':d['doc_id'],'rank':i+1,'score':float(100-i)} for i,d in enumerate(docs)])
    write_rows(data/'frozen.jsonl',[row]);write_rows(data/'zero.jsonl',[row])
    cp=tmp_path/'checkpoint';cp.mkdir();(cp/'manifest.json').write_text('{}')
    config={'model':'model.json','seed':20261026,'benchmarks':[{'dataset_id':'synthetic-benchmark','normalized':'benchmark','frozen_ranking':'frozen.jsonl',
        'frozen_sha256':fingerprint(data/'frozen.jsonl'),'zero_ranking':'zero.jsonl','zero_sha256':fingerprint(data/'zero.jsonl'),
        'benchmark_sha256':{name:fingerprint(root/name) for name in ['documents.jsonl','queries.jsonl','qrels.jsonl']}}]}
    path=tmp_path/'config.json';path.write_text(json.dumps(config));(tmp_path/'model.json').write_text('{}')
    class Backend:
        def score(self,queries,documents,batch_size):
            assert len(queries)==100 and batch_size==1
            return list(range(100,0,-1)),[{'truncated':False} for _ in queries]
        def close(self):pass
    monkeypatch.setattr(workflow,'gpu_runtime',lambda:{'gpu':'synthetic'})
    monkeypatch.setattr(workflow,'load_peft_state',lambda *args:(None,{'inputs':{'config':{'objective':'L0'}}}))
    monkeypatch.setattr(workflow,'build_backend',lambda *args,**kwargs:(Backend(),{'synthetic':True},{}))
    out=tmp_path/'result'
    result=workflow.evaluate_adapter(path,data,cp,out,'L0')
    assert result['datasets'][0]['invariance']=={'queries':1,'candidate_sets_identical':True}
    assert result['datasets'][0]['metrics'][0]['macro']['recall@100']==1
    assert result==workflow.evaluate_adapter(path,data,cp,out,'L0')
    assert fingerprint(data/'frozen.jsonl')==config['benchmarks'][0]['frozen_sha256']
