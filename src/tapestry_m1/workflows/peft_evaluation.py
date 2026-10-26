"""Re-score immutable native Top-100 only; checkpoint selection is already frozen."""
import json
from pathlib import Path
from tapestry_m1.core.io import atomic_json,ContractError,fingerprint,records
from tapestry_m1.data.au_qa_training import write_rows
from tapestry_m1.reranker.peft_backend import build_backend
from tapestry_m1.reranker.ranking import reorder,check_candidates
from tapestry_m1.evaluation.retrieval.evaluator import evaluate
from tapestry_m1.evaluation.statistics.bootstrap import paired_compare
from tapestry_m1.runtime.devices import gpu_runtime
from tapestry_m1.runtime.training_checkpoint import load_peft_state
from tapestry_m1.runtime.provenance import code_fingerprint


def evaluate_adapter(config_path,data_root,checkpoint,output,objective):
    config_path,data_root,checkpoint,output=Path(config_path),Path(data_root),Path(checkpoint),Path(output)
    config=json.loads(config_path.read_text());config['objective']=objective;config['max_tokens']=8192;config['reranker_batch']=1
    spec=json.loads((config_path.parent/config['model']).read_text());_,meta=load_peft_state(checkpoint)
    if meta['inputs']['config']['objective']!=objective:raise ContractError('adapter objective differs')
    for b in config['benchmarks']:
        for name,sha in b['benchmark_sha256'].items():
            if fingerprint(data_root/b['normalized']/name)!=sha:raise ContractError('native benchmark normalized bytes differ')
    identity={'checkpoint_manifest_sha256':fingerprint(checkpoint/'manifest.json'),'config':config,'code':code_fingerprint()}
    if (output/'summary.json').exists():
        saved=json.loads((output/'summary.json').read_text())
        if saved['identity']!=identity:raise ContractError('evaluation identity differs')
        return saved
    output.mkdir(parents=True,exist_ok=True);hardware=gpu_runtime()
    backend,model_identity,_=build_backend(config,spec,data_root,adapter=checkpoint,trainable=False)
    summaries=[]
    for b in config['benchmarks']:
        root=data_root/b['normalized'];frozen=data_root/b['frozen_ranking'];zero_path=data_root/b['zero_ranking']
        if fingerprint(frozen)!=b['frozen_sha256'] or fingerprint(zero_path)!=b['zero_sha256']:raise ContractError('historical ranking hash differs')
        original=list(records(frozen));documents={r['doc_id']:r['text'] for r in records(root/'documents.jsonl')};queries={r['query_id']:r['query_text'] for r in records(root/'queries.jsonl')}
        target=output/b['dataset_id'];target.mkdir(exist_ok=True);results=[];truncations=0
        for index,row in enumerate(original):
            cache=target/f'query-{index:04d}.json';source_identity={'ranking':b['frozen_sha256'],'query_id':row['query_id'],'adapter':identity['checkpoint_manifest_sha256'],'config':fingerprint(config_path),'benchmark':b['benchmark_sha256']}
            if cache.exists():
                saved=json.loads(cache.read_text())
                if saved['identity']!=source_identity:raise ContractError('query score cache identity differs')
                result=saved['ranking'];audit=saved['lengths']
            else:
                ids=[c['doc_id'] for c in row['candidates']]
                if len(ids)!=100:raise ContractError('exact frozen Top-100 required')
                scores,audit=backend.score([queries[row['query_id']]]*len(ids),[documents[d] for d in ids],batch_size=1)
                result={k:v for k,v in row.items() if k not in ['candidates','run_id','seed','fingerprints']}
                result.update(run_id='peft-'+objective+'-'+identity['checkpoint_manifest_sha256'][:16],seed=config['seed'],
                    fingerprints={'model':identity['checkpoint_manifest_sha256'],'config':fingerprint(config_path),'index':b['frozen_sha256']},candidates=reorder(row['candidates'],scores))
                atomic_json(cache,{'identity':source_identity,'ranking':result,'lengths':audit})
            results.append(result);truncations+=sum(a['truncated'] for a in audit)
            print(f'{objective} {b["dataset_id"]} frozen query {index+1}/{len(original)}',flush=True)
        invariance=check_candidates(original,results);prediction=target/'rankings.jsonl';write_rows(prediction,results)
        metrics=evaluate(root,prediction,'linear',100,'test');zero=evaluate(root,zero_path,'linear',100,'test');retrieval=evaluate(root,frozen,'linear',100,'test')
        for name,value in [('metrics.json',metrics),('zero-shot-metrics.json',zero),('retrieval-metrics.json',retrieval)]:atomic_json(target/name,value)
        comparison=paired_compare(zero,metrics,20261018,10000);atomic_json(target/'vs-zero-shot.json',comparison)
        old={r['query_id']:r for r in retrieval['per_query']}
        if any(r['metrics']['recall@100']!=old[r['query_id']]['metrics']['recall@100'] for r in metrics['per_query']):raise ContractError('Recall@100 changed under fixed candidates')
        summaries.append({'dataset_id':b['dataset_id'],'queries':len(results),'invariance':invariance,'truncated_pairs':truncations,
            'metrics':metrics['groups'],'retrieval':retrieval['groups'],'zero_shot':zero['groups'],'comparison':comparison,
            'prediction_sha256':fingerprint(prediction),'metric_sha256':fingerprint(target/'metrics.json')})
    backend.close();result={'status':'frozen-benchmark-evaluation-complete','identity':identity,'objective':objective,'hardware':hardware,'base':model_identity,'datasets':summaries,'training_context':2048,'evaluation_context':8192}
    atomic_json(output/'summary.json',result);return result


def compare_objectives(l0,l1,output):
    l0,l1,output=Path(l0),Path(l1),Path(output);output.mkdir(parents=True,exist_ok=True)
    a=json.loads((l0/'summary.json').read_text());b=json.loads((l1/'summary.json').read_text());results=[]
    if [d['dataset_id'] for d in a['datasets']]!=[d['dataset_id'] for d in b['datasets']]:raise ContractError('evaluation dataset sets differ')
    for d in a['datasets']:
        name=d['dataset_id'];ma=json.loads((l0/name/'metrics.json').read_text());mb=json.loads((l1/name/'metrics.json').read_text())
        comparison=paired_compare(ma,mb,20261018,10000);atomic_json(output/(name+'.json'),comparison);results.append({'dataset_id':name,'comparison':comparison})
    atomic_json(output/'summary.json',{'status':'paired-query-comparison-complete','training_seeds':1,'query_bootstrap_is_not_seed_variance':True,'comparisons':results})
