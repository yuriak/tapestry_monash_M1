"""Resumable base-reranker scoring, controlled group freeze and one L0 run.

Analysis of failure causes and difficulty distributions belongs to independent research.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from tapestry_m1.core.io import ContractError, atomic_json, fingerprint, records
from tapestry_m1.data.au_qa_training import write_rows
from tapestry_m1.data.reranker_hard import near_index, select_hard, assert_comparable
from tapestry_m1.runtime.provenance import code_fingerprint


def now(): return datetime.now(timezone.utc).isoformat(timespec='seconds')


def set_status(root, stage, state, **extra):
    path=root/(stage+'-status.json');temp=root/('.'+stage+'-status.tmp')
    temp.write_text(json.dumps({'stage':stage,'state':state,'updated_at':now(),**extra},indent=2)+'\n')
    os.replace(temp,path)


def reference(config, root):
    for name, sha in config['reference_sha256'].items():
        if fingerprint(root/name) != sha:
            raise ContractError('frozen reference changed: '+name)
    manifest=json.loads((root/'mined/manifest.json').read_text())
    for name, sha in manifest['files'].items():
        if fingerprint(root/'mined'/name)!=sha:raise ContractError('reference mining checksum differs')
    groups=list(records(root/'mined/groups.jsonl'))
    passages=list(records(root/'collection/passages.jsonl'))
    rankings={r['query_id']:r for split in ['train','dev'] for r in records(root/f'mined/{split}-ranking.jsonl')}
    return groups,passages,rankings


def score_teacher(config_path, data_root, ref, root):
    from tapestry_m1.reranker.qwen3 import Qwen3Reranker, PREFIX, SUFFIX
    from tapestry_m1.runtime.assets import verify_model
    from tapestry_m1.runtime.devices import gpu_runtime
    config=json.loads(config_path.read_text());groups,passages,rankings=reference(config,ref)
    teacher=dict(config,max_tokens=config['negative_selection']['teacher_context'],reranker_batch=config['negative_selection']['teacher_batch'])
    spec=json.loads((config_path.parent/config['model']).read_text())
    model_dir,model_identity=verify_model(spec,data_root)
    identity={'source':config['reference_sha256'],'config_sha256':fingerprint(config_path),
        'code':code_fingerprint(),'lock_sha256':fingerprint(os.environ['M1_ENVIRONMENT_LOCK']),
        'model':model_identity,'teacher_context':teacher['max_tokens'],'batch':teacher['reranker_batch'],
        'score':'float32(z_yes)-float32(z_no)','template_sha256':hashlib.sha256((PREFIX+SUFFIX).encode()).hexdigest(),
        'instruction':config['instruction'],'dtype':config['dtype'],'attention_backend':config['attention_backend'],
        'adapter':None,'dropout':'eval','padding_side':'left'}
    cache=root/'teacher';cache.mkdir(exist_ok=True)
    atomic_json(cache/'identity.json',identity)
    # Existing marker never substitutes for verifying all score cache bytes.
    old=json.loads((cache/'manifest.json').read_text()) if (cache/'manifest.json').exists() else None
    if old:
        if old['identity']!=identity:raise ContractError('teacher identity differs')
        for name,sha in old['files'].items():
            if fingerprint(cache/name)!=sha:raise ContractError('teacher cache checksum differs')
        return old
    hardware=gpu_runtime();backend=Qwen3Reranker(model_dir,teacher)
    if hasattr(backend.model,'peft_config') or backend.model.training or backend.tokenizer.padding_side!='left':
        raise ContractError('teacher must be unadapted eval-mode left-padded base')
    if any(m.training for m in backend.model.modules()):raise ContractError('teacher dropout not disabled')
    lookup={p['doc_id']:p for p in passages};files={};pair_count=0
    for i,g in enumerate(groups):
        rank=rankings[g['query_id']]
        if rank['split']!=g['split'] or len(rank['candidates'])!=100:raise ContractError('fixed same-split Top-100 required')
        ids=list(dict.fromkeys([c['doc_id'] for c in rank['candidates']]+[c['doc_id'] for c in g['candidates']]))
        if any(lookup[d]['split']!=g['split'] for d in ids):raise ContractError('teacher cross-split candidate')
        name='query-'+hashlib.sha256(g['query_id'].encode()).hexdigest()+'.json';path=cache/name
        expected={'teacher':hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),'query_id':g['query_id'],'doc_ids':ids}
        if path.exists():
            saved=json.loads(path.read_text())
            if saved['identity']!=expected:raise ContractError('query teacher cache identity differs')
        else:
            scores,lengths=backend.score_logits([g['query_text']]*len(ids),[lookup[d]['text'] for d in ids])
            saved={'identity':expected,'scores':dict(zip(ids,scores)),'lengths':lengths}
            atomic_json(path,saved)
        if set(saved['scores'])!=set(ids) or len(saved['lengths'])!=len(ids) or not all(math.isfinite(s) for s in saved['scores'].values()):
            raise ContractError('incomplete/nonfinite teacher scores')
        files[name]=fingerprint(path);pair_count+=len(ids)
        print(json.dumps({'teacher_queries_complete':i+1,'queries':len(groups),'unique_pairs':pair_count}),flush=True)
    backend.close()
    result={'identity':identity,'files':files,'queries':len(groups),'pairs':pair_count,'hardware':hardware,'status':'teacher-complete'}
    atomic_json(cache/'manifest.json',result);return result


def freeze(config_path,ref,root,teacher):
    config=json.loads(config_path.read_text());groups,passages,rankings=reference(config,ref)
    near=near_index(passages);replacement=[]
    try:
        for g in groups:
            name='query-'+hashlib.sha256(g['query_id'].encode()).hexdigest()+'.json'
            scores=json.loads((root/'teacher'/name).read_text())['scores']
            replacement.append(select_hard(g,passages,rankings[g['query_id']]['candidates'],scores,near))
        gate=assert_comparable(groups,replacement)
    except ContractError as exc:
        atomic_json(root/'readiness.json',{'passed':False,'reason':str(exc),'next_action':'Stop before training; review comparability, never expand the pool.'})
        raise
    if gate['counts']!={'train':1805,'dev':319}:raise ContractError('reviewed original group counts required')
    destination=root/'mined';destination.mkdir(exist_ok=True)
    write_rows(destination/'groups.jsonl',replacement)
    manifest={'schema_version':'m1.reranker_hard.groups.v1','arm':'reranker-hard-L0','counts':gate['counts'],
        'inputs':{'reference_sha256':config['reference_sha256'],'teacher_manifest_sha256':fingerprint(root/'teacher/manifest.json'),
                  'config_sha256':fingerprint(config_path),'code':code_fingerprint()},
        'comparability':gate,'exclusions':[],
        'false_negative_caveat':'Teacher score selects assumed negatives, not certified irrelevance.',
        'files':{'groups.jsonl':fingerprint(destination/'groups.jsonl')}}
    atomic_json(destination/'manifest.json',manifest);atomic_json(root/'readiness.json',gate)
    return manifest


def train(config_path,data_root,ref,root):
    config=json.loads(config_path.read_text());old,_,_=reference(config,ref)
    manifest=json.loads((root/'mined/manifest.json').read_text())
    if not json.loads((root/'readiness.json').read_text())['passed']:raise ContractError('readiness gate failed')
    if fingerprint(root/'mined/groups.jsonl')!=manifest['files']['groups.jsonl']:raise ContractError('new groups changed')
    assert_comparable(old,list(records(root/'mined/groups.jsonl')))
    from tapestry_m1.runtime.training_checkpoint import load_peft_state
    base=root/'training/L0';base.mkdir(parents=True,exist_ok=True)
    while True:
        segments=sorted(base.glob('segment-*'));complete=[];checkpoints=[]
        for segment in segments:
            if (segment/'summary.json').exists():
                summary=json.loads((segment/'summary.json').read_text())
                if summary['status']=='real-pair-objective-complete':complete.append(summary)
            for path in segment.glob('checkpoints/*/manifest.json'):
                _,meta=load_peft_state(path.parent);checkpoints.append((meta['step'],path.parent))
        if complete:
            summary=complete[-1];break
        resume=max(checkpoints,key=lambda x:x[0])[1] if checkpoints else None
        out=base/f'segment-{len(segments):03d}'
        args=[sys.executable,'-m','tapestry_m1.workflows.reranker_training','train',str(config_path),'--data-root',str(data_root),
            '--groups-root',str(root/'mined'),'--objective','L0','--output',str(out)]
        args+=['--resume',str(resume)] if resume else ['--stop-after','5']
        subprocess.run(args,check=True)
    checkpoint=Path(summary['best_checkpoint']);_,meta=load_peft_state(checkpoint,summary['identity'])
    atomic_json(root/'selected_adapter.json',{'checkpoint':str(checkpoint),'manifest_sha256':fingerprint(checkpoint/'manifest.json'),
        'training_identity':summary['identity'],'step':meta['step'],'criterion':'dev BCE only'})
    from tapestry_m1.workflows.peft_evaluation import evaluate_adapter
    # The reusable evaluator performs inference only; research owns A1-vs-reference analysis.
    evaluate_adapter(config_path,data_root,checkpoint,root/'evaluation/L0','L0')
    return {'status':'training-and-native-evaluation-complete','selected_manifest_sha256':fingerprint(checkpoint/'manifest.json')}


def preflight(config_path,data_root,ref,root):
    from tapestry_m1.reranker.qwen3 import Qwen3Reranker
    from tapestry_m1.runtime.assets import verify_model
    from tapestry_m1.runtime.devices import gpu_runtime
    import torch
    config=json.loads(config_path.read_text());groups,_,_=reference(config,ref)
    spec=json.loads((config_path.parent/config['model']).read_text());model_dir,_=verify_model(spec,data_root)
    teacher=dict(config,max_tokens=8192,reranker_batch=8);hardware=gpu_runtime();backend=Qwen3Reranker(model_dir,teacher)
    assert not hasattr(backend.model,'peft_config') and not backend.model.training
    g=groups[0];qs=[g['query_text']]*8;ds=[c['text'] for c in g['candidates']]*2
    logits,_=backend.score_logits(qs,ds);probabilities,_=backend.score(qs,ds)
    error=max(abs(float(torch.sigmoid(torch.tensor(s)))-p) for s,p in zip(logits,probabilities))
    if error>1e-6:raise ContractError('raw teacher score disagrees with yes/no probability')
    backend.close()
    # Recreate both initialization recipes; actual adapter tensors must match before learning.
    import random
    from peft import get_peft_model_state_dict
    from tapestry_m1.reranker.peft_backend import build_backend
    original=json.loads((ref/'training/L0/segment-000/run.json').read_text())['inputs']['config']
    if {k:config[k] for k in original}!=original:
        raise ContractError('A0/A1 training recipe changed beyond negative selection provenance')
    initial_hashes=[]
    for recipe in [original,config]:
        random.seed(recipe['seed']);torch.manual_seed(recipe['seed']);torch.cuda.manual_seed_all(recipe['seed'])
        model,_,_=build_backend(recipe,spec,data_root)
        digest=hashlib.sha256()
        for name,value in sorted(get_peft_model_state_dict(model.model).items()):
            digest.update(name.encode());digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        initial_hashes.append(digest.hexdigest());model.close()
    if initial_hashes[0]!=initial_hashes[1]:raise ContractError('A0/A1 starting adapter weights differ')
    attempts=sorted(root.glob('peft-preflight-*'));out=root/f'peft-preflight-{len(attempts):03d}'
    subprocess.run([sys.executable,'-m','tapestry_m1.workflows.reranker_training','train',str(config_path),
        '--data-root',str(data_root),'--output',str(out),'--preflight'],check=True)
    return {'status':'preflight-complete','teacher_adapter':None,'teacher_probability_max_error':error,'initial_adapter_hashes':initial_hashes,'original_training_recipe_equal':True,
            'hardware':hardware,'peft_preflight':json.loads((out/'preflight.json').read_text())}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['preflight','mine','train'])
    p.add_argument('config',type=Path);p.add_argument('--data-root',required=True,type=Path)
    p.add_argument('--reference-root',required=True,type=Path);p.add_argument('--run-root',required=True,type=Path)
    a=p.parse_args(argv);root=a.run_root.resolve();root.mkdir(parents=True,exist_ok=True);config=a.config.resolve()
    with (root/('.'+a.stage+'.lock')).open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Stage already has an active writer',file=sys.stderr);return 2
        try:
            identity={'code':code_fingerprint(),'config_sha256':fingerprint(config),'lock_sha256':fingerprint(os.environ['M1_ENVIRONMENT_LOCK'])}
            atomic_json(root/'run_identity.json',identity);set_status(root,a.stage,'running',identity=identity)
            if a.stage=='preflight':result=preflight(config,a.data_root,a.reference_root,root)
            elif a.stage=='mine':
                teacher=score_teacher(config,a.data_root,a.reference_root,root);result=freeze(config,a.reference_root,root,teacher)
            else:result=train(config,a.data_root,a.reference_root,root)
            # Keep preflight evidence compact: pair text/score vectors stay private, outside this marker.
            atomic_json(root/(a.stage+'-complete.json'),{'identity':identity,'result':result})
            set_status(root,a.stage,'complete',identity=identity)
            return 0
        except (Exception,KeyboardInterrupt) as exc:
            set_status(root,a.stage,'failed',error_type=type(exc).__name__,message=str(exc))
            print(str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
