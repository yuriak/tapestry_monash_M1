"""Resumable local QA preparation/mining/training/evaluation orchestration."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from tapestry_m1.core.io import atomic_json,ContractError,fingerprint
from tapestry_m1.core.artifacts import identity
from tapestry_m1.data.au_qa_training import prepare
from tapestry_m1.runtime.provenance import code_fingerprint


def now():return datetime.now(timezone.utc).isoformat(timespec='seconds')


def status(root,value):
    path=root/'status.json';temp=root/'.status.tmp'
    temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def preflight(config_path,data_root,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    marker=root/'preflight/preflight.json'
    expected={'code':code_fingerprint(),'config_sha256':fingerprint(config_path)}
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved.get('inputs')!=expected:raise ContractError('preflight identity differs')
        return saved
    paths=sorted(root.glob('preflight-attempt-*'));out=root/f'preflight-attempt-{len(paths):03d}'
    subprocess.run([sys.executable,'-m','tapestry_m1.workflows.reranker_training','train',str(config_path),
        '--data-root',str(data_root),'--output',str(out),'--preflight'],check=True)
    saved=json.loads((out/'preflight.json').read_text())|{'inputs':expected}
    atomic_json(marker,saved);return saved


def pipeline(config_path,data_root,run_root):
    root=Path(run_root);root.mkdir(parents=True,exist_ok=True)
    config_path=Path(config_path).resolve();data_root=Path(data_root).resolve()
    config=json.loads(config_path.read_text())
    run_identity={'code':code_fingerprint(),'config_sha256':fingerprint(config_path),'environment_lock_sha256':fingerprint(os.environ['M1_ENVIRONMENT_LOCK'])}
    atomic_json(root/'pipeline_run.json',{'identity':run_identity,'seed':config['seed'],'planned_epochs':2,'planned_objectives':['L0','L1']})
    state={'started_at':now(),'phase':'preflight','state':'running','identity':run_identity}
    def invoke(arguments):
        status(root,state)
        subprocess.run([sys.executable,'-m',*arguments],check=True)
    try:
        preflight(config_path,data_root,root)
        state['phase']='prepare'
        prepare(data_root,root/'collection',[data_root/b['normalized'] for b in config['benchmarks']],config['seed'])
        state['phase']='mine';status(root,state)
        invoke(['tapestry_m1.workflows.au_peft','mine',str(config_path),'--data-root',str(data_root),'--run-root',str(root)])
        finals={}
        for objective in ['L0','L1']:
            state['phase']='train-'+objective
            base=root/'training'/objective;base.mkdir(parents=True,exist_ok=True)
            while True:
                segments=sorted(base.glob('segment-*'))
                complete=[]
                for segment in segments:
                    if (segment/'summary.json').exists():
                        summary=json.loads((segment/'summary.json').read_text())
                        if summary['status']=='real-pair-objective-complete':complete.append((segment,summary))
                if complete:
                    finals[objective]=complete[-1][1];break
                checkpoints=[]
                for segment in segments:
                    for manifest in segment.glob('checkpoints/*/manifest.json'):
                        meta=json.loads(manifest.read_text());checkpoints.append((meta['step'],manifest.parent))
                resume=max(checkpoints,key=lambda p:p[0])[1] if checkpoints else None
                out=base/f'segment-{len(segments):03d}'
                args=['tapestry_m1.workflows.reranker_training','train',str(config_path),'--data-root',str(data_root),'--groups-root',str(root/'mined'),'--objective',objective,'--output',str(out)]
                if resume:args+=['--resume',str(resume)]
                else:args+=['--stop-after','5']
                invoke(args)
        # Freeze selections before inspecting either benchmark result.
        atomic_json(root/'selected_adapters.json',{o:{'checkpoint':s['best_checkpoint'],'training_summary_identity':s['identity']} for o,s in finals.items()})
        for objective in ['L0','L1']:
            state['phase']='evaluate-'+objective
            invoke(['tapestry_m1.workflows.au_peft','evaluate',str(config_path),'--data-root',str(data_root),'--run-root',str(root),'--objective',objective,'--adapter',finals[objective]['best_checkpoint']])
        from tapestry_m1.workflows.peft_evaluation import compare_objectives
        state['phase']='paired-comparison';status(root,state)
        compare_objectives(root/'evaluation/L0',root/'evaluation/L1',root/'comparisons')
        state.update(state='complete',phase='complete',finished_at=now())
        atomic_json(root/'complete.json',state)
    except BaseException as exc:
        state.update(state='failed',error_type=type(exc).__name__,next_action='Inspect logs/latest checkpoint; rerun the same frozen-source pipeline to continue.')
        raise
    finally:status(root,state)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','mine','pipeline','evaluate','preflight']);p.add_argument('config',type=Path)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--objective',choices=['L0','L1']);p.add_argument('--adapter',type=Path)
    a=p.parse_args(argv);c=json.loads(a.config.read_text());root=a.run_root
    try:
        if a.command=='prepare':
            result=prepare(a.data_root,root/'collection',[a.data_root/b['normalized'] for b in c['benchmarks']],c['seed']);print(json.dumps(result['counts'],indent=2))
        elif a.command=='preflight':preflight(a.config,a.data_root,root)
        elif a.command=='mine':
            from tapestry_m1.workflows.au_qa_mining import mine
            spec=json.loads((a.config.parent/'../../retrieval/qwen3-embedding-4b.json').resolve().read_text())
            recipe={'encoder':spec,'seed':c['seed'],'dtype':c['dtype'],'attention_backend':c['attention_backend'],'max_tokens':8192,
                'instruction':c['instruction'],'embedding_batch':2,'dimension':2560,'embedding_shard':16}
            mine(root/'collection',root/'mined',recipe,a.data_root)
        elif a.command=='evaluate':
            from tapestry_m1.workflows.peft_evaluation import evaluate_adapter
            evaluate_adapter(a.config,a.data_root,a.adapter,root/'evaluation'/a.objective,a.objective)
        else:pipeline(a.config,a.data_root,root)
        return 0
    except (ContractError,OSError,ValueError,subprocess.CalledProcessError) as exc:
        print('AU PEFT workflow stopped: '+str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
