"""Fetch pinned inputs with the user's own upstream permissions and native adapters."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

from .partition import read, write

IL = '28a7f9cdd7e9a33126e7025c9ae14bd2119bbc61'
EI = 'a661e8c14f973398380c8865cf2f27a535aaaf6d'
BIO = '8a845907dc1cff31d42fa6f7bb9c6eef5f3ae6f6'
MODEL = 'e61197ed45024b0ed8a2d74b80b4d909f1255473'
EMBED = '5cf2132abc99cad020ac570b19d031efec650f2b'


def script(scripts, recipe, stage, **paths):
    cmd = ['bash',str(scripts/recipe/(stage+'.sh'))]
    for key,value in paths.items():
        cmd.extend(['--'+key.replace('_','-'),str(value)])
    subprocess.run(cmd,check=True,env={**os.environ,'PYTHON':sys.executable})


def download(repo, rev, name, dest):
    path = hf_hub_download(repo_id=repo,filename=name,revision=rev,repo_type='dataset')
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        for item in (dest, Path(path)):
            with item.open('rb') as f:
                value = hashlib.file_digest(f,'sha256').hexdigest()
            if item == dest:
                expected = value
            elif expected != value:
                raise ValueError(f'Pinned source differs from existing {dest}')
        return
    tmp = dest.with_suffix(dest.suffix+'.partial')
    shutil.copyfile(path,tmp)
    tmp.replace(dest)


def acquire(dataset,root,scripts,config):
    raw,prepared,retrieval = root/'raw',root/'prepared',root/'retrieval'
    root.mkdir(parents=True,exist_ok=True)
    if dataset=='casehold':
        from tapestry_m1.data.casehold import FILES,REVISION
        for name,_,_ in FILES.values():
            download('casehold/casehold',REVISION,'data/all/'+name,raw/name)
        return
    if dataset=='evidence_inference':
        checkout = root/'upstream'
        if not checkout.exists():
            subprocess.run(['git','init',str(checkout)],check=True)
        git = ['git','-C',str(checkout)]
        if subprocess.run(git+['cat-file','-e',EI+'^{commit}'],capture_output=True).returncode:
            subprocess.run(git+['fetch','--depth=1','https://github.com/jayded/evidence-inference',EI],check=True)
        subprocess.run(git+['checkout','--detach',EI],check=True)
        if not raw.exists():
            raw.symlink_to('upstream/annotations',target_is_directory=True)
        if not (prepared/'manifest.json').exists():
            script(scripts,dataset,'prepare',source=raw,output=prepared,
                   sentence_index=config/'ei-sentences.json.gz')
        return
    model = snapshot_download('Qwen/Qwen3-Reranker-0.6B',revision=MODEL,
                              allow_patterns=['*.json','*.txt','*.model'])
    if dataset=='judicialmind':
        acquire_judicial(root,scripts,model)
        return
    if dataset=='bioasq':
        for folder,split in [('text-corpus','test'),('question-answer-passages','train'),
                             ('question-answer-passages','test')]:
            name = f'{folder}/{split}-00000-of-00001.parquet'
            download('enelpol/rag-mini-bioasq',BIO,name,prepared/'raw'/name)
        if not (prepared/'tokens.pkl').exists():
            script(scripts,'bioasq','prepare',data=prepared,model=model)
        # Only ordinal positions are distributed. Resolve against the pinned native IDs.
        data = read(prepared/'data.json')
        docs,queries = sorted(data['documents']),sorted(data['queries'])
        dictionary = hashlib.sha256(json.dumps([docs,queries]).encode()).hexdigest()
        with np.load(config/'bioasq-top1000.npz',allow_pickle=False) as index:
            if dictionary!=str(index['dictionary']) or index['order'].shape!=(len(queries),1000):
                raise ValueError('Unexpected BioASQ query or document dictionary')
            by_query = dict(zip(queries,index['order']))
        frozen = {}
        for split,qs in data['splits'].items():
            frozen[split] = {'gold':{q:data['gold'][q] for q in qs},
                'candidates':{q:[docs[i] for i in by_query[q]] for q in qs}}
        write(retrieval/'frozen.json',frozen)
        return
    for name in ('train_queries','dev_queries','test_queries','statute_candidates','precedent_candidates'):
        download('Exploration-Lab/IL-PCSR',IL,name+'.parquet',raw/(name+'.parquet'))
    if not (retrieval/'completed.json').exists():
        source = root/'whole'
        script(scripts,'il_pcsr','prepare',raw=raw,output=source,model=model)
        if not (prepared/'manifest.json').exists():
            script(scripts,'il_chunk','prepare',source=source,output=prepared,model=model)
        # Four logical shards preserve the historical length batches even on one GPU.
        shards = range(4)
        devices = min(4,int(os.environ.get('GPUS','1')))
        if devices < 1:
            raise ValueError('GPUS must be positive for cold IL retrieval')
        def stage(action, extra):
            def run_device(device):
                for rank in shards[device::devices]:
                    marker = retrieval/f'{action if action=="freeze" else "embed"}-{rank}.json'
                    if marker.exists():
                        continue
                    subprocess.run([sys.executable,'-m','tapestry_m1.reranker.recipes.il_chunk.retrieve',
                        action,'--rank',str(rank),'--local-rank',str(device),'--world','4',
                        '--data',str(prepared),'--output',str(retrieval),*extra],check=True)
            with ThreadPoolExecutor(max_workers=devices) as workers:
                list(workers.map(run_device,range(devices)))
        if not all((retrieval/f'freeze-{rank}.json').exists() for rank in shards):
            embed = snapshot_download('Qwen/Qwen3-Embedding-4B',revision=EMBED)
            stage('encode',['--model',embed,'--reranker',model])
            stage('freeze',[])
        subprocess.run([sys.executable,'-m','tapestry_m1.reranker.recipes.il_chunk.retrieve',
                        'assemble','--source',str(source),'--data',str(prepared),
                        '--output',str(retrieval),'--world','4'],check=True)



def acquire_judicial(root,scripts,model):
    from urllib.parse import quote
    raw,prepared,retrieval = root/'raw',root/'prepared',root/'retrieval'
    native = root/'tax'
    revision = '8ed53a3527c78823ba12e979134dcdac152f8e55'
    for prefix in ('english_world/au/australian_legal_qa','english_india/in/sc_judgments',
                   'english_india/in/rbi_documents'):
        for bucket in 'ABC':
            name = f'{prefix}_{bucket}.parquet'
            download('judicialmind/legal-training-dataset',revision,name,raw/name)
    qp = 'au-australian-tax-guidance-retrieval::'
    dp = 'au-australian-tax-guidance-retrieval-native::'
    for name,target in [('queries','queries'),('corpus','documents'),('default','qrels')]:
        source = native/'upstream'/(name+'.jsonl')
        download('isaacus/australian-tax-guidance-retrieval','982aa11494f04dee16cacf18ae1fbf59f6669845',
                 name+'.jsonl',source)
        with source.open() as reader,(native/(target+'.jsonl')).open('w') as writer:
            for line in reader:
                row=json.loads(line)
                if name=='default':
                    item={'query_id':qp+quote(row['query-id'],safe=''),
                          'doc_id':dp+quote(row['corpus-id'],safe=''),'relevance':row['score']}
                else:
                    key,prefix=('query_id',qp) if name=='queries' else ('doc_id',dp)
                    item={key:prefix+quote(row['_id'],safe=''),'text':row['text']}
                    if name=='queries':
                        item['query_text']=item.pop('text')
                writer.write(json.dumps(item)+'\n')
    if not (prepared/'audit.json').exists():
        script(scripts,'judicialmind','prepare',raw=raw,tax=native,output=prepared)
    if not (prepared/'tokens.pkl').exists():
        script(scripts,'judicialmind','cache_tokens',model=model,data=prepared)
    if not (retrieval/'frozen.json').exists():
        module='tapestry_m1.reranker.recipes.judicialmind.retrieve'
        devices=min(4,int(os.environ.get('GPUS','1')))
        if devices<1:
            raise ValueError('GPUS must be positive for cold JudicialMind retrieval')
        if not all((retrieval/f'rank-{r}.json').exists() for r in range(4)):
            embed=snapshot_download('Qwen/Qwen3-Embedding-4B',revision=EMBED)
            def encode_device(device):
                for rank in range(device,4,devices):
                    subprocess.run([sys.executable,'-m',module,'encode','--world','4','--rank',str(rank),
                        '--local-rank',str(device),'--model',embed,'--data',str(prepared),
                        '--output',str(retrieval)],check=True)
            with ThreadPoolExecutor(max_workers=devices) as workers:
                list(workers.map(encode_device,range(devices)))
        subprocess.run([sys.executable,'-m',module,'freeze','--world','4','--data',str(prepared),
                        '--output',str(retrieval),'--top','500'],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',choices=['casehold','il_pcsr','evidence_inference','bioasq','judicialmind'])
    for name in ['root','scripts','config']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    acquire(a.dataset,a.root,a.scripts,a.config)
