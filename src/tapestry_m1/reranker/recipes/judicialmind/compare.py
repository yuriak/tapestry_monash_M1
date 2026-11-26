"""Distributed fixed-candidate evaluation of supplied reranker checkpoints."""
import argparse,gc,os
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from .data import load,dump
from .model import Tokens,Ranker
from .screen import eval_model,broadcast,root_exists,sync
from .topk import summarize

def main(a):
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30),device_id=torch.device('cuda',local))
    data=load(Path(a.data)/'data.json');tokens=Tokens(a.model,Path(a.data)/'tokens.pkl');frozen=load(Path(a.retrieval)/'frozen.json');prof=load(a.profile);root=Path(a.output)
    for name,adapter in load(a.comparisons).items():
        out=root/name
        if root_exists(out/'completed.json'):continue
        model=Ranker(a.model,adapter=adapter).cuda().eval()
        results={}
        for split in ['dev','test','tax']:
            marker=out/split/'metrics.json'
            if not root_exists(marker):eval_model(model,tokens,frozen[split],out/split,rank,world,prof['eval_tokens'][split],16384 if split=='tax' else 2048)
            if rank==0:
                if split == 'tax':
                    results['tax'] = load(marker)
                    continue
                sizes = [k for k in (100, 200, 500) if all(len(ds) >= k for ds in frozen[split]['candidates'].values())]
                results[split]=summarize(frozen[split],load(out/split/'scores.json'),{q:r['country'] for q,r in data[split].items()}, sizes)
        if rank==0:
            dump(out/'topk.json',results);dump(out/'completed.json',{'complete':True})
            import wandb
            wb=wandb.init(entity=a.entity,project=a.project,group=a.group,name=name,id=f'{a.group}-{name}',resume='allow')
            wb.log({f'{split}/top{k}/{country}/{metric}':value for split,ks in results.items() if split != "tax" for k,cs in ks.items() for country,ms in cs.items() for metric,value in ms.items()});wb.finish()
            print('COMPARISON COMPLETE',name,results,flush=True)
        del model;gc.collect();torch.cuda.empty_cache();sync()
    sync();dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['model','data','retrieval','profile','comparisons','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--entity');p.add_argument('--project');p.add_argument('--group');main(p.parse_args())
