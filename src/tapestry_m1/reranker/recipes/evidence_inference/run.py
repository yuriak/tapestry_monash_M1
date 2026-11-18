"""Standalone Qwen adaptation of the Evidence Inference identifier experiment.

Launch with one process per GPU (torchrun or Slurm); no site-specific paths.
"""
import argparse
from array import array
from collections import OrderedDict
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from .data import dump, sample_indices, length_batches
from .metrics import evaluate, paired_top1_ci

PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
INSTRUCTION = 'Identify a sentence providing evidence about the effect of the intervention compared with the comparator on the specified outcome.'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python':random.getstate(), 'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state()}


def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch']); torch.cuda.set_rng_state(s['cuda'])


class Scorer:
    def __init__(self, model_path, limit, gradient_checkpointing=False):
        self.path, self.limit = model_path, limit
        self.gradient_checkpointing = gradient_checkpointing
        self.token_cache = OrderedDict()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side='left')
        self.prefix = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.no, self.yes = [self.tokenizer.convert_tokens_to_ids(t) for t in ['no','yes']]
        assert self.no != self.yes
        self.audit = {'pairs':0, 'truncated':0, 'query_truncated':0, 'max_tokens':0}

    def load(self, adapter=None, train=False):
        model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.bfloat16,
                  attn_implementation='sdpa', local_files_only=True).cuda()
        model.config.use_cache = False
        if adapter:
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=train)
        elif train:
            model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
                       r=16, lora_alpha=32, lora_dropout=.05, target_modules=['q_proj','v_proj']))
        if train and self.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            model.enable_input_require_grads()
        return model

    def tokenized(self, pairs):
        keys = [tuple(p) for p in pairs]
        missing = list(dict.fromkeys(k for k in keys if k not in self.token_cache))
        budget = self.limit-len(self.prefix)-len(self.suffix)
        assert budget > 0
        if missing:
            text = [f'<Instruct>: {INSTRUCTION}\n<Query>: {q}\n<Document>: {sent}' for q,sent in missing]
            tokens = self.tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
            for (q,sent),t in zip(missing,tokens):
                qt = False
                if len(t)>budget:
                    qt = len(self.tokenizer.encode(f'<Instruct>: {INSTRUCTION}\n<Query>: {q}\n<Document>:',add_special_tokens=False))>=budget
                self.token_cache[(q,sent)] = (array('i',self.prefix+t[:budget]+self.suffix),
                    len(t)>budget, qt, len(t)+len(self.prefix)+len(self.suffix))
        result=[]
        for key in keys:
            ids,truncated,qt,length=self.token_cache[key]
            self.token_cache.move_to_end(key)
            self.audit['pairs']+=1;self.audit['truncated']+=truncated;self.audit['query_truncated']+=qt
            self.audit['max_tokens']=max(self.audit['max_tokens'],length)
            result.append(ids)
        while len(self.token_cache)>100000:self.token_cache.popitem(last=False)
        return result

    def score_ids(self, model, rows):
        x=self.tokenizer.pad({'input_ids':[list(t) for t in rows]},padding=True,return_tensors='pt').to('cuda')
        logits=model(**x,use_cache=False,logits_to_keep=1).logits[:,-1,:]
        return logits[:,self.yes].float()-logits[:,self.no].float()

    def score(self, model, pairs):
        return self.score_ids(model,self.tokenized(pairs))


def evaluation(model, scorer, records, documents, directory, rank, world, batch):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.eval()
    predictions = {}
    scorer.audit = {'pairs':0, 'truncated':0, 'query_truncated':0, 'max_tokens':0}
    start = time.monotonic()
    flat=[]
    for r in records[rank::world]:
        sentences=documents[r['docid']]
        predictions[r['id']]=[None]*len(sentences)
        tokens=scorer.tokenized([(r['query'],s['text']) for s in sentences])
        flat.extend((r['id'],i,t) for i,t in enumerate(tokens))
    # Stable length ordering reduces padding; scatter restores the original candidate order.
    flat.sort(key=lambda x:len(x[2]))
    with torch.inference_mode():
        for offset in range(0,len(flat),batch):
            chunk=flat[offset:offset+batch]
            values=scorer.score_ids(model,[x[2] for x in chunk]).cpu().tolist()
            if not all(math.isfinite(v) for v in values):raise RuntimeError('Nonfinite inference scores')
            for (qid,i,_),v in zip(chunk,values):predictions[qid][i]=v
            if offset//batch%20==0:
                print(f'EVAL {directory.name} rank={rank} pairs={min(offset+batch,len(flat))}/{len(flat)} batch={batch} seconds={time.monotonic()-start:.1f}',flush=True)
    dump(directory/f'rank-{rank}.json', {'predictions':predictions,'audit':scorer.audit})
    dist.barrier(device_ids=[torch.cuda.current_device()])
    result = None
    if rank == 0:
        merged, audits = {}, []
        for k in range(world):
            obj = json.loads((directory/f'rank-{k}.json').read_text())
            merged.update(obj['predictions']);audits.append(obj['audit'])
        assert len(merged)==len(records)
        result = {'primary':evaluate(records,merged),
                  'upstream_gold':evaluate(records,merged,'upstream_positive'),
                  'audit':{k:sum(a[k] for a in audits) for k in ['pairs','truncated','query_truncated']},
                  'seconds':time.monotonic()-start}
        dump(directory/'scores.json',merged);dump(directory/'metrics.json',result)
        for k in range(world): (directory/f'rank-{k}.json').unlink()
        print('EVAL_COMPLETE',str(directory),json.dumps(result),flush=True)
    values = [result]
    dist.broadcast_object_list(values,src=0)
    return values[0]


def wb_start(args, name, config):
    import wandb
    kwargs = dict(project=args.wandb_project, name=name, group=args.group, config=config,
                  id=hashlib.sha256(f'{args.group}:{name}'.encode()).hexdigest()[:16],
                  resume='allow', dir=str(Path(args.output)/'wandb'),
                  settings=wandb.Settings(init_timeout=45, disable_git=True))
    Path(kwargs['dir']).mkdir(parents=True,exist_ok=True)
    try:
        run = wandb.init(**kwargs)
    except Exception as exc:
        print('WANDB_ONLINE_UNAVAILABLE',type(exc).__name__,'recording offline',flush=True)
        kwargs['mode']='offline';run=wandb.init(**kwargs)
    print('WANDB',name,run.url or 'offline',flush=True)
    return run


def train_arm(args, scorer, records, documents, rank, world, ratio, seed):
    name = f'ei-{ratio}n-seed{seed}'
    out = Path(args.output)/name
    out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():
        return
    latest = out/'latest.pt'
    state = torch.load(latest,map_location='cpu',weights_only=False) if latest.exists() else None
    seed_all(seed)
    model = scorer.load(out/f'epoch-{state["epoch"]:02d}' if state else None,train=True)
    ddp = DDP(model, device_ids=[torch.cuda.current_device()],broadcast_buffers=False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=.01)
    train = [r for r in records['train'] if r['positive']]
    steps_per_epoch = math.ceil(len(train)/args.query_batch)
    total = steps_per_epoch*args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,math.ceil(.03*total),total)
    step, first, best, train_seconds = 0, 1, None, 0.
    if state:
        if state['world'] != world: raise ValueError('Resume world size differs')
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        step,first,best,train_seconds = state['step'],state['epoch']+1,state['best'],state['train_seconds']
        restore_rng(state['rng'][rank])
    else:
        seed_all(seed+rank)
    wb = wb_start(args,name,{'ratio':ratio,'seed':seed,'epochs':args.epochs,'lr':1e-5,
        'query_batch':args.query_batch,'world_size':world,'max_length':args.max_length,
        'train_micro':args.train_micro,'eval_batch':args.eval_batch,'train_tokens':args.train_tokens,
        'gradient_checkpointing':args.gradient_checkpointing,
        'lora_rank':16,'warmup_fraction':.03,'base_revision':args.base_revision}) if rank==0 else None
    for epoch in range(first,args.epochs+1):
        order = list(range(len(train)));random.Random(seed+epoch).shuffle(order)
        model.train()
        tic = time.monotonic()
        for offset in range(0,len(order),args.query_batch):
            chosen = order[offset:offset+args.query_batch]
            nreal = len(chosen)
            chosen += [chosen[0]]*(args.query_batch-nreal)
            examples = []
            for position in range(rank,args.query_batch,world):
                r = train[chosen[position]]
                sampled = sample_indices(r,len(documents[r['docid']]),ratio,seed,epoch)
                weight = (world/nreal/len(sampled)) if position<nreal else 0.
                for index,label in sampled:
                    examples.append((r['query'],documents[r['docid']][index]['text'],label,weight))
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.
            token_rows=scorer.tokenized([(x[0],x[1]) for x in examples])
            batches=list(length_batches([(x[2],x[3],t) for x,t in zip(examples,token_rows)],args.train_micro,args.train_tokens))
            for bi,chunk in enumerate(batches):
                final=bi==len(batches)-1
                with nullcontext() if final else ddp.no_sync():
                    scores=scorer.score_ids(ddp,[x[2] for x in chunk])
                    y=torch.tensor([x[0] for x in chunk],device='cuda',dtype=torch.float32)
                    w=torch.tensor([x[1] for x in chunk],device='cuda',dtype=torch.float32)
                    loss=(torch.nn.functional.binary_cross_entropy_with_logits(scores,y,reduction='none')*w).sum()
                    loss.backward()
                    loss_value+=loss.detach().item()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();step+=1
            value=torch.tensor(loss_value,device='cuda');dist.all_reduce(value);value/=world
            if step%10==0 or offset==0:
                if rank==0:
                    log={'step':step,'epoch':epoch,'loss':value.item(),'lr':scheduler.get_last_lr()[0],
                         'grad_norm':float(grad),'epoch_seconds':time.monotonic()-tic}
                    print('TRAIN',name,json.dumps(log),flush=True);wb.log(log,step=step)
                    with (out/'train.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
        train_seconds += time.monotonic()-tic
        ckpt=out/f'epoch-{epoch:02d}'
        if rank==0:model.save_pretrained(ckpt)
        dist.barrier(device_ids=[torch.cuda.current_device()])
        result=evaluation(model,scorer,records['validation'],documents,out/f'dev-{epoch:02d}',rank,world,args.eval_batch)
        key=(result['primary']['top1'],result['primary']['auroc'])
        if best is None or key>tuple(best['key']):best={'epoch':epoch,'key':list(key)}
        states=[None]*world;dist.all_gather_object(states,rng_state())
        if rank==0:
            payload={'epoch':epoch,'step':step,'world':world,'optimizer':optimizer.state_dict(),
                     'scheduler':scheduler.state_dict(),'rng':states,'best':best,'train_seconds':train_seconds}
            torch.save(payload,out/'latest.tmp');(out/'latest.tmp').replace(latest)
            dump(out/'selection.json',{'best':best,'last_epoch':epoch,'train_seconds':train_seconds})
            wb.log({f'dev/{k}':v for k,v in result['primary'].items()},step=step)
        dist.barrier(device_ids=[torch.cuda.current_device()])
    del ddp,model,optimizer,scheduler
    torch.cuda.empty_cache()
    model=scorer.load(out/f'epoch-{best["epoch"]:02d}')
    result=evaluation(model,scorer,records['test'],documents,out/'test',rank,world,args.eval_batch)
    if rank==0:
        dump(out/'completed.json',{'best':best,'train_seconds':train_seconds,'test':result})
        wb.summary.update({'selected_epoch':best['epoch'],'train_seconds':train_seconds,
                           **{f'test/{k}':v for k,v in result['primary'].items()}})
        wb.finish()
    del model
    torch.cuda.empty_cache();dist.barrier(device_ids=[torch.cuda.current_device()])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--query-batch',type=int,default=32)
    p.add_argument('--train-micro',type=int,default=128);p.add_argument('--eval-batch',type=int,default=512)
    p.add_argument('--gradient-checkpointing',action='store_true')
    p.add_argument('--train-tokens',type=int,default=16384)
    p.add_argument('--max-length',type=int,default=512);p.add_argument('--wandb-project',default='reranker-reproduction')
    p.add_argument('--group',required=True);p.add_argument('--base-revision',default='e61197ed45024b0ed8a2d74b80b4d909f1255473')
    args=p.parse_args()
    rank=int(os.environ.get('RANK',os.environ.get('SLURM_PROCID','0')))
    world=int(os.environ.get('WORLD_SIZE',os.environ.get('SLURM_NTASKS','1')))
    local=int(os.environ.get('LOCAL_RANK',os.environ.get('SLURM_LOCALID','0')))
    torch.cuda.set_device(local)
    dist.init_process_group('nccl',rank=rank,world_size=world,
                            device_id=torch.device('cuda',local),timeout=timedelta(minutes=60))
    dist.barrier(device_ids=[local])
    print(f'COLLECTIVE_READY rank={rank} local_device={local}',flush=True)
    assert args.query_batch%world==0
    print(f'WORKER rank={rank}/{world} gpu={torch.cuda.get_device_name()} BF16={torch.cuda.is_bf16_supported()}',flush=True)
    if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 GPU required')
    root=Path(args.data);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    documents=json.loads((root/'documents.json').read_text())
    records={s:json.loads((root/f'{s}.json').read_text()) for s in ['train','validation','test']}
    scorer=Scorer(args.model,args.max_length,args.gradient_checkpointing)
    if not (out/'zero-shot/completed.json').exists():
        model=scorer.load();wb=wb_start(args,'ei-zero-shot',{'max_length':args.max_length}) if rank==0 else None
        for split in ['validation','test']:
            result=evaluation(model,scorer,records[split],documents,out/'zero-shot'/split,rank,world,args.eval_batch)
            if rank==0:wb.log({f'{split}/{k}':v for k,v in result['primary'].items()})
        if rank==0:dump(out/'zero-shot/completed.json',{'complete':True});wb.finish()
        del model;torch.cuda.empty_cache()
    dist.barrier(device_ids=[torch.cuda.current_device()])
    for seed in [42,43,44]:
        for ratio in [2,4]:train_arm(args,scorer,records,documents,rank,world,ratio,seed)
    if rank==0:
        baseline=json.loads((out/'zero-shot/test/scores.json').read_text());summary={}
        for ratio in [2,4]:
            rows=[]
            for seed in [42,43,44]:
                arm=out/f'ei-{ratio}n-seed{seed}'
                result=json.loads((arm/'completed.json').read_text())
                predictions=json.loads((arm/'test/scores.json').read_text())
                result['top1_gain_ci']=paired_top1_ci(records['test'],baseline,predictions,seed=seed)
                rows.append(result)
            summary[f'{ratio}n']={'seeds':rows,'top1_mean':float(np.mean([r['test']['primary']['top1'] for r in rows])),
                                 'top1_std':float(np.std([r['test']['primary']['top1'] for r in rows],ddof=1))}
        dump(out/'summary.json',summary);dump(out/'completed.json',{'complete':True,'arms':6})
        print('BATCH1_COMPLETE',json.dumps(summary),flush=True)
    dist.barrier(device_ids=[torch.cuda.current_device()]);dist.destroy_process_group()


if __name__=='__main__':main()
