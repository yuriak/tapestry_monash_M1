"""Active standard-PEFT trainer: two real epochs with exact complete-group tails."""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time
from tapestry_m1.core.io import atomic_json,ContractError,fingerprint,records,dumps
from tapestry_m1.contracts.reranker_training import validate_groups as validate_synthetic
from tapestry_m1.contracts.reranker_training_v2 import validate_groups
from tapestry_m1.reranker.batches import EpochSampler
from tapestry_m1.reranker.losses import group_loss,relevance_logits
from tapestry_m1.reranker.qwen3 import reranker_text
from tapestry_m1.reranker.peft_backend import build_backend
from tapestry_m1.runtime.devices import gpu_runtime
from tapestry_m1.runtime.provenance import code_fingerprint
from tapestry_m1.runtime.training_checkpoint import save_peft_checkpoint,load_peft_state,restore_rng


def prepare_inputs(backend,groups):
    pairs=[reranker_text(backend.config['instruction'],g['query_text'],c['text']) for g in groups for c in g['candidates']]
    tokens=backend.tokenizer(pairs,padding=False,truncation=False,add_special_tokens=True)['input_ids']
    budget=backend.limit-len(backend.prefix)-len(backend.suffix)
    if budget<1:raise ContractError('context too short for scoring template')
    ids=[backend.prefix+t[:budget]+backend.suffix for t in tokens]
    items=[{'query_id':g['query_id'],'doc_id':c['doc_id'],'source_doc_id':c.get('source_doc_id'),
            'native_tokens':len(t)+len(backend.prefix)+len(backend.suffix),'retained_tokens':len(x),'truncated':len(t)>budget}
           for (g,c),t,x in zip([(g,c) for g in groups for c in g['candidates']],tokens,ids)]
    audit={'pairs':len(tokens),'truncated':sum(x['truncated'] for x in items),'items':items}
    return backend.tokenizer.pad({'input_ids':ids},padding=True,return_tensors='pt').to('cuda'),audit


def scores_for(backend,groups):
    inputs,audit=prepare_inputs(backend,groups)
    logits=backend.model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1,:]
    return relevance_logits(logits,backend.no,backend.yes).reshape(len(groups),4),audit


def evaluate(backend,groups,objective,margin,weight):
    import torch
    backend.model.eval();all_scores=[];truncated=0
    with torch.no_grad():
        for group in groups:
            scores,audit=scores_for(backend,[group]);all_scores.append(scores);truncated+=audit['truncated']
        scores=torch.cat(all_scores);loss,parts=group_loss(scores,objective,margin,weight)
    result={'loss':float(loss),'bce':float(parts['bce']),'margin':float(parts['margin']),
            'active_margin_fraction':float((margin-scores[:,:1]+scores[:,1:]>0).float().mean()),
            'pairwise_accuracy':float((scores[:,:1]>scores[:,1:]).float().mean()),'pairwise_comparisons':len(groups)*3,
            'finite_score_rate':float(torch.isfinite(scores).float().mean()),'truncated':truncated,'scores':scores.cpu().tolist()}
    backend.model.train();return result


def run(config_path,data_root,output,*,groups_root=None,objective=None,resume=None,stop_after=None,preflight=False):
    import torch
    from peft import get_peft_model_state_dict
    config_path,output=Path(config_path).resolve(),Path(output).resolve();config=json.loads(config_path.read_text())
    if config.get('schema_version')!='m1.reranker.run.v2':raise ContractError('v1 native runs are historical; use the PEFT v2 config')
    if objective:config['objective']=objective
    if config['objective'] not in ['L0','L1']:raise ContractError('unknown loss')
    config['seed']=int(config['seed'])
    if preflight:
        groups=list(records(config_path.parent/config['synthetic_groups']));validate_synthetic(groups,synthetic_only=True)
        input_manifest={'synthetic_sha256':fingerprint(config_path.parent/config['synthetic_groups'])}
    else:
        groups_root=Path(groups_root);input_manifest=json.loads((groups_root/'manifest.json').read_text())
        for name,sha in input_manifest['files'].items():
            if fingerprint(groups_root/name)!=sha:raise ContractError('frozen mining artifact changed')
        groups=validate_groups(list(records(groups_root/'groups.jsonl')))
    train=[g for g in groups if g['split']=='train'];dev=[g for g in groups if g['split']=='dev']
    if not train or not dev:raise ContractError('both train and dev required')
    effective,micro=config['effective_groups'],config['micro_groups']
    if type(effective) is not int or type(micro) is not int or min(effective,micro)<1 or micro>effective:raise ContractError('invalid group batch sizes')
    if config['epochs']!=2:raise ContractError('reviewed real recipe requires two complete epochs')
    steps_per_epoch=math.ceil(len(train)/effective);total_steps=config['epochs']*steps_per_epoch
    stop=min(total_steps,stop_after) if stop_after is not None else total_steps
    if output.exists() and any(output.iterdir()):raise ContractError('fresh output required; resume into a separate segment')
    hardware=gpu_runtime()
    if config['dtype']=='bfloat16' and not torch.cuda.is_bf16_supported():raise ContractError('BF16 unsupported')
    torch.set_num_threads(min(8,os.cpu_count() or 1));random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed']);torch.cuda.reset_peak_memory_stats()
    spec=json.loads((config_path.parent/config['model']).read_text())
    if spec['model_id']!='Qwen/Qwen3-Reranker-0.6B' or spec['revision']!='e61197ed45024b0ed8a2d74b80b4d909f1255473':raise ContractError('unreviewed base model')
    backend,model_identity,inventory=build_backend(config,spec,data_root,adapter=resume)
    parameters=[p for p in backend.model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,**config['optimizer']);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    sampler=EpochSampler(len(train),config['seed'])
    lock=Path(os.environ.get('M1_ENVIRONMENT_LOCK',Path(__file__).resolve().parents[3]/'uv.lock'))
    identity={'config':config,'input_manifest':input_manifest,'model':model_identity,'code':code_fingerprint(),
              'runtime_packages':{name:importlib.metadata.version(name) for name in ['torch','transformers','tokenizers','peft','accelerate','numpy','safetensors']},
              'environment_lock_sha256':fingerprint(lock),'peft':importlib.metadata.version('peft'),'steps_per_epoch':steps_per_epoch,'total_steps':total_steps}
    identity_hash=hashlib.sha256(dumps(identity).encode()).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'run.json',{'identity':identity_hash,'inputs':identity,'hardware':hardware,'lora':inventory,
        'token_ids':{'yes':backend.yes,'no':backend.no},'data_status':'synthetic-preflight' if preflight else 'real-au-qa-assumed-negatives'})
    step=0;best=float('inf');best_checkpoint=None;reload_error=None
    if resume:
        state,meta=load_peft_state(resume,identity_hash)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);sampler.load_state_dict(state['sampler'])
        step,best,best_checkpoint=meta['step'],meta['best_dev_loss'],meta['best_checkpoint']
        observed=evaluate(backend,dev,config['objective'],config['margin'],config['margin_weight'])
        reload_error=max(abs(a-b) for ra,rb in zip(observed['scores'],meta['dev']['scores']) for a,b in zip(ra,rb))
        if reload_error>1e-5:raise ContractError('standard PEFT scoring reload differs')
        restore_rng(state['rng'])
    initial=evaluate(backend,dev,config['objective'],config['margin'],config['margin_weight'])
    atomic_json(output/'initial.json',initial|{'resume_from_step':step,'reload_max_abs_error':reload_error})
    if preflight:
        preflight_check(backend,config,spec,data_root,output,optimizer,scheduler,sampler,identity_hash,identity,inventory,train,dev)
        return
    if not step<stop<=total_steps:raise ContractError('resume must advance within two-epoch budget')
    backend.model.train();initial_adapter={n:v.detach().cpu().clone() for n,v in get_peft_model_state_dict(backend.model).items()}
    start=time.monotonic();consumed=0;checks=[]
    with (output/'steps.jsonl').open('x') as log,(output/'lengths.jsonl').open('x') as lengths:
        while step<stop:
            tick=time.monotonic();indices=sampler.take_batch(effective);optimizer.zero_grad(set_to_none=True)
            total=bce=hinge=active=0.;truncated=0
            for offset in range(0,len(indices),micro):
                batch=[train[i] for i in indices[offset:offset+micro]];scores,audit=scores_for(backend,batch)
                loss,parts=group_loss(scores,config['objective'],config['margin'],config['margin_weight']);scale=len(batch)/len(indices)
                (loss*scale).backward();total+=float(loss.detach())*scale;bce+=float(parts['bce'].detach())*scale;hinge+=float(parts['margin'].detach())*scale
                active+=float((config['margin']-scores[:,:1]+scores[:,1:]>0).float().mean())*scale;truncated+=audit['truncated']
                for item in audit['items']:lengths.write(dumps(item|{'step':step+1})+'\n')
                del scores,loss,parts
            grads={n:float(p.grad.float().norm()) for n,p in backend.model.named_parameters() if p.requires_grad and p.grad is not None}
            if not grads or not all(math.isfinite(v) for v in grads.values()) or max(grads.values())<=0:raise ContractError('nonfinite or zero LoRA gradients')
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,config['max_grad_norm'],error_if_nonfinite=True));optimizer.step();scheduler.step();step+=1;consumed+=len(indices)
            record={'step':step,'epoch':sampler.epoch,'groups':len(indices),'query_ids':[train[i]['query_id'] for i in indices],'total':total,'bce':bce,'margin':hinge,'active_margin_fraction':active,
                'gradient_norm':norm,'lr':scheduler.get_last_lr()[0],'seconds':time.monotonic()-tick,'truncated_pairs':truncated,'peak_vram_bytes':torch.cuda.max_memory_allocated(),'finite_score_rate':1.}
            log.write(dumps(record)+'\n');log.flush();lengths.flush();print(dumps({k:record[k] for k in ['step','epoch','total','margin','active_margin_fraction','seconds']}),flush=True)
            if step%config['checkpoint_every']==0 or step%steps_per_epoch==0 or step==stop:
                dev_result=evaluate(backend,dev,config['objective'],config['margin'],config['margin_weight']);checkpoint=output/'checkpoints'/f'step-{step:06d}'
                if dev_result['bce']<best:best=dev_result['bce'];best_checkpoint=str(checkpoint)
                meta={'identity':identity_hash,'step':step,'epoch':sampler.epoch,'best_dev_loss':best,'best_checkpoint':best_checkpoint,'dev':dev_result,'inputs':identity,'consumed_groups_this_segment':consumed}
                save_peft_checkpoint(checkpoint,backend.model,optimizer,scheduler,sampler,meta)
                checks.append({'step':step,'checkpoint':str(checkpoint),'manifest_sha256':fingerprint(checkpoint/'manifest.json')})
                atomic_json(output/f'selection-{step:06d}.json',{'latest':str(checkpoint),'best':best_checkpoint,'best_dev_loss':best,'selection_source':'source-grouped dev BCE only, same criterion for L0/L1'})
    final=evaluate(backend,dev,config['objective'],config['margin'],config['margin_weight'])
    updated=sum(not torch.equal(v.detach().cpu(),initial_adapter[n]) for n,v in get_peft_model_state_dict(backend.model).items())
    summary={'status':'real-pair-objective-complete' if step==total_steps else 'real-training-segment-complete','objective':config['objective'],'step':step,'total_steps':total_steps,'epochs':config['epochs'],
        'consumed_groups_this_segment':consumed,'initial':initial,'final':final,'best_checkpoint':best_checkpoint,'latest_checkpoint':str(checkpoint),'reload_max_abs_error':reload_error,
        'checkpoint_checks':checks,'updated_adapter_tensors':updated,'last_gradient_norms':grads,'peak_vram_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.monotonic()-start,'identity':identity_hash}
    if updated==0:raise ContractError('optimizer did not change adapter tensors')
    atomic_json(output/'summary.json',summary);backend.close();return summary


def preflight_check(backend,config,spec,data_root,output,optimizer,scheduler,sampler,identity_hash,identity,inventory,train,dev):
    import torch
    parameters=[p for p in backend.model.parameters() if p.requires_grad];group=train[0]
    backend.model.eval()
    with torch.no_grad():
        with backend.model.disable_adapter():
            baseline,_=scores_for(backend,[group])
            original_probability,_=backend.score([group['query_text']]*4,[c['text'] for c in group['candidates']],batch_size=4)
        initial_adapter_scores,_=scores_for(backend,[group])
    base_error=float((baseline-initial_adapter_scores).abs().max())
    recipe_error=max(abs(float(x)-y) for x,y in zip(torch.sigmoid(baseline).flatten().cpu(),original_probability))
    if base_error>1e-5 or recipe_error>1e-6:raise ContractError('PEFT initial scoring differs from frozen zero-shot recipe')
    backend.model.train();optimizer.zero_grad(set_to_none=True)
    scores,_=scores_for(backend,[group]);shift=torch.zeros_like(scores);shift[0,0]=scores.detach()[0,0]-scores.detach()[0,1]+1
    forced=scores-shift;loss,parts=group_loss(forced,'L1',config['margin'],config['margin_weight'])
    parts['margin'].backward(retain_graph=True)
    margin_grad=max(float(p.grad.float().norm()) for p in parameters if p.grad is not None)
    if not math.isfinite(margin_grad) or margin_grad<=0:raise ContractError('constructed active hinge has no finite LoRA gradient')
    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();scheduler.step();sampler.take_batch(1)
    dev_result=evaluate(backend,dev,'L0',config['margin'],config['margin_weight']);checkpoint=output/'checkpoint'
    save_peft_checkpoint(checkpoint,backend.model,optimizer,scheduler,sampler,{'identity':identity_hash,'step':1,'dev':dev_result,'best_dev_loss':dev_result['loss'],'best_checkpoint':str(checkpoint),'inputs':identity})
    backend.close();backend,_,_=build_backend(config,spec,data_root,adapter=checkpoint)
    loaded=evaluate(backend,dev,'L0',config['margin'],config['margin_weight'])
    error=max(abs(a-b) for ra,rb in zip(dev_result['scores'],loaded['scores']) for a,b in zip(ra,rb))
    state,_=load_peft_state(checkpoint,identity_hash);parameters=[p for p in backend.model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,**config['optimizer']);optimizer.load_state_dict(state['optimizer']);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.);scheduler.load_state_dict(state['scheduler']);sampler.load_state_dict(state['sampler']);restore_rng(state['rng'])
    scores,_=scores_for(backend,[group]);next_loss,_=group_loss(scores,'L1');optimizer.zero_grad();next_loss.backward();optimizer.step();scheduler.step()
    resumed_optimizer_steps={int(v['step']) for v in optimizer.state.values() if 'step' in v}
    if resumed_optimizer_steps!={2}:raise ContractError('optimizer did not resume to step 2')
    long_group=dict(group);long_group['candidates']=[dict(c,text=c['text']+' synthetic '*2500) for c in group['candidates']]
    optimizer.zero_grad(set_to_none=True);long_scores,audit=scores_for(backend,[long_group]);long_loss,_=group_loss(long_scores,'L1');long_loss.backward()
    if error>1e-5:raise ContractError('independent PEFT reload failed')
    atomic_json(output/'preflight.json',{'status':'peft-preflight-complete','base_vs_zero_adapter_error':base_error,'inference_recipe_error':recipe_error,'constructed_margin':float(parts['margin'].detach()),'margin_gradient_norm':margin_grad,'reload_max_abs_error':error,'optimizer_steps_verified':[1,2],'resumed_optimizer_steps':sorted(resumed_optimizer_steps),
        'long_input_audit':audit,'peak_vram_bytes':torch.cuda.max_memory_allocated(),'lora':inventory})
    backend.close()
