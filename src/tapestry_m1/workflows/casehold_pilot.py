"""Resumable native-five-choice pilot: baseline -> LoRA -> dev selection -> test.

No retrieval or loss sweep. Raw data and all runtime products are caller-owned.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

from tapestry_m1.core.io import ContractError, atomic_json, dumps, fingerprint
from tapestry_m1.data.casehold import FILES, load_split
from tapestry_m1.reranker.choice import aggregate, group_ce, lr_multiplier, metrics, select, stop_reason
from tapestry_m1.reranker.peft_backend import build_backend
from tapestry_m1.reranker.peft_training import prepare_inputs
from tapestry_m1.reranker.qwen3 import Qwen3Reranker, yes_logit_difference
from tapestry_m1.runtime.assets import verify_model
from tapestry_m1.runtime.devices import gpu_runtime
from tapestry_m1.runtime.provenance import code_fingerprint
from tapestry_m1.runtime.training_checkpoint import load_peft_state, restore_rng, rng_state, save_peft_checkpoint


def log(event, **fields):
    print(dumps({'time': datetime.now(timezone.utc).isoformat(), 'event': event, **fields}), flush=True)


def read(path):
    return json.loads(Path(path).read_text())


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def weight_hash(model, adapters):
    import torch
    sha = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        if ('.lora_' in name) == adapters:
            sha.update(name.encode()); sha.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return sha.hexdigest()


def scores_for(backend, groups):
    inputs, audit = prepare_inputs(backend, groups)
    logits = backend.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
    return yes_logit_difference(logits, backend.no, backend.yes).reshape(len(groups), 5), audit


class Schedule:
    def __init__(self, size, seed):
        self.order = list(range(size)); random.Random(seed).shuffle(self.order); self.cursor = 0

    def take(self, groups, size=8):
        if self.cursor+size > len(self.order):
            raise ContractError('refuse repeated/cycled training groups')
        ids = self.order[self.cursor:self.cursor+size]; self.cursor += size
        return [groups[i] for i in ids]

    def state_dict(self):
        return {'order': self.order, 'cursor': self.cursor}

    def load_state_dict(self, state):
        if state['order'] != self.order or not 0 <= state['cursor'] <= len(self.order):
            raise ContractError('schedule changed on resume')
        self.cursor = state['cursor']


def optimizer_for(backend, config):
    import torch
    parameters = [p for p in backend.model.parameters() if p.requires_grad]
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise ContractError('expected FP32 LoRA trainable parameters')
    optimizer = torch.optim.AdamW(parameters, **config['optimizer'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: lr_multiplier(n+1))
    return optimizer, scheduler


def update(backend, groups, optimizer, scheduler, step, config):
    import torch
    backend.model.train(); optimizer.zero_grad(set_to_none=True)
    rows, losses = [], []
    lr = optimizer.param_groups[0]['lr']
    for group in groups:
        scores, _ = scores_for(backend, [group]); loss = group_ce(scores, [group['gold']])
        if not torch.isfinite(loss):
            raise ContractError('nonfinite training loss')
        (loss/len(groups)).backward()
        rows.append(metrics(scores.detach()[0].cpu().tolist(), group['gold'])); losses.append(loss.item())
    params = [p for p in backend.model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(params, config['max_grad_norm'], error_if_nonfinite=True)
    if any(p.grad is not None for p in backend.model.parameters() if not p.requires_grad):
        raise ContractError('base parameter received a gradient')
    optimizer.step(); scheduler.step()
    record = {'step': step, 'loss': sum(losses)/len(losses), 'lr': lr, 'gradient_norm': norm.item(), **aggregate(rows)}
    log('train_update', **record)
    return record


def parity(backend, groups, config):
    """Bounded initialization check; no optimization and no test inputs."""
    import torch
    sample = groups[:20]
    queries = [g['query_text'] for g in sample for _ in g['candidates']]
    docs = [c['text'] for g in sample for c in g['candidates']]
    backend.model.eval()
    single, _ = backend.score_logits(queries, docs, batch_size=1)
    batched, _ = backend.score_logits(queries, docs, batch_size=config['reranker_batch'])
    training = []
    with torch.no_grad():
        for g in sample:
            s, _ = scores_for(backend, [g]); training.extend(s[0].cpu().tolist())
    top = lambda arr: [max(range(5), key=lambda j: (arr[i+j], -j)) for i in range(0,len(arr),5)]
    result = {}
    for name, values in [('batch',batched),('training_path',training)]:
        error = max(abs(a-b) for a,b in zip(single,values))
        agreement = sum(a==b for a,b in zip(top(single),top(values)))/len(sample)
        result[name] = {'max_abs_error':error, 'top1_agreement':agreement}
        if error > config['parity_score_atol'] or agreement < config['parity_rank_agreement_min']:
            raise ContractError('initial scoring parity failed: '+dumps(result))
    result['pairs'] = len(single)
    result['batch1_scores'] = single
    return result


def evaluate(backend, groups, directory, identity, model_key, batch_size):
    """Atomic 32-query chunks: completed chunks never score again on retry."""
    directory.mkdir(parents=True, exist_ok=True)
    contract = {'identity':identity, 'model_key':model_key, 'query_ids':[g['query_id'] for g in groups], 'batch_size':batch_size}
    atomic_json(directory/'contract.json', contract)
    backend.model.eval(); rows = []; started = time.monotonic(); scored = 0
    for start in range(0,len(groups),32):
        chunk = groups[start:start+32]; path = directory/f'chunk-{start:06d}.json'
        if path.exists():
            part = read(path)
        else:
            queries = [g['query_text'] for g in chunk for _ in g['candidates']]
            docs = [c['text'] for g in chunk for c in g['candidates']]
            scores, audits = backend.score_logits(queries, docs, batch_size=batch_size)
            part = []
            for i,g in enumerate(chunk):
                values = scores[i*5:i*5+5]
                part.append({'query_id':g['query_id'], 'gold':g['gold'], 'scores':values,
                             'truncated_pairs':sum(a['truncated'] for a in audits[i*5:i*5+5]), **metrics(values,g['gold'])})
            atomic_json(path, part); scored += len(chunk)
        if [r['query_id'] for r in part] != [g['query_id'] for g in chunk] or any(r['gold'] != g['gold'] for r,g in zip(part,chunk)):
            raise ContractError('cached evaluation row identity mismatch')
        for row in part:
            if any(abs(metrics(row['scores'],row['gold'])[k]-row[k]) > 1e-12 for k in metrics(row['scores'],row['gold'])):
                raise ContractError('cached metrics do not reproduce')
        rows.extend(part)
        if scored and (start % 128 == 0 or len(rows) == len(groups)):
            elapsed = time.monotonic()-started
            log('evaluation_progress', evaluation=directory.name, complete=len(rows), total=len(groups),
                pairs_per_second=5*scored/elapsed, estimated_remaining_seconds=(len(groups)-len(rows))*elapsed/scored)
    summary = aggregate(rows)
    summary['truncated_pairs'] = sum(r['truncated_pairs'] for r in rows)
    summary['chunks_sha256'] = {p.name:fingerprint(p) for p in sorted(directory.glob('chunk-*.json'))}
    atomic_json(directory/'summary.json', summary)
    log('evaluation_complete', evaluation=directory.name, **{k:v for k,v in summary.items() if k!='chunks_sha256'})
    return summary, rows


def smoke(backend, config, groups, output, identity):
    """Disposable update and exact optimizer/sampler/RNG continuation check."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    optimizer, scheduler = optimizer_for(backend,config); schedule = Schedule(len(groups),config['seed'])
    initial_adapter = weight_hash(backend.model, True); base = weight_hash(backend.model, False)
    update(backend, schedule.take(groups), optimizer, scheduler, 1, config)
    after = weight_hash(backend.model, True)
    if after == initial_adapter or weight_hash(backend.model,False) != base:
        raise ContractError('smoke failed: adapter unchanged or base modified')
    checkpoint = output/'smoke-checkpoint'
    if not checkpoint.exists():
        save_peft_checkpoint(checkpoint,backend.model,optimizer,scheduler,schedule,{'identity':identity,'step':1,'adapter_hash':after})
    state, meta = load_peft_state(checkpoint,identity)
    if meta['adapter_hash'] != after:
        raise ContractError('smoke checkpoint differs from deterministic replay')
    # Expected second update from the frozen checkpoint (also valid after a partial retry).
    restore_rng(state['rng'])
    expected = update(backend,schedule.take(groups),optimizer,scheduler,2,config)
    expected_hash = weight_hash(backend.model,True)
    set_peft_model_state_dict(backend.model,load_file(str(checkpoint/'adapter_model.safetensors')))
    optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler']); schedule.load_state_dict(state['sampler']); restore_rng(state['rng'])
    actual = update(backend,schedule.take(groups),optimizer,scheduler,2,config)
    if weight_hash(backend.model,True) != expected_hash or actual['loss'] != expected['loss'] or weight_hash(backend.model,False) != base:
        raise ContractError('exact checkpoint continuation failed')
    atomic_json(output/'smoke.json',{'identity':identity,'nonzero_adapter_update':True,'base_frozen':True,'exact_resume':True})
    log('smoke_passed', exact_resume=True)


def independent_final_audit(base_rows, adapted_rows, config):
    import numpy as np
    if [r['query_id'] for r in base_rows] != [r['query_id'] for r in adapted_rows]:
        raise ContractError('unpaired final predictions')
    # Independent rank implementation: count candidates preceding gold, no argsort.
    arrays = []
    for rows in [base_rows,adapted_rows]:
        result = []
        for r in rows:
            s,y = r['scores'],r['gold']
            rank = 1+sum(v>s[y] or (v==s[y] and j<y) for j,v in enumerate(s))
            values = [float(rank==1),1/rank,1/np.log2(rank+1)]
            if any(abs(a-r[k]) > 1e-12 for a,k in zip(values,['accuracy','mrr','ndcg'])):
                raise ContractError('independent final metric audit failed')
            result.append(values)
        arrays.append(np.array(result))
    delta = arrays[1]-arrays[0]; rng = np.random.default_rng(config['seed'])
    bootstrap = np.empty((config['bootstrap_replicates'],3))
    for start in range(0,len(bootstrap),100):
        count = min(100,len(bootstrap)-start)
        ids = rng.integers(0,len(delta),size=(count,len(delta)))
        bootstrap[start:start+count] = delta[ids].mean(axis=1)
    return {'count':len(delta),'replicates':len(bootstrap),'seed':config['seed'],'independent_rank_audit':True,
            'paired_delta':{k:{'mean':float(delta[:,i].mean()),'ci95':np.quantile(bootstrap[:,i],[.025,.975]).tolist()}
                            for i,k in enumerate(['accuracy','mrr','ndcg'])}}


def run(config_path, raw, data_root, output, preparation):
    import torch
    config_path, raw, output = Path(config_path),Path(raw),Path(output)
    config = read(config_path); output.mkdir(parents=True,exist_ok=True)
    if config['schema_version'] != 'm1.casehold.pilot.v1' or config['checkpoints'] != [5,20,60,120] or config['max_steps'] != 120 or config['effective_groups'] != 8 or config['micro_groups'] != 1:
        raise ContractError('unreviewed pilot schedule')
    hardware = gpu_runtime()
    if not torch.cuda.is_bf16_supported() or torch.cuda.get_device_capability(0)[0] < 8 or config['dtype'] != 'bfloat16':
        raise ContractError('this frozen pilot requires native BF16')
    torch.set_num_threads(2)
    spec = read(config_path.parent/config['model']); model_dir, verified = verify_model(spec,data_root)
    if spec['revision'] != 'e61197ed45024b0ed8a2d74b80b4d909f1255473':
        raise ContractError('unexpected model revision')
    prep = read(preparation)
    if prep['config_sha256'] != fingerprint(config_path) or prep['tokenization']['truncated_fraction'] > config['truncation_fraction_limit']:
        raise ContractError('pretraining tokenization/config gate failed')
    splits = {s:load_split(raw,s) for s in FILES}
    identity_data = {'config':config,'preparation':prep,'code':code_fingerprint(),'model':verified,
        'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft','accelerate','numpy','tokenizers','safetensors']},
        'lock_sha256':fingerprint(os.environ['M1_ENVIRONMENT_LOCK']), 'gpu':hardware['gpu']}
    identity = hashlib.sha256(dumps(identity_data).encode()).hexdigest()
    atomic_json(output/'identity.json',{'identity':identity, 'inputs':identity_data})
    log('run_start', identity=identity, hardware=hardware, resumed=(output/'base-validation').exists())
    seed_all(config['seed'])
    backend = Qwen3Reranker(model_dir,config)
    if {'yes':backend.yes,'no':backend.no} != prep['tokenization']['token_ids']:
        raise ContractError('token IDs changed since CPU preparation')
    if not (output/'parity.json').exists():
        atomic_json(output/'parity.json',parity(backend,splits['validation'],config))
    base, _ = evaluate(backend,splits['validation'],output/'base-validation',identity,'base',config['reranker_batch'])
    backend.close()
    # No optimizer update, including disposable smoke, is allowed before full baseline.
    if not (output/'smoke.json').exists():
        seed_all(config['seed']); backend,_,_ = build_backend(config,spec,data_root)
        parity_result = parity(backend,splits['validation'],config)
        base_scores = read(output/'parity.json')['batch1_scores']
        parity_result['zero_adapter_vs_base_max_error'] = max(abs(a-b) for a,b in zip(base_scores,parity_result['batch1_scores']))
        if parity_result['zero_adapter_vs_base_max_error'] > config['parity_score_atol']:
            raise ContractError('zero adapter differs from base scorer')
        atomic_json(output/'adapter-initial-parity.json',parity_result)
        smoke(backend,config,splits['train'],output,identity); backend.close()
    history = []; reason = None
    existing = [s for s in config['checkpoints'] if (output/f'checkpoint-{s}'/'manifest.json').exists()]
    latest = max(existing, default=0)
    seed_all(config['seed'])
    backend,_,inventory = build_backend(config,spec,data_root,adapter=output/f'checkpoint-{latest}' if latest else None)
    optimizer,scheduler = optimizer_for(backend,config); schedule = Schedule(len(splits['train']),config['seed'])
    frozen_base = weight_hash(backend.model,False)
    if latest:
        state,meta = load_peft_state(output/f'checkpoint-{latest}',identity)
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler']); schedule.load_state_dict(state['sampler']); restore_rng(state['rng'])
    for target in config['checkpoints']:
        checkpoint = output/f'checkpoint-{target}'
        if target < latest:
            if not (output/f'validation-{target}'/'summary.json').exists():
                raise ContractError('later checkpoint exists without earlier completed validation')
            summary = read(output/f'validation-{target}'/'summary.json')
            meta = read(checkpoint/'manifest.json')
        else:
            losses = []
            for step in range(latest+1,target+1):
                record = update(backend,schedule.take(splits['train']),optimizer,scheduler,step,config)
                losses.append(record['loss'])
                # Append attempts: a failed interval may replay, distinguished by run_start in stdout.
                with (output/'training.jsonl').open('a') as handle:
                    handle.write(dumps(record)+'\n'); handle.flush()
            if not checkpoint.exists():
                meta = save_peft_checkpoint(checkpoint,backend.model,optimizer,scheduler,schedule,
                    {'identity':identity,'step':target,'adapter_hash':weight_hash(backend.model,True),
                     'train_loss':sum(losses[-5:])/len(losses[-5:]),'lora':inventory})
            else:
                _,meta = load_peft_state(checkpoint,identity)
            saved_rng = rng_state()
            summary,_ = evaluate(backend,splits['validation'],output/f'validation-{target}',identity,
                                meta['adapter_hash'],config['reranker_batch'])
            restore_rng(saved_rng); latest = target
        history.append({k:summary[k] for k in ['accuracy','mrr','ndcg','margin','dispersion','count']} |
                       {'step':target,'train_loss':meta['train_loss'],'adapter_hash':meta['adapter_hash']})
        reason = stop_reason(base,history)
        log('checkpoint_decision', checkpoint=target, stop_reason=reason, validation=history[-1])
        if reason:
            break
    if weight_hash(backend.model,False) != frozen_base:
        raise ContractError('base weights changed during training')
    chosen = select(history)
    selection = {'identity':identity,'selected':chosen,'validation_history':history,'stop_reason':reason,
                 'rule':'full-validation accuracy, then MRR, then earlier step','test_not_used_for_selection':True}
    atomic_json(output/'selection.json',selection)
    backend.close(); del optimizer,scheduler
    log('selection_frozen', step=chosen['step'], stop_reason=reason)
    # This timestamped marker is committed to disk before either final test scorer starts.
    marker = output/'test-start.json'
    if not marker.exists():
        atomic_json(marker,{'selection_sha256':fingerprint(output/'selection.json'),'time':datetime.now(timezone.utc).isoformat()})
    if read(marker)['selection_sha256'] != fingerprint(output/'selection.json'):
        raise ContractError('selection changed after test was opened')
    backend = Qwen3Reranker(model_dir,config)
    base_test,base_rows = evaluate(backend,splits['test'],output/'base-test',identity,'base',config['reranker_batch']); backend.close()
    backend,_,_ = build_backend(config,spec,data_root,adapter=output/f"checkpoint-{chosen['step']}",trainable=False)
    adapted_test,adapted_rows = evaluate(backend,splits['test'],output/'selected-test',identity,chosen['adapter_hash'],config['reranker_batch']); backend.close()
    audit = independent_final_audit(base_rows,adapted_rows,config)
    result = {'identity':identity,'selection':selection,'base_validation':base,'base_test':base_test,'selected_test':adapted_test,'audit':audit}
    atomic_json(output/'results.json',result)
    atomic_json(output/'completed.json',{'identity':identity,'results_sha256':fingerprint(output/'results.json'),'status':'complete_and_metric_audited'})
    log('pilot_complete', selected_step=chosen['step'], baseline_accuracy=base_test['accuracy'], adapted_accuracy=adapted_test['accuracy'], paired=audit['paired_delta'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['config','raw','data-root','output','preparation']:
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    import fcntl
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a.config,a.raw,a.data_root,a.output,a.preparation)


if __name__ == '__main__':
    main()
