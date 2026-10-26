"""Allocated local LoRA training on gated, fixed four-candidate query groups."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from tapestry_m1.core.io import atomic_json, ContractError, fingerprint, records, dumps
from tapestry_m1.contracts.reranker_training import validate_groups
from tapestry_m1.reranker.batches import GroupSampler
from tapestry_m1.reranker.losses import group_loss, relevance_logits
from tapestry_m1.reranker.qwen3 import Qwen3Reranker, reranker_text
from tapestry_m1.runtime.assets import verify_model
from tapestry_m1.runtime.devices import gpu_runtime
from tapestry_m1.runtime.provenance import code_fingerprint
from tapestry_m1.runtime.training_checkpoint import save_checkpoint, load_checkpoint, restore_rng


def prepare_inputs(backend, groups):
    pairs = [reranker_text(backend.config['instruction'], group['query_text'], c['text'])
             for group in groups for c in group['candidates']]
    tokens = backend.tokenizer(pairs, padding=False, truncation=False, add_special_tokens=True)['input_ids']
    budget = backend.limit - len(backend.prefix) - len(backend.suffix)
    if budget < 1:
        raise ContractError('context too short for scoring prefix/suffix')
    audit = {'pairs': len(tokens), 'truncated': sum(len(t) > budget for t in tokens),
             'max_native_tokens': max(len(t)+len(backend.prefix)+len(backend.suffix) for t in tokens)}
    ids = [backend.prefix+t[:budget]+backend.suffix for t in tokens]
    inputs = backend.tokenizer.pad({'input_ids': ids}, padding=True, return_tensors='pt').to('cuda')
    return inputs, audit


def scores_for(backend, groups):
    inputs, audit = prepare_inputs(backend, groups)
    logits = backend.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
    scores = relevance_logits(logits, backend.no, backend.yes).reshape(len(groups), 4)
    return scores, audit


def evaluate(backend, groups, objective, margin, weight):
    import torch
    backend.model.eval()
    all_scores, truncations = [], 0
    with torch.no_grad():
        for group in groups:
            scores, audit = scores_for(backend, [group])
            all_scores.append(scores.detach())
            truncations += audit['truncated']
        scores = torch.cat(all_scores)
        loss, parts = group_loss(scores, objective, margin, weight)
    backend.model.train()
    return {'loss': float(loss), 'bce': float(parts['bce']), 'margin': float(parts['margin']),
            'pairwise_accuracy': float((scores[:, :1] > scores[:, 1:]).float().mean()),
            'pairwise_comparisons': len(groups)*3, 'finite_score_rate': float(torch.isfinite(scores).float().mean()),
            'truncated': truncations, 'scores': scores.cpu().tolist()}


def run(config_path, data_root, output, *, objective=None, resume=None, stop_after=None, preflight=False):
    import torch
    from tapestry_m1.reranker.lora import inject_lora, adapter_state, load_adapter
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = json.loads(config_path.read_text())
    if objective:
        config['objective'] = objective
    if config['objective'] not in ('L0', 'L1') or config['schema_version'] != 'm1.reranker.run.v1':
        raise ContractError('unsupported training configuration')
    if config['data_gate'] != 'synthetic-only':
        raise ContractError('real-data gate requires a separately reviewed eligibility contract; not enabled')
    groups_path = (config_path.parent/config['groups']).resolve()
    groups = validate_groups(list(records(groups_path)), synthetic_only=True)
    train = [g for g in groups if g['split'] == 'train']
    dev = [g for g in groups if g['split'] == 'dev']
    if not dev:
        raise ContractError('independent synthetic dev groups required')
    effective, micro = config['effective_groups'], config['micro_groups']
    if type(effective) is not int or type(micro) is not int or min(effective, micro) < 1 or effective % micro:
        raise ContractError('effective group count must divide into complete microbatches')
    if not 100 <= config['steps'] <= 300 or not 0 < config['checkpoint_every'] <= config['steps']:
        raise ContractError('bounded warmup requires 100..300 steps and valid checkpoint interval')
    stop = min(config['steps'], stop_after) if stop_after is not None else config['steps']
    if stop < 1:
        raise ContractError('positive stop step required')
    model_spec = json.loads((config_path.parent/config['model']).read_text())
    if model_spec['model_id'] != 'Qwen/Qwen3-Reranker-0.6B' or model_spec['revision'] != 'e61197ed45024b0ed8a2d74b80b4d909f1255473':
        raise ContractError('this warmup requires the reviewed 0.6B revision')
    if output.exists() and any(output.iterdir()):
        raise ContractError('output must be a fresh directory; resume into a separate output')
    hardware = gpu_runtime()
    if config['dtype'] == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise ContractError('BF16 unavailable; record a reviewed precision fallback before running')
    random.seed(config['seed']); torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.cuda.reset_peak_memory_stats()
    model_dir, verified = verify_model(model_spec, data_root)
    backend = Qwen3Reranker(model_dir, config)
    inventory = inject_lora(backend.model, **config['lora'])
    backend.model.train()
    parameters = [p for p in backend.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, **config['optimizer'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    sampler = GroupSampler(len(train), config['seed'])
    lock_path = Path(os.environ.get('M1_ENVIRONMENT_LOCK', Path(__file__).resolve().parents[3]/'uv.lock'))
    identity = {'config': config, 'groups_sha256': fingerprint(groups_path), 'model': verified,
                'code': code_fingerprint(), 'environment_lock_sha256': fingerprint(lock_path)}
    identity_hash = hashlib.sha256(dumps(identity).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'run.json', {'identity': identity_hash, 'inputs': identity, 'hardware': hardware,
               'lora': inventory, 'token_ids': {'no': backend.no, 'yes': backend.yes},
               'data_status': 'synthetic-only / real-data-blocked'})
    step, best, best_step, reload_difference = 0, float('inf'), None, None
    if resume:
        adapter, state, saved = load_checkpoint(resume, identity_hash)
        load_adapter(backend.model, adapter)
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        sampler.load_state_dict(state['sampler'])
        step, best, best_step = saved['step'], saved['best_dev_loss'], saved['best_step']
        observed = evaluate(backend, dev, config['objective'], config['margin'], config['margin_weight'])
        reload_difference = max(abs(a-b) for row_a,row_b in zip(observed['scores'], saved['dev']['scores']) for a,b in zip(row_a,row_b))
        if reload_difference > 1e-5:
            raise ContractError('checkpoint scoring reload differs')
        restore_rng(state['rng'])
    initial = evaluate(backend, dev, config['objective'], config['margin'], config['margin_weight'])
    # eval must not consume stochastic training order; dropout is disabled while evaluating.
    atomic_json(output/'initial.json', initial | {'resume_from_step': step, 'reload_max_abs_error': reload_difference})
    if preflight:
        # Same scoring path at the full configured context, wholly synthetic text.
        long_group = dict(train[0]); long_group['candidates'] = [dict(c, text=c['text']+' synthetic '*2500) for c in train[0]['candidates']]
        long_scores, long_audit = scores_for(backend, [long_group])
        long_loss, _ = group_loss(long_scores, config['objective'], config['margin'], config['margin_weight'])
        long_loss.backward()
        norms = [float(p.grad.float().norm()) for p in parameters if p.grad is not None]
        if not norms or not all(math.isfinite(v) for v in norms) or max(norms) <= 0:
            raise ContractError('preflight has no finite nonzero LoRA gradients')
        atomic_json(output/'preflight.json', {'loss': float(long_loss.detach()), 'audit': long_audit,
                    'gradient_norm_max': max(norms), 'peak_vram_bytes': torch.cuda.max_memory_allocated()})
        backend.close()
        return
    if step >= stop:
        raise ContractError('resume must advance optimizer step')
    initial_adapter = adapter_state(backend.model)
    start = time.monotonic(); consumed = 0; first_gradients = None; checkpoint_checks = []
    with (output/'steps.jsonl').open('x') as log:
        while step < stop:
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            indices = sampler.take(effective)
            bce_value = hinge_value = total_value = 0.; truncated = 0
            for offset in range(0, effective, micro):
                batch = [train[i] for i in indices[offset:offset+micro]]
                scores, audit = scores_for(backend, batch)
                loss, parts = group_loss(scores, config['objective'], config['margin'], config['margin_weight'])
                (loss * (micro/effective)).backward()
                total_value += float(loss.detach())*(micro/effective)
                bce_value += float(parts['bce'].detach())*(micro/effective)
                hinge_value += float(parts['margin'].detach())*(micro/effective)
                truncated += audit['truncated']
                del scores, loss, parts
            grads = {name: float(p.grad.float().norm()) for name,p in backend.model.named_parameters() if p.requires_grad and p.grad is not None}
            if not grads or not all(math.isfinite(v) for v in grads.values()) or max(grads.values()) <= 0:
                raise ContractError('nonfinite/zero training gradients')
            if first_gradients is None:
                first_gradients = grads
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, config['max_grad_norm'], error_if_nonfinite=True))
            optimizer.step(); scheduler.step(); step += 1; consumed += effective
            record = {'step': step, 'query_ids': [train[i]['query_id'] for i in indices], 'groups': effective,
                      'total': total_value, 'bce': bce_value, 'margin': hinge_value, 'gradient_norm': grad_norm,
                      'lr': scheduler.get_last_lr()[0], 'seconds': time.monotonic()-tick, 'truncated': truncated,
                      'peak_vram_bytes': torch.cuda.max_memory_allocated(), 'finite_score_rate': 1.}
            log.write(dumps(record)+'\n'); log.flush()
            print(dumps({k:record[k] for k in ('step','total','bce','margin','seconds')}), flush=True)
            if step % config['checkpoint_every'] == 0 or step == stop:
                dev_result = evaluate(backend, dev, config['objective'], config['margin'], config['margin_weight'])
                selected_best = dev_result['loss'] < best
                if selected_best:
                    best_step = step
                best = min(best, dev_result['loss'])
                checkpoint = output/'checkpoints'/f'step-{step:06d}'
                snapshot = adapter_state(backend.model)
                metadata = {'identity': identity_hash, 'step': step, 'best_dev_loss': best, 'best_step': best_step, 'dev': dev_result, 'inputs': identity,
                            'selection': {'latest': True, 'best': selected_best}, 'config': config,
                            'group_order': sampler.state_dict()['order'], 'consumed_groups_this_segment': consumed}
                save_checkpoint(checkpoint, snapshot, optimizer, scheduler, sampler, metadata)
                # Verify adapter serialization + reload into this exact model, including eval scoring.
                restored, _, _ = load_checkpoint(checkpoint, identity_hash)
                load_adapter(backend.model, restored)
                reloaded = evaluate(backend, dev, config['objective'], config['margin'], config['margin_weight'])
                error = max(abs(a-b) for ra,rb in zip(dev_result['scores'],reloaded['scores']) for a,b in zip(ra,rb))
                if error > 1e-5:
                    raise ContractError('saved adapter scoring reload differs')
                checkpoint_checks.append({'step': step, 'reload_max_abs_error': error})
                atomic_json(output/f'selection-{step:06d}.json', {'latest': str(checkpoint.relative_to(output)), 'is_best': selected_best, 'best_step':best_step, 'dev_loss':dev_result['loss']})
    final = evaluate(backend, dev, config['objective'], config['margin'], config['margin_weight'])
    summary = {'status': 'synthetic-smoke-complete' if step == config['steps'] else 'synthetic-segment-complete',
               'objective': config['objective'], 'step': step, 'initial': initial, 'final': final,
               'reload_max_abs_error': reload_difference, 'checkpoint_checks': checkpoint_checks,
               'consumed_groups': consumed, 'first_gradient_norms': first_gradients,
               'peak_vram_bytes': torch.cuda.max_memory_allocated(), 'elapsed_seconds': time.monotonic()-start,
               'groups_per_second': consumed/(time.monotonic()-start), 'identity':identity_hash,
               'data_status':'synthetic-only / real-data-blocked',
               'updated_adapter_tensors':sum(not torch.equal(v, initial_adapter[n]) for n,v in adapter_state(backend.model).items()),
               'last_gradient_norms':grads}
    atomic_json(output/'summary.json', summary)
    backend.close()
    return summary
