import copy
from pathlib import Path
import pytest
from tapestry_m1.core.io import records,ContractError
from tapestry_m1.contracts.reranker_training_v2 import validate_groups
from tapestry_m1.contracts.reranker_training import validate_groups as strict
from tapestry_m1.reranker.batches import EpochSampler


def real_groups():
    groups=list(records(Path(__file__).parents[1]/'fixtures/training/groups.jsonl'))
    for g in groups:
        g.update(schema_version='m1.reranker.training.v2',synthetic=False,query_source_doc_ids=[g['candidates'][0]['source_doc_id']])
        for i,c in enumerate(g['candidates']):c.update(split=g['split'],source_doc_ids=[c['source_doc_id']],ambiguous=i>0,supervision='assumed' if i else 'source_positive')
    return groups


def test_assumed_negatives_require_v2_and_retain_alias_guards():
    g=real_groups();validate_groups(g)
    with pytest.raises(ContractError):strict(g)
    bad=copy.deepcopy(g);bad[0]['candidates'][1]['source_doc_ids'].append(bad[0]['query_source_doc_ids'][0])
    with pytest.raises(ContractError,match='same-source'):validate_groups(bad)
    bad=copy.deepcopy(g);bad[0]['candidates'][1]['split']='dev'
    with pytest.raises(ContractError,match='another split'):validate_groups(bad)
    bad=copy.deepcopy(g);bad[0]['candidates'][1]['ambiguous']=False
    with pytest.raises(ContractError):validate_groups(bad)


def test_epoch_tail_and_restart_never_oversample():
    a=EpochSampler(11,3);first=a.take_batch(8);tail=a.take_batch(8)
    assert len(tail)==3 and sorted(first+tail)==list(range(11))
    state=a.state_dict();next_epoch=a.take_batch(8)+a.take_batch(8)
    b=EpochSampler(11,3);b.load_state_dict(state)
    assert b.take_batch(8)+b.take_batch(8)==next_epoch and sorted(next_epoch)==list(range(11))


def test_standard_peft_full_state_roundtrip_and_continuation(tmp_path):
    torch=pytest.importorskip('torch');peft=pytest.importorskip('peft')
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from tapestry_m1.runtime.training_checkpoint import save_peft_checkpoint,load_peft_state,restore_rng
    def build():
        torch.manual_seed(4)
        base=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8))
        return base
    model=peft.get_peft_model(build(),peft.LoraConfig(task_type='CAUSAL_LM',r=2,lora_alpha=4,lora_dropout=.1,target_modules=['q_proj','v_proj']))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.);sampler=EpochSampler(3,4)
    ids=torch.tensor([[1,2,3,4],[4,3,2,1]])
    def step(m,o,s):
        m.train();o.zero_grad();logits=m(ids).logits;loss=logits[:,-1,2].sum();loss.backward();o.step();s.step()
    step(model,optimizer,scheduler);sampler.take_batch(2)
    model.eval();before=model(ids).logits.detach()
    save_peft_checkpoint(tmp_path/'cp',model,optimizer,scheduler,sampler,{'identity':'test','step':1})
    loaded=peft.PeftModel.from_pretrained(build(),tmp_path/'cp',is_trainable=True);loaded.eval()
    torch.testing.assert_close(before,loaded(ids).logits,rtol=0,atol=0)
    state,meta=load_peft_state(tmp_path/'cp','test')
    restore_rng(state['rng']);step(model,optimizer,scheduler);expected={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    o=torch.optim.AdamW([p for p in loaded.parameters() if p.requires_grad],lr=.01);o.load_state_dict(state['optimizer']);s=torch.optim.lr_scheduler.LambdaLR(o,lambda _:1.);s.load_state_dict(state['scheduler']);restore_rng(state['rng']);step(loaded,o,s)
    for n,p in loaded.named_parameters():
        if p.requires_grad:torch.testing.assert_close(p,expected[n],rtol=0,atol=0)
    assert (tmp_path/'cp/adapter_config.json').exists() and (tmp_path/'cp/adapter_model.safetensors').exists()
