import copy
import json
from pathlib import Path
import random
import pytest
torch=pytest.importorskip('torch')
from tapestry_m1.core.io import ContractError
from tapestry_m1.reranker.losses import relevance_logits,group_loss
from tapestry_m1.reranker.lora import inject_lora,adapter_state,load_adapter
from tapestry_m1.reranker.batches import GroupSampler
from tapestry_m1.runtime.training_checkpoint import save_checkpoint,load_checkpoint,restore_rng


def test_autograd_matches_independent_scalar_and_group_accumulation():
    expected=json.loads((Path(__file__).parents[1]/'fixtures/training/expected.json').read_text())
    for objective in ['L0','L1']:
        scores=torch.tensor(expected['score_groups'],dtype=torch.float64,requires_grad=True)
        loss,_=group_loss(scores,objective);loss.backward()
        assert float(loss)==pytest.approx(expected[objective]['total'],abs=1e-12)
        torch.testing.assert_close(scores.grad,torch.tensor(expected[objective]['gradients_wrt_score'],dtype=torch.float64),atol=1e-12,rtol=0)
        micros=torch.tensor(expected['score_groups'],dtype=torch.float64,requires_grad=True)
        for i in range(2):(group_loss(micros[i:i+1],objective)[0]/2).backward()
        torch.testing.assert_close(scores.grad,micros.grad)
    logits=torch.randn(2,5,requires_grad=True);relevance_logits(logits,1,3).sum().backward()
    assert torch.equal(logits.grad[:,1],torch.tensor([-1.,-1.])) and torch.equal(logits.grad[:,3],torch.tensor([1.,1.]))


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.q_proj=torch.nn.Linear(3,3);self.v_proj=torch.nn.Linear(3,1)
    def forward(self,x):return self.v_proj(torch.tanh(self.q_proj(x))).reshape(2,4)


def test_full_checkpoint_resume_matches_uninterrupted_with_dropout(tmp_path):
    def build():
        random.seed(9);torch.manual_seed(9)
        model=Tiny();base=copy.deepcopy(model.state_dict());inventory=inject_lora(model,rank=2,alpha=4,dropout=.2)
        assert inventory['modules']==['q_proj','v_proj']
        optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        sched=torch.optim.lr_scheduler.LambdaLR(optim,lambda _:1.)
        sampler=GroupSampler(5,8)
        return model,optim,sched,sampler,base
    x=torch.arange(24,dtype=torch.float32).reshape(8,3)/24
    def step(model,optim,sched,sampler):
        ids=sampler.take(8);optim.zero_grad();loss,_=group_loss(model(x),'L1');loss.backward();optim.step();sched.step();return ids
    a,opt,sch,sampler,base=build()
    for _ in range(3):step(a,opt,sch,sampler)
    save_checkpoint(tmp_path/'saved',adapter_state(a),opt,sch,sampler,{'identity':'fixture','step':3})
    continuation=[step(a,opt,sch,sampler) for _ in range(3)]
    b,opt2,sch2,sampler2,_=build()
    adapter,state,meta=load_checkpoint(tmp_path/'saved','fixture');load_adapter(b,adapter)
    opt2.load_state_dict(state['optimizer']);sch2.load_state_dict(state['scheduler']);sampler2.load_state_dict(state['sampler']);restore_rng(state['rng'])
    assert [step(b,opt2,sch2,sampler2) for _ in range(3)]==continuation
    for name,value in adapter_state(a).items():torch.testing.assert_close(value,adapter_state(b)[name],atol=0,rtol=0)
    assert sch.state_dict()==sch2.state_dict()
    for name,p in a.named_parameters():
        if not p.requires_grad:torch.testing.assert_close(p,base[name.replace('.base.','.')],atol=0,rtol=0)
    with pytest.raises(ContractError,match='identity'):load_checkpoint(tmp_path/'saved','wrong')
    (tmp_path/'saved/adapter.pt').write_bytes(b'corrupt')
    with pytest.raises(ContractError,match='checksum'):load_checkpoint(tmp_path/'saved','fixture')
