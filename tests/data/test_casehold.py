import math
import pytest
import torch
from tapestry_m1.core.io import ContractError
from tapestry_m1.data.casehold import parse_row
from tapestry_m1.reranker.choice import group_ce, metrics, stop_reason, select, lr_multiplier
from tapestry_m1.workflows.casehold_pilot import Schedule, independent_final_audit

@pytest.mark.parametrize('gold',range(5))
def test_native_positions_and_text_preservation(gold):
    row=['123',' context ']+[f'holding {i}' for i in range(5)]+['unused']*5+[str(gold)]
    group=parse_row(row,'validation')
    assert group['query_text']==' context '
    assert group['gold']==gold
    assert [c['text'] for c in group['candidates']]==row[2:7]
    scores=torch.zeros(1,5,requires_grad=True)
    loss=group_ce(scores,[gold]);loss.backward()
    assert loss.item()==pytest.approx(math.log(5))
    assert scores.grad[0,gold]<0
    assert (scores.grad[0,torch.arange(5)!=gold]>0).all()

@pytest.mark.parametrize('label',['5','-1','1.0',''])
def test_bad_labels(label):
    with pytest.raises(ContractError):parse_row(['1','q']+['d']*5+['unused']*5+[label],'test')

def test_permutation_and_stable_ties():
    scores=torch.tensor([[1.,5.,3.,2.,4.]])
    perm=[4,0,3,1,2]
    assert group_ce(scores,[1]).item()==pytest.approx(group_ce(scores[:,perm],[perm.index(1)]).item())
    assert metrics([0.]*5,4)['rank']==5
    assert metrics([0.,0.,1.,0.,0.],2)['accuracy']==1
    assert metrics([5,4,3,2,1],4)['mrr']==.2

def test_schedule_resume_no_cycles():
    s=Schedule(16,42); first=s.take(list(range(16)))
    state=s.state_dict();t=Schedule(16,42);t.load_state_dict(state)
    assert s.take(list(range(16)))==t.take(list(range(16)))
    assert len(set(first+s.order[8:]))==16
    with pytest.raises(ContractError):s.take(list(range(16)))

def test_early_stops_and_tiebreaks():
    base={'accuracy':.7,'dispersion':1.}
    history=[{'step':5,'accuracy':.64,'mrr':.8,'dispersion':1.,'train_loss':1.}]
    assert stop_reason(base,history) is None
    history.append(history[0]|{'step':20,'accuracy':.65})
    assert stop_reason(base,history)=='step5_and20_accuracy_drop_at_least_5pp'
    assert select([history[0],history[0]|{'step':20}])['step']==5
    assert stop_reason(base,[history[0]|{'dispersion':.001}]).startswith('severe')
    history=[history[0]|{'accuracy':.7},history[0]|{'step':60,'accuracy':.68,'train_loss':.5}]
    assert stop_reason(base,history)=='train_loss_improves_but_dev_degrades'

def test_learning_rate_endpoints():
    assert lr_multiplier(1)==.1
    assert lr_multiplier(10)==1.
    assert lr_multiplier(11)<1.
    assert lr_multiplier(120)==0.

def test_independent_audit_and_bootstrap():
    rows=[{'query_id':str(i),'gold':i,'scores':[0.]*5,**metrics([0.]*5,i)} for i in range(5)]
    result=independent_final_audit(rows,rows,{'seed':1,'bootstrap_replicates':10000})
    assert result['independent_rank_audit']
    assert all(v['mean']==0 and v['ci95']==[0.,0.] for v in result['paired_delta'].values())
    rows[0]['accuracy']=0
    with pytest.raises(ContractError):independent_final_audit(rows,rows,{'seed':1,'bootstrap_replicates':1})

def test_pilot_smoke_real_peft_resume_on_tiny_cpu_model(tmp_path,monkeypatch):
    import types
    import peft
    from transformers import Qwen3Config,Qwen3ForCausalLM
    import tapestry_m1.workflows.casehold_pilot as pilot
    torch.manual_seed(4)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8))
    model=peft.get_peft_model(model,peft.LoraConfig(task_type='CAUSAL_LM',r=2,lora_alpha=4,
        lora_dropout=.1,target_modules=['q_proj','v_proj']))
    backend=types.SimpleNamespace(model=model)
    ids=torch.tensor([[1,2,3,4],[2,3,4,1],[3,4,1,2],[4,1,2,3],[1,4,3,2]])
    def score(b,groups):
        logits=b.model(ids).logits[:,-1,:]
        return (logits[:,2]-logits[:,3]).reshape(1,5),{}
    monkeypatch.setattr(pilot,'scores_for',score)
    config={'optimizer':{'lr':1e-5,'weight_decay':.01},'seed':42,'max_grad_norm':1.}
    pilot.smoke(backend,config,[{'gold':i%5} for i in range(24)],tmp_path,'tiny-cpu-identity')
    import json
    assert json.loads((tmp_path/'smoke.json').read_text())['exact_resume']

def test_evaluation_chunks_resume_without_rescoring(tmp_path):
    import types
    import tapestry_m1.workflows.casehold_pilot as pilot
    calls=[]
    def score(q,d,batch_size):
        calls.append(len(q));return [float(i%5) for i in range(len(q))],[{'truncated':False} for _ in q]
    backend=types.SimpleNamespace(model=types.SimpleNamespace(eval=lambda:None),score_logits=score)
    groups=[{'query_id':str(i),'query_text':'q','gold':i%5,'candidates':[{'text':'d'}]*5} for i in range(35)]
    before,_=pilot.evaluate(backend,groups,tmp_path,'id','base',8)
    after,_=pilot.evaluate(backend,groups,tmp_path,'id','base',8)
    assert calls==[160,15] and before==after
    with pytest.raises(FileExistsError):pilot.evaluate(backend,groups,tmp_path,'changed','base',8)
