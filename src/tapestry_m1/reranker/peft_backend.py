"""Standard PEFT ownership and independent loading over pinned base assets."""
from tapestry_m1.core.io import ContractError
from tapestry_m1.reranker.qwen3 import Qwen3Reranker
from tapestry_m1.runtime.assets import verify_model


def build_backend(config, model_spec, data_root, adapter=None, trainable=True):
    from peft import LoraConfig,PeftModel,TaskType,get_peft_model
    root,verified=verify_model(model_spec,data_root)
    backend=Qwen3Reranker(root,config)
    if adapter:
        backend.model=PeftModel.from_pretrained(backend.model,str(adapter),is_trainable=trainable)
    else:
        c=config['lora']
        backend.model=get_peft_model(backend.model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=c['rank'],
            lora_alpha=c['alpha'],lora_dropout=c['dropout'],target_modules=c['targets'],bias='none',revision=model_spec['revision']))
    for c in backend.model.peft_config.values():
        c.base_model_name_or_path=model_spec['model_id'];c.revision=model_spec['revision']
    parameters={n:p for n,p in backend.model.named_parameters() if p.requires_grad}
    if trainable and (not parameters or any('.lora_A.' not in n and '.lora_B.' not in n for n in parameters)):
        raise ContractError('only intended PEFT LoRA parameters may train')
    modules=[n for n,m in backend.model.named_modules() if hasattr(m,'lora_A')]
    if trainable and {n.split('.')[-1] for n in modules}!=set(config['lora']['targets']):raise ContractError('PEFT target modules differ')
    return backend,verified,{'backend':'peft','modules':modules,'trainable_names':list(parameters),
        'trainable_parameters':sum(p.numel() for p in parameters.values()),'total_parameters':sum(p.numel() for p in backend.model.parameters())}
