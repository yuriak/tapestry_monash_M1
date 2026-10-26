"""Small native PyTorch LoRA backend; no dependency or base-checkpoint upgrades."""
import math
import torch
from tapestry_m1.core.io import ContractError


class LoRALinear(torch.nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.dropout = torch.nn.Dropout(dropout)
        # Float32 adapters/optimizer states; base remains in its selected runtime dtype.
        self.lora_A = torch.nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, value):
        update = torch.nn.functional.linear(torch.nn.functional.linear(self.dropout(value.float()), self.lora_A), self.lora_B)
        return self.base(value) + (update * self.scale).to(value.dtype)


def inject_lora(model, *, rank=16, alpha=32, dropout=.05, targets=('q_proj', 'v_proj')):
    if type(rank) is not int or rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
        raise ContractError('invalid LoRA configuration')
    model.requires_grad_(False)
    modules = [(name, module) for name, module in model.named_modules()
               if name.split('.')[-1] in targets and isinstance(module, torch.nn.Linear)]
    if not modules or not set(targets) <= {n.split('.')[-1] for n, _ in modules}:
        raise ContractError('requested LoRA targets not found')
    for name, module in modules:
        parent, _, leaf = name.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model, leaf, LoRALinear(module, rank, alpha, dropout))
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not trainable or any(not n.endswith(('.lora_A', '.lora_B')) for n in trainable):
        raise ContractError('unexpected trainable parameter ownership')
    return {'modules': [n for n, _ in modules], 'trainable_names': list(trainable),
            'trainable_parameters': sum(p.numel() for p in trainable.values()),
            'total_parameters': sum(p.numel() for p in model.parameters()), 'backend': 'native-torch-lora-v1'}


def adapter_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_adapter(model, state):
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(parameters) != set(state) or any(parameters[n].shape != state[n].shape for n in parameters):
        raise ContractError('adapter parameter identity/shape mismatch')
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name].to(parameter.device, parameter.dtype))
