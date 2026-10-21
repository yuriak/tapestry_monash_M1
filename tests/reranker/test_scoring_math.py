import math
import pytest
torch = pytest.importorskip("torch")
from tapestry_m1.core.io import ContractError
from tapestry_m1.reranker.qwen3 import yes_probability, reranker_text


def test_reranker_logits_and_prompt_format():
    logits = torch.tensor([[2.,4.,100.]])
    assert yes_probability(logits,0,1).item() == pytest.approx(1/(1+math.exp(-2)),abs=1e-6)
    with pytest.raises(ContractError):
        yes_probability(logits,1,1)
    with pytest.raises(ContractError):
        yes_probability(torch.tensor([[float("nan"),0.]]),0,1)
    assert reranker_text("TASK","Q","D") == "<Instruct>: TASK\n<Query>: Q\n<Document>: D"
