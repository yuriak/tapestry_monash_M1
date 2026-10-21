import pytest
torch = pytest.importorskip("torch")
from tapestry_m1.core.io import ContractError
from tapestry_m1.retrieval.qwen3 import last_token_pool, embedding_query


def test_embedding_pool_and_query_format():
    hidden = torch.arange(24).reshape(2,3,4)
    right = torch.tensor([[1,1,0],[1,0,0]])
    assert torch.equal(last_token_pool(hidden, right), torch.stack([hidden[0,1],hidden[1,0]]))
    left = torch.tensor([[0,1,1],[1,1,1]])
    assert torch.equal(last_token_pool(hidden,left),hidden[:,-1])
    with pytest.raises(ContractError):
        last_token_pool(hidden,torch.zeros((2,3)))
    assert embedding_query("TASK","Q") == "Instruct: TASK\nQuery:Q"
