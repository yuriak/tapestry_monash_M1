"""Lazy local-only Transformer loading shared by model backends."""
from tapestry_m1.core.io import ContractError


class LocalTransformer:
    def __init__(self, model_dir, config, *, causal_lm):
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False, padding_side="left")
        cls = AutoModelForCausalLM if causal_lm else AutoModel
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[config["dtype"]]
        self.model = cls.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False, torch_dtype=dtype, attn_implementation=config["attention_backend"]).eval().to("cuda")
        self.limit = config["max_tokens"]
        if self.limit < 16 or self.limit > self.model.config.max_position_embeddings:
            raise ContractError("token limit outside model context")

    def close(self):
        import gc
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()
