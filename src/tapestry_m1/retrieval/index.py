"""Bounded shard encoding and memory-bounded exact vector search, no gold labels."""
import json
from pathlib import Path
import time
from tapestry_m1.core.io import ContractError,atomic_json,fingerprint,records
from tapestry_m1.retrieval.vector_cache import VectorCache
from tapestry_m1.core.artifacts import identity
from tapestry_m1.runtime.devices import gpu_runtime
from tapestry_m1.runtime.assets import verify_model
from tapestry_m1.runtime.provenance import code_fingerprint
from tapestry_m1.retrieval.qwen3 import Qwen3Encoder
from tapestry_m1.retrieval.search import exact_topk


def build_index(stage_root,coverage,config,data_root,max_chunks=1024,max_seconds=600):
    if max_chunks<1 or max_seconds<=0:raise ContractError("positive chunk/time budgets required")
    import torch
    runtime=gpu_runtime();model_path,verified=verify_model(config["encoder"],data_root)
    recipe={"policy":coverage["recipe"],"config":config,"hardware":{k:v for k,v in runtime.items() if k!="job_id"},"models":verified,"implementation":code_fingerprint(),"index_implementation":fingerprint(__file__),"index_type":"sharded_normalized_vectors_exact_search"}
    key=identity(recipe);dest=Path(data_root)/"indices"/"au-corpus"/key;dest.mkdir(parents=True,exist_ok=True)
    model=Qwen3Encoder(model_path,config);started=time.monotonic();shards=[];encoded=0;source_counts={};truncated=0
    torch.cuda.reset_peak_memory_stats()
    try:
        stop=False
        for meta in coverage["shards"]:
            file=Path(stage_root)/meta["data_file"]
            if fingerprint(file)!=meta["data_sha256"]:raise ContractError("staged text checksum mismatch")
            rows=list(records(file))
            # Each vector shard is bounded; a full index never concatenates all vectors in RAM.
            for start in range(0,len(rows),config["embedding_shard"]):
                batch=rows[start:start+config["embedding_shard"]]
                if encoded+len(batch)>max_chunks or time.monotonic()-started>=max_seconds:stop=True;break
                ids=[r["doc_id"] for r in batch];cache=VectorCache(dest/"vectors",recipe|{"chunk_file_sha256":meta["data_sha256"],"chunk_start":start},ids,config["dimension"],len(ids))
                cached=cache.read(0)
                if cached is None:
                    vectors,lengths=model.encode([r["text"] for r in batch]);cache.write(0,vectors,lengths)
                _,lengths,final=cache.finish();truncated+=sum(x["truncated"] for x in lengths)
                shards.append({"cache_key":cache.key,"cache_relative_path":str(cache.root.relative_to(dest)),"cache_manifest_sha256":fingerprint(cache.root/"complete.json"),"rows":len(batch),"chunk_ids":ids})
                for row in batch:source_counts[row["source_doc_id"]]=source_counts.get(row["source_doc_id"],0)+1
                encoded+=len(batch)
            if stop:break
        full_doc_counts={d:n for m in coverage["shards"] for d,n in m["source_document_chunks"].items()}
        full=sum(n==full_doc_counts[d] for d,n in source_counts.items())
        complete=coverage["status"]=="complete" and encoded==coverage["chunks_staged"]
        result={"recipe":recipe,"fingerprint":key,"runtime":runtime,"source_documents_total":coverage["source_documents_total"],"source_documents_staged":coverage["source_documents_staged"],"chunks_staged":coverage["chunks_staged"],"chunks_encoded":encoded,"source_documents_with_any_chunk":len(source_counts),"source_documents_fully_indexed":full,"status":"corpus-index-complete" if complete else "partial","shards":shards,"truncated_chunks":truncated,"wall_seconds":time.monotonic()-started,"peak_torch_vram_bytes":torch.cuda.max_memory_allocated(),"vector_bytes":encoded*config["dimension"]*4,"note":"Coverage/resource artifact only; no validated global AU-corpus qrels or quality score."}
        # New coverage snapshots are immutable; resumed builds can extend budget in the same index.
        target=dest/f"coverage-{encoded:09d}.json"
        if not target.exists():atomic_json(target,result)
        elif json.loads(target.read_text())["shards"]!=shards:raise ContractError("index coverage collision")
        return dest,result
    finally:model.close()


def search_index(index_root,manifest,query_vectors,k=100):
    import numpy as np
    root=Path(index_root);by_query=[[] for _ in query_vectors]
    for shard in manifest["shards"]:
        cache_root=root/shard["cache_relative_path"]
        if fingerprint(cache_root/"complete.json")!=shard["cache_manifest_sha256"]:raise ContractError("index vector manifest checksum mismatch")
        meta=json.loads((cache_root/"batch-00000000.json").read_text());array=cache_root/meta["array"]
        if fingerprint(array)!=meta["sha256"] or meta["ids"]!=shard["chunk_ids"]:raise ContractError("index vector checksum/IDs mismatch")
        part=exact_topk(meta["ids"],np.load(array,allow_pickle=False),query_vectors,k)
        for i,rows in enumerate(part):
            candidates=by_query[i]+rows;unique={r["doc_id"] for r in candidates}
            if len(unique)!=len(candidates):raise ContractError("duplicate chunks across index shards")
            by_query[i]=sorted(candidates,key=lambda r:(-r["score"],r["doc_id"]))[:k]
    return [[dict(r,rank=i) for i,r in enumerate(rows,1)] for rows in by_query]
