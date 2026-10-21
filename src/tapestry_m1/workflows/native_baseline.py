"""Native collections only: immutable retrieval, exact-set reranking and evaluation."""
import json
from pathlib import Path
import time
from tapestry_m1.core.io import ContractError, atomic_json, dumps, fingerprint, records
from tapestry_m1.retrieval.encoding import encode_cached
from tapestry_m1.core.artifacts import identity, atomic_bytes
from tapestry_m1.retrieval.qwen3 import Qwen3Encoder
from tapestry_m1.reranker.qwen3 import Qwen3Reranker
from tapestry_m1.retrieval.search import exact_topk
from tapestry_m1.reranker.ranking import reorder, check_candidates
from tapestry_m1.runtime.provenance import code_fingerprint
from tapestry_m1.runtime.assets import verify_model
from tapestry_m1.data.artifacts import input_root
from tapestry_m1.runtime.devices import gpu_runtime
from tapestry_m1.evaluation.retrieval.evaluator import evaluate
from tapestry_m1.evaluation.statistics.bootstrap import paired_compare


def sample_inputs(config,data_root):
    docs=[];queries=[]
    for spec in config["datasets"]:
        root,manifest=input_root(data_root,spec)
        rows=list(records(root/"documents.jsonl"));qs=list(records(root/"queries.jsonl"))
        # Length-extreme samples use source text lengths only, never qrels or model results.
        ordered=sorted(rows,key=lambda r:(len(r["text"]),r["doc_id"]))
        docs.extend([ordered[0],ordered[len(ordered)//2],ordered[-1]])
        queries.extend([qs[0]]*3)
    return docs,queries


def recipe(config,hardware,models):
    return {"config":config,"hardware":{k:v for k,v in hardware.items() if k != "job_id"},"models_verified":models,"implementation":code_fingerprint(),
            "pooling":"last_nonpadding_token","normalization":"L2_float32","similarity":"normalized_dot_float32",
            "tie_break":"score_desc_doc_id_ascending","seed":None}


def preflight(config,data_root):
    import numpy as np
    import torch
    hardware=gpu_runtime();started=time.monotonic();models={};paths={}
    for role in ["encoder","reranker"]:paths[role],models[role]=verify_model(config[role],data_root)
    rec=recipe(config,hardware,models);key=identity(rec);dest=Path(data_root)/"runs"/"preflight"/key
    if (dest/"passed.json").exists():
        saved=json.loads((dest/"passed.json").read_text())
        if saved["recipe"]!=rec or not saved["passed"]:raise ContractError("preflight identity mismatch")
        print("reused verified preflight",key,flush=True);return saved
    docs,queries=sample_inputs(config,data_root);results={}
    for role in ["encoder","reranker"]:
        torch.cuda.reset_peak_memory_stats()
        model=(Qwen3Encoder if role=="encoder" else Qwen3Reranker)(paths[role],config)
        if role=="encoder":
            batch,audit=model.encode([r["text"] for r in docs]);single=np.concatenate([model.encode([r["text"]],batch_size=1)[0] for r in docs])
            qb,qa=model.encode([r["query_text"] for r in queries],is_query=True)
            qs=np.concatenate([model.encode([r["query_text"]],is_query=True,batch_size=1)[0] for r in queries])
            if not all(np.isfinite(x).all() for x in [batch,single,qb,qs]):raise ContractError("nonfinite preflight embeddings")
            distance=float(max(np.max(1-np.sum(batch*single,axis=1)),np.max(1-np.sum(qb*qs,axis=1))))
            if batch.shape[1]!=config["dimension"] or distance>config["preflight_embedding_cosine_distance"]:raise ContractError(f"embedding batch/single preflight failed: {distance}")
            repeat=model.encode([docs[0]["text"]])[0]
            if not np.allclose(single[0],repeat[0],atol=1e-6,rtol=1e-6):raise ContractError("embedding repeat stability failed")
            results[role]={"batch_single_max_cosine_distance":distance,"audit":audit,"query_audit":qa,"repeat_stable":True}
        else:
            batch,audit=model.score([r["query_text"] for r in queries],[r["text"] for r in docs]);single=[model.score([q["query_text"]],[d["text"]],batch_size=1)[0][0] for q,d in zip(queries,docs)]
            error=float(np.max(np.abs(np.asarray(batch)-single)))
            if error>config["preflight_reranker_absolute_error"]:raise ContractError(f"reranker batch/single preflight failed: {error}")
            repeat=model.score([queries[0]["query_text"]],[docs[0]["text"]])[0][0]
            if abs(single[0]-repeat)>1e-6:raise ContractError("reranker repeat stability failed")
            results[role]={"batch_single_max_probability_error":error,"audit":audit,"yes_token_id":model.yes,"no_token_id":model.no,"repeat_stable":True}
        results[role]["peak_vram_bytes"]=torch.cuda.max_memory_allocated();model.close()
    result={"recipe":rec,"runtime":hardware,"fingerprint":key,"passed":True,"results":results,"wall_seconds":time.monotonic()-started,"sample_doc_ids":[r["doc_id"] for r in docs]}
    atomic_json(dest/"passed.json",result);print("preflight passed",key,flush=True);return result


def run_native(config,data_root):
    import torch
    hardware=gpu_runtime();models={};paths={}
    for role in ["encoder","reranker"]:paths[role],models[role]=verify_model(config[role],data_root)
    rec=recipe(config,hardware,models);key=identity(rec)
    passed=Path(data_root)/"runs"/"preflight"/key/"passed.json"
    if not passed.exists() or not json.loads(passed.read_text())["passed"]:raise ContractError("this exact recipe/hardware requires a passed preflight")
    outputs=[]
    for spec in config["datasets"]:
        root,manifest=input_root(data_root,spec);docs=list(records(root/"documents.jsonl"));queries=list(records(root/"queries.jsonl"))
        if any(r["collection_id"]!=spec["collection_id"] for r in docs+queries):raise ContractError("foreign collection")
        ids=[r["doc_id"] for r in docs];query_ids=[r["query_id"] for r in queries]
        if len(ids)!=len(set(ids)) or len(query_ids)!=len(set(query_ids)):raise ContractError("duplicate source identity")
        dskey=identity({"recipe":rec,"source":manifest})
        out=Path(data_root)/"runs"/"native"/spec["dataset_id"]/dskey;out.mkdir(parents=True,exist_ok=True)
        if (out/"result.json").exists():
            completed=json.loads((out/"result.json").read_text())
            if completed["recipe"]!=rec:raise ContractError("completed recipe mismatch")
            for name,sha in completed["artifacts"].items():
                if fingerprint(out/name)!=sha:raise ContractError("completed run checksum mismatch")
            outputs.append(str(out));print("reused completed native run",spec["dataset_id"],flush=True);continue
        started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        enc=Qwen3Encoder(paths["encoder"],config)
        cache_root=Path(data_root)/"embeddings"/spec["collection_id"]
        dv,dl,dc=encode_cached(enc,docs,"text",False,cache_root,rec|{"source":manifest,"kind":"documents"},config)
        qv,ql,qc=encode_cached(enc,queries,"query_text",True,cache_root,rec|{"source":manifest,"kind":"queries"},config)
        lists=exact_topk(ids,dv,qv,config["top_k"]);ranks=[]
        for q,candidates in zip(queries,lists):
            if len(candidates)!=min(100,len(docs)):raise ContractError("incomplete Top-100")
            ranks.append({k:q[k] for k in ["dataset_id","split","query_id","collection_id"]}|{"run_id":"x1_1-"+dskey,"seed":None,
                "fingerprints":{"model":identity(config["encoder"]),"index":identity(dc),"config":key},"candidates":candidates})
        atomic_bytes(out/"x1_1.jsonl",("".join(dumps(r)+"\n" for r in ranks)).encode())
        retrieval_seconds=time.monotonic()-started;retrieval_vram=torch.cuda.max_memory_allocated();enc.close()
        atomic_json(out/"length_audit.json",{"documents":[{"doc_id":r["doc_id"]}|a for r,a in zip(docs,dl)],"queries":[{"query_id":r["query_id"]}|a for r,a in zip(queries,ql)]})
        # Load the immutable serialized retrieval artifact, rather than retrieve again.
        original=list(records(out/"x1_1.jsonl"));retrieval_hash=fingerprint(out/"x1_1.jsonl")
        started=time.monotonic();torch.cuda.reset_peak_memory_stats();reranker=Qwen3Reranker(paths["reranker"],config)
        by_id={r["doc_id"]:r for r in docs};q_by_id={r["query_id"]:r for r in queries};reranked=[];pair_lengths=[]
        for i,row in enumerate(original):
            pair_key=identity({"recipe":rec,"retrieval_sha256":retrieval_hash,"query_id":row["query_id"]})
            score_file=out/"pair_cache"/(pair_key+".json")
            candidate_ids=[r["doc_id"] for r in row["candidates"]]
            if score_file.exists():
                cached=json.loads(score_file.read_text())
                if cached["candidate_ids"]!=candidate_ids or cached["fingerprint"]!=pair_key:raise ContractError("reranker cache identity mismatch")
                scores,audit=cached["scores"],cached["lengths"]
            else:
                scores,audit=reranker.score([q_by_id[row["query_id"]]["query_text"]]*len(candidate_ids),[by_id[d]["text"] for d in candidate_ids])
                atomic_json(score_file,{"fingerprint":pair_key,"candidate_ids":candidate_ids,"scores":scores,"lengths":audit})
            pair_lengths.extend({"query_id":row["query_id"],"doc_id":d}|a for d,a in zip(candidate_ids,audit))
            reranked.append(row|{"run_id":"x1_2-"+dskey,"fingerprints":{"model":identity(config["reranker"]),"index":retrieval_hash,"config":key},"candidates":reorder(row["candidates"],scores)})
            if (i+1)%10==0:print(spec["dataset_id"],"reranked",i+1,"/",len(original),flush=True)
        invariant=check_candidates(original,reranked)
        atomic_bytes(out/"x1_2.jsonl",("".join(dumps(r)+"\n" for r in reranked)).encode())
        reranker_seconds=time.monotonic()-started;reranker_vram=torch.cuda.max_memory_allocated();reranker.close()
        atomic_json(out/"reranker_lengths.json",pair_lengths)
        a=evaluate(root,out/"x1_1.jsonl");b=evaluate(root,out/"x1_2.jsonl");ci=paired_compare(a,b,config["bootstrap_seed"],config["bootstrap_resamples"])
        atomic_json(out/"evaluation-x1_1.json",a);atomic_json(out/"evaluation-x1_2.json",b);atomic_json(out/"paired-comparison.json",ci)
        result={"recipe":rec,"runtime":hardware,"artifacts":{str(f.relative_to(out)):fingerprint(f) for f in out.rglob("*") if f.is_file()},"dataset":spec,"inputs":manifest["output_sha256"],"candidate_invariance":invariant,
            "query_count":len(queries),"document_count":len(docs),"retrieval_sha256":retrieval_hash,"reranker_sha256":fingerprint(out/"x1_2.jsonl"),
            "retrieval_wall_seconds":retrieval_seconds,"reranker_wall_seconds":reranker_seconds,"retrieval_peak_vram_bytes":retrieval_vram,"reranker_peak_vram_bytes":reranker_vram,
            "truncated_documents":sum(r["truncated"] for r in dl),"truncated_queries":sum(r["truncated"] for r in ql),"truncated_pairs":sum(r["truncated"] for r in pair_lengths),
            "encoding_vector_bytes":int(dv.nbytes+qv.nbytes),"output_bytes":sum(f.stat().st_size for f in out.rglob("*") if f.is_file()),"x1_1_macro":a["groups"],"x1_2_macro":b["groups"],"comparison":ci,"status":"benchmark-complete"}
        atomic_json(out/"result.json",result);outputs.append(str(out));print("completed",spec["dataset_id"],flush=True)
    return {"status":"benchmark-complete","outputs":outputs}
