"""Deterministic character-span lineage and atomic source-offset checkpoints."""
import json
from pathlib import Path
from tapestry_m1.core.io import ContractError,atomic_json,dumps,fingerprint
from tapestry_m1.contracts.records import identifier
from tapestry_m1.core.artifacts import identity,atomic_bytes


def chunk_document(doc,tokenizer,policy):
    window,overlap=policy["window"],policy["overlap"]
    if type(window) is not int or type(overlap) is not int or window<=overlap or overlap<0:raise ContractError("invalid chunk window/overlap")
    encoded=tokenizer(doc["text"],add_special_tokens=False,return_offsets_mapping=True,truncation=False)
    offsets=encoded["offset_mapping"];n=len(offsets)
    if n==0:raise ContractError("source document has no tokens")
    collection=doc["collection_id"]+"-chunks-"+identity(policy)[:12]
    for start in range(0,n,window-overlap):
        stop=min(start+window,n);a,b=offsets[start][0],offsets[stop-1][1]
        if b<=a:raise ContractError("token offsets do not resolve a nonempty source span")
        raw=doc["doc_id"]+f"@{start}:{stop}"
        yield {"collection_id":collection,"doc_id":identifier(collection,raw),"text":doc["text"][a:b],"source_doc_id":doc["doc_id"],
            "source_span":{"character_start":a,"character_end":b,"token_start":start,"token_end":stop},"source_provenance":doc.get("provenance"),"policy":policy}
        if stop==n:break


def stage_chunks(corpus_root,output_root,tokenizer,policy,max_documents=None,documents_per_shard=32):
    if max_documents is not None and (type(max_documents) is not int or max_documents<1):raise ContractError("document budget must be positive")
    if documents_per_shard<1:raise ContractError("invalid source shard size")
    root=Path(corpus_root);source=json.loads((root/"manifest.json").read_text())
    if source["status"]!="validated" or source["truncated"]:raise ContractError("chunk staging requires verified complete source")
    source_path=root/"documents.jsonl";source_sha=fingerprint(source_path)
    if source_sha!=source["output_sha256"]["documents.jsonl"]:raise ContractError("corpus source checksum mismatch")
    recipe={"source_sha256":source_sha,"policy":policy,"documents_per_shard":documents_per_shard,"implementation_sha256":fingerprint(__file__),"format_version":1}
    key=identity(recipe);dest=Path(output_root)/key;dest.mkdir(parents=True,exist_ok=True);atomic_json(dest/"identity.json",recipe)
    metas=[];processed=0;offset=0;chunks=0
    for file in sorted(dest.glob("shard-*.json")):
        m=json.loads(file.read_text())
        if m["source_start"]!=processed or m["byte_start"]!=offset or fingerprint(dest/m["data_file"])!=m["data_sha256"]:raise ContractError("chunk shard coverage/checksum mismatch")
        processed=m["source_end"];offset=m["byte_end"];chunks+=m["chunks"];metas.append(m)
    if max_documents is not None and processed>max_documents:raise ContractError("existing staged coverage exceeds requested budget")
    with source_path.open("rb") as handle:
        handle.seek(offset)
        while max_documents is None or processed<max_documents:
            body=[];doc_counts={};begin=processed;byte_start=handle.tell()
            limit=documents_per_shard if max_documents is None else min(documents_per_shard,max_documents-processed)
            for _ in range(limit):
                line=handle.readline()
                if not line:break
                if not line.strip():raise ContractError("blank source rows not supported in source-offset staging")
                doc=json.loads(line);pieces=list(chunk_document(doc,tokenizer,policy))
                body.extend(dumps(r)+"\n" for r in pieces);doc_counts[doc["doc_id"]]=len(pieces);processed+=1
            if processed==begin:break
            name=f"chunks-{begin:08d}-{processed:08d}.jsonl";atomic_bytes(dest/name,"".join(body).encode())
            m={"source_start":begin,"source_end":processed,"byte_start":byte_start,"byte_end":handle.tell(),"data_file":name,"data_sha256":fingerprint(dest/name),"chunks":sum(doc_counts.values()),"source_document_chunks":doc_counts}
            atomic_json(dest/f"shard-{begin:08d}.json",m);metas.append(m);chunks+=m["chunks"]
        eof=handle.tell()==source_path.stat().st_size
    coverage={"recipe":recipe,"fingerprint":key,"source_documents_total":source["observed_counts"]["documents"],"source_documents_staged":processed,"chunks_staged":chunks,"status":"complete" if eof else "partial","shards":metas}
    atomic_json(dest/f"coverage-{processed:08d}.json",coverage)
    return dest,coverage
