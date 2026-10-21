"""Encode query or document records through a checked vector cache."""
from tapestry_m1.retrieval.vector_cache import VectorCache


def encode_cached(model,rows,text_field,is_query,root,rec,config):
    cache=VectorCache(root,rec,[r["query_id" if is_query else "doc_id"] for r in rows],config["dimension"],config["embedding_shard"])
    for start in range(0,len(rows),cache.shard_size):
        if cache.read(start) is None:
            vectors,lengths=model.encode([r[text_field] for r in rows[start:start+cache.shard_size]],is_query=is_query)
            cache.write(start,vectors,lengths)
    vectors,lengths,manifest=cache.finish()
    return vectors,lengths,manifest
