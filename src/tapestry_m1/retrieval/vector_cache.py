"""Content-checked immutable vector cache shards."""
import hashlib
import io
import json
from pathlib import Path
from tapestry_m1.core.io import ContractError, atomic_json, fingerprint
from tapestry_m1.core.artifacts import atomic_bytes, identity


class VectorCache:
    def __init__(self,root,recipe,ids,dimension,shard_size):
        if len(ids)!=len(set(ids)) or not ids:raise ContractError("cache requires nonempty unique IDs")
        if dimension<1 or shard_size<1:raise ContractError("invalid shard size/dimension")
        self.recipe=recipe;self.ids=list(ids);self.dimension=dimension;self.shard_size=shard_size
        self.key=identity({"recipe":recipe,"ids":self.ids,"dimension":dimension,"shard_size":shard_size})
        self.root=Path(root)/self.key;self.root.mkdir(parents=True,exist_ok=True)
        atomic_json(self.root/"identity.json",{"key":self.key,"recipe":recipe,"ids":self.ids,"dimension":dimension,"shard_size":shard_size})
    def expected(self,start):
        if start<0 or start>=len(self.ids) or start%self.shard_size:raise ContractError("invalid shard offset")
        return self.ids[start:start+self.shard_size]
    def read(self,start):
        import numpy as np
        ids=self.expected(start);entry=self.root/f"batch-{start:08d}.json"
        if not entry.exists():return None
        meta=json.loads(entry.read_text())
        if meta["cache_key"]!=self.key or meta["ids"]!=ids:raise ContractError("cache identity/ID collision")
        data=self.root/meta["array"]
        if data.parent!=self.root or fingerprint(data)!=meta["sha256"]:raise ContractError("cache array checksum mismatch")
        array=np.load(data,allow_pickle=False)
        if array.shape!=(len(ids),self.dimension) or not np.isfinite(array).all() or (np.linalg.norm(array,axis=1)==0).any():raise ContractError("invalid cached vectors")
        return array,meta["lengths"]
    def write(self,start,array,lengths):
        import numpy as np
        ids=self.expected(start);array=np.asarray(array,dtype=np.float32)
        if array.shape!=(len(ids),self.dimension) or not np.isfinite(array).all() or (np.linalg.norm(array,axis=1)==0).any():raise ContractError("invalid vectors for cache")
        if len(lengths)!=len(ids):raise ContractError("length audit size differs")
        buf=io.BytesIO();np.save(buf,array,allow_pickle=False);body=buf.getvalue();sha=hashlib.sha256(body).hexdigest()
        name=f"batch-{start:08d}-{sha}.npy";atomic_bytes(self.root/name,body)
        atomic_json(self.root/f"batch-{start:08d}.json",{"cache_key":self.key,"ids":ids,"array":name,"sha256":sha,"lengths":lengths})
    def finish(self):
        import numpy as np
        arrays=[];lengths=[];shards=[]
        for start in range(0,len(self.ids),self.shard_size):
            result=self.read(start)
            if result is None:raise ContractError("incomplete cache cannot be finalized")
            arrays.append(result[0]);lengths.extend(result[1]);shards.append(fingerprint(self.root/f"batch-{start:08d}.json"))
        manifest={"key":self.key,"rows":len(self.ids),"dimension":self.dimension,"shards":shards,"complete":True}
        atomic_json(self.root/"complete.json",manifest)
        return np.concatenate(arrays),lengths,manifest
