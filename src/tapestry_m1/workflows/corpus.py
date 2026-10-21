"""Stage deterministic corpus chunks or encode a budgeted exact-search shard index."""
import argparse
import json
import os
from pathlib import Path
from tapestry_m1.core.io import dumps
from tapestry_m1.runtime.configuration import load_config


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=["stage","encode"])
    p.add_argument("config",type=Path);p.add_argument("corpus_root",type=Path);p.add_argument("--data-root",type=Path,default=Path(os.environ.get("M1_DATA_ROOT","data")))
    p.add_argument("--max-documents",type=int,default=1024);p.add_argument("--max-chunks",type=int,default=1024);p.add_argument("--max-seconds",type=int,default=600)
    p.add_argument("--window",type=int,default=2048);p.add_argument("--overlap",type=int,default=256)
    a=p.parse_args();config=load_config(a.config);spec=config["encoder"]
    from transformers import AutoTokenizer
    from tapestry_m1.data.chunking import stage_chunks
    model=a.data_root/"models"/spec["model_id"].split("/")[-1]/spec["revision"]
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    policy={"window":a.window,"overlap":a.overlap,"tokenizer_revision":spec["tokenizer_revision"],"offset_unit":"python_unicode_character","native_source_text_unchanged":True}
    root,coverage=stage_chunks(a.corpus_root,a.data_root/"chunks",tokenizer,policy,a.max_documents)
    if a.command=="encode":
        from tapestry_m1.retrieval.index import build_index
        root,coverage=build_index(root,coverage,config,a.data_root,a.max_chunks,a.max_seconds)
    print(dumps({"root":str(root),"status":coverage["status"],"coverage_file_rows":coverage.get("chunks_encoded",coverage["chunks_staged"])}))


if __name__=="__main__":main()
