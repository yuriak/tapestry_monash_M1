import hashlib
import io
import json
from pathlib import Path
import pytest
from conftest import write_rows
from tapestry_m1.data.acquisition import download
from tapestry_m1.data.adapters import normalize
from tapestry_m1.data.audit import audit_overlap
from tapestry_m1.io import ContractError, atomic_json, records


class Response(io.BytesIO):
    def __init__(self,data,status=200,headers=None):
        super().__init__(data);self.status=status;self.headers=headers or {"Content-Length":str(len(data))}


def test_download_cache_budget_and_resume(tmp_path,monkeypatch):
    calls=[];body=b"synthetic-payload";path=tmp_path/"source.jsonl"
    def fetch(req,timeout):
        calls.append(req)
        if req.has_header("Range"):
            offset=int(req.get_header("Range").split("=")[1].split("-")[0])
            return Response(body[offset:],206,{"Content-Length":str(len(body)-offset),"Content-Range":f"bytes {offset}-{len(body)-1}/{len(body)}"})
        return Response(body)
    monkeypatch.setattr("urllib.request.urlopen",fetch)
    with pytest.raises(ContractError,match="budget"):download("https://test/source",path,3)
    partial=path.with_name(path.name+".part");partial.write_bytes(body[:4])
    result=download("https://test/source",path,100,hashlib.sha256(body).hexdigest())
    assert result["resumed_bytes"]==4 and path.read_bytes()==body
    count=len(calls)
    assert download("https://test/source",path,100)["cache_reused"]
    assert len(calls)==count
    path.write_bytes(b"tampered")
    with pytest.raises(ContractError,match="checksum"):download("https://test/source",path,100)


@pytest.mark.parametrize("kind",["ignored_range","bad_total","hash","unexpected_partial"])
def test_download_failure_keeps_unpublished(tmp_path,monkeypatch,kind):
    path=tmp_path/"src"
    if kind in ["ignored_range","bad_total"]:
        path.with_name("src.part").write_bytes(b"a");atomic_json(path.with_name("src.part.json"),{"url":"https://test/x"})
    if kind=="bad_total":response=Response(b"b",206,{"Content-Length":"1","Content-Range":"bytes 1-1/3"})
    elif kind=="unexpected_partial":response=Response(b"b",206)
    else:response=Response(b"b")
    monkeypatch.setattr("urllib.request.urlopen",lambda *a,**k:response)
    with pytest.raises(ContractError):download("https://test/x",path,100,"0"*64 if kind=="hash" else None)
    assert not path.exists()


def spec(adapter):
    return {"dataset_id":"au-test","collection_id":"au-test-native","adapter":adapter,"files":["corpus.jsonl"],"repo_id":"synthetic/test","revision":"synthetic-v1","acquisition_allowed":True,"expected_counts":{"documents":2},"unresolved_flags":[]}


def test_stream_normalization_reuse_and_budget(tmp_path):
    raw=tmp_path/"raw";raw.mkdir();out=tmp_path/"norm"
    write_rows(raw/"corpus.jsonl",[{"version_id":str(i),"text":f"Text {i}","jurisdiction":"test","source":"synthetic"} for i in range(3)])
    s=spec("open_au_corpus");result=normalize(s,raw,out,max_rows=2)
    assert result["status"]=="partial" and result["observed_counts"]["documents"]==2
    assert normalize(s,raw,out,max_rows=2)["cache_reused"]
    with pytest.raises(ContractError):normalize(s,raw,out)
    (out/"documents.jsonl").write_text("changed")
    with pytest.raises(ContractError,match="checksum"):normalize(s,raw,out,max_rows=2)


def test_qa_not_evaluation_qrels_and_license_block(tmp_path):
    raw=tmp_path/"raw";raw.mkdir();s=spec("open_au_qa");s["files"]=["qa.jsonl"]
    write_rows(raw/"qa.jsonl",[{"question":"Question?","answer":"Answer.","text":"Positive.","source":{"version_id":"raw:1","text":"Original."}}])
    out=tmp_path/"norm";r=normalize(s,raw,out)
    assert r["observed_counts"]["training_pairs"]==1 and r["observed_counts"]["qrels"]==0
    pair=next(records(out/"training_pairs.jsonl"));assert pair["split"]=="train" and not pair["corpus_identity_resolved"]
    assert pair["positive_text"]=="Original." and pair["formatted_text"]=="Positive."
    s["acquisition_allowed"]=False
    with pytest.raises(ContractError,match="license"):normalize(s,raw,tmp_path/"blocked")


def test_overlap_detects_and_limits_partial(tmp_path):
    bench=tmp_path/"bench";corp=tmp_path/"corp";bench.mkdir();corp.mkdir()
    text="Synthetic guidance text "*8
    write_rows(bench/"documents.jsonl",[{"doc_id":"b::1","text":text}])
    write_rows(corp/"documents.jsonl",[{"doc_id":"c::1","text":"prefix "+text+" suffix","metadata":{"source":"synthetic"}},{"doc_id":"c::2","text":text}])
    atomic_json(corp/"manifest.json",{"status":"partial","truncated":True})
    result=audit_overlap(bench,corp)
    assert len(result["matches"])==2 and not result["full_source_corpus_coverage"]
    limited=audit_overlap(bench,corp,max_documents=1)
    assert not limited["complete_normalized_corpus_scanned"] and limited["corpus_documents_scanned"]==1


def test_atomic_output_refuses_overwrite(tmp_path):
    path=tmp_path/"out.json";atomic_json(path,{"v":1});atomic_json(path,{"v":1})
    with pytest.raises(FileExistsError):atomic_json(path,{"v":2})
