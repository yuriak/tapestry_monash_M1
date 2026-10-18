"""Separate linear-gain/doc-ID-descending tie view, never a published-score claim."""
import argparse
import json
from pathlib import Path
from tapestry_m1.io import records,dumps,atomic_json,fingerprint
from tapestry_m1.inference.artifacts import atomic_bytes
from tapestry_m1.data.schema import ranking
from tapestry_m1.evaluation.evaluator import evaluate


def compatibility_view(root,rankings_path,output):
    output=Path(output);rows=[]
    for row in records(rankings_path):
        ranking(row)
        candidates=sorted(row['candidates'],key=lambda c:c['doc_id'],reverse=True)
        candidates=sorted(candidates,key=lambda c:c['score'],reverse=True)
        rows.append(row|{'run_id':row['run_id']+'-linear-docid-desc','candidates':[dict(c,rank=i) for i,c in enumerate(candidates,1)]})
    atomic_bytes(output/'rankings.jsonl',(''.join(dumps(r)+'\n' for r in rows)).encode())
    result=evaluate(root,output/'rankings.jsonl',gain='linear',mrr_depth=100)
    atomic_json(output/'evaluation.json',result)
    atomic_json(output/'protocol.json',{'source_rankings_sha256':fingerprint(rankings_path),'gain':'linear','tie_break':'score_desc_doc_id_desc','mrr_depth':100,'interpretation':'A separately labelled metric/tie view. Not a full official task runner or verified Appendix B reproduction. The immutable headline rankings are retained.'})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('rankings',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();print(dumps(compatibility_view(a.root,a.rankings,a.output)))


if __name__=='__main__':main()
