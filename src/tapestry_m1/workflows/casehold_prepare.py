"""CPU-only preparation: official data audit and answer-independent token lengths."""
import json
import argparse
from pathlib import Path
from tapestry_m1.core.io import atomic_json,fingerprint
from tapestry_m1.data.casehold import audit,load_split
from tapestry_m1.reranker.qwen3 import PREFIX,SUFFIX,reranker_text
from tapestry_m1.runtime.assets import verify_model
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "config", "data-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root, config_path = args.output, args.config
    root.mkdir(parents=True, exist_ok=True)
    config=json.loads(config_path.read_text())
    spec=json.loads((config_path.parent/config['model']).read_text())
    model_dir,verified=verify_model(spec,args.data_root)
    audit_path=root/'data-audit.json'
    if not audit_path.exists():
        print('Auditing official splits and bounded duplicate screen',flush=True)
        atomic_json(audit_path,audit(args.raw))
    print('Data audit complete; auditing token lengths without model scoring',flush=True)
    tokenizer=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=False,padding_side='left')
    overhead=len(tokenizer.encode(PREFIX,add_special_tokens=False))+len(tokenizer.encode(SUFFIX,add_special_tokens=False))
    counts={'pairs':0,'truncated':0,'max_native_tokens':0,'query_truncated_pairs':0,'document_truncated_pairs':0}
    by_split={}
    for split in ['train','validation','test']:
        groups=load_split(args.raw,split)
        # Audit all official pairs, not only the eventual training subset.
        pair_count=truncated=0
        for start in range(0,len(groups),128):
            pairs=[reranker_text(config['instruction'],g['query_text'],c['text']) for g in groups[start:start+128] for c in g['candidates']]
            tokens=tokenizer(pairs,padding=False,truncation=False,add_special_tokens=True)['input_ids']
            for pair,t in zip(pairs,tokens):
                n=len(t)+overhead; pair_count+=1; counts['pairs']+=1
                counts['max_native_tokens']=max(counts['max_native_tokens'],n)
                if n>config['max_tokens']:
                    truncated+=1;counts['truncated']+=1
                    boundary=pair.rfind('\n<Document>: ')+len('\n<Document>: ')
                    offsets=tokenizer(pair,add_special_tokens=True,return_offsets_mapping=True)['offset_mapping']
                    dropped=offsets[config['max_tokens']-overhead:]
                    counts['query_truncated_pairs']+=int(any(a<boundary-len('\n<Document>: ') and b>a for a,b in dropped))
                    counts['document_truncated_pairs']+=int(any(b>boundary for a,b in dropped))
            if start%4096==0:print(f'token audit {split}: {start}/{len(groups)} queries',flush=True)
        by_split[split]={'queries':len(groups),'pairs':pair_count,'truncated_pairs':truncated}

    counts.update({'truncated_fraction':counts['truncated']/counts['pairs'],'by_split':by_split,
                   'token_ids':{k:tokenizer.convert_tokens_to_ids(k) for k in ['yes','no']},
                   'policy':'right truncate body to max_tokens-prefix-suffix; suffix always retained',
                   'max_tokens':config['max_tokens'],'max_length_frozen_before_baseline':True})
    atomic_json(root/'tokenization-audit.json',counts)
    if counts['truncated_fraction']>config['truncation_fraction_limit']:
        raise RuntimeError('Significant truncation exceeds frozen 1% gate; revise common policy before baseline')
    atomic_json(root/'preparation.json',{'config_sha256':fingerprint(config_path),
        'data_audit':json.loads(audit_path.read_text()),'tokenization':counts,'model_revision':verified['revision']})
    print(json.dumps(counts,indent=2),flush=True)


if __name__ == "__main__":
    main()
