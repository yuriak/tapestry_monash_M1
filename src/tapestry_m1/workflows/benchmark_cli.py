"""Pinned model preflight and benchmark-native zero-shot runs."""
import argparse
import json
import os
from pathlib import Path
import sys
from tapestry_m1.core.io import ContractError,dumps
from tapestry_m1.runtime.configuration import load_config


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','run-native'])
    parser.add_argument('config',type=Path)
    parser.add_argument('--data-root',type=Path,default=Path(os.environ.get('M1_DATA_ROOT','data')))
    args=parser.parse_args(argv)
    try:
        from tapestry_m1.workflows.native_baseline import preflight,run_native
        config=load_config(args.config)
        if config.get("top_k") != 100 or not config.get("datasets"):
            raise ContractError("native benchmark requires Top-100 and explicit datasets")
        result=(preflight if args.command=='preflight' else run_native)(config,args.data_root)
        print(dumps(result));return 0
    except (ContractError,OSError,ValueError,KeyError,TypeError) as exc:
        print(dumps({'status':'error','type':type(exc).__name__,'message':str(exc)}),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
