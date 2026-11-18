"""Write numeric metrics from saved checkpoint JSON files as long-format CSV."""
import argparse
import csv
import json
import sys
from pathlib import Path


def numbers(obj, path=()):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from numbers(value, (*path, key))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from numbers(value, (*path, str(i)))
    elif type(obj) in (int, float):
        yield "/".join(path), obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    writer = csv.writer(sys.stdout)
    writer.writerow(("file", "metric", "value"))
    for path in parser.parse_args().files:
        for metric, value in numbers(json.loads(path.read_text())):
            writer.writerow((path, metric, value))


if __name__ == "__main__":
    main()
