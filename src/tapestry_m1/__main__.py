"""Source-tree command router for the Sovereign Evidence Assistant."""
import importlib
import sys

COMMANDS = {
    "data": "tapestry_m1.data.cli",
    "evaluation": "tapestry_m1.evaluation.cli",
    "models": "tapestry_m1.runtime.assets",
    "benchmark": "tapestry_m1.workflows.benchmark_cli",
    "corpus": "tapestry_m1.workflows.corpus",
    "gateway": "tapestry_m1.gateway.cli",
}


def main():
    if len(sys.argv) == 1 or sys.argv[1] in ("--help", "-h"):
        print("Sovereign Evidence Assistant — source commands\n\n" + "\n".join("  " + command for command in COMMANDS) + "\n\nUse <command> --help for operations. Future components are documented in docs/components/.")
        return 0
    command = sys.argv.pop(1)
    if command not in COMMANDS:
        print("Unknown component command: " + command, file=sys.stderr)
        return 2
    return importlib.import_module(COMMANDS[command]).main()


if __name__ == "__main__":
    raise SystemExit(main())
