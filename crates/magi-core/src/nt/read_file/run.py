"""Reads a text file (read-only).

MacroMagi native tool: reads its arguments as JSON on standard input and
prints its result as JSON on standard output.
"""
import json
import sys


def main(args):
    import os
    path = os.path.expanduser(str(args.get("path", "")).strip())
    if not os.path.isfile(path):
        return {"error": "no such file: " + path}
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read(200_000)
    return {"path": path, "text": text}


if __name__ == "__main__":
    try:
        args = json.loads(sys.stdin.read() or "{}")
        result = main(args if isinstance(args, dict) else {})
    except Exception as error:  # noqa: BLE001 - a tool reports, never crashes
        result = {"error": str(error)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
