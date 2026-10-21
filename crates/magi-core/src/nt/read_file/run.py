"""Reads a text file (read-only), a part at a time: numbered lines from `from` on, `lines` of them.

MacroMagi native tool: reads its arguments as JSON on standard input and
prints its result as JSON on standard output.
"""
import json
import sys


def number(value, default):
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def main(args):
    import os
    path = os.path.expanduser(str(args.get("path", "")).strip())
    if not os.path.isfile(path):
        return {"error": "no such file: " + path}
    start = number(args.get("from"), 1)
    count = min(number(args.get("lines"), 200), 1000)
    with open(path, encoding="utf-8", errors="replace") as handle:
        lines = handle.read(5_000_000).splitlines()
    shown = lines[start - 1:start - 1 + count]
    text = "\n".join("%d| %s" % (start + i, line) for i, line in enumerate(shown))
    result = {"path": path, "lines": len(lines), "text": text}
    if start - 1 + count < len(lines):
        result["more"] = "the file goes on: read from line %d" % (start + count)
    return result


if __name__ == "__main__":
    try:
        args = json.loads(sys.stdin.read() or "{}")
        result = main(args if isinstance(args, dict) else {})
    except Exception as error:  # noqa: BLE001 - a tool reports, never crashes
        result = {"error": str(error)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
