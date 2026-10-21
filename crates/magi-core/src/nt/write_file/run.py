"""Writes a text file – a new one, or all of an existing one. The user is asked first.

MacroMagi native tool: reads its arguments as JSON on standard input and
prints its result as JSON on standard output.
"""
import json
import sys

import os


def main(args):
    path = os.path.expanduser(str(args.get("path", "")).strip())
    if not path:
        return {"error": "say which file"}
    content = str(args.get("text", ""))
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    existed = os.path.isfile(path)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(content)
    return {"path": path, "written": len(content.encode("utf-8")), "replaced": existed,
            "lines": content.count("\n") + (0 if content.endswith("\n") or not content else 1)}


if __name__ == "__main__":
    try:
        args = json.loads(sys.stdin.read() or "{}")
        result = main(args if isinstance(args, dict) else {})
    except Exception as error:  # noqa: BLE001 - a tool reports, never crashes
        result = {"error": str(error)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
