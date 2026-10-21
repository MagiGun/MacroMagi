"""Finds text in the files of a folder (read-only): every line that has it, any case, with its file and line.

MacroMagi native tool: reads its arguments as JSON on standard input and
prints its result as JSON on standard output.
"""
import json
import sys

import os

SKIP = {".git", "node_modules", "target", "__pycache__", ".venv", "venv", "dist", "build", ".idea", ".vs"}


def main(args):
    needle = str(args.get("text", ""))
    folder = os.path.expanduser(str(args.get("folder", "")).strip() or ".")
    if not needle:
        return {"error": "say what to find"}
    if not os.path.isdir(folder):
        return {"error": "no such folder: " + folder}
    wanted = needle.lower()
    found, files = [], 0
    for root, dirs, names in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith("."))
        for name in sorted(names):
            path = os.path.join(root, name)
            try:
                if os.path.getsize(path) > 2_000_000:
                    continue
                with open(path, encoding="utf-8") as handle:
                    lines = handle.read().splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            files += 1
            for number, line in enumerate(lines, 1):
                if wanted in line.lower():
                    found.append("%s:%d: %s" % (os.path.relpath(path, folder).replace(os.sep, "/"), number, line.strip()[:200]))
                    if len(found) >= 60:
                        return {"folder": folder, "matches": found, "more": "there are more: search for something narrower"}
    if not found:
        return {"folder": folder, "matches": [], "note": "not found in %d files" % files}
    return {"folder": folder, "matches": found}


if __name__ == "__main__":
    try:
        args = json.loads(sys.stdin.read() or "{}")
        result = main(args if isinstance(args, dict) else {})
    except Exception as error:  # noqa: BLE001 - a tool reports, never crashes
        result = {"error": str(error)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
