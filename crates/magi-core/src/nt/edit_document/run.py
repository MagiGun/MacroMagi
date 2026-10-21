"""Changes parts of a document, by the numbers open_document gave them, and saves it in its own format with its styles (a copy of the file as it was is kept beside it). The user is asked first.

MacroMagi native tool: reads its arguments as JSON on standard input and
prints its result as JSON on standard output.
"""
import json
import sys

import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import documents  # noqa: E402 - beside this file


def main(args):
    path = os.path.expanduser(str(args.get("path", "")).strip())
    if not os.path.isfile(path):
        return {"error": "no such file: " + path}
    edits = args.get("edits")
    if isinstance(edits, str):
        edits = json.loads(edits)
    if isinstance(edits, dict):
        edits = [edits]
    if not edits:
        return {"error": "say what to change: edits, a list of {\"at\": part, \"text\": new text}"}
    changed = documents.edit(path, edits)
    return {"path": path, "changed": changed, "backup": path + ".magi-backup"}


if __name__ == "__main__":
    try:
        args = json.loads(sys.stdin.read() or "{}")
        result = main(args if isinstance(args, dict) else {})
    except Exception as error:  # noqa: BLE001 - a tool reports, never crashes
        result = {"error": str(error)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
