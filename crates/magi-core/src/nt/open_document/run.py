"""Opens a document as numbered parts of plain text (read-only): .docx, .xlsx, .pptx, .pdf, .csv or text.

MacroMagi native tool: reads its arguments as JSON on standard input and
prints its result as JSON on standard output.
"""
import json
import sys

import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import documents  # noqa: E402 - beside this file

LIMIT = 12000


def main(args):
    path = os.path.expanduser(str(args.get("path", "")).strip())
    if not os.path.isfile(path):
        return {"error": "no such file: " + path}
    found = documents.parts(path)
    start = 0
    begin = str(args.get("from", "")).strip()
    if begin:
        start = next((i for i, (address, _) in enumerate(found) if address.lower() == begin.lower()), 0)
    lines, size = [], 0
    for address, text in found[start:]:
        line = "%s: %s" % (address, text)
        if size + len(line) > LIMIT and lines:
            break
        lines.append(line)
        size += len(line) + 1
    result = {"path": path, "kind": documents.kind_of(path), "parts": len(found), "text": "\n".join(lines),
              "editable": documents.kind_of(path) != "pdf"}
    if start + len(lines) < len(found):
        result["more"] = "the document goes on: open it from %s" % found[start + len(lines)][0]
    return result


if __name__ == "__main__":
    try:
        args = json.loads(sys.stdin.read() or "{}")
        result = main(args if isinstance(args, dict) else {})
    except Exception as error:  # noqa: BLE001 - a tool reports, never crashes
        result = {"error": str(error)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
