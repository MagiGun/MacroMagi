"""Searches the Markdown wiki beside the model (the wiki folder) or MAGI_WIKI: titles, matching lines, paths.

MacroMagi native tool: reads its arguments as JSON on standard input and
prints its result as JSON on standard output.
"""
import json
import sys


def main(args):
    import os
    query = str(args.get("query", "")).lower().split()
    if not query:
        return {"error": "no query"}
    root = os.environ.get("MAGI_WIKI") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wiki")
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return {"error": "no wiki folder at " + root}
    found = []
    for folder, _, files in os.walk(root):
        for name in files:
            if not name.endswith(".md"):
                continue
            path = os.path.join(folder, name)
            with open(path, encoding="utf-8", errors="replace") as handle:
                lines = handle.read().splitlines()
            score = sum(" ".join(lines).lower().count(word) for word in query) + 3 * sum(w in name.lower() for w in query)
            if score:
                hits = [l.strip() for l in lines if any(w in l.lower() for w in query)][:3]
                title = next((l.lstrip("# ") for l in lines if l.startswith("#")), name)
                found.append((score, {"title": title, "path": os.path.relpath(path, root), "lines": hits}))
    found.sort(key=lambda f: -f[0])
    return {"results": [f[1] for f in found[:8]]}


if __name__ == "__main__":
    try:
        args = json.loads(sys.stdin.read() or "{}")
        result = main(args if isinstance(args, dict) else {})
    except Exception as error:  # noqa: BLE001 - a tool reports, never crashes
        result = {"error": str(error)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
