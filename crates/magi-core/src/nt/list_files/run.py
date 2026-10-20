"""Lists a folder (read-only).

MacroMagi native tool: reads its arguments as JSON on standard input and
prints its result as JSON on standard output.
"""
import json
import sys


def main(args):
    import os
    folder = os.path.expanduser(str(args.get("folder", "")).strip() or ".")
    if not os.path.isdir(folder):
        return {"error": "no such folder: " + folder}
    entries = sorted(os.listdir(folder))[:500]
    return {"folder": folder, "files": [e for e in entries if os.path.isfile(os.path.join(folder, e))],
            "folders": [e for e in entries if os.path.isdir(os.path.join(folder, e))]}


if __name__ == "__main__":
    try:
        args = json.loads(sys.stdin.read() or "{}")
        result = main(args if isinstance(args, dict) else {})
    except Exception as error:  # noqa: BLE001 - a tool reports, never crashes
        result = {"error": str(error)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
