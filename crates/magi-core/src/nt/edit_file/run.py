"""Changes a text file: replaces one exact piece of it, there once, with another. The user is asked first.

MacroMagi native tool: reads its arguments as JSON on standard input and
prints its result as JSON on standard output.
"""
import json
import sys

import os


def main(args):
    path = os.path.expanduser(str(args.get("path", "")).strip())
    find, replace = str(args.get("find", "")), str(args.get("replace", ""))
    if not os.path.isfile(path):
        return {"error": "no such file: " + path}
    if not find:
        return {"error": "say what to replace"}
    with open(path, encoding="utf-8", newline="") as handle:
        content = handle.read()
    count = content.count(find)
    if count == 0:
        # Line endings are the usual difference.
        if "\r\n" in content and "\n" in find and "\r\n" not in find:
            find, replace = find.replace("\n", "\r\n"), replace.replace("\n", "\r\n")
            count = content.count(find)
    if count == 0:
        return {"error": "the text to replace is not in the file; read the file again and copy it exactly"}
    if count > 1:
        return {"error": "the text to replace is there %d times; give more of it so it is there once" % count}
    at = content.index(find)
    line = content[:at].count("\n") + 1
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(content.replace(find, replace, 1))
    return {"path": path, "line": line, "removed_lines": find.rstrip("\n").count("\n") + 1,
            "added_lines": replace.rstrip("\n").count("\n") + 1}


if __name__ == "__main__":
    try:
        args = json.loads(sys.stdin.read() or "{}")
        result = main(args if isinstance(args, dict) else {})
    except Exception as error:  # noqa: BLE001 - a tool reports, never crashes
        result = {"error": str(error)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
