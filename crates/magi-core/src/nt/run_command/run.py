"""Runs a command line in a folder and gives back its exit code and what it printed (the last part). The user is asked first.

MacroMagi native tool: reads its arguments as JSON on standard input and
prints its result as JSON on standard output.
"""
import json
import sys

import os
import subprocess


def main(args):
    command = str(args.get("command", "")).strip()
    folder = os.path.expanduser(str(args.get("folder", "")).strip() or os.getcwd())
    if not command:
        return {"error": "say what to run"}
    if not os.path.isdir(folder):
        return {"error": "no such folder: " + folder}
    try:
        done = subprocess.run(command, shell=True, cwd=folder, capture_output=True, timeout=110,
                              stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"error": "it ran for longer than 110 seconds and was stopped"}
    out = (done.stdout + done.stderr).decode("utf-8", errors="replace")
    result = {"exit": done.returncode, "output": out[-6000:]}
    if len(out) > 6000:
        result["cut"] = "only the last 6,000 characters are shown"
    return result


if __name__ == "__main__":
    try:
        args = json.loads(sys.stdin.read() or "{}")
        result = main(args if isinstance(args, dict) else {})
    except Exception as error:  # noqa: BLE001 - a tool reports, never crashes
        result = {"error": str(error)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
