"""Translation with a small offline model (pip install argostranslate, then install its language pairs).

MacroMagi native tool: reads its arguments as JSON on standard input and
prints its result as JSON on standard output.
"""
import json
import sys


def main(args):
    text = str(args.get("text", ""))
    target = str(args.get("to", "en")).strip().lower()
    source = str(args.get("source", "") or "en").strip().lower()
    try:
        import argostranslate.translate
    except ImportError:
        return {"error": "translate needs argostranslate: pip install argostranslate"}
    return {"text": argostranslate.translate.translate(text, source, target), "from": source, "to": target}


if __name__ == "__main__":
    try:
        args = json.loads(sys.stdin.read() or "{}")
        result = main(args if isinstance(args, dict) else {})
    except Exception as error:  # noqa: BLE001 - a tool reports, never crashes
        result = {"error": str(error)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
