"""The readable text of a web page (online).

MacroMagi native tool: reads its arguments as JSON on standard input and
prints its result as JSON on standard output.
"""
import json
import sys


def main(args):
    import html
    import re
    import urllib.request
    url = str(args.get("url", "")).strip()
    if not url.startswith(("http://", "https://")):
        return {"error": "the address must start with http:// or https://"}
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (MacroMagi native tool)"})
    page = urllib.request.urlopen(request, timeout=20).read(2_000_000).decode("utf-8", "replace")
    page = re.sub(r"(?is)<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", page)
    text = html.unescape(re.sub(r"<[^>]+>", " ", page))
    text = re.sub(r"\s+", " ", text).strip()
    limit = int(args.get("limit", 4000))
    return {"text": text[:limit], "more": len(text) > limit}


if __name__ == "__main__":
    try:
        args = json.loads(sys.stdin.read() or "{}")
        result = main(args if isinstance(args, dict) else {})
    except Exception as error:  # noqa: BLE001 - a tool reports, never crashes
        result = {"error": str(error)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
