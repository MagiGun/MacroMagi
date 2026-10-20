"""Web search through DuckDuckGo's HTML page (online).

MacroMagi native tool: reads its arguments as JSON on standard input and
prints its result as JSON on standard output.
"""
import json
import sys


def main(args):
    import html
    import re
    import urllib.parse
    import urllib.request
    query = str(args.get("query", "")).strip()
    if not query:
        return {"error": "no query"}
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (MacroMagi native tool)"})
    page = urllib.request.urlopen(request, timeout=15).read().decode("utf-8", "replace")
    results = []
    for link, title, snippet in re.findall(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</a>',
        page, re.S):
        clean = lambda text: html.unescape(re.sub(r"<[^>]+>", "", text)).strip()
        target = urllib.parse.parse_qs(urllib.parse.urlparse(html.unescape(link)).query).get("uddg", [link])[0]
        results.append({"title": clean(title), "url": target, "snippet": clean(snippet)})
        if len(results) == 5:
            break
    return {"results": results}


if __name__ == "__main__":
    try:
        args = json.loads(sys.stdin.read() or "{}")
        result = main(args if isinstance(args, dict) else {})
    except Exception as error:  # noqa: BLE001 - a tool reports, never crashes
        result = {"error": str(error)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
