"""Reads a MagiDB page: its text (or one part of it), the parts it has and the pages it links to.

MacroMagi native tool: reads its arguments as JSON on standard input and
prints its result as JSON on standard output.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import magidb  # noqa: E402 - beside this file

LIMIT = 6000


def main(args):
    asked = str(args.get("page", "")).strip()
    if not asked:
        return {"error": "say which page to read (a page from db_search)"}
    entries, _ = magidb.index()
    page = magidb.resolve(asked, entries) or (asked if asked.lower().endswith(".md") else asked + ".md")
    try:
        text, where = magidb.read_page(page)
    except Exception:  # noqa: BLE001 - a missing page is an answer
        close = magidb.search(asked.replace("/", " "), entries, limit=3)
        return {"error": "no such page: " + asked, "try": [e["page"] for e in close]}
    meta, body = magidb.front_matter(text)
    parts = re.findall(r"^#{2,4}\s+(.+)$", body, re.M)
    part = str(args.get("part", "")).strip()
    if part:
        wanted = magidb.fold(part)
        found = None
        for match in re.finditer(r"^(#{2,4})\s+(.+)$", body, re.M):
            if magidb.fold(match.group(2)).startswith(wanted):
                found = match
                break
        if found is None:
            return {"page": page, "error": "no part named " + part, "parts": parts}
        level = len(found.group(1))
        rest = body[found.end():]
        end = re.search(r"^#{1,%d}\s" % level, rest, re.M)
        body = found.group(0) + rest[: end.start() if end else len(rest)]
    result = {"page": page, "from": where}
    if meta.get("title"):
        result["title"] = meta["title"]
    if meta.get("version"):
        result["version"] = meta["version"]
    if len(body) > LIMIT:
        result["text"] = body[:LIMIT]
        result["more"] = "the page goes on: read one part of it with part"
    else:
        result["text"] = body
    if parts and not part:
        result["parts"] = parts
    links = magidb.links_of(page, body)
    if links:
        result["links"] = links[:30]
    return result


if __name__ == "__main__":
    magidb.run(main)
