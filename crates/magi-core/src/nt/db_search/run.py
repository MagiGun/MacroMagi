"""Searches MagiDB, Magi's library of how things are done: the pages that best match a query,
with their summaries, best first. Read one with db_read.

MacroMagi native tool: reads its arguments as JSON on standard input and
prints its result as JSON on standard output.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import magidb  # noqa: E402 - beside this file


def main(args):
    query = str(args.get("query", "")).strip()
    if not query:
        return {"error": "say what to search for"}
    entries, where = magidb.index()
    found = magidb.search(query, entries)
    if not found:
        return {"query": query, "results": [], "note": "nothing in MagiDB matches; try other words, or web_search"}
    results = []
    for entry in found:
        result = {"page": entry["page"], "title": entry["title"], "summary": entry["summary"]}
        if entry.get("version"):
            result["version"] = entry["version"]
        results.append(result)
    return {"query": query, "from": where, "results": results}


if __name__ == "__main__":
    magidb.run(main)
