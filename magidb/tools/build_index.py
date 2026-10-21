"""Builds search.json – what db_search reads online – from MagiDB's pages, and checks the links.

    python tools/build_index.py

magidb.py beside this file is a copy of the db_search tool's (crates/magi-core/src/nt/db_search in
MacroMagi): the index is made the way the tool reads it.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import magidb  # noqa: E402 - beside this file

ROOT = os.path.dirname(HERE)


def main():
    entries = magidb.build_index(ROOT)
    pages = {e["page"] for e in entries}
    broken = []
    for entry in entries:
        with open(os.path.join(ROOT, entry["page"]), encoding="utf-8") as handle:
            _, body = magidb.front_matter(handle.read())
        for link in magidb.links_of(entry["page"], body):
            if link["page"] not in pages:
                broken.append("%s -> %s" % (entry["page"], link["page"]))
    with open(os.path.join(ROOT, magidb.INDEX), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(entries, handle, ensure_ascii=False, separators=(",", ":"))
    print("%d pages indexed" % len(entries))
    for line in broken:
        print("broken link:", line)
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
