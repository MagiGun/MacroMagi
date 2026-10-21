"""MagiDB: Magi's library of how things are done – Markdown pages that link to each other.

A page is a Markdown file. Its first heading is its title and its first paragraph its summary,
unless a front matter (between two `---` lines) says otherwise:

    ---
    title: Reading a file in Python
    summary: open() in a with block; read(), readlines() or a loop over the file.
    tags: [python, files]
    version: "3.12"
    tools: [read_file, run_command]
    ---

Pages are read from the local copy first (MAGI_DB_DIR: MacroMagi's data folder's `magidb`,
downloaded under Settings > Runtime, or any folder of pages), then from the online repository
(MAGI_DB_URL), and kept in a cache (MAGI_DB_CACHE) so a page is fetched once. The model does
not choose: it calls the tool, the tool finds the page.

Searching ignores case and accents, so "Python files", "python FILES" and "Pýthon files" find
the same pages.
"""
import json
import math
import os
import re
import time
import unicodedata
import urllib.parse
import urllib.request

DEFAULT_URL = "https://raw.githubusercontent.com/MacroMagi/MagiDB/main/"
INDEX = "search.json"
SKIP = {".git", ".github", "node_modules", "__pycache__", "tools"}
WORD = re.compile(r"[a-z0-9_+#.]+")
STOP = set("a an and are as at be by for from how i in is it its of on or the to what when where which with "
           "do does can my me you your this that".split())


def local_root():
    root = os.environ.get("MAGI_DB_DIR", "").strip()
    return root if root and os.path.isdir(root) else None


def online_root():
    url = os.environ.get("MAGI_DB_URL", "").strip() or DEFAULT_URL
    return url if url.endswith("/") else url + "/"


def cache_root():
    root = os.environ.get("MAGI_DB_CACHE", "").strip() or os.path.join(
        os.path.expanduser("~"), ".cache", "magidb")
    os.makedirs(root, exist_ok=True)
    return root


def fold(text):
    """Lower case without accents: what searching and page names compare."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def words(text):
    return [w.strip(".") for w in WORD.findall(fold(text)) if w.strip(".") and w.strip(".") not in STOP]


def front_matter(text):
    """The page's front matter as a dict, and the text after it."""
    meta = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end > 0:
            for line in text[3:end].splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    value = value.strip().strip('"').strip("'")
                    if value.startswith("[") and value.endswith("]"):
                        value = [v.strip().strip('"').strip("'") for v in value[1:-1].split(",") if v.strip()]
                    meta[key.strip().lower()] = value
            text = text[end + 4:].lstrip("\n")
    return meta, text


def describe(page, text):
    """A page's entry in the search index."""
    meta, body = front_matter(text)
    title = meta.get("title")
    if not title:
        heading = re.search(r"^#\s+(.+)$", body, re.M)
        title = heading.group(1).strip() if heading else os.path.splitext(os.path.basename(page))[0]
    summary = meta.get("summary")
    if not summary:
        for block in re.split(r"\n\s*\n", body):
            block = block.strip()
            if block and not block.startswith(("#", "```", "|", "-", "*", ">")):
                summary = re.sub(r"\s+", " ", block)
                break
    tags = meta.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    headings = re.findall(r"^#{2,4}\s+(.+)$", body, re.M)
    counts = {}
    for word in words(body):
        counts[word] = counts.get(word, 0) + 1
    return {
        "page": page,
        "title": title,
        "summary": (summary or "")[:300],
        "tags": tags,
        "version": meta.get("version", ""),
        "headings": headings[:40],
        "words": counts,
        "length": sum(counts.values()),
    }


def build_index(root):
    """Every page under a folder, as search entries."""
    entries = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith("."))
        for name in sorted(files):
            if name.lower().endswith(".md") and name.lower() not in ("readme.md", "contributing.md", "license.md"):
                path = os.path.join(folder, name)
                page = os.path.relpath(path, root).replace(os.sep, "/")
                with open(path, encoding="utf-8", errors="replace") as handle:
                    entries.append(describe(page, handle.read()))
    return entries


def fetch(url, timeout=20):
    request = urllib.request.Request(url, headers={"User-Agent": "MacroMagi MagiDB"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def cached(name):
    """A file of the online repository: from the cache, else fetched and kept."""
    path = os.path.join(cache_root(), *name.split("/"))
    if os.path.isfile(path):
        # The index is fetched again after a day; pages stay as they were read.
        if name != INDEX or time.time() - os.path.getmtime(path) < 86400:
            with open(path, encoding="utf-8", errors="replace") as handle:
                return handle.read()
    text = fetch(online_root() + urllib.parse.quote(name))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return text


def index():
    """The search index: the local copy's (its search.json, or made from its pages), else the online one."""
    root = local_root()
    if root:
        built = os.path.join(root, INDEX)
        if os.path.isfile(built):
            with open(built, encoding="utf-8") as handle:
                return json.load(handle), "local"
        return build_index(root), "local"
    return json.loads(cached(INDEX)), "online"


def search(query, entries, limit=6):
    terms = words(query)
    if not terms:
        return []
    count = len(entries) or 1
    average = sum(e.get("length", 0) for e in entries) / count or 1
    scored = []
    for entry in entries:
        title = set(words(entry["title"]))
        tags = set(words(" ".join(entry.get("tags", []))))
        heads = set(words(" ".join(entry.get("headings", []))))
        path = set(words(entry["page"].replace("/", " ").replace("-", " ").replace("_", " ")))
        score = 0.0
        for term in terms:
            frequency = entry["words"].get(term, 0)
            having = sum(1 for e in entries if term in e["words"]) or 0.5
            idf = math.log(1 + (count - having + 0.5) / (having + 0.5))
            score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * entry.get("length", 0) / average))
            score += 3.0 * (term in title) + 2.0 * (term in tags) + 1.5 * (term in heads) + 1.5 * (term in path)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda s: -s[0])
    return [entry for _, entry in scored[:limit]]


def resolve(page, entries):
    """A page asked for by its path (with or without .md, any case) or its title."""
    wanted = fold(page.strip().strip("/")).replace("\\", "/")
    if wanted.endswith(".md"):
        wanted = wanted[:-3]
    for entry in entries:
        name = fold(entry["page"])[:-3] if entry["page"].lower().endswith(".md") else fold(entry["page"])
        if name == wanted or name == wanted + "/index":
            return entry["page"]
    for entry in entries:
        if fold(entry["title"]) == wanted:
            return entry["page"]
    tail = [e["page"] for e in entries if fold(e["page"]).endswith("/" + wanted + ".md")]
    return tail[0] if len(tail) == 1 else None


def read_page(page):
    """A page's text: the local copy's, else the online repository's."""
    root = local_root()
    if root:
        path = os.path.join(root, *page.split("/"))
        if os.path.isfile(path):
            with open(path, encoding="utf-8", errors="replace") as handle:
                return handle.read(), "local"
    return cached(page), "online"


def links_of(page, body):
    """The pages a page links to, resolved from its folder."""
    found = []
    base = os.path.dirname(page)
    for text, target in re.findall(r"\[([^\]]+)\]\(([^)\s]+)\)", body):
        if "://" in target or target.startswith("#") or not target.split("#")[0].lower().endswith(".md"):
            continue
        path = os.path.normpath(os.path.join(base, target.split("#")[0])).replace(os.sep, "/")
        if not path.startswith(".."):
            found.append({"page": path, "title": text})
    return found


def run(main):
    import sys
    try:
        args = json.loads(sys.stdin.read() or "{}")
        result = main(args if isinstance(args, dict) else {})
    except Exception as error:  # noqa: BLE001 - a tool reports, never crashes
        result = {"error": str(error)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
