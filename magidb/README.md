# MagiDB

MagiDB is the library Magi models look things up in. A Magi model knows *how to work* – plan, look
up, act, check – but not facts or exact code: when it needs the exact way to read a CSV file in
Python, what an error message means, or how to undo a Git commit, it searches MagiDB with its
`db_search` tool and reads a page with `db_read`.

It is plain Markdown: pages that link to each other, so a model (or a person) can start at
[index.md](index.md) and follow the links.

## How a model reads it

MacroMagi's `db_search` and `db_read` tools look for MagiDB in this order:

1. a folder chosen under *Settings › Runtime › MagiDB folder* (a copy you are writing);
2. the copy downloaded under *Settings › Runtime › MagiDB* (faster, and it works offline);
3. this repository online, page by page, each kept in MacroMagi's cache once read.

The model does not choose: it calls the tool, and the tool finds the page. Searching ignores case
and accents.

## Writing a page

A page is a Markdown file in a topic's folder. Its first heading is its title and its first
paragraph its summary – what search results show, so make it say what the page answers. A front
matter can say more:

```markdown
---
title: Reading a CSV file in Python
summary: csv.DictReader gives each row as a dictionary keyed by the header.
tags: [python, csv, files]
version: "3.12"
tools: [read_file, run_command]
---
# Reading a CSV file in Python
…
```

- **One answer per page**, short: the exact code or command, what each part does, the usual
  mistakes. Long topics are split into pages that link to each other.
- **Parts** (`##` headings) can be read one at a time (`db_read` with `part`).
- **Links** are relative (`[Writing files](writing-files.md)`); `db_read` lists them so the model
  can go on reading.
- **Versions**: say which version of a language or program the page is for (`version`).
- **Workarounds**: when something often fails, say what to do instead.

## The search index

`search.json` is what `db_search` reads online. After adding or changing pages, build it again:

```bash
python tools/build_index.py
```

A local copy without `search.json` is indexed on the fly.

## Licence

The pages are CC BY 4.0 unless a page says otherwise; code in them is MIT – copy it freely.
