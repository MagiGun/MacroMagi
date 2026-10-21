---
title: Reading and writing files in Python
summary: open() in a with block; read() for all of it, a for loop for lines; "w" writes, "a" adds; always give encoding="utf-8".
tags: [python, files, read, write]
version: "3.12"
---
# Reading and writing files in Python

## Read all of it

```python
with open(path, encoding="utf-8") as f:
    text = f.read()
```

## Line by line

```python
with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
```

`f.readlines()` gives a list of lines, each still ending with `\n`.

## Write (replaces the file)

```python
with open(path, "w", encoding="utf-8") as f:
    f.write(text)
```

## Add to the end

```python
with open(path, "a", encoding="utf-8") as f:
    f.write(line + "\n")
```

## Bytes

`open(path, "rb")` and `"wb"` read and write bytes (pictures, zips).

## Mistakes

- Without `encoding="utf-8"`, Windows uses another encoding: `UnicodeDecodeError`.
- `"w"` empties the file first; use `"a"` to add.
- See [paths](paths.md) for building paths and checking a file is there.
