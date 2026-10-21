---
title: SQLite from Python
summary: sqlite3.connect(path); execute with ? placeholders; fetchall; commit after changes.
tags: [sql, sqlite, python, database]
version: "3.12"
---
# SQLite from Python

```python
import sqlite3

with sqlite3.connect("shop.db") as db:
    db.execute("CREATE TABLE IF NOT EXISTS items (name TEXT, price REAL)")
    db.execute("INSERT INTO items VALUES (?, ?)", ("pen", 1.5))
    rows = db.execute("SELECT name, price FROM items WHERE price < ?", (5,)).fetchall()
```

Always `?` placeholders, never f-strings in SQL. The `with` block commits.
