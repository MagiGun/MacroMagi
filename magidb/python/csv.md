---
title: CSV files in Python
summary: csv.DictReader gives each row as a dict keyed by the header; csv.DictWriter writes dicts; open with newline="".
tags: [python, csv, files, tables]
version: "3.12"
---
# CSV files in Python

## Read

```python
import csv

with open("people.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        print(row["name"], row["age"])     # every value is a string
```

`csv.reader(f)` gives lists instead; the first one is the header.

## Write

```python
with open("out.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "total"])
    writer.writeheader()
    writer.writerow({"name": "Ana", "total": 12})
```

## Mistakes

- Values are strings: `int(row["age"])`, `float(row["price"])`.
- `KeyError: 'Name'`: the header is spelled differently – print `reader.fieldnames`.
- Without `newline=""` Windows writes blank lines between rows.
- Another separator: `csv.DictReader(f, delimiter=";")`.
