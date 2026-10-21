---
title: Command-line arguments in Python
summary: argparse.ArgumentParser with add_argument for positional arguments and --options; parse_args() gives them.
tags: [python, argparse, cli, arguments, flags]
version: "3.12"
---
# Command-line arguments in Python

```python
import argparse

parser = argparse.ArgumentParser(description="Totals a CSV file.")
parser.add_argument("path", help="the CSV file")
parser.add_argument("--column", default="amount")
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--limit", type=int, default=10)
args = parser.parse_args()
print(args.path, args.column, args.verbose, args.limit)
```

`python totals.py data.csv --column price -v` – and `--help` is written for you.
