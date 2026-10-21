---
title: Text in Python: strings and f-strings
summary: f"{name}: {total:.2f}" formats; split, join, strip, replace, startswith, lower; slicing text[a:b].
tags: [python, strings, text, format]
version: "3.12"
---
# Text in Python

```python
name, total = "Ana", 12.5
f"{name} owes {total:.2f}"         # 'Ana owes 12.50'
f"{count:,}"                       # '12,345'
f"{share:.0%}"                     # '42%'
"a,b,c".split(",")                 # ['a', 'b', 'c']
", ".join(["a", "b"])              # 'a, b'
"  hi \n".strip()                  # 'hi'
"Hello".lower(), "Hello".upper()
"report.csv".endswith(".csv")
"a-b".replace("-", "_")
text[:10], text[-3:]               # the first 10, the last 3 characters
"word" in text                     # contains
len(text)
```

Strings cannot be changed in place: every method gives a new one.
