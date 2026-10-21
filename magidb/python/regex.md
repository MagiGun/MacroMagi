---
title: Regular expressions in Python
summary: re.search finds the first match, re.findall all of them, re.sub replaces; write patterns as raw strings r"...".
tags: [python, regex, re, patterns, text]
version: "3.12"
---
# Regular expressions in Python

```python
import re

match = re.search(r"(\d+)-(\d+)", "pages 10-12")
if match:
    first, last = match.group(1), match.group(2)
re.findall(r"[\w.]+@[\w.]+", text)       # every e-mail address
re.sub(r"\s+", " ", text)                 # many spaces to one
re.split(r"[,;]\s*", "a, b;c")             # ['a', 'b', 'c']
re.search(r"error", line, re.IGNORECASE)
```

| Pattern | Matches |
| --- | --- |
| `\d` `\w` `\s` | a digit, a word character, white space |
| `.` | any character but a newline |
| `*` `+` `?` | 0 or more, 1 or more, 0 or 1 |
| `^` `$` | start, end (of each line with `re.MULTILINE`) |
| `(…)` | a group, `match.group(1)` |
| `[abc]` `[^abc]` | one of, none of |

Always `r"…"`: without it, `\d` must be written `\\d`.
