---
title: Lists, dictionaries and sorting in Python
summary: sorted(items, key=..., reverse=True); dict.get(key, default); collections.Counter counts; comprehensions build lists and dicts.
tags: [python, lists, dict, sort, counter]
version: "3.12"
---
# Lists, dictionaries and sorting in Python

```python
names = ["bo", "Ana", "cy"]
names.append("Dee")
sorted(names, key=str.lower)                       # a new list
people.sort(key=lambda p: p["age"], reverse=True)  # in place, oldest first
[n.upper() for n in names if len(n) > 2]

ages = {"Ana": 31}
ages.get("Bo", 0)            # 0 when missing (ages["Bo"] is a KeyError)
ages.setdefault("Cy", 20)
for name, age in ages.items():
    ...
{name: len(name) for name in names}

from collections import Counter, defaultdict
Counter(words).most_common(3)          # the 3 commonest, with counts
groups = defaultdict(list)
for p in people:
    groups[p["city"]].append(p)
```

Remove duplicates keeping the order: `list(dict.fromkeys(items))`.
