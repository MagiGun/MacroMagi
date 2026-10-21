---
title: Counting and grouping
summary: A dictionary from key to count (or list); in Python, Counter and defaultdict(list).
tags: [algorithms, counting, grouping, frequency]
---
# Counting and grouping

```
counts = empty map
for each item:
    counts[key of item] = counts[key of item] + 1  (start at 0)
```

```python
from collections import Counter, defaultdict

counts = Counter(word.lower() for word in text.split())
by_city = defaultdict(list)
for person in people:
    by_city[person["city"]].append(person)
```

The commonest: `counts.most_common(5)`. Duplicates: keys whose count is above 1.
