---
title: Searching
summary: A loop for unsorted lists; binary search (bisect) for sorted ones; a set or dict to look things up fast.
tags: [algorithms, search, binary search, bisect]
---
# Searching

- Once in an unsorted list: a loop, or `next((x for x in items if x.id == wanted), None)`.
- Many times: build a dict first, `by_id = {x.id: x for x in items}`, then `by_id.get(wanted)`.
- In a sorted list: binary search.

```
low, high = 0, length - 1
while low <= high:
    middle = (low + high) / 2, rounded down
    if list[middle] == wanted: found
    if list[middle] < wanted: low = middle + 1
    else: high = middle - 1
```

```python
import bisect
i = bisect.bisect_left(values, wanted)
found = i < len(values) and values[i] == wanted
```
