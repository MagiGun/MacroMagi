---
title: Sorting
summary: Use the language's sort with a key; sort by several keys with a tuple; stable sorts keep ties in order.
tags: [algorithms, sorting]
---
# Sorting

Do not write a sort: use the language's, with a key.

```python
sorted(people, key=lambda p: (p["city"], -p["age"]))   # by city, then oldest first
sorted(words, key=len, reverse=True)
```

```js
people.sort((a, b) => a.city.localeCompare(b.city) || b.age - a.age);
```

```rust
people.sort_by(|a, b| a.city.cmp(&b.city).then(b.age.cmp(&a.age)));
```
