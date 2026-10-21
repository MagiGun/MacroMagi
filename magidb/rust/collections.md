---
title: Vec and HashMap in Rust
summary: Vec::new, push, iter().map().collect(); HashMap entry(k).or_insert(0) counts; sort_by_key sorts.
tags: [rust, vec, hashmap, collections]
version: "1.90"
---
# Vec and HashMap in Rust

```rust
use std::collections::HashMap;

let mut items = vec![3, 1, 2];
items.push(4);
items.sort();
let doubled: Vec<i32> = items.iter().map(|x| x * 2).collect();
let big: Vec<&i32> = items.iter().filter(|x| **x > 2).collect();

let mut counts: HashMap<String, usize> = HashMap::new();
for word in text.split_whitespace() {
    *counts.entry(word.to_string()).or_insert(0) += 1;
}
people.sort_by_key(|p| p.age);
people.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
```
