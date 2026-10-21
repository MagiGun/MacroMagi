---
title: Strings in Rust
summary: String owns text, &str borrows it; format! builds; split, trim, parse; push_str adds.
tags: [rust, strings, text]
version: "1.90"
---
# Strings in Rust

```rust
let name = String::from("Ana");
let greeting = format!("Hi {name}, you owe {:.2}", 12.5);
let parts: Vec<&str> = "a,b,c".split(',').collect();
let n: i32 = " 42 ".trim().parse()?;
let mut s = String::new();
s.push_str("abc");
s.contains("b");
s.to_lowercase();
```

A function that only reads text takes `&str`; give it `&my_string`.
