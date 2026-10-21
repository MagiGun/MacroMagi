---
title: Files in Rust
summary: std::fs::read_to_string(path)? reads; std::fs::write(path, text)? writes; create_dir_all makes folders.
tags: [rust, files, fs]
version: "1.90"
---
# Files in Rust

```rust
use std::fs;
use std::path::Path;

let text = fs::read_to_string("data.txt")?;
for line in text.lines() { /* … */ }
fs::write("out.txt", "hello\n")?;
fs::create_dir_all("out/reports")?;
if Path::new("config.toml").exists() { /* … */ }
for entry in fs::read_dir("src")? {
    let entry = entry?;
    println!("{}", entry.path().display());
}
```

`?` needs a function that returns a `Result` – see [errors](errors.md).
