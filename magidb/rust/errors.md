---
title: Errors in Rust: Result and ?
summary: Functions that can fail return Result<T, E>; ? passes the error up; main can return Result<(), Box<dyn Error>>.
tags: [rust, errors, result]
version: "1.90"
---
# Errors in Rust: Result and ?

```rust
use std::error::Error;

fn total(path: &str) -> Result<f64, Box<dyn Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut sum = 0.0;
    for line in text.lines() {
        sum += line.trim().parse::<f64>()?;
    }
    Ok(sum)
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("{}", total("numbers.txt")?);
    Ok(())
}
```

- `match result { Ok(v) => …, Err(e) => … }` to handle it there.
- `.unwrap()` stops the program on an error: fine in tests, not in programs.
- `.ok_or("why")?` turns an `Option` into an error.
