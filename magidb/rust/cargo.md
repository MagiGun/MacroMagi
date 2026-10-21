---
title: Cargo: build, test, add
summary: cargo build, cargo run, cargo test, cargo add name, cargo fmt, cargo clippy.
tags: [rust, cargo, build, test]
version: "1.90"
---
# Cargo

```
cargo new my_tool          # a new program
cargo build --release
cargo run -- --input a.txt # arguments after --
cargo test                 # every #[test]
cargo test total           # tests with "total" in the name
cargo add serde --features derive
cargo fmt
cargo clippy
```

A test:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds() {
        assert_eq!(total(&[1.0, 2.0]), 3.0);
    }
}
```
