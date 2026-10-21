---
title: The command line on Linux and macOS
summary: ls lists, cat shows, grep -rn finds text, find finds files, which finds programs.
tags: [shell, linux, macos, bash]
---
# The command line on Linux and macOS

```
ls -la
cat file.txt
grep -rn "total" src/        # every line with "total", with file and line
find . -name "*.py"
which python3
mkdir -p out/reports
cp a b; mv a b
command > out.txt 2>&1        # its output, errors too, into a file
```

`python3`, not `python`, on many systems.
