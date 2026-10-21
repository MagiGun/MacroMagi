---
title: Running other programs from Python
summary: subprocess.run([...], capture_output=True, text=True, check=True) runs a program and gives its output.
tags: [python, subprocess, commands, shell]
version: "3.12"
---
# Running other programs from Python

```python
import subprocess

done = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, cwd="project")
print(done.returncode)    # 0 is success
print(done.stdout)
print(done.stderr)
```

- `check=True` raises `CalledProcessError` when it fails.
- `timeout=60` stops it after 60 seconds (`TimeoutExpired`).
- Give a list, not one string; `shell=True` with a string only when you need the shell's pipes, and never with text from users.
