---
title: Errors and exceptions in Python
summary: try/except catches a named exception; raise ValueError("why") reports one; finally always runs.
tags: [python, exceptions, errors, try]
version: "3.12"
---
# Errors and exceptions in Python

```python
try:
    value = int(text)
except ValueError:
    value = 0
except (KeyError, IndexError) as error:
    print("missing:", error)
finally:
    cleanup()

if total < 0:
    raise ValueError(f"total cannot be negative: {total}")
```

- Catch the exception you expect, not every one (`except Exception` hides mistakes).
- A traceback's **last line** is the error; the line above it is where it happened. See [Python errors](../errors/python.md).
