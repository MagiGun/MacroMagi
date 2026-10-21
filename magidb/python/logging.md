---
title: Logging in Python
summary: logging.basicConfig(level=logging.INFO) once, then logging.getLogger(__name__).info(...).
tags: [python, logging, debug]
version: "3.12"
---
# Logging in Python

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.info("read %d rows", count)
log.warning("no header in %s", path)
log.debug("only shown at level DEBUG")
```

A `--verbose` flag: `level=logging.DEBUG if args.verbose else logging.INFO`.
