---
title: JSON in Python
summary: json.load(f) and json.dump(data, f, indent=2) for files; json.loads(text) and json.dumps(data) for strings.
tags: [python, json]
version: "3.12"
---
# JSON in Python

```python
import json

with open("data.json", encoding="utf-8") as f:
    data = json.load(f)                 # a dict or a list

with open("out.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

data = json.loads('{"a": 1}')           # from a string
text = json.dumps(data, indent=2)       # to a string
```

- JSON objects become dicts, arrays lists, `null` None, `true` True.
- `json.decoder.JSONDecodeError`: the text is not valid JSON – often a trailing comma or single quotes.
- Dates and sets are not JSON: turn them into strings or lists first (`default=str` in dumps).
