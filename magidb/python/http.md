---
title: Asking a website in Python (HTTP)
summary: urllib.request.urlopen needs nothing installed; requests.get(url, timeout=10).json() is shorter if requests is installed.
tags: [python, http, web, api, requests, urllib]
version: "3.12"
---
# Asking a website in Python (HTTP)

## Without installing anything

```python
import json
import urllib.request

request = urllib.request.Request(url, headers={"User-Agent": "my-script"})
with urllib.request.urlopen(request, timeout=10) as response:
    data = json.load(response)
```

Sending JSON:

```python
body = json.dumps({"name": "Ana"}).encode("utf-8")
request = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
```

## With requests (`pip install requests`)

```python
import requests

response = requests.get(url, params={"q": "rain"}, timeout=10)
response.raise_for_status()      # an error for 4xx and 5xx
data = response.json()
requests.post(url, json={"name": "Ana"}, timeout=10)
```

## Mistakes

- Always set a timeout, or a dead server waits forever.
- `HTTPError 403`: the site wants a User-Agent, or does not allow scripts.
- `ModuleNotFoundError: No module named 'requests'`: `pip install requests` – see [packages](packages.md).
