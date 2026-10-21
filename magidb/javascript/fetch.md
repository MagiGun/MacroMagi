---
title: Asking a server in JavaScript (fetch)
summary: const data = await (await fetch(url)).json(); check response.ok; send JSON with method POST and a Content-Type header.
tags: [javascript, fetch, http, api, async]
---
# Asking a server in JavaScript (fetch)

```js
async function load(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return await response.json();
}

await fetch("/api/items", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ name: "Ana" }),
});
```

- `await` works inside `async` functions (and at the top of modules).
- fetch does not fail on 404 or 500: check `response.ok`.
- A CORS error in the browser means the server does not allow this page to ask it; it is fixed on the server.
