---
title: Files in Node.js
summary: import { readFile, writeFile } from "node:fs/promises"; await readFile(path, "utf8").
tags: [javascript, node, files, fs]
---
# Files in Node.js

```js
import { readFile, writeFile, readdir, mkdir } from "node:fs/promises";
import path from "node:path";

const text = await readFile("data.txt", "utf8");
const data = JSON.parse(await readFile("data.json", "utf8"));
await writeFile("out.json", JSON.stringify(data, null, 2));
await mkdir("out/reports", { recursive: true });
for (const name of await readdir("src")) console.log(path.join("src", name));
```

`import` needs `"type": "module"` in package.json (or a `.mjs` file); otherwise use `require("node:fs")`.
