---
title: JavaScript and npm errors
summary: TypeError undefined, is not a function, CORS, Cannot find module, npm ERR! – what they mean and the fix.
tags: [errors, javascript, npm, node]
---
# JavaScript and npm errors

| Error | Means | Do |
| --- | --- | --- |
| `TypeError: Cannot read properties of undefined (reading 'x')` | the thing before `.x` is undefined | check where it comes from; `obj?.x` |
| `x is not a function` | wrong name, or not what you think | `console.log(typeof x)` |
| `ReferenceError: x is not defined` | never declared, or out of scope | declare it, check spelling |
| `blocked by CORS policy` | the server does not allow this page | fix on the server, or ask through your own server |
| `Cannot find module 'x'` | not installed | `npm install` (or `npm install x`) |
| `SyntaxError: Cannot use import statement outside a module` | CommonJS file | `"type": "module"` in package.json |
| `npm ERR! code ERESOLVE` | versions disagree | read which; `npm install --legacy-peer-deps` as a last resort |
