---
title: Arrays in JavaScript
summary: map, filter, reduce, find, some, sort with a compare function, includes, spread [...a, ...b].
tags: [javascript, arrays, map, filter, sort]
---
# Arrays in JavaScript

```js
const prices = items.map(item => item.price);
const cheap = items.filter(item => item.price < 10);
const total = prices.reduce((sum, p) => sum + p, 0);
const first = items.find(item => item.id === 7);      // undefined when none
items.some(item => item.sold);                        // true or false
items.sort((a, b) => a.price - b.price);              // numbers need the function
names.sort((a, b) => a.localeCompare(b));
names.includes("Ana");
const all = [...a, ...b];
const unique = [...new Set(names)];
```

`sort` changes the array itself; `toSorted` gives a new one.
