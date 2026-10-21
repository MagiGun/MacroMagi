---
title: The page in JavaScript (the DOM)
summary: document.querySelector finds an element; addEventListener reacts; textContent and classList change it.
tags: [javascript, dom, html, browser, events]
---
# The page in JavaScript (the DOM)

```js
const button = document.querySelector("#save");
const rows = document.querySelectorAll(".row");
button.addEventListener("click", () => {
  status.textContent = "Saved";
  status.classList.add("ok");
});
const item = document.createElement("li");
item.textContent = name;
list.append(item);
```

Run code after the page loads: put the `<script>` at the end of `<body>`, or use `<script defer>`.
`textContent` is safe; `innerHTML` with text from users is not.
