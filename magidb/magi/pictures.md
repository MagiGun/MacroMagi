---
title: Pictures
summary: Magi does not see pixels; view_image describes a picture and reads the text in it.
tags: [magi, images, pictures, screenshots]
tools: [view_image]
---
# Pictures

When a request mentions or attaches a picture (`Attached: … (picture)`), call `view_image` with its path first. It gives back:

- `description`: what the picture shows, from a small vision model;
- `text`: the text written in it, line by line (screenshots, signs, documents);
- `width` and `height`.

Answer from those. If the description is missing (the model is not installed), the text and size are still there; say what you could not tell.
