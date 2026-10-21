---
title: Documents
summary: open_document turns .docx, .xlsx, .pptx, .pdf and .csv files into numbered parts; edit_document changes parts and saves the file in its own format.
tags: [magi, documents, word, excel, powerpoint, pdf]
tools: [open_document, edit_document]
---
# Documents

Do not read a .docx or .xlsx with `read_file`: use `open_document`. It gives each part an address:

| File | Parts |
| --- | --- |
| .docx | paragraphs `p1`, `p2`… and table cells `t1.r2.c3` |
| .xlsx | cells `Sheet1!B2` (a value starting with `=` is a formula) |
| .pptx | `s1.2.1`: slide 1, its second shape, the shape's first paragraph |
| .pdf | pages `page1`, `page2`… – read only |
| .csv | cells `r2.c3` |
| text, Markdown, HTML | lines `l1`, `l2`… |

To change it, call `edit_document` with `edits`: a list of `{"at": "p3", "text": "the new text"}`, or `{"insert_after": "p3", "text": "…"}` for a new paragraph or line. The file keeps its styles, and a copy of it as it was is kept beside it (`.magi-backup`).

A long document comes in parts: open it again with `from` set to the part it says.

A PDF cannot be changed: write a new document (Markdown, or a .docx) instead.
