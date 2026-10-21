---
title: Word files in Python
summary: python-docx: Document(path).paragraphs to read, add_paragraph and add_heading to write, save(path).
tags: [documents, word, docx, python]
version: "3.12"
---
# Word files in Python (python-docx)

```python
from docx import Document      # pip install python-docx

doc = Document("report.docx")
for paragraph in doc.paragraphs:
    print(paragraph.style.name, paragraph.text)
for table in doc.tables:
    for row in table.rows:
        print([cell.text for cell in row.cells])

new = Document()
new.add_heading("Summary", level=1)
new.add_paragraph("Sales rose by 4%.")
new.save("summary.docx")
```
