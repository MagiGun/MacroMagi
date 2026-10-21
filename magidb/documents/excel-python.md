---
title: Excel files in Python
summary: openpyxl: load_workbook(path), sheet["B2"].value, iter_rows(values_only=True), save(path).
tags: [documents, excel, xlsx, python, openpyxl]
version: "3.12"
---
# Excel files in Python (openpyxl)

```python
from openpyxl import load_workbook, Workbook     # pip install openpyxl

book = load_workbook("prices.xlsx")              # data_only=True: formulas' values
sheet = book["Sheet1"]                           # or book.active
sheet["B2"].value
for name, price in sheet.iter_rows(min_row=2, values_only=True):
    print(name, price)
sheet["C2"] = "=B2*1.05"
book.save("prices.xlsx")
```

Excel must be closed while the file is saved, or it fails with a permission error.
