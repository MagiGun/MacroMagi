"""Documents as Magi reads them: numbered parts of plain text, whatever the file is.

open_document turns a file into parts with addresses; edit_document changes parts by address
and writes the file back in its own format, keeping its styles:

    .docx   paragraphs p1, p2… and table cells t1.r2.c3 (python-docx)
    .xlsx   cells Sheet1!B2 (openpyxl); a value starting with = is a formula
    .pptx   slide texts s1.2.1: slide 1, its second shape, that shape's first paragraph (python-pptx)
    .pdf    pages page1, page2… (pypdf; read only)
    .csv    cells r2.c3
    .txt .md .html and other text: lines l1, l2…

The packages are installed under Settings > Runtime (Document tools).
"""
import csv
import io
import os
import re
import shutil

KINDS = {".docx": "docx", ".xlsx": "xlsx", ".xlsm": "xlsx", ".pptx": "pptx", ".pdf": "pdf", ".csv": "csv"}


def kind_of(path):
    return KINDS.get(os.path.splitext(path)[1].lower(), "text")


def need(package, name):
    try:
        return __import__(package)
    except ImportError:
        raise RuntimeError("reading %s files needs %s: install Document tools under Settings > Runtime"
                           % (name, package))


def parts(path):
    """The document's parts in order: (address, text)."""
    kind = kind_of(path)
    if kind == "docx":
        docx = need("docx", ".docx")
        document = docx.Document(path)
        found = [("p%d" % i, p.text) for i, p in enumerate(document.paragraphs, 1)]
        for t, table in enumerate(document.tables, 1):
            for r, row in enumerate(table.rows, 1):
                for c, cell in enumerate(row.cells, 1):
                    found.append(("t%d.r%d.c%d" % (t, r, c), cell.text))
        return found
    if kind == "xlsx":
        openpyxl = need("openpyxl", ".xlsx")
        book = openpyxl.load_workbook(path, data_only=False)
        found = []
        for sheet in book.worksheets:
            for row in sheet.iter_rows():
                for cell in row:
                    if cell.value is not None:
                        found.append(("%s!%s" % (sheet.title, cell.coordinate), str(cell.value)))
        return found
    if kind == "pptx":
        pptx = need("pptx", ".pptx")
        deck = pptx.Presentation(path)
        found = []
        for s, slide in enumerate(deck.slides, 1):
            for h, shape in enumerate(slide.shapes, 1):
                if shape.has_text_frame:
                    for p, paragraph in enumerate(shape.text_frame.paragraphs, 1):
                        found.append(("s%d.%d.%d" % (s, h, p), "".join(r.text for r in paragraph.runs)))
        return found
    if kind == "pdf":
        pypdf = need("pypdf", "PDF")
        reader = pypdf.PdfReader(path)
        return [("page%d" % i, (page.extract_text() or "").strip()) for i, page in enumerate(reader.pages, 1)]
    if kind == "csv":
        with open(path, encoding="utf-8", errors="replace", newline="") as handle:
            rows = list(csv.reader(handle))
        return [("r%d.c%d" % (r, c), value) for r, row in enumerate(rows, 1) for c, value in enumerate(row, 1)]
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    if path.lower().endswith((".html", ".htm")):
        text = re.sub(r"(?s)<(script|style).*?</\1>", "", text)
        text = re.sub(r"<br\s*/?>|</p>|</div>|</h\d>|</li>", "\n", text)
        text = re.sub(r"<[^>]+>", "", text)
    return [("l%d" % i, line) for i, line in enumerate(text.splitlines(), 1)]


def backup(path):
    """A copy of the file as it was, beside it, before the first change."""
    copy = path + ".magi-backup"
    if not os.path.exists(copy):
        shutil.copy2(path, copy)
    return copy


def set_text(paragraph, text):
    """A paragraph's text, kept in its first run's style."""
    runs = paragraph.runs
    if not runs:
        paragraph.add_run(text)
        return
    runs[0].text = text
    for run in runs[1:]:
        run.text = ""


def edit(path, edits):
    """Changes parts by address; `insert_after` adds a paragraph (.docx) or a line (text)."""
    kind = kind_of(path)
    if kind == "pdf":
        raise RuntimeError("PDF files cannot be changed here; write a new document instead")
    changed = []
    if kind == "docx":
        docx = need("docx", ".docx")
        document = docx.Document(path)
        paragraphs = list(document.paragraphs)
        for change in edits:
            at, text = str(change.get("at") or change.get("insert_after") or ""), str(change.get("text", ""))
            if at.startswith("t"):
                t, r, c = (int(n) for n in re.findall(r"\d+", at)[:3])
                cell = document.tables[t - 1].rows[r - 1].cells[c - 1]
                set_text(cell.paragraphs[0], text)
                for extra in cell.paragraphs[1:]:
                    set_text(extra, "")
            else:
                paragraph = paragraphs[int(at.lstrip("p")) - 1]
                if change.get("insert_after"):
                    new = paragraph.insert_paragraph_before("")
                    # insert_paragraph_before puts it first; move it after.
                    paragraph._p.addnext(new._p)
                    new.style = paragraph.style
                    set_text(new, text)
                else:
                    set_text(paragraph, text)
            changed.append(at)
        backup(path)
        document.save(path)
    elif kind == "xlsx":
        openpyxl = need("openpyxl", ".xlsx")
        book = openpyxl.load_workbook(path)
        for change in edits:
            at, text = str(change.get("at", "")), change.get("text", "")
            sheet, cell = at.split("!", 1) if "!" in at else (book.active.title, at)
            value = text
            if isinstance(text, str) and not text.startswith("="):
                try:
                    value = float(text) if "." in text else int(text)
                except ValueError:
                    value = text
            book[sheet][cell] = value
            changed.append(at)
        backup(path)
        book.save(path)
    elif kind == "pptx":
        pptx = need("pptx", ".pptx")
        deck = pptx.Presentation(path)
        for change in edits:
            at, text = str(change.get("at", "")), str(change.get("text", ""))
            s, h, p = (int(n) for n in re.findall(r"\d+", at)[:3])
            shape = list(deck.slides[s - 1].shapes)[h - 1]
            set_text(shape.text_frame.paragraphs[p - 1], text)
            changed.append(at)
        backup(path)
        deck.save(path)
    elif kind == "csv":
        with open(path, encoding="utf-8", errors="replace", newline="") as handle:
            rows = list(csv.reader(handle))
        for change in edits:
            r, c = (int(n) for n in re.findall(r"\d+", str(change.get("at", "")))[:2])
            while len(rows) < r:
                rows.append([])
            while len(rows[r - 1]) < c:
                rows[r - 1].append("")
            rows[r - 1][c - 1] = str(change.get("text", ""))
            changed.append(change.get("at"))
        backup(path)
        out = io.StringIO()
        csv.writer(out, lineterminator="\n").writerows(rows)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(out.getvalue())
    else:
        with open(path, encoding="utf-8", errors="replace", newline="") as handle:
            content = handle.read()
        ending = "\r\n" if "\r\n" in content else "\n"
        lines = content.splitlines()
        # From the end, so earlier line numbers stay where they were.
        for change in sorted(edits, key=lambda c: -int(re.sub(r"\D", "", str(c.get("at") or c.get("insert_after") or "0")) or 0)):
            at = str(change.get("at") or change.get("insert_after") or "")
            number = int(at.lstrip("l"))
            if change.get("insert_after"):
                lines.insert(number, str(change.get("text", "")))
            else:
                lines[number - 1] = str(change.get("text", ""))
            changed.append(at)
        backup(path)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(ending.join(lines) + (ending if content.endswith(("\n", "\r")) else ""))
    return changed
