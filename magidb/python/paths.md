---
title: Paths and folders in Python
summary: pathlib.Path joins with /, checks exists(), lists with iterdir() or glob(), makes folders with mkdir(parents=True, exist_ok=True).
tags: [python, paths, folders, pathlib, files]
version: "3.12"
---
# Paths and folders in Python

```python
from pathlib import Path

root = Path("project")
config = root / "config" / "settings.json"   # joins with the right slash
config.exists()        # True or False
config.is_file()
config.parent          # Path("project/config")
config.name            # "settings.json"
config.stem            # "settings"
config.suffix          # ".json"
text = config.read_text(encoding="utf-8")
config.write_text(text, encoding="utf-8")
```

## Folders

```python
Path("out/reports").mkdir(parents=True, exist_ok=True)
for item in Path("src").iterdir():        # one level
    print(item.name, item.is_dir())
for py in Path("src").rglob("*.py"):      # every level
    print(py)
```

## Home and the current folder

`Path.home()`, `Path.cwd()`; `Path("~/notes").expanduser()`.

Delete a file with `path.unlink()`, an empty folder with `path.rmdir()`, a full one with `shutil.rmtree(path)`.
