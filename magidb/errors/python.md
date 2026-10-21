---
title: Python errors
summary: ModuleNotFoundError, FileNotFoundError, KeyError, TypeError, IndentationError, UnicodeDecodeError and more – what they mean and the fix.
tags: [errors, python, traceback]
version: "3.12"
---
# Python errors

The last line of a traceback is the error; the lines above say where.

| Error | Means | Do |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'x'` | the package is not installed | `python -m pip install x` (names differ: see [packages](../python/packages.md)) |
| `FileNotFoundError` | the path is wrong, or relative to another folder | check it with `list_files`; build it from `Path(__file__).parent` |
| `KeyError: 'name'` | the dict has no such key | print the keys; use `.get("name")` |
| `IndexError: list index out of range` | the list is shorter | check `len()` first |
| `TypeError: 'NoneType' object is not subscriptable` | something returned None | find what returned None (a function without `return`) |
| `TypeError: can only concatenate str (not "int") to str` | text + number | `f"{n}"` or `str(n)` |
| `ValueError: invalid literal for int()` | the text is not a number | `.strip()` it, or handle the error |
| `AttributeError: 'X' object has no attribute 'y'` | wrong name, or a different type than you think | `print(type(x))` |
| `IndentationError` | spaces do not line up | four spaces per level, no tabs |
| `UnicodeDecodeError` | the file is not read as UTF-8 | `open(path, encoding="utf-8")`, or `errors="replace"` |
| `PermissionError` | the file is open elsewhere, or protected | close it (Excel), or choose another place |
| `RecursionError` | a function calls itself forever | check the stopping case |
