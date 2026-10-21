---
title: The command line on Windows
summary: dir lists, cd moves, type shows a file, where finds a program; PowerShell commands run with powershell -Command.
tags: [shell, windows, cmd, powershell]
---
# The command line on Windows

| To | cmd | PowerShell |
| --- | --- | --- |
| list a folder | `dir` | `Get-ChildItem` (`ls`) |
| show a file | `type file.txt` | `Get-Content file.txt` |
| find text in files | `findstr /s /i "text" *.py` | `Select-String -Path *.py -Pattern text` |
| find a program | `where python` | `Get-Command python` |
| make a folder | `mkdir out` | `New-Item -ItemType Directory out` |
| copy, move | `copy a b`, `move a b` | `Copy-Item`, `Move-Item` |

`run_command` uses cmd: write `powershell -NoProfile -Command "Get-ChildItem"` for PowerShell.
Paths take `\` or `/`; quote paths with spaces.
