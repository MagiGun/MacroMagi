---
title: Command-line errors
summary: not recognized / command not found, access denied, file in use – what they mean.
tags: [errors, shell, windows, command line]
---
# Command-line errors

| Error | Means | Do |
| --- | --- | --- |
| `'x' is not recognized as an internal or external command` (Windows) | not installed, or not on the PATH | `where x`; install it; or give its full path |
| `x: command not found` | the same, on Linux and macOS | `which x`; `python3` instead of `python` |
| `Access is denied` / `Permission denied` | no right to it | another folder, or ask the user |
| `The process cannot access the file because it is being used by another process` | open in another program | close it there |
| `The system cannot find the path specified` | a folder in the path does not exist | check each part with `list_files` |
