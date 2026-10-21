---
title: When a tool fails
summary: Read the error, fix the call, try another way; never invent a result.
tags: [magi, errors, workarounds, tools]
---
# When a tool fails

Read the error and change something before calling again – the same call fails the same way.

| Error | What to do |
| --- | --- |
| no such file / folder | `list_files` the folder above it; check the spelling and the slashes. |
| edit_file: the text is not in the file | `read_file` again and copy the text exactly (spaces, quotes). |
| edit_file: there 2 times | give more of the text around it, so it is there once. |
| the user did not allow this call | do not try again; say what you wanted to do and ask. |
| the command was not found | the program is not installed: say so, or look up how to install it. |
| a command failed | read its output: the last lines name the problem. `db_search` the error message. |
| nothing in MagiDB matches | other words, or `web_search`. |
| a page or site cannot be reached | try another result, or say it could not be reached. |

Never write a tool's result yourself, and never guess what a file holds: read it.
