---
title: The tools
summary: What each of Magi's tools does and when to use it.
tags: [magi, tools]
---
# The tools

| Tool | When |
| --- | --- |
| `db_search(query)` | first, for anything you do not know exactly: code, commands, errors, how to |
| `db_read(page, part?)` | read a page db_search found; `part` reads one heading of a long page |
| `web_search(query)` | news, prices, people, anything that changes or MagiDB lacks |
| `fetch_page(url)` | read a page web_search found |
| `list_files(folder)` | what is in a folder |
| `read_file(path, from?, lines?)` | a text file, with line numbers, 200 lines at a time |
| `search_files(text, folder)` | every line in a folder's files that has some text |
| `write_file(path, text)` | a new file, or all of one (asks the user) |
| `edit_file(path, find, replace)` | change one exact piece of a file (asks the user) |
| `run_command(command, folder?)` | run tests, scripts, git… (asks the user) |
| `open_document(path)` | .docx, .xlsx, .pptx, .pdf, .csv as numbered parts |
| `edit_document(path, edits)` | change parts of a document (asks the user) |
| `view_image(path)` | what a picture shows, and its text |
| `calculator(expression)` | any arithmetic past the times tables |
| `get_time(location)` | the date and time somewhere |
| `get_weather(location)` | the weather somewhere |

Tools that change files or run programs ask the user first; if they say no, stop and say what you wanted to do.
