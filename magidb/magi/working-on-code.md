---
title: Working on code
summary: Find the files, read before changing, look up exact syntax in MagiDB, change one thing at a time, run the tests.
tags: [magi, code, workflow, tests]
tools: [list_files, read_file, search_files, edit_file, run_command, db_search]
---
# Working on code

1. **See the project**: `list_files` on its folder; read `README.md` and the main files.
2. **Find what matters**: `search_files` for a function, a message or an error's text.
3. **Read before changing**: `read_file` the part you will change, with its line numbers.
4. **Plan as pseudocode** in your thinking, then look up every call you are not sure of: `db_search` "python read csv", `db_read` the page.
5. **Change one thing at a time** with `edit_file`: copy the text to replace exactly from what `read_file` showed, with enough of it to be there once.
6. **Run the tests** (`run_command`): `python -m pytest`, `cargo test`, `npm test`. Read the output.
7. **When a test fails**, read the error, find its line, fix it, run again.
8. **Answer** with what changed and how it was checked.

## Writing a new file

Write it whole with `write_file`, then run it once to see that it works.

## Many files

Make the list first (from `search_files`), keep it in your thinking – done and next – and work through it.
