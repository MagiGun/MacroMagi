# Native tools

The tools a Magi model runs itself. Each folder is one tool:

- `tool.json` – its name, what it does, its arguments (a JSON Schema), how to run it, and `"confirm": true`
  when it changes files or runs programs;
- `run.py` – the program. It reads the arguments as one JSON object on standard input and prints its
  result as one JSON object on standard output (`{"error": "…"}` when it cannot).

When a model calls one of these tools, MacroMagi (and later CodeMagi) runs it on this computer, with the
Python for AI, and gives the result back to the model in a tool turn – as the tool printed it; the model
then goes on, calling more tools or answering. A tool marked `confirm` runs only when the user allows the
call. A tool whose `run.kind` is `host` (`view_image`) has no program of its own: the program running the
model carries it out. Nothing is sent anywhere except by tools that look things up online (`db_search`
and `db_read` when MagiDB is not on this computer, `web_search`, `fetch_page`, `get_weather`).

The model learns *how to work* with tools – which one, with what arguments, what to do with the result –
not how to do their work, so it can still do exact math, open documents, see pictures and find the exact
code it needs.

| Tools | What they are for |
| --- | --- |
| `db_search`, `db_read` | MagiDB, the library of how things are done (`MAGI_DB_DIR`, `MAGI_DB_URL`, `MAGI_DB_CACHE`) |
| `web_search`, `fetch_page` | The web |
| `list_files`, `read_file`, `search_files`, `write_file`, `edit_file` | Folders and text files (`write_file` and `edit_file` ask first) |
| `run_command` | Running a command line (asks first) |
| `open_document`, `edit_document` | Word, Excel, PowerPoint, PDF and CSV files (`edit_document` asks first; needs python-docx, openpyxl, python-pptx and pypdf) |
| `view_image` | Pictures: a small vision model's description (`describe.py`, `MAGI_VISION_MODEL`) and the system's text recognition |
| `calculator`, `get_time`, `get_weather` | Exact math, the time, the weather |
| `transcribe_audio`, `translate`, `wiki_search` | Starting points that need their own small models or packages |

Copy a folder to make a tool of your own: the model sees it once it is in the Native Tools block.
