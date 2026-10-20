# Native tools

The tools a Magi model runs itself. Each folder is one tool:

- `tool.json` – its name, what it does, its arguments (a JSON Schema) and how to run it;
- `run.py` – the program. It reads the arguments as one JSON object on standard input and prints its
  result as one JSON object on standard output (`{"error": "…"}` when it cannot).

When a model calls one of these tools, MacroMagi (and later CodeMagi) runs it on this computer, with the
Python for AI, and gives the result back to the model in a tool turn; the model then goes on, calling
more tools or answering. Nothing is sent anywhere except by tools that look things up online
(`web_search`, `fetch_page`, `get_weather`).

The model learns *how to call* tools – which one, with what arguments – not how to do their work, so a
tiny model can still do exact math, read pictures and sound, and look things up.

Some tools are a starting point that needs a small model of its own (`describe_image`,
`transcribe_audio`, `translate`); each says what to install or where to put the model. Copy a folder to
make a tool of your own: the model sees it once it is in the Tools block.
