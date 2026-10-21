# Changelog

Notable changes to MacroMagi. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

File-format changes are called out explicitly; see [docs/FILE_FORMAT.md](docs/FILE_FORMAT.md) for the compatibility rules.

## 0.1.7.2 – Unreleased

The second update of 0.1.7 (Cargo keeps 0.1.7).

### Added

- **Magi: one architecture, an agent.** File › New text model is **Magi** – about 418 million parameters, 148 million at work per token: 768 wide and 32 layers deep, twelve heads sharing four key/value heads, queries and keys normalized; three layers in four look at the last 1,024 tokens and the fourth at everything; after three plain layers, DeepSeekMoE (a shared and sixteen routed experts, two per token, √Softplus, bias balancing); multi-token prediction while training; 8K tokens in training and YaRN to 128K when it runs; 32K tokens learned where it trains; thinking on, efforts *standard* and *superior*. It knows how to work, not facts or exact code, and trains on a GPU server (File › Export as Python, Colab's G4: about a day).
- **Magi's recipe**: pretraining on three billion tokens of FineWeb-Edu and OpenCoder's code corpus (two Hugging Face Dataset blocks, chained), fine-tuning on smol-smoltalk and the example datasets, a compaction stage, and reinforcement learning with the agent's reward rules.
- **MagiDB**: Magi's library of how things are done – linked Markdown pages (how Magi works, Python, JavaScript, Rust, Git, the command line, SQL, documents, algorithms, error messages), started in the repository's `magidb/` with its search index and how to write pages. `db_search` and `db_read` read a chosen folder, the copy downloaded under Settings › Runtime, or the repository online, page by page and cached; searching ignores case and accents.
- **Magi's tools**: `db_search`, `db_read`, `search_files`, `write_file`, `edit_file`, `run_command`, `open_document`, `edit_document` and `view_image` join `web_search`, `fetch_page`, `list_files`, `read_file` (now numbered lines, 200 at a time), `calculator`, `get_time` and `get_weather`. Documents (.docx, .xlsx, .pptx, .pdf, .csv) are opened as numbered parts of plain text and changed by part, saved in their own format with their styles and a backup. Pictures are described by a small vision model (SmolVLM-256M) and their text read by Windows' text recognition.
- **Calls that change files or run programs ask first**: a window shows the command, the text or the change, with *Allow*, *Allow until MacroMagi closes* and *Don't allow*; a refused call tells the model so. **File format:** a native tool's `confirm`.
- **Tools MacroMagi carries out itself** (`"run": {"kind": "host"}`), such as `view_image`. **File format.**
- **Long work**: up to 100 tool calls a request by default (200 at most). When the work outgrows the context, earlier results are shortened and then the oldest steps left out, keeping the system text and the request – and the trainer learns every turn from exactly that text.
- **`work.jsonl`**: 1,400 long pieces of work, up to 78 tool calls each, every result from the real tools run on real files (`make_work_data.py`): fixing tests, adding options, renaming across up to 30 files, docstrings everywhere, TODO lists, scripts from data (pseudocode, MagiDB, run, fixed), spreadsheets, Word files, PDFs, screenshots, receipts, MagiDB questions, error messages, Git.
- **Attaching pictures and documents** in the AI Agent window: they are named by their path, and the model looks at them or opens them with its tools.
- **Settings › Runtime › Magi's tools**: MagiDB (downloaded for offline use), Document tools (python-docx, openpyxl, python-pptx, pypdf) and Image descriptions (transformers and SmolVLM-256M-Instruct), with the address MagiDB is read from online and a folder of pages to read instead.
- **Reward rules for agents**: *Calls the tool the dataset's answer calls first*, *Calls a tool it is not offered*, *Writes a tool's result itself*.
- **Hugging Face Dataset block**: a dataset on huggingface.co read where training runs – a part at a time, cut into tokens once (about two million tokens a second), never kept in the workspace. A column of `messages` is conversations to fine-tune on; a Text Dataset or another Hugging Face Dataset wired into its *With* input is read along with it.
- **Long runs go on where they stopped**: the trainer saves every 20 minutes and picks up from there when it runs again; the Colab notebook keeps the run and the corpora on Google Drive. Learned tokens it learns itself (from a corpus) come back into the model.
- **Training is faster and leaner**, measured on Magi's layers: local windows as the fused causal kernel on overlapping chunks (about 4× faster than the masked window, also for packed fine-tuning rows), cuDNN's attention when the card has it (about 2×), norms handing on half precision and rotary positions keeping nothing for the backward pass (a window-and-experts layer: 48.8 ms → about 13 ms, 68 → 43 KB a token), a fused softmax in the output head's loss (37% faster), batches split before activations are recomputed, reinforcement learning writing with a key/value cache (14× faster at 3,000-token prompts), long samples prepared 13× faster. A run that stops goes on with the tokens it learned (it learned them again, and a corpus cut into other tokens could have been used).
- **Notify when training ends** (Options, on by default): a sound and a desktop notification when a run of more than a minute finishes or fails. The training view says the time in minutes and hours.

- **Learned tokens.** A text model can cut text into pieces of words learned from its datasets (byte pairs) instead of bytes: about a third as many tokens, so it reads three times the text at once and trains on three times the text in the same time. New models use them; the Text block chooses *Learned* or *Bytes*. The pieces are learned when training starts from zero, in MacroMagi; the trainer reads them as MacroMagi does (checked token for token). **File format:** `text.tokenizer`.
- **The native engine runs DeepSeek's Mixture of Experts** (shared and routed experts, softmax, sigmoid or √Softplus gating, bias balancing, hash routing), so Magi-F1DS answers from its cache like the other Magi models instead of reading the whole conversation again for every token. Checked against PyTorch to within 0.000005.
- **Training survives running out of memory**: it recomputes activations, then splits each batch into parts, and runs the step again.
- **float16 on graphics cards without bfloat16** (Colab's T4, older cards), with a loss scale.
- **A Magi model is one `.szi` file**: the description, the weights, the native tools (and an ONNX model only when asked for) in one zip archive – upload it, download it, open it. Compile Package writes `<project>/<name>.szi`; a workspace exported as Python writes it too, and the Colab notebook downloads it directly. `szi/` folders from before still open. **File format:** `.szi` version 2.
- **The AI Agent window shows every tool call**: a row per call with the tool, its arguments in short and whether it worked, failed or did not run (and why); click it for the arguments and the result in full, with what the model thought before it. A call to a tool the model does not have comes back to it as an error it can read and answer from, instead of ending the answer empty.
- **The console gives times**: after each stage, how long it took and how long the run has taken so far.
- **New example datasets for a router model** (`make_agent_data.py`): `thinking.jsonl` (when to think, step-by-step working checked by code, and arithmetic given to the calculator whenever one is offered), `agent.jsonl` (tools in a row, errors and recovering, what no tool can do), `small_talk.jsonl` (short answers to small talk, so story problems no longer drown it) and `explain.jsonl` (longer answers laid out in steps and lists). Dataset messages can be marked `"learn": false`: read, not learned.
- **Base models: pretrain once, fine-tune in minutes.** *Save as base model…* on a stage block writes the weights kept after it as a `.szi` file; a Load Base Model block pointing at it fine-tunes without pretraining again, taking the base's learned tokens and checking its architecture.
- **The console estimates the time left** after a stage's first steps, and says when the whole run would take more than twenty minutes – better done on a GPU server (File › Export as Python) or from a base model.
- **Hugging Face Dataset block**: a dataset on huggingface.co (FineWeb-Edu by default) read where training runs – a part at a time, cut into tokens once (about two million tokens a second), never kept in the workspace. A column of `messages` is conversations to fine-tune on; a Text Dataset wired into its *With* input is read along with it.
- **Long runs go on where they stopped**: the trainer saves every 20 minutes and picks up from there when it runs again; the Colab notebook keeps the run and the corpus on Google Drive. Learned tokens it learns itself (from a corpus) come back into the model.
- **Edit in bulk** in the dataset editor: find and replace (text or regular expressions) in chosen parts of the samples, delete or keep the ones that match, delete long ones, remove repeats, trim, set effort or tools, take out thinking – on the selected samples, the file or every file, with a preview, and Undo.
- **Generate** in the dataset editor: another AI model – LM Studio, Ollama, llama.cpp, vLLM or any OpenAI-compatible endpoint – writes samples of the kind asked for, each checked before it is added; its thinking can be turned down to write faster.
- **Stage blocks say how much they read**: the tokens, and how many passes over their data – with a warning at too many or too few.

### Changed

- **MagiDB downloads again**: *Settings › Runtime* and the `db_search`/`db_read` tools read MagiDB from `MagiGun/MagiDB` (they looked for `MacroMagi/MagiDB`, which does not exist).
- **AI Agent window: Thinking shown opened** (Settings › Every chat): what the model thought is shown in full, not folded under *Thought*; kept between sessions.
- **File › Export as Python (experimental)**: the trainer tries compiled layers (`torch.compile`, Linux only) and keeps them only when they are faster with the same loss; if compiling fails later in the run (out of memory while compiling for a new batch size, on Colab), it goes on without instead of stopping. Routed experts run as grouped products when PyTorch has them (about 7% faster on a Magi layer); Muon's step is fused (30% faster, a third of the rounding error); multi-token prediction's attention is padded to take cuDNN (2× faster); a batch that leaves memory so full that every step waits for PyTorch to empty its cache is split in parts (the compiled run on Colab's G4: 8 rows at 92 of 96 GB, 3.89 s a step, slower than 4 rows plain); on Linux, memory grows in segments (`expandable_segments`); saving the run every 20 minutes no longer stops training while several GB are written to Google Drive.
- **Local windows are block-aligned**: a token sees its own block of *window* tokens and the block before (between one and two windows back), in training and in the native engine alike – the window the fast kernel can run.
- **Only Magi.** Presets › Commons offers Magi alone; the small, 12M, 30M and F1DS presets and the well-known layouts (Qwen3, Gemma 3, Mistral, GLM-4, DeepSeek-V3/V4, Llama) are gone. Their blocks are still in the block list.
- **Tool results read exactly as the tools print them**, keys in the tool's order – as the training data writes them (they were re-sorted alphabetically). A tool's arguments are listed in the order it declares them (`read_file(path, from?: integer, lines?: integer)`).
- **The example datasets are an agent's**: `math.jsonl`, `python.jsonl`, `basics.md`, `facts.md` and `reasoning.jsonl` are gone – arithmetic goes to the calculator, facts and exact code to MagiDB and the web.

- **The AI Agent window** is laid out again, as chat apps are: the chat list and the settings keep their width and nothing slides under them; requests are bubbles on the right as wide as their text, answers are text in a column of reading width with code blocks, inline code, bold, headings and lists; the message box is one rounded bar that grows as you write. A narrow window hides the chat list rather than squeezing the messages.
- **Compaction is learned with fine-tuning** (its samples mixed in like any file) instead of in a stage of its own after it, which made small models end every answer with `</compact>`. A Context Compaction stage mixes the other requests in too. The example recipe has no separate compaction stage; models note that they learned it (`trained.compacts`).
- **Training lists the model's own tools** in samples without `"tools"` – as the model sees them whenever it runs; `"tools": []` lists none.
- The example recipe pretrains on every file (requests and answers as plain text too) and fine-tunes for 600 steps.
- **Models learn much faster from zero.** A tied embedding table starts small (as GPT-2's), and Muon gets two learning rates tuned on the Magi presets: 0.004 for the layers and 0.02 for the embeddings and norms (*Learning rate (layers)* and *(embeddings)* on the Optimizer block); Optimizer blocks still on the old defaults take them when opened. After 300 steps of pretraining the 12M Magi's loss is 1.93 instead of 2.77, and the small one's 2.02 instead of 2.41. **File format:** the optimizer's `embed_lr`.
- **Less graphics memory**: the output head's loss is worked out a slice at a time, its gradients in the same pass – 7.5 GB instead of 9.7 GB for the 12M Magi at 16 rows of 2048 tokens, 3.7 GB instead of 6.0 for the small one, at the same speed.
- **Fine-tuning is about 40% faster**: requests are packed into rows only as long as each step's longest one needs, and into the first row they fit in – 0.35 s a step instead of 0.46 on the 12M Magi, with a little more learned in each; the next step's rows are packed while the graphics card works.
- **Less memory when idle**: training no longer writes an ONNX model of a text model the native engine runs (unless Compile Package asks for an ONNX file), an ONNX copy kept beside such a model's weights is dropped when it opens, and the AI Agent window keeps only what chatting needs of the Magic tab's model – and nothing once closed. A trained 21M-parameter model took about 450 MB with the AI Agent window opened once; the copies behind most of that are gone.
- **Tools are listed a line each** (`web_search(query): Searches the web…`) in new models' system text, instead of with their JSON parameters: 207 tokens instead of 377 for the example model's seven tools, in every sample it trains on – about half as many again per fine-tuning step. Models made before keep the form they learned. **File format:** `text.tool_list`.
- `conversations.jsonl` leaves out the conversations that name the assistant that wrote them, which taught small models to say they were Open Assistant.
- The trainer says what mixed precision means (the math in bfloat16 or float16, the weights in float32).

### Removed

- **GGUF, LM Studio, llama.cpp and Hugging Face exports**: Compile Package writes the `.magic` workspace, the `.szi` Magi model and, if asked, an ONNX file; GGUF models no longer import or run, and llama.cpp is no longer offered under Settings › Runtime and Hardware. Old files open; the settings they had for these are ignored.
- **New text model from a base** and **New large text model** in the File menu: New text model is Magi. A stage's *Save as base model…* and the Load Base Model block remain.

### Fixed

- Fine-tuning compaction samples on their own made a model start answers oddly and end them with `</compact>`.
- Small settings such as learning rates showed as 0.00 and dragged in steps of 0.02; they show their digits now (0.004) and drag in proportion.
- On Windows, training that needed more graphics memory than was free ran about ten times slower instead of saving memory: NVIDIA's driver lends system memory rather than failing. Training now keeps within the free memory, so it recomputes activations or splits batches instead.

## 0.1.7.1 – Unreleased

Cargo keeps the version 0.1.7 (a package version has three numbers); this is its first update.

### Added

- **Training from a stage.** Each training keeps the weights after every stage (in the project's `checkpoints/`). Train then offers: everything from the start, the rest from after any stage with kept weights, or one stage again on the trained model. Stage blocks have *Train from here* and *Only this, on the trained model*.
- **The AI Agent window** (the robot button, top right): chats with the Magic tab's model or a `.szi`, `.magic` or `.gguf` file, in a window of its own – chats kept and named by the model, per-chat settings (system prompt, thinking, effort, sampling, context, native tools, compaction), text files attached to messages, sent messages edited, deleted or answered again. A preview of CodeMagi.
- **Compaction in use**: a Context Compaction stage in the example recipe with its own data (`compaction.jsonl`), and the AI Agent window writes a long chat's earlier part short with it.
- **`knowing.jsonl`**: greeting back, keeping small talk short, saying when it does not know, searching instead of guessing, asking when a request is unclear.
- **Dataset editor**: undo and redo; *+ Tool call* lists the tools with their arguments (the ones not on for the model marked) and fills one field per argument; calls to tools the model does not have are marked; *Scan for issues* points out answers in a row; *Scan all files* in the Text Dataset block marks each file with issues.
- **Magi-V4 (experts)** in Presets › Commons: Magi with DeepSeek-V4's Mixture of Experts in every layer.
- **Colab**: File › Export as Python also writes `colab.ipynb`, which trains the model on Google Colab.

### Changed

- **Training is about five times faster** on a graphics card: bfloat16 with full-precision weights, fused norms and AdamW, Muon batched by shape. Fine-tuning packs requests into each row (a step learns from a few dozen), and datasets are mixed by the square root of their size. The 12M Magi: about 0.3 s a step instead of 1.5 s on an RTX 5070.
- **With thinking on, answers without reasoning are learned after an empty thinking span**, so a model no longer works out a greeting like a math problem.
- The Magi presets add DeepSeek-V4's multi-token prediction while training (the finished model and its GGUF are unchanged).
- The dataset editor's *Model's view* is gone: hovering over a sample's kind shows the training text.
- LM Studio's model.yaml adds a repeat penalty and an answer-length cap, so small models stop.

## 0.1.7 – Unreleased

### Added

- **A new dataset editor.** Samples show as the chat they teach – the request, the model's reasoning, answer and tool calls (name and arguments), tool results, the system prompt – and are edited right there. *Model's view* shows a sample exactly as training writes it for the model, chat tags highlighted; *Raw* shows the JSON lines. The editor pops out into a window of its own, and shows 25 samples a page.
- **Reasoning efforts per sample**, as chips: *Default* and each of the model's levels. A sample can have several (`"effort": ["standard", "superior"]`): it is one entry, learned at each level. Tick samples (Shift for a range) to set their efforts, duplicate, copy or remove them together.
- **Scan for issues** in the dataset editor: samples with problems, repeats, samples longer than the context window, efforts without samples, and words a fine-tuning or reinforcement file uses that the pretraining text never does.
- **Efforts without samples are covered**: training learns such a level from the nearest level's samples and says so; the Reasoning Effort block shows how many samples each level has and warns about empty ones.
- **Native Tools** block, apart from **Tools**: Tools are the ones the program running the model carries out (the AI Agent block, or another app); Native Tools are the ones a Magi model runs itself. Canvases from before get a Native Tools block when their Tools block held native tools.
- Model blocks in macros load **.magic, .szi, .gguf and .onnx** files.
- **Example data from open datasets**, about 9 MB: simple texts to pretrain on (Cosmopedia), human-written conversations (OpenAssistant), math with step-by-step reasoning at both efforts (GSM8K), Python (MBPP, self-oss-instruct) and tool calls (glaive). `open_data.py` rebuilds them; `SOURCES.md` credits them and gives their licences.
- **Magi (12M)** in Presets › Commons: the Magi recipe at about 12 million parameters, which learns more from that data.
- LM Studio shows the **tool use** badge on models with a tool tag: the chat template now lists tools and writes calls and results the way LM Studio recognizes, with the same prompt text as before. The Compile panel shows which badges LM Studio will show.

### Changed

- **Compile Package** (was *Compile .magic Package*) has a switch for each thing it saves: *.magic workspace* (with *With the datasets* and *With the pictures* under it), *Magi Model (.szi)*, *GGUF file*, *Add to LM Studio* (with its own precision, apart from the GGUF file), *Hugging Face folder*, *ONNX file*.
- **Use in Macro** adds Start → AI Load Model → AI Agent → Log → AI Unload Model.
- Magi-F1's reasoning efforts are *standard* (the default) and *superior*.
- The Reasoning Effort block explains with an example how samples are given a level.
- **Reasoning in datasets is learned only with the Thinking block** chained in and on; without it, it is left out (before, the model thought when its datasets did). The dataset editor marks such reasoning, and *Scan for issues* lists it.
- Window titles read “*Magi • MacroMagi”: a dot between the document and MacroMagi, and a star for unsaved changes, instead of a dot and a dash.

### Fixed

- LM Studio listed a model added from MacroMagi twice – once as the model, once as the bare GGUF file. The file is now marked as the model's own (LM Studio's list is backed up first).
- **When a macro stops, every AI model it loaded is unloaded** (llama.cpp's servers stopped), whether or not an AI Unload Model block ran.
- Some notes had runs of spaces in the middle of a sentence.

## 0.1.6 – Unreleased

### Added

- **The Magi architecture.** Magi models are small and do the hard work through **native tools**: each a folder (`nt/<tool>/` with `tool.json` and `run.py`) that MacroMagi runs on this computer, giving the result back to the model until it answers. MacroMagi comes with calculator, web_search, fetch_page, get_time, get_weather, read_file, list_files, describe_image, transcribe_audio, translate and wiki_search (the vision, audio and translation tools are starting points that need their small models installed). Tick them in the new **Tools** block.
- **The Magi package (`.szi`)**: a folder with the model's description, weights, ONNX model and native tools. Compile writes one folder per project – the `.magic` workspace (no model inside when packaged), `szi/`, and as chosen `gguf/`, `huggingface/` and `onnx/`. File › Open and Load Base Model take `.szi` files.
- **The native engine**: models of Transformer Layers trained here run straight from their weights with a key/value cache – each token is one step, and a request that starts like the last one only reads what is new. On the example it is about a thousand times faster than the ONNX path (0.8 ms a token), with the same scores to within 0.00006. Prompt Model and macros use it.
- **Long contexts.** The **Context Scaler** block trains at 2K–8K tokens and runs at up to 256K with YaRN (or NTK), stretching positions only past the trained context so short texts stay exactly as learned; llama.cpp gets the YaRN flags from MacroMagi when a longer context is asked for, and the exports can carry it always (off by default: it changes short texts). The **Memory & Cache** block keeps a running model's memory in a hot tier, a warm tier (F16 or 8-bit) and a cold tier on disk, can keep the system text's memory on disk under its SHA-256, and can read only the most relevant blocks; llama.cpp gets the cache type (F16, Q8_0, Q4_0) and flash attention. Prompt Model shows the tiers filling.
- **Magi-F1's recipe**: the example canvas pretrains on its own dataset of plain language and reasoning, fine-tunes on every kind of request, then runs a short reinforcement stage – with Muon, which is now every Optimizer block's default. Reasoning effort is *standard* or *superior*, apart from thinking on or off. New example data: simple facts, reasoning at both efforts.
- **Macro blocks AI Load Model and AI Unload Model**, and AI Agent settings for thinking, reasoning effort, context window and running native tools.
- **File › Export as Python** on the Magic tab: a folder with `run.py`, the plan, the datasets, the native tools, the trainer and a README, which trains the same model without MacroMagi and writes its `.szi`.
- Compiled models go into LM Studio by default when LM Studio is installed, so its Reasoning Effort choice shows (it needs the model.yaml).

### Changed

- **Text & Tools** became two blocks, **Text** and **Tools**. The warning about the system prompt shows only when it differs from the one the model was trained with.
- Reasoning effort levels start with *none* (the default).
- The Commons menu scrolls when the window is short; the Magic tab's start buttons are centred.
- Settings › Runtime shows *Uninstall* on installed parts; Settings › Engine calls MacroMagi's own Python *Embedded Python*, apart from a Python found on the computer.
- The dataset editor's filter is *With problems (n)* and explains what a problem is; empty requests and answers are problems.

### Fixed

- Installing llama.cpp for CUDA 12 or 13 failed with "the download did not contain llama-server": the CUDA runtime's zip was taken for the program.

## 0.1.5 – Unreleased

### Added

- **The Magic tab is a node canvas** for building, training, trying and packaging AI models. It has the Macro tab's panning, zooming, wires, groups and searchable block list, a panel for the selected block (its settings, and what is wired into it), and a console with the loss as a chart.
- **Every wire means something.** The *Model* wire runs from a start through the architecture and training stages to Compile .magic Package and Prompt Model / Test Model. Setup blocks (Model Info, Text & Tools, Screen Area, Answers…) chain into the start's *Setup* input, and the model is what they say. *Feed-forward* wires take a Mixture of Experts or Persona Router into layer blocks. *Optimizer* wires take an Optimizer into the stages it drives. Blocks that are not connected where they would be used say so on their faces.
- **The banner works for the Magic tab**:
  - **File**: New text model, New vision model, Open, Open recent, Import (GGUF, ONNX project, `.zip`/`.onnx`), Save, Save as, Close.
  - **Options**: *Optimize environment when training*.
  - **Presets**: *Commons*, saved Magic blocks, *Save selected blocks*.
  - **Add**, **View**, undo/redo, and **Train** / **Stop training**.
- **Presets › Commons**: Magi (MacroMagi's own layout for small agents), Qwen3, Qwen3 Thinking, Gemma 3, Mistral, GLM-4, DeepSeek-V4, DeepSeek-V3 and Llama, each scaled down, put in place of the canvas's architecture (with the setup blocks they need).
- **Thinking** and **Reasoning Effort** setup blocks: training learns answers with and without reasoning and at each effort level; Prompt Model and LM Studio get a Thinking switch and a Reasoning Effort choice.
- **Transformer Layers' finer options**: normalized queries and keys, GeGLU, norms before and after each part, local windows with global layers, partial rotary positions, attention biases; Model Size adds tied and scaled embeddings.
- **Architecture blocks**:
  - attention: Transformer Layers (the Llama layout: RMSNorm, rotary positions, grouped key/value heads, SwiGLU), Sliding Window Attention, Latent Attention (MLA), Compressed Sparse Attention (V4's CSA with its lightning indexer), Heavily Compressed Attention (V4's HCA), Mamba-2 / SSM;
  - feed-forward: Mixture of Experts (DeepSeekMoE with softmax, sigmoid or √softplus gating, bias or loss balancing, hash routing), Dual-Persona Router;
  - whole-model: Hyper-Connections (V4's mHC), Multi-Token Prediction, Encoder/Decoder Split, Vision Encoder.
- **Training on your computer** with PyTorch, on the CPU or one or several NVIDIA graphics cards:
  - stages: Pretraining with a context curriculum, Fine-Tuning (all weights, LoRA or DoRA, with "what not to do" samples), Reinforcement Learning with reward rules, Context Compaction, Vision Training with negative pictures;
  - an Optimizer block per stage: AdamW, Muon as DeepSeek-V4 trains with it, or SGD, with cosine or constant schedules, warm-up, clipping, accumulation, recomputed activations and offloaded optimizer state.
- **A new text model is a working example**: Magi, a 3.5M-parameter assistant with Thinking and Reasoning Effort, four general tools, and small original datasets – plain English, conversation, reasoning at three efforts, tool use with results, chat titles, answers to avoid. It trains in a few minutes on a recent graphics card.
- **Compile .magic Package** saves the model into the folder you choose when training ends: `.magic`, and for Llama, Qwen3 and Gemma 3 layouts a **GGUF** file (F16, Q8_0 or Q4_0, with its size and precision for LM Studio) and a **Hugging Face** folder (Llama, Qwen3). The chat template takes OpenAI-style messages, tools, `enable_thinking` and `reasoning_effort`; llama.cpp reads thinking, answer and tool calls apart. Its panel has *Save package now*, *Add to LM Studio* (the GGUF plus a `model.yaml` with Enable Thinking and Reasoning Effort), *Use in Macro* and *Open folder*.
- **Prompt Model** keeps the model loaded (*Load*, *Unload*), runs it with the built-in engine or llama.cpp, and sets the context window, thinking, effort, longest answer and temperature; it shows the answer, thinking and tool calls. **Test Model** watches the screen or tries a picture.
- **GGUF models**: import them (*File › Import*), prompt them, and run them in macros with llama.cpp, on the graphics cards you choose.
- **Text Dataset** block and **dataset editor**: `.jsonl`, `.md` and `.txt` files in the `.magic` file, searched, filtered, checked (JSON, sample shape, every tool call) and edited in place, with *Add an example* and *Add the examples*. **Vision Dataset** block: pictures per answer and pictures it was mistaken for, with thumbnails.
- **AI Agent** block: asks a text model (ONNX or GGUF) and gives its answer, thinking, first tool call and arguments; optionally carries out press, type, click, move, wait and log calls. *Use in Macro* adds a text model with Start → AI Agent → Log.
- **Settings › Runtime**: what trains and runs models, each with its version, *Install*, *Uninstall* and *Install again*. It covers PyTorch for NVIDIA graphics cards (CUDA 12.8) or for the CPU, onnx, llama.cpp (CUDA 13, CUDA 12, Vulkan, CPU, from its latest release), and ONNX Runtime for the CPU and for graphics cards. It shows which Python training uses.
- **Settings › Hardware**: the processor, memory and graphics cards. It chooses where training runs (Automatic, the CPU, or chosen cards), how GGUF models use cards (which build, how many layers, which cards), and where ONNX models run.
- **Options › Optimize environment when training** (on by default): while a model trains, the Magic tab shows only the training, and the background picture and see-through panels are off.
- If MacroMagi ever fails, it writes `crash.log` into the data folder and says so, instead of closing without a word.
- Settings › Runtime and Hardware are looked at when MacroMagi starts and whenever something that needs them fails; *Check again* looks now.

### Changed

- Opening a `.magic` file from an earlier version builds its canvas from it, with its setup blocks chained into a *Use As It Is* start.
- Settings › AI became **Runtime** and **Hardware**. PyTorch and onnx are listed separately, so an installed PyTorch shows as installed.
- AI Watch and AI Model report an error when given a text model.
- On the Magic tab, Ctrl+N, Ctrl+O, Ctrl+S, Ctrl+Shift+S, Ctrl+Z and Ctrl+Y work on the model.

### Fixed

- The dataset editor closed MacroMagi when a sample was added.
- Installing llama.cpp failed when its latest release had no Windows builds; the newest recent release that has them is used.
- Every prompt reloaded the model.
- llama.cpp servers MacroMagi started could outlive it after a crash; Windows now closes them with it.
- LM Studio showed no size or precision for exported GGUF files, and the example model named every chat "type".

### File format

`.magic` additions, all optional; earlier files load unchanged:
- in `magic.json`: `kind`, `text` (with `tool_tag`, `thinking` and `effort`), `architecture`, `studio` (the canvas), `datasets`, `trained` and `gguf` (a GGUF file the model runs);
- in the archive: `model/weights.safetensors` and `data/text/…`;
- `negative` on pictures, and `weights` and `datasets` under `include`.

New macro block `ai.agent`.

## 0.1.4

### Added

- **Files** blocks: **Read File** (all the text, one line, or a picture), **Write to File** (set the file, add to it, or change one line; a picture is saved as a PNG, replacing one of the same name), **Find in Text** (on which line a word is, what that line says, on how many lines), **Folder Contents** and **Make Folder**. Paths have a Browse button and can be wired in; relative paths go into the `files` folder in the data folder.
- **Screenshot** block: a picture of the display the pointer is on, a chosen display or an area, kept while the macro runs. Pictures from Screenshot or Read File can be wired into **Image Detection** (its new *Picture* input) or **Write to File**.
- **Switch to Display** block: moves the pointer to the middle of a display, and can click there.
- **Current Window** block: the title and program of the window in front, its display, the pointer's display and its middle.
- **Every drop-down can be searched**, the Stop block's *Continue from* included.
- **Auto Typer**: a list of characters to leave out, with Space, Line break and Tab buttons.
- **Recording**: *Skip task switches* (off by default) leaves out Alt+Tab, Win+Tab or a click on another display – the first, the last and those in between, each switchable. The recording editor can add *Move* and *Look* rows, and pick where a movement starts or ends on screen.
- **Recordings** can be deleted from a Play Recording block's drop-down; recordings no block plays are deleted when the macro is saved (*Automatically delete unassigned recordings when saving*, in Settings › Recording and Options › Behaviours; on by default).
- **File › Save as** can save a `.mp` preset – of the selected blocks, or the whole workspace. **File › Export presets** exports the workspace, the selection, or one of your presets.

### Changed

- A Smart recording gets the same conversion as the editor's *Smart* button when it ends, so it is as accurate as converting it by hand.
- The Presets menu lists only your presets, without the ones stored in the open file.
- The AI settings page looks for Python by itself, so PyTorch's status is right without visiting the Engine page; a PyTorch that is found offers *Uninstall* instead of *Install*.
- The program picker of Switch to Window stays open while you type in its search box, and lists programs with windows first.

### File format

All additions are optional; earlier files load unchanged. New blocks `vision.screenshot`, `window.display`, `window.current`, `files.read`, `files.write`, `files.find`, `files.list`, `files.make_folder`; Image Detection's `picture` input; the Auto Typer's `ignore`.

## 0.1.3

### Added

- **Auto Typer** block: types text character by character – its own text, a linked variable, or a value wired in – at a speed in characters per second with random variation, longer pauses after punctuation, optional typos corrected with Backspace, trimming, skipping the first characters, a limit, a choice of what line breaks do, and Enter at the end. Outputs how many characters it typed.
- **Text Scan** block: reads the text in an area of the screen with the text recognition built into Windows (nothing to download). Outputs the text, whether a given text is there, where it is (to click it), and the first number in it. Can enlarge small text and read a chosen language.
- **Nested groups**: grouping blocks that are already in a group puts the new group inside it, drawn inside its box. Clicking a group's title selects the group itself; right-click it for *Select everything in it* and *Move the selection into* it. *Take out of its group* moves a block one level out.
- **AI Watch** has an *Acted* output: the blocks under it run each time the model acts, with *Times acted* and *Last answer* up to date, and AI Watch then goes on watching. *Done* follows when its time is up (or a Stop leaves the loop).
- **Lightweight mode** (the feather at the top right on the Macro tab): only Run, Stop and the console, over the theme and background, without the editor.
- **Key pickers** can be searched and have *Press a key…*, which takes the next key pressed. In the Input block and for a model's key in the Magic tab they also offer mouse buttons, so a model can click instead of pressing a key.
- **Close buttons** on the console and the Variables panel.
- **About**: click the logo at the top left, or see *Settings › General › About*. MacroMagi is developed and created by [MagiGun](https://github.com/MagiGun) and [uvxszi](https://github.com/uvxszi).
- Download progress shows as a small bar.
- The bin next to an image (Image Detection) or a sound (Sound Recognition) also deletes it from the macro when no other block uses it. Files on your computer are never touched: loaded files are copied into the macro.

### Changed

- **Image Detection and Sound Recognition**: groups and items are one list. Each group shows its images or sounds, the ones in no group come last, and an item is dragged by its handle onto a group to put it in. The group drop-downs are gone.
- **Settings › Engine** no longer starts anything when opened. *Check* looks for Pythons and tries the one macros use; Python is also looked for once, the first time a macro runs, and what was found is remembered. The drop-down shows what *(auto)* stands for, with its version and file. Python is downloaded – and uninstalled – here.
- **Settings › AI** lists ONNX Runtime, its graphics card build and PyTorch, with *Check again* at the top and *Uninstall* instead of *Remove*. PyTorch goes into the Python macros use.
- On Windows, keys are sent as scan codes, as a keyboard sends them, so games that ignore other key input receive them.
- The Comment block's color has a color picker.
- Rows of buttons continue on the next line in a narrow window instead of being cut off.
- The interface size has the usual reset arrow instead of a *100%* button; the theme color swatches no longer touch the edge.
- "Colour" is spelled "color" everywhere.
- Releases are built for Windows only, for now.

### Fixed

- AI Watch's *Times acted* and *Last answer* reached other blocks only after it stopped watching – by default never.
- Installing PyTorch asked for Python to be downloaded even when macros were running on an installed one; a Python from the Microsoft Store was not found.
- Opening Settings › Engine or › AI flashed command windows.
- The "→" in some messages showed as a box.
- Links (the credits, "Get Python from python.org") did not open the browser.

### File format

All additions are optional; earlier files load unchanged. Groups may have a `parent` (a group inside another); in files where a block was in several groups, it stays in the smallest and the groups nest. New blocks `input.autotype` and `vision.read_text`. AI Watch has the flow output `acted`. A model's key in `.magic` files may be a mouse button (`"Mouse:Left"`). See [docs/FILE_FORMAT.md](docs/FILE_FORMAT.md).

### Known limitations

- Text Scan uses the text recognition built into Windows; it reads only the languages Windows has text recognition for (usually the display language), and does not work on Linux or macOS yet.
- Linux and macOS packages are not built for now; building from source still works.

## 0.1.2

### Added

- **Magic tab**, beside Auto Clicker and Macro, for small vision AI models that macros use.
  - **Import** an ONNX model, or a project folder or ZIP that contains one. The importer reads the project's files – it never runs them – for the class names, which classes call for action, the color normalisation, the screen area, the key and the adjustable settings, and lists what it found and what it guessed.
  - **Configure** the name and description; the screen area (a square in the middle of a display that scales with it, or a fixed area picked on screen); each answer's name, whether it acts, how sure the model must be and how long to wait; the key, how long it is held and the cooldown; adjustable settings that move several values at once; how pictures are prepared; and the pictures to keep for training and fine-tuning.
  - **Try it**: watch the screen live and see what the model sees and how sure it is of each answer, with or without pressing the key, or try it on a picture file.
  - **Save** `.magic` files: the model, its setup, ready-made blocks and its pictures, each optional.
  - **Use in Macro** adds the model to the open macro with blocks that run it. Using it again updates the model and the blocks that use it.
- **Models made for the DBD Auto Skill Check project** run as they do there: importing its repository or its packaged app (folder or ZIP) finds its ten answers, the five that act, the 51% confidence, the 5 ms frontier delay and its screen area, with sliders for the three settings it offers.
- **AI blocks**: **AI Watch** keeps looking and presses the model's key when it should (or only watches), for a time or until stopped, as often as allowed; **AI Model** looks once and gives the answer, how sure the model is, and whether to act. Each block can change the model's adjustable settings for itself, load a `.magic` file, or open its model in the Magic tab.
- **Settings → AI**: run models **automatically** (each way is measured on the model and the fastest kept; the CPU when a graphics card is not faster), on the **CPU**, or on a chosen **graphics card**.
- **Downloads** on the AI settings page (also *Options → Downloads…*): ONNX Runtime (faster on the CPU), ONNX Runtime for graphics cards (DirectX 12, Windows), Python and PyTorch, each with its size, Install and Remove. Copies already on the computer are detected.
- **Python**: the Windows package now comes with Python. *Settings → Engine* lists every Python found, with its version, in a drop-down, and offers *Download Python* when there is none.
- **Image Detection** looks for several images in one block, in order of priority. Images can be put in groups; an image or a group can have its own search area, otherwise the block's is used. *Found when* any or every image is there; **Detail** (Full, Half, A third, A quarter) trades accuracy for speed. New outputs: *Which image*, *Its group*, *How many*. The block shows up to four of its images, and Find (Ctrl+F) finds blocks by the names of their images and groups.
- **Sound Recognition** listens for several sounds in one block, in groups that can have their own accuracy; *Found when* any or every sound has been heard; *Recognise from the first* recognises long sounds while they are still playing. New outputs: *Which sound*, *Its group*, *How many*.
- **Microphone choice** for Sound Detection and Sound Recognition; the system's default when none is chosen.
- **Clear** button in the console.
- Variables can keep their value from one run to the next: *Reset at the start of each run* in the Variables panel. New variables keep their value and start empty; a kept value can be emptied from the panel.
- Blocks that need the Python engine show a warning sign when it is not available; it explains why and opens *Settings → Engine*. The native engine skips them, with a warning in the console, instead of stopping.
- A `.magic` file opened with MacroMagi, or named on the command line, opens in the Magic tab.

### Changed

- The screen picker's display bar can be dragged, opens under the pointer, and stays inside the display.
- *Strict hotkey modifiers* and *Stop on task switch* are under *Behaviours* in the Options menu.
- The Log block's message is empty by default, so it logs the value wired into it.
- The banner logo takes the theme color, like the buttons.
- Sound recognition is cheaper on long sounds: what has been heard is analysed once, not on every look.

### Fixed

- The Presets menu became very wide once it held a preset.

### File format

All additions are optional; files from 0.1.0 and 0.1.1 load unchanged. New `.magic` format for AI models. In `.magi` files: `model` assets; Image Detection has `images`, `groups`, `mode` and `detail` (the single `image` of older files is still read); Sound Recognition has `sounds`, `groups`, `mode` and `match_ms` (likewise for `sound`); sound blocks have `device`; variables have `reset` (`true` when missing). Presets may embed models. See [docs/FILE_FORMAT.md](docs/FILE_FORMAT.md).

### Known limitations

- Training and fine-tuning are not in the Magic tab yet. Pictures can already be collected into `.magic` files by class, and PyTorch can be downloaded for it.
- Graphics cards are used on Windows only; elsewhere models run on the CPU.
- Image Detection runs on the CPU, where its search is fastest.
- A model keeps the licence of the project it came from; the Magic tab shows the licence an import found. Models imported from DBD Auto Skill Check are GPL-3.0, so `.magic` files and macros that contain one carry those terms when shared.

## 0.1.1

### Added

- **Variables.** A macro has named variables with starting values. Any block setting can follow a variable through the link button beside it, so several blocks share one value and change together. New blocks: **Variable** (read one), **Set Variable** (set, add, subtract, multiply, divide, append – from the setting or from a wired value) and **Counter** (count passes, with a limit and a *Reached* output). A **Variables** panel lists them with their starting values and their values in the latest run. Python scripts read and change them through `magi.vars`, and a Python Script block can store its result in one.
- **Logic for values**: **Math** (add, subtract, multiply, divide, remainder, power, min, max, rounding, random…), **Compare** (equal, less, greater, contains, starts with, ends with) and **Text** (join, length, upper/lower case, trim, replace, read as a number).
- **Rate** block: a list of inputs done one after another, a set time apart. The time can follow a variable or be wired in from another block. **Variable Rate Editor** window for its timing and steps.
- **Definition** and **Function** blocks: build a piece of a macro once under a Definition and run it from any number of Function blocks, passing a value in through a variable.
- **Repeat until / while** a condition, checked before every round.
- **Stop** can now end only the current round of a loop, leave the loop, or continue from another chosen block.
- **Clipboard** block (the text on the clipboard) and **Notification** block (a sound and a desktop notification). *Options → Notify when a macro finishes* does the same whenever a macro ends by itself.
- **Sound Detection** (loud enough, optionally within a pitch range) and **Sound Recognition** (an uploaded WAV, MP3, OGG or FLAC sound, recognised with an adjustable accuracy at any volume), listening to what the computer plays or to the microphone.
- **Comments**: notes on the canvas that fold up and resize by their corner, in any color.
- **Groups**: put blocks in a named, colored box (Ctrl+G); drag the box to move them together, and turn the whole group off to skip every block in it.
- **Find** (Ctrl+F): search blocks by name, custom name, type or settings, and groups by name.
- **Several displays**: every click position and search area remembers the display it is on, chosen in its settings; Image and Color Detection search a chosen display; the screen picker and Screen targets open on the display MacroMagi is on and switch between displays; recordings note when the pointer moves to another display.
- **Recording Editor** window: drag rows by their grip to reorder them, **Cut** (remove the part between the handles and close the gap) beside **Trim**, and rows for key down/up, button down/up and display changes. Recordings use *down* and *up* rows only when other inputs happen while a key or button is held; otherwise one row with its hold time.
- **Recording options** in a menu beside Record: what to record and how to record movement (Smart Recording by default).
- **Reset buttons** beside every block setting and every Auto Clicker value that differs from its default.
- **View** menu: show or hide the grid, the block activity glow and the variables.
- The block list shows every block. When a wire is dropped on the canvas, the blocks it can connect to come first and the others after them.
- A fourth example, *Variables, functions and groups*.
- A new icon, used for the window, the executable and desktop shortcuts.

### Changed

- **Auto Clicker**: the units after "per" read second, minute, hour, day; Variation sits under High-speed mode; the speed-limit note says to turn on High-speed mode to go faster; Two keys alternate by default, with *Alternating* first; Key and Two keys hide the settings that only matter for the mouse.
- A Play Recording block's information card shows what the recording holds and a **Recording Editor** button; the block's settings show the timeline, handles, Trim, Cut, speed and Smart, without the rows.
- **Screen targets** no longer darkens the screen, redraws without flickering, and a double-click on a target or its line in the list shows its block in the editor.
- The Options menu is narrower; the console, the block settings panel, the settings window and the program rules window keep the size you give them.
- Number settings with a wide range (such as loudness or pitch) change at a sensible pace when dragged.

### Fixed

- Hotkeys could not be assigned, and did not fire, while MacroMagi's own window was in front. The window library's Raw Input keyboard registration made Windows skip MacroMagi's keyboard hook; MacroMagi now takes Raw Input over itself.
- Panels and windows grew wider by themselves.
- The screen picker and Screen targets could crash on very large displays.

### File format

All additions are optional; files from 0.1.0 load unchanged. Projects may have `variables` and graph `groups`; any setting may be a variable binding (`{"$var": "name"}`); positions and areas may have a `display`; recordings may contain `key_down`, `key_up`, `mouse_down`, `mouse_up` and `display` strokes; assets may be `audio` (WAV). See [docs/FILE_FORMAT.md](docs/FILE_FORMAT.md).

### Known limitations

- Listening to what the computer plays needs a loopback or monitor input on Linux and macOS (Windows has one built in).
- Switch to Window, program rules and smart recording are still Windows-only; gamepad input still cannot be sent.

## 0.1.0

The first version. Everything is new.

### Added

- **Auto Clicker** tab: mouse button, key or two keys (together or alternating); single/double/triple clicks with an adjustable gap; speed as clicks per second/minute/hour/day or as an interval from milliseconds to 365 days; speed variation; fixed hold time or duty cycle; course correction (on by default); high-speed mode up to 1000 clicks per second, with a warning and automatic hold shortening; click limit, time limit, and corner and edge stops; click at the pointer or a picked position; toggle or hold-to-run hotkey; separate Start and Stop buttons; undo/redo; presets; "Open as macro".
- **Macro** tab: a node graph editor with pan and zoom, box and multi-selection, copy/paste (carrying images and recordings), duplicate, disable, undo/redo, a searchable, keyboard-driven and resizable block list, a settings panel generated from block definitions, a console, and three built-in examples.
- **Banner menus** per tab: File (both tabs), Options, Presets, Open as macro or Add and View, undo/redo; buttons for settings, keep-on-top and hide-from-screen-captures.
- **Blocks**: Start, Wait, Repeat, Decision, Wait Until, Stop, Log, Input, Auto Click, Play Recording, Image Detection, Color Detection, Switch to Window, Logic, Random Chance, Python Script.
- **Switch to Window**: bring a program to the front by title or program name, aware of windowed, borderless and exclusive fullscreen, with a "Not found" branch.
- **View → Screen targets**: every click position and search area shown over a frozen screenshot, numbered in run order, draggable and resizable.
- **Recording** of keyboard, mouse buttons, mouse movement, scroll wheel and gamepad input, stored as press time plus hold duration per input. **Smart recording** detects games that lock the pointer and records relative motion; Play Recording can replay movement as recorded, as screen positions or as relative motion; older recordings convert from the block's context menu.
- **Recording editor**: a spreadsheet of every input with every field editable for every kind of input, changeable row types, add/duplicate/delete, a timeline with trim handles, speed changes, and a pop-out window.
- **Image Detection** by color-aware template matching with a coarse-to-fine search, an adjustable accuracy threshold (default 80%), an optional search area, and a screen picker for capturing images, areas, points and colors.
- **Two engines** with identical behaviour: Python (standard library only, auto-detected, with Python Script blocks and a `magi` scripting API) and native Rust.
- **Hotkeys** from any keyboard key (Escape and function keys included), mouse button or gamepad button, alone or combined; Auto Clicker, Run, Record, stop-everything and any number of extra stop hotkeys; all listed in Settings with conflict warnings and a reset button. Run and Record show their hotkeys and ask before starting when theirs has been removed.
- **Options** menu: strict hotkey modifiers, stop on task switch (Alt+Tab, Win+Tab), auto-stop hotkeys, program rules (whitelist or blacklist with search and per-program stop or pause), stopping the Auto Clicker while a hotkey is edited, dry run, and creating a desktop shortcut.
- **Appearance**: monochrome by default; a theme color that recolors the whole interface; Light, Dark or System theme (System by default); a background picture or animated GIF/WebP from a file or web address, with opacity, blur, darkness and panel opacity.
- **Settings window**: movable and resizable, in pages (General, Appearance, Hotkeys, Recording, Engine), with "Reset all hotkeys" and "Reset all settings".
- **Export** dialog: choose to include presets, Options and Auto Clicker settings in a `.magi` file; files that carry them offer to apply them on opening.
- **Safety**: keys and buttons held by a macro are released when it stops; dry-run mode; on Windows, input sent by MacroMagi never triggers its own hotkeys or ends up in its own recordings.
- **Memory**: an idle Python process is stopped after a few minutes, and memory is handed back to the system while the window is minimised.
- **Files**: the `.magi` project format (a ZIP with a readable `project.json` and linked assets) and the `.mp` preset format (JSON), both at format version 1. *Format:* projects may carry `options` and `clicker`; recordings may contain `look` strokes and clicks with `here`; Auto Clicker settings gained the fields listed in FILE_FORMAT.md. All additions are optional; earlier files load unchanged.
- **Command line**: `run` (with `--engine` and `--dry-run`), `nodes`, `example`.
- **Releases**: a GitHub Actions workflow builds Windows, Linux and macOS packages and publishes a release only when every build succeeds.
- **Terms**: MacroMagi is source available under [TERMS.md](TERMS.md). [AGENTS.md](AGENTS.md) guides AI coding agents and logs their revisions.

### Fixed

- Hotkeys that never took effect, or only after a restart: Windows silently removes slow keyboard hooks, which left only mouse hotkeys working. The listener now runs at high priority and a watchdog reinstalls the hooks. Any key and combination can be assigned and works immediately.
- The *Add* button and the *Add block* context-menu entry opened the block list and closed it again at once.
- The first entry of the block list was drawn as tall as the whole list.
- Typing a percentage into *Interface size* set the wrong size; the size now changes only after the slider is released.
- *Trim start* in the recording editor appeared to do nothing (recordings already start at the first input); trimming is now done with handles on the timeline.
- Clicks landing on the Auto Clicker's start button no longer stop it.

### Known limitations

- Linux and macOS builds are implemented but untested. Switch to Window, program rules and smart recording are Windows-only for now.
- Gamepad input can trigger macros and be recorded, but cannot be sent.
- Whole-screen searches, the screen picker and the screen-targets view use the primary monitor only (lifted in 0.1.1).
