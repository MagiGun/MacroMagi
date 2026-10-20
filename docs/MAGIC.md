# The Magic tab

The Magic tab builds, trains, tries and packages AI models for macros. It uses a canvas like the Macro tab's.

- **Text models** read a request and write an answer. They can think first, at a chosen effort, and call the tools they are offered. The **AI Agent** block runs them in a macro and can carry out their press, type and click calls.
- **Vision models** look at part of the screen and pick an answer. The **AI Watch** and **AI Model** blocks run them.

A model is one `.magic` file: the trained model, its weights, its datasets, its setup and the canvas. A text model built from Transformer Layers in the Llama, Qwen3 or Gemma 3 layout can also be saved as a **GGUF** file for LM Studio and llama.cpp (with LM Studio's Enable Thinking and Reasoning Effort controls), and as a **Hugging Face** folder. GGUF models from elsewhere can be imported and used. The file format is in [FILE_FORMAT.md](FILE_FORMAT.md#magic--ai-models).

## Contents

- [Quick start](#quick-start)
- [The canvas](#the-canvas)
- [Blocks](#blocks)
- [Common architectures](#common-architectures)
- [Datasets](#datasets)
- [Training](#training)
- [Packages: .magic, GGUF, Hugging Face](#packages-magic-gguf-hugging-face)
- [Running models](#running-models)
- [Settings › Runtime and Hardware](#settings--runtime-and-hardware)
- [How text is read and written](#how-text-is-read-and-written)
- [What the architecture blocks are](#what-the-architecture-blocks-are)
- [Limits](#limits)
- [For developers](#for-developers)

## Quick start

1. **Settings › Runtime**: install **PyTorch** (the NVIDIA build if you have an NVIDIA graphics card) and **onnx**.
2. **File › New text model** (on the Magic tab). It opens as a working example: **Magi**, a 3.5-million-parameter assistant in the Magi layout, with Thinking and Reasoning Effort set up, four general tools (web search, a calculator, weather and time) and six small example datasets – see [The example datasets](#the-example-datasets).
3. Select **Compile .magic Package** and choose a folder (**Save into → Browse**).
4. Press **Train** in the banner. On an RTX 5070 the example trains in about five minutes; on a processor it takes much longer. With *Options › Optimize environment when training* (on by default), the tab shows only the training while it runs.
5. When training ends, the model is saved into the folder by itself: `Magi.magic` and `Magi-F16.gguf`.
6. Select **Prompt Model**, write *What is 487 times 26?*, and press **Send**. The answer shows what the model thought, the tool it called and everything it wrote. **Load** keeps the model in memory, so it is not loaded again for every request. With llama.cpp installed (*Settings › Runtime*), Prompt Model runs the compiled GGUF, which answers in well under a second; the built-in engine reads the whole 2048-token context again for every token it writes and can take a minute.
7. **Use in Macro** (in Compile .magic Package) adds the model to the open macro with Start → AI Agent → Log blocks.

**Add to LM Studio** (in Compile .magic Package) puts the GGUF file into LM Studio with its controls: see [Using a model in LM Studio](#using-a-model-in-lm-studio).

## The canvas

Five kinds of wire connect the blocks. Each block's panel (select the block) lists what is wired into it under **Connections**, and what it would use if nothing were.

| Wire | From → to | What it carries |
| --- | --- | --- |
| **Model** | start → architecture blocks → training stages → Compile .magic Package → Prompt Model / Test Model | The model, left to right. Architecture blocks come before the stages. |
| **Setup** | setup blocks → … → the start's *Setup* input | The model's own settings. Setup blocks chain into each other; the model is what the blocks on the chain say. A setup block that is not on the chain is marked *Not used*. |
| **Feed-forward** | Mixture of Experts or Dual-Persona Router → a layer block's *Feed-forward* input | What replaces that layer block's plain feed-forward layer. One block can feed several layer blocks. |
| **Data** | Text Dataset / Vision Dataset → a training stage's data inputs | What the stage learns from. |
| **Optimizer** | Optimizer → a training stage's *Optimizer* input | How that stage learns. Without one, a stage uses AdamW with the defaults. One Optimizer block can feed every stage. |

Blocks show warnings on their faces, such as *Wire a Text Dataset into it* or *Choose a folder to save into*. The **Train** button says in its tooltip what is still missing. With nothing selected, the panel shows the pipeline from start to end and a short *How it works*.

The banner holds the tab's menus, as on the Macro tab:

| Menu | On the Magic tab |
| --- | --- |
| **File** | New text model (Ctrl+N), New vision model, Open (Ctrl+O), Open recent, Import (a GGUF model, an ONNX project folder, a `.zip` or `.onnx`), Save (Ctrl+S), Save as, Close the model. |
| **Options** | The usual options, plus *Optimize environment when training*. |
| **Presets** | **Commons** (well-known architectures), your saved Magic blocks, and *Save selected blocks*. |
| **Add** | The block list (right-clicking the canvas opens it too). |
| **View** | Console, Block settings, Grid, Find, Fit to view, Actual size. |
| Undo / Redo | Every change to the canvas and the pictures (Ctrl+Z, Ctrl+Y). |
| **Train** / **Stop training** | Runs the pipeline. |

## Blocks

### Start

| Block | What it does |
| --- | --- |
| **Start from Zero** | A brand-new model with random weights, text or vision. |
| **Load Base Model** | Starts from a trained model's weights, to train it further: this model's own (empty path) or another `.magic` file trained in MacroMagi. Keep the base model's architecture blocks. |
| **Use As It Is** | The model as it is, without training: an imported GGUF or ONNX model, or one trained before. Change its setup, try it, package it. |

All three take the setup chain in their *Setup* input.

### Setup

| Block | For | Settings |
| --- | --- | --- |
| **Model Info** | all | Name, what it does, author, licence. Also shows the model's size and how it was trained. |
| **Text & Tools** | text | The system prompt; temperature, how many likely tokens to choose among, the longest answer; the tools it may call (*Add general tools* adds web_search, calculator, get_weather and get_time; *Add MacroMagi's tools* adds press, type, click, move, wait and log); its personas; its chat tags. |
| **Thinking** | text | The model can be asked to think or not; *Think unless asked not to* sets what it does by default. Chained in, training learns each answer with reasoning both ways, and the Prompt Model block and LM Studio get a Thinking switch. |
| **Reasoning Effort** | text | Levels of how hard the model thinks (low, medium, high by default) and the default one. The level is named in the system text; datasets mark samples with `"effort"`. The Prompt Model block and LM Studio get a Reasoning Effort choice. |
| **Screen Area** | vision | A square in the middle of a display (it scales with the display) or a fixed area. |
| **Answers** | vision | Each answer's name, whether it acts, how sure the model must be, and how long to wait. Training learns these answers. |
| **When It Acts** | vision | The key or mouse button, how long it is held, and how long answers are then ignored. |
| **Sliders** | vision | Sliders that move several values at once, offered again in every AI block that uses the model. |
| **Picture Preparation** | vision | Picture size, layout, color order and normalisation. Training and importing set it. |

### Architecture

| Block | What it adds |
| --- | --- |
| **Model Size** | *Width* (numbers per token), *Attention heads*, *Key/value heads* (fewer: smaller and faster; equal to the heads: plain attention), **Context window** (the longest text the model reads at once, in tokens – here a token is a byte of text or a tag; for vision, the side of the pictures in pixels), *Positions* (rotary or learned), *Tie input and output* (the output head reuses the embedding table) and *Scale embeddings* (by √width, as Gemma does). |
| **Transformer Layers** | Standard layers: attention over earlier tokens, then a gated feed-forward layer. *Finer options* shows what sets the open model families apart: *Normalize queries and keys* (Qwen3, Gemma 3), *Feed-forward gate* (SwiGLU or GeGLU), *Normalization* (before each part, or before and after: Gemma, GLM), *Local window* with *Every Nth layer sees all* (Gemma 3), *Rotary share* (GLM: 0.5) and *Biases on queries, keys, values* (GLM, Qwen2). Models of these export to GGUF when they match a layout llama.cpp knows: Llama, Qwen3 or Gemma 3. |
| **Sliding Window Attention** | Layers that attend only to the last *Window* tokens. |
| **Latent Attention (MLA)** | DeepSeek-V2/V3's multi-head latent attention: keys and values rebuilt from a small latent per token, with a separate rotary part. |
| **Compressed Sparse Attention** | DeepSeek-V4's CSA: every *Pool every* tokens become one entry; a lightning indexer picks *Entries per token* of them for each token, beside a window of recent tokens. |
| **Heavily Compressed Attention** | DeepSeek-V4's HCA: larger pooled entries, all attended to, beside a window of recent tokens. |
| **Mamba-2 / SSM** | State-space layers: a small memory that fades over time, instead of attention. |
| **Mixture of Experts** | DeepSeekMoE, for the layer blocks it is wired into. Shared experts see every token; each token also picks *Experts per token* of the routed experts. *Choosing experts*: softmax (V2), sigmoid (V3) or √softplus (V4). *Balancing*: a bias with no extra loss (V3/V4) or a balance loss. *Fixed experts per token*: hash routing, as V4's first layers use. |
| **Dual-Persona Router** | Two personas of the reasoning, each inside its own tag with its own expert: by default Gun (strict code and tool calls) and Magi (critique, breaking loops). Wired into a layer block. |
| **Hyper-Connections (mHC)** | DeepSeek-V4's manifold-constrained hyper-connections: several parallel residual streams, mixed by doubly stochastic matrices. Applies to the whole model. |
| **Multi-Token Prediction** | While training, an extra layer also predicts the token after next. It is left out of the finished model. |
| **Encoder/Decoder Split** | The request is read in both directions; only the answer is written token by token. |
| **Vision Encoder** | Vision models: convolution layers, then the answers. |

Every layer block has *Layers* (how many) and *Feed-forward width* (used when no Mixture of Experts or Persona Router is wired in).

### Data

| Block | What it holds |
| --- | --- |
| **Text Dataset** | Text files kept in the `.magic` file, in two lists: *What to do* and *What not to do*. Add files, start a new `.jsonl` or `.md`, add the examples, or open a file in the dataset editor. |
| **Vision Dataset** | Pictures of each answer, and pictures an answer was mistaken for, with thumbnails. *Add…* for each; right-click a picture to remove it. |

### Training

| Block | What it does |
| --- | --- |
| **Pretraining** | Teaches a new model the language: next-token prediction on all the text in its dataset. *Start the context at* grows the context from short to full (a curriculum). |
| **Fine-Tuning** | Teaches the model to answer: requests and answers from *What to do* become likelier, answers from *What not to do* less likely. All weights, or LoRA or DoRA adapters folded back in afterwards. |
| **Reinforcement Learning** | Several answers per request are scored by reward rules, and the better-than-average ones become likelier (group-relative policy gradient). |
| **Context Compaction** | Teaches the model to rewrite a long context into a short one, from `{"context", "compact"}` samples. |
| **Vision Training** | Teaches a vision model its answers from a Vision Dataset. |
| **Optimizer** | For the stages it is wired into: AdamW, Muon (with AdamW for embeddings and norms, as DeepSeek-V4 does) or SGD; learning rate; weight decay; cosine or constant schedule with warm-up; gradient clipping; accumulation; recomputing activations and keeping optimizer state in system memory to save graphics memory. |

**Reward rules.** Each answer scores the sum of the rules it meets. The defaults: +1 for the dataset's answer, +1 for well-formed tool calls, +5 for at most 24 tokens, −3 for a loop (4 tokens repeated 3 times), −1 for a broken tool call.

### Package and trying

| Block | What it does |
| --- | --- |
| **Compile .magic Package** | Where and how the finished model is saved: *Save into* (a folder; training needs one), *File name*, what the `.magic` file includes, whether to also write a `.gguf` (with its *GGUF precision*: F16, Q8_0 or Q4_0) and a Hugging Face folder, and *Also add it to LM Studio*. *Save package now*, *Add to LM Studio*, *Use in Macro* and *Open folder* are in its panel. Training saves the package by itself when it ends. |
| **Prompt Model** | Text models. **Model**: *Run with* the built-in engine (ONNX) or llama.cpp (the compiled GGUF), and *Load*/*Unload* – a loaded model stays in memory, so requests answer at once. **Answering**: the *Context window* (GGUF: adjustable, up to what the model learned; ONNX: the one from Model Size), *Thinking* and *Reasoning effort* (when those blocks are chained in), and under *More* the longest answer and the temperature. Then write a request, *Send* (Ctrl+Enter), and read the answer, what it thought, its tool calls and everything it wrote. The settings are kept in the block. |
| **Test Model** | Vision models: watch the screen live, or try a picture file. |

## Common architectures

**Presets › Commons** puts a well-known architecture in place of the canvas's architecture blocks. The blocks after it move to make room, and the stages stay wired; setup blocks it needs (Thinking, Reasoning Effort) join the setup chain. They are scaled down to train on a desktop computer; raise *Width*, *Layers* and the experts for bigger models.

| Architecture | Blocks | Exports |
| --- | --- | --- |
| **Magi (small)** | Model Size (192 wide, 6 heads, 2 key/value heads, 2048 tokens, tied) · 8 Transformer Layers with queries and keys normalized · Thinking · Reasoning Effort | GGUF (as qwen3), Hugging Face. MacroMagi's own recipe for small agents; the example model uses it. |
| **Qwen3 (small)** | Model Size (256 wide, 8 heads, 4 key/value heads, tied) · 6 Transformer Layers with QK-norm and SwiGLU | GGUF (qwen3), Hugging Face (Qwen3ForCausalLM). |
| **Qwen3 Thinking (small)** | The same, with Thinking and Reasoning Effort | GGUF with LM Studio's Enable Thinking toggle and Reasoning Effort choice. |
| **Gemma 3 (small)** | Model Size (scaled and tied embeddings) · 6 Transformer Layers with QK-norm, GeGLU, norms before and after each part, five local layers (256-token window) to one global | GGUF (gemma3). |
| **Mistral (small)** | The Llama layout with 4 query heads per key/value head and a large rotary base | GGUF (llama), Hugging Face. |
| **GLM-4 (small)** | Biases on queries, keys and values, rotary positions on half of each head, norms before and after each part | Runs in MacroMagi. |
| **DeepSeek-V4 (small)** | Model Size (192 wide, 512 tokens) · Hyper-Connections (4 streams) · Multi-Token Prediction · Sliding Window Attention, then Compressed Sparse and Heavily Compressed Attention in turn · Mixture of Experts with √softplus gating and bias balancing, hash-routed in the first layer | Train it with the Optimizer on Muon. Runs in MacroMagi; not a GGUF architecture. |
| **DeepSeek-V3 (small)** | Model Size · Multi-Token Prediction · a dense Latent Attention layer, then Latent Attention layers with a sigmoid Mixture of Experts | Runs in MacroMagi; not a GGUF architecture. |
| **Llama (small)** | Model Size (256 wide, 8 heads, 4 key/value heads) · 6 Transformer Layers | GGUF (llama), Hugging Face. |

Qwen3, Gemma 3, GLM-4 and Mistral publish their model weights openly; these entries reproduce their layouts (the shape of the network), not their weights or training.

### Making your own architecture

An architecture is just the blocks between the start and the first training stage. To make one – your own "Magi" – start from a Commons entry or from Model Size and Transformer Layers, change the numbers and the finer options, or mix in other layer blocks, Mixture of Experts and Hyper-Connections. Then select those blocks and use **Presets › Save selected blocks**: it appears under Presets, ready for other models. The Compile block's panel says which GGUF architecture a model exports as; architectures llama.cpp does not know still train, run and package in MacroMagi.

The Magi layout itself is a set of choices for small models: more layers than its width would suggest (depth helps small models reason), grouped key/value heads (a smaller cache), normalized queries and keys (steadier training at higher learning rates), a tied output head (fewer weights), and a long context, because MacroMagi's tokens are bytes and a few tool descriptions take hundreds of them.

The DeepSeek-V4 blocks follow the DeepSeek-V4 technical report ("Towards Highly Efficient Million-Token Context Intelligence", 2026). DeepSeek-V4-Flash, for example, has 43 layers of width 4096, sliding-window attention in its first two layers, then CSA (pooling every 4 tokens, top 512) and HCA (pooling every 128) interleaved. It has 256 routed experts and 1 shared expert, 6 per token, hash routing in its first three MoE layers, 4 hyper-connection streams, and is trained with Muon. The Commons version keeps the design and shrinks the numbers.

## Datasets

Text datasets are `.jsonl` files (one JSON object per line), or `.md` and `.txt` files (plain text, split at blank lines, for pretraining):

```json
{"prompt": "What is 15% of 80?", "thinking": "10% is 8 and 5% is 4: 12.", "response": "12.", "effort": "low"}
{"tools": [{"name": "get_weather", "description": "…", "parameters": {…}}], "messages": [{"role": "user", "content": "Is it warm in Madrid?"}, {"role": "assistant", "content": "", "tool_calls": [{"name": "get_weather", "arguments": {"location": "Madrid"}}]}, {"role": "tool", "content": "{\"temperature_c\": 27}"}, {"role": "assistant", "content": "Yes – 27 °C."}]}
{"prompt": "What is 5 plus 7?", "rejected": "5 plus 7 is 5 plus 7 is"}
{"context": "a long conversation …", "compact": "the short version"}
{"text": "plain text to pretrain on"}
```

- `prompt`/`response` and `messages` teach answers. In a conversation every assistant message is learned, with what came before it as the request.
- `thinking` is the reasoning before an answer (on a sample or an assistant message; `<think>…</think>` at the start of an answer works too). With the Thinking block, each answer with reasoning is learned both with it and, after an empty thinking span, without it.
- `effort` names the reasoning effort the sample is written at; samples without one are at the default level.
- `tools` lists the tools the sample offers (MacroMagi's `{"name", "description", "parameters"}` or OpenAI's `{"type": "function", "function": {…}}`). Without it, no tools are listed. Assistant messages call them with `tool_calls`; `tool` messages carry the results.
- `system` (or a first system message) replaces the model's own system prompt (Text & Tools).
- `prompt`/`rejected` (or `bad`) is an answer not to give; it may be broken on purpose. `context`/`compact` trains compaction.

### The example datasets

A new text model comes with six small files, written for MacroMagi (no text is copied from elsewhere) by [`make_examples.py`](../crates/magi-core/src/nodes/examples/make_examples.py):

| File | What it teaches |
| --- | --- |
| `language.md` | Plain, general English to pretrain on: everyday things, numbers, time, how questions and good answers work. |
| `chat.jsonl` | Greetings, short definitions, rewriting and fixing sentences, and a few multi-turn conversations. |
| `reasoning.jsonl` | Small problems – arithmetic, comparisons, logic, counting – each with its thinking at three efforts: a line at *low*, a few steps at *medium*, steps and a check at *high*. |
| `tools.jsonl` | Using general tools: web search and reading a page, a calculator, weather, time, notes and files; two calls at once; a failing tool; no fitting tool; asking where the user is; answering straight away when no tool is needed. |
| `titles.jsonl` | Naming a conversation in two to five words when a chat program asks, as LM Studio does (as JSON `{"title": …}`, in `<title>` tags, or plain). |
| `avoid.jsonl` | Answers not to give: made-up tool results, tools that are not offered, broken calls, `type` as a title, repeating. |

MacroMagi's macro tools (press, type, click…) are not in them; *Add MacroMagi's tools* in Text & Tools and datasets of your own teach those.

The **dataset editor** opens a file in its own window. You can search, show only the samples with problems, and add, duplicate, edit and delete samples. *Add an example* inserts one sample of each kind. Each sample is checked as you type:

- it must be valid JSON, of one of the shapes above;
- a conversation needs an assistant message, and roles must be system, user, assistant or tool;
- every tool call must be closed, be JSON with a `name`, and have an object as its `arguments`;
- a call must name one of the sample's `tools`, or, without them, of the model's tools;
- an `effort` must be one of the model's levels.

*Keep changes* puts the samples into the model; saving the model writes them into the `.magic` file.

## Training

Training runs in Python with **PyTorch** and **onnx**, installed under *Settings › Runtime* into the *Python for AI*, which is the Python macros use. It runs on the graphics cards chosen under *Settings › Hardware*. Several NVIDIA cards share each batch.

When you press **Train**:

1. MacroMagi checks the pipeline (the Train button's tooltip says what is missing) and that PyTorch is installed.
2. It writes a plan, the datasets, the base weights and the prepared pictures into `data/training`.
3. It runs the trainer. Before training, the trainer checks onnx is installed. The console shows each stage, its progress, the loss (as a chart), the mean reward and any warnings.
4. When training ends, the trained model goes into the open document, and the package is saved into Compile's folder. The console lists the files.

**Stop training** ends the Python process; the model stays as it was. To train a model further, start from **Load Base Model** with the same architecture blocks.

## Packages: .magic, GGUF, Hugging Face

**Compile .magic Package** writes, into its folder:

| File | When | For |
| --- | --- | --- |
| `<name>.magic` | always | MacroMagi: the model (ONNX), its weights (safetensors), datasets, pictures, setup and canvas, as chosen. |
| `<name>-F16.gguf` (or `-Q8_0`, `-Q4_0`) | *Also a .gguf*, for Transformer Layers in the Llama, Qwen3 or Gemma 3 layout | LM Studio, llama.cpp, Ollama, and MacroMagi itself. The architecture llama.cpp knows (llama, qwen3 or gemma3); matrices as *GGUF precision* says, norms in 32-bit floats; the name, size (`general.size_label`, such as 3.5M) and precision LM Studio lists; MacroMagi's tokens and chat template inside. |
| `<name>-huggingface/` | *Also a Hugging Face folder*, Llama and Qwen3 layouts | `transformers` and the Hugging Face Hub: `config.json` (LlamaForCausalLM or Qwen3ForCausalLM), `model.safetensors`, `tokenizer.json`, `tokenizer_config.json` with the chat template, `generation_config.json`, `chat_template.jinja` and a model card `README.md`. |

They give the same answers. When this was written, the example model was run in llama.cpp (build 11748), in `transformers` 5.3 and in MacroMagi's ONNX runtime: the chat template wrote the same prompt in all three, `transformers` matched ONNX to within 0.00004, and llama.cpp chose the same tokens. llama.cpp's server reads the thinking, the answer and the tool calls apart (`reasoning_content`, `content`, `tool_calls`).

### Using a model in LM Studio

**Add to LM Studio** in the Compile block (or *Also add it to LM Studio*, which does it at every compile) copies the `.gguf` into LM Studio's models folder, `~/.lmstudio/models/<author>/<Name>-GGUF/`, and writes a `model.yaml` with a `manifest.json` into `~/.lmstudio/hub/models/<author>/<name>/` – the [model.yaml](https://modelyaml.org) format LM Studio's own models use. *Author* comes from Model Info (MacroMagi when empty). The model.yaml gives LM Studio:

- the size and context window, whether the model reasons and is trained for tools;
- the sampling the model wants (temperature, top-k) and its thinking tags, so LM Studio shows its thinking apart;
- an **Enable Thinking** toggle (with the Thinking block) and a **Reasoning Effort** choice (with the Reasoning Effort block). They set `enable_thinking` and `reasoning_effort`, which the chat template reads.

Restart LM Studio if it was open. Without model.yaml – the `.gguf` alone, or `lms import` – LM Studio still lists the model with its size and precision and uses its chat template.

LM Studio names a new chat by asking the model for a 2–5 word title. The example datasets teach that (`titles.jsonl`); a model trained without such samples may answer with something odd, as earlier examples did with `type`.

### Uploading to Hugging Face

Upload the `-huggingface` folder as a model repository (for example with `huggingface-cli upload <you>/<model> <folder>`), and the `.gguf` beside it if you like. The model card says which licence the model has, taken from Model Info.

## Running models

| Model | Runs with | Notes |
| --- | --- | --- |
| A text model trained here | ONNX, in MacroMagi (built-in engine, or ONNX Runtime) | Any architecture. |
| A GGUF model (imported, or exported from here) | llama.cpp (*Settings › Runtime*) | On the graphics cards chosen under *Settings › Hardware*. Fast: it keeps a cache while writing. A model trained here runs from its compiled GGUF when llama.cpp is installed (Prompt Model › *Run with*). |
| A vision model | ONNX, in MacroMagi | On the CPU or a graphics card (ONNX Runtime). |

**Importing a GGUF model** (*File › Import › A GGUF model*): the file stays where it is, and the `.magic` file points at it, because GGUF models are often gigabytes. The model opens with *Use As It Is*, Model Info and Text & Tools, Compile and Prompt Model. Its own chat template is used; MacroMagi's tools are described in its system prompt, so the AI Agent block can run it too.

**AI Agent** (in the Macro tab's AI category) asks a text model – ONNX or GGUF:

| Setting / port | Meaning |
| --- | --- |
| *Model* | A text model (a `.magic` file). |
| *Request* (setting or input) | What to ask. A wired value wins. |
| *Carry out its tool calls* | Off by default. On: it carries out `press {key}`, `type {text}`, `click {x, y, button}`, `move {x, y}`, `wait {ms}` and `log {text}`. |
| *Longest answer*, *Temperature* | 0, and below 0, use the model's own settings. |
| Outputs | *Answer*, *Thinking*, *Tool* and *Arguments* (its first tool call), *Tool calls* (how many), *Everything it wrote*. |

## Settings › Runtime and Hardware

**Runtime** lists what trains and runs AI models, LM Studio-style, each with its version, *Install* or *Installed*, and a menu to uninstall or install again:

| Group | Parts |
| --- | --- |
| *Python for AI* | Which Python training uses and the packages go into (the Python macros use; *Change…* opens Settings › Engine). |
| Training models | PyTorch for NVIDIA graphics cards (CUDA 12.8), PyTorch for the CPU (one replaces the other), onnx. |
| Running GGUF models | llama.cpp CUDA 13 (RTX 20s to RTX 50s), CUDA 12 (older NVIDIA cards), Vulkan (any vendor), CPU. Downloaded from llama.cpp's latest GitHub release. |
| Running ONNX models | ONNX Runtime, ONNX Runtime for graphics cards. |

What fits the computer is marked *Recommended here*. MacroMagi looks at all of this when it starts and again when something that needs these parts fails; *Check again* at the top of the page looks now. llama.cpp is installed from the newest of its recent releases that has the build, since its very latest release can be one without builds.

**Hardware** shows the processor, memory and graphics cards, and chooses:

- **Training**: *Automatic* (the first NVIDIA card PyTorch sees), *The CPU*, or *Chosen cards*. With several cards ticked, they share each batch.
- **GGUF models**: which llama.cpp build runs them, how much of a model goes on graphics cards (all of it, or some layers), and which cards (none ticked: llama.cpp chooses; several: the model is split between them).
- **ONNX models**: where vision models and ONNX text models run.

## How text is read and written

Tokens are the 256 byte values of UTF-8 text, followed by the special tags in the order the model lists them, each one token. This works for any language, and it is the same in MacroMagi, the trainer and the exports. A request is written as:

```text
<|system|>the system prompt

Reasoning effort: high

You can use these tools:
- get_weather: Gets the current weather for a place. Parameters: {"properties": {"location": {"type": "string"}}, …}
To use one, write <tool_call>{"name": "<tool name>", "arguments": {...}}</tool_call>. Its result comes back in the next turn.<|end|><|user|>the request<|end|><|assistant|>
```

The effort line is there with the Reasoning Effort block, the tools when the model has any. The model answers until it writes `<|end|>` or reaches the longest answer. It can think inside `<think>…</think>`; with thinking off, the answer starts after an empty `<think></think>`. It calls tools in one syntax, and a tool's result comes back in its own turn:

```text
<tool_call>{"name": "get_weather", "arguments": {"location": "Oslo"}}</tool_call><|end|><|tool|>{"temperature_c": 4}<|end|><|assistant|>It's 4 °C in Oslo.<|end|>
```

Earlier answers are read back without their thinking, except those since the last request. The GGUF and Hugging Face chat template writes exactly this, from OpenAI-style messages, `tools`, `enable_thinking` and `reasoning_effort`. Models made before tool turns existed keep `<tool>{"name", "args"}</tool>` and read results as user turns.

With the Dual-Persona Router, each persona's turn is inside its own tags, such as `<gun>…</gun>`, and counts as thinking. In GGUF and Hugging Face exports, bytes are written as GPT-2's byte characters with no merges, so other tools read the same tokens.

## What the architecture blocks are

The blocks are clean-room implementations of published designs, in plain PyTorch ([`runtime/python/magi_train/blocks.py`](../runtime/python/magi_train/blocks.py)):

| Block | Implementation |
| --- | --- |
| Transformer Layers | RMSNorm, rotary positions (halves layout), grouped-query attention, SwiGLU – the Llama layout, so the weights map one-to-one onto GGUF and Hugging Face. The finer options add RMSNorm per head on queries and keys before they turn, GeGLU (tanh-approximated GELU), RMSNorm on each part's output before it is added back, a sliding window on all but every *N*th layer, rotary positions on a share of each head, query/key/value biases, a tied output head and √width-scaled embeddings. |
| Latent Attention | Queries from an optional normalized latent; keys and values from a normalized latent per token, plus a rotary key part shared by the heads (DeepSeek-V2/V3). |
| Compressed Sparse Attention | Per channel, a softmax over learned scores plus position biases pools each block of *m* tokens together with the block before (2*m* candidates). Queries come from a latent and are normalized per head. One shared key/value head is used (MQA). The lightning indexer scores pooled entries as Σₕ wₕ · ReLU(qₕ · k). The top *k* eligible entries are chosen by rank, so ONNX needs no TopK. The indexer learns from dense attention over the entries through a KL loss. The token also sees a window of recent tokens. There is a learned sink per head. Rotary positions cover part of the channels, and outputs are rotated back by their own position. Heads are projected out in groups. |
| Heavily Compressed Attention | The same, with larger non-overlapping blocks attended to densely. |
| Mixture of Experts | Shared experts plus routed SwiGLU experts. Affinity by softmax, sigmoid or √softplus; choice by affinity plus a balancing bias that is nudged after each step (or a balance loss). Weights are normalized over the chosen experts. Hash routing fixes experts by token id. While training, each expert runs only on its tokens; exported models compute all experts and weight them. |
| Hyper-Connections | *n* residual streams. Before each sublayer, a sigmoid-weighted mix of them; after it, a Sinkhorn-normalized (doubly stochastic) mixing of the streams plus the output weighted back in. All three depend on the input, through an RMSNorm of the flattened streams, and start near a plain residual connection. |
| Multi-Token Prediction | The final hidden state and the next token's embedding, normalized and projected, go through one more layer to predict the token after next, weighted into the loss. Left out of the finished model. |
| Muon | Nesterov momentum, orthogonalized by Newton–Schulz (eight fast steps, then two settling ones), rescaled by √max(n, m) and a constant, with decoupled weight decay. AdamW for embeddings, the head, norms and gates. |

Not reproduced: DeepSeek's FP4/FP8 arithmetic, quantization-aware training and their exact sizes. MacroMagi trains in 32-bit floats. Exported GGUF files use 16-bit floats, or Q8_0 or Q4_0 blocks (32 values with one scale, as llama.cpp writes them).

## Limits

- **ONNX text models recompute the context for every token.** With a long context window that is slow even for small models; a GGUF export run with llama.cpp keeps a cache and is many times faster.
- **Only Transformer Layers in the Llama, Qwen3 or Gemma 3 layout export to GGUF** (Hugging Face: Llama and Qwen3). GLM- and DeepSeek-style models run in MacroMagi; llama.cpp's support for those expects their own sizes and names.
- **llama.cpp keeps a model within the context window it learned.** For models trained here, the Prompt Model block can lower it, not raise it.
- **Small models learn formats more than knowledge.** The 3.5M-parameter example picks the right tool and thinks at each effort, but copies details such as a city's name unreliably; bigger models and more data do better.
- **No vision input for text models yet.** A text model cannot take pictures (as Qwen's vision models do in LM Studio): that needs an image encoder block, picture-and-text datasets and an `mmproj` file llama.cpp reads.
- **Vision Encoder** is the only vision architecture. There are no OCR or audio input blocks yet.
- **Fine-tuning needs weights.** Imported ONNX and GGUF models can be used, not trained further.

## For developers

The tab turns the canvas into a JSON plan. In [`nodes/studio.rs`](../crates/magi-core/src/nodes/studio.rs), `pipeline` follows the Model wire, `derive` makes the model from the setup chain, and `plan` writes the plan. The trainer, [`runtime/python/magi_train`](../runtime/python/magi_train), reads it:

```json
{
  "kind": "text",
  "out_dir": "…/training/out",
  "devices": [0],
  "text": { "context": 2048, "specials": ["<|end|>", "…"], "tools": [], "tool_tag": "<|tool|>",
            "thinking": { "on_by_default": true }, "effort": { "levels": ["low", "medium", "high"], "default": "medium" },
            "…": "the model's text spec" },
  "architecture": {
    "width": 128, "heads": 4, "kv_heads": 2, "context": 256, "positions": "rope", "rope_theta": 10000,
    "hyper_connections": { "streams": 4, "sinkhorn": 20 }, "mtp": { "weight": 0.3 }, "prefix_split": false,
    "layers": [ { "attention": { "kind": "csa", "compress": 4, "top_k": 16, "window": 32 },
                  "ffn": { "kind": "moe", "experts": 8, "shared": 1, "top_k": 2, "gating": "sqrt_softplus", "balance": "bias" } } ]
  },
  "start": { "mode": "zero", "weights": null },
  "datasets": { "7.positive": ["…/data/0-language.md", "…/data/1-chat.jsonl"] },
  "stages": [
    { "type": "pretrain", "data": ["7.positive"], "steps": 400, "batch": 16, "curriculum_start": 0,
      "optimizer": { "name": "adamw", "lr": 0.002, "schedule": "cosine", "warmup": 20, "clip": 1.0, "accumulate": 1 } },
    { "type": "sft", "data": ["7.positive"], "negatives": ["7.negative"], "adapter": "full", "steps": 600, "batch": 16 }
  ]
}
```

The trainer prints one JSON object per line: `stage`, `progress` (`step`, `steps`, `loss`, `reward`), `log`, `warning`, `error` and `done`. It writes `model.onnx` (fixed context; input `tokens`, output `logits`; vision: `input` and `logits`), `weights.safetensors` (no pickle) and `report.json`. The GGUF and Hugging Face writers are in [`magi-core/src/export.rs`](../crates/magi-core/src/export.rs). Running GGUF models with llama.cpp is in [`magi-ml/src/llama.rs`](../crates/magi-ml/src/llama.rs).

Tests that go the whole way (they need Python with PyTorch and onnx, given by `MAGI_TEST_PYTHON`; `MAGI_TEST_GPU` picks a card, and `MAGI_TEST_LLAMA` is a folder with llama.cpp):

```bash
cargo test -p magi-app a_model_trains_and_answers_through_the_interface -- --ignored
cargo test -p magi-app a_vision_model_trains_through_the_interface -- --ignored
```
