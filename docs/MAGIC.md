# The Magic tab

The Magic tab builds, trains, sets up and tries AI models for macros, on a canvas like the Macro tab's. It handles two kinds of model:

- **Text models** read a request and write an answer. They can think first, inside thinking tags, and call tools in one fixed syntax. In a macro the **AI Agent** block asks them, and it can carry out the tool calls that press, type and click.
- **Vision models** look at part of the screen and pick an answer. In a macro the **AI Watch** and **AI Model** blocks run them.

A model is one `.magic` file. It holds the trained model (ONNX), its weights for further training, its datasets, its setup and the canvas itself. The format is specified in [FILE_FORMAT.md](FILE_FORMAT.md#magic--ai-models).

## Contents

- [The canvas](#the-canvas)
- [Blocks](#blocks)
- [Datasets](#datasets)
- [Training](#training)
- [Text models in macros](#text-models-in-macros)
- [How text is read and written](#how-text-is-read-and-written)
- [What the architecture blocks are](#what-the-architecture-blocks-are)
- [Limits](#limits)
- [For developers: the training plan](#for-developers-the-training-plan)

## The canvas

**New → Text model** or **New → Vision model** starts with a ready pipeline. A model travels left to right along the **Model** wire:

```text
Start from Zero → Model Size → Transformer → Pretraining → Fine-Tuning → Compile .magic Package
                                                  ↑             ↑   ↑
                                             Text Dataset ──────┘   └── (what not to do)
```

- **A start**: a new model (*Start from Zero*), a trained model's weights (*Load Base Model*), or the model as it is (*Inference / Config Only*).
- **Architecture blocks**, in order: the model's size, then its layers.
- **Training stages**, in order, each fed by a dataset.
- **Compile .magic Package** at the end says what the file includes. **Use in Macro** puts the model into the open macro.

The model's own setup is shown by **inspector blocks**, grouped in a box called *The model's setup*: Model Info, Text & Tools, Try It for a text model; Screen Area, Answers, When It Acts, Adjustable Settings and Picture Preparation as well for a vision model. Their faces show the current values, and selecting one opens its settings on the right. They do not have to be wired.

Opening a `.magic` file made before 0.1.5 builds this canvas from the file: an imported model gets *Inference / Config Only* and its inspector blocks, so it can be set up and used straight away. A model trained in MacroMagi opens with *Load Base Model*, its own architecture blocks and a Fine-Tuning stage.

Selecting a block shows its settings in the panel on the right. With nothing selected, the panel shows what the pipeline will do, or what is wrong with it. **Train** stays off until the pipeline can run, and its tooltip says why. While training, the console at the bottom shows each stage, its progress, the loss (with a chart), the mean reward during reinforcement learning, and any warnings.

## Blocks

### Start

| Block | What it does |
| --- | --- |
| **Start from Zero** | A new model with fresh weights, shaped by the architecture blocks after it. *Model*: text or vision. |
| **Load Base Model** | Starts from trained weights to fine-tune or train further: this model's own (empty path), or another `.magic` file trained in MacroMagi. Keep the base model's architecture blocks. |
| **Inference / Config Only** | The model as it is. Set it up and try it without training. |

### Model (inspectors)

| Block | Settings |
| --- | --- |
| **Model Info** | Name, what it does, author, licence; its kind, size, and what training it went through. |
| **Text & Tools** | The system prompt; temperature, how many likely tokens to choose among, the longest answer; the tools it may call (name, description, arguments as a JSON Schema – *Add MacroMagi's tools* adds press, type, click, move, wait and log); its personas; its chat tags. |
| **Try It** | Text: ask it something, and see its answer, its thinking, its tool calls and everything it wrote. Vision: watch the screen live, or try a picture file. |
| **Screen Area** | Vision: a square in the middle of a display (it scales with the display) or a fixed area. |
| **Answers** | Vision: each answer's name, whether it acts, how sure the model must be, and how long to wait. |
| **When It Acts** | Vision: the key or mouse button, how long it is held, and how long answers are ignored afterwards. |
| **Adjustable Settings** | Vision: sliders that move several values at once, offered again in every AI block that uses the model. |
| **Picture Preparation** | Vision: the picture size, layout, color order and normalisation. |

### Architecture

| Block | What it adds |
| --- | --- |
| **Model Size** | *Width* (numbers per token), *Heads* (attention heads), *Context*: the longest text in tokens, or the side of a vision model's pictures in pixels. |
| **Transformer** | Standard layers: attention, then a feed-forward layer. |
| **Compressed Sparse Attention** | Attention whose keys and values are made from a small *latent* per token, optionally shared by the block's layers. Each token sees a sliding window of the tokens before it, plus every n-th token. |
| **Micro-MoE Router** | A mixture of experts: shared experts every token goes through, and many small routed experts of which each token uses a few. |
| **Encoder/Decoder Split** | Everything up to the assistant tag (the request, tool output) is read in both directions; only the answer is written token by token. |
| **Mamba-2 / SSM** | State-space layers: a learned decay over time instead of attention. |
| **Dual-Persona Router** | Two personas of the reasoning (by default *Gun*, for strict code and tool calls, and *Magi*, for critique and breaking loops), each with its own tag and its own experts, chosen by the tags. |
| **Vision Encoder** | Vision: convolution layers, then the answers. |

### Data

| Block | What it holds |
| --- | --- |
| **Text Dataset** | Text files kept in the `.magic` file, in two lists with an output each: *What to do* and *What not to do*. Select it to add files, start a new `.jsonl` or `.md`, or open a file in the dataset editor. |
| **Vision Dataset** | The model's pictures: per answer, pictures that show it and pictures it was mistaken for. Thumbnails, *Add…*, and right-click to remove. |

### Training

| Block | What it does |
| --- | --- |
| **Pretraining** | Learns the language: next-token prediction on all the text in its dataset, in random windows. With the Optimizer's curriculum on, the windows grow from a short start to the full context. |
| **Fine-Tuning** | Learns to answer: requests and answers from *What to do*. Answers from *What not to do* are made less likely, as strongly as *“Not to do” weight* says. *Train*: all weights, or LoRA or DoRA adapters (rank and strength) that are folded back into the weights afterwards. |
| **Reinforcement Learning** | For each request, several answers are drawn and scored by the reward rules; the better-than-average ones are made more likely. |
| **Context Compaction** | Learns to turn a long context into a short one, from `{"context", "compact"}` samples. |
| **Vision Training** | Learns to tell the answers apart from the pictures. Pictures of what an answer is *not* make that answer less likely for them. |
| **Optimizer & Memory** | Applies to every stage: AdamW, Muon (for the layers' weight matrices, with AdamW for the rest) or SGD; learning rate; weight decay; the context curriculum; recomputing activations to save memory; optimizer state in system memory while training on a graphics card; accumulating batches; CPU or graphics card. |

**Reward rules.** Each answer scores the sum of the rules it meets. The default set gives +1 for the dataset's answer, +1 for well-formed tool calls, +5 for an answer of at most 24 tokens, −3 for a loop (a run of 4 tokens repeated 3 times) and −1 for a broken tool call.

| Rule | Met when the answer… |
| --- | --- |
| Says… / Does not say… | contains (or does not contain) a text |
| Matches the pattern… | matches a regular expression |
| Gives the dataset's answer | equals the sample's response |
| Calls tools correctly / Writes a broken tool call | has only well-formed tool calls / has a malformed one |
| Repeats itself (a loop) | repeats a run of tokens the given number of times |
| Is short (fast) | has at most the given number of tokens |
| Finishes its answer | ends its turn before running out of tokens |

### Package

| Block | What it does |
| --- | --- |
| **Compile .magic Package** | What the file includes: the model, its weights, the text datasets, the pictures, the blocks to use it with. The canvas, architecture, tokens and chat tags, tool schemas, answers and thresholds are always included. Save and Save as are here too. |
| **Use in Macro** | Puts the model into the open macro: a text model with Start → AI Agent → Log, a vision model with Start → AI Watch. Using it again after a change updates the model in the macro. |

## Datasets

Text datasets are `.jsonl` files (one JSON object per line), or `.md` and `.txt` files (plain text, split at blank lines, for pretraining). A `.jsonl` line is one of:

```json
{"prompt": "Press the Enter key", "response": "<think>One tap.</think><tool>{\"name\": \"press\", \"args\": {\"key\": \"Enter\"}}</tool>Done."}
{"messages": [{"role": "system", "content": "…"}, {"role": "user", "content": "Say hello"}, {"role": "assistant", "content": "Hello!"}]}
{"prompt": "Press the Enter key", "rejected": "<tool>{\"name\": \"press\" …"}
{"context": "a long conversation …", "compact": "the short version"}
{"text": "plain text to pretrain on"}
```

`prompt`/`response` and `messages` teach answers; a conversation's last assistant message is the answer, and the messages before it are the request. `prompt`/`rejected` (or `bad`) is an answer not to give. `context`/`compact` trains context compaction. Files in *What not to do* are all used as answers to avoid.

**The dataset editor** opens a file in its own window. It can search the samples, show only those with problems, add, duplicate, edit and delete them. Each sample is checked as you type:

- it must be valid JSON, of one of the shapes above;
- a conversation needs an assistant message, and roles must be system, user, assistant or tool;
- every tool call must be closed, be JSON with a `name`, and have an object as its `args`;
- when the model lists its tools, a call must name one of them.

*Keep changes* puts the samples back into the model; saving the model writes them into the `.magic` file.

## Training

Training runs in Python with **PyTorch** and **onnx**. *Settings → AI → Downloads* installs both into the Python macros use, or uses copies already there. The tab writes the plan, the dataset files and the prepared pictures into `data/training` in the data folder. It then runs `runtime/python/magi_train_run.py` and reads its progress line by line. When training ends, the new ONNX model, its weights and a record of the training go into the open model. Nothing leaves the computer.

- Training uses a graphics card when PyTorch can use one (a CUDA build of PyTorch); otherwise the CPU. The PyTorch that *Settings → AI* installs is the CPU build. That is fine for the small models this tab is for, and slow for large ones.
- **Stop training** ends the Python process. The model is left as it was.
- To train a model further, replace *Start from Zero* with *Load Base Model* (empty path: this model's weights), keep the architecture blocks, and add the stages to run.
- Models imported from elsewhere come as ONNX without weights. They can be set up and used, but not fine-tuned.
- Vision training prepares the pictures at the Model Size block's *Context* (the side, in pixels), with the Picture Preparation block's normalisation. The trained model reads pictures that size, channels first, and gives probabilities.

## Text models in macros

**AI Agent** (in the AI category of the Macro tab) asks a text model:

| Setting / port | Meaning |
| --- | --- |
| *Model* | A text model asset (a `.magic` file). |
| *Request* (setting or input) | What to ask. A wired value wins. |
| *Carry out its tool calls* | Off by default. On: the block carries out `press {key}`, `type {text}`, `click {x, y, button}`, `move {x, y}`, `wait {ms}` and `log {text}` as the model calls them. |
| *Longest answer*, *Temperature* | 0 and below 0 use the model's own settings. |
| Outputs | *Answer* (without thinking and tool calls), *Thinking*, *Tool* and *Arguments* (of its first tool call, as JSON), *Tool calls* (how many), *Everything it wrote*. |

Tools other than the six above are not carried out. Wire *Tool* and *Arguments* into a Decision and the blocks that do the work. AI Watch and AI Model report an error for a text model, and AI Agent reports one for a vision model.

## How text is read and written

Tokens are the 256 byte values of UTF-8 text, followed by the special tags in the order the model lists them (`text.specials`), each one token. This is simple, works for any language, and is the same in MacroMagi and in training. A request is written as:

```text
<|system|>the system prompt<|end|><|user|>the request<|end|><|assistant|>
```

The model answers until it writes `<|end|>` or reaches the longest answer. It can think inside `<think>…</think>`, and calls tools in one syntax only:

```text
<tool>{"name": "press", "args": {"key": "Enter"}}</tool>
```

With the Dual-Persona Router, each persona's turn is inside its own tag pair, such as `<gun>…</gun>`. A persona's turn counts as thinking: it is in *Thinking*, not in *Answer*. Chat tags can be changed under Text & Tools › Chat tags. A model has to be trained again after its tags change.

## What the architecture blocks are

The blocks follow published techniques. They are written in plain PyTorch in [`runtime/python/magi_train/blocks.py`](../runtime/python/magi_train/blocks.py):

| Block | Implementation |
| --- | --- |
| Compressed Sparse Attention | Keys and values are projected up from a small per-token latent (as in multi-head latent attention). The mask is a sliding window plus every n-th token. With *Share the cache across layers*, the block's layers reuse the first layer's latent. *Cache precision* is recorded in the file for runtimes; MacroMagi keeps 32-bit values, and there is no FP4 arithmetic. |
| Micro-MoE Router | Shared experts plus top-k routed experts with softmax gates and a load-balancing loss. Every expert is computed and weighted: correct, but no faster than a dense layer of the same total size. |
| Encoder/Decoder Split | A prefix-LM mask: positions up to the assistant tag attend to each other in both directions. It is one stack of layers, not two. |
| Mamba-2 / SSM | The quadratic ("dual") form of a scalar-decay state-space layer: exact, computed in parallel over the sequence. |
| Dual-Persona Router | A feed-forward expert per persona plus a neutral one. Each token is routed by the persona tag it is inside. |

The names follow the ideas they come from, such as DeepSeek's MoE and latent attention and the Mamba-2 paper. They are clean-room implementations of those published ideas. They do not copy any model's internals, and they are not compatible with any model's weights.

## Limits

- **No cache while generating.** Every new token runs the model over the whole context again (padded to the context length), so answers of hundreds of tokens from a large model are slow. Small models answer in well under a second.
- **Vision Encoder** is the only vision architecture. There are no OCR or audio input blocks yet.
- **Training is for small models.** Nothing stops a large architecture, but the CPU build of PyTorch trains it slowly.
- **Fine-tuning needs weights.** Models imported as ONNX cannot be trained further.

## For developers: the training plan

The tab turns the canvas into a JSON plan ([`nodes/studio.rs`](../crates/magi-core/src/nodes/studio.rs): `pipeline` follows the model wire, `plan` writes the plan). The trainer ([`runtime/python/magi_train`](../runtime/python/magi_train)) reads it:

```json
{
  "kind": "text",
  "out_dir": "…/training/out",
  "seed": 0,
  "text": { "context": 256, "specials": ["<|end|>", "…"], "personas": [], "…": "the model's text spec" },
  "architecture": { "width": 128, "heads": 4, "blocks": [ { "kind": "transformer", "params": { "layers": 2, "mult": 4 } } ] },
  "start": { "mode": "zero", "weights": null },
  "datasets": { "7.positive": ["…/data/0-hi.jsonl"], "7.negative": ["…"] },
  "stages": [
    { "type": "pretrain", "data": ["7.positive"], "negatives": [], "steps": 300, "batch": 8, "lr": 0 },
    { "type": "sft", "data": ["7.positive"], "negatives": ["7.negative"], "adapter": "lora", "rank": 8, "alpha": 16, "steps": 200, "batch": 4, "negative_weight": 0.5 },
    { "type": "rl", "data": ["7.positive"], "group": 4, "max_new": 48, "temperature": 0.9, "rules": [ { "kind": "exact", "score": 1 } ] }
  ],
  "optimizer": { "name": "adamw", "lr": 0.003, "weight_decay": 0.01, "checkpointing": false, "offload": false, "accumulate": 1, "device": "auto", "curriculum": { "start": 32, "end": 256 } }
}
```

A vision plan has `"kind": "vision"`, a `vision` entry (`pictures`: the prepared pictures file, a JSON header line followed by float32 `[3, h, w]` tensors; `classes`: how many answers) and a `vision` stage.

The trainer prints one JSON object per line: `stage`, `progress` (`step`, `steps`, `loss`, and `reward` in reinforcement learning), `log`, `warning`, `error` and `done` (the report also written to `report.json`). It writes `model.onnx` (input `tokens` `[1, length]` and output `logits` for text; input `input` `[1, 3, h, w]` for vision) and `weights.safetensors` (read and written without pickle, so loading weights cannot run code).

`cargo test -p magi-app a_new_text_model_trains_end_to_end -- --ignored` runs the whole way – canvas, plan, trainer, the trained model answering – with the Python named by `MAGI_TEST_PYTHON` (or `python`), which needs PyTorch and onnx.
