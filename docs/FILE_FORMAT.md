# File formats

MacroMagi has three file types: **`.magi`** (a macro), **`.mp`** (a preset) and **`.magic`** (an AI model). All are designed to be readable, diffable and easy to produce with other tools.

- [`.magi` – projects](#magi--projects)
- [`.mp` – presets](#mp--presets)
- [`.magic` – AI models](#magic--ai-models)
- [Shared vocabulary](#shared-vocabulary)
- [Versioning](#versioning)

## `.magi` – projects

A `.magi` file is a standard ZIP archive. Rename it to `.zip` to open it.

```text
project.json           required – the macro
assets/<id>.png        images
assets/<id>.json       recordings
assets/<id>.wav        sounds (for Sound Recognition)
assets/<id>.magic      AI models (for the AI blocks)
presets/<nn>-<name>.mp bundled presets (optional)
```

`project.json` lists every other file in the archive. A reader never has to guess file names, and files not listed are ignored.

### `project.json`

```json
{
  "format": "macromagi.project",
  "version": 1,
  "app": "0.1.2",
  "meta": { "name": "Click when ready", "description": "", "author": "" },
  "engine": "auto",
  "graph": {
    "nodes": [
      { "id": 1, "type": "flow.start", "pos": [0.0, 0.0],
        "params": { "hotkey": "F7", "mode": "toggle", "run_on_start": true } },
      { "id": 2, "type": "input.action", "pos": [0.0, 120.0], "label": "Click it",
        "params": { "actions": [ { "kind": "mouse", "button": "Left", "mode": "tap", "clicks": 1 } ],
                    "hold_ms": { "$var": "hold" } } },
      { "id": 3, "type": "vision.find_image", "pos": [-260.0, 100.0],
        "params": { "images": [ { "image": "img-9f2c61d0a4b7e113" } ], "region": null, "threshold": 0.8 } }
    ],
    "wires": [
      { "from": { "node": 1, "port": "out" },      "to": { "node": 2, "port": "in" } },
      { "from": { "node": 3, "port": "position" }, "to": { "node": 2, "port": "position" } }
    ],
    "groups": [
      { "id": 1, "name": "Find and click", "color": [63, 127, 230], "nodes": [2, 3] }
    ]
  },
  "assets": [
    { "id": "img-9f2c61d0a4b7e113", "name": "Ready button", "kind": "image", "file": "assets/img-9f2c61d0a4b7e113.png" }
  ],
  "presets": [],
  "variables": [ { "name": "hold", "value": 30 }, { "name": "total", "value": null, "reset": false } ],
  "options": { "stop_on_task_switch": true, "stop_hotkeys": ["Escape"] }
}
```

| Field | Meaning |
| --- | --- |
| `format` | Always `macromagi.project`. Anything else is rejected. |
| `version` | Format version, see [Versioning](#versioning). |
| `app` | The MacroMagi version that wrote the file. Informational. |
| `meta` | Name, description and author. All optional. |
| `engine` | `auto` (follow the app setting), `python` or `native`. |
| `graph` | The nodes, wires and [groups](#groups). |
| `assets` | One entry per file under `assets/`. |
| `presets` | Paths of bundled `.mp` files inside the archive. |
| `variables` | Optional. The macro's [variables](#variables) and the values they start with. |
| `options` | Optional. [Automation options](#automation-options) that travel with the macro. |
| `clicker` | Optional. [Auto Clicker settings](#kind-auto_clicker) that travel with the macro. |

`options` and `clicker` are written only when the user chooses to include them in the Export dialog. When a file that has them is opened, MacroMagi offers to use them.

### Automation options

Rules about when MacroMagi may act. All fields are optional.

```json
{
  "strict_modifiers": false,
  "stop_on_task_switch": false,
  "stop_hotkeys": ["Escape", "ControlLeft+AltLeft+KeyY"],
  "stop_when_editing_hotkey": true,
  "process_rules": {
    "enabled": true,
    "mode": "whitelist",
    "whitelist": [ { "name": "game.exe", "enabled": true, "action": "pause" } ],
    "blacklist": []
  }
}
```

| Field | Meaning |
| --- | --- |
| `strict_modifiers` | `true`: a hotkey fires only when exactly its modifier keys are held. |
| `stop_on_task_switch` | Stop everything on Alt+Tab or Win+Tab. |
| `stop_hotkeys` | More [hotkeys](#shared-vocabulary) that stop everything. |
| `stop_when_editing_hotkey` | Stop the Auto Clicker when a hotkey field starts listening. |
| `process_rules.mode` | `whitelist` (input only reaches the listed programs) or `blacklist` (never reaches them). |
| `process_rules.*list` | Executable names, matched without regard to case, each with `action` `stop` or `pause`. |

### Nodes

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | A positive integer, unique within the graph. |
| `type` | yes | The block type, e.g. `flow.delay`. See the [block reference](NODES.md). |
| `pos` | yes | `[x, y]` of the node's top-left corner on the canvas. |
| `params` | no | The block's settings. Missing settings take their defaults. |
| `enabled` | no | `false` to disable the block. Omitted when `true`. |
| `label` | no | A custom title. Omitted when empty. |

The meaning of `params` is defined per block type. `macromagi nodes` prints every block's parameters with their kinds and defaults.

A node whose `type` is unknown to the reading app is kept untouched and written back as it was.

### Variables

```json
"variables": [ { "name": "hold", "value": 30 }, { "name": "greeting", "value": "hello" } ]
```

A variable has a `name` and a starting `value` (a number, text, true/false or `null` for none). Set Variable, Counter and Python scripts change them while a macro runs.

`reset` says whether a run starts from `value`. It is `true` when omitted, which is how every variable behaved before 0.1.2. With `"reset": false` the variable keeps what the last run left in it, for as long as MacroMagi is open; `value` is then only where the first run starts. Variables made in 0.1.2 and later start this way, with no value.

**Bound settings.** Any setting of a block can be bound to a variable instead of holding a value of its own. It is then stored as

```json
"hold_ms": { "$var": "hold" }
```

and takes the variable's value each time the block runs, converted to the kind of value the setting expects (a number for durations and counts, text for text, true/false for switches). A binding to a variable that does not exist uses the setting's default.

### Groups

```json
"groups": [
  { "id": 1, "name": "Find and click", "color": [63, 127, 230], "nodes": [2, 3], "enabled": false },
  { "id": 2, "name": "Retry", "nodes": [4], "parent": 1 }
]
```

| Field | Meaning |
| --- | --- |
| `id` | A positive integer, unique among the groups. |
| `name` | Shown on the group's box. |
| `color` | Optional `[r, g, b]`; omitted for the theme color. |
| `nodes` | Ids of the blocks in the group. Ids of missing blocks are dropped on load; a group left empty is dropped. |
| `enabled` | Optional; `false` turns the group off, which skips every block in it – and in the groups inside it – as if each were disabled. Omitted when `true`. |
| `parent` | Optional. The `id` of the group this one is inside. Omitted for a group at the top level. |

A block is directly in one group at most: `nodes` lists only the blocks that are not in a group inside it. Files from before 0.1.3 could list a block in several groups; such a block stays in the smallest of them, and a group whose blocks were all in a larger group is put inside it.

### Wires

A wire goes **from an output port to an input port**.

- **Flow** wires end at the port named `in`. A flow output has at most one wire.
- **Value** wires end at a named value input. A value input has at most one wire.

Port names come from the block's definition. Ports of a Decision are derived from its cases: case `"id": "2"` has the condition input `cond:2` and the flow output `case:2`.

Wires that refer to missing nodes are dropped on load.

### Assets

| `kind` | File | Contents |
| --- | --- | --- |
| `image` | `assets/<id>.png` | A PNG image. |
| `recording` | `assets/<id>.json` | A [recording](#recordings). |
| `audio` | `assets/<id>.wav` | A sound to listen for: a 16-bit mono WAV file, 16 kHz, at most 10 seconds. Uploaded MP3, OGG and FLAC files are converted to this. |
| `model` | `assets/<id>.magic` | An AI model: a [`.magic` file](#magic--ai-models) without training pictures. |

Blocks refer to assets by `id` in their parameters (`"image": "img-…"`, `"recording": "rec-1"`, `"sound": "snd-…"`, `"model": "mdl-…"`). Image, sound and model ids are derived from the content, so the same picture, sound or model is stored once.

### Image and sound lists

An Image Detection block looks for a list of images, and a Sound Recognition block listens for a list of sounds. Each may belong to a named group.

```json
"params": {
  "images": [
    { "image": "img-…", "group": "Buttons" },
    { "image": "img-…", "group": "Buttons", "region": { "x": 0, "y": 900, "w": 600, "h": 180 } },
    { "image": "img-…" }
  ],
  "groups": [ { "name": "Buttons", "region": { "x": 0, "y": 800, "w": 1920, "h": 280 } } ],
  "region": null,
  "mode": "any",
  "detail": "half",
  "threshold": 0.8
}
```

| Field | Meaning |
| --- | --- |
| `images[].region` | Optional. Where this image is looked for. Without it, its group's `region`, and without that, the block's. |
| `groups` | The groups' names and optional search areas. |
| `mode` | `any`: found when one of the images is there (the first in the list wins); `all`: only when every one is. |
| `detail` | `full`, `half`, `third` or `quarter`: the screen and the images are shrunk by that much before comparing. |

Sound Recognition has `sounds` (`{ "sound": "snd-…", "group": "…" }`), `groups` (`{ "name": "…", "accuracy": 0.8 }`, where `accuracy` is optional and overrides the block's), `mode` and `match_ms` (recognise a sound by its first milliseconds; `0` for the whole sound). Sound Detection and Sound Recognition have `device`: the name of the microphone, or `""` for the system's default.

Blocks saved before 0.1.2 have a single `"image"` or `"sound"` instead. It is still read, as the first entry of the list.

### Pictures and paths

Screenshot and Read File output pictures, which only exist while the macro runs: a value `{"$picture": "picture-1", "width": 16, "height": 12}` names one, and the pixels stay with the run. Image Detection looks for a picture wired into its `picture` input instead of its list; Write to File saves one as a PNG.

File blocks have a `path`: a full path, or one relative to the `files` folder in MacroMagi's data folder.

### AI blocks

AI Watch and AI Model have `model` (the id of a `model` asset) and `settings`: values for the model's adjustable settings, by key, for this block only. A key that is missing takes the model's own value. AI Watch has two flow outputs: `acted`, followed each time the model acts, and `out`, followed when its time is up.

```json
"params": { "model": "mdl-3a…", "settings": { "thresh": 60 }, "display": 0, "press": true, "duration_ms": 0, "max_fps": 0 }
```

AI Agent (`ai.agent`) runs a text model. `max_tokens` 0, `temperature` below 0, `context` 0, `thinking` `"model"` and an empty `effort` use the model's own settings. `native_tools` runs the model's native tools, at most `max_steps` calls per request.

```json
"params": { "model": "mdl-7c…", "request": "Say hello", "run_tools": false, "max_tokens": 0, "temperature": -1,
            "thinking": "model", "effort": "", "context": 0, "native_tools": true, "max_steps": 6 }
```

AI Load Model (`ai.load`) and AI Unload Model (`ai.unload`) have `model` only.

### Recordings

A recording is a list of **strokes** – one per thing the user did – rather than a raw event log.

```json
{
  "duration_ms": 1620.0,
  "screen": [1920, 1080],
  "strokes": [
    { "at": 0.0,   "hold": 70.0,  "type": "key",    "key": "KeyH" },
    { "at": 610.0, "hold": 420.0, "type": "move",   "path": [[0.0, 640, 360], [14.0, 648, 369]] },
    { "at": 1080.0,"hold": 85.0,  "type": "mouse",  "button": "Left", "x": 900, "y": 360 },
    { "at": 1100.0,"hold": 300.0, "type": "mouse",  "button": "Left", "x": 10, "y": 10, "end": { "x": 200, "y": 90 } },
    { "at": 1380.0,"hold": 0.0,   "type": "scroll", "dx": 0, "dy": -2 },
    { "at": 1500.0,"hold": 90.0,  "type": "look",   "path": [[0.0, 12, -3], [8.0, 15, -2]] },
    { "at": 1530.0,"hold": 40.0,  "type": "mouse",  "button": "Left", "x": 960, "y": 540, "here": true },
    { "at": 1600.0,"hold": 0.0,   "type": "display", "index": 1 },
    { "at": 1650.0,"hold": 0.0,   "type": "key_down", "key": "ShiftLeft" },
    { "at": 1700.0,"hold": 60.0,  "type": "key",    "key": "KeyA" },
    { "at": 1790.0,"hold": 0.0,   "type": "key_up", "key": "ShiftLeft" }
  ]
}
```

Every stroke has:

- `at` – milliseconds from the start of the recording to the **press**;
- `hold` – milliseconds until the **release** (zero for strokes that have none).

| `type` | Extra fields | Notes |
| --- | --- | --- |
| `key` | `key` | |
| `mouse` | `button`, `x`, `y`, optional `end`, optional `here` | With `end`, a drag: pressed at `x, y`, released at `end`. With `here: true`, clicked wherever the pointer is (recorded while a game had the pointer locked). |
| `move` | `path` | A list of `[ms since the stroke began, x, y]` screen positions. |
| `look` | `path` | A list of `[ms since the stroke began, dx, dy]`: relative mouse motion, as a game camera reads it. |
| `scroll` | `dx`, `dy` | Wheel notches. Positive `dy` is up, positive `dx` is right. |
| `pad_button` | `button` | |
| `pad_axis` | `axis`, `value` | |
| `key_down`, `key_up` | `key` | A key pressed while other inputs happened, and its release. A key pressed and released with nothing else in between is a single `key` stroke with a `hold`. |
| `mouse_down`, `mouse_up` | `button`, `x`, `y`, optional `here` | The same for mouse buttons. |
| `display` | `index` | The positions of the strokes after it are relative to this display (0 is the primary display). Recordings made across displays have one wherever the pointer moved to another display. Positions before the first `display` stroke are on the whole desktop. |

Strokes need not be sorted; playback orders events by time (strokes with the same time keep their order). `screen` is the size of the primary screen at recording time, used to find the centre a game re-centres the pointer on, and kept for future rescaling.

A Play Recording block's `mouse` parameter says how movement is replayed: `recorded` (each stroke as it was recorded), `absolute` (`move` strokes only) or `relative` (everything as relative motion).

## `.mp` – presets

A preset is a single JSON file.

```json
{
  "format": "macromagi.preset",
  "version": 1,
  "app": "0.1.2",
  "name": "Rapid fire",
  "description": "",
  "kind": "auto_clicker",
  "data": { "…": "…" }
}
```

### `"kind": "auto_clicker"`

`data` is the Auto Clicker's settings – the same object as a project's `clicker` field, and, without `hotkey` and `activation`, the `clicker` parameter of an Auto Click block:

```json
{
  "target":       { "kind": "mouse", "button": "Left" },
  "clicks":       1,
  "double_click_delay_ms": 40.0,
  "speed_mode":   "rate",
  "interval_ms":  100.0,
  "rate_unit":    "seconds",
  "interval_unit": "ms",
  "variation":    { "on": false, "value": 15.0 },
  "hold_ms":      10.0,
  "duty_cycle":   { "on": false, "value": { "min": 30.0, "max": 50.0 } },
  "limit":        { "kind": "forever" },
  "time_limit":   { "on": false, "value": 60000.0 },
  "time_limit_unit": "seconds",
  "position":     { "kind": "cursor" },
  "corner_stop":  { "on": false, "value": [25, 25, 25, 25] },
  "edge_stop":    { "on": false, "value": [5, 5, 5, 5] },
  "course_correction": true,
  "high_speed":   false,
  "hotkey":       "F6",
  "activation":   "toggle"
}
```

| Field | Values |
| --- | --- |
| `target` | `{ "kind": "mouse", "button": … }`, `{ "kind": "key", "key": … }` or `{ "kind": "dual_keys", "first": …, "second": …, "mode": "alternate" \| "together" }` (`alternate` when omitted) |
| `clicks` | `1`, `2` or `3` (mouse only); `double_click_delay_ms` is the pause between them |
| `interval_ms` | Time from one click to the next. This is the speed; `speed_mode`, `rate_unit` and `interval_unit` only say how it is shown. |
| `*_unit` | `ms`, `seconds`, `minutes`, `hours` or `days` |
| `variation` | Vary each interval by up to this many percent |
| `duty_cycle` | Hold each click for a random share (percent) of the interval, instead of `hold_ms` |
| `limit` | `{ "kind": "forever" }` or `{ "kind": "count", "count": 100 }` |
| `time_limit` | Stop after this many milliseconds (at most 365 days) |
| `position` | `{ "kind": "cursor" }` or `{ "kind": "fixed", "x": 300, "y": 400, "display": 0 }` (`display` optional, see [screen positions](#shared-vocabulary)) |
| `corner_stop` | Stop within this many pixels of a corner: top-left, top-right, bottom-left, bottom-right |
| `edge_stop` | Stop within this many pixels of an edge: top, right, bottom, left |
| `course_correction` | Keep clicks on schedule by shortening the wait after a late click |
| `high_speed` | Allow up to 1000 clicks per second instead of 100 |
| `activation` | `toggle` or `hold` |

Missing fields take their defaults. Files from the first version, which had only `target`, `clicks`, `interval_ms`, `jitter_ms`, `hold_ms`, `limit`, `position`, `hotkey` and `activation`, load unchanged.

### `"kind": "nodes"`

`data` is a piece of a graph, plus the assets it uses, embedded so the file is self-contained:

```json
{
  "nodes": [ { "id": 1, "type": "vision.find_image", "pos": [0.0, 0.0], "params": { "image": "img-…" } } ],
  "wires": [],
  "assets": [
    { "id": "img-…", "name": "Ready button", "kind": "image", "base64": "iVBORw0KGgo…" },
    { "id": "rec-1", "name": "Take 1", "kind": "recording", "recording": { "duration_ms": 0.0, "strokes": [] } },
    { "id": "snd-…", "name": "Alarm", "kind": "audio", "wav_base64": "UklGRi…" },
    { "id": "mdl-…", "name": "Ready check", "kind": "model", "magic_base64": "UEsDBB…" }
  ]
}
```

A nodes preset may also carry `groups` whose blocks are all in it. When it is inserted, its nodes get fresh ids and its assets are imported (images, sounds and models are de-duplicated by content). Variables are not part of a preset: settings bound to a variable the receiving macro does not have use their defaults until it is created. This is also the format MacroMagi puts on the clipboard when blocks are copied, prefixed with `macromagi-nodes:`.

## `.magic` – AI workspaces and models

A `.magic` file is a ZIP archive: a Magic tab **workspace** (the canvas, the setup, the datasets and pictures) and, unless a Magi package holds it, the AI model itself – a vision model or a text model. The Magic tab reads and writes it; macros embed it, with the model, as a `model` asset.

```text
magic.json                    required – the description below
model/model.onnx              the model, in ONNX format (left out when include.model is false)
model/weights.safetensors     its weights (models trained in MacroMagi; optional)
data/text/<file>              text datasets: .jsonl, .md or .txt (optional)
data/train/<class>/<file>     pictures of each answer, one folder per class (optional)
data/not/<class>/<file>       pictures an answer was mistaken for (optional)
data/tune/<class>/<file>      fine-tuning pictures of earlier versions (optional)
nt/<tool>/<file>              the model's own native tools (optional)
```

Since 0.1.6, compiling a text model writes a Magi package (below) and the workspace's `package` names it; the workspace then leaves out `model/` (`include.model` and `include.weights` false), and opening it takes the model from the package – the path given, or `szi/<file>` beside the workspace. A model asset in a macro always holds its model, and its weights when MacroMagi's native engine runs it.

The weights are in the [safetensors](https://github.com/huggingface/safetensors) layout: an 8-byte header length, a JSON header, then raw little-endian tensors. They are data only; loading them runs no code.

### `magic.json`

```json
{
  "format": "macromagi.magic",
  "version": 1,
  "app": "0.1.2",
  "name": "Ready check",
  "description": "Presses Space when the ready prompt is on screen.",
  "input": { "name": "input", "width": 224, "height": 224, "layout": "nchw", "color": "rgb",
             "scale": 0.00392157, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "softmax": true },
  "capture": { "area": { "kind": "center", "height_share": 0.2074 }, "display": 0 },
  "classes": [
    { "label": "Nothing", "act": false, "threshold": 0.5, "delay_ms": 0 },
    { "label": "Ready", "act": true, "threshold": 0.6, "delay_ms": 0 }
  ],
  "action": { "key": "Space", "hold_ms": 5, "cooldown_ms": 500 },
  "settings": [
    { "key": "thresh", "label": "Confidence", "kind": "percent", "min": 1, "max": 100, "default": 60, "value": 60,
      "targets": [ { "kind": "threshold", "class": 1 } ] }
  ],
  "include": { "model": true, "blocks": true, "training": true, "finetune": true },
  "training": [ { "file": "data/train/1/0001-ready.png", "class": 1 } ]
}
```

| Field | Meaning |
| --- | --- |
| `format`, `version`, `app` | As for projects: `macromagi.magic`, the format version, the MacroMagi version that wrote it. |
| `name`, `description`, `author`, `source`, `license` | About the model. `source` and `license` record where an imported model came from and its licence. |
| `input` | How a picture becomes the model's input: resized to `width × height`, laid out `nchw` (channels first) or `nhwc`, in `rgb` or `bgr` order, each value `v` (0–255) becoming `(v × scale − mean) / std` per channel. `softmax: true` turns raw scores into probabilities. `name` is the model's input name. |
| `capture` | The screen area shown to the model: `{ "kind": "center", "height_share": … }` – a square in the middle of display `display`, as tall as that share of the display's height – or `{ "kind": "region", "rect": { "x", "y", "w", "h", "display" } }`. |
| `classes` | One per answer the model gives, in its output order. `act`: whether this answer calls for the action; `threshold` (0–1): how sure the model must be; `delay_ms`: how long to wait before acting. |
| `action` | The [key or mouse button](#shared-vocabulary) to press (`"Space"`, `"Mouse:Left"`), how long to hold it, and how long to ignore answers after acting. |
| `settings` | Adjustable settings, shown as sliders in the Magic tab and in each AI block. `kind` is `percent` or `milliseconds`. Each sets its `targets`: `{ "kind": "threshold", "class": n }`, `{ "kind": "delay", "class": n }`, `{ "kind": "cooldown" }` or `{ "kind": "hold" }`. A setting's `value` is applied over the values it targets. |
| `blocks` | Optional. Blocks to use the model with – `nodes`, `wires` and `groups` as in a [nodes preset](#kind-nodes), without assets; added to a macro along with the model. |
| `include` | What the archive holds besides `magic.json`: the model, the blocks, training pictures, fine-tuning pictures. |
| `training`, `finetune` | The pictures under `data/`, each with the class it shows. `negative: true` marks a picture of what the class is *not*. |
| `kind` | `vision` (the default, and what files without it are) or `text`. |
| `text` | Text models: how they read and write – see below. |
| `architecture` | The shape of a model trained in MacroMagi, as the trainer builds it: `width`, `heads`, `kv_heads`, `context`, `positions` (`rope` or `learned`), `rope_theta`, optional `hyper_connections` (`{"streams", "sinkhorn"}`) and `mtp` (`{"weight"}`), `prefix_split`, `tied` and `embed_scale` (optional booleans), `layers` (one per layer: `{"attention": {"kind": "full" \| "window" \| "mla" \| "csa" \| "hca" \| "ssm", …}, "ffn": {"kind": "dense" \| "moe" \| "persona", …}, "norm": "pre" \| "sandwich"}`; `full` attention may have `qk_norm`, `bias`, `rotary` and `window`, a `dense` feed-forward `activation`: `swiglu` or `geglu`), `conv` for vision models, and `blocks`: the canvas blocks it was made from, each `{ "kind", "params", "ffn" }`. |
| `studio` | The Magic tab's canvas: `nodes`, `wires` and `groups` as in a [nodes preset](#kind-nodes), with the Magic tab's block types (`magic.*`, listed in [MAGIC.md](MAGIC.md#blocks)). Setup blocks keep their part of the model in `params.value` (Model Info: `name`, `description`, `author`, `license`, `source`; Text: the `text` object; Tools: the `text.tools` the program carries out; Native Tools: the `text.tools` with `native`; Thinking: `text.thinking`; Reasoning Effort: `text.effort`; Screen Area: `capture`; Answers: `classes`; When It Acts: `action`; Sliders: `settings`; Picture Preparation: `input`), and the description above is what the setup blocks chained into the start say. Files without it get a canvas built from their description. |
| `gguf` | Before 0.1.7.2: a GGUF model file the model ran with llama.cpp. MacroMagi no longer reads it. |
| `datasets` | The text datasets in the archive, by path (`data/text/…`). |
| `trained` | What training the model went through: `stages` (`pretrain`, `sft`, `rl`, `compaction`, `vision`), `steps`, the final `loss`, `parameters`. |
| `include` | Also `weights` (`model/weights.safetensors`) and `datasets`. |

A text model's `text`:

```json
"text": {
  "specials": ["<|end|>", "<|system|>", "<|user|>", "<|assistant|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<compact>", "</compact>", "<|tool|>"],
  "context": 2048,
  "system_tag": "<|system|>", "user_tag": "<|user|>", "assistant_tag": "<|assistant|>", "end_tag": "<|end|>",
  "think_tags": ["<think>", "</think>"], "tool_tags": ["<tool_call>", "</tool_call>"], "compact_tags": ["<compact>", "</compact>"],
  "tool_tag": "<|tool|>",
  "thinking": { "on_by_default": true },
  "effort": { "levels": ["low", "medium", "high"], "default": "medium" },
  "personas": [ { "name": "Gun", "tag": "<gun>", "role": "Strict code and tool execution" } ],
  "tools": [ { "name": "press", "description": "taps a key", "parameters": { "type": "object", "properties": { "key": { "type": "string" } } } } ],
  "system": "You press keys for the user.",
  "temperature": 0.7, "top_k": 40, "max_tokens": 200
}
```

Tokens are the 256 byte values of UTF-8 text (ids 0–255), then – with a learned tokenizer, `"tokenizer": {"kind": "learned", "size": 4096, "merges": [[a, b], …]}` – one token per merge (ids 256 on, in order: merge *i* joins tokens *a* and *b* into token 256 + *i*), then each of `specials` in order; special tags are matched longest first. A learned tokenizer cuts the text between tags into chunks (Python: `rb" ?[A-Za-z\x80-\xff]+| ?[0-9]| ?[^\sA-Za-z0-9\x80-\xff]+|[\r\n]+|[ \t\x0b\x0c]+(?![^\s])|[ \t\x0b\x0c]+"`) and joins each chunk's bytes by the lowest-numbered merge present, all its places at once, until none applies ([`tokenizer.rs`](../crates/magi-core/src/tokenizer.rs)). Without `tokenizer` (or with `"kind": "bytes"`), tokens are bytes. A request is `system_tag system end_tag user_tag request end_tag assistant_tag` (the system part only when there is one), and the answer ends at `end_tag`. The system part is the system prompt, then `Reasoning effort: <level>` (with `effort`), then the tools (with `tool_tag`), separated by blank lines – [MAGIC.md](MAGIC.md#how-text-is-read-and-written) shows it. With `thinking`, a request that does not think ends with an empty `think_tags` pair. Tool calls are `tool_tags.0` + `{"name": …, "arguments": {…}}` + `tool_tags.1`, and a tool's result is `tool_tag result end_tag`. `tool_tag`, `thinking` and `effort` are optional: files without `tool_tag` (made before it) keep their tokens, write calls with `"args"`, and read results as user turns. The ONNX model takes `tokens` (int64, `[1, length]`) and gives `logits` (`[1, length, vocabulary]`).

Since 0.1.6, `text.tools[].native` marks a native tool, `memory` is the Memory & Cache block's settings (`precision`, `hot_mb`, `warm_mb`, `disk_dir`, `disk_gb`, `persist_prefix`, `read_blocks`, `flash_attention`), `architecture.rope_scaling` is the Context Scaler's (`kind`: `yarn`, `ntk` or `none`; `factor`, `original`, `target`), `trained.system` is the system prompt the model was trained with, and `package` names its Magi package.

A model asset inside a `.magi` file is the same archive with `training`, `finetune` and the datasets left out, and the weights too unless the native engine runs the model.

## `.szi` – Magi models

A Magi model is **one file**: download it, open it, and it runs – native tools included. Compile Package (*Magi Model (.szi)*) writes it as `<project>/<name>.szi`, and a workspace exported as Python writes `<name>.szi` beside `run.py`. It is a zip archive:

```text
model.json            the description: JSON, below
weights.safetensors   the weights (MacroMagi's native engine runs these), stored uncompressed
model.onnx            the same model for ONNX runtimes, uncompressed (only when Compile Package asks for an ONNX file,
                      or the native engine cannot run the architecture)
nt/<tool>/tool.json   each native tool: {"name", "description", "parameters", "run": {"kind": "python" | "command" | "host", "entry", "timeout_ms"}, "confirm"?}
nt/<tool>/run.py      its program: arguments as JSON on standard input, its result as JSON on standard output
README.md             what it is
```

`model.json` is `{"format": "magi.szi", "version": 2, "app", "family", "files": {"weights", "onnx", "native_tools"}, …}` followed by the fields of `magic.json` (name, `kind`, `text`, `architecture`, `memory`, `trained`…) without the canvas and the datasets. A native tool with `"confirm": true` (it writes files or runs programs) runs only when the user allows each call; one whose `run.kind` is `host` has no program of its own – the program running the model carries it out, by its `entry` name (MacroMagi's `view_image`) – and other files in its folder are helpers it may use (`describe.py`). Tool results go back to the model as the tool printed them, keys in the tool's order. `family` is `Magi-F` (up to a billion parameters), `Magi-S` (up to thirty billion) or `Magi-T`. The weights and the ONNX model are stored, not compressed, so a reader can take them straight from the file. MacroMagi opens it with File › Open or in the AI Agent window, takes it as a base model, and CodeMagi will run it.

**Version 1** (0.1.6 to 0.1.7.1) was a folder: a `<name>.szi` JSON description – the same fields, `"version": 1` – beside `weights.safetensors`, `model.onnx` and `nt/`. MacroMagi still opens it (the folder or its `.szi` file); a workspace saved with it finds its package in `szi/` beside it.

## Shared vocabulary

**Keys** are named after the W3C `KeyboardEvent.code` values and identify *physical* keys: `KeyA`, `Digit1`, `F6`, `ShiftLeft`, `ArrowUp`, `Numpad5`, `NumpadEnter`… A key MacroMagi has no name for is written `Raw<code>` with its platform key code. The full list is the `keys!` table in [`crates/magi-core/src/input.rs`](../crates/magi-core/src/input.rs).

**Mouse buttons**: `Left`, `Right`, `Middle`, `Back`, `Forward`.

**Gamepad buttons** are named by position: `South`, `East`, `West`, `North`, `LeftBumper`, `RightBumper`, `LeftTrigger`, `RightTrigger`, `Select`, `Start`, `Guide`, `LeftStick`, `RightStick`, `DPadUp`, `DPadDown`, `DPadLeft`, `DPadRight`.

**Hotkeys** are the buttons to hold, joined by `+`: `F6`, `ControlLeft+ShiftLeft+KeyK`, `Mouse:Back`, `ControlLeft+Pad:South`. Modifier keys are written as their left-hand variant and match either side. An empty string means "no hotkey".

**Screen positions** are `{ "x": …, "y": … }`, **screen areas** `{ "x": …, "y": …, "w": …, "h": … }`, both in physical pixels. With `"display": n` they are relative to the top-left corner of display `n`; without it, to the top-left of the primary display (the desktop's origin). Displays are numbered the same way on every machine: the primary display is 0, the others follow from left to right, then top to bottom. A position on a display that is not connected falls back to the primary display.

**Durations** are milliseconds and may be fractional.

**Colors** are `#RRGGBB`.

## Versioning

`version` is a single integer per format, currently `1` for all three.

- A reader **rejects** files with a higher version than it knows, with a message asking the user to update.
- Within a version, changes are additive: new optional fields with defaults, new block types, new stroke types. Readers ignore fields they do not know. (`options`, `clicker`, `look` strokes and the extended Auto Clicker settings were added this way in 0.1.0; `variables`, bound settings, `groups`, `display` on positions and areas, the `key_down`/`key_up`/`mouse_down`/`mouse_up`/`display` strokes and `audio` assets in 0.1.1; `model` assets, image and sound lists with groups, `mode`, `detail`, `match_ms`, `device` and the variables' `reset` in 0.1.2; group `parent`, the `input.autotype` and `vision.read_text` blocks, AI Watch's `acted` output and mouse buttons as a model's key in 0.1.3; the `vision.screenshot`, `window.display`, `window.current` and `files.*` blocks, Image Detection's `picture` input and the Auto Typer's `ignore` in 0.1.4; the `ai.agent` block and the `.magic` fields `kind`, `text`, `architecture`, `studio`, `datasets`, `trained`, `gguf`, `negative` and the weights in 0.1.5; the `ai.load` and `ai.unload` blocks, AI Agent's `thinking`, `effort`, `context`, `native_tools` and `max_steps`, the `.magic` fields `package`, `memory`, `rope_scaling`, `trained.system` and `tools[].native`, `nt/` files, and `.szi` models in 0.1.6; the `magic.native_tools` block, Compile Package's `magic` and `lmstudio_quant`, and a list of levels as a dataset sample's `effort` in 0.1.7; `text.tokenizer`, `text.tool_list`, `trained.compacts`, the Optimizer block's `embed_lr`, a dataset message's `learn`, `.szi` version 2 (one file), the `magic.corpus` block, a native tool's `confirm` and `host` run kind, and the reward rules `same_tool`, `unknown_tool` and `made_up_result` in 0.1.7.2. 0.1.7.2 also stopped writing GGUF, LM Studio and Hugging Face exports: Compile Package's `gguf`, `quant`, `lmstudio`, `lmstudio_quant` and `huggingface`, the Context Scaler's `static_in_exports`, the Memory & Cache block's `flash_attention` and the model's `gguf` path are ignored when read. Files from earlier versions load unchanged; earlier versions may refuse files that use later additions.)
- A change that old readers would misread bumps the version, and the new reader migrates old files on load.
