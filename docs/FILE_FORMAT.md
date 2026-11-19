# File formats

MacroMagi has two file types: **`.magi`** (a macro, and as chosen your settings and presets) and **`.mp`** (a macro or Auto Clicker preset). A `.magi` can carry presets inside it. Both are designed to be readable, diffable and easy to produce with other tools. The `.magic` and `.szi` files of versions before 0.1.8 (AI workspaces and models) are no longer read or written; see [AI files of earlier versions](#ai-files-of-earlier-versions).

- [`.magi` – projects](#magi--projects)
- [`.mp` – presets](#mp--presets)
- [AI files of earlier versions](#ai-files-of-earlier-versions)
- [Shared vocabulary](#shared-vocabulary)
- [Versioning](#versioning)

## `.magi` – projects

A `.magi` file is a standard ZIP archive. Rename it to `.zip` to open it.

```text
project.json           required – the macro
assets/<id>.png        images
assets/<id>.json       recordings
assets/<id>.wav        sounds (for Sound Recognition)
presets/<nn>-<name>.mp bundled presets (optional)
```

`project.json` lists every other file in the archive. A reader never has to guess file names, and files not listed are ignored.

### `project.json`

```json
{
  "format": "macromagi.project",
  "version": 1,
  "app": "0.1.7",
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
| `model` | `assets/<id>.szi` or `.magic` | An AI model of a version before 0.1.8, for its AI blocks. It is kept, byte for byte, so such a macro opens and saves unchanged, but nothing runs it. |

Blocks refer to assets by `id` in their parameters (`"image": "img-…"`, `"recording": "rec-1"`, `"sound": "snd-…"`). Image and sound ids are derived from the content, so the same picture or sound is stored once.

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

### AI Vision

An AI Vision block (`ai.vision`) has `groups`: each a `name` and the ids of its example pictures (image assets). It learns from them when the macro runs; nothing it learned is saved. `region` and `display` say where it looks, as for Image Detection; `threshold` is how sure (0–1) it must be of a group for `found`.

```json
"params": {
  "groups": [
    { "name": "Menu", "images": ["img-…", "img-…"] },
    { "name": "Game", "images": ["img-…", "img-…", "img-…"] }
  ],
  "region": null,
  "display": 0,
  "threshold": 0.8
}
```

Its outputs: `found` (true/false), `group` (the group's name when found, else empty), `confidence` (0–1) and `likeliest` (the likeliest group's name, sure or not). A picture wired into its `picture` input is looked at instead of the screen.

The AI blocks of versions before 0.1.8 – `ai.watch`, `ai.classify`, `ai.agent`, `ai.load` and `ai.unload` – are kept as they are, with their settings, but are unknown blocks now: the editor shows them, and engines treat them as not runnable.

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
  "app": "0.1.7",
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
    { "id": "snd-…", "name": "Alarm", "kind": "audio", "wav_base64": "UklGRi…" }
  ]
}
```

A nodes preset may also carry `groups` whose blocks are all in it. When it is inserted, its nodes get fresh ids and its assets are imported (images and sounds are de-duplicated by content; a model of an earlier version, `"kind": "model"` with `magic_base64`, is kept as it is). Variables are not part of a preset: settings bound to a variable the receiving macro does not have use their defaults until it is created. This is also the format MacroMagi puts on the clipboard when blocks are copied, prefixed with `macromagi-nodes:`.

## AI files of earlier versions

Versions 0.1.5 to 0.1.7 had a Magic tab that built and trained AI models, with two file types of its own: **`.magic`**, a ZIP archive of a workspace (its canvas, setup, datasets and pictures – and, before 0.1.8, sometimes the model), and **`.szi`**, a ZIP archive of a trained model (`model.json`, `weights.safetensors` and its native tools). MacroMagi 0.1.8 neither reads nor writes them; both are ordinary ZIP archives whose description files are JSON, readable with any tool. A macro that embedded such a model keeps it as a `model` asset (see [Assets](#assets)).

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
