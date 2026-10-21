//! AI models, as the Magic tab prepares them and macros use them.
//!
//! A `.magic` file is a ZIP archive holding a small vision model (an ONNX
//! file) together with everything needed to use it: how to prepare a picture
//! for it, which part of the screen to show it, what its answers mean, what to
//! do when it gives one, which of those settings a person may adjust, the
//! blocks to use it with, and – optionally – the pictures it was trained or
//! fine-tuned on.
//!
//! ```text
//! magic.json                 required – the description below
//! model/model.onnx           the model (left out when "include the model" is off)
//! model/weights.safetensors  its weights, to train further (optional)
//! data/train/…               training pictures, one folder per class (optional)
//! data/tune/…                fine-tuning pictures, one folder per class (optional)
//! data/text/…                text datasets: .jsonl, .md, .txt (optional)
//! nt/<tool>/…                a model's own native tools (optional)
//! ```
//!
//! Text models trained in MacroMagi are packaged as `.szi` folders
//! ([`crate::szi`]); their `.magic` file is then the workspace – canvas,
//! setup, datasets, pictures – and points at the package (`package`) instead
//! of holding the model.
//!
//! A model is a *vision* model (a picture in, an answer out) or a *text*
//! model (text in, text out, with tools and personas). Both are made, trained
//! and tried in the Magic tab, whose canvas (`studio`) is saved in the file.
//!
//! Macros embed a model as an asset: the `.magic` file without its data.

use std::collections::BTreeMap;
use std::io::{Read, Seek, Write};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use crate::error::{Error, Result};
use crate::geom::{Displays, Rect};
use crate::graph::Fragment;
use crate::input::{Button, Key};

pub const FORMAT: &str = "macromagi.magic";
pub const FORMAT_VERSION: u32 = 1;
pub const EXTENSION: &str = "magic";
const MANIFEST: &str = "magic.json";
const MODEL_FILE: &str = "model/model.onnx";
const WEIGHTS_FILE: &str = "model/weights.safetensors";

/// The standard ImageNet color statistics most vision models are trained with.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// How the numbers of a picture are laid out in the model's input.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    /// Channels first: `[1, 3, height, width]` (PyTorch's habit).
    #[default]
    Nchw,
    /// Channels last: `[1, height, width, 3]` (TensorFlow's habit).
    Nhwc,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColorOrder {
    #[default]
    Rgb,
    Bgr,
}

/// How a picture is turned into the numbers the model reads: resized to
/// `width × height`, each color value `v` (0–255) becoming
/// `(v × scale − mean) / std`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InputSpec {
    /// The name of the model's input.
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub layout: Layout,
    pub color: ColorOrder,
    pub scale: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// The model gives raw scores: turn them into probabilities (softmax).
    pub softmax: bool,
}

impl Default for InputSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            width: 224,
            height: 224,
            layout: Layout::Nchw,
            color: ColorOrder::Rgb,
            scale: 1.0 / 255.0,
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
            softmax: true,
        }
    }
}

/// Which part of the screen the model looks at.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CaptureArea {
    /// A square in the middle of the display, as tall as this share of the
    /// display's height – so it covers the same part of the picture at any
    /// resolution.
    Center { height_share: f64 },
    /// A fixed area.
    Region { rect: Rect },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CaptureSpec {
    pub area: CaptureArea,
    /// The display a centred square is on (0 = primary).
    pub display: u8,
}

impl Default for CaptureSpec {
    fn default() -> Self {
        // 224 pixels on a 1080-pixel-high display.
        Self { area: CaptureArea::Center { height_share: 224.0 / 1080.0 }, display: 0 }
    }
}

impl CaptureSpec {
    /// The area on the desktop, given the displays. `display` overrides the
    /// spec's own display for a centred square.
    pub fn rect(&self, displays: &Displays, display: Option<u8>) -> Option<Rect> {
        match self.area {
            CaptureArea::Region { rect } => Some(displays.rect_to_global(rect)),
            CaptureArea::Center { height_share } => {
                let screen = displays.get(display.unwrap_or(self.display)).or_else(|| displays.get(0))?;
                let side = ((f64::from(screen.h) * height_share).round() as u32).clamp(8, screen.w.min(screen.h));
                Some(Rect::new(
                    screen.x + ((screen.w - side) / 2) as i32,
                    screen.y + ((screen.h - side) / 2) as i32,
                    side,
                    side,
                ))
            }
        }
    }
}

/// One answer a classifying model can give.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ClassSpec {
    pub label: String,
    /// Whether this answer calls for the action.
    pub act: bool,
    /// How sure the model must be, `0.0..=1.0`, before acting on it.
    pub threshold: f64,
    /// Milliseconds to wait before acting.
    pub delay_ms: f64,
}

/// What is done when the model gives an answer that calls for action.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ActionSpec {
    /// A key, or a mouse button. Files from before mouse buttons hold a key's name, which reads as that key.
    pub key: Button,
    /// How long the key is held.
    pub hold_ms: f64,
    /// After acting, answers are ignored for this long, so one event is not
    /// acted on twice.
    pub cooldown_ms: f64,
}

impl Default for ActionSpec {
    fn default() -> Self {
        Self { key: Button::Key(Key::Space), hold_ms: 5.0, cooldown_ms: 500.0 }
    }
}

/// What an adjustable setting changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SettingTarget {
    /// A class's threshold; the setting is a percentage.
    Threshold {
        class: usize,
    },
    /// A class's delay, in milliseconds.
    Delay {
        class: usize,
    },
    Cooldown,
    Hold,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SettingKind {
    #[default]
    Percent,
    Milliseconds,
}

/// A setting a person may adjust – in the Magic tab, or per block in a macro –
/// that changes one or more values of the model's configuration at once.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelSetting {
    pub key: String,
    pub label: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub help: String,
    pub kind: SettingKind,
    pub min: f64,
    pub max: f64,
    pub default: f64,
    pub value: f64,
    pub targets: Vec<SettingTarget>,
}

/// A picture kept for training or fine-tuning, with the class it shows.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DataItem {
    /// Its path inside the archive.
    pub file: String,
    pub class: usize,
    /// A picture of what the class is *not* (a false positive to unlearn).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub negative: bool,
}

/// What a model takes in and gives out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    /// A picture in, an answer (a class) out.
    #[default]
    Vision,
    /// Text in, text out – with tools and personas: an agent.
    Text,
}

impl ModelKind {
    pub fn label(self) -> &'static str {
        match self {
            ModelKind::Vision => "Vision",
            ModelKind::Text => "Text / agent",
        }
    }
}

/// One voice of a text model's reasoning, marked by its own tag.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Persona {
    pub name: String,
    /// The tag that opens its turn, such as `<gun>`; `</gun>` closes it.
    pub tag: String,
    /// What it is for.
    pub role: String,
}

/// A tool a text model may call.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// Its arguments, as a JSON Schema object.
    pub parameters: Value,
    /// A native tool: one the model's `nt` folder holds, which MacroMagi runs
    /// itself and answers ([`crate::native_tools`]).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub native: bool,
}

/// A tool call found in a text model's answer:
/// `<tool_call>{"name": …, "arguments": {…}}</tool_call>` (older models write `"args"`).
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub args: Value,
}

/// A text model that can be asked to think or not. Not thinking is an
/// empty thinking span the answer starts with.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ThinkingSpec {
    /// Think unless the request says not to.
    pub on_by_default: bool,
}

impl Default for ThinkingSpec {
    fn default() -> Self {
        Self { on_by_default: true }
    }
}

/// How hard a text model reasons: the levels it was trained with – by default
/// none, low, medium and high – named in the system text as `Reasoning effort:
/// high`. It is apart from thinking: a model can think briefly at effort none,
/// and not think at all when thinking is off, whatever the effort.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EffortSpec {
    pub levels: Vec<String>,
    pub default: String,
}

impl Default for EffortSpec {
    fn default() -> Self {
        Self { levels: vec!["none".into(), "low".into(), "medium".into(), "high".into()], default: "none".into() }
    }
}

impl EffortSpec {
    /// The level a request asks for, or the default.
    pub fn level<'a>(&'a self, asked: &'a str) -> &'a str {
        if self.levels.iter().any(|l| l == asked) { asked } else { &self.default }
    }
}

/// How a text model cuts text into tokens.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TokenizerSpec {
    /// `bytes` (or empty): every byte is a token.
    /// `learned`: pieces learned from the datasets ([`crate::tokenizer`]) –
    /// about a third as many tokens, read by MacroMagi only.
    pub kind: String,
    /// A learned tokenizer's size: bytes, pieces and tags together.
    pub size: u32,
    /// The learned pieces, in order: each joins two tokens into one, whose id
    /// is 256 + its place. Learned when training starts from zero.
    pub merges: Vec<(u32, u32)>,
}

impl TokenizerSpec {
    /// A learned tokenizer of about `size` tokens (its pieces learned at the first training).
    pub fn learned_of(size: u32) -> Self {
        Self { kind: "learned".into(), size, merges: Vec::new() }
    }

    pub fn learned(&self) -> bool {
        self.kind == "learned"
    }

    /// Bytes as tokens: what every model was before learned tokenizers.
    pub fn is_bytes(&self) -> bool {
        !self.learned() && self.merges.is_empty()
    }
}

/// JSON the way Python's `json.dumps(…, sort_keys=True)` writes it (", " and
/// ": ", keys in order), so prompts written here and in training agree.
/// A copy with every object's keys in alphabetical order.
pub fn sorted_keys(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            Value::Object(keys.into_iter().map(|k| (k.clone(), sorted_keys(&map[k]))).collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted_keys).collect()),
        other => other.clone(),
    }
}

pub fn spaced_json(value: &Value) -> String {
    match value {
        Value::Array(items) => format!("[{}]", items.iter().map(spaced_json).collect::<Vec<_>>().join(", ")),
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(k, v)| format!("{}: {}", Value::String(k.clone()), spaced_json(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => other.to_string(),
    }
}

/// How a text model reads and writes: its tokens, its chat tags, its personas
/// and tools, and how it generates.
///
/// Tokens are the 256 byte values of UTF-8 text, then the special tags in
/// `specials`, in order – simple, and the same in MacroMagi and in training.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TextSpec {
    /// Special tags, each one token (ids after the bytes and the learned pieces, in this order).
    pub specials: Vec<String>,
    /// How text is cut into tokens: bytes, or pieces learned from the datasets.
    #[serde(skip_serializing_if = "TokenizerSpec::is_bytes")]
    pub tokenizer: TokenizerSpec,
    /// The longest text, in tokens, the model reads at once.
    pub context: u32,
    pub system_tag: String,
    pub user_tag: String,
    pub assistant_tag: String,
    /// Ends a turn; generation stops at it.
    pub end_tag: String,
    pub think_tags: (String, String),
    /// The one way of writing a tool call: `<tool_call>{"name": …, "arguments": {…}}</tool_call>`.
    pub tool_tags: (String, String),
    /// Opens a turn with a tool's result. Models with one also read the
    /// tools they may use in their system text (see [`TextSpec::system_text`]);
    /// models made before it have none and read results as user turns.
    #[serde(default)]
    pub tool_tag: String,
    /// Whether the model can be asked not to think (and thinks unless asked).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingSpec>,
    /// Levels of reasoning effort the model knows, named in its system text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<EffortSpec>,
    pub compact_tags: (String, String),
    pub personas: Vec<Persona>,
    pub tools: Vec<ToolSpec>,
    /// How the system text lists the tools: `short` (a line each: `name(argument, other?): what it
    /// does`) or, empty, each with its JSON parameters – as models made before 0.1.7.2 read them.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub tool_list: String,
    /// What the model is told before every request.
    pub system: String,
    pub temperature: f32,
    pub top_k: u32,
    pub max_tokens: u32,
}

impl Default for TextSpec {
    fn default() -> Self {
        let mut spec = Self {
            specials: Vec::new(),
            tokenizer: TokenizerSpec::default(),
            context: 256,
            system_tag: "<|system|>".into(),
            user_tag: "<|user|>".into(),
            assistant_tag: "<|assistant|>".into(),
            end_tag: "<|end|>".into(),
            think_tags: ("<think>".into(), "</think>".into()),
            tool_tags: ("<tool_call>".into(), "</tool_call>".into()),
            tool_tag: "<|tool|>".into(),
            thinking: None,
            effort: None,
            compact_tags: ("<compact>".into(), "</compact>".into()),
            personas: Vec::new(),
            tools: Vec::new(),
            tool_list: String::new(),
            system: String::new(),
            temperature: 0.7,
            top_k: 40,
            max_tokens: 200,
        };
        spec.ensure_specials();
        spec
    }
}

impl TextSpec {
    /// Every tag the template and the personas use is a special token.
    pub fn ensure_specials(&mut self) {
        let mut wanted = vec![
            self.end_tag.clone(),
            self.system_tag.clone(),
            self.user_tag.clone(),
            self.assistant_tag.clone(),
            self.think_tags.0.clone(),
            self.think_tags.1.clone(),
            self.tool_tags.0.clone(),
            self.tool_tags.1.clone(),
            self.compact_tags.0.clone(),
            self.compact_tags.1.clone(),
            self.tool_tag.clone(),
        ];
        for persona in &self.personas {
            let name = persona.tag.trim_start_matches('<').trim_end_matches('>');
            if !name.is_empty() {
                wanted.push(format!("<{name}>"));
                wanted.push(format!("</{name}>"));
            }
        }
        for tag in wanted {
            if !tag.is_empty() && !self.specials.contains(&tag) {
                self.specials.push(tag);
            }
        }
    }

    pub fn vocab_size(&self) -> usize {
        256 + self.tokenizer.merges.len() + self.specials.len()
    }

    /// Where the special tags' ids start: after the bytes and the learned pieces.
    fn tags_from(&self) -> u32 {
        256 + self.tokenizer.merges.len() as u32
    }

    /// Text as token ids: special tags first (longest match), the rest as
    /// UTF-8 bytes – joined into learned pieces, with a learned tokenizer.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut order: Vec<(usize, &String)> = self.specials.iter().enumerate().collect();
        order.sort_by_key(|(_, tag)| std::cmp::Reverse(tag.len()));
        let bpe = (!self.tokenizer.merges.is_empty()).then(|| crate::tokenizer::shared(&self.tokenizer.merges));
        let tags_from = self.tags_from();
        let bytes = text.as_bytes();
        let mut ids = Vec::with_capacity(bytes.len());
        let plain = |ids: &mut Vec<u32>, part: &[u8]| match &bpe {
            Some(bpe) => ids.extend(bpe.encode(part)),
            None => ids.extend(part.iter().map(|b| u32::from(*b))),
        };
        let (mut i, mut start) = (0, 0);
        'outer: while i < bytes.len() {
            if bytes[i] == b'<' {
                for (index, tag) in &order {
                    if !tag.is_empty() && bytes[i..].starts_with(tag.as_bytes()) {
                        plain(&mut ids, &bytes[start..i]);
                        ids.push(tags_from + *index as u32);
                        i += tag.len();
                        start = i;
                        continue 'outer;
                    }
                }
            }
            i += 1;
        }
        plain(&mut ids, &bytes[start..]);
        ids
    }

    /// Token ids as text. Unknown ids are left out.
    pub fn decode(&self, ids: &[u32]) -> String {
        let bpe = (!self.tokenizer.merges.is_empty()).then(|| crate::tokenizer::shared(&self.tokenizer.merges));
        let tags_from = self.tags_from();
        let mut bytes = Vec::with_capacity(ids.len() * 3);
        for &id in ids {
            match id {
                0..=255 => bytes.push(id as u8),
                _ if id < tags_from => {
                    if let Some(piece) = bpe.as_ref().and_then(|b| b.bytes(id)) {
                        bytes.extend_from_slice(piece);
                    }
                }
                _ => {
                    if let Some(tag) = self.specials.get((id - tags_from) as usize) {
                        bytes.extend_from_slice(tag.as_bytes());
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// The id of a special tag.
    pub fn special_id(&self, tag: &str) -> Option<u32> {
        self.specials.iter().position(|s| s == tag).map(|i| self.tags_from() + i as u32)
    }

    /// Learns the pieces of a learned tokenizer from texts – what the model
    /// will train on – and keeps them. Returns how many it learned.
    pub fn learn_tokens<'a>(&mut self, texts: impl IntoIterator<Item = &'a str>) -> usize {
        if !self.tokenizer.learned() {
            return 0;
        }
        self.ensure_specials();
        // The system text and the tools are in nearly every request: they are learned too.
        let system = self.system_text("", "", &self.tools);
        let size = (self.tokenizer.size as usize).clamp(512, 65_536);
        let count = size.saturating_sub(256 + self.specials.len());
        let tags = self.specials.clone();
        let mut all: Vec<&str> = texts.into_iter().collect();
        all.extend(std::iter::repeat_n(system.as_str(), 50));
        let merges = crate::tokenizer::learn(all, &tags, count);
        self.tokenizer.merges = merges;
        self.tokenizer.merges.len()
    }

    /// Tags the model writes as text (thinking, tool calls, personas), which
    /// chat programs show; the turn tags are control tokens they hide.
    pub fn is_text_tag(&self, tag: &str) -> bool {
        ![&self.end_tag, &self.system_tag, &self.user_tag, &self.assistant_tag, &self.tool_tag]
            .iter()
            .any(|t| *t == tag)
    }

    /// The tools a model may use, as its system text lists them.
    pub fn tools_text(&self, tools: &[ToolSpec]) -> String {
        let mut text = "You can use these tools:".to_owned();
        for tool in tools {
            if self.tool_list == "short" {
                text.push_str(&format!(
                    "\n- {}({}): {}",
                    tool.name,
                    Self::signature(&tool.parameters),
                    tool.description.trim()
                ));
            } else {
                text.push_str(&format!(
                    "\n- {}: {} Parameters: {}",
                    tool.name,
                    tool.description.trim(),
                    // As models before the short list were trained: keys in alphabetical order.
                    spaced_json(&sorted_keys(&tool.parameters))
                ));
            }
        }
        text.push_str(&format!(
            "\nTo use one, write {}{{\"name\": \"<tool name>\", \"arguments\": {{...}}}}{}. Its result comes back in the next turn.",
            self.tool_tags.0, self.tool_tags.1
        ));
        text
    }

    /// A tool's arguments, short: their names in order, `?` after an optional one, and the type
    /// when it is not text – `expression`, `path, language?`, `x: number, y: number`.
    pub fn signature(parameters: &Value) -> String {
        let required: Vec<&str> = parameters
            .get("required")
            .and_then(Value::as_array)
            .map(|r| r.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let Some(properties) = parameters.get("properties").and_then(Value::as_object) else { return String::new() };
        // In the order the tool declares them, as the trainer writes them (tokens.py, signature).
        properties
            .keys()
            .map(|name| {
                let kind = properties[name].get("type").and_then(Value::as_str).unwrap_or("string");
                let optional = if required.contains(&name.as_str()) { "" } else { "?" };
                if kind == "string" { format!("{name}{optional}") } else { format!("{name}{optional}: {kind}") }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The system text of a request: `system` (the model's own when empty),
    /// the reasoning effort and the tools, separated by blank lines.
    pub fn system_text(&self, system: &str, effort: &str, tools: &[ToolSpec]) -> String {
        let mut parts = Vec::new();
        let system = if system.trim().is_empty() { self.system.trim() } else { system.trim() };
        if !system.is_empty() {
            parts.push(system.to_owned());
        }
        if let Some(spec) = &self.effort {
            parts.push(format!("Reasoning effort: {}", spec.level(effort)));
        }
        if !self.tool_tag.is_empty() && !tools.is_empty() {
            parts.push(self.tools_text(tools));
        }
        parts.join("\n\n")
    }

    /// Whether a request thinks: as asked, else as the model does by default.
    pub fn thinks(&self, asked: Option<bool>) -> bool {
        match &self.thinking {
            Some(spec) => asked.unwrap_or(spec.on_by_default),
            None => true,
        }
    }

    /// A request as the model reads it: the system text, the request, and the
    /// tag after which it answers.
    pub fn prompt(&self, user: &str) -> String {
        self.prompt_with(user, None, "")
    }

    /// A request as the model reads it, thinking or not and at an effort level
    /// (`None` and empty follow the model). The model's tools are listed.
    pub fn prompt_with(&self, user: &str, thinking: Option<bool>, effort: &str) -> String {
        let mut text = String::new();
        let system = self.system_text("", effort, &self.tools);
        if !system.is_empty() {
            text.push_str(&format!("{}{}{}", self.system_tag, system, self.end_tag));
        }
        text.push_str(&format!("{}{}{}{}", self.user_tag, user, self.end_tag, self.assistant_tag));
        if self.thinking.is_some() && !self.thinks(thinking) {
            text.push_str(&self.think_tags.0);
            text.push_str(&self.think_tags.1);
        }
        text
    }

    /// A conversation as the model reads it, ready for its next answer: the
    /// system text (`system`, the model's own when empty, with the effort and
    /// the tools), then each turn – `("user" | "assistant" | "tool", text)`,
    /// an answer as the model wrote it (its tool calls included, its
    /// reasoning left out, as the chat template does) – and the tag after
    /// which it answers.
    pub fn chat_prompt(
        &self,
        system: &str,
        turns: &[(String, String)],
        thinking: Option<bool>,
        effort: &str,
    ) -> String {
        let mut text = String::new();
        let head = self.system_text(system, effort, &self.tools);
        if !head.is_empty() {
            text.push_str(&format!("{}{head}{}", self.system_tag, self.end_tag));
        }
        for (role, content) in turns {
            match role.as_str() {
                "assistant" => {
                    let content = match content.rfind(&self.think_tags.1) {
                        Some(at) if !self.think_tags.1.is_empty() => {
                            content[at + self.think_tags.1.len()..].trim_start()
                        }
                        _ => content.as_str(),
                    };
                    text.push_str(&format!("{}{content}{}", self.assistant_tag, self.end_tag));
                }
                "tool" if !self.tool_tag.is_empty() => {
                    text.push_str(&format!("{}{content}{}", self.tool_tag, self.end_tag))
                }
                "tool" => text.push_str(&format!("{}Tool result: {content}{}", self.user_tag, self.end_tag)),
                _ => text.push_str(&format!("{}{content}{}", self.user_tag, self.end_tag)),
            }
        }
        text.push_str(&self.assistant_tag);
        if self.thinking.is_some() && !self.thinks(thinking) {
            text.push_str(&self.think_tags.0);
            text.push_str(&self.think_tags.1);
        }
        text
    }

    /// The tool calls in an answer, in order. Calls that are not well formed are left out.
    pub fn tool_calls(&self, text: &str) -> Vec<ToolCall> {
        let (open, close) = (&self.tool_tags.0, &self.tool_tags.1);
        let mut calls = Vec::new();
        let mut rest = text;
        while let Some(start) = rest.find(open.as_str()) {
            let after = &rest[start + open.len()..];
            let Some(end) = after.find(close.as_str()) else { break };
            if let Ok(Value::Object(call)) = serde_json::from_str::<Value>(after[..end].trim())
                && let Some(name) = call.get("name").and_then(Value::as_str)
            {
                let args = call.get("arguments").or_else(|| call.get("args")).cloned().unwrap_or(Value::Null);
                // Some programs send the arguments as a JSON string.
                let args = match args {
                    Value::String(text) => serde_json::from_str(&text).unwrap_or(Value::String(text)),
                    other => other,
                };
                calls.push(ToolCall { name: name.to_owned(), args });
            }
            rest = &after[end + close.len()..];
        }
        calls
    }

    /// What an answer thought: the text inside its thinking tags (and its personas' tags).
    pub fn thinking(&self, text: &str) -> String {
        let mut tags = vec![self.think_tags.clone()];
        for persona in &self.personas {
            let name = persona.tag.trim_start_matches('<').trim_end_matches('>');
            if !name.is_empty() {
                tags.push((format!("<{name}>"), format!("</{name}>")));
            }
        }
        let mut parts = Vec::new();
        for (open, close) in &tags {
            let mut rest = text;
            while let Some(start) = rest.find(open.as_str()) {
                let after = &rest[start + open.len()..];
                let end = after.find(close.as_str()).unwrap_or(after.len());
                let part = after[..end].trim();
                if !part.is_empty() {
                    parts.push(part.to_owned());
                }
                rest = &after[(end + close.len()).min(after.len())..];
            }
        }
        parts.join("\n")
    }

    /// The answer without its thinking and tool calls: what it says.
    pub fn visible(&self, text: &str) -> String {
        let mut out = text.to_owned();
        let mut spans = vec![self.think_tags.clone(), self.tool_tags.clone(), self.compact_tags.clone()];
        // A persona's turn is reasoning, too.
        for persona in &self.personas {
            let name = persona.tag.trim_start_matches('<').trim_end_matches('>');
            if !name.is_empty() {
                spans.push((format!("<{name}>"), format!("</{name}>")));
            }
        }
        for (open, close) in &spans {
            while let (Some(a), Some(b)) = (out.find(open.as_str()), out.find(close.as_str())) {
                if b < a {
                    break;
                }
                out.replace_range(a..b + close.len(), "");
            }
        }
        out.replace(&self.end_tag, "").trim().to_owned()
    }
}

/// One block of a model's architecture, as the Magic tab's canvas put it
/// together: the block's type (`magic.layers`, `magic.csa`…), its settings,
/// and the feed-forward block wired into it, if any.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ArchBlock {
    pub kind: String,
    pub params: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ffn: Option<Box<ArchBlock>>,
}

/// The shape of a model trained in MacroMagi, as the trainer builds it
/// (`runtime/python/magi_train/blocks.py`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Architecture {
    /// Numbers per token.
    pub width: u32,
    pub heads: u32,
    /// Key/value heads (grouped-query attention); equal to `heads` for plain attention.
    pub kv_heads: u32,
    /// The longest text, in tokens (text), or the side of the pictures (vision).
    pub context: u32,
    /// `rope` (rotary) or `learned`.
    pub positions: String,
    pub rope_theta: f64,
    /// Manifold-constrained hyper-connections: `{"streams": n, "sinkhorn": iterations}`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyper_connections: Option<Value>,
    /// Multi-token prediction while training: `{"weight": w}`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtp: Option<Value>,
    pub prefix_split: bool,
    /// The output head shares the embedding table.
    pub tied: bool,
    /// Embeddings are scaled by √width (Gemma).
    pub embed_scale: bool,
    /// How positions stretch past the trained context when the model runs
    /// (the Context Scaler block); trained without it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rope_scaling: Option<RopeScaling>,
    /// One per layer: `{"attention": {"kind": …, …}, "ffn": {"kind": …, …}, "norm": "pre" | "sandwich"}`.
    pub layers: Vec<Value>,
    /// Vision models: the convolutions, `{"layers": n, "channels": c}`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conv: Option<Value>,
    /// The canvas blocks it was made from, in order, to rebuild them.
    pub blocks: Vec<ArchBlock>,
}

impl Default for Architecture {
    fn default() -> Self {
        Self {
            width: 128,
            heads: 4,
            kv_heads: 4,
            context: 256,
            positions: "rope".into(),
            rope_theta: 10000.0,
            hyper_connections: None,
            mtp: None,
            prefix_split: false,
            tied: false,
            embed_scale: false,
            rope_scaling: None,
            layers: Vec::new(),
            conv: None,
            blocks: Vec::new(),
        }
    }
}

/// Running a model past the context it was trained with: rotary positions
/// stretched by `factor`, from `original` (the trained context) to `target`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RopeScaling {
    /// `yarn` (YaRN: low frequencies stretched, high ones kept, attention
    /// sharpened by 0.1·ln(factor)+1) or `ntk` (the rotary base raised).
    pub kind: String,
    pub factor: f64,
    pub original: u32,
    pub target: u32,
}

impl Default for RopeScaling {
    fn default() -> Self {
        Self { kind: "yarn".into(), factor: 32.0, original: 8192, target: 262_144 }
    }
}

impl RopeScaling {
    /// YaRN's attention temperature: how much sharper attention gets.
    pub fn attention_factor(&self) -> f64 {
        if self.kind == "yarn" && self.factor > 1.0 { 0.1 * self.factor.ln() + 1.0 } else { 1.0 }
    }
}

/// Where a text model keeps its attention memory (the keys and values of the
/// text it has read) while it runs: a hot tier at full precision, a warm tier
/// in system memory at `precision`, and a cold tier in files on a disk. The
/// system text's memory can be kept on disk, so it is never read again.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MemorySpec {
    /// `f32`, `f16` or `q8_0`: how the warm tier keeps the cache.
    pub precision: String,
    /// The hot tier: the first block (its attention sinks) and the latest blocks, at full precision.
    pub hot_mb: u32,
    /// The warm tier, in system memory.
    pub warm_mb: u32,
    /// The cold tier's folder; empty: MacroMagi's data folder.
    pub disk_dir: String,
    pub disk_gb: u32,
    /// Keep the system text's memory on disk, found again by its SHA-256.
    pub persist_prefix: bool,
    /// Blocks of 256 tokens each new token reads (the first, the latest and
    /// the most relevant); 0: all of them, exactly.
    pub read_blocks: u32,
}

impl Default for MemorySpec {
    fn default() -> Self {
        Self {
            precision: "f16".into(),
            hot_mb: 512,
            warm_mb: 4096,
            disk_dir: String::new(),
            disk_gb: 16,
            persist_prefix: true,
            read_blocks: 0,
        }
    }
}

impl Architecture {
    /// Why MacroMagi's native engine (with its cache) cannot run the model,
    /// or `None` when it can: it runs Transformer Layers with rotary positions.
    pub fn native_problem(&self) -> Option<&'static str> {
        if self.conv.is_some() {
            return Some("it is a vision model");
        }
        if self.positions != "rope" {
            return Some("it has learned positions");
        }
        if self.hyper_connections.is_some() || self.prefix_split {
            return Some("it has hyper-connections or an encoder/decoder split");
        }
        if self.layers.is_empty() {
            return Some("it has no layers");
        }
        for layer in &self.layers {
            if layer.pointer("/attention/kind").and_then(Value::as_str).unwrap_or("full") != "full" {
                return Some("it has layers other than Transformer Layers");
            }
            if !matches!(layer.pointer("/ffn/kind").and_then(Value::as_str).unwrap_or("dense"), "dense" | "moe") {
                return Some("it has persona routing");
            }
        }
        None
    }

    /// The local window of a model with one (Gemma 3), in tokens.
    pub fn local_window(&self) -> Option<u64> {
        self.layers.iter().filter_map(|l| l.pointer("/attention/window").and_then(Value::as_u64)).find(|w| *w > 0)
    }
}

/// What training a model went through, for the record.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingRecord {
    /// The stages run, in order (`pretrain`, `sft`, `rl`…).
    pub stages: Vec<String>,
    pub steps: u64,
    /// The loss at the end of the last stage.
    pub loss: f64,
    /// The size of the model, in parameters.
    pub parameters: u64,
    /// The system prompt it was trained with.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Whether it learned to write a conversation short (compaction samples).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub compacts: bool,
}

impl TrainingRecord {
    /// Whether the model can write a long conversation short: it learned
    /// compaction samples, with fine-tuning or in a Context Compaction stage.
    pub fn can_compact(&self) -> bool {
        self.compacts || self.stages.iter().any(|s| s == "compaction")
    }
}

/// What a `.magic` file carries besides its description.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Includes {
    pub model: bool,
    pub blocks: bool,
    pub training: bool,
    pub finetune: bool,
    /// The weights to train further (`model/weights.safetensors`).
    pub weights: bool,
    /// Text datasets (`data/text/…`).
    pub datasets: bool,
}

impl Default for Includes {
    fn default() -> Self {
        Self { model: true, blocks: true, training: true, finetune: true, weights: true, datasets: true }
    }
}

/// The description of a model: `magic.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MagicModel {
    pub kind: ModelKind,
    pub name: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub author: String,
    /// Where the model came from, and its licence, as found when importing.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub source: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub license: String,
    pub input: InputSpec,
    pub capture: CaptureSpec,
    pub classes: Vec<ClassSpec>,
    pub action: ActionSpec,
    pub settings: Vec<ModelSetting>,
    /// Blocks to use the model with, ready to add to a macro.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocks: Option<Fragment>,
    pub include: Includes,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub training: Vec<DataItem>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub finetune: Vec<DataItem>,
    /// How a text model reads, writes and calls tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextSpec>,
    /// The shape of a model trained in MacroMagi.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<Architecture>,
    /// The Magic tab's canvas: the blocks that set up, train and try the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub studio: Option<Fragment>,
    /// Text datasets in the file (`data/text/…`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub datasets: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trained: Option<TrainingRecord>,
    /// The `.szi` package that holds the model (its weights and ONNX file):
    /// set when the model was compiled, and the `.magic` file is a workspace.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package: Option<String>,
    /// Where its attention memory is kept while it runs (the Memory & Cache block).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<MemorySpec>,
}

impl Default for MagicModel {
    fn default() -> Self {
        Self {
            kind: ModelKind::Vision,
            name: "New model".into(),
            description: String::new(),
            author: String::new(),
            source: String::new(),
            license: String::new(),
            input: InputSpec::default(),
            capture: CaptureSpec::default(),
            classes: Vec::new(),
            action: ActionSpec::default(),
            settings: Vec::new(),
            blocks: None,
            include: Includes::default(),
            training: Vec::new(),
            finetune: Vec::new(),
            text: None,
            architecture: None,
            studio: None,
            datasets: Vec::new(),
            trained: None,
            package: None,
            memory: None,
        }
    }
}

/// What a model's answer means, after its configuration is applied.
#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    pub class: usize,
    pub label: String,
    /// The probability of that class, `0.0..=1.0`.
    pub confidence: f64,
    /// The answer calls for the action, and the model is sure enough.
    pub act: bool,
    pub delay_ms: f64,
}

impl MagicModel {
    /// The configuration with the adjustable settings applied: each setting's
    /// own value, or the one in `overrides` (by key) when it has one.
    pub fn configured(&self, overrides: &Map<String, Value>) -> MagicModel {
        let mut out = self.clone();
        for setting in &self.settings {
            let value = overrides.get(&setting.key).and_then(Value::as_f64).unwrap_or(setting.value);
            let value = value.clamp(setting.min.min(setting.max), setting.max.max(setting.min));
            for target in &setting.targets {
                match *target {
                    SettingTarget::Threshold { class } => {
                        if let Some(c) = out.classes.get_mut(class) {
                            c.threshold = match setting.kind {
                                SettingKind::Percent => value / 100.0,
                                SettingKind::Milliseconds => value,
                            };
                        }
                    }
                    SettingTarget::Delay { class } => {
                        if let Some(c) = out.classes.get_mut(class) {
                            c.delay_ms = value;
                        }
                    }
                    SettingTarget::Cooldown => out.action.cooldown_ms = value,
                    SettingTarget::Hold => out.action.hold_ms = value,
                }
            }
        }
        out
    }

    /// What the probabilities a model gave mean: the likeliest class, and
    /// whether to act on it.
    pub fn decide(&self, probabilities: &[f32]) -> Decision {
        let (class, confidence) = probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or((0, 0.0), |(i, p)| (i, f64::from(*p)));
        let spec = self.classes.get(class);
        Decision {
            class,
            label: spec.map_or_else(|| format!("Class {class}"), |c| c.label.clone()),
            confidence,
            act: spec.is_some_and(|c| c.act && confidence >= c.threshold),
            delay_ms: spec.map_or(0.0, |c| c.delay_ms),
        }
    }

    /// The label of a class.
    pub fn label(&self, class: usize) -> String {
        self.classes.get(class).map_or_else(|| format!("Class {class}"), |c| c.label.clone())
    }
}

/// A whole `.magic` file in memory.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MagicFile {
    pub model: MagicModel,
    /// The ONNX model. Empty when the file does not include it.
    pub onnx: Vec<u8>,
    /// Training and fine-tuning pictures and text datasets, by path inside the archive.
    pub data: BTreeMap<String, Vec<u8>>,
    /// The weights (safetensors), to train the model further. Empty when there are none.
    pub weights: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    format: String,
    version: u32,
    #[serde(default)]
    app: String,
    #[serde(flatten)]
    model: MagicModel,
}

impl MagicFile {
    pub fn write_to<W: Write + Seek>(&self, writer: W) -> Result<()> {
        let mut zip = ZipWriter::new(writer);
        let text = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        let stored = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
        let mut model = self.model.clone();
        let include = model.include;
        if include.model && !self.onnx.is_empty() {
            zip.start_file(MODEL_FILE, text)?;
            zip.write_all(&self.onnx)?;
        }
        if include.weights && !self.weights.is_empty() {
            zip.start_file(WEIGHTS_FILE, stored)?;
            zip.write_all(&self.weights)?;
        }
        if !include.blocks {
            model.blocks = None;
        }
        model.datasets.retain(|file| include.datasets && self.data.contains_key(file));
        for file in &model.datasets {
            zip.start_file(file.as_str(), text)?;
            zip.write_all(&self.data[file])?;
        }
        // The model's own native tools.
        for (path, bytes) in self.data.iter().filter(|(path, _)| path.starts_with("nt/")) {
            zip.start_file(path.as_str(), text)?;
            zip.write_all(bytes)?;
        }
        let keep = |item: &DataItem, on: bool| on && self.data.contains_key(&item.file);
        model.training.retain(|item| keep(item, include.training));
        model.finetune.retain(|item| keep(item, include.finetune));
        for item in model.training.iter().chain(&model.finetune) {
            zip.start_file(item.file.as_str(), stored)?;
            zip.write_all(&self.data[&item.file])?;
        }
        let manifest = Manifest { format: FORMAT.into(), version: FORMAT_VERSION, app: crate::VERSION.into(), model };
        zip.start_file(MANIFEST, text)?;
        zip.write_all(serde_json::to_string_pretty(&manifest)?.as_bytes())?;
        zip.finish()?;
        Ok(())
    }

    pub fn read_from<R: Read + Seek>(reader: R) -> Result<Self> {
        let mut zip =
            ZipArchive::new(reader).map_err(|_| Error::format("this is not a MacroMagi model (.magic) file"))?;
        let manifest = read(&mut zip, MANIFEST)
            .map_err(|_| Error::format("this archive has no magic.json; it is not a MacroMagi model"))?;
        let header: Value = serde_json::from_slice(&manifest)?;
        if header.get("format").and_then(Value::as_str) != Some(FORMAT) {
            return Err(Error::format("magic.json is not a MacroMagi model description"));
        }
        let version = header.get("version").and_then(Value::as_u64).unwrap_or(0);
        if version > u64::from(FORMAT_VERSION) {
            return Err(Error::format(format!(
                "this model was saved by a newer MacroMagi (model format {version}); please update"
            )));
        }
        let manifest: Manifest = serde_json::from_value(header)?;
        let onnx = read(&mut zip, MODEL_FILE).unwrap_or_default();
        let mut data = BTreeMap::new();
        let files = manifest.model.training.iter().chain(&manifest.model.finetune).map(|item| &item.file);
        for file in files.chain(&manifest.model.datasets) {
            if let Ok(bytes) = read(&mut zip, file) {
                data.insert(file.clone(), bytes);
            }
        }
        let tools: Vec<String> = zip
            .file_names()
            .filter_map(|name| name.ok().map(|n| n.into_owned()))
            .filter(|n| n.starts_with("nt/") && !n.ends_with('/'))
            .collect();
        for name in tools {
            if let Ok(bytes) = read(&mut zip, &name) {
                data.insert(name, bytes);
            }
        }
        let weights = read(&mut zip, WEIGHTS_FILE).unwrap_or_default();
        Ok(Self { model: manifest.model, onnx, data, weights })
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut out = std::io::Cursor::new(Vec::new());
        self.write_to(&mut out)?;
        Ok(out.into_inner())
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::read_from(std::io::Cursor::new(bytes))
    }

    /// The file as a macro keeps it: the model and its configuration, without
    /// the training data.
    pub fn for_macro(&self) -> Result<Vec<u8>> {
        let mut copy = self.clone();
        // Text models the native engine runs go with their weights alone: it runs them from those, fast.
        let native = self.model.architecture.as_ref().is_some_and(|a| a.native_problem().is_none());
        let weights_alone = native && !self.weights.is_empty();
        copy.model.include = Includes {
            model: !weights_alone,
            blocks: true,
            training: false,
            finetune: false,
            weights: native,
            datasets: false,
        };
        // The canvas is for making the model, not for using it.
        copy.model.studio = None;
        copy.to_bytes()
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        Ok(std::fs::write(path, self.to_bytes()?)?)
    }

    pub fn load(path: &std::path::Path) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }
}

fn read<R: Read + Seek>(zip: &mut ZipArchive<R>, name: &str) -> Result<Vec<u8>> {
    let mut entry = zip.by_name(name).map_err(|_| Error::format(format!("`{name}` is missing")))?;
    let mut bytes = Vec::with_capacity(entry.size() as usize);
    entry.read_to_end(&mut bytes)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_are_listed_short_when_the_model_reads_them_so() {
        let spec = TextSpec {
            tool_list: "short".into(),
            tools: vec![
                ToolSpec {
                    name: "web_search".into(),
                    description: "Searches the web.".into(),
                    parameters: serde_json::json!({"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
                    native: true,
                },
                ToolSpec {
                    name: "click".into(),
                    description: "Clicks.".into(),
                    parameters: serde_json::json!({"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "button": {"type": "string"}}, "required": ["x", "y"]}),
                    native: false,
                },
            ],
            ..TextSpec::default()
        };
        let text = spec.tools_text(&spec.tools);
        assert!(
            text.starts_with(
                "You can use these tools:
- web_search(query): Searches the web.
- click(x: number, y: number, button?): Clicks.
To use one"
            ),
            "{text}"
        );
        if let Some(out) = std::env::var_os("MAGI_TEST_TOOLS_OUT") {
            std::fs::write(out, serde_json::to_vec(&serde_json::json!({ "spec": spec, "text": text })).unwrap())
                .unwrap();
        }
    }

    #[test]
    fn a_chat_reads_as_its_requests_do() {
        let mut spec = TextSpec {
            thinking: Some(ThinkingSpec { on_by_default: false }),
            effort: Some(EffortSpec::default()),
            ..TextSpec::default()
        };
        spec.ensure_specials();
        let one = spec.chat_prompt("", &[("user".into(), "Hi".into())], None, "high");
        assert_eq!(one, spec.prompt_with("Hi", None, "high"), "one request reads as a request does");
        let turns = [
            ("user".into(), "Hi".into()),
            ("assistant".into(), "<think>A greeting.</think>Hello!".into()),
            ("user".into(), "Bye".into()),
        ];
        let chat = spec.chat_prompt("Be brief.", &turns, Some(true), "");
        assert!(chat.starts_with("<|system|>Be brief.\n\nReasoning effort: none<|end|>"), "{chat}");
        assert!(
            chat.contains("<|assistant|>Hello!<|end|><|user|>Bye<|end|><|assistant|>"),
            "earlier reasoning is left out"
        );
        assert!(chat.ends_with("<|assistant|>"), "thinking on: no empty thinking span");
    }

    #[test]
    fn requests_list_tools_effort_and_thinking() {
        let mut spec = TextSpec {
            system: "Be brief.".into(),
            tools: vec![ToolSpec {
                name: "web_search".into(),
                description: "Searches the web.".into(),
                parameters: serde_json::json!({"type": "object", "properties": {"query": {"type": "string"}}}),
                native: false,
            }],
            thinking: Some(ThinkingSpec::default()),
            effort: Some(EffortSpec::default()),
            ..TextSpec::default()
        };
        spec.ensure_specials();
        let prompt = spec.prompt_with("hi", Some(false), "high");
        assert!(prompt.starts_with(
            "<|system|>Be brief.\n\nReasoning effort: high\n\nYou can use these tools:\n- web_search: Searches the web. \
             Parameters: {\"properties\": {\"query\": {\"type\": \"string\"}}, \"type\": \"object\"}"
        ));
        assert!(prompt.ends_with("<|user|>hi<|end|><|assistant|><think></think>"));
        assert!(spec.prompt_with("hi", None, "extreme").contains("Reasoning effort: none"));
        assert!(spec.prompt("hi").ends_with("<|assistant|>"));
        let calls =
            spec.tool_calls(r#"<tool_call>{"name": "web_search", "arguments": "{\"query\": \"rust\"}"}</tool_call>"#);
        assert_eq!(calls[0].args, serde_json::json!({"query": "rust"}));
        assert!(spec.is_text_tag("<think>") && !spec.is_text_tag("<|tool|>"));
    }

    #[test]
    fn old_text_specs_keep_their_tokens() {
        // Saved before tool turns existed: no tool tag, so no new token.
        let old: TextSpec =
            serde_json::from_str(r#"{"specials": ["<|end|>"], "tool_tags": ["<tool>", "</tool>"]}"#).unwrap();
        assert!(old.tool_tag.is_empty());
        let mut again = old.clone();
        again.ensure_specials();
        assert!(!again.specials.iter().any(|s| s == "<|tool|>"));
    }

    #[test]
    fn text_is_bytes_and_special_tags_and_comes_back_the_same() {
        let spec = TextSpec {
            personas: vec![Persona { name: "Gun".into(), tag: "<gun>".into(), role: String::new() }],
            ..TextSpec::default()
        };
        let mut spec = spec;
        spec.ensure_specials();
        let text = "<|user|>héllo<|end|><gun>ok</gun>";
        let ids = spec.encode(text);
        assert_eq!(ids[0], spec.special_id("<|user|>").unwrap());
        assert_eq!(ids.len(), 1 + "héllo".len() + 1 + 1 + 2 + 1);
        assert_eq!(spec.decode(&ids), text);
        assert_eq!(spec.vocab_size(), 256 + spec.specials.len());
        assert!(spec.encode("<notatag>").iter().all(|id| *id < 256));
    }

    #[test]
    fn learned_tokens_are_fewer_and_come_back_the_same() {
        let mut spec = TextSpec { tokenizer: TokenizerSpec::learned_of(600), ..TextSpec::default() };
        let corpus = "The weather is nice today. The weather was cold yesterday. What is the weather like? ".repeat(30);
        let learned = spec.learn_tokens([corpus.as_str()]);
        assert!(learned > 20, "{learned}");
        assert_eq!(spec.vocab_size(), 256 + learned + spec.specials.len());
        let text = "<|user|>What is the weather like today?<|end|><|assistant|>Nice and héllo.<|end|>";
        let ids = spec.encode(text);
        assert_eq!(spec.decode(&ids), text);
        assert!(ids.len() * 2 < text.len(), "{} tokens for {} bytes", ids.len(), text.len());
        // Tags come after the pieces, and are still one token each.
        assert_eq!(ids[0], spec.special_id("<|user|>").unwrap());
        assert!(spec.special_id("<|end|>").unwrap() >= 256 + learned as u32);
        // Text the pieces never saw still reads byte by byte.
        assert_eq!(spec.decode(&spec.encode("zqx<think>ok</think>")), "zqx<think>ok</think>");
        // Its JSON keeps the pieces; a byte model's has no tokenizer at all.
        let json = serde_json::to_value(&spec).unwrap();
        assert_eq!(json["tokenizer"]["merges"].as_array().unwrap().len(), learned);
        assert!(serde_json::to_value(TextSpec::default()).unwrap().get("tokenizer").is_none());
    }

    #[test]
    fn tool_calls_are_read_and_the_visible_answer_has_none() {
        let spec = TextSpec { tool_tags: ("<tool>".into(), "</tool>".into()), ..TextSpec::default() };
        let answer = r#"<think>press it</think>Sure.<tool>{"name": "press", "args": {"key": "Space"}}</tool><tool>broken</tool><|end|>"#;
        let calls = spec.tool_calls(answer);
        assert_eq!(calls, vec![ToolCall { name: "press".into(), args: serde_json::json!({ "key": "Space" }) }]);
        assert_eq!(spec.visible(answer), "Sure.");
    }

    #[test]
    fn a_text_model_keeps_its_weights_datasets_and_canvas() {
        let mut file = MagicFile::default();
        file.model.kind = ModelKind::Text;
        file.model.text = Some(TextSpec::default());
        file.model.datasets = vec!["data/text/chat.jsonl".into()];
        file.data.insert("data/text/chat.jsonl".into(), br#"{"prompt": "hi", "response": "hello"}"#.to_vec());
        file.weights = vec![1, 2, 3];
        file.model.studio = Some(Fragment::default());
        let back = MagicFile::from_bytes(&file.to_bytes().unwrap()).unwrap();
        assert_eq!(back.model.kind, ModelKind::Text);
        assert_eq!(back.weights, [1, 2, 3]);
        assert!(back.data.contains_key("data/text/chat.jsonl"));
        // A macro keeps neither the weights, the datasets nor the canvas.
        let in_macro = MagicFile::from_bytes(&file.for_macro().unwrap()).unwrap();
        assert!(in_macro.weights.is_empty() && in_macro.data.is_empty() && in_macro.model.studio.is_none());
        // Files from before text models open as vision models.
        let old: MagicModel = serde_json::from_str(r#"{ "name": "Old" }"#).unwrap();
        assert_eq!(old.kind, ModelKind::Vision);
    }

    #[test]
    fn the_key_to_press_may_be_a_mouse_button_and_old_files_keep_their_key() {
        let old: ActionSpec = serde_json::from_str(r#"{ "key": "Space", "hold_ms": 5.0 }"#).unwrap();
        assert_eq!(old.key, Button::Key(Key::Space));
        let click: ActionSpec = serde_json::from_str(r#"{ "key": "Mouse:Left" }"#).unwrap();
        assert_eq!(click.key, Button::Mouse(crate::input::MouseButton::Left));
    }

    fn two_classes() -> MagicModel {
        MagicModel {
            name: "Test".into(),
            classes: vec![
                ClassSpec { label: "Nothing".into(), ..ClassSpec::default() },
                ClassSpec { label: "Press now".into(), act: true, threshold: 0.5, delay_ms: 0.0 },
            ],
            settings: vec![ModelSetting {
                key: "thresh".into(),
                label: "Confidence".into(),
                kind: SettingKind::Percent,
                min: 50.0,
                max: 100.0,
                default: 60.0,
                value: 60.0,
                targets: vec![SettingTarget::Threshold { class: 1 }],
                ..ModelSetting::default()
            }],
            ..MagicModel::default()
        }
    }

    #[test]
    fn settings_change_what_the_model_acts_on() {
        let model = two_classes().configured(&Map::new());
        assert_eq!(model.classes[1].threshold, 0.6);
        let sure = model.decide(&[0.3, 0.7]);
        assert!(sure.act && sure.label == "Press now");
        assert!(!model.decide(&[0.45, 0.55]).act, "not sure enough");
        let mut overrides = Map::new();
        overrides.insert("thresh".into(), serde_json::json!(51));
        assert!(two_classes().configured(&overrides).decide(&[0.45, 0.55]).act);
        assert!(!model.decide(&[0.9, 0.1]).act, "the likeliest class does not act");
    }

    #[test]
    fn a_centred_square_scales_with_the_display() {
        let displays = Displays::new(vec![Rect::new(0, 0, 1920, 1080), Rect::new(1920, 0, 2560, 1440)]);
        let spec = CaptureSpec::default();
        assert_eq!(spec.rect(&displays, None), Some(Rect::new(848, 428, 224, 224)));
        let other = spec.rect(&displays, Some(1)).unwrap();
        assert_eq!((other.w, other.x, other.y), (299, 1920 + 1130, 570));
    }

    #[test]
    fn files_round_trip_and_leave_out_what_is_switched_off() {
        let mut file = MagicFile { model: two_classes(), onnx: vec![1, 2, 3], ..MagicFile::default() };
        file.data.insert("data/train/1/a.png".into(), vec![9; 10]);
        file.model.training.push(DataItem { file: "data/train/1/a.png".into(), class: 1, negative: false });
        let back = MagicFile::from_bytes(&file.to_bytes().unwrap()).unwrap();
        assert_eq!(back, file);

        let slim = MagicFile::from_bytes(&file.for_macro().unwrap()).unwrap();
        assert!(slim.data.is_empty() && slim.model.training.is_empty());
        assert_eq!(slim.onnx, vec![1, 2, 3]);
        assert!(MagicFile::from_bytes(b"not a zip").is_err());
    }
}
