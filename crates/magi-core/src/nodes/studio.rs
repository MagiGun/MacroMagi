//! The blocks of the Magic tab: a canvas for making, training and setting up
//! AI models. They never run in a macro; the Magic tab reads the canvas.
//!
//! Every block passes the model along a `model` wire: a starting point,
//! the architecture blocks in order, the training stages in order, and the
//! package. Datasets plug into the training stages. The *inspector* blocks
//! (Model Info, Screen Area, Answers…) show and edit the model's setup and
//! may hang off the model wire anywhere.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::{Category, DataType, NodeDef, ParamDef, ParamKind, Registry};
use crate::graph::{Fragment, Graph, Group, Node, NodeId, PortRef, Wire};
use crate::magic::{ArchBlock, Architecture, MagicModel, ModelKind, Persona, TextSpec};

/// Block types of the Magic tab.
pub mod kinds {
    pub const ZERO: &str = "magic.zero";
    pub const BASE: &str = "magic.base";
    pub const CONFIG: &str = "magic.config";

    pub const INFO: &str = "magic.info";
    pub const INPUT: &str = "magic.input";
    pub const ANSWERS: &str = "magic.answers";
    pub const ACTION: &str = "magic.action";
    pub const SETTINGS: &str = "magic.settings";
    pub const PREP: &str = "magic.prep";
    pub const TEXT: &str = "magic.text";
    pub const TRY: &str = "magic.try";

    pub const SIZE: &str = "magic.size";
    pub const TRANSFORMER: &str = "magic.transformer";
    pub const SPARSE: &str = "magic.sparse_attention";
    pub const MOE: &str = "magic.moe";
    pub const SPLIT: &str = "magic.prefix_split";
    pub const SSM: &str = "magic.ssm";
    pub const PERSONA: &str = "magic.persona";
    pub const CONV: &str = "magic.conv";

    pub const TEXT_DATA: &str = "magic.text_data";
    pub const IMAGE_DATA: &str = "magic.image_data";

    pub const PRETRAIN: &str = "magic.pretrain";
    pub const FINETUNE: &str = "magic.finetune";
    pub const RL: &str = "magic.rl";
    pub const COMPACTION: &str = "magic.compaction";
    pub const VISION_TRAIN: &str = "magic.vision_train";
    pub const OPTIMIZER: &str = "magic.optimizer";

    pub const COMPILE: &str = "magic.compile";
    pub const USE: &str = "magic.use";

    /// Blocks that edit the model's setup rather than settings of their own.
    pub const INSPECTORS: [&str; 8] = [INFO, INPUT, ANSWERS, ACTION, SETTINGS, PREP, TEXT, TRY];
}

/// The wire a model travels along.
pub const MODEL: &str = "model";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ZeroParams {
    /// `text` or `vision`.
    pub kind: String,
}

impl Default for ZeroParams {
    fn default() -> Self {
        Self { kind: "text".into() }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BaseParams {
    /// Empty: this model's own weights. Otherwise a `.magic` file to start from.
    pub path: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SizeParams {
    pub width: u32,
    pub heads: u32,
    /// Text models: the longest text, in tokens. Vision models: the picture's side.
    pub context: u32,
}

impl Default for SizeParams {
    fn default() -> Self {
        Self { width: 128, heads: 4, context: 256 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LayersParams {
    pub layers: u32,
    pub mult: u32,
}

impl Default for LayersParams {
    fn default() -> Self {
        Self { layers: 2, mult: 4 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SparseParams {
    pub layers: u32,
    pub kv_rank: u32,
    pub window: u32,
    pub stride: u32,
    pub share_kv: bool,
    /// The precision a runtime may keep the latent cache in (recorded for runtimes; MacroMagi keeps 32 bits).
    pub kv_bits: String,
}

impl Default for SparseParams {
    fn default() -> Self {
        Self { layers: 2, kv_rank: 32, window: 64, stride: 16, share_kv: true, kv_bits: "4".into() }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MoeParams {
    pub layers: u32,
    pub experts: u32,
    pub shared: u32,
    pub top_k: u32,
    pub mult: u32,
}

impl Default for MoeParams {
    fn default() -> Self {
        Self { layers: 1, experts: 8, shared: 1, top_k: 2, mult: 2 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SsmParams {
    pub layers: u32,
    pub state: u32,
}

impl Default for SsmParams {
    fn default() -> Self {
        Self { layers: 2, state: 16 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PersonaParams {
    pub layers: u32,
    pub first: String,
    pub first_role: String,
    pub second: String,
    pub second_role: String,
}

impl Default for PersonaParams {
    fn default() -> Self {
        Self {
            layers: 1,
            first: "Gun".into(),
            first_role: "Strict code and tool execution".into(),
            second: "Magi".into(),
            second_role: "Breaking loops, critique, routing visual tools".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ConvParams {
    pub layers: u32,
    pub channels: u32,
}

impl Default for ConvParams {
    fn default() -> Self {
        Self { layers: 3, channels: 32 }
    }
}

/// A Text Dataset block: files in the `.magic` file (`data/text/…`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TextDataParams {
    /// What to do: right answers, good tool calls, reasoning.
    pub positive: Vec<String>,
    /// What not to do: answers to avoid, broken tool calls, loops.
    pub negative: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StageParams {
    pub steps: u32,
    pub batch: u32,
    /// 0 uses the Optimizer block's learning rate.
    pub lr: f64,
}

impl Default for StageParams {
    fn default() -> Self {
        Self { steps: 300, batch: 8, lr: 0.0 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FinetuneParams {
    /// `full`, `lora` or `dora`.
    pub method: String,
    pub rank: u32,
    pub alpha: f64,
    pub steps: u32,
    pub batch: u32,
    pub lr: f64,
    /// How strongly "what not to do" counts.
    pub negative_weight: f64,
}

impl Default for FinetuneParams {
    fn default() -> Self {
        Self { method: "lora".into(), rank: 8, alpha: 16.0, steps: 200, batch: 4, lr: 0.0, negative_weight: 0.5 }
    }
}

/// One reward rule of reinforcement learning.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RewardRule {
    /// `contains`, `not_contains`, `regex`, `exact`, `tool_valid`,
    /// `tool_malformed`, `loop`, `short` or `ends`.
    pub kind: String,
    pub text: String,
    pub score: f64,
    /// `short`: at most this many tokens. `loop`: the n-gram length.
    pub tokens: u32,
    /// `loop`: how many repeats count as a loop.
    pub repeats: u32,
}

impl Default for RewardRule {
    fn default() -> Self {
        Self { kind: "contains".into(), text: String::new(), score: 1.0, tokens: 32, repeats: 3 }
    }
}

impl RewardRule {
    pub const KINDS: [(&'static str, &'static str); 9] = [
        ("contains", "Says…"),
        ("not_contains", "Does not say…"),
        ("regex", "Matches the pattern…"),
        ("exact", "Gives the dataset's answer"),
        ("tool_valid", "Calls tools correctly"),
        ("tool_malformed", "Writes a broken tool call"),
        ("loop", "Repeats itself (a loop)"),
        ("short", "Is short (fast)"),
        ("ends", "Finishes its answer"),
    ];
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RlParams {
    pub steps: u32,
    /// Answers drawn per request; each is scored against the others.
    pub group: u32,
    pub max_new: u32,
    pub temperature: f64,
    pub lr: f64,
    pub rules: Vec<RewardRule>,
}

impl Default for RlParams {
    fn default() -> Self {
        Self {
            steps: 50,
            group: 4,
            max_new: 48,
            temperature: 0.9,
            lr: 0.0,
            rules: vec![
                RewardRule { kind: "exact".into(), score: 1.0, ..RewardRule::default() },
                RewardRule { kind: "tool_valid".into(), score: 1.0, ..RewardRule::default() },
                RewardRule { kind: "short".into(), score: 5.0, tokens: 24, ..RewardRule::default() },
                RewardRule { kind: "loop".into(), score: -3.0, tokens: 4, ..RewardRule::default() },
                RewardRule { kind: "tool_malformed".into(), score: -1.0, ..RewardRule::default() },
            ],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OptimizerParams {
    /// `adamw`, `muon` or `sgd`.
    pub name: String,
    pub lr: f64,
    pub weight_decay: f64,
    /// Train on short texts first, growing to the full context.
    pub curriculum: bool,
    pub curriculum_start: u32,
    pub checkpointing: bool,
    pub offload: bool,
    pub accumulate: u32,
    /// `auto`, `cpu` or `gpu`.
    pub device: String,
}

impl Default for OptimizerParams {
    fn default() -> Self {
        Self {
            name: "adamw".into(),
            lr: 0.003,
            weight_decay: 0.01,
            curriculum: false,
            curriculum_start: 32,
            checkpointing: false,
            offload: false,
            accumulate: 1,
            device: "auto".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CompileParams {
    pub weights: bool,
    pub datasets: bool,
    pub pictures: bool,
    pub blocks: bool,
}

impl Default for CompileParams {
    fn default() -> Self {
        Self { weights: true, datasets: true, pictures: true, blocks: true }
    }
}

/// A mirrored description, for inspector blocks: the lines on the block.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InspectorParams {
    pub summary: Vec<String>,
}

fn model_in(def: NodeDef) -> NodeDef {
    def.data_in(MODEL, "Model", DataType::Any)
}

fn model_out(def: NodeDef) -> NodeDef {
    def.data_out(MODEL, "Model", DataType::Any)
}

fn lines<T: serde::de::DeserializeOwned + Default>(
    node: &crate::graph::Node,
    f: impl Fn(T) -> Vec<String>,
) -> Vec<String> {
    f(node.params_or_default())
}

fn int(min: i64, max: i64, unit: &str) -> ParamKind {
    ParamKind::int(min, max, unit)
}

fn inspector(registry: &mut Registry, id: &str, name: &str, description: &str, keywords: &[&str]) {
    registry.register(
        model_in(NodeDef::new(id, name, Category::ModelSetup, description))
            .keywords(keywords)
            .param(ParamDef::new("summary", "Summary", ParamKind::Text { multiline: true }).hidden())
            .defaults(&InspectorParams::default())
            .summarize(|node, _| lines(node, |p: InspectorParams| p.summary)),
    );
}

/// The Magic tab's blocks.
pub fn registry() -> Registry {
    use kinds::*;
    let mut r = Registry::empty();

    // ---- where to start ----
    r.register(
        model_out(NodeDef::new(
            ZERO,
            "Start from Zero",
            Category::Start,
            "A new model with fresh weights, shaped by the architecture blocks after it: pretrain it, then fine-tune it.",
        ))
        .keywords(&["new", "pretrain", "scratch", "zero", "start"])
        .param(ParamDef::new("kind", "Model", ParamKind::choice(&[("text", "Text / agent"), ("vision", "Vision")])))
        .defaults(&ZeroParams::default())
        .summarize(|node, _| lines(node, |p: ZeroParams| vec![if p.kind == "vision" { "A new vision model" } else { "A new text model" }.into()])),
    );
    r.register(
        model_out(NodeDef::new(
            BASE,
            "Load Base Model",
            Category::Start,
            "Starts from a model's trained weights – this model's own, or another .magic file trained in MacroMagi – to \
             fine-tune it or train it further. Keep the base model's architecture blocks.",
        ))
        .keywords(&["base", "load", "fine-tune", "adapt", "lora", "weights", "safetensors"])
        .param(ParamDef::new("path", "Base model", ParamKind::Path { folder: false }).help("Empty: this model's own weights."))
        .defaults(&BaseParams::default())
        .summarize(|node, _| {
            lines(node, |p: BaseParams| {
                vec![if p.path.trim().is_empty() {
                    "This model's weights".into()
                } else {
                    p.path.rsplit(['/', '\\']).next().unwrap_or_default().to_owned()
                }]
            })
        }),
    );
    r.register(
        model_out(NodeDef::new(
            CONFIG,
            "Inference / Config Only",
            Category::Start,
            "The model as it is: set its answers, tools and thresholds, and try it, without training.",
        ))
        .keywords(&["inference", "config", "configure", "try", "existing"])
        .defaults(&serde_json::Map::new())
        .summarize(|_, _| vec!["The model as it is".into()]),
    );

    // ---- the model's setup ----
    inspector(
        &mut r,
        INFO,
        "Model Info",
        "The model's name, what it does, who made it, its licence, kind and size.",
        &["info", "name", "author", "license"],
    );
    inspector(
        &mut r,
        INPUT,
        "Screen Area",
        "Vision models: the part of the screen the model looks at.",
        &["screen", "area", "capture", "input", "region"],
    );
    inspector(
        &mut r,
        ANSWERS,
        "Answers",
        "Vision models: what each answer means, whether to act on it, how sure the model must be, and how long to wait.",
        &["answers", "classes", "confidence", "threshold"],
    );
    inspector(
        &mut r,
        ACTION,
        "When It Acts",
        "Vision models: the key or mouse button to press, how long, and the cooldown.",
        &["action", "key", "press", "cooldown"],
    );
    inspector(
        &mut r,
        SETTINGS,
        "Adjustable Settings",
        "Sliders that change several values at once, offered again in every block that uses the model.",
        &["settings", "sliders"],
    );
    inspector(
        &mut r,
        PREP,
        "Picture Preparation",
        "Vision models: the picture's size, layout, color order and normalization.",
        &["normalization", "mean", "std", "resize"],
    );
    inspector(
        &mut r,
        TEXT,
        "Text & Tools",
        "Text models: the system prompt, chat tags, personas, the tools it may call and how it writes.",
        &["tools", "prompt", "system", "chat", "personas", "tokens"],
    );
    inspector(
        &mut r,
        TRY,
        "Try It",
        "Try the model: on the screen or a picture (vision), or with a request (text).",
        &["try", "test", "live", "chat"],
    );

    // ---- architecture ----
    let arch = |r: &mut Registry, def: NodeDef| r.register(model_out(model_in(def)));
    arch(
        &mut r,
        NodeDef::new(SIZE, "Model Size", Category::Architecture, "How wide the model is, how many attention heads it has, and its context: the longest text it reads, in tokens.")
            .keywords(&["size", "width", "heads", "context", "dimensions"])
            .param(ParamDef::new("width", "Width", int(8, 4096, "")))
            .param(ParamDef::new("heads", "Heads", int(1, 64, "")))
            .param(ParamDef::new("context", "Context", int(8, 8192, "")).help("Text models: the longest text, in tokens. Vision models: the side of the pictures, in pixels."))
            .defaults(&SizeParams::default())
            .summarize(|node, _| lines(node, |p: SizeParams| vec![format!("{} wide · {} heads", p.width, p.heads), format!("context {}", p.context)])),
    );
    arch(
        &mut r,
        NodeDef::new(
            TRANSFORMER,
            "Transformer",
            Category::Architecture,
            "Standard transformer layers: attention, then a feed-forward layer.",
        )
        .keywords(&["transformer", "attention", "layers", "backbone"])
        .param(ParamDef::new("layers", "Layers", int(1, 64, "")))
        .param(ParamDef::new("mult", "Feed-forward width", int(1, 16, "× width")))
        .defaults(&LayersParams::default())
        .summarize(|node, _| lines(node, |p: LayersParams| vec![format!("{} layers", p.layers)])),
    );
    arch(
        &mut r,
        NodeDef::new(
            SPARSE,
            "Compressed Sparse Attention",
            Category::Architecture,
            "Attention whose keys and values come from a small latent (a much smaller cache), optionally shared by \
             the block's layers, looking at a sliding window plus every n-th token. The cache precision is recorded \
             for runtimes.",
        )
        .keywords(&["sparse", "attention", "csa", "kv", "cache", "latent", "deepseek", "fp4", "window"])
        .param(ParamDef::new("layers", "Layers", int(1, 64, "")))
        .param(
            ParamDef::new("kv_rank", "Latent size", int(4, 1024, ""))
                .help("Keys and values are made from this many numbers per token."),
        )
        .param(
            ParamDef::new("window", "Window", int(0, 8192, "tokens"))
                .help("Each token sees this many before it. 0: all."),
        )
        .param(
            ParamDef::new("stride", "Also every", int(0, 1024, "th token"))
                .help("And these tokens, wherever they are. 0: none."),
        )
        .param(ParamDef::new("share_kv", "Share the cache across layers", ParamKind::Bool))
        .param(ParamDef::new(
            "kv_bits",
            "Cache precision (runtimes)",
            ParamKind::choice(&[("16", "16 bits"), ("8", "8 bits"), ("4", "4 bits (FP4)")]),
        ))
        .defaults(&SparseParams::default())
        .summarize(|node, _| {
            lines(node, |p: SparseParams| {
                vec![
                    format!("{} layers · latent {}", p.layers, p.kv_rank),
                    format!("window {} · every {}", p.window, p.stride),
                ]
            })
        }),
    );
    arch(
        &mut r,
        NodeDef::new(
            MOE,
            "Micro-MoE Router",
            Category::Architecture,
            "Fine-grained mixture of experts: shared experts every token goes through, and many small routed experts \
             of which each token uses only a few.",
        )
        .keywords(&["moe", "experts", "router", "mixture", "deepseek", "routed", "shared"])
        .param(ParamDef::new("layers", "Layers", int(1, 64, "")))
        .param(ParamDef::new("experts", "Routed experts", int(1, 256, "")))
        .param(ParamDef::new("shared", "Shared experts", int(0, 16, "")))
        .param(ParamDef::new("top_k", "Experts per token", int(1, 16, "")))
        .param(ParamDef::new("mult", "Expert width", int(1, 16, "× width")))
        .defaults(&MoeParams::default())
        .summarize(|node, _| {
            lines(node, |p: MoeParams| {
                vec![format!("{} layers", p.layers), format!("{} + {} shared · top {}", p.experts, p.shared, p.top_k)]
            })
        }),
    );
    arch(
        &mut r,
        NodeDef::new(
            SPLIT,
            "Encoder/Decoder Split",
            Category::Architecture,
            "The request and tool output (everything up to the assistant tag) is read all at once, both ways – fast to \
             prefill – and only the answer is written token by token (a prefix language model).",
        )
        .keywords(&["prefix", "encoder", "decoder", "ced", "split", "prefill"])
        .defaults(&serde_json::Map::new())
        .summarize(|_, _| vec!["Prefill both ways, decode causally".into()]),
    );
    arch(
        &mut r,
        NodeDef::new(
            SSM,
            "Mamba-2 / SSM",
            Category::Architecture,
            "State-space layers (Mamba-2's quadratic form): a learned decay over time instead of attention.",
        )
        .keywords(&["mamba", "ssm", "state space", "recurrent", "linear"])
        .param(ParamDef::new("layers", "Layers", int(1, 64, "")))
        .param(ParamDef::new("state", "State size", int(2, 256, "")))
        .defaults(&SsmParams::default())
        .summarize(|node, _| lines(node, |p: SsmParams| vec![format!("{} layers · state {}", p.layers, p.state)])),
    );
    arch(
        &mut r,
        NodeDef::new(
            PERSONA,
            "Dual-Persona Router",
            Category::Architecture,
            "Two personas of the reasoning, each with its own tag and its own experts, routed by the tags: one for \
             strict code and tool calls, one for critique and breaking loops. Tool calls keep one syntax.",
        )
        .keywords(&["persona", "thinking", "reasoning", "gun", "magi", "router", "dual"])
        .param(ParamDef::new("layers", "Layers", int(1, 64, "")))
        .param(ParamDef::new("first", "First persona", ParamKind::Text { multiline: false }))
        .param(ParamDef::new("first_role", "Its role", ParamKind::Text { multiline: false }))
        .param(ParamDef::new("second", "Second persona", ParamKind::Text { multiline: false }))
        .param(ParamDef::new("second_role", "Its role", ParamKind::Text { multiline: false }))
        .defaults(&PersonaParams::default())
        .summarize(|node, _| {
            lines(node, |p: PersonaParams| vec![format!("{} & {}", p.first, p.second), format!("{} layers", p.layers)])
        }),
    );
    arch(
        &mut r,
        NodeDef::new(
            CONV,
            "Vision Encoder",
            Category::Architecture,
            "Vision models: convolution layers that turn a picture into features, then the answers.",
        )
        .keywords(&["vision", "cnn", "convolution", "encoder", "image"])
        .param(ParamDef::new("layers", "Layers", int(1, 8, "")))
        .param(ParamDef::new("channels", "Channels", int(4, 512, "")))
        .defaults(&ConvParams::default())
        .summarize(|node, _| {
            lines(node, |p: ConvParams| vec![format!("{} layers · {} channels", p.layers, p.channels)])
        }),
    );

    // ---- data ----
    r.register(
        NodeDef::new(
            TEXT_DATA,
            "Text Dataset",
            Category::Data,
            "Text files in the .magic file – .jsonl requests and answers, conversations and tool traces, .md and .txt \
             text. “What to do” and “What not to do” go into training stages. Select it to edit the samples.",
        )
        .keywords(&["dataset", "text", "jsonl", "markdown", "samples", "data", "negative"])
        .data_out("positive", "What to do", DataType::Any)
        .data_out("negative", "What not to do", DataType::Any)
        .param(ParamDef::new("positive", "What to do", ParamKind::Text { multiline: false }).hidden())
        .param(ParamDef::new("negative", "What not to do", ParamKind::Text { multiline: false }).hidden())
        .defaults(&TextDataParams::default())
        .summarize(|node, _| {
            lines(node, |p: TextDataParams| {
                vec![format!("{} files to do", p.positive.len()), format!("{} not to do", p.negative.len())]
            })
        }),
    );
    r.register(
        NodeDef::new(
            IMAGE_DATA,
            "Vision Dataset",
            Category::Data,
            "Pictures of each answer, and pictures of what an answer is not (false positives).",
        )
        .keywords(&["dataset", "pictures", "images", "vision", "positive", "negative"])
        .data_out("pictures", "Pictures", DataType::Any)
        .defaults(&serde_json::Map::new())
        .summarize(|_, _| vec!["The model's pictures".into()]),
    );

    // ---- training ----
    let stage = |r: &mut Registry, def: NodeDef| r.register(model_out(model_in(def)));
    stage(
        &mut r,
        NodeDef::new(
            PRETRAIN,
            "Pretraining",
            Category::Training,
            "Learns the language: predicting the next token of plain text, in batches.",
        )
        .keywords(&["pretrain", "pretraining", "language model", "tokens"])
        .data_in("data", "Text", DataType::Any)
        .param(ParamDef::new("steps", "Steps", int(1, 1_000_000, "")))
        .param(ParamDef::new("batch", "Batch", int(1, 4096, "")))
        .param(ParamDef::new("lr", "Learning rate", ParamKind::float(0.0, 1.0, "")).help("0: the Optimizer block's."))
        .defaults(&StageParams::default())
        .summarize(|node, _| lines(node, |p: StageParams| vec![format!("{} steps · batch {}", p.steps, p.batch)])),
    );
    stage(
        &mut r,
        NodeDef::new(
            FINETUNE,
            "Fine-Tuning",
            Category::Training,
            "Learns to answer: requests and answers from “What to do”, and less likely the answers from “What not to \
             do”. All weights, or small LoRA or DoRA adapters that are folded back in.",
        )
        .keywords(&["fine-tune", "sft", "lora", "dora", "adapter", "instruct"])
        .data_in("data", "What to do", DataType::Any)
        .data_in("negative", "What not to do", DataType::Any)
        .param(ParamDef::new(
            "method",
            "Train",
            ParamKind::choice(&[("full", "All weights"), ("lora", "LoRA adapters"), ("dora", "DoRA adapters")]),
        ))
        .param(ParamDef::new("rank", "Adapter rank", int(1, 256, "")))
        .param(ParamDef::new("alpha", "Adapter strength", ParamKind::float(0.1, 256.0, "")))
        .param(ParamDef::new("steps", "Steps", int(1, 1_000_000, "")))
        .param(ParamDef::new("batch", "Batch", int(1, 4096, "")))
        .param(ParamDef::new("lr", "Learning rate", ParamKind::float(0.0, 1.0, "")).help("0: the Optimizer block's."))
        .param(ParamDef::new("negative_weight", "“Not to do” weight", ParamKind::float(0.0, 10.0, "")))
        .defaults(&FinetuneParams::default())
        .summarize(|node, _| {
            lines(node, |p: FinetuneParams| vec![format!("{} · {} steps", p.method.to_uppercase(), p.steps)])
        }),
    );
    stage(
        &mut r,
        NodeDef::new(
            RL,
            "Reinforcement Learning",
            Category::Training,
            "Several answers per request are scored by reward rules – correctness, tool calls, speed, loops – and \
             the better-than-average ones are made more likely.",
        )
        .keywords(&["rl", "reinforcement", "reward", "grpo", "score", "verifiable"])
        .data_in("data", "Requests", DataType::Any)
        .param(ParamDef::new("steps", "Steps", int(1, 100_000, "")))
        .param(ParamDef::new("group", "Answers per request", int(2, 64, "")))
        .param(ParamDef::new("max_new", "Longest answer", int(1, 4096, "tokens")))
        .param(ParamDef::new("temperature", "Temperature", ParamKind::float(0.05, 2.0, "")))
        .param(ParamDef::new("lr", "Learning rate", ParamKind::float(0.0, 1.0, "")).help("0: the Optimizer block's."))
        .param(ParamDef::new("rules", "Reward rules", ParamKind::Text { multiline: false }).hidden())
        .defaults(&RlParams::default())
        .summarize(|node, _| {
            lines(node, |p: RlParams| {
                vec![format!("{} steps · {} answers each", p.steps, p.group), format!("{} reward rules", p.rules.len())]
            })
        }),
    );
    stage(
        &mut r,
        NodeDef::new(
            COMPACTION,
            "Context Compaction",
            Category::Training,
            "Learns to turn a long context into a dense, compact one, from {\"context\", \"compact\"} samples.",
        )
        .keywords(&["compaction", "compact", "summarize", "context", "memory"])
        .data_in("data", "Samples", DataType::Any)
        .param(ParamDef::new("steps", "Steps", int(1, 1_000_000, "")))
        .param(ParamDef::new("batch", "Batch", int(1, 4096, "")))
        .param(ParamDef::new("lr", "Learning rate", ParamKind::float(0.0, 1.0, "")))
        .defaults(&StageParams::default())
        .summarize(|node, _| lines(node, |p: StageParams| vec![format!("{} steps", p.steps)])),
    );
    stage(
        &mut r,
        NodeDef::new(
            VISION_TRAIN,
            "Vision Training",
            Category::Training,
            "Learns to tell the answers apart from pictures; negative pictures of an answer make it less likely.",
        )
        .keywords(&["vision", "train", "classifier", "pictures"])
        .data_in("pictures", "Pictures", DataType::Any)
        .param(ParamDef::new("steps", "Steps", int(1, 1_000_000, "")))
        .param(ParamDef::new("batch", "Batch", int(1, 4096, "")))
        .param(ParamDef::new("lr", "Learning rate", ParamKind::float(0.0, 1.0, "")))
        .defaults(&StageParams::default())
        .summarize(|node, _| lines(node, |p: StageParams| vec![format!("{} steps · batch {}", p.steps, p.batch)])),
    );
    r.register(
        NodeDef::new(
            OPTIMIZER,
            "Optimizer & Memory",
            Category::Training,
            "How every stage learns: the optimizer (AdamW, or Muon for the layers' weights), the learning rate, a \
             context-length curriculum, and saving memory (recomputing activations, keeping optimizer state in \
             system memory).",
        )
        .keywords(&["optimizer", "muon", "adamw", "memory", "vram", "offload", "curriculum", "learning rate"])
        .param(ParamDef::new(
            "name",
            "Optimizer",
            ParamKind::choice(&[("adamw", "AdamW"), ("muon", "Muon (+ AdamW)"), ("sgd", "SGD")]),
        ))
        .param(ParamDef::new("lr", "Learning rate", ParamKind::float(0.000_001, 1.0, "")))
        .param(ParamDef::new("weight_decay", "Weight decay", ParamKind::float(0.0, 1.0, "")))
        .param(ParamDef::new("curriculum", "Grow the context while pretraining", ParamKind::Bool))
        .param(ParamDef::new("curriculum_start", "Start at", int(8, 8192, "tokens")).show_if("curriculum", true))
        .param(ParamDef::new("checkpointing", "Recompute activations (less memory)", ParamKind::Bool))
        .param(ParamDef::new("offload", "Optimizer state in system memory (graphics cards)", ParamKind::Bool))
        .param(ParamDef::new("accumulate", "Accumulate", int(1, 1024, "batches")))
        .param(ParamDef::new(
            "device",
            "Train on",
            ParamKind::choice(&[("auto", "Automatic"), ("cpu", "The CPU"), ("gpu", "A graphics card")]),
        ))
        .defaults(&OptimizerParams::default())
        .summarize(|node, _| lines(node, |p: OptimizerParams| vec![format!("{} · {}", p.name, p.lr)])),
    );

    // ---- packaging ----
    r.register(
        model_in(NodeDef::new(
            COMPILE,
            "Compile .magic Package",
            Category::Package,
            "Bundles the trained model (ONNX), its weights, its tokens and tool schemas, its architecture and its \
             thresholds into the .magic file, with what else it includes.",
        ))
        .keywords(&["compile", "package", "export", "save", "magic", "bundle"])
        .param(ParamDef::new("weights", "Weights, to train further", ParamKind::Bool))
        .param(ParamDef::new("datasets", "Text datasets", ParamKind::Bool))
        .param(ParamDef::new("pictures", "Pictures", ParamKind::Bool))
        .param(ParamDef::new("blocks", "Blocks to use it with", ParamKind::Bool))
        .defaults(&CompileParams::default())
        .summarize(|_, _| vec!["Into the .magic file".into()]),
    );
    r.register(
        model_in(NodeDef::new(
            USE,
            "Use in Macro",
            Category::Package,
            "Puts the model into the open macro, with blocks that run it.",
        ))
        .keywords(&["macro", "use", "export", "agent", "watch"])
        .defaults(&serde_json::Map::new())
        .summarize(|_, _| vec!["Into the Macro tab".into()]),
    );
    r
}

// ---- the canvas of a model ----------------------------------------------------

fn place(nodes: &mut Vec<Node>, registry: &Registry, kind: &str, pos: [f32; 2], params: Option<Value>) -> NodeId {
    let id = nodes.len() as NodeId + 1;
    let mut node = Node::new(id, kind, pos, registry.get(kind).map(|d| d.default_params()).unwrap_or_default());
    if let Some(Value::Object(map)) = params {
        node.params.extend(map);
    }
    nodes.push(node);
    id
}

fn wire(wires: &mut Vec<Wire>, from: NodeId, out: &str, to: NodeId, input: &str) {
    wires.push(Wire { from: PortRef::new(from, out), to: PortRef::new(to, input) });
}

/// The architecture blocks a recorded architecture is made of, as blocks of the canvas.
fn arch_nodes(architecture: &Architecture) -> Vec<(&'static str, Value)> {
    architecture
        .blocks
        .iter()
        .filter_map(|block| {
            let kind = match block.kind.as_str() {
                "transformer" => kinds::TRANSFORMER,
                "sparse_attention" => kinds::SPARSE,
                "moe" => kinds::MOE,
                "prefix_split" => kinds::SPLIT,
                "ssm" => kinds::SSM,
                "persona_router" => kinds::PERSONA,
                "conv" => kinds::CONV,
                _ => return None,
            };
            Some((kind, Value::Object(block.params.clone())))
        })
        .collect()
}

/// The canvas a model opens with when it has none: its setup as inspector
/// blocks, and a pipeline that fits it – training from zero for a new model,
/// fine-tuning for one with weights, setting it up for one made elsewhere.
pub fn default_canvas(model: &MagicModel, has_weights: bool, has_onnx: bool) -> Fragment {
    use kinds::*;
    let registry = registry();
    let (mut nodes, mut wires) = (Vec::new(), Vec::new());
    let text = model.kind == ModelKind::Text;
    const STEP: f32 = 250.0;

    // The pipeline, left to right.
    let entry = if has_weights {
        place(&mut nodes, &registry, BASE, [0.0, 0.0], None)
    } else if has_onnx {
        place(&mut nodes, &registry, CONFIG, [0.0, 0.0], None)
    } else {
        let kind = if text { "text" } else { "vision" };
        place(&mut nodes, &registry, ZERO, [0.0, 0.0], Some(serde_json::json!({ "kind": kind })))
    };
    let mut last = entry;
    let mut x = STEP;
    if has_weights || !has_onnx {
        let architecture = model.architecture.clone().unwrap_or_else(|| Architecture {
            blocks: if text {
                vec![ArchBlock { kind: "transformer".into(), params: Map::new() }]
            } else {
                vec![ArchBlock { kind: "conv".into(), params: Map::new() }]
            },
            ..Architecture::default()
        });
        let context = if text { model.text.as_ref().map_or(256, |t| t.context) } else { model.input.width.max(8) };
        let size = serde_json::json!({ "width": architecture.width, "heads": architecture.heads, "context": context });
        let id = place(&mut nodes, &registry, SIZE, [x, 0.0], Some(size));
        wire(&mut wires, last, MODEL, id, MODEL);
        (last, x) = (id, x + STEP);
        for (kind, params) in arch_nodes(&architecture) {
            let id = place(&mut nodes, &registry, kind, [x, 0.0], Some(params));
            wire(&mut wires, last, MODEL, id, MODEL);
            (last, x) = (id, x + STEP);
        }
        let data_y = 200.0;
        if text {
            let data = place(
                &mut nodes,
                &registry,
                TEXT_DATA,
                [x - STEP * 0.5, data_y],
                Some(serde_json::json!({ "positive": model.datasets })),
            );
            let stages: &[&str] = if has_weights { &[FINETUNE] } else { &[PRETRAIN, FINETUNE] };
            for kind in stages {
                let id = place(&mut nodes, &registry, kind, [x, 0.0], None);
                wire(&mut wires, last, MODEL, id, MODEL);
                wire(&mut wires, data, "positive", id, "data");
                if *kind == FINETUNE {
                    wire(&mut wires, data, "negative", id, "negative");
                }
                (last, x) = (id, x + STEP);
            }
        } else {
            let data = place(&mut nodes, &registry, IMAGE_DATA, [x - STEP * 0.5, data_y], None);
            let id = place(&mut nodes, &registry, VISION_TRAIN, [x, 0.0], None);
            wire(&mut wires, last, MODEL, id, MODEL);
            wire(&mut wires, data, "pictures", id, "pictures");
            (last, x) = (id, x + STEP);
        }
        place(&mut nodes, &registry, OPTIMIZER, [x - STEP, data_y + 130.0], None);
    } else if !text {
        place(&mut nodes, &registry, IMAGE_DATA, [x, 200.0], None);
    }
    let compile = place(&mut nodes, &registry, COMPILE, [x, 0.0], None);
    wire(&mut wires, last, MODEL, compile, MODEL);
    let used = place(&mut nodes, &registry, USE, [x + STEP, 0.0], None);
    wire(&mut wires, last, MODEL, used, MODEL);

    // The model's setup, in a box of its own under the pipeline.
    let inspectors: &[&str] =
        if text { &[INFO, TEXT, TRY] } else { &[INFO, INPUT, ANSWERS, ACTION, SETTINGS, PREP, TRY] };
    let mut setup = std::collections::BTreeSet::new();
    for (index, kind) in inspectors.iter().enumerate() {
        let (column, row) = (index % 4, index / 4);
        setup.insert(place(&mut nodes, &registry, kind, [column as f32 * STEP, 520.0 + row as f32 * 130.0], None));
    }
    let group =
        Group { id: 1, name: "The model's setup".into(), color: None, nodes: setup, parent: None, enabled: true };
    let mut fragment = Fragment { nodes, wires, groups: vec![group] };
    let mut all: Vec<&mut Node> = fragment.nodes.iter_mut().collect();
    refresh(&mut all, model, 0);
    fragment
}

/// What a model looks like, in a few lines, for its inspector blocks.
pub fn summary(kind: &str, model: &MagicModel, size_bytes: usize) -> Vec<String> {
    use kinds::*;
    let text = model.text.clone().unwrap_or_default();
    match kind {
        INFO => {
            let name = if model.name.trim().is_empty() { "Untitled model".to_owned() } else { model.name.clone() };
            let mut size = model.kind.label().to_owned();
            if let Some(trained) = &model.trained
                && trained.parameters > 0
            {
                size += &format!(" · {:.1}M parameters", trained.parameters as f64 / 1e6);
            } else if size_bytes > 0 {
                size += &format!(" · {:.1} MB", size_bytes as f64 / (1024.0 * 1024.0));
            }
            vec![name, size]
        }
        INPUT => vec![match &model.capture.area {
            crate::magic::CaptureArea::Center { height_share } => {
                format!("A {:.0}-px square in the middle", height_share * 1080.0)
            }
            crate::magic::CaptureArea::Region { rect } => format!("{}×{} at {}, {}", rect.w, rect.h, rect.x, rect.y),
        }],
        ANSWERS => {
            let mut lines: Vec<String> = model
                .classes
                .iter()
                .take(4)
                .map(|c| {
                    if c.act { format!("{} – act at {:.0}%", c.label, c.threshold * 100.0) } else { c.label.clone() }
                })
                .collect();
            if model.classes.len() > 4 {
                lines.push(format!("and {} more", model.classes.len() - 4));
            }
            if lines.is_empty() {
                lines.push("No answers yet".into());
            }
            lines
        }
        ACTION => vec![format!("Press {} for {:.0} ms", model.action.key.label(), model.action.hold_ms)],
        SETTINGS => vec![match model.settings.len() {
            0 => "No sliders".into(),
            1 => "1 slider".into(),
            n => format!("{n} sliders"),
        }],
        PREP => vec![format!("{}×{} pictures", model.input.width, model.input.height)],
        TEXT => vec![
            format!("{} tools · {} personas", text.tools.len(), text.personas.len()),
            format!("{} tokens of context", text.context),
        ],
        TRY => vec![if model.kind == ModelKind::Text { "Ask it something".into() } else { "Watch the screen".into() }],
        _ => Vec::new(),
    }
}

/// Puts the model's setup on its inspector blocks.
pub fn refresh<N: std::ops::DerefMut<Target = Node>>(nodes: &mut [N], model: &MagicModel, size_bytes: usize) {
    for node in nodes.iter_mut() {
        if kinds::INSPECTORS.contains(&node.kind.as_str()) {
            let lines = summary(&node.kind, model, size_bytes);
            if node.param::<Vec<String>>("summary").as_ref() != Some(&lines) {
                node.set_param("summary", lines);
            }
        }
    }
}

/// One training stage of the pipeline.
#[derive(Clone, Debug, PartialEq)]
pub struct Stage {
    pub node: NodeId,
    pub kind: String,
    pub params: Map<String, Value>,
    /// The Text Dataset outputs feeding it, as `(block, port)`.
    pub data: Vec<(NodeId, String)>,
    pub negatives: Vec<(NodeId, String)>,
}

/// The model wire, followed from its start to its end.
#[derive(Clone, Debug, PartialEq)]
pub struct Pipeline {
    pub entry: Node,
    pub size: SizeParams,
    pub architecture: Architecture,
    pub personas: Option<PersonaParams>,
    pub stages: Vec<Stage>,
    pub optimizer: OptimizerParams,
    pub compile: Option<CompileParams>,
}

impl Pipeline {
    pub fn kind(&self) -> ModelKind {
        match self.entry.kind.as_str() {
            kinds::ZERO if self.entry.params_or_default::<ZeroParams>().kind == "vision" => ModelKind::Vision,
            kinds::ZERO => ModelKind::Text,
            _ if self.architecture.blocks.iter().any(|b| b.kind == "conv") => ModelKind::Vision,
            _ if self.stages.iter().any(|s| s.kind == kinds::VISION_TRAIN) => ModelKind::Vision,
            _ => ModelKind::Text,
        }
    }

    pub fn trains(&self) -> bool {
        !self.stages.is_empty()
    }
}

/// Follows the model wire from the start block through the architecture and
/// training blocks. Inspector and packaging blocks may hang off it anywhere.
pub fn pipeline(graph: &Graph) -> Result<Pipeline, String> {
    use kinds::*;
    let enabled = |n: &&Node| n.enabled && graph.is_active(n.id);
    let entries: Vec<&Node> =
        graph.nodes().filter(enabled).filter(|n| [ZERO, BASE, CONFIG].contains(&n.kind.as_str())).collect();
    let entry = match entries.as_slice() {
        [] => {
            return Err(
                "Add a block to start from: Start from Zero, Load Base Model or Inference / Config Only.".into()
            );
        }
        [one] => *one,
        _ => return Err("There is more than one block to start from; keep one.".into()),
    };
    let chain_kinds =
        [SIZE, TRANSFORMER, SPARSE, MOE, SPLIT, SSM, PERSONA, CONV, PRETRAIN, FINETUNE, RL, COMPACTION, VISION_TRAIN];
    let mut pipeline = Pipeline {
        entry: entry.clone(),
        size: SizeParams::default(),
        architecture: Architecture::default(),
        personas: None,
        stages: Vec::new(),
        optimizer: graph
            .nodes()
            .filter(enabled)
            .find(|n| n.kind == OPTIMIZER)
            .map(|n| n.params_or_default())
            .unwrap_or_default(),
        compile: None,
    };
    let mut current = entry.id;
    let mut seen = std::collections::BTreeSet::from([current]);
    loop {
        let next: Vec<&Node> = graph
            .wires()
            .iter()
            .filter(|w| w.from.node == current && w.from.port == MODEL && w.to.port == MODEL)
            .filter_map(|w| graph.node(w.to.node))
            .filter(enabled)
            .collect();
        if let Some(compile) = next.iter().find(|n| n.kind == COMPILE) {
            pipeline.compile = Some(compile.params_or_default());
        }
        let onward: Vec<&&Node> = next.iter().filter(|n| chain_kinds.contains(&n.kind.as_str())).collect();
        let node = match onward.as_slice() {
            [] => break,
            [one] => **one,
            _ => return Err("The model wire splits after one block; a model is built along one line.".into()),
        };
        if !seen.insert(node.id) {
            break;
        }
        current = node.id;
        let arch = |kind: &str| ArchBlock { kind: kind.into(), params: node.params.clone() };
        match node.kind.as_str() {
            SIZE => {
                pipeline.size = node.params_or_default();
                pipeline.architecture.width = pipeline.size.width;
                pipeline.architecture.heads = pipeline.size.heads;
            }
            TRANSFORMER => pipeline.architecture.blocks.push(arch("transformer")),
            SPARSE => pipeline.architecture.blocks.push(arch("sparse_attention")),
            MOE => pipeline.architecture.blocks.push(arch("moe")),
            SPLIT => pipeline.architecture.blocks.push(arch("prefix_split")),
            SSM => pipeline.architecture.blocks.push(arch("ssm")),
            CONV => pipeline.architecture.blocks.push(arch("conv")),
            PERSONA => {
                pipeline.personas = Some(node.params_or_default());
                pipeline.architecture.blocks.push(arch("persona_router"));
            }
            _ => {
                let sources = |port: &str| -> Vec<(NodeId, String)> {
                    graph.data_source(node.id, port).map(|p| vec![(p.node, p.port.clone())]).unwrap_or_default()
                };
                pipeline.stages.push(Stage {
                    node: node.id,
                    kind: node.kind.clone(),
                    params: node.params.clone(),
                    data: sources(if node.kind == VISION_TRAIN { "pictures" } else { "data" }),
                    negatives: sources("negative"),
                });
            }
        }
    }
    // Architecture blocks must come before the first training stage.
    let first_stage = pipeline.stages.first().map(|s| s.node);
    if let Some(first) = first_stage {
        let mut after = false;
        let mut current = entry.id;
        while let Some(w) = graph.wires().iter().find(|w| {
            w.from.node == current
                && w.from.port == MODEL
                && graph.node(w.to.node).is_some_and(|n| chain_kinds.contains(&n.kind.as_str()) && n.enabled)
        }) {
            current = w.to.node;
            after |= current == first;
            let kind = graph.node(current).map(|n| n.kind.as_str()).unwrap_or_default();
            if after && [SIZE, TRANSFORMER, SPARSE, MOE, SPLIT, SSM, PERSONA, CONV].contains(&kind) {
                return Err("Put the architecture blocks before the training stages.".into());
            }
            if current == entry.id {
                break;
            }
        }
    }
    if entry.kind == CONFIG && pipeline.trains() {
        return Err(
            "Inference / Config Only does not train. To train, start from zero or from a base model's weights.".into(),
        );
    }
    Ok(pipeline)
}

/// Where the files a training run needs were written.
pub struct PlanFiles<'a> {
    /// A Text Dataset's file (its path inside the `.magic` file) on disk.
    pub dataset: &'a dyn Fn(&str) -> Option<String>,
    /// The weights to start from, on disk.
    pub weights: Option<String>,
    /// The prepared pictures (vision).
    pub pictures: Option<String>,
    pub out_dir: String,
}

/// The settings of a text model as the pipeline shapes it: its context and personas.
pub fn shaped_text(pipeline: &Pipeline, model: &MagicModel) -> TextSpec {
    let mut text = model.text.clone().unwrap_or_default();
    text.context = pipeline.size.context;
    if let Some(p) = &pipeline.personas {
        let persona = |name: &str, role: &str| Persona {
            name: name.to_owned(),
            tag: format!("<{}>", name.trim().to_lowercase().replace(' ', "_")),
            role: role.to_owned(),
        };
        text.personas = vec![persona(&p.first, &p.first_role), persona(&p.second, &p.second_role)];
    }
    text.ensure_specials();
    text
}

/// The training plan the trainer (`runtime/python/magi_train`) reads.
pub fn plan(pipeline: &Pipeline, graph: &Graph, model: &MagicModel, files: &PlanFiles<'_>) -> Result<Value, String> {
    use serde_json::json;
    let kind = pipeline.kind();
    let mut datasets = Map::new();
    // A dataset output with no files adds nothing.
    let mut key_of = |(node, port): &(NodeId, String)| -> Result<Option<String>, String> {
        let block = graph.node(*node).ok_or("a dataset block is missing")?;
        let params: TextDataParams = block.params_or_default();
        let list = if port == "negative" { &params.negative } else { &params.positive };
        if list.is_empty() {
            return Ok(None);
        }
        let key = format!("{node}.{port}");
        let mut paths = Vec::new();
        for file in list {
            paths.push(Value::from(
                (files.dataset)(file).ok_or_else(|| format!("The dataset file “{file}” is missing."))?,
            ));
        }
        datasets.insert(key.clone(), Value::Array(paths));
        Ok(Some(key))
    };
    let mut stages = Vec::new();
    for stage in &pipeline.stages {
        let data: Vec<String> =
            stage.data.iter().map(&mut key_of).collect::<Result<Vec<_>, _>>()?.into_iter().flatten().collect();
        let negatives: Vec<String> =
            stage.negatives.iter().map(&mut key_of).collect::<Result<Vec<_>, _>>()?.into_iter().flatten().collect();
        let mut entry = stage.params.clone();
        let ty = match stage.kind.as_str() {
            kinds::PRETRAIN => "pretrain",
            kinds::FINETUNE => {
                let p: FinetuneParams = serde_json::from_value(Value::Object(stage.params.clone())).unwrap_or_default();
                entry.insert("adapter".into(), Value::from(p.method));
                "sft"
            }
            kinds::RL => {
                let p: RlParams = serde_json::from_value(Value::Object(stage.params.clone())).unwrap_or_default();
                let rules: Vec<Value> = p
                    .rules
                    .iter()
                    .map(|r| {
                        json!({ "kind": r.kind, "text": r.text, "score": r.score, "tokens": r.tokens,
                                "ngram": r.tokens, "repeats": r.repeats })
                    })
                    .collect();
                entry.insert("rules".into(), Value::Array(rules));
                "rl"
            }
            kinds::COMPACTION => "compaction",
            kinds::VISION_TRAIN => "vision",
            _ => continue,
        };
        if kind == ModelKind::Text && ty != "vision" && data.is_empty() {
            let name = registry().get(&stage.kind).map(|d| d.name.clone()).unwrap_or_default();
            return Err(format!("{name} has nothing to learn from: wire a Text Dataset with files into it."));
        }
        if kind == ModelKind::Vision && ty != "vision" {
            return Err("A vision model trains with Vision Training only.".into());
        }
        if kind == ModelKind::Text && ty == "vision" {
            return Err("Vision Training is for vision models.".into());
        }
        entry.insert("type".into(), Value::from(ty));
        entry.insert("data".into(), json!(data));
        entry.insert("negatives".into(), json!(negatives));
        stages.push(Value::Object(entry));
    }
    let o = &pipeline.optimizer;
    let mut optimizer = json!({
        "name": o.name, "lr": o.lr, "weight_decay": o.weight_decay, "checkpointing": o.checkpointing,
        "offload": o.offload, "accumulate": o.accumulate.max(1), "device": o.device,
    });
    if o.curriculum {
        optimizer["curriculum"] =
            json!({ "start": o.curriculum_start.min(pipeline.size.context), "end": pipeline.size.context });
    }
    let mut plan = json!({
        "kind": if kind == ModelKind::Text { "text" } else { "vision" },
        "out_dir": files.out_dir,
        "seed": 0,
        "architecture": pipeline.architecture,
        "start": { "mode": pipeline.entry.kind.trim_start_matches("magic."), "weights": files.weights },
        "datasets": datasets,
        "stages": stages,
        "optimizer": optimizer,
    });
    match kind {
        ModelKind::Text => {
            plan["text"] = serde_json::to_value(shaped_text(pipeline, model)).map_err(|e| e.to_string())?
        }
        ModelKind::Vision => {
            let pictures = files.pictures.clone().ok_or("The model has no pictures to learn from.")?;
            if model.classes.len() < 2 {
                return Err("A vision model needs at least two answers (Answers block).".into());
            }
            plan["vision"] = json!({ "pictures": pictures, "classes": model.classes.len() });
        }
    }
    Ok(plan)
}

// ---- text datasets ----------------------------------------------------------

/// Whether a dataset file holds one JSON sample per line (`.jsonl`), rather
/// than plain text split at blank lines (`.md`, `.txt`).
pub fn is_jsonl(file: &str) -> bool {
    let lower = file.to_lowercase();
    lower.ends_with(".jsonl") || lower.ends_with(".json")
}

/// The samples of a dataset file: its lines (`.jsonl`) or its paragraphs.
pub fn split_samples(file: &str, content: &str) -> Vec<String> {
    if is_jsonl(file) {
        content.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_owned).collect()
    } else {
        content
            .replace("\r\n", "\n")
            .split("\n\n")
            .map(str::trim)
            .filter(|b| !b.is_empty())
            .map(str::to_owned)
            .collect()
    }
}

/// A dataset file from its samples.
pub fn join_samples(file: &str, samples: &[String]) -> String {
    let separator = if is_jsonl(file) { "\n" } else { "\n\n" };
    let mut out = samples.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(separator);
    out.push('\n');
    out
}

/// What kind of sample one is, or what is wrong with it: the JSON, the
/// fields the trainer reads, and every tool call in it.
pub fn check_sample(jsonl: bool, sample: &str, text: &TextSpec) -> Result<&'static str, String> {
    let (kind, strings) = if jsonl {
        let value: Value = serde_json::from_str(sample).map_err(|e| format!("Not valid JSON: {e}"))?;
        let Value::Object(item) = &value else {
            return Err("A sample is a JSON object, such as {\"prompt\": …, \"response\": …}.".into());
        };
        let text_of = |key: &str| item.get(key).map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_owned));
        if let Some(messages) = item.get("messages") {
            let Value::Array(messages) = messages else { return Err("“messages” must be a list.".into()) };
            let mut strings = Vec::new();
            for message in messages {
                let role = message.get("role").and_then(Value::as_str).unwrap_or_default();
                if !["system", "user", "assistant", "tool"].contains(&role) {
                    return Err(format!("A message's role must be system, user, assistant or tool, not “{role}”."));
                }
                strings.push(message.get("content").and_then(Value::as_str).unwrap_or_default().to_owned());
            }
            if !messages.iter().any(|m| m.get("role").and_then(Value::as_str) == Some("assistant")) {
                return Err("A conversation needs an answer: a message with the role “assistant”.".into());
            }
            ("Conversation", strings)
        } else if let (Some(context), Some(compact)) = (text_of("context"), text_of("compact")) {
            ("Compaction", vec![context, compact])
        } else if let Some(prompt) = text_of("prompt") {
            if let Some(bad) = text_of("rejected").or_else(|| text_of("bad")) {
                ("Not to answer", vec![prompt, bad])
            } else if let Some(response) = text_of("response") {
                ("Request and answer", vec![prompt, response])
            } else {
                return Err("A request needs its “response” (or “rejected” for what not to answer).".into());
            }
        } else if let Some(text) = text_of("text") {
            ("Text", vec![text])
        } else {
            return Err(
                "Not a sample the trainer reads: give prompt and response, messages, context and compact, or text."
                    .into(),
            );
        }
    } else {
        ("Text", vec![sample.to_owned()])
    };
    for string in &strings {
        check_tools(string, text)?;
    }
    Ok(kind)
}

/// Every tool call in a text must be written the one way: `<tool>{"name": …, "args": {…}}</tool>`.
fn check_tools(text: &str, spec: &TextSpec) -> Result<(), String> {
    let (open, close) = (spec.tool_tags.0.as_str(), spec.tool_tags.1.as_str());
    let mut rest = text;
    while let Some(start) = rest.find(open) {
        let after = &rest[start + open.len()..];
        let end = after.find(close).ok_or_else(|| format!("A tool call is not closed with {close}."))?;
        let call: Value = serde_json::from_str(after[..end].trim())
            .map_err(|_| format!("A tool call is not JSON: {}", after[..end].trim()))?;
        if !call.get("name").is_some_and(Value::is_string) {
            return Err("A tool call needs a “name”.".into());
        }
        if call.get("args").is_some_and(|a| !a.is_object()) {
            return Err("A tool call's “args” must be an object.".into());
        }
        if let Some(name) = call.get("name").and_then(Value::as_str)
            && !spec.tools.is_empty()
            && !spec.tools.iter().any(|t| t.name == name)
        {
            return Err(format!("The tool “{name}” is not one of the model's tools (Text & Tools)."));
        }
        rest = &after[end + close.len()..];
    }
    if rest.contains(close) {
        return Err(format!("{close} without {open}."));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_magic_block_has_complete_defaults() {
        let registry = registry();
        let all = [
            kinds::ZERO,
            kinds::BASE,
            kinds::CONFIG,
            kinds::SIZE,
            kinds::TRANSFORMER,
            kinds::SPARSE,
            kinds::MOE,
            kinds::SPLIT,
            kinds::SSM,
            kinds::PERSONA,
            kinds::CONV,
            kinds::TEXT_DATA,
            kinds::IMAGE_DATA,
            kinds::PRETRAIN,
            kinds::FINETUNE,
            kinds::RL,
            kinds::COMPACTION,
            kinds::VISION_TRAIN,
            kinds::OPTIMIZER,
            kinds::COMPILE,
            kinds::USE,
        ];
        for id in all.into_iter().chain(kinds::INSPECTORS) {
            assert!(registry.get(id).is_some(), "{id}");
        }
        assert!(Registry::builtin().get(kinds::ZERO).is_none(), "macros never see them");
    }

    fn text_model() -> MagicModel {
        MagicModel { kind: ModelKind::Text, text: Some(TextSpec::default()), ..MagicModel::default() }
    }

    #[test]
    fn a_new_text_model_gets_a_pipeline_that_trains_it() {
        let canvas = default_canvas(&text_model(), false, false);
        let graph = Graph::from(canvas);
        let pipeline = pipeline(&graph).unwrap();
        assert_eq!(pipeline.entry.kind, kinds::ZERO);
        assert_eq!(pipeline.kind(), ModelKind::Text);
        let stages: Vec<&str> = pipeline.stages.iter().map(|s| s.kind.as_str()).collect();
        assert_eq!(stages, [kinds::PRETRAIN, kinds::FINETUNE]);
        assert_eq!(pipeline.architecture.blocks[0].kind, "transformer");
        assert!(pipeline.compile.is_some());
        assert_eq!(pipeline.stages[1].negatives.len(), 1, "fine-tuning learns what not to do too");

        // The dataset block's files, as the plan sees them.
        let mut graph = graph;
        let data = graph.nodes().find(|n| n.kind == kinds::TEXT_DATA).unwrap().id;
        graph.node_mut(data).unwrap().set_params(&TextDataParams {
            positive: vec!["data/text/a.jsonl".into()],
            negative: vec!["data/text/b.jsonl".into()],
        });
        let pipeline = super::pipeline(&graph).unwrap();
        let files = PlanFiles {
            dataset: &|file: &str| Some(format!("/tmp/{}", file.rsplit('/').next().unwrap())),
            weights: None,
            pictures: None,
            out_dir: "/tmp/out".into(),
        };
        let plan = plan(&pipeline, &graph, &text_model(), &files).unwrap();
        assert_eq!(plan["kind"], "text");
        assert_eq!(plan["stages"][0]["type"], "pretrain");
        assert_eq!(plan["stages"][1]["type"], "sft");
        assert_eq!(plan["stages"][1]["adapter"], "lora");
        let key = plan["stages"][1]["negatives"][0].as_str().unwrap().to_owned();
        assert_eq!(plan["datasets"][&key][0], "/tmp/b.jsonl");
        assert_eq!(plan["text"]["context"], 256);
    }

    #[test]
    fn the_pipeline_explains_what_is_wrong() {
        // A model made elsewhere is only set up, and its setup is on the canvas.
        let vision = MagicModel { classes: vec![Default::default(); 2], ..MagicModel::default() };
        let graph = Graph::from(default_canvas(&vision, false, true));
        let found = pipeline(&graph).unwrap();
        assert!(!found.trains());
        assert!(graph.nodes().any(|n| n.kind == kinds::ANSWERS));
        assert!(graph.nodes().any(|n| n.kind == kinds::INPUT));

        // Training stages after Inference / Config Only are refused, with a reason.
        let mut graph = graph;
        let config = graph.nodes().find(|n| n.kind == kinds::CONFIG).unwrap().id;
        let stage = graph.add_with(kinds::VISION_TRAIN, [0.0, 0.0], Map::new());
        graph.connect(&registry(), PortRef::new(config, MODEL), PortRef::new(stage, MODEL)).unwrap();
        assert!(pipeline(&graph).unwrap_err().contains("does not train"));

        // Two starts.
        let mut two = Graph::from(default_canvas(&text_model(), false, false));
        two.add_with(kinds::ZERO, [0.0, 0.0], Map::new());
        assert!(pipeline(&two).unwrap_err().contains("more than one"));

        // A stage without its dataset.
        let mut bare = Graph::from(default_canvas(&text_model(), false, false));
        let data = bare.nodes().find(|n| n.kind == kinds::TEXT_DATA).unwrap().id;
        bare.remove(data);
        let found = pipeline(&bare).unwrap();
        let files = PlanFiles { dataset: &|_: &str| None, weights: None, pictures: None, out_dir: String::new() };
        assert!(plan(&found, &bare, &text_model(), &files).unwrap_err().contains("nothing to learn from"));
    }

    #[test]
    fn a_trained_model_opens_with_its_own_architecture() {
        let mut model = text_model();
        model.architecture = Some(Architecture {
            width: 64,
            heads: 2,
            blocks: vec![
                ArchBlock { kind: "ssm".into(), params: Map::new() },
                ArchBlock { kind: "moe".into(), params: Map::new() },
            ],
        });
        let graph = Graph::from(default_canvas(&model, true, true));
        let found = pipeline(&graph).unwrap();
        assert_eq!(found.entry.kind, kinds::BASE);
        let kinds: Vec<&str> = found.architecture.blocks.iter().map(|b| b.kind.as_str()).collect();
        assert_eq!((found.architecture.width, found.architecture.heads, kinds), (64, 2, vec!["ssm", "moe"]));
        assert_eq!(found.stages.len(), 1, "fine-tuning");
    }

    #[test]
    fn dataset_samples_are_checked() {
        let spec = TextSpec::default();
        let check = |sample: &str| check_sample(true, sample, &spec);
        assert_eq!(check(r#"{"prompt": "hi", "response": "yo"}"#), Ok("Request and answer"));
        assert_eq!(check(r#"{"prompt": "hi", "rejected": "no"}"#), Ok("Not to answer"));
        assert_eq!(check(r#"{"context": "long", "compact": "short"}"#), Ok("Compaction"));
        assert_eq!(check(r#"{"text": "plain"}"#), Ok("Text"));
        let tool = r#"{"prompt": "go", "response": "<tool>{\"name\": \"press\", \"args\": {\"key\": \"A\"}}</tool>"}"#;
        assert_eq!(check(tool), Ok("Request and answer"));
        assert!(check("{not json").is_err());
        assert!(check(r#"{"prompt": "hi"}"#).unwrap_err().contains("response"));
        assert!(check(r#"{"messages": [{"role": "user", "content": "hi"}]}"#).unwrap_err().contains("assistant"));
        assert!(check(r#"{"prompt": "go", "response": "<tool>{\"args\": {}}</tool>"}"#).unwrap_err().contains("name"));
        assert!(
            check(r#"{"prompt": "go", "response": "<tool>{\"name\": \"x\"}"}"#).unwrap_err().contains("not closed")
        );
        // Plain text is checked for tool calls too.
        assert!(check_sample(false, "a <tool>oops</tool>", &spec).is_err());

        let samples = split_samples("a.md", "one\r\n\r\ntwo\n\n\nthree");
        assert_eq!(samples, ["one", "two", "three"]);
        assert_eq!(join_samples("a.md", &samples), "one\n\ntwo\n\nthree\n");
        assert_eq!(split_samples("a.jsonl", "{}\n\n{}\n"), ["{}", "{}"]);
    }
}
