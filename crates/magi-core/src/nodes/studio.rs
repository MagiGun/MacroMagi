//! The blocks of the Magic tab: a canvas for making, training, setting up
//! and trying AI models. They never run in a macro; the Magic tab reads the
//! canvas.
//!
//! Four kinds of wire connect them:
//!
//! - **Model** – the model itself, left to right: a start, the architecture
//!   blocks in order, the training stages in order, the package, and the
//!   blocks that try the result.
//! - **Setup** – the model's own settings (its name, tools, answers…): setup
//!   blocks chain into each other and into the start's *Setup* input. What a
//!   block on that chain says is what the model is.
//! - **Feed-forward** – a Mixture of Experts or Persona Router block, wired
//!   into the layer blocks that use it instead of a plain feed-forward layer.
//! - **Data** and **Optimizer** – datasets and optimizer settings, wired into
//!   the training stages that use them.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::{Category, DataType, NodeDef, ParamDef, ParamKind, Registry};
use crate::graph::{Fragment, Graph, Group, Node, NodeId, PortRef, Wire};
use crate::magic::{
    ActionSpec, ArchBlock, Architecture, CaptureSpec, ClassSpec, EffortSpec, InputSpec, MagicModel, ModelKind,
    ModelSetting, Persona, TextSpec, ThinkingSpec, ToolSpec,
};

/// Block types of the Magic tab.
pub mod kinds {
    pub const ZERO: &str = "magic.zero";
    pub const BASE: &str = "magic.base";
    pub const CONFIG: &str = "magic.config";

    pub const INFO: &str = "magic.info";
    pub const TEXT: &str = "magic.text";
    pub const THINKING: &str = "magic.thinking";
    pub const EFFORT: &str = "magic.effort";
    pub const INPUT: &str = "magic.input";
    pub const ANSWERS: &str = "magic.answers";
    pub const ACTION: &str = "magic.action";
    pub const SETTINGS: &str = "magic.settings";
    pub const PREP: &str = "magic.prep";

    pub const SIZE: &str = "magic.size";
    pub const LAYERS: &str = "magic.layers";
    pub const WINDOW: &str = "magic.window";
    pub const MLA: &str = "magic.mla";
    pub const CSA: &str = "magic.csa";
    pub const HCA: &str = "magic.hca";
    pub const SSM: &str = "magic.ssm";
    pub const MOE: &str = "magic.moe";
    pub const PERSONA: &str = "magic.persona";
    pub const HYPER: &str = "magic.hyper";
    pub const MTP: &str = "magic.mtp";
    pub const SPLIT: &str = "magic.prefix_split";
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
    pub const PROMPT: &str = "magic.prompt";
    pub const TEST: &str = "magic.test";

    /// Blocks that hold a part of the model's own settings, on the Setup wire.
    pub const SETUP_BLOCKS: [&str; 9] = [INFO, TEXT, THINKING, EFFORT, INPUT, ANSWERS, ACTION, SETTINGS, PREP];
    /// Setup blocks that switch something on: without one, it is off.
    pub const SWITCHES: [&str; 2] = [THINKING, EFFORT];
    /// Blocks that start a pipeline.
    pub const STARTS: [&str; 3] = [ZERO, BASE, CONFIG];
    /// Blocks that add layers (and take a feed-forward block).
    pub const LAYER_BLOCKS: [&str; 6] = [LAYERS, WINDOW, MLA, CSA, HCA, SSM];
    /// Architecture blocks on the model wire.
    pub const ARCHITECTURE: [&str; 11] = [SIZE, LAYERS, WINDOW, MLA, CSA, HCA, SSM, HYPER, MTP, SPLIT, CONV];
    /// Training stages.
    pub const STAGES: [&str; 5] = [PRETRAIN, FINETUNE, RL, COMPACTION, VISION_TRAIN];
}

/// The wire a model travels along.
pub const MODEL: &str = "model";
/// The wire a model's own settings travel along.
pub const SETUP: &str = "setup";
/// A feed-forward block's output, and the layer blocks' input for it.
pub const FFN: &str = "ffn";
/// An Optimizer block's output, and the training stages' input for it.
pub const OPTIMIZER: &str = "optimizer";

// ---- block settings -----------------------------------------------------------

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

/// A setup block: its part of the model's settings, as JSON.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SetupParams {
    pub value: Value,
    /// What the canvas says about the block (set by the Magic tab).
    pub note: String,
}

/// The name and description of a model, for the Model Info block.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InfoSection {
    pub name: String,
    pub description: String,
    pub author: String,
    pub license: String,
    pub source: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SizeParams {
    pub width: u32,
    pub heads: u32,
    pub kv_heads: u32,
    /// Text: the longest text, in tokens. Vision: the pictures' side, in pixels.
    pub context: u32,
    /// `rope` or `learned`.
    pub positions: String,
    pub rope_theta: f64,
    /// The output head shares the embedding table.
    pub tied: bool,
    /// Embeddings are scaled by √width (Gemma).
    pub embed_scale: bool,
}

impl Default for SizeParams {
    fn default() -> Self {
        Self {
            width: 128,
            heads: 4,
            kv_heads: 2,
            context: 256,
            positions: "rope".into(),
            rope_theta: 10000.0,
            tied: false,
            embed_scale: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LayersParams {
    pub layers: u32,
    /// The plain feed-forward layer's width, times the model's width.
    pub mult: f64,
    /// Shows the finer options in the panel.
    pub more: bool,
    /// RMSNorm on each head's queries and keys (Qwen3, Gemma 3).
    pub qk_norm: bool,
    /// `swiglu` or `geglu`.
    pub activation: String,
    /// `pre` (before each part) or `sandwich` (before and after: Gemma, GLM).
    pub norm: String,
    /// Local layers look only at the last this-many tokens; 0: every layer sees all.
    pub window: u32,
    /// With a window: every this-many-th layer sees all (Gemma 3: 6).
    pub global_every: u32,
    /// The share of each head that rotary positions turn (GLM: 0.5).
    pub rotary: f64,
    /// Biases on queries, keys and values (GLM, Qwen2).
    pub bias: bool,
}

impl Default for LayersParams {
    fn default() -> Self {
        Self {
            layers: 4,
            mult: 2.75,
            more: false,
            qk_norm: false,
            activation: "swiglu".into(),
            norm: "pre".into(),
            window: 0,
            global_every: 6,
            rotary: 1.0,
            bias: false,
        }
    }
}

impl LayersParams {
    /// The attention of one layer, `index` counted over the whole model.
    pub fn attention(&self, index: usize) -> Value {
        let local =
            self.window > 0 && (self.global_every == 0 || !(index + 1).is_multiple_of(self.global_every as usize));
        let mut attention = json!({ "kind": "full" });
        if self.qk_norm {
            attention["qk_norm"] = json!(true);
        }
        if self.bias {
            attention["bias"] = json!(true);
        }
        if self.rotary < 1.0 {
            attention["rotary"] = json!(self.rotary);
        }
        if local {
            attention["window"] = json!(self.window);
        }
        attention
    }
}

/// The Prompt Model block: how requests are asked.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PromptParams {
    /// The context window, in tokens, for GGUF models (0: the model's own).
    pub context: u32,
    /// The longest answer, in tokens (0: the model's own).
    pub max_tokens: u32,
    /// Below 0: the model's own.
    pub temperature: f64,
    /// `model`, `on` or `off`.
    pub thinking: String,
    /// Empty: the model's default.
    pub effort: String,
    /// `auto`, `onnx` (the built-in engine) or `gguf` (llama.cpp).
    pub engine: String,
    pub note: String,
}

impl Default for PromptParams {
    fn default() -> Self {
        Self {
            context: 0,
            max_tokens: 0,
            temperature: -1.0,
            thinking: "model".into(),
            effort: String::new(),
            engine: "auto".into(),
            note: String::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WindowParams {
    pub layers: u32,
    pub window: u32,
    pub mult: f64,
}

impl Default for WindowParams {
    fn default() -> Self {
        Self { layers: 1, window: 64, mult: 2.75 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MlaParams {
    pub layers: u32,
    pub kv_rank: u32,
    pub q_rank: u32,
    pub rope_dim: u32,
    pub mult: f64,
}

impl Default for MlaParams {
    fn default() -> Self {
        Self { layers: 2, kv_rank: 64, q_rank: 96, rope_dim: 16, mult: 2.75 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CsaParams {
    pub layers: u32,
    pub compress: u32,
    pub top_k: u32,
    pub window: u32,
    pub indexer_heads: u32,
    pub indexer_dim: u32,
    pub rope_dim: u32,
    pub mult: f64,
}

impl Default for CsaParams {
    fn default() -> Self {
        Self {
            layers: 1,
            compress: 4,
            top_k: 16,
            window: 32,
            indexer_heads: 4,
            indexer_dim: 32,
            rope_dim: 16,
            mult: 2.75,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HcaParams {
    pub layers: u32,
    pub compress: u32,
    pub window: u32,
    pub rope_dim: u32,
    pub mult: f64,
}

impl Default for HcaParams {
    fn default() -> Self {
        Self { layers: 1, compress: 32, window: 32, rope_dim: 16, mult: 2.75 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SsmParams {
    pub layers: u32,
    pub state: u32,
    pub mult: f64,
}

impl Default for SsmParams {
    fn default() -> Self {
        Self { layers: 2, state: 16, mult: 2.75 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MoeParams {
    pub experts: u32,
    pub shared: u32,
    pub top_k: u32,
    /// Each expert's width, times the model's width.
    pub mult: f64,
    /// `softmax`, `sigmoid` or `sqrt_softplus`.
    pub gating: String,
    /// `aux` (a balance loss) or `bias` (auxiliary-loss-free).
    pub balance: String,
    /// Experts fixed per token by a hash of its id.
    pub hash: bool,
}

impl Default for MoeParams {
    fn default() -> Self {
        Self {
            experts: 8,
            shared: 1,
            top_k: 2,
            mult: 1.0,
            gating: "sigmoid".into(),
            balance: "bias".into(),
            hash: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PersonaParams {
    pub first: String,
    pub first_role: String,
    pub second: String,
    pub second_role: String,
    pub mult: f64,
}

impl Default for PersonaParams {
    fn default() -> Self {
        Self {
            first: "Gun".into(),
            first_role: "Strict code and tool execution".into(),
            second: "Magi".into(),
            second_role: "Breaking loops, critique, routing visual tools".into(),
            mult: 2.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HyperParams {
    pub streams: u32,
    pub sinkhorn: u32,
}

impl Default for HyperParams {
    fn default() -> Self {
        Self { streams: 4, sinkhorn: 20 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MtpParams {
    pub weight: f64,
}

impl Default for MtpParams {
    fn default() -> Self {
        Self { weight: 0.3 }
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
    /// What to do: right answers, good tool calls, text to learn the language from.
    pub positive: Vec<String>,
    /// What not to do: answers to avoid, broken tool calls, loops.
    pub negative: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PretrainParams {
    pub steps: u32,
    pub batch: u32,
    /// 0: always the whole context. Otherwise the context grows from this many tokens.
    pub curriculum_start: u32,
}

impl Default for PretrainParams {
    fn default() -> Self {
        Self { steps: 400, batch: 16, curriculum_start: 0 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StageParams {
    pub steps: u32,
    pub batch: u32,
}

impl Default for StageParams {
    fn default() -> Self {
        Self { steps: 300, batch: 16 }
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
    /// How strongly "what not to do" counts.
    pub negative_weight: f64,
}

impl Default for FinetuneParams {
    fn default() -> Self {
        Self { method: "full".into(), rank: 8, alpha: 16.0, steps: 600, batch: 16, negative_weight: 0.5 }
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
    /// `short`: at most this many tokens. `loop`: the length of the repeated run.
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
    pub rules: Vec<RewardRule>,
}

impl Default for RlParams {
    fn default() -> Self {
        Self {
            steps: 60,
            group: 4,
            max_new: 64,
            temperature: 0.9,
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
    /// Muon's own learning rate, for the layers' weight matrices.
    pub muon_lr: f64,
    pub weight_decay: f64,
    /// `constant` or `cosine`.
    pub schedule: String,
    pub warmup: u32,
    /// Largest gradient norm; 0: no clipping.
    pub clip: f64,
    pub accumulate: u32,
    pub checkpointing: bool,
    pub offload: bool,
}

impl Default for OptimizerParams {
    fn default() -> Self {
        Self {
            name: "adamw".into(),
            lr: 0.002,
            muon_lr: 0.02,
            weight_decay: 0.01,
            schedule: "cosine".into(),
            warmup: 20,
            clip: 1.0,
            accumulate: 1,
            checkpointing: false,
            offload: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CompileParams {
    /// Where the package (and its exports) is saved after training.
    pub folder: String,
    /// The file's name, without `.magic`; empty: the model's name.
    pub file: String,
    pub weights: bool,
    pub datasets: bool,
    pub pictures: bool,
    pub blocks: bool,
    /// Also a `.gguf` file, for llama.cpp, LM Studio and others.
    pub gguf: bool,
    /// Also a Hugging Face folder (`config.json`, `model.safetensors`, tokenizer).
    pub huggingface: bool,
    /// How the GGUF file keeps its matrices: `F16`, `Q8_0` or `Q4_0`.
    pub quant: String,
    /// Also put the GGUF file into LM Studio, with its `model.yaml`.
    pub lmstudio: bool,
}

impl Default for CompileParams {
    fn default() -> Self {
        Self {
            folder: String::new(),
            file: String::new(),
            weights: true,
            datasets: true,
            pictures: true,
            blocks: true,
            gguf: true,
            huggingface: false,
            quant: "F16".into(),
            lmstudio: false,
        }
    }
}

/// Settings of blocks with nothing of their own but a note.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NoteParams {
    pub note: String,
}

// ---- the registry -------------------------------------------------------------

fn model_in(def: NodeDef) -> NodeDef {
    def.data_in(MODEL, "Model", DataType::Any)
}

fn model_out(def: NodeDef) -> NodeDef {
    def.data_out(MODEL, "Model", DataType::Any)
}

fn chain(def: NodeDef) -> NodeDef {
    model_out(model_in(def))
}

fn int(min: i64, max: i64, unit: &str) -> ParamKind {
    ParamKind::int(min, max, unit)
}

fn float(min: f64, max: f64, unit: &str) -> ParamKind {
    ParamKind::float(min, max, unit)
}

fn note_param() -> ParamDef {
    ParamDef::new("note", "Note", ParamKind::Text { multiline: false }).hidden()
}

/// A block's face: its lines, then what the canvas says about it.
fn with_note(node: &Node, mut lines: Vec<String>) -> Vec<String> {
    if let Some(note) = node.param::<String>("note").filter(|n| !n.is_empty()) {
        lines.push(format!("⚠ {note}"));
    }
    lines
}

fn layers_line(layers: u32) -> String {
    if layers == 1 { "1 layer".into() } else { format!("{layers} layers") }
}

fn setup_block(registry: &mut Registry, id: &str, name: &str, description: &str, keywords: &[&str]) {
    registry.register(
        NodeDef::new(id, name, Category::ModelSetup, description)
            .keywords(keywords)
            .data_in(SETUP, "Setup", DataType::Any)
            .data_out(SETUP, "Setup", DataType::Any)
            .param(ParamDef::new("value", "Value", ParamKind::Text { multiline: true }).hidden())
            .param(note_param())
            .defaults(&SetupParams::default())
            .summarize(|node, _| {
                with_note(node, setup_lines(&node.kind, &node.params_or_default::<SetupParams>().value))
            }),
    );
}

/// The Magic tab's blocks.
pub fn registry() -> Registry {
    use kinds::*;
    let mut r = Registry::empty();

    // ---- where to start ----
    let start = |def: NodeDef| model_out(def.data_in(SETUP, "Setup", DataType::Any)).param(note_param());
    r.register(
        start(NodeDef::new(
            ZERO,
            "Start from Zero",
            Category::Start,
            "A brand-new model with random weights. The architecture blocks after it decide its shape; the \
             training stages after those teach it. Plug the setup blocks (Model Info, Text & Tools…) into Setup.",
        ))
        .keywords(&["new", "pretrain", "scratch", "zero", "start", "begin"])
        .param(ParamDef::new("kind", "Model", ParamKind::choice(&[("text", "Text / agent"), ("vision", "Vision")])))
        .defaults(&json!({ "kind": "text", "note": "" }))
        .summarize(|node, _| {
            let p: ZeroParams = node.params_or_default();
            with_note(node, vec![if p.kind == "vision" { "A new vision model" } else { "A new text model" }.into()])
        }),
    );
    r.register(
        start(NodeDef::new(
            BASE,
            "Load Base Model",
            Category::Start,
            "Starts from a trained model's weights, to train it further or fine-tune it: this model's own (leave \
             the path empty), or another .magic file trained in MacroMagi. The architecture blocks after it must \
             be the base model's.",
        ))
        .keywords(&["base", "load", "fine-tune", "adapt", "lora", "weights", "continue", "resume"])
        .param(
            ParamDef::new("path", "Base model", ParamKind::Path { folder: false })
                .help("Empty: this model's own weights."),
        )
        .defaults(&json!({ "path": "", "note": "" }))
        .summarize(|node, _| {
            let p: BaseParams = node.params_or_default();
            let line = if p.path.trim().is_empty() {
                "This model's weights".into()
            } else {
                p.path.rsplit(['/', '\\']).next().unwrap_or_default().to_owned()
            };
            with_note(node, vec![line])
        }),
    );
    r.register(
        start(NodeDef::new(
            CONFIG,
            "Use As It Is",
            Category::Start,
            "The model as it is – one imported from elsewhere, or trained before – without training: change its \
             setup, try it, package it.",
        ))
        .keywords(&["inference", "config", "configure", "try", "existing", "imported", "as is"])
        .defaults(&json!({ "note": "" }))
        .summarize(|node, _| with_note(node, vec!["The model as it is".into()])),
    );

    // ---- the model's setup ----
    setup_block(
        &mut r,
        INFO,
        "Model Info",
        "The model's name, what it does, who made it and its licence. Chain it into the start's Setup input.",
        &["info", "name", "author", "license", "licence", "description"],
    );
    setup_block(
        &mut r,
        TEXT,
        "Text & Tools",
        "Text models: what the model is told before every request (the system prompt), the tools it may call, \
         how freely it writes and how long its answers may be. Chain it into the start's Setup input.",
        &["tools", "prompt", "system", "chat", "temperature", "tokens", "agent"],
    );
    setup_block(
        &mut r,
        THINKING,
        "Thinking",
        "Text models: the model can be asked to think before answering, or not. Chained in, training teaches \
         both (reasoning in the datasets is learned with and without), the Prompt Model block gets a Thinking \
         switch and LM Studio an Enable Thinking toggle. Without it, the model thinks when its datasets did.",
        &["thinking", "reasoning", "think", "toggle", "enable_thinking", "chain of thought"],
    );
    setup_block(
        &mut r,
        EFFORT,
        "Reasoning Effort",
        "Text models: levels of how hard the model thinks, such as low, medium and high. The system text names \
         the level; datasets mark samples with \"effort\". The Prompt Model block and LM Studio get a Reasoning \
         Effort choice.",
        &["effort", "reasoning", "levels", "low", "high", "reasoning_effort"],
    );
    setup_block(
        &mut r,
        INPUT,
        "Screen Area",
        "Vision models: the part of the screen the model looks at in a macro. Chain it into the start's Setup input.",
        &["screen", "area", "capture", "input", "region"],
    );
    setup_block(
        &mut r,
        ANSWERS,
        "Answers",
        "Vision models: the answers the model gives, whether to act on each, how sure it must be and how long to \
         wait. Training learns these answers. Chain it into the start's Setup input.",
        &["answers", "classes", "labels", "confidence", "threshold"],
    );
    setup_block(
        &mut r,
        ACTION,
        "When It Acts",
        "Vision models: the key or mouse button a macro presses when the model acts, how long, and how long \
         to ignore it afterwards. Chain it into the start's Setup input.",
        &["action", "key", "press", "cooldown"],
    );
    setup_block(
        &mut r,
        SETTINGS,
        "Sliders",
        "Vision models: sliders that change several values at once (how sure, how long to wait), offered again \
         in every macro block that uses the model. Chain it into the start's Setup input.",
        &["settings", "sliders", "adjustable"],
    );
    setup_block(
        &mut r,
        PREP,
        "Picture Preparation",
        "Vision models: how a picture is turned into numbers for the model – its size, layout, color order and \
         normalization. Importing and training set this; change it only if a model needs something else.",
        &["normalization", "mean", "std", "resize", "preprocess"],
    );

    // ---- architecture ----
    let ffn_in = |def: NodeDef| def.data_in(FFN, "Feed-forward", DataType::Any);
    r.register(
        chain(NodeDef::new(
            SIZE,
            "Model Size",
            Category::Architecture,
            "How big the model is: its width (numbers per token), attention heads, key/value heads (fewer is \
             smaller and faster), and its context – the longest text it reads, in tokens (or, for a vision model, \
             the pictures' side in pixels). Put it first, after the start.",
        ))
        .keywords(&["size", "width", "heads", "context", "dimensions", "hidden", "gqa", "rope"])
        .param(
            ParamDef::new("width", "Width", int(16, 8192, ""))
                .help("Numbers per token. Bigger learns more, and is slower."),
        )
        .param(ParamDef::new("heads", "Attention heads", int(1, 128, "")).help("The width must divide by it."))
        .param(
            ParamDef::new("kv_heads", "Key/value heads", int(1, 128, ""))
                .help("Heads must divide by it. Equal to the heads: plain attention."),
        )
        .param(ParamDef::new("context", "Context window", int(8, 32768, "")).help(
            "Text models: the longest text the model reads at once, in tokens (here a token is a byte of \
                     text or a tag, so 1024 is about a page). Training and the exports use it; the Prompt Model \
                     block can run GGUF models with a different one. Vision models: the side of the pictures, in \
                     pixels.",
        ))
        .param(
            ParamDef::new(
                "positions",
                "Positions",
                ParamKind::choice(&[("rope", "Rotary (RoPE)"), ("learned", "Learned")]),
            )
            .help("Rotary is what Llama and DeepSeek use, and what GGUF needs."),
        )
        .param(ParamDef::new("rope_theta", "Rotary base", float(100.0, 10_000_000.0, "")).show_if("positions", "rope"))
        .param(
            ParamDef::new("tied", "Tie input and output", ParamKind::Bool)
                .help("The output head reuses the embedding table: fewer weights (Qwen3 small models, Gemma)."),
        )
        .param(
            ParamDef::new("embed_scale", "Scale embeddings", ParamKind::Bool)
                .help("Embeddings are multiplied by √width before the first layer (Gemma)."),
        )
        .defaults(&SizeParams::default())
        .summarize(|node, _| {
            let p: SizeParams = node.params_or_default();
            vec![
                format!("{} wide · {} heads ({} kv)", p.width, p.heads, p.kv_heads),
                format!("context window {} tokens", p.context),
            ]
        }),
    );
    let mult = || {
        ParamDef::new("mult", "Feed-forward width", float(0.25, 16.0, "× width"))
            .help("The plain feed-forward layer, when no Mixture of Experts or Persona Router is wired in.")
    };
    r.register(
        ffn_in(chain(NodeDef::new(
            LAYERS,
            "Transformer Layers",
            Category::Architecture,
            "Standard layers (the Llama layout): every token looks at every earlier token (attention with rotary \
             positions), then a SwiGLU feed-forward layer – or the experts wired into Feed-forward. Models made \
             only of these export to GGUF for LM Studio and llama.cpp.",
        )))
        .keywords(&["transformer", "attention", "layers", "llama", "backbone", "dense", "gqa"])
        .param(ParamDef::new("layers", "Layers", int(1, 256, "")))
        .param(mult())
        .param(
            ParamDef::new("more", "Finer options", ParamKind::Bool)
                .help("What sets Qwen, Gemma, GLM and Mistral apart from Llama."),
        )
        .param(
            ParamDef::new("qk_norm", "Normalize queries and keys", ParamKind::Bool)
                .help("RMSNorm on each head's queries and keys before they turn: steadier training (Qwen3, Gemma 3).")
                .show_if("more", true),
        )
        .param(
            ParamDef::new(
                "activation",
                "Feed-forward gate",
                ParamKind::choice(&[("swiglu", "SwiGLU (Llama, Qwen, Mistral)"), ("geglu", "GeGLU (Gemma)")]),
            )
            .show_if("more", true),
        )
        .param(
            ParamDef::new(
                "norm",
                "Normalization",
                ParamKind::choice(&[
                    ("pre", "Before each part (Llama, Qwen)"),
                    ("sandwich", "Before and after each part (Gemma, GLM)"),
                ]),
            )
            .show_if("more", true),
        )
        .param(
            ParamDef::new("window", "Local window", int(0, 32768, "tokens"))
                .help(
                    "0: every layer sees the whole text. Otherwise most layers look only at this many recent \
                     tokens – cheaper for long texts – and every few layers see everything (Gemma 3).",
                )
                .show_if("more", true),
        )
        .param(
            ParamDef::new("global_every", "Every Nth layer sees all", int(0, 64, ""))
                .help("With a local window: Gemma 3 has five local layers, then a global one (6).")
                .show_if("more", true),
        )
        .param(
            ParamDef::new("rotary", "Rotary share", float(0.1, 1.0, ""))
                .help("How much of each head rotary positions turn: 1 for most models, 0.5 for GLM.")
                .show_if("more", true),
        )
        .param(
            ParamDef::new("bias", "Biases on queries, keys, values", ParamKind::Bool)
                .help("GLM and Qwen2 have them.")
                .show_if("more", true),
        )
        .param(note_param())
        .defaults(&json!({ "layers": 4, "mult": 2.75, "more": false, "qk_norm": false, "activation": "swiglu",
                           "norm": "pre", "window": 0, "global_every": 6, "rotary": 1.0, "bias": false, "note": "" }))
        .summarize(|node, _| {
            let p: LayersParams = node.params_or_default();
            let mut traits = Vec::new();
            if p.qk_norm {
                traits.push("QK-norm".to_owned());
            }
            if p.activation == "geglu" {
                traits.push("GeGLU".to_owned());
            }
            if p.norm == "sandwich" {
                traits.push("sandwich norms".to_owned());
            }
            if p.window > 0 {
                traits.push(format!("{}-token window", p.window));
            }
            if p.rotary < 1.0 {
                traits.push(format!("{:.0}% rotary", p.rotary * 100.0));
            }
            let mut lines = vec![layers_line(p.layers)];
            if !traits.is_empty() {
                lines.push(traits.join(" · "));
            }
            with_note(node, lines)
        }),
    );
    r.register(
        ffn_in(chain(NodeDef::new(
            WINDOW,
            "Sliding Window Attention",
            Category::Architecture,
            "Layers whose attention looks only at the last few tokens: cheap for long texts. DeepSeek-V4 starts \
             with these.",
        )))
        .keywords(&["sliding", "window", "local", "attention", "swa"])
        .param(ParamDef::new("layers", "Layers", int(1, 256, "")))
        .param(ParamDef::new("window", "Window", int(1, 32768, "tokens")))
        .param(mult())
        .param(note_param())
        .defaults(&json!({ "layers": 1, "window": 64, "mult": 2.75, "note": "" }))
        .summarize(|node, _| {
            let p: WindowParams = node.params_or_default();
            with_note(node, vec![layers_line(p.layers), format!("last {} tokens", p.window)])
        }),
    );
    r.register(
        ffn_in(chain(NodeDef::new(
            MLA,
            "Latent Attention (MLA)",
            Category::Architecture,
            "DeepSeek-V2/V3's multi-head latent attention: keys and values are rebuilt from a small latent per \
             token, so a model keeps far less in memory while it writes. A separate rotary part carries positions.",
        )))
        .keywords(&["mla", "latent", "attention", "deepseek", "v3", "kv", "cache", "compressed"])
        .param(ParamDef::new("layers", "Layers", int(1, 256, "")))
        .param(
            ParamDef::new("kv_rank", "Key/value latent", int(4, 4096, ""))
                .help("Numbers kept per token for keys and values."),
        )
        .param(ParamDef::new("q_rank", "Query latent", int(0, 4096, "")).help("0: queries straight from the token."))
        .param(ParamDef::new("rope_dim", "Rotary part", int(2, 512, "")).help("Channels that carry positions."))
        .param(mult())
        .param(note_param())
        .defaults(&json!({ "layers": 2, "kv_rank": 64, "q_rank": 96, "rope_dim": 16, "mult": 2.75, "note": "" }))
        .summarize(|node, _| {
            let p: MlaParams = node.params_or_default();
            with_note(node, vec![layers_line(p.layers), format!("latent {}", p.kv_rank)])
        }),
    );
    r.register(
        ffn_in(chain(NodeDef::new(
            CSA,
            "Compressed Sparse Attention",
            Category::Architecture,
            "DeepSeek-V4's CSA: every few tokens are pooled into one entry, a small “lightning indexer” picks the \
             entries that matter to each token, and the token attends to those plus a window of recent tokens. \
             Built for very long texts.",
        )))
        .keywords(&["csa", "sparse", "compressed", "attention", "deepseek", "v4", "indexer", "dsa"])
        .param(ParamDef::new("layers", "Layers", int(1, 256, "")))
        .param(ParamDef::new("compress", "Pool every", int(1, 1024, "tokens")))
        .param(
            ParamDef::new("top_k", "Entries per token", int(1, 4096, ""))
                .help("How many pooled entries the indexer picks for each token."),
        )
        .param(ParamDef::new("window", "Recent tokens", int(1, 32768, "")))
        .param(ParamDef::new("indexer_heads", "Indexer heads", int(1, 128, "")))
        .param(ParamDef::new("indexer_dim", "Indexer size", int(4, 1024, "")))
        .param(ParamDef::new("rope_dim", "Rotary part", int(2, 512, "")))
        .param(mult())
        .param(note_param())
        .defaults(
            &json!({ "layers": 1, "compress": 4, "top_k": 16, "window": 32, "indexer_heads": 4, "indexer_dim": 32,
                           "rope_dim": 16, "mult": 2.75, "note": "" }),
        )
        .summarize(|node, _| {
            let p: CsaParams = node.params_or_default();
            with_note(
                node,
                vec![layers_line(p.layers), format!("pool {} · top {} · window {}", p.compress, p.top_k, p.window)],
            )
        }),
    );
    r.register(
        ffn_in(chain(NodeDef::new(
            HCA,
            "Heavily Compressed Attention",
            Category::Architecture,
            "DeepSeek-V4's HCA: many tokens are pooled into each entry, and every token attends to all the pooled \
             entries before it plus a window of recent tokens. DeepSeek-V4 alternates it with Compressed Sparse \
             Attention.",
        )))
        .keywords(&["hca", "compressed", "attention", "deepseek", "v4", "pooled"])
        .param(ParamDef::new("layers", "Layers", int(1, 256, "")))
        .param(ParamDef::new("compress", "Pool every", int(1, 4096, "tokens")))
        .param(ParamDef::new("window", "Recent tokens", int(1, 32768, "")))
        .param(ParamDef::new("rope_dim", "Rotary part", int(2, 512, "")))
        .param(mult())
        .param(note_param())
        .defaults(&json!({ "layers": 1, "compress": 32, "window": 32, "rope_dim": 16, "mult": 2.75, "note": "" }))
        .summarize(|node, _| {
            let p: HcaParams = node.params_or_default();
            with_note(node, vec![layers_line(p.layers), format!("pool {} · window {}", p.compress, p.window)])
        }),
    );
    r.register(
        ffn_in(chain(NodeDef::new(
            SSM,
            "Mamba-2 / SSM",
            Category::Architecture,
            "State-space layers (Mamba-2): instead of attention, each token updates a small memory that fades over \
             time. Good at long, flowing text; often mixed with attention layers.",
        )))
        .keywords(&["mamba", "ssm", "state space", "recurrent", "linear"])
        .param(ParamDef::new("layers", "Layers", int(1, 256, "")))
        .param(ParamDef::new("state", "Memory size", int(2, 512, "")))
        .param(mult())
        .param(note_param())
        .defaults(&json!({ "layers": 2, "state": 16, "mult": 2.75, "note": "" }))
        .summarize(|node, _| {
            let p: SsmParams = node.params_or_default();
            with_note(node, vec![layers_line(p.layers), format!("memory {}", p.state)])
        }),
    );
    r.register(
        NodeDef::new(
            MOE,
            "Mixture of Experts",
            Category::Architecture,
            "DeepSeekMoE: replaces the feed-forward layer of the layer blocks it is wired into. Shared experts \
             see every token; each token also picks a few of many small routed experts, so the model holds a lot \
             but uses little per token. Wire its output into a layer block's Feed-forward input.",
        )
        .keywords(&["moe", "experts", "router", "mixture", "deepseek", "routed", "shared", "sparse"])
        .data_out(FFN, "Feed-forward", DataType::Any)
        .param(ParamDef::new("experts", "Routed experts", int(1, 1024, "")))
        .param(ParamDef::new("shared", "Shared experts", int(0, 16, "")))
        .param(ParamDef::new("top_k", "Experts per token", int(1, 64, "")))
        .param(ParamDef::new("mult", "Expert width", float(0.125, 16.0, "× width")))
        .param(
            ParamDef::new(
                "gating",
                "Choosing experts",
                ParamKind::choice(&[
                    ("softmax", "Softmax (V2)"),
                    ("sigmoid", "Sigmoid (V3)"),
                    ("sqrt_softplus", "√Softplus (V4)"),
                ]),
            )
            .help("How a token scores the experts. Sigmoid and √Softplus are normalized over the chosen ones."),
        )
        .param(
            ParamDef::new(
                "balance",
                "Balancing",
                ParamKind::choice(&[("bias", "Bias, no extra loss (V3/V4)"), ("aux", "Balance loss")]),
            )
            .help("Keeps tokens spread over the experts."),
        )
        .param(
            ParamDef::new("hash", "Fixed experts per token (hash)", ParamKind::Bool)
                .help("DeepSeek-V4's first layers: each token always uses the same experts."),
        )
        .param(note_param())
        .defaults(&json!({ "experts": 8, "shared": 1, "top_k": 2, "mult": 1.0, "gating": "sigmoid", "balance": "bias",
                           "hash": false, "note": "" }))
        .summarize(|node, _| {
            let p: MoeParams = node.params_or_default();
            let routing = if p.hash { " · hashed" } else { "" };
            with_note(
                node,
                vec![format!("{} routed + {} shared", p.experts, p.shared), format!("{} per token{routing}", p.top_k)],
            )
        }),
    );
    r.register(
        NodeDef::new(
            PERSONA,
            "Dual-Persona Router",
            Category::Architecture,
            "Two personas of the model's reasoning, each written inside its own tag and with its own feed-forward \
             expert: by default Gun (strict code and tool calls) and Magi (critique, breaking loops). Tool calls \
             keep one syntax. Wire its output into a layer block's Feed-forward input.",
        )
        .keywords(&["persona", "thinking", "reasoning", "gun", "magi", "router", "dual"])
        .data_out(FFN, "Feed-forward", DataType::Any)
        .param(ParamDef::new("first", "First persona", ParamKind::Text { multiline: false }))
        .param(ParamDef::new("first_role", "Its role", ParamKind::Text { multiline: false }))
        .param(ParamDef::new("second", "Second persona", ParamKind::Text { multiline: false }))
        .param(ParamDef::new("second_role", "Its role", ParamKind::Text { multiline: false }))
        .param(ParamDef::new("mult", "Expert width", float(0.25, 16.0, "× width")))
        .param(note_param())
        .defaults(&json!({ "first": "Gun", "first_role": "Strict code and tool execution", "second": "Magi",
                           "second_role": "Breaking loops, critique, routing visual tools", "mult": 2.0, "note": "" }))
        .summarize(|node, _| {
            let p: PersonaParams = node.params_or_default();
            with_note(node, vec![format!("<{}> & <{}>", tag_of(&p.first), tag_of(&p.second))])
        }),
    );
    r.register(
        chain(NodeDef::new(
            HYPER,
            "Hyper-Connections (mHC)",
            Category::Architecture,
            "DeepSeek-V4's manifold-constrained hyper-connections: the model carries several parallel streams \
             instead of one, mixed by learned, balanced weights around every layer. Steadier training for deep \
             models. Applies to the whole model.",
        ))
        .keywords(&["mhc", "hyper", "connections", "residual", "deepseek", "v4", "streams"])
        .param(ParamDef::new("streams", "Streams", int(2, 16, "")))
        .param(ParamDef::new("sinkhorn", "Balancing rounds", int(1, 100, "")))
        .defaults(&HyperParams::default())
        .summarize(|node, _| vec![format!("{} streams", node.params_or_default::<HyperParams>().streams)]),
    );
    r.register(
        chain(NodeDef::new(
            MTP,
            "Multi-Token Prediction",
            Category::Architecture,
            "While training, an extra small layer also predicts the token after next, which teaches the model \
             to plan ahead (DeepSeek-V3/V4). It is left out of the finished model.",
        ))
        .keywords(&["mtp", "multi-token", "prediction", "deepseek", "lookahead"])
        .param(ParamDef::new("weight", "Weight", float(0.0, 2.0, "")).help("How much the extra prediction counts."))
        .defaults(&MtpParams::default())
        .summarize(|node, _| vec![format!("weight {}", node.params_or_default::<MtpParams>().weight)]),
    );
    r.register(
        chain(NodeDef::new(
            SPLIT,
            "Encoder/Decoder Split",
            Category::Architecture,
            "The request (everything up to the answer) is read in both directions at once, and only the answer is \
             written token by token. Applies to the whole model.",
        ))
        .keywords(&["prefix", "encoder", "decoder", "split", "bidirectional"])
        .defaults(&Map::new())
        .summarize(|_, _| vec!["Request both ways".into()]),
    );
    r.register(
        chain(NodeDef::new(
            CONV,
            "Vision Encoder",
            Category::Architecture,
            "Vision models: convolution layers that turn a picture into features, then the answers.",
        ))
        .keywords(&["vision", "cnn", "convolution", "encoder", "image"])
        .param(ParamDef::new("layers", "Layers", int(1, 8, "")))
        .param(ParamDef::new("channels", "Channels", int(4, 512, "")))
        .defaults(&ConvParams::default())
        .summarize(|node, _| {
            let p: ConvParams = node.params_or_default();
            vec![format!("{} · {} channels", layers_line(p.layers), p.channels)]
        }),
    );

    // ---- data ----
    r.register(
        NodeDef::new(
            TEXT_DATA,
            "Text Dataset",
            Category::Data,
            "Text the model learns from, kept in the .magic file: .jsonl requests and answers, conversations and \
             tool calls, and .md or .txt text. “What to do” feeds Pretraining and Fine-Tuning; “What not to do” \
             feeds Fine-Tuning's input of the same name. Select it to add, write and check samples.",
        )
        .keywords(&["dataset", "text", "jsonl", "markdown", "samples", "data", "negative", "examples"])
        .data_out("positive", "What to do", DataType::Any)
        .data_out("negative", "What not to do", DataType::Any)
        .param(ParamDef::new("positive", "What to do", ParamKind::Text { multiline: false }).hidden())
        .param(ParamDef::new("negative", "What not to do", ParamKind::Text { multiline: false }).hidden())
        .param(note_param())
        .defaults(&json!({ "positive": [], "negative": [], "note": "" }))
        .summarize(|node, _| {
            let p: TextDataParams = node.params_or_default();
            let files = |n: usize| if n == 1 { "1 file".to_owned() } else { format!("{n} files") };
            with_note(
                node,
                vec![format!("To do: {}", files(p.positive.len())), format!("Not to do: {}", files(p.negative.len()))],
            )
        }),
    );
    r.register(
        NodeDef::new(
            IMAGE_DATA,
            "Vision Dataset",
            Category::Data,
            "Pictures the model learns from: pictures of each answer, and pictures an answer was mistaken for. \
             Wire it into Vision Training. Select it to add and remove pictures.",
        )
        .keywords(&["dataset", "pictures", "images", "vision", "positive", "negative"])
        .data_out("pictures", "Pictures", DataType::Any)
        .param(note_param())
        .defaults(&json!({ "note": "" }))
        .summarize(|node, _| with_note(node, vec!["The model's pictures".into()])),
    );

    // ---- training ----
    let stage = |def: NodeDef| chain(def).data_in(OPTIMIZER, "Optimizer", DataType::Any).param(note_param());
    r.register(
        stage(
            NodeDef::new(
                PRETRAIN,
                "Pretraining",
                Category::Training,
                "Teaches a new model the language: it reads all the text in its dataset and learns to predict each \
                 next token. Comes first for a model started from zero.",
            )
            .data_in("data", "Text", DataType::Any),
        )
        .keywords(&["pretrain", "pretraining", "language model", "tokens", "next token"])
        .param(ParamDef::new("steps", "Steps", int(1, 10_000_000, "")).help("How many batches to learn from."))
        .param(ParamDef::new("batch", "Batch", int(1, 4096, "")).help("Texts learned from at once."))
        .param(
            ParamDef::new("curriculum_start", "Start the context at", int(0, 32768, "tokens"))
                .help("0: always the whole context. Otherwise texts start this short and grow to the full context."),
        )
        .defaults(&json!({ "steps": 400, "batch": 16, "curriculum_start": 0, "note": "" }))
        .summarize(|node, _| {
            let p: PretrainParams = node.params_or_default();
            with_note(node, vec![format!("{} steps · batch {}", p.steps, p.batch)])
        }),
    );
    r.register(
        stage(
            NodeDef::new(
                FINETUNE,
                "Fine-Tuning",
                Category::Training,
                "Teaches the model to answer: the requests and answers in “What to do” become more likely, and the \
                 answers in “What not to do” less likely. Train all weights, or small LoRA or DoRA adapters that \
                 are folded back in afterwards.",
            )
            .data_in("data", "What to do", DataType::Any)
            .data_in("negative", "What not to do", DataType::Any),
        )
        .keywords(&["fine-tune", "finetune", "sft", "lora", "dora", "adapter", "instruct"])
        .param(ParamDef::new("method", "Train", ParamKind::choice(&[("full", "All weights"), ("lora", "LoRA adapters"), ("dora", "DoRA adapters")])))
        .param(ParamDef::new("rank", "Adapter rank", int(1, 512, "")).help("Bigger adapters learn more.").show_if("method", "lora"))
        .param(ParamDef::new("alpha", "Adapter strength", float(0.1, 512.0, "")).show_if("method", "lora"))
        .param(ParamDef::new("steps", "Steps", int(1, 10_000_000, "")))
        .param(ParamDef::new("batch", "Batch", int(1, 4096, "")))
        .param(ParamDef::new("negative_weight", "“Not to do” weight", float(0.0, 10.0, "")).help("How strongly answers to avoid count."))
        .defaults(&json!({ "method": "full", "rank": 8, "alpha": 16.0, "steps": 600, "batch": 16, "negative_weight": 0.5, "note": "" }))
        .summarize(|node, _| {
            let p: FinetuneParams = node.params_or_default();
            let method = match p.method.as_str() {
                "lora" => format!("LoRA rank {}", p.rank),
                "dora" => format!("DoRA rank {}", p.rank),
                _ => "All weights".into(),
            };
            with_note(node, vec![format!("{method} · {} steps", p.steps)])
        }),
    );
    r.register(
        stage(
            NodeDef::new(
                RL,
                "Reinforcement Learning",
                Category::Training,
                "The model answers each request several times; its reward rules score the answers (right answer, \
                 good tool calls, short, no loops…), and the better-than-average answers become more likely. Comes \
                 after fine-tuning.",
            )
            .data_in("data", "Requests", DataType::Any),
        )
        .keywords(&["rl", "reinforcement", "reward", "grpo", "score", "rlhf"])
        .param(ParamDef::new("steps", "Steps", int(1, 1_000_000, "")))
        .param(ParamDef::new("group", "Answers per request", int(2, 64, "")))
        .param(ParamDef::new("max_new", "Longest answer", int(1, 8192, "tokens")))
        .param(
            ParamDef::new("temperature", "Temperature", float(0.05, 2.0, "")).help("How varied the tried answers are."),
        )
        .param(ParamDef::new("rules", "Reward rules", ParamKind::Text { multiline: false }).hidden())
        .defaults(&{
            let mut v = serde_json::to_value(RlParams::default()).expect("serializable");
            v["note"] = json!("");
            v
        })
        .summarize(|node, _| {
            let p: RlParams = node.params_or_default();
            with_note(
                node,
                vec![
                    format!("{} steps · {} answers each", p.steps, p.group),
                    format!("{} reward rules", p.rules.len()),
                ],
            )
        }),
    );
    r.register(
        stage(
            NodeDef::new(
                COMPACTION,
                "Context Compaction",
                Category::Training,
                "Teaches the model to rewrite a long context – a conversation, a log – into a short one that keeps \
                 what matters, from {\"context\", \"compact\"} samples.",
            )
            .data_in("data", "Samples", DataType::Any),
        )
        .keywords(&["compaction", "compact", "summarize", "context", "memory"])
        .param(ParamDef::new("steps", "Steps", int(1, 10_000_000, "")))
        .param(ParamDef::new("batch", "Batch", int(1, 4096, "")))
        .defaults(&json!({ "steps": 300, "batch": 16, "note": "" }))
        .summarize(|node, _| {
            let p: StageParams = node.params_or_default();
            with_note(node, vec![format!("{} steps · batch {}", p.steps, p.batch)])
        }),
    );
    r.register(
        stage(
            NodeDef::new(
                VISION_TRAIN,
                "Vision Training",
                Category::Training,
                "Teaches a vision model its answers from the pictures of a Vision Dataset; pictures an answer was \
                 mistaken for make it less likely there.",
            )
            .data_in("pictures", "Pictures", DataType::Any),
        )
        .keywords(&["vision", "train", "classifier", "pictures"])
        .param(ParamDef::new("steps", "Steps", int(1, 10_000_000, "")))
        .param(ParamDef::new("batch", "Batch", int(1, 4096, "")))
        .defaults(&json!({ "steps": 300, "batch": 16, "note": "" }))
        .summarize(|node, _| {
            let p: StageParams = node.params_or_default();
            with_note(node, vec![format!("{} steps · batch {}", p.steps, p.batch)])
        }),
    );
    r.register(
        NodeDef::new(
            OPTIMIZER,
            "Optimizer",
            Category::Training,
            "How the training stages it is wired into learn: the optimizer (AdamW, or Muon as DeepSeek-V4 uses, \
             with AdamW for embeddings and norms), the learning rate and its schedule, and ways to save memory. \
             Wire its output into each stage's Optimizer input; one block can feed several stages.",
        )
        .keywords(&["optimizer", "muon", "adamw", "memory", "vram", "offload", "learning rate", "schedule", "warmup"])
        .data_out(OPTIMIZER, "Optimizer", DataType::Any)
        .param(ParamDef::new(
            "name",
            "Optimizer",
            ParamKind::choice(&[("adamw", "AdamW"), ("muon", "Muon (+ AdamW)"), ("sgd", "SGD")]),
        ))
        .param(
            ParamDef::new("lr", "Learning rate", float(0.000_001, 1.0, ""))
                .help("How big each step is. Too high: the loss jumps around; too low: it barely moves."),
        )
        .param(ParamDef::new("muon_lr", "Muon learning rate", float(0.000_01, 1.0, "")).show_if("name", "muon"))
        .param(ParamDef::new("weight_decay", "Weight decay", float(0.0, 1.0, "")))
        .param(ParamDef::new(
            "schedule",
            "Schedule",
            ParamKind::choice(&[("cosine", "Cosine decay"), ("constant", "Constant")]),
        ))
        .param(ParamDef::new("warmup", "Warm up over", int(0, 1_000_000, "steps")))
        .param(ParamDef::new("clip", "Clip gradients at", float(0.0, 100.0, "")).help("0: no clipping."))
        .param(
            ParamDef::new("accumulate", "Accumulate", int(1, 1024, "batches"))
                .help("Learns from several batches per step: a bigger batch in the same memory."),
        )
        .param(ParamDef::new("checkpointing", "Recompute activations (less memory)", ParamKind::Bool))
        .param(
            ParamDef::new("offload", "Optimizer state in system memory", ParamKind::Bool)
                .help("Less graphics memory, a little slower."),
        )
        .param(note_param())
        .defaults(&{
            let mut v = serde_json::to_value(OptimizerParams::default()).expect("serializable");
            v["note"] = json!("");
            v
        })
        .summarize(|node, _| {
            let p: OptimizerParams = node.params_or_default();
            let name = match p.name.as_str() {
                "muon" => "Muon",
                "sgd" => "SGD",
                _ => "AdamW",
            };
            with_note(node, vec![format!("{name} · {}", p.lr), format!("{} schedule", p.schedule)])
        }),
    );

    // ---- packaging ----
    r.register(
        chain(NodeDef::new(
            COMPILE,
            "Compile .magic Package",
            Category::Package,
            "Saves the finished model after training: a .magic file (the model, its weights, tokens, tools, \
             architecture and setup) in the folder you choose, and if you like a .gguf file for LM Studio and \
             llama.cpp and a Hugging Face folder. Training needs a folder here.",
        ))
        .keywords(&["compile", "package", "export", "save", "magic", "gguf", "huggingface", "bundle", "folder"])
        .param(ParamDef::new("folder", "Save into", ParamKind::Path { folder: true }))
        .param(
            ParamDef::new("file", "File name", ParamKind::Text { multiline: false })
                .help("Without .magic. Empty: the model's name."),
        )
        .param(ParamDef::new("weights", "Weights, to train further", ParamKind::Bool))
        .param(ParamDef::new("datasets", "Text datasets", ParamKind::Bool))
        .param(ParamDef::new("pictures", "Pictures", ParamKind::Bool))
        .param(ParamDef::new("blocks", "Blocks to use it with", ParamKind::Bool))
        .param(
            ParamDef::new("gguf", "Also a .gguf (LM Studio, llama.cpp)", ParamKind::Bool)
                .help("For models of Transformer Layers in the Llama, Qwen3 or Gemma 3 layout."),
        )
        .param(
            ParamDef::new(
                "quant",
                "GGUF precision",
                ParamKind::choice(&[
                    ("F16", "F16 – exact, the largest"),
                    ("Q8_0", "Q8_0 – half the size, nearly exact"),
                    ("Q4_0", "Q4_0 – a quarter of the size, loses some quality"),
                ]),
            )
            .help("How the .gguf keeps its weights. LM Studio shows it as the quantization.")
            .show_if("gguf", true),
        )
        .param(
            ParamDef::new("lmstudio", "Also add it to LM Studio", ParamKind::Bool)
                .help("Copies the .gguf into LM Studio's models, with a model.yaml for its controls.")
                .show_if("gguf", true),
        )
        .param(
            ParamDef::new("huggingface", "Also a Hugging Face folder", ParamKind::Bool)
                .help("config.json, model.safetensors and the tokenizer (Llama and Qwen3 layouts)."),
        )
        .param(note_param())
        .defaults(&{
            let mut v = serde_json::to_value(CompileParams::default()).expect("serializable");
            v["note"] = json!("");
            v
        })
        .summarize(|node, _| {
            let p: CompileParams = node.params_or_default();
            let place = if p.folder.trim().is_empty() {
                "No folder chosen".to_owned()
            } else {
                format!("Into {}", p.folder.rsplit(['/', '\\']).find(|s| !s.is_empty()).unwrap_or_default())
            };
            let mut formats = vec![".magic"];
            let gguf = format!(".gguf {}", p.quant);
            if p.gguf {
                formats.push(&gguf);
            }
            if p.gguf && p.lmstudio {
                formats.push("LM Studio");
            }
            if p.huggingface {
                formats.push("Hugging Face");
            }
            with_note(node, vec![place, formats.join(" · ")])
        }),
    );
    r.register(
        model_in(NodeDef::new(
            PROMPT,
            "Prompt Model",
            Category::Package,
            "Talk to the trained text model: write a request and see its answer, what it thought and the tools it \
             called. Wire it after Compile .magic Package.",
        ))
        .keywords(&["prompt", "try", "test", "chat", "ask", "talk", "context", "load"])
        .param(ParamDef::new("context", "Context window", int(0, 1_048_576, "tokens")).hidden())
        .param(ParamDef::new("max_tokens", "Longest answer", int(0, 65536, "tokens")).hidden())
        .param(ParamDef::new("temperature", "Temperature", float(-1.0, 2.0, "")).hidden())
        .param(ParamDef::new("thinking", "Thinking", ParamKind::Text { multiline: false }).hidden())
        .param(ParamDef::new("effort", "Reasoning effort", ParamKind::Text { multiline: false }).hidden())
        .param(ParamDef::new("engine", "Run with", ParamKind::Text { multiline: false }).hidden())
        .param(note_param())
        .defaults(&PromptParams::default())
        .summarize(|node, _| {
            let p: PromptParams = node.params_or_default();
            let context = if p.context == 0 {
                "Context window: the model's".to_owned()
            } else {
                format!("Context window: {} tokens", p.context)
            };
            with_note(node, vec!["Ask it something".into(), context])
        }),
    );
    r.register(
        model_in(NodeDef::new(
            TEST,
            "Test Model",
            Category::Package,
            "Try the vision model: let it watch the screen live, or show it a picture, and see its answers. Wire \
             it after Compile .magic Package.",
        ))
        .keywords(&["test", "try", "live", "watch", "picture"])
        .param(note_param())
        .defaults(&json!({ "note": "" }))
        .summarize(|node, _| with_note(node, vec!["Watch the screen".into()])),
    );
    r
}

/// A persona's tag from its name: `Gun` → `gun`.
pub fn tag_of(name: &str) -> String {
    let tag: String = name.trim().to_lowercase().chars().map(|c| if c.is_alphanumeric() { c } else { '_' }).collect();
    if tag.is_empty() { "persona".into() } else { tag }
}

// ---- setup sections -----------------------------------------------------------

/// The part of a model a setup block holds, as JSON.
pub fn section_of(kind: &str, model: &MagicModel) -> Value {
    fn to<T: Serialize>(value: &T) -> Value {
        serde_json::to_value(value).unwrap_or(Value::Null)
    }
    match kind {
        kinds::INFO => to(&InfoSection {
            name: model.name.clone(),
            description: model.description.clone(),
            author: model.author.clone(),
            license: model.license.clone(),
            source: model.source.clone(),
        }),
        kinds::TEXT => to(&model.text.clone().unwrap_or_default()),
        kinds::THINKING => to(&model.text.as_ref().and_then(|t| t.thinking.clone()).unwrap_or_default()),
        kinds::EFFORT => to(&model.text.as_ref().and_then(|t| t.effort.clone()).unwrap_or_default()),
        kinds::INPUT => to(&model.capture),
        kinds::ANSWERS => to(&model.classes),
        kinds::ACTION => to(&model.action),
        kinds::SETTINGS => to(&model.settings),
        kinds::PREP => to(&model.input),
        _ => Value::Null,
    }
}

fn parse<T: serde::de::DeserializeOwned + Default>(value: &Value) -> T {
    serde_json::from_value(value.clone()).unwrap_or_default()
}

/// Puts a setup block's part into a model.
pub fn apply_section(kind: &str, value: &Value, model: &mut MagicModel) {
    match kind {
        kinds::INFO => {
            let info: InfoSection = parse(value);
            (model.name, model.description, model.author, model.license, model.source) =
                (info.name, info.description, info.author, info.license, info.source);
        }
        kinds::TEXT => {
            let mut text: TextSpec = if value.is_null() { TextSpec::default() } else { parse(value) };
            // The canvas decides these; the block's copy may be stale.
            if let Some(current) = &model.text {
                text.context = current.context;
                text.personas = current.personas.clone();
            }
            text.ensure_specials();
            model.text = Some(text);
        }
        kinds::THINKING => {
            if let Some(text) = &mut model.text {
                text.thinking = Some(if value.is_null() { ThinkingSpec::default() } else { parse(value) });
            }
        }
        kinds::EFFORT => {
            if let Some(text) = &mut model.text {
                let mut effort: EffortSpec = if value.is_null() { EffortSpec::default() } else { parse(value) };
                effort.levels.retain(|l| !l.trim().is_empty());
                if effort.levels.is_empty() {
                    effort = EffortSpec::default();
                }
                if !effort.levels.contains(&effort.default) {
                    effort.default = effort.levels[effort.levels.len() / 2].clone();
                }
                text.effort = Some(effort);
            }
        }
        kinds::INPUT => model.capture = parse::<CaptureSpec>(value),
        kinds::ANSWERS => model.classes = parse::<Vec<ClassSpec>>(value),
        kinds::ACTION => model.action = parse::<ActionSpec>(value),
        kinds::SETTINGS => model.settings = parse::<Vec<ModelSetting>>(value),
        kinds::PREP => model.input = parse::<InputSpec>(value),
        _ => {}
    }
}

/// What a setup block's part says, in a few lines for its face.
pub fn setup_lines(kind: &str, value: &Value) -> Vec<String> {
    match kind {
        kinds::INFO => {
            let info: InfoSection = parse(value);
            let name = if info.name.trim().is_empty() { "Untitled model".to_owned() } else { info.name };
            let mut lines = vec![name];
            if !info.license.is_empty() {
                lines.push(format!("Licence: {}", info.license));
            }
            lines
        }
        kinds::TEXT => {
            let text: TextSpec = if value.is_null() { TextSpec::default() } else { parse(value) };
            let tools = match text.tools.len() {
                0 => "No tools".to_owned(),
                1 => "1 tool".to_owned(),
                n => format!("{n} tools"),
            };
            vec![tools, format!("temperature {} · up to {} tokens", text.temperature, text.max_tokens)]
        }
        kinds::THINKING => {
            let thinking: ThinkingSpec = if value.is_null() { ThinkingSpec::default() } else { parse(value) };
            vec![
                if thinking.on_by_default { "Thinks unless asked not to" } else { "Thinks only when asked" }.into(),
                "A toggle in LM Studio".into(),
            ]
        }
        kinds::EFFORT => {
            let effort: EffortSpec = if value.is_null() { EffortSpec::default() } else { parse(value) };
            vec![effort.levels.join(" · "), format!("Default: {}", effort.default)]
        }
        kinds::INPUT => {
            let capture: CaptureSpec = parse(value);
            vec![match capture.area {
                crate::magic::CaptureArea::Center { height_share } => {
                    format!("A {:.0}-px square in the middle", height_share * 1080.0)
                }
                crate::magic::CaptureArea::Region { rect } => {
                    format!("{}×{} at {}, {}", rect.w, rect.h, rect.x, rect.y)
                }
            }]
        }
        kinds::ANSWERS => {
            let classes: Vec<ClassSpec> = parse(value);
            let mut lines: Vec<String> = classes
                .iter()
                .take(4)
                .map(|c| {
                    if c.act { format!("{} – act at {:.0}%", c.label, c.threshold * 100.0) } else { c.label.clone() }
                })
                .collect();
            if classes.len() > 4 {
                lines.push(format!("and {} more", classes.len() - 4));
            }
            if lines.is_empty() {
                lines.push("No answers yet".into());
            }
            lines
        }
        kinds::ACTION => {
            let action: ActionSpec = parse(value);
            vec![format!("Press {} for {:.0} ms", action.key.label(), action.hold_ms)]
        }
        kinds::SETTINGS => {
            let settings: Vec<ModelSetting> = parse(value);
            vec![match settings.len() {
                0 => "No sliders".into(),
                1 => "1 slider".into(),
                n => format!("{n} sliders"),
            }]
        }
        kinds::PREP => {
            let input: InputSpec = parse(value);
            vec![format!("{}×{} pictures", input.width, input.height)]
        }
        _ => Vec::new(),
    }
}

// ---- following the wires --------------------------------------------------------

fn active(graph: &Graph, node: &Node) -> bool {
    node.enabled && graph.is_active(node.id)
}

/// The setup blocks chained into a block's Setup input, furthest first.
pub fn setup_chain(graph: &Graph, start: NodeId) -> Vec<NodeId> {
    let mut chain = Vec::new();
    let mut seen = BTreeSet::from([start]);
    let mut current = start;
    while let Some(source) = graph.data_source(current, SETUP) {
        if !seen.insert(source.node) {
            break;
        }
        if graph.node(source.node).is_some_and(|n| active(graph, n) && kinds::SETUP_BLOCKS.contains(&n.kind.as_str())) {
            chain.push(source.node);
        }
        current = source.node;
    }
    chain.reverse();
    chain
}

/// The start block of the canvas, when there is exactly one.
pub fn start_block(graph: &Graph) -> Result<&Node, String> {
    let starts: Vec<&Node> =
        graph.nodes().filter(|n| active(graph, n) && kinds::STARTS.contains(&n.kind.as_str())).collect();
    match starts.as_slice() {
        [] => Err("Add a block to start from: Start from Zero, Load Base Model or Use As It Is.".into()),
        [one] => Ok(*one),
        _ => Err("There is more than one start block; keep one (or turn the others off).".into()),
    }
}

/// One training stage of the pipeline.
#[derive(Clone, Debug, PartialEq)]
pub struct Stage {
    pub node: NodeId,
    pub kind: String,
    pub params: Map<String, Value>,
    /// The dataset outputs feeding it, as `(block, port)`.
    pub data: Vec<(NodeId, String)>,
    pub negatives: Vec<(NodeId, String)>,
    pub optimizer: OptimizerParams,
    /// Whether an Optimizer block is wired in (or the defaults are used).
    pub has_optimizer: bool,
}

/// The model wire, followed from its start to its end.
#[derive(Clone, Debug, PartialEq)]
pub struct Pipeline {
    pub entry: Node,
    pub architecture: Architecture,
    pub personas: Option<PersonaParams>,
    pub stages: Vec<Stage>,
    pub compile: Option<(NodeId, CompileParams)>,
    /// Every block on the model wire, in order (start first).
    pub path: Vec<NodeId>,
}

impl Pipeline {
    pub fn kind(&self, model: &MagicModel) -> ModelKind {
        match self.entry.kind.as_str() {
            kinds::ZERO if self.entry.params_or_default::<ZeroParams>().kind == "vision" => ModelKind::Vision,
            kinds::ZERO => ModelKind::Text,
            _ if self.architecture.conv.is_some() => ModelKind::Vision,
            _ if self.stages.iter().any(|s| s.kind == kinds::VISION_TRAIN) => ModelKind::Vision,
            _ if self.stages.iter().any(|s| s.kind != kinds::VISION_TRAIN) => ModelKind::Text,
            _ => model.kind,
        }
    }

    pub fn trains(&self) -> bool {
        !self.stages.is_empty()
    }
}

/// The blocks after a block on the model wire.
fn next_on_wire(graph: &Graph, node: NodeId) -> Vec<&Node> {
    graph
        .wires()
        .iter()
        .filter(|w| w.from.node == node && w.from.port == MODEL && w.to.port == MODEL)
        .filter_map(|w| graph.node(w.to.node))
        .filter(|n| active(graph, n))
        .collect()
}

/// The feed-forward layer of a layer block: the block wired in, or a plain one.
fn ffn_spec(graph: &Graph, node: &Node, mult: f64) -> (Value, Option<ArchBlock>, Option<PersonaParams>) {
    let provider = graph.data_source(node.id, FFN).and_then(|p| graph.node(p.node)).filter(|n| active(graph, n));
    match provider {
        Some(p) if p.kind == kinds::MOE => {
            let m: MoeParams = p.params_or_default();
            let spec = json!({ "kind": "moe", "experts": m.experts, "shared": m.shared, "top_k": m.top_k,
                               "mult": m.mult, "gating": m.gating, "balance": m.balance, "hash": m.hash });
            (spec, Some(ArchBlock { kind: p.kind.clone(), params: p.params.clone(), ffn: None }), None)
        }
        Some(p) if p.kind == kinds::PERSONA => {
            let persona: PersonaParams = p.params_or_default();
            let spec = json!({ "kind": "persona", "mult": persona.mult });
            (spec, Some(ArchBlock { kind: p.kind.clone(), params: p.params.clone(), ffn: None }), Some(persona))
        }
        _ => (json!({ "kind": "dense", "mult": mult }), None, None),
    }
}

/// Follows the model wire from the start through the architecture blocks,
/// the training stages and the package.
pub fn pipeline(graph: &Graph) -> Result<Pipeline, String> {
    use kinds::*;
    let entry = start_block(graph)?;
    let mut arch = Architecture::default();
    let mut size_seen = false;
    let mut pipeline = Pipeline {
        entry: entry.clone(),
        architecture: Architecture::default(),
        personas: None,
        stages: Vec::new(),
        compile: None,
        path: vec![entry.id],
    };
    let mut current = entry.id;
    let mut seen = BTreeSet::from([current]);
    let mut stage_seen = false;
    loop {
        let next = next_on_wire(graph, current);
        let onward: Vec<&Node> = next
            .iter()
            .copied()
            .filter(|n| {
                ARCHITECTURE.contains(&n.kind.as_str()) || STAGES.contains(&n.kind.as_str()) || n.kind == COMPILE
            })
            .collect();
        let node = match onward.as_slice() {
            [] => break,
            [one] => *one,
            _ => return Err("The Model wire splits into two paths; a model is built along one line.".into()),
        };
        if !seen.insert(node.id) {
            break;
        }
        current = node.id;
        pipeline.path.push(node.id);
        let kind = node.kind.as_str();
        if ARCHITECTURE.contains(&kind) && stage_seen {
            return Err("Put the architecture blocks before the training stages.".into());
        }
        if ARCHITECTURE.contains(&kind) && pipeline.compile.is_some()
            || STAGES.contains(&kind) && pipeline.compile.is_some()
        {
            return Err("Compile .magic Package comes last on the Model wire.".into());
        }
        let record = |ffn: Option<ArchBlock>| ArchBlock {
            kind: node.kind.clone(),
            params: node.params.clone(),
            ffn: ffn.map(Box::new),
        };
        match kind {
            SIZE => {
                let p: SizeParams = node.params_or_default();
                (arch.width, arch.heads, arch.kv_heads, arch.context) = (p.width, p.heads, p.kv_heads, p.context);
                arch.positions = p.positions.clone();
                arch.rope_theta = p.rope_theta;
                (arch.tied, arch.embed_scale) = (p.tied, p.embed_scale);
                size_seen = true;
                arch.blocks.push(record(None));
            }
            HYPER => {
                let p: HyperParams = node.params_or_default();
                arch.hyper_connections = Some(json!({ "streams": p.streams, "sinkhorn": p.sinkhorn }));
                arch.blocks.push(record(None));
            }
            MTP => {
                arch.mtp = Some(json!({ "weight": node.params_or_default::<MtpParams>().weight }));
                arch.blocks.push(record(None));
            }
            SPLIT => {
                arch.prefix_split = true;
                arch.blocks.push(record(None));
            }
            CONV => {
                let p: ConvParams = node.params_or_default();
                arch.conv = Some(json!({ "layers": p.layers, "channels": p.channels }));
                arch.blocks.push(record(None));
            }
            LAYERS | WINDOW | MLA | CSA | HCA | SSM => {
                let layers = node.param::<u32>("layers").unwrap_or(1).max(1);
                let mult = node.param::<f64>("mult").unwrap_or(2.75);
                let (ffn, provider, persona) = ffn_spec(graph, node, mult);
                if persona.is_some() {
                    pipeline.personas = persona;
                }
                let options: LayersParams = node.params_or_default();
                let attention = match kind {
                    LAYERS => json!({ "kind": "full" }),
                    WINDOW => json!({ "kind": "window", "window": node.param::<u32>("window").unwrap_or(64) }),
                    MLA => {
                        let p: MlaParams = node.params_or_default();
                        json!({ "kind": "mla", "kv_rank": p.kv_rank, "q_rank": p.q_rank, "rope_dim": p.rope_dim })
                    }
                    CSA => {
                        let p: CsaParams = node.params_or_default();
                        json!({ "kind": "csa", "compress": p.compress, "top_k": p.top_k, "window": p.window,
                                "indexer_heads": p.indexer_heads, "indexer_dim": p.indexer_dim, "rope_dim": p.rope_dim })
                    }
                    HCA => {
                        let p: HcaParams = node.params_or_default();
                        json!({ "kind": "hca", "compress": p.compress, "window": p.window, "rope_dim": p.rope_dim })
                    }
                    _ => json!({ "kind": "ssm", "state": node.param::<u32>("state").unwrap_or(16) }),
                };
                for _ in 0..layers {
                    if kind == LAYERS {
                        let mut ffn = ffn.clone();
                        if options.activation == "geglu" && ffn["kind"] == "dense" {
                            ffn["activation"] = json!("geglu");
                        }
                        let mut layer = json!({ "attention": options.attention(arch.layers.len()), "ffn": ffn });
                        if options.norm == "sandwich" {
                            layer["norm"] = json!("sandwich");
                        }
                        arch.layers.push(layer);
                    } else {
                        arch.layers.push(json!({ "attention": attention, "ffn": ffn }));
                    }
                }
                arch.blocks.push(record(provider));
            }
            COMPILE => pipeline.compile = Some((node.id, node.params_or_default())),
            _ => {
                stage_seen = true;
                let source = |port: &str| {
                    graph.data_source(node.id, port).map(|p| vec![(p.node, p.port.clone())]).unwrap_or_default()
                };
                let optimizer =
                    graph.data_source(node.id, OPTIMIZER).and_then(|p| graph.node(p.node)).filter(|n| active(graph, n));
                pipeline.stages.push(Stage {
                    node: node.id,
                    kind: node.kind.clone(),
                    params: node.params.clone(),
                    data: source(if kind == VISION_TRAIN { "pictures" } else { "data" }),
                    negatives: source("negative"),
                    optimizer: optimizer.map(|n| n.params_or_default()).unwrap_or_default(),
                    has_optimizer: optimizer.is_some(),
                });
            }
        }
    }
    if !size_seen && pipeline.trains() && entry.kind == ZERO {
        // Without a Model Size block, the defaults.
        let p = SizeParams::default();
        (arch.width, arch.heads, arch.kv_heads, arch.context) = (p.width, p.heads, p.kv_heads, p.context);
    }
    if arch.heads == 0 || arch.width % arch.heads != 0 {
        return Err(format!(
            "Model Size: the width ({}) must divide by the attention heads ({}).",
            arch.width, arch.heads
        ));
    }
    if arch.kv_heads == 0 || arch.heads % arch.kv_heads != 0 {
        return Err(format!(
            "Model Size: the heads ({}) must divide by the key/value heads ({}).",
            arch.heads, arch.kv_heads
        ));
    }
    if entry.kind == CONFIG && pipeline.trains() {
        return Err("Use As It Is does not train. To train, start from zero or from a base model's weights.".into());
    }
    pipeline.architecture = arch;
    Ok(pipeline)
}

// ---- the model from the canvas -------------------------------------------------

/// Builds the model's settings from the canvas – the setup blocks chained into
/// the start, the architecture's context and personas – and writes notes on
/// blocks that are not connected where they would be used.
pub fn derive(graph: &mut Graph, model: &mut MagicModel, has_onnx: bool) {
    let start = start_block(graph).map(|n| (n.id, n.kind.clone(), n.params.clone())).ok();
    let pipeline = pipeline(graph).ok();
    // The kind: a model not trained yet is what its Start from Zero says.
    if let Some((_, kind, params)) = &start
        && kind == kinds::ZERO
        && !has_onnx
    {
        let zero: ZeroParams = serde_json::from_value(Value::Object(params.clone())).unwrap_or_default();
        model.kind = if zero.kind == "vision" { ModelKind::Vision } else { ModelKind::Text };
    }
    // The context and personas come from the architecture.
    if model.kind == ModelKind::Text {
        let mut text = model.text.clone().unwrap_or_default();
        if let Some(p) = &pipeline
            && p.trains()
        {
            text.context = p.architecture.context;
        }
        text.personas = pipeline.as_ref().and_then(|p| p.personas.clone()).map_or_else(Vec::new, |p| {
            vec![
                Persona { name: p.first.clone(), tag: format!("<{}>", tag_of(&p.first)), role: p.first_role },
                Persona { name: p.second.clone(), tag: format!("<{}>", tag_of(&p.second)), role: p.second_role },
            ]
        });
        text.ensure_specials();
        model.text = Some(text);
    }
    // The setup chained into the start; what is not chained in is the default.
    let chain = start.as_ref().map(|(id, ..)| setup_chain(graph, *id)).unwrap_or_default();
    for kind in kinds::SETUP_BLOCKS {
        let used = chain.iter().rev().find_map(|id| graph.node(*id).filter(|n| n.kind == kind));
        if kinds::SWITCHES.contains(&kind) && used.is_none() {
            if let Some(text) = &mut model.text {
                match kind {
                    kinds::THINKING => text.thinking = None,
                    _ => text.effort = None,
                }
            }
            continue;
        }
        let value = used.map(|n| n.params_or_default::<SetupParams>().value).unwrap_or(Value::Null);
        apply_section(kind, &value, model);
    }
    // A model that is not text has no text settings.
    if model.kind == ModelKind::Vision {
        model.text = None;
    }
    write_notes(graph, model, start.as_ref().map(|s| s.0), pipeline.as_ref(), &chain);
}

fn write_notes(
    graph: &mut Graph,
    model: &MagicModel,
    start: Option<NodeId>,
    pipeline: Option<&Pipeline>,
    chain: &[NodeId],
) {
    use kinds::*;
    let text = model.kind == ModelKind::Text;
    let on_path: BTreeSet<NodeId> = pipeline.map(|p| p.path.iter().copied().collect()).unwrap_or_default();
    let mut notes: Vec<(NodeId, String)> = Vec::new();
    let has_output =
        |graph: &Graph, id: NodeId, port: &str| graph.wires().iter().any(|w| w.from.node == id && w.from.port == port);
    for node in graph.nodes() {
        let kind = node.kind.as_str();
        let note = if SETUP_BLOCKS.contains(&kind) {
            let vision_only = [INPUT, ANSWERS, ACTION, SETTINGS, PREP].contains(&kind);
            if !chain.contains(&node.id) {
                "Not used: chain it into the start's Setup input".into()
            } else if text && vision_only {
                "Only vision models use this".into()
            } else if !text && kind == TEXT {
                "Only text models use this".into()
            } else {
                String::new()
            }
        } else if STARTS.contains(&kind) {
            if Some(node.id) != start && start.is_some() {
                String::new()
            } else if chain.is_empty() && graph.data_source(node.id, SETUP).is_none() {
                "Plug Model Info into Setup to name it".into()
            } else {
                String::new()
            }
        } else if kinds::LAYER_BLOCKS.contains(&kind) || STAGES.contains(&kind) || kind == COMPILE {
            if !on_path.contains(&node.id) {
                "Not on the Model wire".into()
            } else if STAGES.contains(&kind) {
                let data_port = if kind == VISION_TRAIN { "pictures" } else { "data" };
                if graph.data_source(node.id, data_port).is_none() {
                    if kind == VISION_TRAIN {
                        "Wire a Vision Dataset into Pictures".into()
                    } else {
                        "Wire a Text Dataset into it".into()
                    }
                } else if graph.data_source(node.id, OPTIMIZER).is_none() {
                    "Default optimizer (AdamW)".into()
                } else {
                    String::new()
                }
            } else if kind == COMPILE && node.param::<String>("folder").is_none_or(|f| f.trim().is_empty()) {
                "Choose a folder to save into".into()
            } else {
                String::new()
            }
        } else if [MOE, PERSONA].contains(&kind) {
            if has_output(graph, node.id, FFN) {
                String::new()
            } else {
                "Wire it into a layer block's Feed-forward".into()
            }
        } else if kind == OPTIMIZER {
            if has_output(graph, node.id, OPTIMIZER) { String::new() } else { "Wire it into a training stage".into() }
        } else if kind == TEXT_DATA {
            let p: TextDataParams = node.params_or_default();
            if p.positive.is_empty() && p.negative.is_empty() {
                "No files yet: select it to add some".into()
            } else if !has_output(graph, node.id, "positive") && !has_output(graph, node.id, "negative") {
                "Wire it into a training stage".into()
            } else {
                String::new()
            }
        } else if kind == IMAGE_DATA {
            if model.training.is_empty() && model.finetune.is_empty() {
                "No pictures yet: select it to add some".into()
            } else if !has_output(graph, node.id, "pictures") {
                "Wire it into Vision Training".into()
            } else {
                String::new()
            }
        } else if [PROMPT, TEST].contains(&kind) {
            if (kind == PROMPT) != text {
                if text { "Use Prompt Model for text models".into() } else { "Use Test Model for vision models".into() }
            } else {
                String::new()
            }
        } else {
            continue;
        };
        notes.push((node.id, note));
    }
    for (id, note) in notes {
        if let Some(node) = graph.node_mut(id)
            && node.param::<String>("note").unwrap_or_default() != note
        {
            node.set_param("note", note);
        }
    }
}

// ---- the canvas a model opens with --------------------------------------------

struct Builder {
    registry: Registry,
    nodes: Vec<Node>,
    wires: Vec<Wire>,
}

impl Builder {
    fn new() -> Self {
        Self { registry: registry(), nodes: Vec::new(), wires: Vec::new() }
    }

    fn add(&mut self, kind: &str, pos: [f32; 2], params: Value) -> NodeId {
        let id = self.nodes.len() as NodeId + 1;
        let mut node =
            Node::new(id, kind, pos, self.registry.get(kind).map(|d| d.default_params()).unwrap_or_default());
        if let Value::Object(map) = params {
            node.params.extend(map);
        }
        self.nodes.push(node);
        id
    }

    fn wire(&mut self, from: NodeId, out: &str, to: NodeId, input: &str) {
        self.wires.push(Wire { from: PortRef::new(from, out), to: PortRef::new(to, input) });
    }
}

const STEP: f32 = 250.0;

/// The canvas a model opens with when it has none: its setup chained into a
/// start, and a pipeline that fits it – training from zero for a new model,
/// training further for one with weights, using it as it is for one made
/// elsewhere.
pub fn default_canvas(model: &MagicModel, has_weights: bool, has_onnx: bool, datasets: &[(String, bool)]) -> Fragment {
    use kinds::*;
    let mut b = Builder::new();
    let text = model.kind == ModelKind::Text;
    let row = 0.0;

    // The setup, chained into the start (below the pipeline).
    let mut setup_kinds: Vec<&str> =
        if text { vec![INFO, TEXT] } else { vec![INFO, ANSWERS, INPUT, ACTION, SETTINGS, PREP] };
    if let Some(spec) = model.text.as_ref().filter(|_| text) {
        if spec.thinking.is_some() {
            setup_kinds.push(THINKING);
        }
        if spec.effort.is_some() {
            setup_kinds.push(EFFORT);
        }
    }
    let mut setup = BTreeSet::new();
    let mut previous: Option<NodeId> = None;
    for (index, kind) in setup_kinds.iter().enumerate() {
        let (column, line) = (index % 3, index / 3);
        let id = b.add(
            kind,
            [column as f32 * STEP - 3.0 * STEP, 260.0 + line as f32 * 140.0],
            json!({ "value": section_of(kind, model) }),
        );
        if let Some(prev) = previous {
            b.wire(prev, SETUP, id, SETUP);
        }
        previous = Some(id);
        setup.insert(id);
    }

    let entry = if has_weights {
        b.add(BASE, [0.0, row], json!({}))
    } else if has_onnx {
        b.add(CONFIG, [0.0, row], json!({}))
    } else {
        b.add(ZERO, [0.0, row], json!({ "kind": if text { "text" } else { "vision" } }))
    };
    if let Some(last) = previous {
        b.wire(last, SETUP, entry, SETUP);
    }
    let mut last = entry;
    let mut x = STEP;
    let place = |b: &mut Builder, kind: &str, params: Value, last: &mut NodeId, x: &mut f32| -> NodeId {
        let id = b.add(kind, [*x, row], params);
        b.wire(*last, MODEL, id, MODEL);
        *last = id;
        *x += STEP;
        id
    };

    if has_weights || !has_onnx {
        // The architecture: the model's own, or a small one that trains in a few minutes.
        let blocks: Vec<ArchBlock> = match &model.architecture {
            Some(a) if !a.blocks.is_empty() => a.blocks.clone(),
            _ if text => magi_blocks(),
            _ => vec![
                ArchBlock {
                    kind: SIZE.into(),
                    params: obj(
                        json!({ "width": 64, "heads": 4, "kv_heads": 4, "context": model.input.width.clamp(16, 512) }),
                    ),
                    ffn: None,
                },
                ArchBlock { kind: CONV.into(), params: Map::new(), ffn: None },
            ],
        };
        let mut providers: Vec<(NodeId, ArchBlock)> = Vec::new();
        for block in &blocks {
            let id = place(&mut b, &block.kind, Value::Object(block.params.clone()), &mut last, &mut x);
            if let Some(ffn) = &block.ffn {
                let existing =
                    providers.iter().find(|(_, p)| p.kind == ffn.kind && p.params == ffn.params).map(|(id, _)| *id);
                let provider = existing.unwrap_or_else(|| {
                    let provider_id = b.add(&ffn.kind, [x - STEP, row - 190.0], Value::Object(ffn.params.clone()));
                    providers.push((provider_id, (**ffn).clone()));
                    provider_id
                });
                b.wire(provider, FFN, id, FFN);
            }
        }
        let optimizer = b.add(OPTIMIZER, [x + STEP * 0.5, row - 210.0], json!({}));
        if text {
            let positive: Vec<String> = datasets.iter().filter(|(_, n)| !n).map(|(f, _)| f.clone()).collect();
            let negative: Vec<String> = datasets.iter().filter(|(_, n)| *n).map(|(f, _)| f.clone()).collect();
            let data = b.add(TEXT_DATA, [x, row + 230.0], json!({ "positive": positive, "negative": negative }));
            let stages: &[&str] = if has_weights { &[FINETUNE] } else { &[PRETRAIN, FINETUNE] };
            for kind in stages {
                // The example's many small datasets need longer fine-tuning than the default.
                let params = if *kind == FINETUNE && !has_weights { json!({ "steps": 1200 }) } else { json!({}) };
                let id = place(&mut b, kind, params, &mut last, &mut x);
                b.wire(data, "positive", id, "data");
                if *kind == FINETUNE {
                    b.wire(data, "negative", id, "negative");
                }
                b.wire(optimizer, OPTIMIZER, id, OPTIMIZER);
            }
        } else {
            let data = b.add(IMAGE_DATA, [x, row + 230.0], json!({}));
            let id = place(&mut b, VISION_TRAIN, json!({}), &mut last, &mut x);
            b.wire(data, "pictures", id, "pictures");
            b.wire(optimizer, OPTIMIZER, id, OPTIMIZER);
        }
    }
    let compile = place(&mut b, COMPILE, json!({ "gguf": text }), &mut last, &mut x);
    let try_kind = if text { PROMPT } else { TEST };
    let try_it = b.add(try_kind, [x, row], json!({}));
    b.wire(compile, MODEL, try_it, MODEL);

    let group =
        Group { id: 1, name: "The model's setup".into(), color: None, nodes: setup, parent: None, enabled: true };
    Fragment { nodes: b.nodes, wires: b.wires, groups: vec![group] }
}

fn obj(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

// ---- common architectures -------------------------------------------------------

/// An architecture to start from: blocks that replace the canvas's own, and
/// setup blocks it adds to the setup chain (Thinking, Reasoning Effort).
pub struct Common {
    pub name: &'static str,
    pub description: &'static str,
    pub blocks: Vec<ArchBlock>,
    pub setup: Vec<&'static str>,
}

/// The Magi layout: MacroMagi's own recipe for small agents. Deep and narrow
/// (more layers than its width would suggest), grouped key/value heads,
/// queries and keys normalized for steady training, a tied output head to
/// save weights, a long rotary base, and a 2048-token context window: tokens
/// are bytes, and four tools in the system text take about 900. It exports
/// as Qwen3 in GGUF.
pub fn magi_blocks() -> Vec<ArchBlock> {
    use kinds::*;
    vec![
        block(
            SIZE,
            json!({ "width": 192, "heads": 6, "kv_heads": 2, "context": 2048, "rope_theta": 100000.0, "tied": true }),
        ),
        block(LAYERS, json!({ "layers": 8, "mult": 3.0, "more": true, "qk_norm": true })),
    ]
}

fn block(kind: &str, params: Value) -> ArchBlock {
    ArchBlock { kind: kind.into(), params: obj(params), ffn: None }
}

fn with_ffn(mut b: ArchBlock, ffn: ArchBlock) -> ArchBlock {
    b.ffn = Some(Box::new(ffn));
    b
}

/// Well-known architectures, scaled down to train on a desktop computer.
pub fn commons() -> Vec<Common> {
    use kinds::*;
    let v4_moe = block(
        MOE,
        json!({ "experts": 16, "shared": 1, "top_k": 2, "mult": 0.5, "gating": "sqrt_softplus", "balance": "bias" }),
    );
    let v4_hash = block(
        MOE,
        json!({ "experts": 16, "shared": 1, "top_k": 2, "mult": 0.5, "gating": "sqrt_softplus", "balance": "bias", "hash": true }),
    );
    let v3_moe = block(
        MOE,
        json!({ "experts": 16, "shared": 1, "top_k": 2, "mult": 0.5, "gating": "sigmoid", "balance": "bias" }),
    );
    vec![
        Common {
            name: "Magi (small)",
            description: "MacroMagi's own recipe for small agents: deep and narrow, grouped key/value heads, queries and \
                          keys normalized, a tied output head, and Thinking and Reasoning Effort set up. Exports to GGUF \
                          (as Qwen3) with LM Studio's Enable Thinking and Reasoning Effort controls. To make your own, \
                          change the blocks, select them and use Presets › Save selected blocks.",
            blocks: magi_blocks(),
            setup: vec![THINKING, EFFORT],
        },
        Common {
            name: "Qwen3 (small)",
            description: "Qwen3's layout at desktop size: grouped key/value heads, queries and keys normalized \
                          (QK-norm), SwiGLU, a tied output head like Qwen3's small models. Exports to GGUF as qwen3 \
                          and to Hugging Face.",
            blocks: vec![
                block(
                    SIZE,
                    json!({ "width": 256, "heads": 8, "kv_heads": 4, "context": 1024, "rope_theta": 1000000.0, "tied": true }),
                ),
                block(LAYERS, json!({ "layers": 6, "mult": 3.0, "more": true, "qk_norm": true })),
            ],
            setup: Vec::new(),
        },
        Common {
            name: "Qwen3 Thinking (small)",
            description: "Qwen3's layout with reasoning set up: Thinking (on unless switched off) and Reasoning Effort \
                          (low, medium, high). Train it on datasets with \"thinking\" and \"effort\" – the example \
                          reasoning.jsonl has both. In LM Studio it shows the Enable Thinking toggle and a Reasoning \
                          Effort choice.",
            blocks: vec![
                block(
                    SIZE,
                    json!({ "width": 256, "heads": 8, "kv_heads": 4, "context": 1024, "rope_theta": 1000000.0, "tied": true }),
                ),
                block(LAYERS, json!({ "layers": 6, "mult": 3.0, "more": true, "qk_norm": true })),
            ],
            setup: vec![THINKING, EFFORT],
        },
        Common {
            name: "Gemma 3 (small)",
            description: "Gemma 3's layout at desktop size: GeGLU, norms before and after each part, queries and keys \
                          normalized, scaled and tied embeddings, and five local layers (a 256-token window) for each \
                          global one. Exports to GGUF as gemma3.",
            blocks: vec![
                block(
                    SIZE,
                    json!({ "width": 256, "heads": 4, "kv_heads": 1, "context": 1024, "rope_theta": 10000.0,
                                    "tied": true, "embed_scale": true }),
                ),
                block(
                    LAYERS,
                    json!({ "layers": 6, "mult": 4.0, "more": true, "qk_norm": true, "activation": "geglu",
                                      "norm": "sandwich", "window": 256, "global_every": 6 }),
                ),
            ],
            setup: Vec::new(),
        },
        Common {
            name: "Mistral (small)",
            description: "Mistral's layout at desktop size: the Llama layout with four query heads per key/value head and \
                          a large rotary base for long texts. Exports to GGUF (as llama) and to Hugging Face.",
            blocks: vec![
                block(
                    SIZE,
                    json!({ "width": 256, "heads": 8, "kv_heads": 2, "context": 1024, "rope_theta": 1000000.0 }),
                ),
                block(LAYERS, json!({ "layers": 6, "mult": 3.5 })),
            ],
            setup: Vec::new(),
        },
        Common {
            name: "GLM-4 (small)",
            description: "GLM-4's layout at desktop size: biases on queries, keys and values, rotary positions on half \
                          of each head, and norms before and after each part. Runs in MacroMagi; not exported to GGUF.",
            blocks: vec![
                block(SIZE, json!({ "width": 256, "heads": 8, "kv_heads": 2, "context": 1024 })),
                block(
                    LAYERS,
                    json!({ "layers": 6, "mult": 3.0, "more": true, "norm": "sandwich", "rotary": 0.5, "bias": true }),
                ),
            ],
            setup: Vec::new(),
        },
        Common {
            name: "DeepSeek-V4 (small)",
            description: "DeepSeek-V4's design at desktop size: hyper-connections, sliding-window attention first, then \
                          compressed sparse and heavily compressed attention in turn, DeepSeekMoE with √Softplus gating \
                          (hash-routed in the first layer) and multi-token prediction. Train it with Muon. Runs in \
                          MacroMagi; not a GGUF architecture.",
            blocks: vec![
                block(SIZE, json!({ "width": 192, "heads": 4, "kv_heads": 4, "context": 512 })),
                block(HYPER, json!({ "streams": 4, "sinkhorn": 20 })),
                block(MTP, json!({ "weight": 0.3 })),
                with_ffn(block(WINDOW, json!({ "layers": 1, "window": 64 })), v4_hash),
                with_ffn(block(CSA, json!({ "layers": 1, "compress": 4, "top_k": 16, "window": 64 })), v4_moe.clone()),
                with_ffn(block(HCA, json!({ "layers": 1, "compress": 32, "window": 64 })), v4_moe.clone()),
                with_ffn(block(CSA, json!({ "layers": 1, "compress": 4, "top_k": 16, "window": 64 })), v4_moe.clone()),
                with_ffn(block(HCA, json!({ "layers": 1, "compress": 32, "window": 64 })), v4_moe),
            ],
            setup: Vec::new(),
        },
        Common {
            name: "DeepSeek-V3 (small)",
            description: "DeepSeek-V3's design at desktop size: multi-head latent attention, a dense first layer, then \
                          DeepSeekMoE layers with sigmoid gating and bias balancing, and multi-token prediction. Runs \
                          in MacroMagi; not a GGUF architecture.",
            blocks: vec![
                block(SIZE, json!({ "width": 192, "heads": 4, "kv_heads": 4, "context": 512 })),
                block(MTP, json!({ "weight": 0.3 })),
                block(MLA, json!({ "layers": 1, "kv_rank": 64, "q_rank": 96, "rope_dim": 16 })),
                with_ffn(block(MLA, json!({ "layers": 3, "kv_rank": 64, "q_rank": 96, "rope_dim": 16 })), v3_moe),
            ],
            setup: Vec::new(),
        },
        Common {
            name: "Llama (small)",
            description: "The Llama layout: rotary positions, grouped key/value heads and SwiGLU feed-forward layers. \
                          Exports to GGUF for LM Studio and llama.cpp, and to Hugging Face.",
            blocks: vec![
                block(SIZE, json!({ "width": 256, "heads": 8, "kv_heads": 4, "context": 512 })),
                block(LAYERS, json!({ "layers": 6, "mult": 2.75 })),
            ],
            setup: Vec::new(),
        },
    ]
}

/// Puts a common architecture on the canvas in place of its architecture
/// blocks (or, without a pipeline, at `at`). Returns the new blocks.
pub fn apply_common(graph: &mut Graph, common: &Common, at: [f32; 2]) -> Vec<NodeId> {
    use kinds::*;
    let reg = registry();
    let defaults = |kind: &str| reg.get(kind).map(|d| d.default_params()).unwrap_or_default();
    // Where the architecture goes: after the start, before what followed it.
    let path = pipeline(graph).map(|p| p.path).unwrap_or_default();
    let start = path.first().copied();
    let old: Vec<NodeId> = path
        .iter()
        .copied()
        .filter(|id| graph.node(*id).is_some_and(|n| ARCHITECTURE.contains(&n.kind.as_str())))
        .collect();
    let after = path.iter().copied().find(|id| {
        graph.node(*id).is_some_and(|n| !ARCHITECTURE.contains(&n.kind.as_str()) && !STARTS.contains(&n.kind.as_str()))
    });
    let origin = start.and_then(|id| graph.node(id)).map(|n| [n.pos[0] + STEP, n.pos[1]]).unwrap_or(at);
    // Feed-forward blocks only the old architecture used go too.
    let providers: Vec<NodeId> = old
        .iter()
        .filter_map(|id| graph.data_source(*id, FFN).map(|p| p.node))
        .filter(|p| graph.wires().iter().filter(|w| w.from.node == *p).all(|w| old.contains(&w.to.node)))
        .collect();
    let old_width = old.len() as f32 * STEP;
    for id in old.iter().chain(&providers) {
        graph.remove(*id);
    }
    let new_width = common.blocks.len() as f32 * STEP;
    // Blocks to the right move over to make room.
    let shift = new_width - old_width;
    if start.is_some() {
        for node in graph.nodes_mut() {
            if node.pos[0] >= origin[0] - 1.0 {
                node.pos[0] += shift;
            }
        }
    }
    let mut created = Vec::new();
    let mut previous = start;
    let mut made: Vec<(ArchBlock, NodeId)> = Vec::new();
    for (index, b) in common.blocks.iter().enumerate() {
        let mut params = defaults(&b.kind);
        params.extend(b.params.clone());
        let id = graph.add_with(&b.kind, [origin[0] + index as f32 * STEP, origin[1]], params);
        created.push(id);
        if let Some(prev) = previous {
            let _ = graph.connect(&reg, PortRef::new(prev, MODEL), PortRef::new(id, MODEL));
        }
        previous = Some(id);
        if let Some(ffn) = &b.ffn {
            let provider = made.iter().find(|(p, _)| p.kind == ffn.kind && p.params == ffn.params).map(|(_, id)| *id);
            let provider = provider.unwrap_or_else(|| {
                let mut params = defaults(&ffn.kind);
                params.extend(ffn.params.clone());
                let pid = graph.add_with(&ffn.kind, [origin[0] + index as f32 * STEP, origin[1] - 190.0], params);
                made.push(((**ffn).clone(), pid));
                created.push(pid);
                pid
            });
            let _ = graph.connect(&reg, PortRef::new(provider, FFN), PortRef::new(id, FFN));
        }
    }
    if let (Some(last), Some(next)) = (previous, after) {
        let _ = graph.connect(&reg, PortRef::new(last, MODEL), PortRef::new(next, MODEL));
    }
    // Setup blocks it needs join the end of the setup chain, unless they are in it.
    if let Some(start) = start {
        for kind in &common.setup {
            let chain = setup_chain(graph, start);
            if chain.iter().any(|id| graph.node(*id).is_some_and(|n| n.kind == *kind)) {
                continue;
            }
            let last = graph.data_source(start, SETUP).map(|p| p.node);
            let below = last
                .and_then(|id| graph.node(id))
                .map_or([origin[0] - STEP, origin[1] + 260.0], |n| [n.pos[0] + STEP, n.pos[1]]);
            let id = graph.add_with(kind, below, defaults(kind));
            created.push(id);
            if let Some(last) = last {
                if let Some(wire) = graph.wires().iter().find(|w| w.from.node == last && w.to.node == start).cloned() {
                    graph.disconnect(&wire);
                }
                let _ = graph.connect(&reg, PortRef::new(last, SETUP), PortRef::new(id, SETUP));
            }
            let _ = graph.connect(&reg, PortRef::new(id, SETUP), PortRef::new(start, SETUP));
        }
    }
    created
}

// ---- the training plan ----------------------------------------------------------

/// Where the files a training run needs were written.
pub struct PlanFiles<'a> {
    /// A Text Dataset's file (its path inside the `.magic` file) on disk.
    pub dataset: &'a dyn Fn(&str) -> Option<String>,
    /// The weights to start from, on disk.
    pub weights: Option<String>,
    /// The prepared pictures (vision).
    pub pictures: Option<String>,
    pub out_dir: String,
    /// CUDA devices to train on; empty: the CPU.
    pub devices: Vec<u32>,
}

/// The training plan the trainer (`runtime/python/magi_train`) reads.
pub fn plan(pipeline: &Pipeline, graph: &Graph, model: &MagicModel, files: &PlanFiles<'_>) -> Result<Value, String> {
    let kind = pipeline.kind(model);
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
        entry.remove("note");
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
                    .map(|r| json!({ "kind": r.kind, "text": r.text, "score": r.score, "tokens": r.tokens, "ngram": r.tokens, "repeats": r.repeats }))
                    .collect();
                entry.insert("rules".into(), Value::Array(rules));
                "rl"
            }
            kinds::COMPACTION => "compaction",
            kinds::VISION_TRAIN => "vision",
            _ => continue,
        };
        let name = registry().get(&stage.kind).map(|d| d.name.clone()).unwrap_or_default();
        if kind == ModelKind::Text && ty != "vision" && data.is_empty() {
            return Err(format!("{name} has nothing to learn from: wire a Text Dataset with files into it."));
        }
        if kind == ModelKind::Vision && ty != "vision" {
            return Err(format!("{name} is for text models; a vision model trains with Vision Training."));
        }
        if kind == ModelKind::Text && ty == "vision" {
            return Err("Vision Training is for vision models.".into());
        }
        let mut optimizer = serde_json::to_value(&stage.optimizer).map_err(|e| e.to_string())?;
        if let Some(o) = optimizer.as_object_mut() {
            o.remove("note");
        }
        entry.insert("type".into(), Value::from(ty));
        entry.insert("data".into(), json!(data));
        entry.insert("negatives".into(), json!(negatives));
        entry.insert("optimizer".into(), optimizer);
        stages.push(Value::Object(entry));
    }
    let mut architecture = serde_json::to_value(&pipeline.architecture).map_err(|e| e.to_string())?;
    if let Some(a) = architecture.as_object_mut() {
        a.remove("blocks");
    }
    let mut plan = json!({
        "kind": if kind == ModelKind::Text { "text" } else { "vision" },
        "out_dir": files.out_dir,
        "seed": 0,
        "architecture": architecture,
        "start": { "mode": pipeline.entry.kind.trim_start_matches("magic."), "weights": files.weights },
        "datasets": datasets,
        "stages": stages,
        "devices": files.devices,
    });
    match kind {
        ModelKind::Text => {
            if pipeline.architecture.layers.is_empty() {
                return Err(
                    "The model has no layers: add Transformer Layers (or another layer block) after Model Size.".into(),
                );
            }
            let mut text = model.text.clone().unwrap_or_default();
            text.context = pipeline.architecture.context;
            text.ensure_specials();
            plan["text"] = serde_json::to_value(text).map_err(|e| e.to_string())?;
        }
        ModelKind::Vision => {
            let pictures = files
                .pictures
                .clone()
                .ok_or("The model has no pictures to learn from: add some in the Vision Dataset block.")?;
            if model.classes.len() < 2 {
                return Err("A vision model needs at least two answers (the Answers block).".into());
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
    if !jsonl {
        check_tools(sample, text, None)?;
        return Ok("Text");
    }
    let value: Value = serde_json::from_str(sample).map_err(|e| format!("Not valid JSON: {e}"))?;
    let Value::Object(item) = &value else {
        return Err("A sample is a JSON object, such as {\"prompt\": …, \"response\": …}.".into());
    };
    let text_of = |key: &str| item.get(key).map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_owned));
    // The tools a sample offers, by name; without “tools”, the model's.
    let offered: Option<Vec<String>> = match item.get("tools") {
        None => None,
        Some(Value::Array(tools)) => Some(
            tools
                .iter()
                .map(|tool| {
                    tool.get("function")
                        .unwrap_or(tool)
                        .get("name")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                        .ok_or_else(|| "Each tool in “tools” needs a “name”.".to_owned())
                })
                .collect::<Result<_, _>>()?,
        ),
        Some(_) => return Err("“tools” must be a list of tools.".into()),
    };
    if let Some(effort) = item.get("effort") {
        let effort = effort.as_str().ok_or("“effort” is a level's name, such as \"high\".")?;
        if let Some(levels) = &text.effort
            && !levels.levels.iter().any(|l| l == effort)
        {
            return Err(format!(
                "“effort” is “{effort}”, which is not one of the model's levels ({}).",
                levels.levels.join(", ")
            ));
        }
    }
    let check = |s: &str| check_tools(s, text, offered.as_deref());
    if let Some(messages) = item.get("messages") {
        let Value::Array(messages) = messages else { return Err("“messages” must be a list.".into()) };
        for message in messages {
            let role = message.get("role").and_then(Value::as_str).unwrap_or_default();
            if !["system", "user", "assistant", "tool"].contains(&role) {
                return Err(format!("A message's role must be system, user, assistant or tool, not “{role}”."));
            }
            match message.get("content") {
                Some(Value::String(content)) => check(content)?,
                None | Some(Value::Null) if role == "assistant" => {}
                Some(other) if role == "tool" => check(&other.to_string())?,
                _ => return Err(format!("A {role} message needs its “content” as text.")),
            }
            if let Some(thinking) = message.get("thinking")
                && !thinking.is_string()
            {
                return Err("“thinking” is text.".into());
            }
            if let Some(calls) = message.get("tool_calls") {
                let Value::Array(calls) = calls else { return Err("“tool_calls” must be a list.".into()) };
                for call in calls {
                    check_call(call.get("function").unwrap_or(call), text, offered.as_deref())?;
                }
            }
        }
        if !messages.iter().any(|m| m.get("role").and_then(Value::as_str) == Some("assistant")) {
            return Err("A conversation needs an answer: a message with the role “assistant”.".into());
        }
        let tools = messages.iter().any(|m| m.get("tool_calls").is_some() || m.get("role") == Some(&json!("tool")));
        return Ok(if tools { "Tool use" } else { "Conversation" });
    }
    if let (Some(context), Some(compact)) = (text_of("context"), text_of("compact")) {
        check(&context)?;
        check(&compact)?;
        return Ok("Compaction");
    }
    if let Some(prompt) = text_of("prompt") {
        check(&prompt)?;
        if text_of("rejected").or_else(|| text_of("bad")).is_some() {
            // An answer not to give may be broken on purpose: only the request is checked.
            return Ok("Not to answer");
        }
        let response =
            text_of("response").ok_or("A request needs its “response” (or “rejected” for an answer not to give).")?;
        check(&response)?;
        return Ok(if item.get("thinking").is_some_and(|t| t.as_str().is_some_and(|t| !t.is_empty())) {
            "Reasoning"
        } else {
            "Request and answer"
        });
    }
    if let Some(plain) = text_of("text") {
        check(&plain)?;
        return Ok("Text");
    }
    Err("Not a sample the trainer reads: give prompt and response, messages, context and compact, or text.".into())
}

/// A tool call as JSON: a name the sample (or the model) offers, and its arguments.
fn check_call(call: &Value, spec: &TextSpec, offered: Option<&[String]>) -> Result<(), String> {
    let name = call.get("name").and_then(Value::as_str).ok_or("A tool call needs a “name”.")?;
    let arguments = call.get("arguments").or_else(|| call.get("args"));
    if arguments.is_some_and(|a| !a.is_object() && !a.is_string()) {
        return Err("A tool call's “arguments” must be an object.".into());
    }
    match offered {
        Some(names) if !names.iter().any(|n| n == name) => {
            Err(format!("The tool “{name}” is not one of the sample's “tools”."))
        }
        None if !spec.tools.is_empty() && !spec.tools.iter().any(|t| t.name == name) => {
            Err(format!("The tool “{name}” is not one of the model's tools (Text & Tools)."))
        }
        _ => Ok(()),
    }
}

/// Every tool call in a text must be written the one way:
/// `<tool_call>{"name": …, "arguments": {…}}</tool_call>`.
fn check_tools(text: &str, spec: &TextSpec, offered: Option<&[String]>) -> Result<(), String> {
    let (open, close) = (spec.tool_tags.0.as_str(), spec.tool_tags.1.as_str());
    let mut rest = text;
    while let Some(start) = rest.find(open) {
        let after = &rest[start + open.len()..];
        let end = after.find(close).ok_or_else(|| format!("A tool call is not closed with {close}."))?;
        let call: Value = serde_json::from_str(after[..end].trim())
            .map_err(|_| format!("A tool call is not JSON: {}", after[..end].trim()))?;
        check_call(&call, spec, offered)?;
        rest = &after[end + close.len()..];
    }
    if rest.contains(close) {
        return Err(format!("{close} without {open}."));
    }
    Ok(())
}

// ---- the example workspace ------------------------------------------------------

/// The system prompt of the example text model.
pub const EXAMPLE_SYSTEM: &str = "You are Magi, a small assistant made in MacroMagi. Answer clearly and briefly, and use a tool when you need facts you do not have.";

/// The tools the example text model offers: general ones, as its datasets
/// use them (`examples/make_examples.py`). MacroMagi's own tools (press,
/// type…) are added in Text & Tools.
pub fn example_tools() -> Vec<ToolSpec> {
    let tool = |name: &str, description: &str, argument: &str| ToolSpec {
        name: name.into(),
        description: description.into(),
        parameters: json!({ "type": "object", "properties": { argument: { "type": "string" } }, "required": [argument] }),
    };
    vec![
        tool("web_search", "Searches the web and returns short results.", "query"),
        tool("calculator", "Works out a math expression.", "expression"),
        tool("get_weather", "Gets the current weather for a place.", "location"),
        tool("get_time", "Gets the current date and time for a place.", "location"),
    ]
}

/// The example datasets a new text model starts with, written by
/// `examples/make_examples.py`: plain English to pretrain on, everyday
/// conversation, reasoning at three effort levels, using tools, naming
/// conversations, and answers not to give. Paths are inside the `.magic`
/// file; `true` marks “what not to do”.
pub fn example_datasets() -> Vec<(String, String, bool)> {
    let file = |name: &str, content: &str, negative: bool| (format!("data/text/{name}"), content.to_owned(), negative);
    vec![
        file("language.md", include_str!("examples/language.md"), false),
        file("chat.jsonl", include_str!("examples/chat.jsonl"), false),
        file("reasoning.jsonl", include_str!("examples/reasoning.jsonl"), false),
        file("tools.jsonl", include_str!("examples/tools.jsonl"), false),
        file("titles.jsonl", include_str!("examples/titles.jsonl"), false),
        file("avoid.jsonl", include_str!("examples/avoid.jsonl"), true),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn every_kind() -> Vec<&'static str> {
        use kinds::*;
        let mut all: Vec<&str> =
            vec![ZERO, BASE, CONFIG, MOE, PERSONA, TEXT_DATA, IMAGE_DATA, OPTIMIZER, COMPILE, PROMPT, TEST];
        all.extend(SETUP_BLOCKS);
        all.extend(ARCHITECTURE);
        all.extend(STAGES);
        all
    }

    #[test]
    fn every_magic_block_has_complete_defaults() {
        let registry = registry();
        for id in every_kind() {
            assert!(registry.get(id).is_some(), "{id}");
        }
        assert_eq!(registry.len(), every_kind().len(), "every block is listed");
        assert!(Registry::builtin().get(kinds::ZERO).is_none(), "macros never see them");
    }

    fn text_model() -> MagicModel {
        MagicModel {
            kind: ModelKind::Text,
            text: Some(TextSpec {
                thinking: Some(ThinkingSpec::default()),
                effort: Some(EffortSpec::default()),
                tools: example_tools(),
                ..TextSpec::default()
            }),
            name: "Helper".into(),
            ..MagicModel::default()
        }
    }

    fn files() -> Vec<(String, bool)> {
        example_datasets().into_iter().map(|(f, _, n)| (f, n)).collect()
    }

    fn plan_files<'a>(dataset: &'a dyn Fn(&str) -> Option<String>) -> PlanFiles<'a> {
        PlanFiles { dataset, weights: None, pictures: None, out_dir: "/tmp/out".into(), devices: vec![] }
    }

    #[test]
    fn a_new_text_model_gets_a_working_pipeline() {
        let mut graph = Graph::from(default_canvas(&text_model(), false, false, &files()));
        let found = pipeline(&graph).unwrap();
        assert_eq!(found.entry.kind, kinds::ZERO);
        let stages: Vec<&str> = found.stages.iter().map(|s| s.kind.as_str()).collect();
        assert_eq!(stages, [kinds::PRETRAIN, kinds::FINETUNE]);
        assert!(found.stages.iter().all(|s| s.has_optimizer), "the optimizer is wired into both stages");
        assert_eq!(found.architecture.layers.len(), 8);
        assert_eq!(found.architecture.gguf_arch(), Some("qwen3"), "the Magi layout");
        assert!(found.architecture.tied);
        assert!(found.compile.is_some());

        // The setup chain makes the model what it says.
        let mut model = MagicModel::default();
        derive(&mut graph, &mut model, false);
        assert_eq!((model.kind, model.name.as_str()), (ModelKind::Text, "Helper"));
        let text = model.text.as_ref().unwrap();
        assert_eq!(text.context, 2048, "room for the tools: a token is a byte");
        assert!(text.thinking.is_some() && text.effort.is_some(), "Thinking and Reasoning Effort are chained in");
        // Unchaining Thinking switches it off.
        let thinking = graph.nodes().find(|n| n.kind == kinds::THINKING).unwrap().id;
        graph.node_mut(thinking).unwrap().enabled = false;
        derive(&mut graph, &mut model, false);
        assert!(model.text.as_ref().unwrap().thinking.is_none());
        graph.node_mut(thinking).unwrap().enabled = true;
        // Unchaining Model Info leaves the model unnamed, and the block says why.
        let info = graph.nodes().find(|n| n.kind == kinds::INFO).unwrap().id;
        let wire = graph.wires().iter().find(|w| w.from.node == info).cloned().unwrap();
        graph.disconnect(&wire);
        derive(&mut graph, &mut model, false);
        assert_eq!(model.name, "");
        assert!(graph.node(info).unwrap().param::<String>("note").unwrap().contains("Setup"));

        let found = pipeline(&graph).unwrap();
        let dataset = |file: &str| Some(format!("/tmp/{}", file.rsplit('/').next().unwrap()));
        let plan = plan(&found, &graph, &model, &plan_files(&dataset)).unwrap();
        assert_eq!(plan["kind"], "text");
        assert_eq!(plan["stages"][0]["type"], "pretrain");
        assert_eq!(plan["stages"][1]["type"], "sft");
        assert_eq!(plan["stages"][1]["optimizer"]["name"], "adamw");
        assert_eq!(plan["architecture"]["layers"].as_array().unwrap().len(), 8);
        assert_eq!(plan["architecture"]["layers"][0]["attention"], json!({ "kind": "full", "qk_norm": true }));
        assert_eq!(plan["text"]["effort"]["default"], "medium");
        assert!(plan["architecture"].get("blocks").is_none());
        let key = plan["stages"][1]["negatives"][0].as_str().unwrap().to_owned();
        assert_eq!(plan["datasets"][&key][0], "/tmp/avoid.jsonl");
    }

    #[test]
    fn the_pipeline_explains_what_is_wrong() {
        // A model made elsewhere is only used as it is, with its setup on the canvas.
        let vision = MagicModel { classes: vec![Default::default(); 2], ..MagicModel::default() };
        let mut graph = Graph::from(default_canvas(&vision, false, true, &[]));
        assert!(!pipeline(&graph).unwrap().trains());
        let mut model = vision.clone();
        derive(&mut graph, &mut model, true);
        assert_eq!(model.classes.len(), 2, "the Answers block keeps them");

        // Training stages after Use As It Is are refused, with a reason.
        let config = graph.nodes().find(|n| n.kind == kinds::CONFIG).unwrap().id;
        let compile = graph.nodes().find(|n| n.kind == kinds::COMPILE).unwrap().id;
        let wire = graph.wires().iter().find(|w| w.from.node == config && w.to.node == compile).cloned().unwrap();
        graph.disconnect(&wire);
        let stage = graph.add_with(kinds::VISION_TRAIN, [0.0, 0.0], Map::new());
        graph.connect(&registry(), PortRef::new(config, MODEL), PortRef::new(stage, MODEL)).unwrap();
        assert!(pipeline(&graph).unwrap_err().contains("does not train"));

        // Two starts.
        let mut two = Graph::from(default_canvas(&text_model(), false, false, &files()));
        two.add_with(kinds::ZERO, [0.0, 0.0], Map::new());
        assert!(pipeline(&two).unwrap_err().contains("more than one"));

        // A stage without its dataset.
        let mut bare = Graph::from(default_canvas(&text_model(), false, false, &files()));
        let data = bare.nodes().find(|n| n.kind == kinds::TEXT_DATA).unwrap().id;
        bare.remove(data);
        let found = pipeline(&bare).unwrap();
        let none = |_: &str| None;
        assert!(plan(&found, &bare, &text_model(), &plan_files(&none)).unwrap_err().contains("nothing to learn from"));

        // Heads that do not divide the width.
        let mut odd = Graph::from(default_canvas(&text_model(), false, false, &files()));
        let size = odd.nodes().find(|n| n.kind == kinds::SIZE).unwrap().id;
        odd.node_mut(size).unwrap().set_param("heads", 3);
        assert!(pipeline(&odd).unwrap_err().contains("divide"));
    }

    #[test]
    fn a_common_architecture_replaces_the_canvas_architecture() {
        let mut graph = Graph::from(default_canvas(&text_model(), false, false, &files()));
        let deepseek = commons().into_iter().find(|c| c.name.starts_with("DeepSeek-V4")).unwrap();
        apply_common(&mut graph, &deepseek, [0.0, 0.0]);
        let found = pipeline(&graph).unwrap();
        assert_eq!(found.stages.len(), 2, "the stages stay wired after it");
        let a = &found.architecture;
        assert!(a.hyper_connections.is_some() && a.mtp.is_some());
        let kinds: Vec<&str> = a.layers.iter().map(|l| l["attention"]["kind"].as_str().unwrap()).collect();
        assert_eq!(kinds, ["window", "csa", "hca", "csa", "hca"]);
        assert_eq!(a.layers[0]["ffn"]["hash"], true);
        assert_eq!(a.layers[1]["ffn"]["gating"], "sqrt_softplus");
        assert!(!a.gguf_compatible());
        // Applying another one replaces it, feed-forward blocks included.
        let llama = commons().into_iter().find(|c| c.name.starts_with("Llama")).unwrap();
        apply_common(&mut graph, &llama, [0.0, 0.0]);
        assert!(!graph.nodes().any(|n| n.kind == kinds::MOE));
        assert_eq!(pipeline(&graph).unwrap().architecture.gguf_arch(), Some("llama"));
    }

    #[test]
    fn the_common_architectures_export_as_they_say() {
        let expected = [
            ("Magi (small)", Some("qwen3")),
            ("Qwen3 (small)", Some("qwen3")),
            ("Qwen3 Thinking (small)", Some("qwen3")),
            ("Gemma 3 (small)", Some("gemma3")),
            ("Mistral (small)", Some("llama")),
            ("GLM-4 (small)", None),
            ("DeepSeek-V4 (small)", None),
            ("DeepSeek-V3 (small)", None),
            ("Llama (small)", Some("llama")),
        ];
        let names: Vec<&str> = commons().iter().map(|c| c.name).collect();
        assert_eq!(names, expected.map(|(n, _)| n));
        for (name, arch) in expected {
            let mut model = MagicModel { text: Some(TextSpec::default()), ..text_model() };
            let mut graph = Graph::from(default_canvas(&model, false, false, &files()));
            let common = commons().into_iter().find(|c| c.name == name).unwrap();
            apply_common(&mut graph, &common, [0.0, 0.0]);
            let found = pipeline(&graph).unwrap();
            assert_eq!(found.architecture.gguf_arch(), arch, "{name}");
            derive(&mut graph, &mut model, false);
            let text = model.text.as_ref().unwrap();
            let reasons = common.setup.contains(&kinds::THINKING);
            assert_eq!(text.thinking.is_some(), reasons, "{name}: Thinking is chained in");
            assert_eq!(text.effort.is_some(), common.setup.contains(&kinds::EFFORT), "{name}");
        }
        // Gemma 3: five local layers, then a global one.
        let gemma = commons().into_iter().find(|c| c.name.starts_with("Gemma")).unwrap();
        let mut graph = Graph::from(default_canvas(&text_model(), false, false, &files()));
        apply_common(&mut graph, &gemma, [0.0, 0.0]);
        let a = pipeline(&graph).unwrap().architecture;
        let windows: Vec<u64> = a.layers.iter().map(|l| l["attention"]["window"].as_u64().unwrap_or(0)).collect();
        assert_eq!(windows, [256, 256, 256, 256, 256, 0]);
        assert_eq!(a.layers[0]["norm"], "sandwich");
        assert_eq!(a.layers[0]["ffn"]["activation"], "geglu");
        assert!(a.embed_scale && a.tied);
    }

    #[test]
    fn a_trained_model_opens_with_its_own_architecture() {
        let mut graph = Graph::from(default_canvas(&text_model(), false, false, &files()));
        apply_common(&mut graph, &commons()[1], [0.0, 0.0]);
        let trained = MagicModel { architecture: Some(pipeline(&graph).unwrap().architecture), ..text_model() };
        let reopened = Graph::from(default_canvas(&trained, true, true, &[]));
        let found = pipeline(&reopened).unwrap();
        assert_eq!(found.entry.kind, kinds::BASE);
        assert_eq!(found.architecture.layers, trained.architecture.as_ref().unwrap().layers);
        assert_eq!(found.stages.len(), 1, "fine-tuning");
    }

    #[test]
    fn dataset_samples_are_checked() {
        let spec = TextSpec { effort: Some(EffortSpec::default()), ..TextSpec::default() };
        let check = |sample: &str| check_sample(true, sample, &spec);
        assert_eq!(check(r#"{"prompt": "hi", "response": "yo"}"#), Ok("Request and answer"));
        assert_eq!(
            check(r#"{"prompt": "hi", "thinking": "easy", "response": "yo", "effort": "low"}"#),
            Ok("Reasoning")
        );
        assert!(check(r#"{"prompt": "hi", "response": "yo", "effort": "extreme"}"#).unwrap_err().contains("levels"));
        assert_eq!(check(r#"{"prompt": "hi", "rejected": "no"}"#), Ok("Not to answer"));
        assert_eq!(check(r#"{"context": "long", "compact": "short"}"#), Ok("Compaction"));
        assert_eq!(check(r#"{"text": "plain"}"#), Ok("Text"));
        let tool = r#"{"prompt": "go", "response": "<tool_call>{\"name\": \"press\", \"arguments\": {\"key\": \"A\"}}</tool_call>"}"#;
        assert_eq!(check(tool), Ok("Request and answer"));
        let conversation = r#"{"tools": [{"name": "web_search"}], "messages": [{"role": "user", "content": "news?"},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "web_search", "arguments": {"query": "news"}}]},
            {"role": "tool", "content": "rain"}, {"role": "assistant", "content": "It rains."}]}"#
            .replace('\n', " ");
        assert_eq!(check(&conversation), Ok("Tool use"));
        assert!(
            check(
                &conversation.replace("\"name\": \"web_search\", \"arguments\"", "\"name\": \"other\", \"arguments\"")
            )
            .unwrap_err()
            .contains("“tools”")
        );
        assert!(check("{not json").is_err());
        assert!(check(r#"{"prompt": "hi"}"#).unwrap_err().contains("response"));
        assert!(check(r#"{"messages": [{"role": "user", "content": "hi"}]}"#).unwrap_err().contains("assistant"));
        assert!(
            check(r#"{"prompt": "go", "response": "<tool_call>{\"arguments\": {}}</tool_call>"}"#)
                .unwrap_err()
                .contains("name")
        );
        assert!(
            check(r#"{"prompt": "go", "response": "<tool_call>{\"name\": \"x\"}"}"#)
                .unwrap_err()
                .contains("not closed")
        );
        assert!(check_sample(false, "a <tool_call>oops</tool_call>", &spec).is_err());

        let samples = split_samples("a.md", "one\r\n\r\ntwo\n\n\nthree");
        assert_eq!(samples, ["one", "two", "three"]);
        assert_eq!(join_samples("a.md", &samples), "one\n\ntwo\n\nthree\n");
        assert_eq!(split_samples("a.jsonl", "{}\n\n{}\n"), ["{}", "{}"]);
    }

    #[test]
    fn the_examples_are_valid_samples() {
        let spec = text_model().text.unwrap();
        for (file, content, _) in example_datasets() {
            let samples = split_samples(&file, &content);
            assert!(samples.len() >= 5, "{file}");
            for sample in &samples {
                let checked = check_sample(is_jsonl(&file), sample, &spec);
                assert!(checked.is_ok(), "{file}: {sample}: {checked:?}");
            }
        }
        let kinds: BTreeSet<&str> = example_datasets()
            .iter()
            .filter(|(f, ..)| is_jsonl(f))
            .flat_map(|(f, content, _)| split_samples(f, content).into_iter().map(move |s| (f.clone(), s)))
            .filter_map(|(f, s)| check_sample(is_jsonl(&f), &s, &spec).ok())
            .collect();
        assert_eq!(
            kinds,
            BTreeSet::from(["Conversation", "Not to answer", "Reasoning", "Request and answer", "Tool use"]),
            "every kind of sample is shown"
        );
    }
}
