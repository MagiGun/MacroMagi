//! AI blocks: the models prepared in the Magic tab, used in a macro.
//!
//! A model asset (a `.magic` file) carries its own configuration – which part
//! of the screen it looks at, what its answers mean, what to do about them.
//! These blocks run it; their `settings` adjust the model's declared settings
//! for this one block.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::{Category, DataType, NodeDef, ParamDef, ParamKind, Registry, ids, ports};
use crate::assets::AssetStore;
use crate::graph::Node;
use crate::text::format_ms;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AiParams {
    /// Id of the model asset.
    pub model: String,
    /// Values for the model's adjustable settings, by key.
    pub settings: Map<String, Value>,
    /// The display a model that looks at the middle of the screen looks at.
    pub display: u8,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AiWatchParams {
    pub model: String,
    pub settings: Map<String, Value>,
    pub display: u8,
    /// Press the model's key when it says so. Off: only watch (and count).
    pub press: bool,
    /// Stop watching after this long, and continue; zero watches until the macro stops.
    pub duration_ms: f64,
    /// At most this many looks per second; zero for as many as possible.
    pub max_fps: f64,
}

impl Default for AiWatchParams {
    fn default() -> Self {
        Self { model: String::new(), settings: Map::new(), display: 0, press: true, duration_ms: 0.0, max_fps: 0.0 }
    }
}

fn model_name(assets: &AssetStore, id: &str) -> String {
    assets.get(id).map_or_else(|| "No model chosen".to_owned(), |a| a.name.clone())
}

fn summary(node: &Node, assets: &AssetStore) -> Vec<String> {
    let p: AiWatchParams = node.params_or_default();
    let mut lines = vec![model_name(assets, &p.model)];
    if node.kind == ids::AI_WATCH {
        lines.push(if p.press { "Acts on what it sees".into() } else { "Only watches".into() });
        if p.duration_ms > 0.0 {
            lines.push(format!("For {}", format_ms(p.duration_ms)));
        }
    }
    if p.display > 0 {
        lines.push(format!("Display {}", p.display + 1));
    }
    if !p.settings.is_empty() {
        lines.push(format!("{} settings changed", p.settings.len()));
    }
    lines
}

/// An AI Agent block: asks a text model and, if set to, does what it calls for.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AiAgentParams {
    pub model: String,
    /// The request, when none is wired in.
    pub request: String,
    /// Carry out the tool calls MacroMagi knows: press, type, click, move, wait and log.
    pub run_tools: bool,
    /// The longest answer, in tokens; 0 uses the model's own setting.
    pub max_tokens: u32,
    /// How freely it writes; below 0 uses the model's own setting.
    pub temperature: f64,
}

impl Default for AiAgentParams {
    fn default() -> Self {
        Self { model: String::new(), request: String::new(), run_tools: false, max_tokens: 0, temperature: -1.0 }
    }
}

/// The tools an AI Agent block can carry out itself, with their arguments.
pub const AGENT_TOOLS: [(&str, &str); 6] = [
    ("press", "{\"key\": \"Enter\"} – taps a key"),
    ("type", "{\"text\": \"…\"} – types text"),
    ("click", "{\"x\": 10, \"y\": 20, \"button\": \"left\"} – clicks, where given"),
    ("move", "{\"x\": 10, \"y\": 20} – moves the pointer"),
    ("wait", "{\"ms\": 500} – waits"),
    ("log", "{\"text\": \"…\"} – writes to the console"),
];

fn agent_summary(node: &Node, assets: &AssetStore) -> Vec<String> {
    let p: AiAgentParams = node.params_or_default();
    let mut lines = vec![model_name(assets, &p.model)];
    if !p.request.trim().is_empty() {
        let request: String = p.request.trim().chars().take(40).collect();
        lines.push(format!("“{request}”"));
    }
    if p.run_tools {
        lines.push("Carries out its tool calls".into());
    }
    lines
}

/// AI Watch's flow output taken each time it acts.
pub const ACTED: &str = "acted";

fn model_param() -> ParamDef {
    ParamDef::new("model", "Model", ParamKind::Model)
        .help("A model prepared in the Magic tab. Use “Use in Macro” there, or load a .magic file.")
}

fn settings_param() -> ParamDef {
    ParamDef::new("settings", "Model settings", ParamKind::ModelSettings)
        .help("The model's own adjustable settings, changed for this block only.")
}

fn display_param() -> ParamDef {
    ParamDef::new("display", "Display", ParamKind::Display)
        .help("For models that look at the middle of the screen: which display.")
}

pub(super) fn register(registry: &mut Registry) {
    registry.register(
        NodeDef::new(
            ids::AI_WATCH,
            "AI Watch",
            Category::Ai,
            "Keeps showing part of the screen to an AI model and does what the model is set up to do – for \
             example press a key the moment it sees the right thing. Each time it acts, the blocks under \
             “Acted” run (with “Times acted” and “Last answer” up to date), then it watches again. Runs until \
             the macro stops, or for as long as set; then it continues from “Done”.",
        )
        .keywords(&["ai", "model", "magic", "vision", "watch", "neural", "onnx", "skill check", "detect"])
        .flow_in()
        .flow_out(ACTED, "Acted")
        .flow_out(ports::OUT, "Done")
        .data_out("hits", "Times acted", DataType::Number)
        .data_out("label", "Last answer", DataType::Text)
        .param(model_param())
        .param(settings_param())
        .param(display_param())
        .param(ParamDef::new("press", "Press the model's key", ParamKind::Bool).help(
            "Off: the model only watches and runs the blocks under “Acted” – try it safely, or do something \
             else than pressing the key.",
        ))
        .param(
            ParamDef::new("duration_ms", "Stop after", ParamKind::Duration)
                .help("0 keeps watching until the macro is stopped."),
        )
        .param(
            ParamDef::new("max_fps", "Looks per second", ParamKind::float(0.0, 1000.0, ""))
                .help("0 looks as often as the computer can. A limit saves power."),
        )
        .defaults(&AiWatchParams::default())
        .summarize(summary),
    );

    registry.register(
        NodeDef::new(
            ids::AI_CLASSIFY,
            "AI Model",
            Category::Ai,
            "Shows part of the screen to an AI model once and gives its answer: the likeliest class, how \
             sure the model is, and whether the model's setup says to act on it.",
        )
        .keywords(&["ai", "model", "magic", "vision", "classify", "neural", "onnx", "recognise", "recognize"])
        .data_out("act", "Act", DataType::Bool)
        .data_out("label", "Answer", DataType::Text)
        .data_out("class", "Class number", DataType::Number)
        .data_out("confidence", "Confidence", DataType::Number)
        .data_out("delay", "Delay (ms)", DataType::Number)
        .param(model_param())
        .param(settings_param())
        .param(display_param())
        .defaults(&AiParams::default())
        .summarize(summary),
    );

    registry.register(
        NodeDef::new(
            ids::AI_AGENT,
            "AI Agent",
            Category::Ai,
            "Asks a text model made in the Magic tab and gives its answer, what it thought, and the tool it \
             called with its arguments. Set to, it carries out the tool calls MacroMagi knows – press, type, \
             click, move, wait and log – and other tools are left to the blocks after it.",
        )
        .keywords(&["ai", "agent", "text", "llm", "language model", "chat", "tool", "magic", "prompt", "ask"])
        .flow_in()
        .flow_out(ports::OUT, "Then")
        .data_in("request", "Request", DataType::Text)
        .data_out("answer", "Answer", DataType::Text)
        .data_out("thinking", "Thinking", DataType::Text)
        .data_out("tool", "Tool", DataType::Text)
        .data_out("args", "Arguments", DataType::Text)
        .data_out("calls", "Tool calls", DataType::Number)
        .data_out("raw", "Everything it wrote", DataType::Text)
        .param(model_param())
        .param(
            ParamDef::new("request", "Request", ParamKind::Text { multiline: true })
                .help("Used when nothing is wired into “Request”."),
        )
        .param(ParamDef::new("run_tools", "Carry out its tool calls", ParamKind::Bool).help(
            "press {key}, type {text}, click {x, y, button}, move {x, y}, wait {ms} and log {text}. Off: the \
             block only reports them.",
        ))
        .param(
            ParamDef::new("max_tokens", "Longest answer", ParamKind::int(0, 8192, "tokens"))
                .help("0: the model's own setting."),
        )
        .param(
            ParamDef::new("temperature", "Temperature", ParamKind::float(-1.0, 2.0, ""))
                .help("0 always writes the likeliest; higher is freer. Below 0: the model's own setting."),
        )
        .defaults(&AiAgentParams::default())
        .summarize(agent_summary),
    );
}
