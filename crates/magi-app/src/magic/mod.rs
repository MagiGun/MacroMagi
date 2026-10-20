//! The Magic tab's side of things that is not drawing: importing models, the
//! model being edited, and trying it out live.

pub mod import;
pub mod train;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use magi_core::geom::Displays;
use magi_core::graph::{Fragment, Node, PortRef, Wire};
use magi_core::host::Host;
use magi_core::input::InputEvent;
use magi_core::magic::{DataItem, Decision, MagicFile, MagicModel, ModelKind};
use magi_core::nodes::ai::AiWatchParams;
use magi_core::nodes::flow::StartParams;
use magi_core::nodes::notes::CommentParams;
use magi_core::nodes::{Registry, ids, ports, studio};
use magi_core::project::Project;

use crate::history::History;

/// What undo and redo bring back: the canvas (the setup is in its blocks)
/// and the pictures. Data files are only dropped when the model is saved,
/// so undoing a removal finds them again.
#[derive(Clone, PartialEq)]
pub struct Snapshot {
    graph: magi_core::graph::Graph,
    training: Vec<DataItem>,
    finetune: Vec<DataItem>,
}

/// The model open in the Magic tab.
pub struct MagicDoc {
    pub file: MagicFile,
    pub path: Option<PathBuf>,
    /// What the importer found and guessed.
    pub notes: Vec<String>,
    /// The model's shapes, read when it was loaded.
    pub shape: Option<magi_ml::Shape>,
    /// The Magic tab's canvas, kept in `file.model.studio` too. The model's
    /// settings are made from it (`studio::derive`).
    pub studio: Project,
    /// Bumped when the model, weights or data files change, which the
    /// description alone does not show.
    revision: u64,
    /// The description and revision as last saved, to tell whether there are changes.
    saved: Option<(MagicModel, u64)>,
    /// The description and revision as last put into the macro, where it is kept too.
    kept: Option<(MagicModel, u64)>,
    history: History<Snapshot>,
}

impl MagicDoc {
    pub fn new(mut file: MagicFile, path: Option<PathBuf>, notes: Vec<String>) -> Self {
        let shape = (!file.onnx.is_empty()).then(|| magi_ml::inspect(&file.onnx).ok()).flatten();
        // A model from before the canvas gets one, built from its setup.
        let canvas = file.model.studio.take().unwrap_or_else(|| {
            let datasets: Vec<(String, bool)> = file.model.datasets.iter().map(|f| (f.clone(), false)).collect();
            studio::default_canvas(&file.model, !file.weights.is_empty(), !file.onnx.is_empty(), &datasets)
        });
        let studio = Project { graph: magi_core::graph::Graph::from(canvas), ..Project::new() };
        let history = History::new(&Snapshot {
            graph: studio.graph.clone(),
            training: file.model.training.clone(),
            finetune: file.model.finetune.clone(),
        });
        let mut doc = Self { file, path, notes, shape, studio, revision: 0, saved: None, kept: None, history };
        doc.sync_canvas();
        doc.history = History::new(&doc.snapshot());
        doc.saved = doc.path.as_ref().map(|_| (doc.file.model.clone(), 0));
        doc
    }

    /// Makes the model what the canvas says – its setup from the setup blocks
    /// chained into the start, its datasets from the Text Dataset blocks –
    /// notes what is not connected, and puts the canvas into the model.
    pub fn sync_canvas(&mut self) {
        let has_onnx = !self.file.onnx.is_empty();
        studio::derive(&mut self.studio.graph, &mut self.file.model, has_onnx);
        // The sliders' values, on what they change.
        let configured = self.file.model.configured(&serde_json::Map::new());
        if configured != self.file.model {
            self.file.model = configured;
        }
        let mut datasets: Vec<String> = Vec::new();
        for node in self.studio.graph.nodes().filter(|n| n.kind == studio::kinds::TEXT_DATA) {
            let params: studio::TextDataParams = node.params_or_default();
            for file in params.positive.into_iter().chain(params.negative) {
                if self.file.data.contains_key(&file) && !datasets.contains(&file) {
                    datasets.push(file);
                }
            }
        }
        if self.file.model.datasets != datasets {
            self.file.model.datasets = datasets;
        }
        let canvas: Fragment = self.studio.graph.clone().into();
        if self.file.model.studio.as_ref() != Some(&canvas) {
            self.file.model.studio = Some(canvas);
        }
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            graph: self.studio.graph.clone(),
            training: self.file.model.training.clone(),
            finetune: self.file.model.finetune.clone(),
        }
    }

    /// Call once per frame; `settled` is false during a drag or while typing.
    pub fn track_history(&mut self, settled: bool) {
        let snapshot = self.snapshot();
        self.history.track(&snapshot, settled);
    }

    pub fn can_undo(&self) -> bool {
        self.history.can_undo(&self.snapshot())
    }

    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    pub fn undo(&mut self) {
        let mut snapshot = self.snapshot();
        self.history.undo(&mut snapshot);
        self.restore(snapshot);
    }

    pub fn redo(&mut self) {
        let mut snapshot = self.snapshot();
        self.history.redo(&mut snapshot);
        self.restore(snapshot);
    }

    fn restore(&mut self, snapshot: Snapshot) {
        self.studio.graph = snapshot.graph;
        self.file.model.training = snapshot.training;
        self.file.model.finetune = snapshot.finetune;
        self.sync_canvas();
    }

    /// Notes a change to the model, its weights or its data files.
    pub fn files_changed(&mut self) {
        self.revision += 1;
    }

    /// Puts what training made into the model.
    pub fn apply_trained(&mut self, trained: train::Trained) {
        self.files_changed();
        self.file.onnx = trained.onnx;
        self.file.weights = trained.weights;
        let model = &mut self.file.model;
        model.kind = trained.kind;
        model.architecture = Some(trained.architecture);
        model.trained = Some(trained.record);
        if let Some(input) = trained.input {
            model.input = input;
            // The Picture Preparation block follows what the model was trained with.
            let value = studio::section_of(studio::kinds::PREP, &self.file.model);
            for node in self.studio.graph.nodes_mut().filter(|n| n.kind == studio::kinds::PREP) {
                node.set_param("value", value.clone());
            }
        }
        self.shape = magi_ml::inspect(&self.file.onnx).ok();
        self.sync_canvas();
    }

    /// The Compile block's settings: the one on the Model wire, or any.
    pub fn compile_settings(&self) -> Option<studio::CompileParams> {
        studio::pipeline(&self.studio.graph).ok().and_then(|p| p.compile).map(|(_, params)| params).or_else(|| {
            self.studio.graph.nodes().find(|n| n.kind == studio::kinds::COMPILE).map(|n| n.params_or_default())
        })
    }

    /// Why the canvas cannot train the model now, if it cannot.
    pub fn training_problem(&self) -> Option<String> {
        let pipeline = match studio::pipeline(&self.studio.graph) {
            Ok(pipeline) => pipeline,
            Err(problem) => return Some(problem),
        };
        if !pipeline.trains() {
            return Some("Add a training stage after the model's start (and its architecture) to train it.".into());
        }
        if pipeline.entry.kind == studio::kinds::BASE
            && pipeline.entry.params_or_default::<studio::BaseParams>().path.trim().is_empty()
            && self.file.weights.is_empty()
        {
            return Some(
                "This model has no weights to train further: use Start from Zero, or choose a base model file.".into(),
            );
        }
        match &pipeline.compile {
            None => {
                return Some(
                    "Put Compile .magic Package at the end of the Model wire: it saves the trained model.".into(),
                );
            }
            Some((_, params)) if params.folder.trim().is_empty() => {
                return Some("Choose the folder Compile .magic Package saves into (select the block).".into());
            }
            _ => {}
        }
        let pictures = self.file.model.training.len() + self.file.model.finetune.len();
        let files = studio::PlanFiles {
            dataset: &|file: &str| self.file.data.contains_key(file).then(|| file.to_owned()),
            weights: None,
            pictures: (pictures > 0).then(String::new),
            out_dir: String::new(),
            devices: Vec::new(),
        };
        studio::plan(&pipeline, &self.studio.graph, &self.file.model, &files).err()
    }

    /// Saves the package as the Compile block says: the `.magic` file in its
    /// folder, and a `.gguf` and a Hugging Face folder when asked for and
    /// possible. Returns what was written, and what was skipped and why.
    pub fn compile(&mut self) -> Result<(Vec<PathBuf>, Vec<String>), String> {
        let params = self.compile_settings().ok_or("Add a Compile .magic Package block to the canvas.")?;
        if params.folder.trim().is_empty() {
            return Err("Choose the folder Compile .magic Package saves into.".into());
        }
        let folder = PathBuf::from(params.folder.trim());
        std::fs::create_dir_all(&folder)
            .map_err(|e| format!("The folder {} could not be made: {e}", folder.display()))?;
        let title = self.title();
        let stem = file_stem(if params.file.trim().is_empty() { &title } else { params.file.trim() });
        let include = &mut self.file.model.include;
        include.weights = params.weights;
        include.datasets = params.datasets;
        (include.training, include.finetune) = (params.pictures, params.pictures);
        include.blocks = params.blocks;
        if self.file.model.blocks.is_none() {
            self.file.model.blocks = Some(starter_blocks(&self.file.model, "", &Registry::builtin()));
        }
        let path = folder.join(format!("{stem}.magic"));
        self.file.save(&path).map_err(|e| format!("Could not save {}: {e}", path.display()))?;
        self.mark_saved(path.clone());
        let mut written = vec![path];
        let mut skipped = Vec::new();
        let text = self.file.model.kind == ModelKind::Text;
        let exportable =
            self.file.model.architecture.as_ref().is_some_and(|a| a.gguf_compatible()) && !self.file.weights.is_empty();
        if (params.gguf || params.huggingface) && text && !exportable {
            skipped.push(
                "GGUF and Hugging Face need a model trained in MacroMagi from Transformer Layers only; the .magic \
                 file was saved."
                    .to_owned(),
            );
        } else if text {
            if params.gguf {
                let quant = magi_core::export::Quant::parse(&params.quant).unwrap_or_default();
                // The quantization ends the name, as LM Studio and others expect.
                let gguf = folder.join(format!("{stem}-{}.gguf", quant.name()));
                let bytes = magi_core::export::gguf_with(&self.file.model, &self.file.weights, quant)
                    .map_err(|e| e.to_string())?;
                std::fs::write(&gguf, bytes).map_err(|e| format!("Could not save {}: {e}", gguf.display()))?;
                written.push(gguf.clone());
                if params.lmstudio {
                    match self.add_to_lmstudio(&gguf) {
                        Ok(paths) => written.extend(paths),
                        Err(error) => skipped.push(error),
                    }
                }
            }
            if params.huggingface {
                let hf = folder.join(format!("{stem}-huggingface"));
                magi_core::export::huggingface(&self.file.model, &self.file.weights, &hf).map_err(|e| e.to_string())?;
                written.push(hf);
            }
        }
        Ok((written, skipped))
    }

    /// Puts a GGUF file of this model into LM Studio: the file into its
    /// models folder (`<owner>/<Name>-GGUF/`), and a `model.yaml` with a
    /// `manifest.json` into its hub folder, so LM Studio shows the model's
    /// size, context, Enable Thinking and Reasoning Effort. Returns what was written.
    pub fn add_to_lmstudio(&self, gguf: &std::path::Path) -> Result<Vec<PathBuf>, String> {
        let root = lmstudio_root().ok_or(
            "LM Studio was not found (its folder is .lmstudio in your home folder). Install and open it once first.",
        )?;
        let model = &self.file.model;
        let exported = magi_core::export::describe(model, &self.file.weights).map_err(|e| e.to_string())?;
        let owner = lmstudio_owner(model);
        let repo = format!("{}-GGUF", file_stem(&self.title()).replace(' ', "-"));
        let folder = root.join("models").join(&owner).join(&repo);
        std::fs::create_dir_all(&folder).map_err(|e| format!("Could not make {}: {e}", folder.display()))?;
        let name = gguf.file_name().ok_or("The GGUF file has no name.")?;
        let copy = folder.join(name);
        std::fs::copy(gguf, &copy).map_err(|e| format!("Could not copy the GGUF file into LM Studio: {e}"))?;
        let bytes = std::fs::metadata(&copy).map(|m| m.len()).unwrap_or_default();
        let hub = root
            .join("hub")
            .join("models")
            .join(magi_core::export::lmstudio_key(&owner))
            .join(magi_core::export::lmstudio_key(&model.name));
        std::fs::create_dir_all(&hub).map_err(|e| format!("Could not make {}: {e}", hub.display()))?;
        let yaml = magi_core::export::lmstudio_model_yaml(model, &exported, &owner, &repo, bytes)
            .map_err(|e| e.to_string())?;
        std::fs::write(hub.join("model.yaml"), yaml).map_err(|e| format!("Could not write model.yaml: {e}"))?;
        std::fs::write(hub.join("manifest.json"), magi_core::export::lmstudio_manifest(model, &owner, &repo))
            .map_err(|e| format!("Could not write manifest.json: {e}"))?;
        Ok(vec![copy, hub.join("model.yaml")])
    }

    /// The GGUF file the last compile wrote for this model, if it is there.
    pub fn compiled_gguf(&self) -> Option<PathBuf> {
        let params = self.compile_settings()?;
        if !params.gguf || params.folder.trim().is_empty() {
            return None;
        }
        let title = self.title();
        let stem = file_stem(if params.file.trim().is_empty() { &title } else { params.file.trim() });
        let quant = magi_core::export::Quant::parse(&params.quant).unwrap_or_default();
        Some(PathBuf::from(params.folder.trim()).join(format!("{stem}-{}.gguf", quant.name()))).filter(|p| p.is_file())
    }

    pub fn is_dirty(&self) -> bool {
        self.saved.as_ref().is_none_or(|(model, revision)| *model != self.file.model || *revision != self.revision)
    }

    /// Whether closing it would lose changes: ones neither saved to a file
    /// nor put into the macro.
    pub fn would_lose_changes(&self) -> bool {
        self.is_dirty()
            && self
                .kept
                .as_ref()
                .is_none_or(|(model, revision)| *model != self.file.model || *revision != self.revision)
    }

    /// Notes that the model as it is now is in the macro.
    pub fn mark_kept(&mut self) {
        self.kept = Some((self.file.model.clone(), self.revision));
    }

    pub fn mark_saved(&mut self, path: PathBuf) {
        self.path = Some(path);
        self.saved = Some((self.file.model.clone(), self.revision));
    }

    pub fn title(&self) -> String {
        if self.file.model.name.trim().is_empty() { "Untitled model".into() } else { self.file.model.name.clone() }
    }

    /// Replaces the model itself, keeping the configuration.
    pub fn replace_onnx(&mut self, onnx: Vec<u8>) -> Result<(), String> {
        let shape = magi_ml::inspect(&onnx).map_err(|e| e.0)?;
        self.file.model.input = shape.suggested(self.file.model.input.clone());
        let value = studio::section_of(studio::kinds::PREP, &self.file.model);
        for node in self.studio.graph.nodes_mut().filter(|n| n.kind == studio::kinds::PREP) {
            node.set_param("value", value.clone());
        }
        self.file.onnx = onnx;
        self.shape = Some(shape);
        self.files_changed();
        Ok(())
    }
}

/// LM Studio's own folder: `.lmstudio` in the home folder (or `MAGI_LMSTUDIO_DIR`).
pub fn lmstudio_root() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MAGI_LMSTUDIO_DIR") {
        return Some(PathBuf::from(dir));
    }
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
    Some(PathBuf::from(home).join(".lmstudio")).filter(|p| p.is_dir())
}

/// Whose model LM Studio files it under: its author, or MacroMagi.
fn lmstudio_owner(model: &MagicModel) -> String {
    let owner: String = model.author.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_').collect();
    if owner.is_empty() { "MacroMagi".into() } else { owner }
}

/// A name usable as a file name.
pub fn file_stem(name: &str) -> String {
    let cleaned: String =
        name.chars().map(|c| if c.is_alphanumeric() || " -_.()".contains(c) { c } else { '-' }).collect();
    let cleaned = cleaned.trim().trim_matches('.').to_owned();
    if cleaned.is_empty() { "model".into() } else { cleaned }
}

/// A new model, with a canvas ready to train it: for a text model, an
/// example agent with its datasets, which trains in a minute or two.
pub fn new_model(text: bool) -> MagicFile {
    let mut model = MagicModel {
        name: if text { "Magi" } else { "My vision model" }.into(),
        description: if text {
            "A small assistant: it chats, thinks at three effort levels and uses tools such as web search.".into()
        } else {
            "Tells the screen's states apart.".into()
        },
        ..MagicModel::default()
    };
    let mut data = std::collections::BTreeMap::new();
    let mut datasets = Vec::new();
    if text {
        model.kind = ModelKind::Text;
        // General tools, as the example datasets use them; MacroMagi's own are a button away in Text & Tools.
        let mut spec = magi_core::magic::TextSpec {
            temperature: 0.3,
            max_tokens: 200,
            system: studio::EXAMPLE_SYSTEM.into(),
            tools: studio::example_tools(),
            thinking: Some(magi_core::magic::ThinkingSpec::default()),
            effort: Some(magi_core::magic::EffortSpec::default()),
            ..Default::default()
        };
        spec.ensure_specials();
        model.text = Some(spec);
        for (file, content, negative) in studio::example_datasets() {
            data.insert(file.clone(), content.into_bytes());
            datasets.push((file, negative));
        }
    } else {
        model.kind = ModelKind::Vision;
        model.input.width = 64;
        model.input.height = 64;
        model.classes = vec![
            magi_core::magic::ClassSpec { label: "Yes".into(), act: true, threshold: 0.8, ..Default::default() },
            magi_core::magic::ClassSpec { label: "No".into(), threshold: 0.5, ..Default::default() },
        ];
    }
    model.studio = Some(studio::default_canvas(&model, false, false, &datasets));
    model.datasets = datasets.iter().map(|(f, _)| f.clone()).collect();
    MagicFile { model, data, ..MagicFile::default() }
}

/// One of the tools the AI Agent block carries out, as a tool of a model.
pub fn agent_tool(name: &str, about: &str) -> magi_core::magic::ToolSpec {
    let property = |ty: &str| serde_json::json!({ "type": ty });
    let properties = match name {
        "press" => serde_json::json!({ "key": property("string") }),
        "type" | "log" => serde_json::json!({ "text": property("string") }),
        "click" => {
            serde_json::json!({ "x": property("number"), "y": property("number"), "button": property("string") })
        }
        "move" => serde_json::json!({ "x": property("number"), "y": property("number") }),
        "wait" => serde_json::json!({ "ms": property("number") }),
        _ => serde_json::json!({}),
    };
    magi_core::magic::ToolSpec {
        name: name.to_owned(),
        description: about.split(" – ").nth(1).unwrap_or(about).to_owned(),
        parameters: serde_json::json!({ "type": "object", "properties": properties }),
    }
}

// The blocks a model comes with: a Start block and an AI Watch block (an AI
/// Agent and a Log block for a text model), with a comment saying how to use
/// them. `model` is the asset id in the macro.
pub fn starter_blocks(model: &MagicModel, asset: &str, registry: &Registry) -> Fragment {
    let defaults = |kind: &str| registry.get(kind).map(|def| def.default_params()).unwrap_or_default();
    if model.kind == ModelKind::Text {
        return text_starter_blocks(model, asset, registry);
    }
    let start = Node::new(1, ids::START, [0.0, 0.0], {
        let mut params = defaults(ids::START);
        params.extend(
            serde_json::to_value(StartParams::default()).ok().and_then(|v| v.as_object().cloned()).unwrap_or_default(),
        );
        params
    });
    let watch_params =
        AiWatchParams { model: asset.to_owned(), display: model.capture.display, ..AiWatchParams::default() };
    let mut watch = Node::new(2, ids::AI_WATCH, [0.0, 130.0], defaults(ids::AI_WATCH));
    watch.set_params(&watch_params);
    let note = CommentParams {
        text: format!(
            "“{}” watches the screen while this runs and presses {} when it should. Press Run (or give the Start \
             block a hotkey) to start, and the stop hotkey to stop. Its settings are in the AI Watch block.",
            model.name,
            model.action.key.label()
        ),
        width: 260.0,
        height: 130.0,
        ..CommentParams::default()
    };
    let mut comment = Node::new(3, ids::COMMENT, [260.0, 0.0], defaults(ids::COMMENT));
    comment.label = model.name.clone();
    comment.set_params(&note);
    Fragment {
        nodes: vec![start, watch, comment],
        wires: vec![Wire { from: PortRef::new(1, ports::OUT), to: PortRef::new(2, ports::FLOW_IN) }],
        groups: Vec::new(),
    }
}

fn text_starter_blocks(model: &MagicModel, asset: &str, registry: &Registry) -> Fragment {
    let defaults = |kind: &str| registry.get(kind).map(|def| def.default_params()).unwrap_or_default();
    let mut start = Node::new(1, ids::START, [0.0, 0.0], defaults(ids::START));
    start.set_params(&StartParams::default());
    let mut agent = Node::new(2, ids::AI_AGENT, [0.0, 130.0], defaults(ids::AI_AGENT));
    agent.set_params(&magi_core::nodes::ai::AiAgentParams {
        model: asset.to_owned(),
        request: "Say hello".into(),
        ..Default::default()
    });
    let log = Node::new(3, ids::LOG, [0.0, 330.0], defaults(ids::LOG));
    let note = CommentParams {
        text: format!(
            "“{}” answers the request in the AI Agent block, and the Log block writes its answer. Turn on \
             “Carry out its tool calls” to let it press, type and click; its other tools come out of “Tool” and \
             “Arguments” for the blocks after it.",
            model.name
        ),
        width: 280.0,
        height: 140.0,
        ..CommentParams::default()
    };
    let mut comment = Node::new(4, ids::COMMENT, [280.0, 0.0], defaults(ids::COMMENT));
    comment.label = model.name.clone();
    comment.set_params(&note);
    Fragment {
        nodes: vec![start, agent, log, comment],
        wires: vec![
            Wire { from: PortRef::new(1, ports::OUT), to: PortRef::new(2, ports::FLOW_IN) },
            Wire { from: PortRef::new(2, ports::OUT), to: PortRef::new(3, ports::FLOW_IN) },
            Wire { from: PortRef::new(2, "answer"), to: PortRef::new(3, "value") },
        ],
        groups: Vec::new(),
    }
}

/// What the live test shows.
#[derive(Default)]
pub struct TestState {
    /// The picture the model saw last (RGBA, as captured) and its size.
    pub picture: Option<(Vec<u8>, u32, u32)>,
    /// Bumped with every new picture, so the screen knows to update.
    pub frame: u64,
    pub probabilities: Vec<f32>,
    pub decision: Option<Decision>,
    pub looks_per_second: f64,
    pub engine: String,
    pub acted: u64,
    pub error: Option<String>,
}

/// Runs a model on the screen, over and over, in the background.
pub struct Tester {
    pub state: Arc<Mutex<TestState>>,
    /// The configuration in use; changes made while testing take effect at once.
    pub config: Arc<Mutex<MagicModel>>,
    stop: Arc<AtomicBool>,
}

impl Tester {
    /// Starts testing. With `press`, the model's key is pressed (through
    /// `host`) whenever the model says to act.
    pub fn start(
        onnx: Vec<u8>,
        config: MagicModel,
        displays: Displays,
        host: Arc<dyn Host>,
        press: bool,
        repaint: impl Fn() + Send + 'static,
    ) -> Self {
        let state = Arc::new(Mutex::new(TestState::default()));
        let config = Arc::new(Mutex::new(config));
        let stop = Arc::new(AtomicBool::new(false));
        let (for_state, for_config, for_stop) = (Arc::clone(&state), Arc::clone(&config), Arc::clone(&stop));
        std::thread::Builder::new()
            .name("magi-magic-test".into())
            .spawn(move || {
                run_test(&onnx, &for_config, &displays, host.as_ref(), press, &for_state, &for_stop, &repaint)
            })
            .expect("a thread can be started");
        Self { state, config, stop }
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl Drop for Tester {
    fn drop(&mut self) {
        self.stop();
    }
}

#[allow(clippy::too_many_arguments)]
fn run_test(
    onnx: &[u8],
    config: &Mutex<MagicModel>,
    displays: &Displays,
    host: &dyn Host,
    press: bool,
    state: &Mutex<TestState>,
    stop: &AtomicBool,
    repaint: &dyn Fn(),
) {
    let lock = || state.lock().unwrap_or_else(|p| p.into_inner());
    let spec = config.lock().unwrap_or_else(|p| p.into_inner()).input.clone();
    let mut model = match magi_ml::Model::load(onnx, &spec) {
        Ok(model) => model,
        Err(error) => {
            lock().error = Some(error.0);
            repaint();
            return;
        }
    };
    lock().engine = model.engine().label();
    let (mut looks, mut counted_from, mut last_repaint) = (0u32, Instant::now(), Instant::now());
    let mut resting_until: Option<Instant> = None;
    while !stop.load(Ordering::Relaxed) {
        let current = config.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let Some(rect) = current.capture.rect(displays, None) else {
            lock().error = Some("there is no display to look at".into());
            break;
        };
        let frame = match host.capture(Some(rect)) {
            Ok(frame) => frame,
            Err(error) => {
                lock().error = Some(error.0);
                break;
            }
        };
        let probabilities = match model.classify(&frame.rgba, frame.width, frame.height) {
            Ok(p) => p,
            Err(error) => {
                lock().error = Some(error.0);
                break;
            }
        };
        let decision = current.decide(&probabilities);
        looks += 1;
        let resting = resting_until.is_some_and(|until| Instant::now() < until);
        let act = decision.act && !resting;
        if act {
            resting_until = Some(
                Instant::now()
                    + Duration::from_secs_f64((decision.delay_ms + current.action.cooldown_ms).max(0.0) / 1000.0),
            );
        }
        {
            let mut shown = lock();
            shown.probabilities = probabilities;
            shown.decision = Some(decision.clone());
            if act {
                shown.acted += 1;
            }
            if last_repaint.elapsed() >= Duration::from_millis(33) {
                shown.picture = Some((frame.rgba, frame.width, frame.height));
                shown.frame += 1;
            }
            if counted_from.elapsed() >= Duration::from_secs(1) {
                shown.looks_per_second = f64::from(looks) / counted_from.elapsed().as_secs_f64();
                (looks, counted_from) = (0, Instant::now());
            }
        }
        if act && press {
            std::thread::sleep(Duration::from_secs_f64(decision.delay_ms.max(0.0) / 1000.0));
            let (down, up) = match current.action.key {
                magi_core::input::Button::Key(key) => (InputEvent::KeyDown { key }, InputEvent::KeyUp { key }),
                magi_core::input::Button::Mouse(button) => {
                    (InputEvent::ButtonDown { button }, InputEvent::ButtonUp { button })
                }
                magi_core::input::Button::Pad(_) => continue,
            };
            let _ = host.send(&down);
            std::thread::sleep(Duration::from_secs_f64(current.action.hold_ms.max(0.0) / 1000.0));
            let _ = host.send(&up);
        }
        if last_repaint.elapsed() >= Duration::from_millis(33) {
            last_repaint = Instant::now();
            repaint();
        }
    }
    repaint();
}

#[cfg(test)]
mod tests {
    use magi_core::magic::{ClassSpec, InputSpec};

    use super::*;

    #[test]
    fn a_model_comes_with_blocks_that_run_it() {
        let registry = Registry::builtin();
        let model = MagicModel {
            name: "Watcher".into(),
            classes: vec![ClassSpec { label: "go".into(), act: true, ..ClassSpec::default() }],
            input: InputSpec::default(),
            ..MagicModel::default()
        };
        let blocks = starter_blocks(&model, "mdl-1", &registry);
        let mut graph = magi_core::graph::Graph::new();
        let ids = graph.insert_fragment(&blocks, [0.0, 0.0]);
        assert_eq!(ids.len(), 3);
        let watch = graph.nodes().find(|n| n.kind == ids::AI_WATCH).unwrap();
        assert_eq!(watch.param::<String>("model").as_deref(), Some("mdl-1"));
        let watch = watch.id;
        assert_eq!(graph.prune(&registry), 0, "the wire fits");
        let start = graph.nodes().find(|n| n.kind == ids::START).unwrap().id;
        assert_eq!(graph.flow_target(start, ports::OUT).map(|t| t.node), Some(watch));
    }
}
