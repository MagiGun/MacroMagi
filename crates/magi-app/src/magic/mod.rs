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
use magi_core::magic::{Decision, MagicFile, MagicModel, ModelKind};
use magi_core::nodes::ai::AiWatchParams;
use magi_core::nodes::flow::StartParams;
use magi_core::nodes::notes::CommentParams;
use magi_core::nodes::{Registry, ids, ports, studio};
use magi_core::project::Project;

/// The model open in the Magic tab.
pub struct MagicDoc {
    pub file: MagicFile,
    pub path: Option<PathBuf>,
    /// What the importer found and guessed.
    pub notes: Vec<String>,
    /// The model's shapes, read when it was loaded.
    pub shape: Option<magi_ml::Shape>,
    /// The Magic tab's canvas, kept in `file.model.studio` too.
    pub studio: Project,
    /// Bumped when the model, weights or data files change, which the
    /// description alone does not show.
    revision: u64,
    /// The description and revision as last saved, to tell whether there are changes.
    saved: Option<(MagicModel, u64)>,
    /// The description and revision as last put into the macro, where it is kept too.
    kept: Option<(MagicModel, u64)>,
}

impl MagicDoc {
    pub fn new(mut file: MagicFile, path: Option<PathBuf>, notes: Vec<String>) -> Self {
        // The values the settings decide, as the tab keeps them, so a file
        // written by hand does not look changed as soon as it is open.
        file.model = file.model.configured(&serde_json::Map::new());
        let shape = (!file.onnx.is_empty()).then(|| magi_ml::inspect(&file.onnx).ok()).flatten();
        // A model from before the canvas gets one, built from its setup.
        let canvas =
            file.model.studio.take().unwrap_or_else(|| {
                studio::default_canvas(&file.model, !file.weights.is_empty(), !file.onnx.is_empty())
            });
        let studio = Project { graph: magi_core::graph::Graph::from(canvas), ..Project::new() };
        let mut doc = Self { file, path, notes, shape, studio, revision: 0, saved: None, kept: None };
        doc.sync_canvas();
        doc.saved = doc.path.as_ref().map(|_| (doc.file.model.clone(), 0));
        doc
    }

    /// Puts the model's setup on the canvas's inspector blocks, and the
    /// canvas into the model.
    pub fn sync_canvas(&mut self) {
        let model = &mut self.file.model;
        // A model not trained yet is what its Start from Zero block makes.
        if self.file.onnx.is_empty()
            && let Some(zero) = self.studio.graph.nodes().find(|n| n.kind == studio::kinds::ZERO && n.enabled)
        {
            let params: studio::ZeroParams = zero.params_or_default();
            model.kind = if params.kind == "vision" { ModelKind::Vision } else { ModelKind::Text };
        }
        if model.kind == ModelKind::Text && model.text.is_none() {
            model.text = Some(magi_core::magic::TextSpec::default());
        }
        // The datasets are the Text Dataset blocks' files.
        let mut datasets: Vec<String> = Vec::new();
        for node in self.studio.graph.nodes().filter(|n| n.kind == studio::kinds::TEXT_DATA) {
            let params: studio::TextDataParams = node.params_or_default();
            for file in params.positive.into_iter().chain(params.negative) {
                if self.file.data.contains_key(&file) && !datasets.contains(&file) {
                    datasets.push(file);
                }
            }
        }
        if model.datasets != datasets {
            model.datasets = datasets;
        }
        let mut nodes: Vec<&mut magi_core::graph::Node> = self.studio.graph.nodes_mut().collect();
        studio::refresh(&mut nodes, &self.file.model, self.file.onnx.len());
        let canvas: Fragment = self.studio.graph.clone().into();
        if self.file.model.studio.as_ref() != Some(&canvas) {
            self.file.model.studio = Some(canvas);
        }
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
        if let Some(text) = trained.text {
            model.text = Some(text);
        }
        if let Some(input) = trained.input {
            model.input = input;
        }
        self.shape = magi_ml::inspect(&self.file.onnx).ok();
    }

    /// Why the canvas cannot train the model now, if it cannot.
    pub fn training_problem(&self) -> Option<String> {
        let pipeline = match studio::pipeline(&self.studio.graph) {
            Ok(pipeline) => pipeline,
            Err(problem) => return Some(problem),
        };
        if !pipeline.trains() {
            return Some("Add a training stage after the model's start to train it.".into());
        }
        if pipeline.entry.kind == studio::kinds::BASE
            && pipeline.entry.params_or_default::<studio::BaseParams>().path.trim().is_empty()
            && self.file.weights.is_empty()
        {
            return Some("This model has no weights to train further: start from zero, or load a base model.".into());
        }
        let pictures = self.file.model.training.len() + self.file.model.finetune.len();
        let files = studio::PlanFiles {
            dataset: &|file: &str| self.file.data.contains_key(file).then(|| file.to_owned()),
            weights: None,
            pictures: (pictures > 0).then(String::new),
            out_dir: String::new(),
        };
        studio::plan(&pipeline, &self.studio.graph, &self.file.model, &files).err()
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
        self.file.onnx = onnx;
        self.shape = Some(shape);
        self.files_changed();
        Ok(())
    }
}

/// A new model, with a canvas ready to train it.
pub fn new_model(text: bool) -> MagicFile {
    let mut model =
        MagicModel { name: if text { "New text model" } else { "New vision model" }.into(), ..MagicModel::default() };
    if text {
        model.kind = ModelKind::Text;
        model.text = Some(magi_core::magic::TextSpec::default());
    } else {
        model.kind = ModelKind::Vision;
        model.input.width = 64;
        model.input.height = 64;
        model.classes = vec![
            magi_core::magic::ClassSpec { label: "Yes".into(), act: true, threshold: 0.8, ..Default::default() },
            magi_core::magic::ClassSpec { label: "No".into(), threshold: 0.5, ..Default::default() },
        ];
    }
    MagicFile { model, ..MagicFile::default() }
}

/// The blocks a model comes with: a Start block and an AI Watch block (an AI
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
