//! Training a model in the background: writing the plan and the files it
//! needs, running the trainer (`runtime/python/magi_train`) and reading what
//! it reports, line by line.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use magi_core::magic::{Architecture, InputSpec, Layout, MagicFile, ModelKind, TrainingRecord};
use magi_core::nodes::studio::{self, Pipeline, PlanFiles};
use serde_json::Value;

/// One line of the console.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Info,
    Stage,
    Warning,
    Error,
    Done,
}

/// What a finished run made.
pub struct Trained {
    pub kind: ModelKind,
    pub onnx: Vec<u8>,
    pub weights: Vec<u8>,
    pub record: TrainingRecord,
    pub architecture: Architecture,
    pub input: Option<InputSpec>,
    /// Tokens the trainer learned (from a corpus read where it ran), to keep with the model.
    pub tokenizer: Option<magi_core::magic::TokenizerSpec>,
}

#[derive(Default)]
pub struct Telemetry {
    pub lines: Vec<(Level, String)>,
    pub stage: String,
    /// Step and steps of the stage running.
    pub progress: Option<(u64, u64)>,
    /// Loss by step, over the whole run.
    pub losses: Vec<[f64; 2]>,
    /// The mean reward by step, while reinforcement learning.
    pub rewards: Vec<[f64; 2]>,
    /// How the run ended, once it has.
    pub outcome: Option<Result<(), String>>,
    /// What it made, until the Magic tab takes it.
    pub trained: Option<Trained>,
    steps_before: u64,
    stage_started: Option<Instant>,
    /// A failure was passed on (so the optional parts are checked once for it).
    pub reported: bool,
}

impl Telemetry {
    pub fn push(&mut self, level: Level, text: impl Into<String>) {
        self.line(level, text);
    }

    /// Seconds since the stage running began.
    pub fn stage_seconds(&self) -> f64 {
        self.stage_started.map_or(0.0, |t| t.elapsed().as_secs_f64())
    }

    fn line(&mut self, level: Level, text: impl Into<String>) {
        self.lines.push((level, text.into()));
        if self.lines.len() > 2000 {
            self.lines.drain(..500);
        }
    }
}

/// A training run.
pub struct TrainJob {
    pub state: Arc<Mutex<Telemetry>>,
    child: Arc<Mutex<Option<Child>>>,
    pub started: Instant,
}

/// What a run needs, taken from the open model.
pub struct Request {
    pub python: PathBuf,
    pub runtime: PathBuf,
    /// A folder of its own for this run.
    pub work: PathBuf,
    pub file: MagicFile,
    pub graph: magi_core::graph::Graph,
    pub pipeline: Pipeline,
    /// CUDA devices to train on; empty: the CPU.
    pub devices: Vec<u32>,
    /// Weights to start from instead of what the canvas's start says: a
    /// checkpoint, or the trained model.
    pub from: Option<PathBuf>,
    /// Where to save the weights after a stage (by its block), and what to note with them.
    pub checkpoints: std::collections::BTreeMap<magi_core::graph::NodeId, (PathBuf, Value)>,
}

impl TrainJob {
    pub fn start(request: Request, repaint: impl Fn() + Send + 'static) -> Self {
        let state = Arc::new(Mutex::new(Telemetry::default()));
        let child = Arc::new(Mutex::new(None));
        let (for_state, for_child) = (Arc::clone(&state), Arc::clone(&child));
        std::thread::Builder::new()
            .name("magi-train".into())
            .spawn(move || {
                let lock = || for_state.lock().unwrap_or_else(|p| p.into_inner());
                let result = run(&request, &for_state, &for_child, &repaint);
                let mut state = lock();
                match result {
                    Ok(trained) => {
                        state.line(Level::Done, "Training finished; the model is updated.");
                        state.trained = Some(trained);
                        state.outcome = Some(Ok(()));
                    }
                    Err(error) => {
                        state.line(Level::Error, error.clone());
                        state.outcome = Some(Err(error));
                    }
                }
                drop(state);
                repaint();
            })
            .expect("a thread can be started");
        Self { state, child, started: Instant::now() }
    }

    pub fn stop(&self) {
        if let Some(child) = self.child.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
            let _ = child.kill();
        }
    }

    pub fn is_running(&self) -> bool {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).outcome.is_none()
    }
}

impl Drop for TrainJob {
    fn drop(&mut self) {
        self.stop();
    }
}

fn run(
    request: &Request,
    state: &Mutex<Telemetry>,
    child_slot: &Mutex<Option<Child>>,
    repaint: &dyn Fn(),
) -> Result<Trained, String> {
    let lock = || state.lock().unwrap_or_else(|p| p.into_inner());
    let say = |level: Level, text: String| {
        lock().line(level, text);
        repaint();
    };
    let work = &request.work;
    let _ = std::fs::remove_dir_all(work);
    let data_dir = work.join("data");
    let out_dir = work.join("out");
    std::fs::create_dir_all(&data_dir).map_err(|e| format!("The training folder could not be made: {e}"))?;
    let model = &request.file.model;
    let kind = request.pipeline.kind(&request.file.model);

    // The datasets, as files.
    let mut written = std::collections::BTreeMap::new();
    for (index, (name, bytes)) in request.file.data.iter().enumerate() {
        if !name.starts_with("data/text/") {
            continue;
        }
        let file = name.rsplit('/').next().unwrap_or("data.txt");
        let path = data_dir.join(format!("{index}-{file}"));
        std::fs::write(&path, bytes).map_err(|e| format!("A dataset could not be written: {e}"))?;
        written.insert(name.clone(), path.to_string_lossy().into_owned());
    }

    // The weights to start from.
    let mut weights = None;
    if let Some(from) = &request.from {
        weights = Some(from.to_string_lossy().into_owned());
    } else if request.pipeline.entry.kind == studio::kinds::BASE {
        let params: studio::BaseParams = request.pipeline.entry.params_or_default();
        let bytes = if params.path.trim().is_empty() {
            request.file.weights.clone()
        } else {
            let base = magi_core::szi::open(Path::new(params.path.trim()))
                .map_err(|e| format!("The base model could not be opened: {e}"))?;
            base.weights
        };
        if bytes.is_empty() {
            return Err("The base model has no weights to train further. Only models trained in MacroMagi (with \
                        their weights included) can be fine-tuned; start from zero instead."
                .into());
        }
        let path = data_dir.join("base.safetensors");
        std::fs::write(&path, bytes).map_err(|e| format!("The base weights could not be written: {e}"))?;
        weights = Some(path.to_string_lossy().into_owned());
    }

    // The pictures, prepared as the model will see them (vision).
    let mut input = None;
    let mut pictures = None;
    if kind == ModelKind::Vision {
        let side = request.pipeline.architecture.context.clamp(8, 1024);
        let spec = InputSpec { width: side, height: side, layout: Layout::Nchw, softmax: true, ..model.input.clone() };
        say(Level::Info, "Preparing the pictures…".into());
        let path = data_dir.join("pictures.bin");
        let count = write_pictures(&request.file, &spec, &path)?;
        if count == 0 {
            return Err("There are no pictures to learn from: add some in the Vision Dataset block.".into());
        }
        say(Level::Info, format!("{count} pictures ready."));
        pictures = Some(path.to_string_lossy().into_owned());
        input = Some(spec);
    }

    let files = PlanFiles {
        dataset: &|name: &str| written.get(name).cloned(),
        weights,
        pictures,
        out_dir: out_dir.to_string_lossy().into_owned(),
        devices: request.devices.clone(),
    };
    let mut plan = studio::plan(&request.pipeline, &request.graph, model, &files)?;
    // Checkpoints: the weights after a stage, for a later training to start from.
    if let Some(stages) = plan.get_mut("stages").and_then(Value::as_array_mut) {
        for stage in stages {
            let node = stage.get("node").and_then(Value::as_u64);
            if let Some((path, meta)) = node.and_then(|n| request.checkpoints.get(&n)) {
                stage["checkpoint"] = serde_json::json!({ "path": path.to_string_lossy(), "meta": meta });
            }
        }
    }
    let plan_path = work.join("plan.json");
    std::fs::write(&plan_path, serde_json::to_vec_pretty(&plan).unwrap_or_default())
        .map_err(|e| format!("The plan could not be written: {e}"))?;

    // The trainer.
    let script = request.runtime.join("magi_train_run.py");
    let mut command = Command::new(&request.python);
    command
        .arg("-u")
        .arg(&script)
        .arg(&plan_path)
        .current_dir(work)
        .env("PYTHONIOENCODING", "utf-8")
        // Corpora, cut into tokens once, outlive the training folder (emptied for every run).
        .env("MAGI_CORPUS_DIR", work.parent().unwrap_or(work).join("corpora"))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    say(Level::Info, format!("Training with {}.", request.python.display()));
    let mut child = command.spawn().map_err(|e| format!("Python could not be started: {e}"))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    *child_slot.lock().unwrap_or_else(|p| p.into_inner()) = Some(child);

    // Errors Python prints (a missing package, a crash) go to the console too.
    let errors = Arc::new(Mutex::new(Vec::<String>::new()));
    let for_errors = Arc::clone(&errors);
    let reader = stderr.map(|stderr| {
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                for_errors.lock().unwrap_or_else(|p| p.into_inner()).push(line);
            }
        })
    });

    let mut done = None;
    let mut failed = None;
    if let Some(stdout) = stdout {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let Ok(event) = serde_json::from_str::<Value>(&line) else {
                say(Level::Info, line);
                continue;
            };
            let text = |key: &str| event.get(key).and_then(Value::as_str).unwrap_or_default().to_owned();
            let number = |key: &str| event.get(key).and_then(Value::as_f64);
            match event.get("event").and_then(Value::as_str).unwrap_or_default() {
                "stage" => {
                    let mut s = lock();
                    s.steps_before = s.losses.last().map_or(0, |l| l[0] as u64);
                    s.stage = text("stage");
                    s.progress = None;
                    s.stage_started = Some(Instant::now());
                    let name = stage_name(&s.stage).to_owned();
                    s.line(Level::Stage, format!("{name}…"));
                }
                "progress" => {
                    let mut s = lock();
                    let step = number("step").unwrap_or(0.0) as u64;
                    s.progress = Some((step, number("steps").unwrap_or(0.0) as u64));
                    let x = (s.steps_before + step) as f64;
                    if let Some(loss) = number("loss").filter(|l| l.is_finite()) {
                        s.losses.push([x, loss]);
                    }
                    if let Some(reward) = number("reward") {
                        s.rewards.push([x, reward]);
                    }
                }
                "log" => say(Level::Info, text("message")),
                "warning" => say(Level::Warning, text("message")),
                "error" => failed = Some(text("message")),
                "done" => done = Some(event.clone()),
                _ => {}
            }
            repaint();
        }
    }
    let status = child_slot.lock().unwrap_or_else(|p| p.into_inner()).as_mut().map(Child::wait);
    if let Some(reader) = reader {
        let _ = reader.join();
    }
    let errors = errors.lock().unwrap_or_else(|p| p.into_inner()).clone();
    if let Some(message) = failed {
        return Err(message);
    }
    let Some(report) = done else {
        let succeeded = matches!(status, Some(Ok(s)) if s.success());
        let tail: Vec<&String> = errors.iter().rev().take(12).collect::<Vec<_>>().into_iter().rev().collect();
        let detail = tail.iter().map(|l| l.as_str()).collect::<Vec<_>>().join("\n");
        if detail.contains("No module named 'torch'") {
            return Err("PyTorch is not installed in this Python. Install it under Settings › Runtime.".into());
        }
        return Err(if succeeded || detail.is_empty() {
            "Training stopped.".into()
        } else {
            format!("Training failed:\n{detail}")
        });
    };

    // Text models the native engine runs come out as weights alone.
    let onnx = std::fs::read(out_dir.join("model.onnx")).unwrap_or_default();
    let weights = std::fs::read(out_dir.join("weights.safetensors")).unwrap_or_default();
    if onnx.is_empty() && (weights.is_empty() || kind == ModelKind::Vision) {
        return Err("Training finished, but no model came out of it: the trainer could not export it. The lines \
                    above say why."
            .into());
    }
    let record = TrainingRecord {
        stages: report
            .get("stages")
            .and_then(Value::as_array)
            .map(|s| s.iter().filter_map(Value::as_str).map(str::to_owned).collect())
            .unwrap_or_default(),
        steps: report.get("steps").and_then(Value::as_u64).unwrap_or(0),
        loss: report.get("loss").and_then(Value::as_f64).unwrap_or(f64::NAN),
        parameters: report.get("parameters").and_then(Value::as_u64).unwrap_or(0),
        system: None,
        compacts: report.get("compacts").and_then(Value::as_bool).unwrap_or(false),
    };
    if let Some(seconds) = report.get("seconds").and_then(Value::as_f64) {
        say(Level::Info, format!("{} steps in {seconds:.0} s · {} parameters.", record.steps, record.parameters));
    }
    // Tokens the trainer learned, from a corpus it read.
    let tokenizer = std::fs::read(out_dir.join("tokenizer.json")).ok().and_then(|b| serde_json::from_slice(&b).ok());
    Ok(Trained { kind, onnx, weights, record, architecture: request.pipeline.architecture.clone(), input, tokenizer })
}

/// A stage's name, as people say it.
pub fn stage_name(stage: &str) -> &str {
    match stage {
        "pretrain" => "Pretraining",
        "sft" => "Fine-tuning",
        "rl" => "Reinforcement learning",
        "compaction" => "Context compaction",
        "vision" => "Vision training",
        other => other,
    }
}

/// Writes the model's pictures as the trainer reads them: a JSON header line,
/// then each picture as float32 `[3, h, w]`. Returns how many there are.
pub(crate) fn write_pictures(file: &MagicFile, spec: &InputSpec, path: &Path) -> Result<usize, String> {
    let mut items = Vec::new();
    let mut body: Vec<u8> = Vec::new();
    for item in file.model.training.iter().chain(&file.model.finetune) {
        let Some(bytes) = file.data.get(&item.file) else { continue };
        let Ok(bitmap) = magi_vision::decode(bytes) else { continue };
        let values = magi_ml::prepare(&bitmap.rgba, bitmap.width, bitmap.height, spec);
        for value in values {
            body.extend_from_slice(&value.to_le_bytes());
        }
        items.push(serde_json::json!({ "class": item.class, "negative": item.negative }));
    }
    let header = serde_json::json!({ "width": spec.width, "height": spec.height, "items": items });
    let mut out = std::fs::File::create(path).map_err(|e| format!("The pictures could not be written: {e}"))?;
    let mut line = serde_json::to_vec(&header).unwrap_or_default();
    line.push(b'\n');
    out.write_all(&line)
        .and_then(|()| out.write_all(&body))
        .map_err(|e| format!("The pictures could not be written: {e}"))?;
    Ok(items.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use magi_core::nodes::studio::kinds;

    /// The whole way, as the Magic tab does it: a new text model (Magi, made
    /// tiny on its example files), the trainer in Python, the trained model back
    /// in the document, the package compiled into a folder – and the model
    /// answering from the saved file.
    #[test]
    #[ignore = "needs Python with PyTorch and onnx (MAGI_TEST_PYTHON, or python on the PATH)"]
    fn a_new_text_model_trains_end_to_end() {
        let mut doc = crate::magic::MagicDoc::new(crate::magic::new_model(true), None, Vec::new());
        crate::magic::tiny(&mut doc);
        let folder = std::env::temp_dir().join(format!("magi-compile-test-{}", std::process::id()));
        let compile = doc.studio.graph.nodes().find(|n| n.kind == kinds::COMPILE).unwrap().id;
        doc.studio.graph.node_mut(compile).unwrap().set_param("folder", folder.to_string_lossy());
        doc.sync_canvas();
        assert_eq!(doc.training_problem(), None);
        let pipeline = studio::pipeline(&doc.studio.graph).unwrap();
        let work = std::env::temp_dir().join(format!("magi-train-test-{}", std::process::id()));
        let devices: Vec<u32> = std::env::var("MAGI_TEST_GPU").ok().and_then(|g| g.parse().ok()).into_iter().collect();
        let request = Request {
            python: std::env::var_os("MAGI_TEST_PYTHON").map_or_else(|| "python".into(), PathBuf::from),
            runtime: Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/python"),
            work: work.clone(),
            file: doc.file.clone(),
            graph: doc.studio.graph.clone(),
            pipeline,
            devices,
            from: None,
            checkpoints: Default::default(),
        };
        let job = TrainJob::start(request, || {});
        while job.is_running() {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let mut state = job.state.lock().unwrap();
        let lines: Vec<&String> = state.lines.iter().map(|(_, l)| l).collect();
        assert_eq!(state.outcome, Some(Ok(())), "{lines:#?}");
        assert!(!state.losses.is_empty());
        let trained = state.trained.take().unwrap();
        drop(state);
        doc.apply_trained(trained);
        assert_eq!(doc.file.model.trained.as_ref().unwrap().stages, ["pretrain", "sft", "compaction", "rl"]);

        let (written, skipped) = doc.compile().unwrap();
        assert!(skipped.is_empty(), "{skipped:?}");
        assert!(written.iter().any(|p| p.extension().is_some_and(|e| e == "szi")), "{written:?}");
        // The workspace holds no model: opening it finds the model in the .szi package.
        let saved = magi_core::szi::open(&written[0]).unwrap();
        // The native engine runs it from its weights: no ONNX model is made for it.
        assert!(!saved.weights.is_empty() && saved.onnx.is_empty());
        let spec = saved.model.text.clone().unwrap();
        let architecture = saved.model.architecture.clone().unwrap();
        let memory = saved.model.memory.clone().unwrap_or_default();
        let mut model =
            magi_ml::TextModel::load_native(&architecture, &saved.weights, &memory, &magi_ml::tools::cache_dir())
                .unwrap();
        // A tiny model says little that makes sense; it must run and answer.
        let answer = magi_ml::ask(&mut model, &spec, "List the files in C:/work", 40, 0.0, 1).unwrap();
        assert!(!answer.raw.is_empty(), "{answer:?}");
        let _ = std::fs::remove_dir_all(work);
        let _ = std::fs::remove_dir_all(folder);
    }
}
