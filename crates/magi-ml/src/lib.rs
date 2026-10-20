//! # magi-ml
//!
//! Runs the small vision models of the Magic tab on pictures of the screen.
//!
//! Two engines can run a model:
//!
//! * the **built-in** one ([`tract_onnx`]), pure Rust and always there – CPU only;
//! * **ONNX Runtime**, when it has been downloaded (see [`use_onnxruntime`]):
//!   usually faster on the CPU, and able to use a graphics card – any
//!   DirectX 12 card through DirectML on Windows, CoreML on macOS.
//!
//! [`Device::Auto`] tries every way available, measures each on the model,
//! and keeps the fastest; a graphics card that turns out slower than the CPU
//! (common for tiny models) is not used.
//!
//! Nothing here touches the screen: callers capture pixels and pass them in.

mod prepare;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, RwLock};
use std::time::Instant;

use magi_core::magic::{InputSpec, Layout};
use thiserror::Error;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::infer::Factoid;
use tract_onnx::tract_hir::internal::DimLike;

pub use prepare::{prepare, resize_rgb, softmax};

pub mod llama;
pub mod native;
pub mod tools;
pub use llama::{LlamaConfig, ask_gguf, set_llama};

#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[error("{0}")]
pub struct MlError(pub String);

impl MlError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

/// Where models run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Device {
    /// Whatever is fastest for the model.
    #[default]
    Auto,
    Cpu,
    /// A graphics card, by its number in the system's list.
    Gpu {
        index: u32,
    },
}

/// How a model ended up running.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Engine {
    BuiltIn,
    OnnxRuntimeCpu,
    /// ONNX Runtime on a graphics card.
    OnnxRuntimeGpu {
        index: u32,
    },
}

impl Engine {
    pub fn label(self) -> String {
        match self {
            Engine::BuiltIn => "CPU (built-in)".into(),
            Engine::OnnxRuntimeCpu => "CPU (ONNX Runtime)".into(),
            Engine::OnnxRuntimeGpu { index } => format!("Graphics card {} (ONNX Runtime)", index + 1),
        }
    }
}

// ---- ONNX Runtime ------------------------------------------------------------

struct Runtime {
    environment: ort::environment::Environment,
    /// The runtime can use graphics cards.
    gpu: bool,
    path: PathBuf,
}

static RUNTIME: OnceLock<Result<Runtime, String>> = OnceLock::new();
static DEVICE: RwLock<Device> = RwLock::new(Device::Auto);
/// The engine measured fastest for each model (by a hash of its bytes).
static FASTEST: Mutex<Option<HashMap<u64, Engine>>> = Mutex::new(None);

/// Loads ONNX Runtime from `library` (`onnxruntime.dll`, `libonnxruntime.so`
/// or `.dylib`). It can only be loaded once per run of the program: later
/// calls report how the first went. `gpu` says whether this build of the
/// runtime can use graphics cards.
pub fn use_onnxruntime(library: &Path, gpu: bool) -> Result<(), MlError> {
    let loaded = RUNTIME.get_or_init(|| {
        let builder = ort::init_from(library).map_err(|e| format!("could not load ONNX Runtime: {e}"))?;
        let environment = builder.with_name("macromagi").build().map_err(|e| e.to_string())?;
        Ok(Runtime { environment, gpu, path: library.to_owned() })
    });
    loaded.as_ref().map(drop).map_err(|e| MlError::new(e.clone()))
}

/// The ONNX Runtime in use, if one is loaded: its file, and whether it can use graphics cards.
pub fn onnxruntime() -> Option<(PathBuf, bool)> {
    RUNTIME.get()?.as_ref().ok().map(|r| (r.path.clone(), r.gpu))
}

/// Where models should run from now on. Models already loaded keep their choice.
pub fn set_device(device: Device) {
    *DEVICE.write().unwrap_or_else(|p| p.into_inner()) = device;
    *FASTEST.lock().unwrap_or_else(|p| p.into_inner()) = None;
}

pub fn device() -> Device {
    *DEVICE.read().unwrap_or_else(|p| p.into_inner())
}

fn hash(bytes: &[u8]) -> u64 {
    // FNV-1a: enough to tell models apart.
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3))
}

// ---- models ------------------------------------------------------------------

enum Backend {
    BuiltIn(std::sync::Arc<TypedRunnableModel>),
    OnnxRuntime { session: ort::session::Session, input: String },
}

/// The shapes a model reads and gives.
#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    pub input_name: String,
    /// Dimensions of the first input; `None` for a dimension that varies.
    pub input: Vec<Option<usize>>,
    /// Dimensions of the first output, where known.
    pub output: Vec<Option<usize>>,
}

impl Shape {
    /// The input spec this shape suggests: size and layout.
    pub fn suggested(&self, mut spec: InputSpec) -> InputSpec {
        spec.name = self.input_name.clone();
        if let [_, a, b, c] = self.input.as_slice() {
            match (a, b, c) {
                (Some(3), Some(h), Some(w)) => {
                    spec.layout = Layout::Nchw;
                    (spec.width, spec.height) = (*w as u32, *h as u32);
                }
                (Some(h), Some(w), Some(3)) => {
                    spec.layout = Layout::Nhwc;
                    (spec.width, spec.height) = (*w as u32, *h as u32);
                }
                _ => {}
            }
        }
        spec
    }

    /// How many answers (classes) the model gives.
    pub fn classes(&self) -> Option<usize> {
        self.output.last().copied().flatten()
    }
}

/// Reads a model's input and output shapes.
pub fn inspect(onnx: &[u8]) -> Result<Shape, MlError> {
    let model = tract_onnx::onnx()
        .model_for_read(&mut std::io::Cursor::new(onnx))
        .map_err(|e| MlError::new(format!("this is not an ONNX model MacroMagi can read: {e}")))?;
    let dims = |fact: &InferenceFact| -> Vec<Option<usize>> {
        fact.shape.dims().map(|d| d.concretize().and_then(|d| d.to_usize().ok())).collect()
    };
    let input = model.input_outlets().map_err(|e| MlError::new(e.to_string()))?[0];
    let input_fact = model.outlet_fact(input).map_err(|e| MlError::new(e.to_string()))?;
    let mut shape =
        Shape { input_name: model.node(input.node).name.clone(), input: dims(input_fact), output: Vec::new() };
    // The output shape is only known once the input's is.
    if let Ok(typed) = model.into_optimized()
        && let Some(outlet) = typed.output_outlets().ok().and_then(|o| o.first().copied())
        && let Ok(fact) = typed.outlet_fact(outlet)
    {
        shape.output = fact.shape.iter().map(|d| d.to_usize().ok()).collect();
    }
    Ok(shape)
}

/// A model ready to run.
pub struct Model {
    backend: Backend,
    spec: InputSpec,
    engine: Engine,
}

impl Model {
    /// Loads a model, choosing where it runs from the current [`Device`].
    pub fn load(onnx: &[u8], spec: &InputSpec) -> Result<Model, MlError> {
        let wanted = device();
        let candidates = candidates(wanted);
        if candidates.is_empty() {
            return Err(MlError::new("ONNX Runtime with graphics card support is not installed"));
        }
        let engine = if candidates.len() == 1 {
            candidates[0]
        } else {
            let key = hash(onnx);
            let known = FASTEST.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert_default().get(&key).copied();
            match known {
                Some(engine) => engine,
                None => {
                    let fastest = fastest(onnx, spec, &candidates);
                    FASTEST.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert_default().insert(key, fastest);
                    fastest
                }
            }
        };
        Self::load_on(onnx, spec, engine)
    }

    /// Loads a model on a particular engine.
    pub fn load_on(onnx: &[u8], spec: &InputSpec, engine: Engine) -> Result<Model, MlError> {
        let backend = match engine {
            Engine::BuiltIn => {
                let shape: [usize; 4] = match spec.layout {
                    Layout::Nchw => [1, 3, spec.height as usize, spec.width as usize],
                    Layout::Nhwc => [1, spec.height as usize, spec.width as usize, 3],
                };
                let model = tract_onnx::onnx()
                    .model_for_read(&mut std::io::Cursor::new(onnx))
                    .and_then(|m| m.with_input_fact(0, f32::fact(shape).into()))
                    .and_then(|m| m.into_optimized())
                    .and_then(|m| m.into_runnable())
                    .map_err(|e| MlError::new(format!("the built-in engine cannot run this model: {e}")))?;
                Backend::BuiltIn(model)
            }
            Engine::OnnxRuntimeCpu | Engine::OnnxRuntimeGpu { .. } => {
                let runtime = RUNTIME
                    .get()
                    .and_then(|r| r.as_ref().ok())
                    .ok_or_else(|| MlError::new("ONNX Runtime is not installed"))?;
                let error = |e: ort::Error| MlError::new(format!("ONNX Runtime cannot run this model: {e}"));
                let mut builder = ort::session::Session::builder(&runtime.environment).map_err(error)?;
                let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(4));
                builder = builder.with_intra_threads(threads).map_err(|e| error(e.into()))?;
                if let Engine::OnnxRuntimeGpu { index } = engine {
                    let provider = gpu_provider(index);
                    builder = builder.with_execution_providers([provider]).map_err(|e| error(e.into()))?;
                }
                let session = builder.commit_from_memory(onnx).map_err(error)?;
                let input = session.inputs().first().map(|i| i.name().to_owned()).unwrap_or_default();
                Backend::OnnxRuntime { session, input }
            }
        };
        Ok(Model { backend, spec: spec.clone(), engine })
    }

    pub fn engine(&self) -> Engine {
        self.engine
    }

    pub fn spec(&self) -> &InputSpec {
        &self.spec
    }

    /// Runs the model on prepared input (see [`prepare`]); returns its output.
    pub fn run(&mut self, input: Vec<f32>) -> Result<Vec<f32>, MlError> {
        let shape: Vec<usize> = match self.spec.layout {
            Layout::Nchw => vec![1, 3, self.spec.height as usize, self.spec.width as usize],
            Layout::Nhwc => vec![1, self.spec.height as usize, self.spec.width as usize, 3],
        };
        match &mut self.backend {
            Backend::BuiltIn(model) => {
                let tensor = Tensor::from_shape(&shape, &input).map_err(|e| MlError::new(e.to_string()))?;
                let outputs = model.run(tvec!(tensor.into())).map_err(|e| MlError::new(e.to_string()))?;
                let view = outputs[0].to_plain_array_view::<f32>().map_err(|e| MlError::new(e.to_string()))?;
                Ok(view.iter().copied().collect())
            }
            Backend::OnnxRuntime { session, input: name } => {
                let error = |e: ort::Error| MlError::new(e.to_string());
                let shape: Vec<i64> = shape.iter().map(|d| *d as i64).collect();
                let tensor = ort::value::Tensor::from_array((shape, input)).map_err(error)?;
                let outputs = session.run(ort::inputs![name.as_str() => tensor]).map_err(error)?;
                let (_, data) = outputs[0].try_extract_tensor::<f32>().map_err(error)?;
                Ok(data.to_vec())
            }
        }
    }

    /// The probabilities of each class for a picture (tightly packed RGBA).
    pub fn classify(&mut self, rgba: &[u8], width: u32, height: u32) -> Result<Vec<f32>, MlError> {
        let input = prepare(rgba, width, height, &self.spec);
        let output = self.run(input)?;
        Ok(if self.spec.softmax { softmax(&output) } else { output })
    }
}

#[cfg(windows)]
fn gpu_provider(index: u32) -> ort::ep::ExecutionProviderDispatch {
    ort::ep::DirectML::default().with_device_id(index as i32).build().error_on_failure()
}

#[cfg(target_os = "macos")]
fn gpu_provider(_index: u32) -> ort::ep::ExecutionProviderDispatch {
    ort::ep::CoreML::default().build().error_on_failure()
}

#[cfg(all(not(windows), not(target_os = "macos")))]
fn gpu_provider(_index: u32) -> ort::ep::ExecutionProviderDispatch {
    ort::ep::CPU::default().build()
}

/// Every engine the device setting allows, given what is installed.
fn candidates(device: Device) -> Vec<Engine> {
    let runtime = RUNTIME.get().and_then(|r| r.as_ref().ok());
    let gpu = runtime.is_some_and(|r| r.gpu);
    match device {
        Device::Cpu => {
            let mut list = vec![Engine::BuiltIn];
            if runtime.is_some() {
                list.push(Engine::OnnxRuntimeCpu);
            }
            list
        }
        Device::Gpu { index } => {
            if gpu {
                vec![Engine::OnnxRuntimeGpu { index }]
            } else {
                Vec::new()
            }
        }
        Device::Auto => {
            let mut list = vec![Engine::BuiltIn];
            if runtime.is_some() {
                list.push(Engine::OnnxRuntimeCpu);
            }
            if gpu {
                list.push(Engine::OnnxRuntimeGpu { index: 0 });
            }
            list
        }
    }
}

/// Times each engine on the model and returns the fastest that works.
fn fastest(onnx: &[u8], spec: &InputSpec, candidates: &[Engine]) -> Engine {
    let input = vec![0.0f32; (spec.width * spec.height * 3) as usize];
    let mut best = (f64::INFINITY, Engine::BuiltIn);
    for &engine in candidates {
        let Ok(mut model) = Model::load_on(onnx, spec, engine) else { continue };
        // A few runs to warm up, then the median of a few more.
        if (0..3).any(|_| model.run(input.clone()).is_err()) {
            continue;
        }
        let mut times: Vec<f64> = (0..9)
            .map(|_| {
                let start = Instant::now();
                let _ = model.run(input.clone());
                start.elapsed().as_secs_f64()
            })
            .collect();
        times.sort_by(f64::total_cmp);
        if times[times.len() / 2] < best.0 {
            best = (times[times.len() / 2], engine);
        }
    }
    best.1
}

// ---- text models ---------------------------------------------------------------

enum TextBackend {
    BuiltIn(std::sync::Arc<TypedRunnableModel>),
    OnnxRuntime(ort::session::Session),
    /// MacroMagi's own engine, with a key/value cache ([`native`]).
    Native(Box<native::NativeModel>),
}

/// Draws numbers for sampling: xorshift64*, enough for picking tokens.
pub(crate) struct Random(u64);

impl Random {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub(crate) fn next(&mut self) -> f64 {
        let state = &mut self.0;
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// How a text model writes.
#[derive(Clone, Debug, PartialEq)]
pub struct Generation {
    pub max_tokens: usize,
    /// 0 always takes the likeliest token.
    pub temperature: f32,
    /// Only the `top_k` likeliest tokens are drawn from (0: all).
    pub top_k: usize,
    /// Tokens that end the answer (not included in it).
    pub stop: Vec<u32>,
    pub seed: u64,
}

/// A text model ready to run: token ids in (`tokens`, [1, length]), a score
/// for every possible next token out (`logits`, [1, length, vocabulary]).
pub struct TextModel {
    backend: TextBackend,
    context: usize,
}

impl TextModel {
    /// Loads a text model on ONNX Runtime when it is installed, else on the built-in engine.
    pub fn load(onnx: &[u8], context: usize) -> Result<TextModel, MlError> {
        let backend = match RUNTIME.get().and_then(|r| r.as_ref().ok()) {
            Some(runtime) => {
                let error = |e: ort::Error| MlError::new(format!("ONNX Runtime cannot run this model: {e}"));
                let mut builder = ort::session::Session::builder(&runtime.environment).map_err(error)?;
                TextBackend::OnnxRuntime(builder.commit_from_memory(onnx).map_err(error)?)
            }
            None => {
                let mut model = tract_onnx::onnx()
                    .model_for_read(&mut std::io::Cursor::new(onnx))
                    .map_err(|e| MlError::new(format!("this is not a text model MacroMagi can read: {e}")))?;
                // Always the full context: see `next_scores`.
                let fact = i64::fact([1, context.max(1)]);
                model.set_input_fact(0, fact.into()).map_err(|e| MlError::new(e.to_string()))?;
                let runnable = model
                    .into_optimized()
                    .and_then(|m| m.into_runnable())
                    .map_err(|e| MlError::new(format!("the built-in engine cannot run this model: {e}")))?;
                TextBackend::BuiltIn(runnable)
            }
        };
        Ok(TextModel { backend, context: context.max(1) })
    }

    /// Loads a model trained in MacroMagi on the native engine, from its
    /// weights: a key/value cache, prefix reuse, context scaling and tiered
    /// memory. `data_dir` holds the cache's files unless `memory` says otherwise.
    pub fn load_native(
        architecture: &magi_core::magic::Architecture,
        weights: &[u8],
        memory: &magi_core::magic::MemorySpec,
        data_dir: &Path,
    ) -> Result<TextModel, MlError> {
        let model = native::NativeModel::load(architecture, weights, memory, data_dir)?;
        let context = model.context;
        Ok(TextModel { backend: TextBackend::Native(Box::new(model)), context })
    }

    /// Whether it runs on the native engine (with its cache).
    pub fn is_native(&self) -> bool {
        matches!(self.backend, TextBackend::Native(_))
    }

    /// What the native engine's cache holds.
    pub fn telemetry(&self) -> Option<native::KvTelemetry> {
        match &self.backend {
            TextBackend::Native(model) => Some(model.telemetry()),
            _ => None,
        }
    }

    /// The scores for the token after `tokens` (the last `context` of them).
    ///
    /// The model always reads its full context: the tokens, then padding.
    /// It reads causally, so padding after a token never changes the scores
    /// at it – and one fixed size runs fastest.
    pub fn next_scores(&mut self, tokens: &[u32]) -> Result<Vec<f32>, MlError> {
        if let TextBackend::Native(model) = &mut self.backend {
            return model.scores(&tokens[tokens.len().saturating_sub(self.context)..], 0);
        }
        let window = &tokens[tokens.len().saturating_sub(self.context)..];
        if window.is_empty() {
            return Err(MlError::new("there is nothing to continue"));
        }
        let last = window.len() - 1;
        let mut ids: Vec<i64> = window.iter().map(|&t| i64::from(t)).collect();
        ids.resize(self.context, 0);
        let all: Vec<f32> = match &mut self.backend {
            TextBackend::BuiltIn(model) => {
                let tensor = Tensor::from_shape(&[1, ids.len()], &ids).map_err(|e| MlError::new(e.to_string()))?;
                let outputs = model.run(tvec!(tensor.into())).map_err(|e| MlError::new(e.to_string()))?;
                outputs[0]
                    .to_plain_array_view::<f32>()
                    .map_err(|e| MlError::new(e.to_string()))?
                    .iter()
                    .copied()
                    .collect()
            }
            TextBackend::Native(_) => unreachable!("answered above"),
            TextBackend::OnnxRuntime(session) => {
                let error = |e: ort::Error| MlError::new(e.to_string());
                let tensor =
                    ort::value::Tensor::from_array((vec![1i64, ids.len() as i64], ids.clone())).map_err(error)?;
                let outputs = session.run(ort::inputs!["tokens" => tensor]).map_err(error)?;
                outputs[0].try_extract_tensor::<f32>().map_err(error)?.1.to_vec()
            }
        };
        let vocab = all.len() / self.context;
        Ok(all[last * vocab..(last + 1) * vocab].to_vec())
    }

    /// Writes an answer after `prompt`. Returns its tokens, without the stop token.
    pub fn generate(&mut self, prompt: &[u32], how: &Generation) -> Result<Vec<u32>, MlError> {
        self.generate_after(prompt, how, 0)
    }

    /// Generates, telling the native engine that the first `prefix` tokens
    /// (the system text) may be kept on disk.
    pub fn generate_after(&mut self, prompt: &[u32], how: &Generation, prefix: usize) -> Result<Vec<u32>, MlError> {
        if let TextBackend::Native(model) = &mut self.backend {
            return model.generate(prompt, how, prefix);
        }
        let mut tokens = prompt.to_vec();
        let mut answer = Vec::new();
        let mut random = Random::new(how.seed);
        for _ in 0..how.max_tokens {
            let scores = self.next_scores(&tokens)?;
            let next = pick(&scores, how.temperature, how.top_k, random.next());
            if how.stop.contains(&next) {
                break;
            }
            tokens.push(next);
            answer.push(next);
        }
        Ok(answer)
    }
}

/// What a text model said to a request.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Answer {
    /// Everything it wrote, tags and all.
    pub raw: String,
    /// What it says, without its thinking and tool calls.
    pub text: String,
    /// What it thought (inside the thinking tags).
    pub thinking: String,
    /// The tool calls it made, in order.
    pub tools: Vec<magi_core::magic::ToolCall>,
}

/// How to answer one request. Zero `max_tokens`, a negative `temperature`
/// and zero `context` use the model's own.
#[derive(Clone, Debug, PartialEq)]
pub struct AskOptions {
    pub max_tokens: usize,
    pub temperature: f32,
    pub seed: u64,
    /// The context window, in tokens (GGUF models; ONNX models have theirs built in).
    pub context: usize,
    /// Think before answering; `None` follows the model.
    pub thinking: Option<bool>,
    /// The reasoning effort, one of the model's levels; empty follows the model.
    pub effort: String,
    /// llama.cpp: the key/value cache's type (`f16`, `q8_0`, `q4_0`); empty: its own.
    pub kv_cache: String,
    /// llama.cpp: flash attention (a quantized cache needs it).
    pub flash_attention: bool,
    /// llama.cpp: the model's context scaling, passed when a longer context than it trained at is asked for.
    pub scaling: Option<magi_core::magic::RopeScaling>,
}

impl Default for AskOptions {
    fn default() -> Self {
        Self {
            max_tokens: 0,
            temperature: -1.0,
            seed: 0,
            context: 0,
            thinking: None,
            effort: String::new(),
            kv_cache: String::new(),
            flash_attention: false,
            scaling: None,
        }
    }
}

impl AskOptions {
    /// The context scaling of a model trained in MacroMagi.
    pub fn with_scaling(mut self, architecture: Option<&magi_core::magic::Architecture>) -> Self {
        self.scaling = architecture.and_then(|a| a.rope_scaling.clone());
        self
    }

    /// The cache settings of a model's Memory & Cache block.
    pub fn with_memory(mut self, memory: Option<&magi_core::magic::MemorySpec>) -> Self {
        if let Some(memory) = memory {
            self.kv_cache = if memory.precision == "f32" { "f32".into() } else { memory.precision.clone() };
            self.flash_attention = memory.flash_attention;
        }
        self
    }
}

/// Asks a text model something, with the options of a request.
pub fn ask_with(
    model: &mut TextModel,
    spec: &magi_core::magic::TextSpec,
    request: &str,
    options: &AskOptions,
) -> Result<Answer, MlError> {
    let raw = complete(model, spec, &spec.prompt_with(request, options.thinking, &options.effort), options)?;
    Ok(Answer { text: spec.visible(&raw), thinking: spec.thinking(&raw), tools: spec.tool_calls(&raw), raw })
}

/// How long the system part of a prompt is, in tokens: the system prompt,
/// effort and tools, the same request after request.
pub fn system_tokens(spec: &magi_core::magic::TextSpec, text: &str) -> usize {
    text.find(&spec.end_tag)
        .filter(|_| text.starts_with(&spec.system_tag))
        .map_or(0, |end| spec.encode(&text[..end + spec.end_tag.len()]).len())
}

/// What a model writes after a prompt (already in its chat template), as raw text.
pub fn complete(
    model: &mut TextModel,
    spec: &magi_core::magic::TextSpec,
    text: &str,
    options: &AskOptions,
) -> Result<String, MlError> {
    let mut prompt = spec.encode(text);
    // A shorter context window than the model's: only the latest part of the text.
    if options.context > 0 && prompt.len() > options.context {
        prompt.drain(..prompt.len() - options.context);
    }
    let prefix = system_tokens(spec, text).min(prompt.len());
    let how = Generation {
        max_tokens: if options.max_tokens == 0 { spec.max_tokens as usize } else { options.max_tokens },
        temperature: if options.temperature < 0.0 { spec.temperature } else { options.temperature },
        top_k: spec.top_k as usize,
        stop: spec.special_id(&spec.end_tag).into_iter().collect(),
        seed: options.seed,
    };
    Ok(spec.decode(&model.generate_after(&prompt, &how, prefix)?))
}

/// Asks a text model something, the way its spec says to: the chat template
/// around the request, its own way of writing, stopping at the end of its turn.
/// `max_tokens` 0 and `temperature` below 0 use the spec's.
pub fn ask(
    model: &mut TextModel,
    spec: &magi_core::magic::TextSpec,
    request: &str,
    max_tokens: usize,
    temperature: f32,
    seed: u64,
) -> Result<Answer, MlError> {
    ask_with(model, spec, request, &AskOptions { max_tokens, temperature, seed, ..Default::default() })
}

/// The next token: the likeliest (temperature 0), or one drawn from the `top_k` likeliest.
fn pick(scores: &[f32], temperature: f32, top_k: usize, draw: f64) -> u32 {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]));
    if temperature <= 0.0 || order.len() == 1 {
        return order[0] as u32;
    }
    let keep = if top_k == 0 { order.len() } else { top_k.min(order.len()) };
    let best = scores[order[0]];
    let weights: Vec<f64> = order[..keep].iter().map(|&i| f64::from((scores[i] - best) / temperature).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut target = draw * total;
    for (index, weight) in order[..keep].iter().zip(&weights) {
        if target < *weight {
            return *index as u32;
        }
        target -= weight;
    }
    order[keep - 1] as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trained_text_model_answers_on_the_built_in_engine() {
        // Trained in MacroMagi's trainer to answer "hi" with "yo", before tool turns existed:
        // its tags are the older ones, and it has no tool tag (models of then keep working).
        let onnx = include_bytes!("../tests/data/tiny_text.onnx");
        let mut spec = magi_core::magic::TextSpec {
            context: 32,
            specials: Vec::new(),
            tool_tags: ("<tool>".into(), "</tool>".into()),
            tool_tag: String::new(),
            ..magi_core::magic::TextSpec::default()
        };
        spec.ensure_specials();
        let mut model = TextModel::load(onnx, 32).unwrap();
        let prompt = spec.encode(&spec.prompt("hi"));
        // Different lengths go through the same model.
        assert_eq!(model.next_scores(&prompt).unwrap().len(), spec.vocab_size());
        assert_eq!(model.next_scores(&prompt[..3]).unwrap().len(), spec.vocab_size());
        let how = Generation {
            max_tokens: 10,
            temperature: 0.0,
            top_k: 0,
            stop: vec![spec.special_id(&spec.end_tag).unwrap()],
            seed: 1,
        };
        let answer = model.generate(&prompt, &how).unwrap();
        assert_eq!(spec.decode(&answer), "yo");
        assert_eq!(pick(&[0.1, 3.0, 0.2], 0.0, 0, 0.5), 1);
    }

    /// A tiny ONNX model, built by hand: the average of each color channel of
    /// a 4×4 picture – three "classes" that are red, green and blue.
    pub fn channel_means() -> Vec<u8> {
        use prost::Message;
        use tract_onnx::pb::*;
        let dim = |v: i64| tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(v)),
            denotation: String::new(),
        };
        let tensor_type = |dims: &[i64]| TypeProto {
            denotation: String::new(),
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: 1,
                shape: Some(TensorShapeProto { dim: dims.iter().map(|d| dim(*d)).collect() }),
            })),
        };
        let graph = GraphProto {
            name: "means".into(),
            node: vec![
                NodeProto {
                    input: vec!["x".into()],
                    output: vec!["pooled".into()],
                    name: "pool".into(),
                    op_type: "GlobalAveragePool".into(),
                    ..NodeProto::default()
                },
                NodeProto {
                    input: vec!["pooled".into()],
                    output: vec!["y".into()],
                    name: "flatten".into(),
                    op_type: "Flatten".into(),
                    ..NodeProto::default()
                },
            ],
            input: vec![ValueInfoProto {
                name: "x".into(),
                r#type: Some(tensor_type(&[1, 3, 4, 4])),
                ..ValueInfoProto::default()
            }],
            output: vec![ValueInfoProto {
                name: "y".into(),
                r#type: Some(tensor_type(&[1, 3])),
                ..ValueInfoProto::default()
            }],
            ..GraphProto::default()
        };
        let model = ModelProto {
            ir_version: 7,
            opset_import: vec![OperatorSetIdProto { domain: String::new(), version: 13 }],
            graph: Some(graph),
            ..ModelProto::default()
        };
        model.encode_to_vec()
    }

    #[test]
    fn a_model_is_inspected_and_run_on_a_picture() {
        let onnx = channel_means();
        let shape = inspect(&onnx).unwrap();
        assert_eq!(shape.input, [Some(1), Some(3), Some(4), Some(4)]);
        assert_eq!(shape.classes(), Some(3));
        let spec = shape.suggested(InputSpec { mean: [0.0; 3], std: [1.0; 3], ..InputSpec::default() });
        assert_eq!((spec.width, spec.height, spec.name.as_str()), (4, 4, "x"));

        let mut model = Model::load_on(&onnx, &spec, Engine::BuiltIn).unwrap();
        // A 20×20 picture, mostly green.
        let rgba: Vec<u8> = (0..400).flat_map(|_| [30u8, 220, 40, 255]).collect();
        let probabilities = model.classify(&rgba, 20, 20).unwrap();
        let best = probabilities.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert_eq!(best, 1, "{probabilities:?}");
        assert!(inspect(b"nonsense").is_err());
    }

    /// The engine tests use a copy of the tiny model. If this fails after
    /// changing `channel_means`, write its bytes to that file again.
    #[test]
    fn the_engine_tests_have_the_tiny_model() {
        let copy = include_bytes!("../../magi-engine/tests/data/channel_means.onnx");
        assert_eq!(copy.as_slice(), channel_means().as_slice());
    }

    #[test]
    fn without_onnx_runtime_everything_runs_on_the_built_in_engine() {
        assert_eq!(candidates(Device::Cpu), [Engine::BuiltIn]);
        assert_eq!(candidates(Device::Auto), [Engine::BuiltIn]);
        assert!(candidates(Device::Gpu { index: 0 }).is_empty());
    }
}

#[cfg(test)]
mod exported_models {
    use super::*;

    /// With MAGI_TEST_ONNX (a text model's ONNX) and MAGI_TEST_CONTEXT: the
    /// built-in engine loads it and gives a score for every token.
    #[test]
    #[ignore = "needs an ONNX text model made by the trainer"]
    fn the_built_in_engine_runs_a_trained_text_model() {
        let path = std::env::var("MAGI_TEST_ONNX").expect("MAGI_TEST_ONNX");
        let context: usize = std::env::var("MAGI_TEST_CONTEXT").ok().and_then(|c| c.parse().ok()).unwrap_or(256);
        let onnx = std::fs::read(path).unwrap();
        let spec = magi_core::magic::TextSpec { context: context as u32, ..Default::default() };
        let mut model = TextModel::load(&onnx, context).unwrap();
        let prompt = spec.encode(&spec.prompt("Press Enter"));
        let scores = model.next_scores(&prompt).unwrap();
        assert_eq!(scores.len(), spec.vocab_size());
        assert!(scores.iter().all(|s| s.is_finite()));
        let started = std::time::Instant::now();
        let how = Generation { max_tokens: 20, temperature: 0.0, top_k: 0, stop: vec![256], seed: 1 };
        let answer = model.generate(&prompt, &how).unwrap();
        println!("{} tokens in {:?}: {:?}", answer.len(), started.elapsed(), spec.decode(&answer));
    }
}
