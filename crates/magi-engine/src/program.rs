//! A frozen copy of a macro, ready to run.
//!
//! The editor keeps changing its project; a run must not see half an edit. A
//! [`Program`] is an immutable snapshot of the graph and its assets that any
//! number of runs can share.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use magi_audio::Fingerprint;
use magi_core::assets::AssetStore;
use magi_core::graph::{Graph, Node, NodeId};
use magi_core::magic::{MagicFile, MagicModel, ModelKind};
use magi_core::nodes::flow::DefineParams;
use magi_core::nodes::{EngineSupport, NodePorts, Registry, ids};
use magi_core::project::Project;
use magi_core::variables::{self, Variable};
use magi_vision::Bitmap;
use serde_json::{Map, Value};

use crate::EngineError;

pub struct Program {
    /// The graph as it runs: blocks in disabled groups are disabled.
    pub graph: Graph,
    pub assets: AssetStore,
    pub registry: Arc<Registry>,
    /// The variables and their starting values.
    pub variables: Vec<Variable>,
    /// Decoded template images by id and shrink factor, filled in on first use.
    images: Mutex<HashMap<(String, u32), Arc<Bitmap>>>,
    /// AI models, loaded on first use.
    models: Mutex<HashMap<String, Arc<LoadedModel>>>,
    /// Text models, loaded on first use.
    text_models: Mutex<HashMap<String, Arc<LoadedText>>>,
    /// Fingerprints of the sounds listened for, by id and how much of the
    /// sound is used (whole milliseconds; 0 for all of it), filled in on first use.
    sounds: Mutex<HashMap<(String, u64), Arc<Fingerprint>>>,
}

impl Program {
    pub fn new(graph: Graph, assets: AssetStore, registry: Arc<Registry>) -> Self {
        Self {
            graph: graph.with_groups_applied(),
            assets,
            registry,
            variables: Vec::new(),
            images: Mutex::new(HashMap::new()),
            sounds: Mutex::new(HashMap::new()),
            models: Mutex::new(HashMap::new()),
            text_models: Mutex::new(HashMap::new()),
        }
    }

    pub fn from_project(project: &Project, registry: Arc<Registry>) -> Self {
        let mut program = Self::new(project.graph.clone(), project.assets.clone(), registry);
        program.variables = project.variables.clone();
        program
    }

    pub fn node(&self, id: NodeId) -> Result<&Node, EngineError> {
        self.graph.node(id).ok_or_else(|| EngineError::new(format!("block #{id} no longer exists")))
    }

    /// A block's settings as it should run them: defaults filled in, and
    /// settings bound to variables replaced by the variables' current values.
    pub fn resolved_params(&self, node: &Node, lookup: impl Fn(&str) -> Option<Value>) -> Map<String, Value> {
        let defaults = self.registry.get(&node.kind).map(|def| def.default_params()).unwrap_or_default();
        let mut params = defaults.clone();
        params.extend(node.params.clone());
        if variables::has_bindings(&params) { variables::resolve(&params, &defaults, lookup) } else { params }
    }

    /// The Definition block with this name.
    pub fn definition(&self, name: &str) -> Option<&Node> {
        self.graph
            .nodes()
            .filter(|node| node.kind == ids::DEFINE)
            .find(|node| node.params_or_default::<DefineParams>().name == name)
    }

    /// The decoded pixels of an image asset.
    pub fn image(&self, id: &str) -> Result<Arc<Bitmap>, EngineError> {
        self.image_shrunk(id, 1)
    }

    /// The decoded pixels of an image asset, shrunk by `factor`.
    pub fn image_shrunk(&self, id: &str, factor: u32) -> Result<Arc<Bitmap>, EngineError> {
        if id.is_empty() {
            return Err(EngineError::new("no image has been chosen"));
        }
        let key = (id.to_owned(), factor.max(1));
        if let Some(bitmap) = self.images.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
            return Ok(Arc::clone(bitmap));
        }
        let bitmap = if factor <= 1 {
            let bytes = self.assets.image(id).ok_or_else(|| EngineError::new(format!("image `{id}` is missing")))?;
            Arc::new(magi_vision::decode(bytes).map_err(|e| EngineError::new(e.to_string()))?)
        } else {
            let full = self.image_shrunk(id, 1)?;
            Arc::new(magi_vision::shrink(full.as_ref().as_ref(), factor))
        };
        self.images.lock().unwrap_or_else(|p| p.into_inner()).insert(key, Arc::clone(&bitmap));
        Ok(bitmap)
    }

    /// The fingerprint of a sound asset.
    pub fn sound(&self, id: &str) -> Result<Arc<Fingerprint>, EngineError> {
        self.sound_start(id, 0.0)
    }

    /// The fingerprint of the first `ms` milliseconds of a sound asset (all of
    /// it for zero).
    pub fn sound_start(&self, id: &str, ms: f64) -> Result<Arc<Fingerprint>, EngineError> {
        if id.is_empty() {
            return Err(EngineError::new("no sound has been chosen"));
        }
        let key = (id.to_owned(), ms.max(0.0).round() as u64);
        if let Some(fingerprint) = self.sounds.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
            return Ok(Arc::clone(fingerprint));
        }
        let fingerprint = if key.1 == 0 {
            let bytes = self.assets.audio(id).ok_or_else(|| EngineError::new(format!("sound `{id}` is missing")))?;
            let audio = magi_audio::decode(bytes).map_err(|e| EngineError::new(e.to_string()))?;
            let fingerprint = Fingerprint::of(&audio);
            if fingerprint.is_empty() {
                return Err(EngineError::new("the sound to listen for is too short"));
            }
            Arc::new(fingerprint)
        } else {
            Arc::new(self.sound_start(id, 0.0)?.prefix(ms))
        };
        self.sounds.lock().unwrap_or_else(|p| p.into_inner()).insert(key, Arc::clone(&fingerprint));
        Ok(fingerprint)
    }

    /// An AI model asset, ready to run.
    pub fn model(&self, id: &str) -> Result<Arc<LoadedModel>, EngineError> {
        if id.is_empty() {
            return Err(EngineError::new("no AI model has been chosen"));
        }
        if let Some(model) = self.models.lock().unwrap_or_else(|p| p.into_inner()).get(id) {
            return Ok(Arc::clone(model));
        }
        let bytes = self.assets.model(id).ok_or_else(|| EngineError::new(format!("model `{id}` is missing")))?;
        let file = MagicFile::from_bytes(bytes).map_err(|e| EngineError::new(e.to_string()))?;
        if file.model.kind == ModelKind::Text {
            return Err(EngineError::new("this is a text model: run it with an AI Agent block"));
        }
        if file.onnx.is_empty() {
            return Err(EngineError::new("this .magic file does not include the model itself"));
        }
        let model = magi_ml::Model::load(&file.onnx, &file.model.input).map_err(|e| EngineError::new(e.0))?;
        let loaded = Arc::new(LoadedModel { spec: file.model, model: Mutex::new(model) });
        self.models.lock().unwrap_or_else(|p| p.into_inner()).insert(id.to_owned(), Arc::clone(&loaded));
        Ok(loaded)
    }

    /// A text model, loaded on first use.
    pub fn text_model(&self, id: &str) -> Result<Arc<LoadedText>, EngineError> {
        if id.is_empty() {
            return Err(EngineError::new("no AI model has been chosen"));
        }
        if let Some(model) = self.text_models.lock().unwrap_or_else(|p| p.into_inner()).get(id) {
            return Ok(Arc::clone(model));
        }
        let bytes = self.assets.model(id).ok_or_else(|| EngineError::new(format!("model `{id}` is missing")))?;
        let file = MagicFile::from_bytes(bytes).map_err(|e| EngineError::new(e.to_string()))?;
        let Some(spec) = file.model.text.clone().filter(|_| file.model.kind == ModelKind::Text) else {
            return Err(EngineError::new("this is a vision model: run it with an AI Watch or AI Model block"));
        };
        let native =
            file.model.architecture.as_ref().filter(|a| a.native_problem().is_none() && !file.weights.is_empty());
        let runner = if let Some(path) = &file.model.gguf {
            TextRunner::Gguf { path: path.into(), own: file.model.architecture.is_some() }
        } else if let Some(architecture) = native {
            // MacroMagi's own engine, with its cache: fast.
            let memory = file.model.memory.clone().unwrap_or_default();
            let model =
                magi_ml::TextModel::load_native(architecture, &file.weights, &memory, &magi_ml::tools::cache_dir())
                    .map_err(|e| EngineError::new(e.0))?;
            TextRunner::Onnx(Mutex::new(model))
        } else {
            if file.onnx.is_empty() {
                return Err(EngineError::new("this .magic file does not include the model itself; train it first"));
            }
            let model =
                magi_ml::TextModel::load(&file.onnx, spec.context as usize).map_err(|e| EngineError::new(e.0))?;
            TextRunner::Onnx(Mutex::new(model))
        };
        let scaling = file.model.architecture.as_ref().and_then(|a| a.rope_scaling.clone());
        let loaded = Arc::new(LoadedText { spec, runner, data: file.data, memory: file.model.memory, scaling });
        self.text_models.lock().unwrap_or_else(|p| p.into_inner()).insert(id.to_owned(), Arc::clone(&loaded));
        Ok(loaded)
    }

    /// Unloads a text model: it loads again on its next request.
    pub fn unload_text_model(&self, id: &str) {
        let removed = self.text_models.lock().unwrap_or_else(|p| p.into_inner()).remove(id);
        if let Some(loaded) = removed
            && let TextRunner::Gguf { path, .. } = &loaded.runner
        {
            magi_ml::llama::stop_server(path);
        }
    }

    /// Unloads every AI model the macro loaded – text models (stopping their
    /// llama.cpp servers) and vision models – as when it stops. Returns how
    /// many text models were loaded.
    pub fn unload_models(&self) -> usize {
        self.models.lock().unwrap_or_else(|p| p.into_inner()).clear();
        let removed: Vec<Arc<LoadedText>> =
            std::mem::take(&mut *self.text_models.lock().unwrap_or_else(|p| p.into_inner())).into_values().collect();
        for loaded in &removed {
            if let TextRunner::Gguf { path, .. } = &loaded.runner {
                magi_ml::llama::stop_server(path);
            }
        }
        removed.len()
    }

    /// Whether any enabled node can only be run by the Python engine.
    pub fn needs_python(&self) -> bool {
        self.graph
            .nodes()
            .filter(|node| node.enabled)
            .any(|node| self.registry.get(&node.kind).is_some_and(|def| def.engines == EngineSupport::PythonOnly))
    }

    /// The program as the Python runtime receives it.
    ///
    /// Every node carries its complete parameters (defaults filled in) and the
    /// output a disabled node passes execution to, so the runtime needs no
    /// copy of the node definitions. Nodes with settings bound to variables
    /// are marked, so the runtime asks for them to be resolved before running
    /// such a node. Assets are listed by name so scripts can refer to them.
    pub fn to_python(&self) -> serde_json::Value {
        let nodes: Vec<_> = self
            .graph
            .nodes()
            .map(|node| {
                let mut params = self.registry.get(&node.kind).map(|def| def.default_params()).unwrap_or_default();
                params.extend(node.params.clone());
                serde_json::json!({
                    "id": node.id,
                    "type": node.kind,
                    "enabled": node.enabled,
                    "bound": variables::has_bindings(&params),
                    "params": params,
                    "pass_through": pass_through(&self.registry.ports(node)),
                })
            })
            .collect();
        let assets: Vec<_> = self
            .assets
            .iter()
            .map(|asset| serde_json::json!({ "id": asset.id, "name": asset.name, "kind": asset.kind().id() }))
            .collect();
        serde_json::json!({ "graph": { "nodes": nodes, "wires": self.graph.wires() }, "assets": assets })
    }
}

/// An AI model of the macro, loaded.
pub struct LoadedText {
    pub spec: magi_core::magic::TextSpec,
    pub runner: TextRunner,
    /// The model's own files (its native tools).
    pub data: std::collections::BTreeMap<String, Vec<u8>>,
    pub memory: Option<magi_core::magic::MemorySpec>,
    /// Its context scaling (llama.cpp gets it past the trained context).
    pub scaling: Option<magi_core::magic::RopeScaling>,
}

/// How a text model runs: in MacroMagi (the native engine, or its ONNX
/// model), or a GGUF file with llama.cpp.
pub enum TextRunner {
    Onnx(Mutex<magi_ml::TextModel>),
    Gguf { path: std::path::PathBuf, own: bool },
}

impl LoadedText {
    /// Asks the model; with `max_steps` above 0 its native tools run until it
    /// answers (models from elsewhere only answer).
    pub fn ask(
        &self,
        request: &str,
        options: &magi_ml::AskOptions,
        max_steps: usize,
    ) -> Result<magi_ml::tools::AgentAnswer, magi_ml::MlError> {
        let mut options = options.clone().with_memory(self.memory.as_ref());
        options.scaling = self.scaling.clone();
        let mut run = |call: &magi_core::magic::ToolCall| magi_ml::tools::run_call(&self.spec, &self.data, call);
        match &self.runner {
            TextRunner::Onnx(model) => {
                let mut generate = |prompt: &str| {
                    let mut model = model.lock().unwrap_or_else(|p| p.into_inner());
                    magi_ml::complete(&mut model, &self.spec, prompt, &options)
                };
                magi_ml::tools::agent(&self.spec, request, &options, max_steps, &mut generate, &mut run)
            }
            TextRunner::Gguf { path, own: true } => {
                let mut generate = |prompt: &str| magi_ml::llama::complete_gguf(path, &self.spec, prompt, &options);
                magi_ml::tools::agent(&self.spec, request, &options, max_steps, &mut generate, &mut run)
            }
            TextRunner::Gguf { path, own: false } => {
                let answer = magi_ml::llama::ask_gguf_with(path, &self.spec, false, request, &options)?;
                Ok(magi_ml::tools::AgentAnswer { answer, steps: Vec::new() })
            }
        }
    }

    /// Loads it now (llama.cpp: starts its server).
    pub fn warm_up(&self) -> Result<(), magi_ml::MlError> {
        match &self.runner {
            TextRunner::Gguf { path, own } => magi_ml::llama::load_gguf(
                path,
                &self.spec,
                *own,
                &magi_ml::AskOptions::default().with_memory(self.memory.as_ref()),
            ),
            TextRunner::Onnx(_) => Ok(()),
        }
    }
}

pub struct LoadedModel {
    /// Its configuration, as saved (before a block's own settings).
    pub spec: MagicModel,
    pub model: Mutex<magi_ml::Model>,
}

/// The flow output a disabled block passes execution to: `out` if it has
/// one, otherwise its first output.
pub(crate) fn pass_through(ports: &NodePorts) -> Option<String> {
    ports
        .flow_out
        .iter()
        .find(|port| port.id == magi_core::nodes::ports::OUT)
        .or_else(|| ports.flow_out.first())
        .map(|port| port.id.clone())
}
