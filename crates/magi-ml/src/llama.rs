//! Running GGUF models with llama.cpp: MacroMagi starts its `llama-server`
//! (downloaded under Settings › Runtime) for a model the first time it is
//! asked something, keeps it while MacroMagi runs, and talks to it over a
//! local port.
//!
//! Models trained in MacroMagi are sent their own tokens and answer in them,
//! so their tags (thinking, tool calls) come back exactly. Other GGUF models
//! are asked through their own chat template, with MacroMagi's tool syntax
//! explained in the system prompt.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use magi_core::magic::TextSpec;
use serde_json::{Value, json};

use crate::{Answer, MlError};

/// Where llama.cpp's server is, and how it should use the machine.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LlamaConfig {
    /// `llama-server` (`llama-server.exe` on Windows).
    pub server: PathBuf,
    /// Layers on graphics cards: 0 keeps the model on the CPU, a large number puts all of it there.
    pub gpu_layers: u32,
    /// The devices to use (`CUDA0`, `Vulkan1`…), as `llama-server --list-devices` names them; empty: its own choice.
    pub devices: Vec<String>,
    /// How to share the model between several devices, such as `3,1`.
    pub split: Option<String>,
}

static CONFIG: RwLock<Option<LlamaConfig>> = RwLock::new(None);

/// Tells MacroMagi where llama.cpp is (or that it is not installed).
pub fn set_llama(config: Option<LlamaConfig>) {
    let changed = *CONFIG.read().unwrap_or_else(|p| p.into_inner()) != config;
    *CONFIG.write().unwrap_or_else(|p| p.into_inner()) = config;
    if changed {
        // Servers started with other settings start again when next asked.
        stop_servers();
    }
}

pub fn llama() -> Option<LlamaConfig> {
    CONFIG.read().unwrap_or_else(|p| p.into_inner()).clone()
}

struct Server {
    child: Child,
    port: u16,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Running servers, by model and context window.
type Servers = HashMap<(PathBuf, usize, String), Arc<Mutex<Server>>>;

static SERVERS: Mutex<Option<Servers>> = Mutex::new(None);

/// Puts a server into a job that Windows closes when MacroMagi ends, however
/// it ends, so no server is left holding a graphics card's memory.
#[cfg(windows)]
fn close_with_us(child: &std::process::Child) {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation, SetInformationJobObject,
    };
    static JOB: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let job = *JOB.get_or_init(|| {
        // SAFETY: plain Win32 calls; the information struct is zeroed and sized as the call expects.
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return 0;
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                (&raw const info).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            job as usize
        }
    });
    if job != 0 {
        // SAFETY: the job handle lives for the whole process; the child's handle is valid while it runs.
        unsafe {
            AssignProcessToJobObject(job as _, child.as_raw_handle() as _);
        }
    }
}

#[cfg(not(windows))]
fn close_with_us(_child: &std::process::Child) {}

/// Stops the llama.cpp server of one model (unloads it).
pub fn stop_server(model: &Path) {
    if let Some(servers) = SERVERS.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
        servers.retain(|key, _| key.0 != model);
    }
}

/// Whether a model is loaded in a llama.cpp server now.
pub fn is_loaded(model: &Path) -> bool {
    SERVERS.lock().unwrap_or_else(|p| p.into_inner()).as_ref().is_some_and(|m| m.keys().any(|k| k.0 == model))
}

/// Stops every llama.cpp server MacroMagi started.
pub fn stop_servers() {
    SERVERS.lock().unwrap_or_else(|p| p.into_inner()).take();
}

fn free_port() -> Result<u16, MlError> {
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| MlError::new(format!("no free local port: {e}")))?;
    Ok(listener.local_addr().map_err(|e| MlError::new(e.to_string()))?.port())
}

fn not_installed() -> MlError {
    MlError::new("Running GGUF models needs llama.cpp: install it under Settings › Runtime.")
}

/// The server of a model, started when needed.
fn server(model: &Path, context: usize, options: &crate::AskOptions) -> Result<Arc<Mutex<Server>>, MlError> {
    let config = llama().ok_or_else(not_installed)?;
    if !config.server.is_file() {
        return Err(not_installed());
    }
    let cache = format!(
        "{}{}{}",
        options.kv_cache,
        if options.flash_attention { "+fa" } else { "" },
        options.scaling.as_ref().map_or(0.0, |s| s.factor)
    );
    let key = (model.to_path_buf(), context.max(256), cache);
    if let Some(found) = SERVERS.lock().unwrap_or_else(|p| p.into_inner()).as_ref().and_then(|m| m.get(&key)).cloned() {
        let alive = found.lock().unwrap_or_else(|p| p.into_inner()).child.try_wait().ok().flatten().is_none();
        if alive {
            return Ok(found);
        }
    }
    if !model.is_file() {
        return Err(MlError::new(format!("the model file {} is not there any more", model.display())));
    }
    let port = free_port()?;
    let mut command = Command::new(&config.server);
    command
        .arg("--model")
        .arg(model)
        .args(["--host", "127.0.0.1", "--port", &port.to_string()])
        .args(["--ctx-size", &context.max(256).to_string()])
        .args(["--n-gpu-layers", &config.gpu_layers.to_string()])
        .args(["--parallel", "1"])
        .args(["--flash-attn", if options.flash_attention { "on" } else { "auto" }])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    if !config.devices.is_empty() {
        command.args(["--device", &config.devices.join(",")]);
    }
    // Past the trained context, positions are stretched (unless the file does it itself).
    if let Some(s) = options.scaling.as_ref().filter(|s| !s.static_in_exports && s.factor > 1.0 && s.kind != "none")
        && context > s.original as usize
    {
        let factor = (context as f64 / f64::from(s.original.max(1))).min(s.factor);
        if s.kind == "yarn" {
            command.args(["--rope-scaling", "yarn", "--yarn-orig-ctx", &s.original.to_string()]);
        } else {
            // NTK has no llama.cpp switch of its own: linear interpolation is the nearest.
            command.args(["--rope-scaling", "linear"]);
        }
        command.args(["--rope-scale", &format!("{factor}")]);
    }
    // A quantized cache holds four or eight times the text in the same memory.
    if ["f32", "f16", "q8_0", "q4_0"].contains(&options.kv_cache.as_str()) {
        command.args(["--cache-type-k", &options.kv_cache, "--cache-type-v", &options.kv_cache]);
    }
    if let Some(split) = &config.split {
        command.args(["--tensor-split", split]);
    }
    if let Some(dir) = config.server.parent() {
        command.current_dir(dir);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = command.spawn().map_err(|e| MlError::new(format!("llama.cpp could not be started: {e}")))?;
    close_with_us(&child);
    // Whatever it prints goes nowhere, but must be read, or it would stall.
    let errors = Arc::new(Mutex::new(String::new()));
    if let Some(mut stderr) = child.stderr.take() {
        let errors = Arc::clone(&errors);
        std::thread::spawn(move || {
            let mut buffer = [0u8; 4096];
            while let Ok(n) = stderr.read(&mut buffer) {
                if n == 0 {
                    break;
                }
                let mut text = errors.lock().unwrap_or_else(|p| p.into_inner());
                text.push_str(&String::from_utf8_lossy(&buffer[..n]));
                if text.len() > 20_000 {
                    let cut = text.len() - 10_000;
                    text.drain(..cut);
                }
            }
        });
    }
    let server = Arc::new(Mutex::new(Server { child, port }));
    // Loading takes a moment; it answers /health with 200 once ready.
    let began = Instant::now();
    loop {
        if let Ok(Some(status)) = server.lock().unwrap_or_else(|p| p.into_inner()).child.try_wait() {
            let log = errors.lock().unwrap_or_else(|p| p.into_inner()).clone();
            let reason =
                log.lines().rev().find(|l| l.contains("error") || l.contains("failed")).unwrap_or("").trim().to_owned();
            return Err(MlError::new(format!("llama.cpp stopped ({status}) while loading the model. {reason}")));
        }
        if let Ok((200, _)) = request(port, "GET", "/health", None, Duration::from_secs(2)) {
            break;
        }
        if began.elapsed() > Duration::from_secs(300) {
            return Err(MlError::new("llama.cpp took too long to load the model"));
        }
        std::thread::sleep(Duration::from_millis(150));
    }
    // One server per model: a new context window replaces the old one.
    let mut servers = SERVERS.lock().unwrap_or_else(|p| p.into_inner());
    let servers = servers.get_or_insert_with(HashMap::new);
    servers.retain(|other, _| other.0 != key.0);
    servers.insert(key, Arc::clone(&server));
    Ok(server)
}

/// A plain HTTP/1.1 request to the local server; the status and the body.
fn request(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&Value>,
    timeout: Duration,
) -> Result<(u16, Vec<u8>), MlError> {
    let error = |e: std::io::Error| MlError::new(format!("llama.cpp did not answer: {e}"));
    let mut stream = TcpStream::connect(("127.0.0.1", port)).map_err(error)?;
    stream.set_read_timeout(Some(timeout)).map_err(error)?;
    let payload = body.map(|b| serde_json::to_vec(b).unwrap_or_default()).unwrap_or_default();
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    stream.write_all(head.as_bytes()).map_err(error)?;
    stream.write_all(&payload).map_err(error)?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response).map_err(error)?;
    let split = response
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| MlError::new("llama.cpp sent a broken answer"))?;
    let head = String::from_utf8_lossy(&response[..split]).into_owned();
    let status = head.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut body = response[split + 4..].to_vec();
    if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        body = unchunk(&body);
    }
    Ok((status, body))
}

fn unchunk(body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(end) = body[at..].windows(2).position(|w| w == b"\r\n") {
        let size = usize::from_str_radix(String::from_utf8_lossy(&body[at..at + end]).trim(), 16).unwrap_or(0);
        at += end + 2;
        if size == 0 || at + size > body.len() {
            break;
        }
        out.extend_from_slice(&body[at..at + size]);
        at += size + 2;
    }
    out
}

/// The system prompt a GGUF model from elsewhere gets: the model's own, and
/// how to call MacroMagi's tools.
fn system_for(spec: &TextSpec) -> String {
    let mut system = spec.system.trim().to_owned();
    if !spec.tools.is_empty() {
        let tools: Vec<String> =
            spec.tools.iter().map(|t| format!("- {}: {} Arguments: {}", t.name, t.description, t.parameters)).collect();
        system.push_str(&format!(
            "\n\nYou can use these tools:\n{}\nTo use one, write {}{{\"name\": \"<tool>\", \"args\": {{...}}}}{} – \
             JSON exactly in that form, one tool call per tag pair. Then say briefly what you did.",
            tools.join("\n"),
            spec.tool_tags.0,
            spec.tool_tags.1
        ));
    }
    system.trim().to_owned()
}

/// Asks a GGUF model something. `own`: the model was trained in MacroMagi
/// and reads its tokens. `max_tokens` 0 and `temperature` below 0 use the spec's.
pub fn ask_gguf(
    model: &Path,
    spec: &TextSpec,
    own: bool,
    request_text: &str,
    max_tokens: usize,
    temperature: f32,
    seed: u64,
) -> Result<Answer, MlError> {
    ask_gguf_with(
        model,
        spec,
        own,
        request_text,
        &crate::AskOptions { max_tokens, temperature, seed, ..Default::default() },
    )
}

/// Loads a GGUF model now (starts its server), so the first request is quick.
pub fn load_gguf(model: &Path, spec: &TextSpec, own: bool, options: &crate::AskOptions) -> Result<(), MlError> {
    server(model, gguf_context(spec, own, options.context), options).map(|_| ())
}

fn gguf_context(spec: &TextSpec, own: bool, context: usize) -> usize {
    match (context, own) {
        (0, true) => spec.context as usize,
        (0, false) => 8192,
        (n, _) => n,
    }
}

/// Asks a GGUF model something, with the options of a request.
pub fn ask_gguf_with(
    model: &Path,
    spec: &TextSpec,
    own: bool,
    request_text: &str,
    options: &crate::AskOptions,
) -> Result<Answer, MlError> {
    let raw = if own {
        complete_gguf(model, spec, &spec.prompt_with(request_text, options.thinking, &options.effort), options)?
    } else {
        let context = gguf_context(spec, own, options.context);
        let server = server(model, context, options)?;
        let port = server.lock().unwrap_or_else(|p| p.into_inner()).port;
        let max = if options.max_tokens == 0 { spec.max_tokens as usize } else { options.max_tokens };
        let temperature = if options.temperature < 0.0 { spec.temperature } else { options.temperature };
        let (seed, timeout) = (options.seed, Duration::from_secs(600));
        let mut messages = Vec::new();
        let system = system_for(spec);
        if !system.is_empty() {
            messages.push(json!({ "role": "system", "content": system }));
        }
        messages.push(json!({ "role": "user", "content": request_text }));
        let body = json!({ "messages": messages, "max_tokens": max, "temperature": temperature, "top_k": spec.top_k, "seed": seed as u32 });
        let (status, bytes) = request(port, "POST", "/v1/chat/completions", Some(&body), timeout)?;
        let answer: Value =
            serde_json::from_slice(&bytes).map_err(|_| MlError::new(format!("llama.cpp answered {status}")))?;
        if status != 200 {
            return Err(MlError::new(format!(
                "llama.cpp: {}",
                answer.pointer("/error/message").and_then(Value::as_str).unwrap_or("it failed")
            )));
        }
        let message = &answer["choices"][0]["message"];
        let thinking = message["reasoning_content"].as_str().unwrap_or_default();
        let content = message["content"].as_str().unwrap_or_default();
        if thinking.is_empty() {
            content.to_owned()
        } else {
            format!("{}{thinking}{}{content}", spec.think_tags.0, spec.think_tags.1)
        }
    };
    Ok(Answer { text: spec.visible(&raw), thinking: spec.thinking(&raw), tools: spec.tool_calls(&raw), raw })
}

/// What a model trained in MacroMagi writes after a prompt (in its chat
/// template), as raw text: sent and read back as token ids.
pub fn complete_gguf(
    model: &Path,
    spec: &TextSpec,
    text: &str,
    options: &crate::AskOptions,
) -> Result<String, MlError> {
    let context = gguf_context(spec, true, options.context);
    let server = server(model, context, options)?;
    let port = server.lock().unwrap_or_else(|p| p.into_inner()).port;
    let max = if options.max_tokens == 0 { spec.max_tokens as usize } else { options.max_tokens };
    let temperature = if options.temperature < 0.0 { spec.temperature } else { options.temperature };
    let mut prompt: Vec<u32> = spec.encode(text);
    if prompt.len() + 1 > context {
        prompt.drain(..prompt.len() + 1 - context);
    }
    let body = json!({
        "prompt": prompt, "n_predict": max, "temperature": temperature, "top_k": spec.top_k,
        "seed": options.seed as u32, "return_tokens": true, "cache_prompt": true,
    });
    let (status, bytes) = request(port, "POST", "/completion", Some(&body), Duration::from_secs(600))?;
    let answer: Value =
        serde_json::from_slice(&bytes).map_err(|_| MlError::new(format!("llama.cpp answered {status}")))?;
    if status != 200 {
        return Err(MlError::new(format!(
            "llama.cpp: {}",
            answer.pointer("/error/message").and_then(Value::as_str).unwrap_or("it failed")
        )));
    }
    let end = spec.special_id(&spec.end_tag);
    let tokens: Vec<u32> = answer["tokens"]
        .as_array()
        .map(|t| t.iter().filter_map(Value::as_u64).map(|t| t as u32).take_while(|t| Some(*t) != end).collect())
        .unwrap_or_default();
    Ok(if tokens.is_empty() { answer["content"].as_str().unwrap_or_default().to_owned() } else { spec.decode(&tokens) })
}

/// The devices llama.cpp can use, as `(id, description)`: `("CUDA0", "NVIDIA GeForce RTX 5070 (12199 MiB…)")`.
pub fn list_devices(server: &Path) -> Vec<(String, String)> {
    let mut command = Command::new(server);
    command.arg("--list-devices").stdin(Stdio::null());
    if let Some(dir) = server.parent() {
        command.current_dir(dir);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }
    let Ok(output) = command.output() else { return Vec::new() };
    let text = String::from_utf8_lossy(&output.stdout).into_owned() + &String::from_utf8_lossy(&output.stderr);
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            let (id, rest) = line.split_once(':')?;
            (!id.contains(' ') && !id.is_empty() && id.chars().any(|c| c.is_ascii_digit()))
                .then(|| (id.to_owned(), rest.trim().to_owned()))
        })
        .collect()
}
