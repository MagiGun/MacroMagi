//! Running a model's native tools, and the agent loop around them.
//!
//! A native tool ([`magi_core::native_tools`]) runs on this computer: its
//! files are put in a folder of the cache (once per version), its program
//! gets the call's arguments as JSON on standard input, and what it prints is
//! the result. The agent loop asks the model, runs the native tools it called,
//! gives back their results in tool turns, and asks again – until the model
//! answers without calling one, or calls a tool that is not native (which is
//! left to whoever asked), or the steps run out.
//!
//! A tool marked `confirm` (it writes files or runs programs) runs only when
//! the user allows it: the call waits in [`pending_approvals`] until the
//! program showing them answers with [`answer_approval`]. A tool run by the
//! `host` is carried out by a function the program registered
//! ([`set_host_tool`]): reading a picture's text, say, which needs the system.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Mutex, RwLock, mpsc};
use std::time::{Duration, Instant};

use magi_core::magic::{TextSpec, ToolCall};
use magi_core::native_tools::{NativeTool, RunSpec};
use serde_json::{Value, json};

use crate::{Answer, AskOptions, MlError};

static PYTHON: RwLock<Option<PathBuf>> = RwLock::new(None);
static CACHE: RwLock<Option<PathBuf>> = RwLock::new(None);
static ENV: RwLock<BTreeMap<String, String>> = RwLock::new(BTreeMap::new());

/// A tool the program running the model carries out itself (`"run": {"kind": "host"}`).
pub type HostTool = Box<dyn Fn(&Value) -> Result<Value, String> + Send + Sync>;
static HOST: RwLock<BTreeMap<String, HostTool>> = RwLock::new(BTreeMap::new());

/// A call waiting for the user to allow it.
pub struct Approval {
    pub id: u64,
    pub call: ToolCall,
    answer: mpsc::Sender<bool>,
}

struct Approvals {
    /// Whether anyone shows the calls to the user; without, they are refused.
    shown: bool,
    next: u64,
    pending: Vec<Approval>,
    /// Tools the user allowed for the rest of the session.
    always: BTreeSet<String>,
    notify: Option<Box<dyn Fn() + Send + Sync>>,
}

static APPROVALS: Mutex<Approvals> =
    Mutex::new(Approvals { shown: false, next: 1, pending: Vec::new(), always: BTreeSet::new(), notify: None });

/// Calls that ask first are shown to the user, who allows or refuses them; `notify` is called
/// when one arrives (to redraw the window). Without this, such calls are refused.
pub fn show_approvals(notify: Box<dyn Fn() + Send + Sync>) {
    let mut approvals = APPROVALS.lock().unwrap_or_else(|p| p.into_inner());
    approvals.shown = true;
    approvals.notify = Some(notify);
}

/// The calls waiting for the user: their ids and the calls.
pub fn pending_approvals() -> Vec<(u64, ToolCall)> {
    APPROVALS.lock().unwrap_or_else(|p| p.into_inner()).pending.iter().map(|a| (a.id, a.call.clone())).collect()
}

/// The user's answer to a call; `always` allows its tool for the rest of the session.
pub fn answer_approval(id: u64, allow: bool, always: bool) {
    let mut approvals = APPROVALS.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(index) = approvals.pending.iter().position(|a| a.id == id) {
        let approval = approvals.pending.remove(index);
        if allow && always {
            approvals.always.insert(approval.call.name.clone());
        }
        let _ = approval.answer.send(allow);
    }
}

/// Asks the user about a call, waiting up to ten minutes for the answer.
fn approved(call: &ToolCall) -> bool {
    let (sender, receiver) = mpsc::channel();
    {
        let mut approvals = APPROVALS.lock().unwrap_or_else(|p| p.into_inner());
        if approvals.always.contains(&call.name) {
            return true;
        }
        if !approvals.shown {
            return false;
        }
        let id = approvals.next;
        approvals.next += 1;
        approvals.pending.push(Approval { id, call: call.clone(), answer: sender });
        if let Some(notify) = &approvals.notify {
            notify();
        }
    }
    receiver.recv_timeout(Duration::from_secs(600)).unwrap_or(false)
}

/// Registers a tool the program carries out itself.
pub fn set_host_tool(name: &str, run: HostTool) {
    HOST.write().unwrap_or_else(|p| p.into_inner()).insert(name.to_owned(), run);
}

/// Environment variables native tools run with (where MagiDB is, say).
pub fn set_tool_env(key: &str, value: Option<String>) {
    let mut env = ENV.write().unwrap_or_else(|p| p.into_inner());
    match value {
        Some(value) => env.insert(key.to_owned(), value),
        None => env.remove(key),
    };
}

/// The Python native tools run with (the Python for AI).
pub fn set_python(python: Option<PathBuf>) {
    *PYTHON.write().unwrap_or_else(|p| p.into_inner()) = python;
}

/// Where native tools' files and the native engine's cache files go.
pub fn set_cache_dir(dir: PathBuf) {
    *CACHE.write().unwrap_or_else(|p| p.into_inner()) = Some(dir);
}

pub fn cache_dir() -> PathBuf {
    CACHE.read().unwrap_or_else(|p| p.into_inner()).clone().unwrap_or_else(|| std::env::temp_dir().join("MacroMagi"))
}

/// Runs one native tool with `args`; its `files` are its folder's.
pub fn run_native(tool: &NativeTool, files: &[(String, Vec<u8>)], args: &Value) -> Result<Value, String> {
    // A folder per version of the tool's files, so changed tools are never stale.
    let mut all = Vec::new();
    for (path, bytes) in files {
        all.extend_from_slice(path.as_bytes());
        all.extend_from_slice(bytes);
    }
    let hash: String = crate::native::sha256(&all)[..8].iter().map(|b| format!("{b:02x}")).collect();
    let dir = cache_dir().join("nt").join(format!("{}-{hash}", tool.name));
    if !dir.join("tool.json").is_file() {
        for (path, bytes) in files {
            let target = dir.join(path);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            std::fs::write(target, bytes).map_err(|e| e.to_string())?;
        }
    }
    run_in(&dir, &tool.run, args)
}

fn run_in(dir: &Path, run: &RunSpec, args: &Value) -> Result<Value, String> {
    if run.kind == "host" {
        let host = HOST.read().unwrap_or_else(|p| p.into_inner());
        let tool = host.get(&run.entry).ok_or_else(|| format!("this program cannot run the tool {}", run.entry))?;
        return tool(args);
    }
    let entry = dir.join(&run.entry);
    let mut command = if run.kind == "command" {
        Command::new(&entry)
    } else {
        let python = PYTHON
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
            .ok_or("native tools need Python: choose one under Settings › Engine")?;
        let mut command = Command::new(python);
        command.arg("-I").arg(&entry);
        command
    };
    command.envs(ENV.read().unwrap_or_else(|p| p.into_inner()).iter());
    command
        .current_dir(dir)
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = command.spawn().map_err(|e| format!("the tool could not be started: {e}"))?;
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(args.to_string().as_bytes());
    }
    let mut stdout = child.stdout.take().ok_or("no output")?;
    let reader = std::thread::spawn(move || {
        let mut out = Vec::new();
        let _ = stdout.read_to_end(&mut out);
        out
    });
    let began = Instant::now();
    let limit = Duration::from_millis(run.timeout_ms.max(1000));
    loop {
        if child.try_wait().map_err(|e| e.to_string())?.is_some() {
            break;
        }
        if began.elapsed() > limit {
            let _ = child.kill();
            return Err(format!("the tool took longer than {} s", limit.as_secs()));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let out = reader.join().unwrap_or_default();
    let text = String::from_utf8_lossy(&out);
    serde_json::from_str(text.trim()).map_err(|_| {
        let mut errors = String::new();
        if let Some(mut stderr) = child.stderr.take() {
            let _ = stderr.read_to_string(&mut errors);
        }
        format!(
            "the tool did not answer in JSON: {}",
            if errors.trim().is_empty() { text.trim() } else { errors.trim() }
        )
    })
}

/// One tool the agent loop ran.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    /// What the model thought before it called the tool (its first call of a turn only).
    pub thinking: String,
    pub call: ToolCall,
    pub result: Value,
}

/// What the agent loop ended with: the last answer, and the tools it ran on the way.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AgentAnswer {
    pub answer: Answer,
    pub steps: Vec<Step>,
}

/// The text after an answer that called tools: the answer's turn closed, a
/// tool turn per result, and the next answer's turn opened.
pub fn continue_with(spec: &TextSpec, raw: &str, results: &[Value], thinking: Option<bool>) -> String {
    let mut text = format!("{raw}{}", spec.end_tag);
    for result in results {
        let content = magi_core::magic::spaced_json(result);
        if spec.tool_tag.is_empty() {
            text.push_str(&format!("{}Tool result: {content}{}", spec.user_tag, spec.end_tag));
        } else {
            text.push_str(&format!("{}{content}{}", spec.tool_tag, spec.end_tag));
        }
    }
    text.push_str(&spec.assistant_tag);
    if spec.thinking.is_some() && !spec.thinks(thinking) {
        text.push_str(&spec.think_tags.0);
        text.push_str(&spec.think_tags.1);
    }
    text
}

/// Asks a model, running its native tools for it. `generate` writes the
/// model's continuation of a prompt (as raw text); `run` runs a native tool,
/// or gives `None` for a tool that is not one.
pub fn agent(
    spec: &TextSpec,
    request: &str,
    options: &AskOptions,
    max_steps: usize,
    generate: &mut dyn FnMut(&str) -> Result<String, MlError>,
    run: &mut dyn FnMut(&ToolCall) -> Option<Value>,
) -> Result<AgentAnswer, MlError> {
    let prompt = spec.prompt_with(request, options.thinking, &options.effort);
    agent_from(spec, prompt, options, max_steps, generate, run)
}

/// Tool results the agent loop keeps whole when a long piece of work outgrows the context:
/// the latest ones. Older ones are shortened, oldest first, until the text fits.
pub const KEEP_WHOLE: usize = 6;
/// What is left of a shortened result, in characters.
const SHORTENED: usize = 240;
/// Rounds (a call and its result) kept when even shortened results do not fit: the latest.
pub const KEEP_ROUNDS: usize = 12;

/// What stands for the rounds left out of a very long piece of work.
pub fn left_out(rounds: usize) -> Value {
    json!({ "note": format!("{rounds} earlier steps are left out to make room; your notes say where the work stands") })
}

/// A tool result as it is kept once the work has moved on: its start, and a note.
pub fn shortened(result: &Value) -> Value {
    let text = magi_core::magic::spaced_json(result);
    if text.chars().count() <= SHORTENED {
        return result.clone();
    }
    let start: String = text.chars().take(SHORTENED).collect();
    json!({ "shortened": format!("{start}…"), "note": "an earlier result, shortened: call the tool again to see it all" })
}

/// The agent loop from a prompt of any length – a whole conversation, say
/// ([`TextSpec::chat_prompt`]) – ending where the model answers. When the work
/// outgrows the context (`options.context`, else the model's), the results of
/// earlier calls are shortened, oldest first, keeping the latest [`KEEP_WHOLE`]
/// whole; if even that is too long, the oldest rounds are left out (keeping the
/// latest [`KEEP_ROUNDS`]), with a note – so the start (the system text and the
/// request) is never lost. The model keeps track of the work in its thinking.
pub fn agent_from(
    spec: &TextSpec,
    prompt: String,
    options: &AskOptions,
    max_steps: usize,
    generate: &mut dyn FnMut(&str) -> Result<String, MlError>,
    run: &mut dyn FnMut(&ToolCall) -> Option<Value>,
) -> Result<AgentAnswer, MlError> {
    let window = if options.context > 0 { options.context } else { spec.context as usize };
    let room = if options.max_tokens > 0 { options.max_tokens } else { spec.max_tokens as usize };
    let budget = window.saturating_sub(room).max(window / 2);
    let mut steps = Vec::new();
    // Each round: what the model wrote, and the results it was given.
    let mut rounds: Vec<(String, Vec<Value>)> = Vec::new();
    let (mut short, mut dropped) = (0, 0);
    let mut text = prompt.clone();
    for _ in 0..=max_steps {
        let raw = generate(&text)?;
        let answer =
            Answer { text: spec.visible(&raw), thinking: spec.thinking(&raw), tools: spec.tool_calls(&raw), raw };
        if answer.tools.is_empty() || steps.len() >= max_steps {
            return Ok(AgentAnswer { answer, steps });
        }
        let mut results = Vec::new();
        for (n, call) in answer.tools.iter().enumerate() {
            match run(call) {
                Some(result) => {
                    results.push(result.clone());
                    let thinking = if n == 0 { answer.thinking.clone() } else { String::new() };
                    steps.push(Step { thinking, call: call.clone(), result });
                }
                // Not a native tool: the asker carries it out.
                None => return Ok(AgentAnswer { answer, steps }),
            }
        }
        rounds.push((answer.raw, results));
        let build = |short: usize, dropped: usize| {
            let mut text = prompt.clone();
            if dropped > 0 {
                text.push_str(&continue_with(spec, "", &[left_out(dropped)], options.thinking));
            }
            for (index, (raw, results)) in rounds.iter().enumerate().skip(dropped) {
                let kept: Vec<Value> =
                    if index < short { results.iter().map(shortened).collect() } else { results.clone() };
                text.push_str(&continue_with(spec, raw, &kept, options.thinking));
            }
            text
        };
        text = build(short, dropped);
        // Too long: shorten the oldest whole results until it fits, then leave out the oldest rounds.
        while short + KEEP_WHOLE < rounds.len() && spec.encode(&text).len() > budget {
            short += 1;
            text = build(short, dropped);
        }
        while dropped + KEEP_ROUNDS < rounds.len() && spec.encode(&text).len() > budget {
            dropped += 1;
            text = build(short, dropped);
        }
    }
    Err(MlError::new("the agent loop ended without an answer"))
}

/// Runs a call as one of the model's native tools, if it is one: `data` is
/// the model's files (its own `nt/…`), the built-in tools otherwise.
pub fn run_call(spec: &TextSpec, data: &std::collections::BTreeMap<String, Vec<u8>>, call: &ToolCall) -> Option<Value> {
    if !spec.tools.iter().any(|t| t.native && t.name == call.name) {
        return None;
    }
    let Some(tool) = magi_core::native_tools::tool(&call.name, data) else {
        return Some(json!({ "error": format!("the native tool {} is missing", call.name) }));
    };
    if tool.confirm && !approved(call) {
        return Some(json!({ "error": "the user did not allow this call" }));
    }
    let files = magi_core::native_tools::files(&call.name, data);
    Some(run_native(&tool, &files, &call.args).unwrap_or_else(|error| json!({ "error": error })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tool_loop_feeds_results_back_until_the_answer() {
        let mut spec = TextSpec {
            tools: vec![magi_core::magic::ToolSpec { name: "calculator".into(), native: true, ..Default::default() }],
            ..TextSpec::default()
        };
        spec.ensure_specials();
        let mut asked = Vec::new();
        let mut generate = |prompt: &str| {
            asked.push(prompt.to_owned());
            Ok(if asked.len() == 1 {
                r#"<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>"#.to_owned()
            } else {
                "It is 4.".to_owned()
            })
        };
        let mut run = |call: &ToolCall| (call.name == "calculator").then(|| json!({ "result": 4 }));
        let done = agent(&spec, "2+2?", &AskOptions::default(), 4, &mut generate, &mut run).unwrap();
        assert_eq!(done.answer.text, "It is 4.");
        assert_eq!(done.steps.len(), 1);
        assert!(asked[1].ends_with("</tool_call><|end|><|tool|>{\"result\": 4}<|end|><|assistant|>"), "{}", asked[1]);
    }

    #[test]
    fn long_work_shortens_old_results_and_keeps_the_start() {
        let mut spec = TextSpec {
            tools: vec![magi_core::magic::ToolSpec { name: "read_file".into(), native: true, ..Default::default() }],
            context: 1200,
            max_tokens: 100,
            ..TextSpec::default()
        };
        spec.ensure_specials();
        let mut asked: Vec<String> = Vec::new();
        let mut generate = |prompt: &str| {
            asked.push(prompt.to_owned());
            Ok(if asked.len() <= 12 {
                format!(
                    r#"<tool_call>{{"name": "read_file", "arguments": {{"path": "f{}.txt"}}}}</tool_call>"#,
                    asked.len()
                )
            } else {
                "Done.".to_owned()
            })
        };
        let mut run = |_: &ToolCall| Some(json!({ "text": "x".repeat(300) }));
        let done = agent(&spec, "Read them all.", &AskOptions::default(), 20, &mut generate, &mut run).unwrap();
        assert_eq!((done.answer.text.as_str(), done.steps.len()), ("Done.", 12));
        let last = asked.last().unwrap();
        assert!(last.contains("Read them all."), "the request stays");
        assert!(last.contains("an earlier result, shortened"));
        assert_eq!(last.matches(&"x".repeat(300)).count(), KEEP_WHOLE, "the latest results stay whole");

        // Far longer work: the oldest rounds are left out, the latest kept.
        let mut asked: Vec<String> = Vec::new();
        let mut generate = |prompt: &str| {
            asked.push(prompt.to_owned());
            Ok(if asked.len() <= 40 {
                format!(
                    r#"<tool_call>{{"name": "read_file", "arguments": {{"path": "f{}.txt"}}}}</tool_call>"#,
                    asked.len()
                )
            } else {
                "Done.".to_owned()
            })
        };
        let done = agent(&spec, "Read them all.", &AskOptions::default(), 50, &mut generate, &mut run).unwrap();
        assert_eq!(done.steps.len(), 40);
        let last = asked.last().unwrap();
        assert!(last.contains("Read them all.") && last.contains("earlier steps are left out"));
        assert!(last.contains("f40.txt") && !last.contains("\"f1.txt\""));
    }
}
