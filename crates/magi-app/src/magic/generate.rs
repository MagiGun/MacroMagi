//! Generating samples with another AI model: the dataset editor asks a model
//! that serves an OpenAI-compatible API – LM Studio's local server, Ollama,
//! vLLM, llama.cpp's server, or an external one – for samples of a dataset,
//! in MacroMagi's format, a batch at a time. Each sample is checked as the
//! editor checks one typed by hand, and repeats are left out.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use magi_core::magic::TextSpec;
use magi_core::nodes::studio;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Where samples are generated: an OpenAI-compatible endpoint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GeneratorSettings {
    /// The API's base address, up to `/v1`.
    pub url: String,
    /// The model to ask; empty: the first the endpoint lists.
    pub model: String,
    /// Only for endpoints that need one (most local servers do not).
    pub key: String,
    /// How much a thinking model thinks before it writes: `model` (as it does), `low` or `none` –
    /// sent as `reasoning_effort` (and `/no_think`, which Qwen models read); far faster without.
    pub thinking: String,
}

impl Default for GeneratorSettings {
    fn default() -> Self {
        Self {
            url: "http://localhost:1234/v1".into(),
            model: String::new(),
            key: String::new(),
            thinking: "none".into(),
        }
    }
}

/// The thinking settings, for a choice.
pub const THINKING: [(&str, &str); 3] = [("none", "No thinking"), ("low", "Little"), ("model", "As the model does")];

/// Endpoints people use, to fill in the address.
pub const ENDPOINTS: [(&str, &str); 4] = [
    ("LM Studio", "http://localhost:1234/v1"),
    ("Ollama", "http://localhost:11434/v1"),
    ("llama.cpp server", "http://localhost:8080/v1"),
    ("vLLM", "http://localhost:8000/v1"),
];

/// What kind of samples to generate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Answers,
    Reasoning,
    Conversations,
    ToolUse,
}

impl Kind {
    pub const ALL: [(Kind, &'static str); 4] = [
        (Kind::Answers, "Requests and answers"),
        (Kind::Reasoning, "Reasoning"),
        (Kind::Conversations, "Conversations"),
        (Kind::ToolUse, "Tool use"),
    ];
}

pub struct Request {
    pub settings: GeneratorSettings,
    /// What the samples should be about, in the person's words.
    pub prompt: String,
    pub count: usize,
    pub kind: Kind,
    /// The model the dataset is for: its tools and reasoning efforts.
    pub spec: TextSpec,
    /// The samples there already, so none are repeated.
    pub existing: Vec<String>,
}

/// How a generation goes, read by the editor as it goes.
#[derive(Default)]
pub struct Progress {
    /// Checked samples, not yet taken by the editor.
    pub made: Vec<String>,
    pub total: usize,
    /// Lines the model wrote that were not samples, or were repeats.
    pub rejected: usize,
    pub status: String,
    pub error: Option<String>,
    pub done: bool,
}

pub struct Job {
    pub progress: Arc<Mutex<Progress>>,
    stop: Arc<AtomicBool>,
}

impl Job {
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(600))).build().into()
}

/// The models an endpoint serves.
pub fn models(settings: &GeneratorSettings) -> Result<Vec<String>, String> {
    let mut request = agent().get(format!("{}/models", settings.url.trim_end_matches('/')));
    if !settings.key.trim().is_empty() {
        request = request.header("Authorization", format!("Bearer {}", settings.key.trim()));
    }
    let text = request
        .call()
        .map_err(|e| format!("Could not reach {}: {e}", settings.url))?
        .body_mut()
        .read_to_string()
        .map_err(|e| e.to_string())?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|_| "The endpoint did not answer as an OpenAI-compatible API.".to_owned())?;
    Ok(value["data"].as_array().into_iter().flatten().filter_map(|m| m["id"].as_str().map(str::to_owned)).collect())
}

/// What MacroMagi's own native tools give back, so generated samples show their results as they are.
const RESULTS: [(&str, &str); 7] = [
    ("calculator", r#"{"result": 12662}"#),
    ("get_time", r#"{"location": "Tokyo", "time": "18:30", "date": "Tuesday 20 October 2026", "zone": "Asia/Tokyo"}"#),
    ("get_weather", r#"{"location": "Paris", "temperature_c": 14.2, "conditions": "partly cloudy", "wind_kmh": 11.3}"#),
    ("web_search", r#"{"results": [{"title": "…", "url": "https://…", "snippet": "…"}]}"#),
    ("fetch_page", r#"{"text": "the page's text…", "more": false}"#),
    ("read_file", r#"{"path": "C:/notes/todo.md", "text": "the file's text…"}"#),
    ("list_files", r#"{"folder": "C:/notes", "files": ["todo.md"], "folders": ["old"]}"#),
];

/// Whether every call of a conversation is followed by its result – one tool message per call.
fn calls_answered(sample: &Value) -> bool {
    let Some(messages) = sample.get("messages").and_then(Value::as_array) else { return true };
    let is_tool = |m: &Value| m.get("role").and_then(Value::as_str) == Some("tool");
    messages.iter().enumerate().all(|(index, message)| {
        let calls = message.get("tool_calls").and_then(Value::as_array).map_or(0, Vec::len);
        calls == 0 || messages[index + 1..].iter().take_while(|m| is_tool(m)).count() == calls
    })
}

/// What the model is told: MacroMagi's sample format for the kind, the dataset's model, the rules.
fn instructions(request: &Request) -> String {
    let spec = &request.spec;
    let format = match request.kind {
        Kind::Answers => r#"{"prompt": "a request", "response": "the answer"}"#.to_owned(),
        Kind::Reasoning => {
            let levels = spec.effort.as_ref().map(|e| e.levels.join(" or ")).unwrap_or_else(|| "standard".into());
            format!(
                r#"{{"prompt": "a problem", "thinking": "the reasoning, step by step", "response": "the short answer", "effort": "{levels}"}}"#
            ) + "\nAt the lower effort the reasoning is short and only for problems that need it; at the higher one every \
                 problem is reasoned about and harder ones are checked at the end."
        }
        Kind::Conversations => {
            r#"{"messages": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}, {"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}]}"#.to_owned()
        }
        Kind::ToolUse => {
            let tools: Vec<String> = spec
                .tools
                .iter()
                .map(|t| {
                    let result = RESULTS.iter().find(|(name, _)| *name == t.name).map(|(_, r)| format!(" It gives back {r}"));
                    format!("- {}({}): {}{}", t.name, TextSpec::signature(&t.parameters), t.description, result.unwrap_or_default())
                })
                .collect();
            format!(
                "{}\nThe assistant may call these tools, and only these:\n{}\nA tool's result is a JSON object in a \
                 \"tool\" message – one for each call, in order; when a tool fails it gives {{\"error\": \"…\"}} and the \
                 assistant recovers or says so. The answer uses the result's facts exactly. When no tool can do what is \
                 asked, the assistant says so plainly without calling one.",
                r#"{"messages": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "", "thinking": "why this tool", "tool_calls": [{"name": "tool_name", "arguments": {"arg": "value"}}]}, {"role": "tool", "content": "{\"result\": …}"}, {"role": "assistant", "content": "the answer, from the result"}]}"#,
                tools.join("\n")
            )
        }
    };
    format!(
        "You write training samples for a small AI assistant called Magi. Write each sample as one JSON object on one \
         line, in exactly this form:\n{format}\n\nRules: one JSON object per line and nothing else – no numbering, no \
         commentary, no code fences. Every answer must be correct, clear and as short as the request allows. Vary the \
         wording, the topics and the length; do not repeat a sample."
    )
}

/// One batch of samples: the model's lines, read as samples.
fn ask(request: &Request, how_many: usize, avoid: &[String]) -> Result<String, String> {
    let settings = &request.settings;
    let mut user = format!("Write {how_many} samples about: {}", request.prompt.trim());
    if settings.thinking == "none" {
        // Qwen's switch for answering without thinking; other models take it as an empty line.
        user.push_str("\n/no_think");
    }
    if !avoid.is_empty() {
        user.push_str("\n\nDifferent from these, already written:\n");
        user.push_str(&avoid.join("\n"));
    }
    let mut body = json!({
        "messages": [
            { "role": "system", "content": instructions(request) },
            { "role": "user", "content": user },
        ],
        "temperature": 0.9,
        // Room for a thinking model's reasoning, then the samples.
        "max_tokens": 6000 + 600 * how_many,
        "stream": false,
    });
    if !settings.model.trim().is_empty() {
        body["model"] = json!(settings.model.trim());
    }
    if settings.thinking == "none" || settings.thinking == "low" {
        body["reasoning_effort"] = json!(settings.thinking);
    }
    let send = |body: &Value| -> Result<String, String> {
        let mut call = agent()
            .post(format!("{}/chat/completions", settings.url.trim_end_matches('/')))
            .header("Content-Type", "application/json");
        if !settings.key.trim().is_empty() {
            call = call.header("Authorization", format!("Bearer {}", settings.key.trim()));
        }
        call.send(serde_json::to_string(body).map_err(|e| e.to_string())?)
            .map_err(|e| format!("The endpoint did not answer: {e}"))?
            .body_mut()
            .read_to_string()
            .map_err(|e| e.to_string())
    };
    // An endpoint that does not know reasoning_effort is asked again without it.
    let text = match send(&body) {
        Err(_) if body.get("reasoning_effort").is_some() => {
            body.as_object_mut().map(|b| b.remove("reasoning_effort"));
            send(&body)?
        }
        other => other?,
    };
    let value: Value = serde_json::from_str(&text)
        .map_err(|_| "The endpoint did not answer as an OpenAI-compatible API.".to_owned())?;
    if let Some(error) = value.get("error") {
        return Err(format!(
            "The endpoint said: {}",
            error.get("message").and_then(Value::as_str).unwrap_or("an error")
        ));
    }
    let choice = &value["choices"][0];
    let content = choice["message"]["content"].as_str().unwrap_or_default();
    // A thinking model's reasoning, when the server leaves it in the answer.
    let content = match content.rfind("</think>") {
        Some(end) => &content[end + "</think>".len()..],
        None => content,
    };
    if content.trim().is_empty() && choice["finish_reason"] == "length" {
        return Err("The model ran out of room while it was thinking: choose a model that thinks less, or one without thinking.".into());
    }
    Ok(content.to_owned())
}

/// The samples in a model's answer: its JSON lines that check out as samples of the dataset.
pub fn samples_in(answer: &str, spec: &TextSpec) -> (Vec<String>, usize) {
    let mut good = Vec::new();
    let mut bad = 0;
    for line in answer.lines().map(str::trim) {
        if line.is_empty() || line.starts_with("```") {
            continue;
        }
        let line = line.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ')' || c == '-').trim();
        match serde_json::from_str::<Value>(line) {
            Ok(Value::Object(item)) => {
                let written = magi_core::dataset::to_line(&item);
                if studio::check_sample(true, &written, spec).is_ok() && calls_answered(&Value::Object(item)) {
                    good.push(written);
                } else {
                    bad += 1;
                }
            }
            _ => bad += 1,
        }
    }
    (good, bad)
}

/// Generates the samples in the background; the editor takes them as they come.
pub fn start(request: Request, repaint: impl Fn() + Send + 'static) -> Job {
    let progress = Arc::new(Mutex::new(Progress::default()));
    let stop = Arc::new(AtomicBool::new(false));
    let (state, stopped) = (Arc::clone(&progress), Arc::clone(&stop));
    std::thread::spawn(move || {
        let lock = || state.lock().unwrap_or_else(|p| p.into_inner());
        let mut seen: std::collections::HashSet<String> = request.existing.iter().cloned().collect();
        let mut recent: Vec<String> = request.existing.iter().rev().take(6).cloned().collect();
        let mut empty = 0;
        while lock().total < request.count && !stopped.load(Ordering::Relaxed) {
            let left = request.count - lock().total;
            let batch = left.min(10);
            lock().status = format!("Asking the model for {batch} more…");
            repaint();
            match ask(&request, batch, &recent) {
                Ok(answer) => {
                    let (good, bad) = samples_in(&answer, &request.spec);
                    let mut state = lock();
                    state.rejected += bad;
                    let mut added = 0;
                    for sample in good {
                        if state.total >= request.count {
                            break;
                        }
                        if seen.insert(sample.clone()) {
                            recent.push(sample.clone());
                            state.made.push(sample);
                            state.total += 1;
                            added += 1;
                        } else {
                            state.rejected += 1;
                        }
                    }
                    if recent.len() > 6 {
                        let extra = recent.len() - 6;
                        recent.drain(..extra);
                    }
                    empty = if added == 0 { empty + 1 } else { 0 };
                    if empty >= 3 {
                        state.error = Some(
                            "The model wrote no usable samples three times in a row: try a clearer request or another model."
                                .into(),
                        );
                        break;
                    }
                }
                Err(error) => {
                    lock().error = Some(error);
                    break;
                }
            }
            repaint();
        }
        let mut state = lock();
        state.done = true;
        state.status = if stopped.load(Ordering::Relaxed) { "Stopped.".into() } else { "Done.".into() };
        drop(state);
        repaint();
    });
    Job { progress, stop }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real endpoint: `MAGI_TEST_GENERATOR_URL=http://localhost:1234/v1 [MAGI_TEST_GENERATOR_MODEL=…] cargo test -p magi-app generates_from -- --ignored --nocapture`
    #[test]
    #[ignore = "needs an OpenAI-compatible endpoint (MAGI_TEST_GENERATOR_URL)"]
    fn generates_from_a_real_endpoint() {
        let Some(url) = std::env::var_os("MAGI_TEST_GENERATOR_URL") else { return };
        let settings = GeneratorSettings {
            url: url.to_string_lossy().into_owned(),
            model: std::env::var("MAGI_TEST_GENERATOR_MODEL").unwrap_or_default(),
            key: String::new(),
            thinking: std::env::var("MAGI_TEST_GENERATOR_THINKING").unwrap_or_else(|_| "none".into()),
        };
        println!("models: {:?}", models(&settings));
        let mut spec = TextSpec { tools: magi_core::nodes::studio::example_tools(), ..TextSpec::default() };
        spec.ensure_specials();
        for (kind, prompt) in
            [(Kind::Answers, "everyday cooking questions"), (Kind::ToolUse, "the weather and the time in cities")]
        {
            let request = Request {
                settings: settings.clone(),
                prompt: prompt.into(),
                count: 3,
                kind,
                spec: spec.clone(),
                existing: Vec::new(),
            };
            let job = start(request, || {});
            loop {
                std::thread::sleep(Duration::from_millis(500));
                let progress = job.progress.lock().unwrap();
                if progress.done {
                    println!(
                        "{kind:?}: {} made, {} left out, error {:?}",
                        progress.total, progress.rejected, progress.error
                    );
                    for sample in &progress.made {
                        println!("  {sample}");
                    }
                    assert!(progress.error.is_none() && progress.total == 3);
                    break;
                }
            }
        }
    }

    #[test]
    fn a_models_lines_are_read_as_checked_samples() {
        let spec = TextSpec::default();
        let answer = "```jsonl\n1. {\"prompt\": \"Hi\", \"response\": \"Hello!\"}\n{\"prompt\": \"\", \"response\": \"\"}\nnot json\n{\"messages\": [{\"role\": \"user\", \"content\": \"a\"}, {\"role\": \"assistant\", \"content\": \"b\"}]}\n```";
        let (good, bad) = samples_in(answer, &spec);
        assert_eq!(good.len(), 2, "{good:?}");
        assert_eq!(bad, 2);
        assert!(good[0].starts_with("{\"prompt\": \"Hi\""));
    }
}
