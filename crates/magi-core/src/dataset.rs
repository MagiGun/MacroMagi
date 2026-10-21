//! Text dataset samples beyond their format: the reasoning efforts a sample
//! is written at, how an edited sample is written back as a JSON line, and
//! checks across a model's whole training set (efforts without samples,
//! words fine-tuning uses that pretraining never does).

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};

use crate::magic::{EffortSpec, TextSpec, ToolSpec, spaced_json};

/// The reasoning efforts a sample is written at: none (it counts as the
/// model's default level), one (`"effort": "superior"`) or several
/// (`"effort": ["standard", "superior"]`, learned at each).
pub fn efforts(item: &Map<String, Value>) -> Vec<String> {
    let mut levels: Vec<String> = Vec::new();
    let mut add = |level: &str| {
        let level = level.trim();
        if !level.is_empty() && !levels.iter().any(|l| l == level) {
            levels.push(level.to_owned());
        }
    };
    match item.get("effort") {
        Some(Value::String(level)) => add(level),
        Some(Value::Array(list)) => list.iter().filter_map(Value::as_str).for_each(add),
        _ => {}
    }
    levels
}

/// Sets the efforts of a sample: none removes `"effort"` (the default
/// level), one is written as a name and several as a list.
pub fn set_efforts(item: &mut Map<String, Value>, levels: &[String]) {
    match levels {
        [] => {
            item.remove("effort");
        }
        [one] => {
            item.insert("effort".into(), Value::String(one.clone()));
        }
        several => {
            item.insert("effort".into(), Value::Array(several.iter().cloned().map(Value::String).collect()));
        }
    }
}

/// Whether a sample is a request or a conversation – what reasoning effort
/// applies to – rather than plain text or a compaction.
pub fn is_request(item: &Map<String, Value>) -> bool {
    item.contains_key("prompt") || item.contains_key("messages")
}

/// The levels a sample is learned at, each a level of `spec`: its own, or
/// the default when it names none (or none of the model's).
pub fn learned_at(item: &Map<String, Value>, spec: &EffortSpec) -> Vec<String> {
    let own: Vec<String> = efforts(item).into_iter().filter(|l| spec.levels.contains(l)).collect();
    if own.is_empty() { vec![spec.default.clone()] } else { own }
}

/// How many samples each level of `spec` is learned from, across dataset
/// files (`(name, content)`), in the order of the levels.
pub fn effort_counts<'a>(
    spec: &EffortSpec,
    files: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Vec<(String, usize)> {
    let mut counts: BTreeMap<String, usize> = spec.levels.iter().map(|l| (l.clone(), 0)).collect();
    for (name, content) in files {
        if !crate::nodes::studio::is_jsonl(name) {
            continue;
        }
        for line in content.lines() {
            if let Ok(Value::Object(item)) = serde_json::from_str::<Value>(line.trim())
                && is_request(&item)
            {
                for level in learned_at(&item, spec) {
                    *counts.entry(level).or_default() += 1;
                }
            }
        }
    }
    spec.levels.iter().map(|l| (l.clone(), counts.get(l).copied().unwrap_or_default())).collect()
}

/// The level whose samples a level without any is also learned from, as the
/// trainer does: the nearest level that has samples, the lower one first.
/// `None` when the level has samples of its own, or no level has any.
pub fn donor(counts: &[(String, usize)], level: &str) -> Option<String> {
    let index = counts.iter().position(|(l, _)| l == level)?;
    if counts[index].1 > 0 {
        return None;
    }
    (1..counts.len())
        .flat_map(|distance| [index.checked_sub(distance), Some(index + distance)])
        .flatten()
        .find_map(|i| counts.get(i).filter(|(_, n)| *n > 0).map(|(l, _)| l.clone()))
}

/// The order a sample's fields are written in: what it is set up with, then
/// the request and its answer.
const FIELD_ORDER: [&str; 13] = [
    "system", "tools", "effort", "prompt", "thinking", "response", "rejected", "bad", "messages", "context", "compact",
    "text", "label",
];

/// A sample as one JSON line, its fields in a readable order (the rest after
/// them, alphabetically), spaced as the example datasets are.
pub fn to_line(item: &Map<String, Value>) -> String {
    let mut keys: Vec<&String> = item.keys().collect();
    keys.sort_by_key(|k| (FIELD_ORDER.iter().position(|f| f == k).unwrap_or(FIELD_ORDER.len()), (*k).clone()));
    let fields: Vec<String> =
        keys.iter().map(|k| format!("{}: {}", Value::String((*k).clone()), spaced_json(&item[*k]))).collect();
    format!("{{{}}}", fields.join(", "))
}

/// A request or conversation as training writes it out for the model, chat
/// tags and all (`runtime/python/magi_train/data.py`, with the reasoning
/// kept – when the model has a Thinking block; without one, reasoning is not
/// learned): the system text with the effort and the sample's tools, the
/// turns, tool calls and tool results. `None` for other samples.
pub fn as_trained(spec: &TextSpec, item: &Map<String, Value>, effort: &str) -> Option<String> {
    let string = |value: Option<&Value>| match value {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    };
    let mut messages: Vec<Map<String, Value>> = match (item.get("messages"), item.get("prompt")) {
        (Some(Value::Array(list)), _) => list.iter().filter_map(|m| m.as_object().cloned()).collect(),
        (_, Some(prompt)) => {
            let mut user = Map::new();
            user.insert("role".into(), "user".into());
            user.insert("content".into(), Value::String(string(Some(prompt))));
            let mut answer = Map::new();
            answer.insert("role".into(), "assistant".into());
            let response = item.get("response").or_else(|| item.get("rejected")).or_else(|| item.get("bad"));
            answer.insert("content".into(), Value::String(string(response)));
            if let Some(thinking) = item.get("thinking") {
                answer.insert("thinking".into(), thinking.clone());
            }
            vec![user, answer]
        }
        _ => return None,
    };
    let mut system = string(item.get("system"));
    if messages.first().is_some_and(|m| m.get("role").and_then(Value::as_str) == Some("system")) {
        system = string(messages.remove(0).get("content"));
    }
    // Without "tools", the model's own (as when it runs); "tools": [] lists none.
    let tools: Vec<ToolSpec> = match item.get("tools") {
        Some(Value::Array(list)) => list
            .iter()
            .filter_map(|tool| serde_json::from_value(tool.get("function").unwrap_or(tool).clone()).ok())
            .collect(),
        _ => spec.tools.clone(),
    };
    let mut out = String::new();
    let head = spec.system_text(&system, effort, &tools);
    if !head.is_empty() {
        out.push_str(&format!("{}{head}{}", spec.system_tag, spec.end_tag));
    }
    let (think_open, think_close) = &spec.think_tags;
    for message in &messages {
        let content = string(message.get("content"));
        match message.get("role").and_then(Value::as_str).unwrap_or("user") {
            "assistant" => {
                out.push_str(&spec.assistant_tag);
                // Reasoning is learned only with the Thinking block.
                let thinking = if spec.thinking.is_some() { string(message.get("thinking")) } else { String::new() };
                if !thinking.trim().is_empty() {
                    out.push_str(&format!("{think_open}{}{think_close}", thinking.trim()));
                }
                out.push_str(&content);
                if let Some(Value::Array(calls)) = message.get("tool_calls") {
                    for call in calls {
                        let call = call.get("function").unwrap_or(call);
                        let arguments = call.get("arguments").or_else(|| call.get("args")).cloned().unwrap_or_default();
                        let arguments = match arguments {
                            Value::String(text) => serde_json::from_str(&text).unwrap_or(Value::String(text)),
                            other => other,
                        };
                        out.push_str(&format!(
                            "{}{{\"name\": {}, \"arguments\": {}}}{}",
                            spec.tool_tags.0,
                            Value::String(string(call.get("name"))),
                            spaced_json(&arguments),
                            spec.tool_tags.1
                        ));
                    }
                }
            }
            "tool" if !spec.tool_tag.is_empty() => out.push_str(&format!("{}{content}", spec.tool_tag)),
            "tool" => out.push_str(&format!("{}Tool result: {content}", spec.user_tag)),
            "system" => out.push_str(&format!("{}{content}", spec.system_tag)),
            _ => out.push_str(&format!("{}{content}", spec.user_tag)),
        }
        out.push_str(&spec.end_tag);
    }
    Some(out)
}

/// The text of a sample a model reads or writes – requests, answers,
/// reasoning, tool results – without its keys, tools or tool calls.
pub fn texts(sample: &str, jsonl: bool) -> Vec<String> {
    if !jsonl {
        return vec![sample.to_owned()];
    }
    let Ok(Value::Object(item)) = serde_json::from_str::<Value>(sample) else { return Vec::new() };
    let mut out = Vec::new();
    let mut take = |value: Option<&Value>| {
        if let Some(Value::String(text)) = value {
            out.push(text.clone());
        }
    };
    for key in ["system", "prompt", "thinking", "response", "rejected", "bad", "context", "compact", "text"] {
        take(item.get(key));
    }
    if let Some(Value::Array(messages)) = item.get("messages") {
        for message in messages {
            take(message.get("content"));
            take(message.get("thinking"));
        }
    }
    out
}

/// The words of a text, lowercase: letters, with apostrophes and hyphens
/// inside them, two letters or more.
pub fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !(c.is_alphabetic() || c == '\'' || c == '-'))
        .map(|w| w.trim_matches(|c: char| c == '\'' || c == '-'))
        .filter(|w| w.chars().filter(|c| c.is_alphabetic()).count() >= 2)
        .map(str::to_lowercase)
}

/// Words the samples use that the pretraining text never does, with how
/// often each comes up, the most frequent first. A model learns its words
/// in pretraining; ones it first meets in fine-tuning it learns from those
/// samples alone.
pub fn unknown_words<'a>(
    pretraining: impl IntoIterator<Item = &'a str>,
    samples: impl IntoIterator<Item = &'a str>,
) -> Vec<(String, usize)> {
    let known: BTreeSet<String> = pretraining.into_iter().flat_map(words).collect();
    let mut unknown: BTreeMap<String, usize> = BTreeMap::new();
    for text in samples {
        for word in words(text) {
            if !known.contains(&word) {
                *unknown.entry(word).or_default() += 1;
            }
        }
    }
    let mut list: Vec<(String, usize)> = unknown.into_iter().collect();
    list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    list
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn item(value: Value) -> Map<String, Value> {
        value.as_object().cloned().unwrap()
    }

    #[test]
    fn efforts_are_one_or_several_and_default_without() {
        let spec = EffortSpec { levels: vec!["standard".into(), "superior".into()], default: "standard".into() };
        let mut sample = item(json!({ "prompt": "Hi", "response": "Hello." }));
        assert_eq!(learned_at(&sample, &spec), ["standard"]);
        set_efforts(&mut sample, &["standard".into(), "superior".into()]);
        assert_eq!(sample["effort"], json!(["standard", "superior"]));
        assert_eq!(learned_at(&sample, &spec), ["standard", "superior"]);
        set_efforts(&mut sample, &["superior".into()]);
        assert_eq!(sample["effort"], json!("superior"));
        assert_eq!(to_line(&sample), r#"{"effort": "superior", "prompt": "Hi", "response": "Hello."}"#);
        set_efforts(&mut sample, &[]);
        assert!(!sample.contains_key("effort"));
    }

    #[test]
    fn a_level_without_samples_learns_from_the_nearest() {
        let spec = EffortSpec {
            levels: vec!["none".into(), "low".into(), "medium".into(), "high".into()],
            default: "none".into(),
        };
        let data = "{\"prompt\": \"a\", \"response\": \"b\"}\n{\"prompt\": \"a\", \"response\": \"b\", \"effort\": [\"none\", \"high\"]}\n{\"text\": \"plain\"}\n";
        let counts = effort_counts(&spec, [("d.jsonl", data), ("p.md", "{\"prompt\": \"x\"}")]);
        assert_eq!(counts, [("none".into(), 2), ("low".into(), 0), ("medium".into(), 0), ("high".into(), 1)]);
        assert_eq!(donor(&counts, "low").as_deref(), Some("none"));
        assert_eq!(donor(&counts, "medium").as_deref(), Some("high"));
        assert_eq!(donor(&counts, "high"), None);
    }

    #[test]
    fn words_fine_tuning_uses_and_pretraining_does_not() {
        let samples = texts(r#"{"prompt": "What's the capital of Peru?", "response": "Lima."}"#, true);
        let unknown =
            unknown_words(["The capital of France is Paris. What is it?"], samples.iter().map(String::as_str));
        let words: Vec<&str> = unknown.iter().map(|(w, _)| w.as_str()).collect();
        assert_eq!(words, ["lima", "peru", "what's"]);
    }
}
