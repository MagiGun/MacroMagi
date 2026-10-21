//! Editing a dataset in bulk: find and replace (plain or a regular expression)
//! in chosen parts of every sample, deleting samples that match or are too
//! long, setting their reasoning effort and tools, taking out their reasoning,
//! trimming and removing repeats – each as one change that can be previewed,
//! applied to one file or to all of them, and taken back.
//!
//! A sample of a `.jsonl` file is edited through its JSON, part by part, so an
//! edit never breaks it; plain text files (`.md`, `.txt`) are edited as they are.

use std::collections::HashSet;

use regex::{Regex, RegexBuilder};
use serde_json::{Map, Value};

use crate::dataset;

/// Which parts of a sample an edit reads and changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// Every text the model reads or writes.
    All,
    /// What the user asks: prompts, user messages, the context to write short, system text.
    Requests,
    /// What the model answers: responses, assistant messages, compacted text, answers not to give.
    Answers,
    Thinking,
    /// The arguments of tool calls (their text values) and the tools' names.
    ToolCalls,
    /// What tools gave back.
    ToolResults,
    /// The sample's line as it is written (JSON for `.jsonl`): anything, but checked to stay a sample.
    Raw,
}

impl Part {
    pub const ALL: [(Part, &'static str); 7] = [
        (Part::All, "Any text"),
        (Part::Requests, "Requests"),
        (Part::Answers, "Answers"),
        (Part::Thinking, "Thinking"),
        (Part::ToolCalls, "Tool calls"),
        (Part::ToolResults, "Tool results"),
        (Part::Raw, "Whole line"),
    ];
}

/// What to match: text, or a regular expression.
#[derive(Clone, Debug, PartialEq)]
pub struct Pattern {
    pub text: String,
    pub regex: bool,
    pub case_sensitive: bool,
    pub whole_words: bool,
}

impl Pattern {
    fn compile(&self) -> Result<Regex, String> {
        if self.text.is_empty() {
            return Err("Write what to look for.".into());
        }
        let mut source = if self.regex { self.text.clone() } else { regex::escape(&self.text) };
        if self.whole_words {
            source = format!(r"\b(?:{source})\b");
        }
        RegexBuilder::new(&source)
            .case_insensitive(!self.case_sensitive)
            .build()
            .map_err(|e| format!("Not a regular expression: {e}"))
    }
}

/// One edit, applied to every sample it reaches.
#[derive(Clone, Debug, PartialEq)]
pub enum Edit {
    /// Replace what matches (`$1`… are a regular expression's groups).
    Replace { pattern: Pattern, with: String, part: Part },
    /// Delete the samples that match – or, `keep`, those that do not.
    Delete { pattern: Pattern, part: Part, keep: bool },
    /// Delete samples longer than this many characters (their text the model reads and writes).
    DeleteLonger { characters: usize },
    /// Set the reasoning effort (`None`: the model's default, the field taken out).
    SetEffort(Option<String>),
    /// List the model's own tools (`"tools"` taken out), or none (`"tools": []`).
    SetTools { none: bool },
    /// Take out the reasoning, keeping the answers.
    StripThinking,
    /// Trim spaces at the ends of every text, and the lines of plain text.
    Trim,
    /// Delete samples that repeat an earlier one exactly.
    Dedupe,
}

/// What an edit did, or would do.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub changed: usize,
    pub deleted: usize,
    /// Samples a whole-line edit would have broken: left as they were.
    pub skipped: usize,
}

impl Report {
    pub fn add(&mut self, other: Report) {
        self.changed += other.changed;
        self.deleted += other.deleted;
        self.skipped += other.skipped;
    }

    pub fn is_empty(&self) -> bool {
        self.changed == 0 && self.deleted == 0
    }
}

/// Applies `edit` to the samples (`only`: just those, by index), and says what it did.
pub fn apply(
    samples: &mut Vec<String>,
    jsonl: bool,
    edit: &Edit,
    only: Option<&HashSet<usize>>,
) -> Result<Report, String> {
    let reached = |index: usize| only.is_none_or(|set| set.contains(&index));
    let mut report = Report::default();
    match edit {
        Edit::Replace { pattern, with, part } => {
            let regex = pattern.compile()?;
            for (_, sample) in samples.iter_mut().enumerate().filter(|(i, _)| reached(*i)) {
                match replaced(sample, jsonl, &regex, with, *part) {
                    Some(Ok(new)) => {
                        *sample = new;
                        report.changed += 1;
                    }
                    Some(Err(())) => report.skipped += 1,
                    None => {}
                }
            }
        }
        Edit::Delete { pattern, part, keep } => {
            let regex = pattern.compile()?;
            let before = samples.len();
            let mut index = 0;
            samples.retain(|sample| {
                let here = index;
                index += 1;
                !reached(here) || matches(sample, jsonl, &regex, *part) == *keep
            });
            report.deleted = before - samples.len();
        }
        Edit::DeleteLonger { characters } => {
            let before = samples.len();
            let mut index = 0;
            samples.retain(|sample| {
                let here = index;
                index += 1;
                !reached(here)
                    || dataset::texts(sample, jsonl).iter().map(|t| t.chars().count()).sum::<usize>() <= *characters
            });
            report.deleted = before - samples.len();
        }
        Edit::Dedupe => {
            let mut seen = HashSet::new();
            let before = samples.len();
            let mut index = 0;
            samples.retain(|sample| {
                let here = index;
                index += 1;
                let key = normalized(sample, jsonl);
                !reached(here) || seen.insert(key)
            });
            report.deleted = before - samples.len();
        }
        Edit::SetEffort(_) | Edit::SetTools { .. } | Edit::StripThinking | Edit::Trim => {
            for (index, sample) in samples.iter_mut().enumerate() {
                if !reached(index) {
                    continue;
                }
                let new = if jsonl {
                    edited_object(sample, edit)
                } else {
                    (edit == &Edit::Trim).then(|| trimmed_text(sample))
                };
                if let Some(new) = new.filter(|n| n != sample) {
                    *sample = new;
                    report.changed += 1;
                }
            }
        }
    }
    Ok(report)
}

/// The sample as compared for repeats: its JSON written one way, or its text trimmed.
fn normalized(sample: &str, jsonl: bool) -> String {
    if jsonl && let Ok(Value::Object(item)) = serde_json::from_str::<Value>(sample) {
        return dataset::to_line(&item);
    }
    sample.trim().to_owned()
}

fn trimmed_text(text: &str) -> String {
    text.lines().map(str::trim_end).collect::<Vec<_>>().join("\n").trim().to_owned()
}

/// A JSON sample with an edit of its fields; `None` when it is not one.
fn edited_object(sample: &str, edit: &Edit) -> Option<String> {
    let Ok(Value::Object(mut item)) = serde_json::from_str::<Value>(sample) else { return None };
    match edit {
        Edit::SetEffort(Some(level)) => {
            item.insert("effort".into(), Value::from(level.clone()));
        }
        Edit::SetEffort(None) => {
            item.remove("effort");
        }
        Edit::SetTools { none: true } => {
            item.insert("tools".into(), Value::Array(Vec::new()));
        }
        Edit::SetTools { none: false } => {
            item.remove("tools");
        }
        Edit::StripThinking => {
            item.remove("thinking");
            if let Some(Value::Array(messages)) = item.get_mut("messages") {
                for message in messages.iter_mut().filter_map(Value::as_object_mut) {
                    message.remove("thinking");
                }
            }
        }
        Edit::Trim => visit(&mut item, Part::All, &mut |text| {
            let trimmed = text.trim().to_owned();
            (trimmed != *text).then_some(trimmed)
        }),
        _ => {}
    }
    Some(dataset::to_line(&item))
}

/// Whether some text of the sample's `part` matches.
fn matches(sample: &str, jsonl: bool, regex: &Regex, part: Part) -> bool {
    if !jsonl || part == Part::Raw {
        return regex.is_match(sample);
    }
    let Ok(Value::Object(mut item)) = serde_json::from_str::<Value>(sample) else { return regex.is_match(sample) };
    let mut found = false;
    visit(&mut item, part, &mut |text| {
        found |= regex.is_match(text);
        None
    });
    found
}

/// The sample with its `part` replaced: `None` when nothing matched, an error when a whole-line
/// replacement would no longer be a sample.
fn replaced(sample: &str, jsonl: bool, regex: &Regex, with: &str, part: Part) -> Option<Result<String, ()>> {
    if !jsonl || part == Part::Raw {
        if !regex.is_match(sample) {
            return None;
        }
        let new = regex.replace_all(sample, with).into_owned();
        if jsonl && !matches!(serde_json::from_str::<Value>(&new), Ok(Value::Object(_))) {
            return Some(Err(()));
        }
        return Some(Ok(new));
    }
    let Ok(Value::Object(mut item)) = serde_json::from_str::<Value>(sample) else { return None };
    let mut changed = false;
    visit(&mut item, part, &mut |text| {
        if regex.is_match(text) {
            changed = true;
            Some(regex.replace_all(text, with).into_owned())
        } else {
            None
        }
    });
    changed.then(|| Ok(dataset::to_line(&item)))
}

/// Calls `change` on each text of the sample's `part`; what it gives back takes its place.
fn visit(item: &mut Map<String, Value>, part: Part, change: &mut dyn FnMut(&String) -> Option<String>) {
    let wants = |p: Part| part == Part::All || part == p;
    let field = |item: &mut Map<String, Value>, key: &str, change: &mut dyn FnMut(&String) -> Option<String>| {
        if let Some(Value::String(text)) = item.get_mut(key)
            && let Some(new) = change(text)
        {
            *text = new;
        }
    };
    if wants(Part::Requests) {
        for key in ["prompt", "context", "system"] {
            field(item, key, change);
        }
    }
    if wants(Part::Answers) {
        for key in ["response", "compact", "rejected", "bad"] {
            field(item, key, change);
        }
    }
    if wants(Part::Thinking) {
        field(item, "thinking", change);
    }
    if part == Part::All {
        field(item, "text", change);
    }
    let Some(Value::Array(messages)) = item.get_mut("messages") else { return };
    for message in messages.iter_mut().filter_map(Value::as_object_mut) {
        let role = message.get("role").and_then(Value::as_str).unwrap_or("user").to_owned();
        let reads = match role.as_str() {
            "user" | "system" => wants(Part::Requests),
            "assistant" => wants(Part::Answers),
            "tool" => wants(Part::ToolResults),
            _ => false,
        };
        if reads {
            field(message, "content", change);
        }
        if wants(Part::Thinking) {
            field(message, "thinking", change);
        }
        if wants(Part::ToolCalls)
            && let Some(Value::Array(calls)) = message.get_mut("tool_calls")
        {
            for call in calls.iter_mut().filter_map(Value::as_object_mut) {
                let call = match call.get_mut("function") {
                    Some(Value::Object(_)) => call.get_mut("function").and_then(Value::as_object_mut).expect("checked"),
                    _ => call,
                };
                field(call, "name", change);
                if let Some(arguments) = call.get_mut("arguments") {
                    strings(arguments, change);
                }
            }
        }
    }
}

/// Every text inside a JSON value (a tool call's arguments).
fn strings(value: &mut Value, change: &mut dyn FnMut(&String) -> Option<String>) {
    match value {
        Value::String(text) => {
            if let Some(new) = change(text) {
                *text = new;
            }
        }
        Value::Array(items) => items.iter_mut().for_each(|v| strings(v, change)),
        Value::Object(map) => map.values_mut().for_each(|v| strings(v, change)),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(text: &str, regex: bool) -> Pattern {
        Pattern { text: text.into(), regex, case_sensitive: false, whole_words: false }
    }

    fn lines(samples: &[&str]) -> Vec<String> {
        samples.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn replacing_in_answers_leaves_requests_and_keeps_samples_whole() {
        let mut samples = lines(&[
            r#"{"prompt": "Who is Open Assistant?", "response": "I am Open Assistant, made by LAION."}"#,
            r#"{"messages": [{"role": "user", "content": "hi Open Assistant"}, {"role": "assistant", "content": "Hi! I'm Open Assistant."}]}"#,
        ]);
        let edit =
            Edit::Replace { pattern: pattern("open assistant", false), with: "Magi".into(), part: Part::Answers };
        let report = apply(&mut samples, true, &edit, None).unwrap();
        assert_eq!(report.changed, 2);
        assert!(samples[0].contains("Who is Open Assistant?") && samples[0].contains("I am Magi, made by LAION."));
        assert!(samples[1].contains("hi Open Assistant") && samples[1].contains("Hi! I'm Magi."));
        // Regular expressions, with groups.
        let edit = Edit::Replace {
            pattern: pattern(r"made by (\w+)", true),
            with: "made in MacroMagi (not $1)".into(),
            part: Part::All,
        };
        apply(&mut samples, true, &edit, None).unwrap();
        assert!(samples[0].contains("made in MacroMagi (not LAION)"));
    }

    #[test]
    fn a_whole_line_edit_that_would_break_a_sample_is_skipped() {
        let mut samples = lines(&[r#"{"prompt": "a", "response": "b"}"#]);
        let edit = Edit::Replace { pattern: pattern("}", false), with: "".into(), part: Part::Raw };
        let report = apply(&mut samples, true, &edit, None).unwrap();
        assert_eq!((report.changed, report.skipped), (0, 1));
        assert_eq!(samples[0], r#"{"prompt": "a", "response": "b"}"#);
    }

    #[test]
    fn deleting_setting_and_removing_repeats() {
        let mut samples = lines(&[
            r#"{"prompt": "hi", "response": "hello", "thinking": "A greeting."}"#,
            r#"{"prompt": "hi",  "response": "hello", "thinking": "A greeting."}"#,
            r#"{"prompt": "What is 2+2?", "response": "4."}"#,
            r#"{"prompt": "Tell me a long story", "response": "Once upon a time there was a very long story that goes on."}"#,
        ]);
        assert_eq!(
            apply(&mut samples, true, &Edit::Dedupe, None).unwrap().deleted,
            1,
            "the same sample, spaced otherwise"
        );
        assert_eq!(apply(&mut samples, true, &Edit::DeleteLonger { characters: 40 }, None).unwrap().deleted, 1);
        let only: HashSet<usize> = [1].into();
        let report = apply(&mut samples, true, &Edit::SetEffort(Some("superior".into())), Some(&only)).unwrap();
        assert_eq!(report.changed, 1);
        assert!(samples[1].contains("\"effort\": \"superior\"") && !samples[0].contains("effort"));
        apply(&mut samples, true, &Edit::StripThinking, None).unwrap();
        assert!(!samples[0].contains("thinking"));
        apply(&mut samples, true, &Edit::SetTools { none: true }, None).unwrap();
        assert!(samples.iter().all(|s| s.contains("\"tools\": []")));
        let keep = Edit::Delete { pattern: pattern(r"\d", true), part: Part::Requests, keep: true };
        assert_eq!(apply(&mut samples, true, &keep, None).unwrap().deleted, 1);
        assert_eq!(samples.len(), 1);
    }

    #[test]
    fn tool_calls_and_plain_text() {
        let mut samples = lines(&[
            r#"{"messages": [{"role": "user", "content": "weather in Paris"}, {"role": "assistant", "content": "", "tool_calls": [{"name": "weather", "arguments": {"city": "Paris"}}]}, {"role": "tool", "content": "{\"c\": 14}"}, {"role": "assistant", "content": "14 °C."}]}"#,
        ]);
        let edit =
            Edit::Replace { pattern: pattern("^weather$", true), with: "get_weather".into(), part: Part::ToolCalls };
        assert_eq!(apply(&mut samples, true, &edit, None).unwrap().changed, 1);
        assert!(samples[0].contains("\"name\": \"get_weather\"") && samples[0].contains("weather in Paris"));
        let mut text = lines(&["Some   text.  \n  more  ", "Other words."]);
        apply(&mut text, false, &Edit::Trim, None).unwrap();
        assert_eq!(text[0], "Some   text.\n  more");
        let edit = Edit::Replace {
            pattern: Pattern { whole_words: true, ..pattern("words", false) },
            with: "text".into(),
            part: Part::All,
        };
        apply(&mut text, false, &edit, None).unwrap();
        assert_eq!(text[1], "Other text.");
    }
}
