//! Native tools: the tools a Magi model runs itself.
//!
//! A model's `nt` folder holds one folder per tool – its `tool.json` (name,
//! description, arguments as a JSON Schema, how to run it) and its program,
//! usually `run.py`, which reads the arguments as JSON on standard input and
//! prints its result as JSON. When the model calls one, the program runs on
//! this computer and its result goes back to the model in a tool turn, so a
//! tiny model only has to learn *which* tool to call and *how*, not the work.
//!
//! MacroMagi comes with a set of them (`src/nt`): MagiDB (db_search,
//! db_read), web_search and fetch_page, files (list_files, read_file,
//! search_files, write_file, edit_file), run_command, documents
//! (open_document, edit_document), view_image, calculator, get_time,
//! get_weather, transcribe_audio, translate and wiki_search. Tools that change
//! files or run programs ask the user first (`confirm`). A model's own files under
//! `nt/<tool>/` in its `.magic` data take the place of the built-in ones.

use std::collections::BTreeMap;
use std::path::Path;

use include_dir::{Dir, include_dir};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::magic::ToolSpec;

static BUILT_IN: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/src/nt");

/// How a native tool runs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RunSpec {
    /// `python` (its entry with the Python for AI), `command` (its entry as a program) or `host`
    /// (carried out by the program running the model, which knows it by its entry's name).
    pub kind: String,
    pub entry: String,
    pub timeout_ms: u64,
}

impl Default for RunSpec {
    fn default() -> Self {
        Self { kind: "python".into(), entry: "run.py".into(), timeout_ms: 60_000 }
    }
}

/// A native tool's `tool.json`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NativeTool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub run: RunSpec,
    /// It changes files or runs programs: the user is asked before each call.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub confirm: bool,
}

impl NativeTool {
    /// The tool as the model's tools list it.
    pub fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
            native: true,
        }
    }
}

/// The native tools MacroMagi comes with.
pub fn built_in() -> Vec<NativeTool> {
    let mut tools: Vec<NativeTool> = BUILT_IN
        .dirs()
        .filter_map(|dir| dir.get_file(dir.path().join("tool.json")))
        .filter_map(|file| serde_json::from_slice(file.contents()).ok())
        .collect();
    tools.sort_by(|a, b| a.name.cmp(&b.name));
    tools
}

/// A native tool's files, by path inside its folder: the model's own
/// (`nt/<name>/…` in its data), else the built-in ones.
pub fn files(name: &str, data: &BTreeMap<String, Vec<u8>>) -> Vec<(String, Vec<u8>)> {
    let prefix = format!("nt/{name}/");
    let own: Vec<(String, Vec<u8>)> = data
        .iter()
        .filter_map(|(path, bytes)| path.strip_prefix(&prefix).map(|rest| (rest.to_owned(), bytes.clone())))
        .collect();
    if !own.is_empty() {
        return own;
    }
    BUILT_IN
        .get_dir(name)
        .map(|dir| {
            dir.files()
                .map(|file| {
                    let rest = file.path().strip_prefix(name).unwrap_or(file.path());
                    (rest.to_string_lossy().replace('\\', "/"), file.contents().to_vec())
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The `tool.json` of a native tool the model has.
pub fn tool(name: &str, data: &BTreeMap<String, Vec<u8>>) -> Option<NativeTool> {
    let files = files(name, data);
    let json = files.iter().find(|(path, _)| path == "tool.json")?;
    serde_json::from_slice(&json.1).ok()
}

/// Writes the native tools a model lists into `into` (its `nt` folder), with the README.
pub fn write_folder(tools: &[ToolSpec], data: &BTreeMap<String, Vec<u8>>, into: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(into)?;
    if let Some(readme) = BUILT_IN.get_file("README.md") {
        std::fs::write(into.join("README.md"), readme.contents())?;
    }
    for tool in tools.iter().filter(|t| t.native) {
        for (path, bytes) in files(&tool.name, data) {
            let target = into.join(&tool.name).join(&path);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(target, bytes)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_built_in_tools_are_complete() {
        let tools = built_in();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        for wanted in ["calculator", "web_search", "get_weather", "get_time", "view_image", "transcribe_audio"] {
            assert!(names.contains(&wanted), "{wanted}");
        }
        for tool in &tools {
            assert!(tool.parameters["type"] == "object", "{}", tool.name);
            let files = files(&tool.name, &BTreeMap::new());
            // A tool the program carries out itself has no program of its own.
            assert!(tool.run.kind == "host" || files.iter().any(|(p, _)| *p == tool.run.entry), "{}", tool.name);
            assert!(tool.spec().native);
        }
        // A model's own copy wins.
        let mut data = BTreeMap::new();
        data.insert("nt/calculator/tool.json".to_owned(), br#"{"name": "calculator", "description": "mine"}"#.to_vec());
        assert_eq!(tool("calculator", &data).unwrap().description, "mine");
    }
}
