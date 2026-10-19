//! Blocks that work with files and folders: reading text or pictures,
//! writing them, searching text, listing and making folders.
//!
//! Paths are the user's own: a full path, or one relative to the `files`
//! folder in MacroMagi's data folder.

use serde::{Deserialize, Serialize};

use super::{Category, DataType, NodeDef, ParamDef, ParamKind, Registry, ids, ports};

/// What a Read File block reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadMode {
    /// The whole text.
    #[default]
    Text,
    /// One line of the text.
    Line,
    /// A picture (PNG, JPEG, BMP, GIF, WebP).
    Picture,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ReadFileParams {
    pub path: String,
    pub mode: ReadMode,
    /// The line to read, counting from 1.
    pub line: u64,
}

impl Default for ReadFileParams {
    fn default() -> Self {
        Self { path: String::new(), mode: ReadMode::Text, line: 1 }
    }
}

/// How a Write to File block writes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteMode {
    /// Replace what the file holds.
    #[default]
    Replace,
    /// Add to the end.
    Append,
    /// Change one line.
    Line,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WriteFileParams {
    pub path: String,
    pub mode: WriteMode,
    /// The line to change, counting from 1.
    pub line: u64,
    /// What to write, when nothing is wired into "Value".
    pub text: String,
    /// When adding, start on a new line.
    pub new_line: bool,
}

impl Default for WriteFileParams {
    fn default() -> Self {
        Self { path: String::new(), mode: WriteMode::Replace, line: 1, text: String::new(), new_line: true }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FindTextParams {
    /// What to look for.
    pub find: String,
    pub match_case: bool,
}

/// What a Folder Contents block lists.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ListKind {
    #[default]
    Files,
    Folders,
    Both,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ListFolderParams {
    pub path: String,
    /// Only names containing this (or ending in it, like `.png`); empty for all.
    pub filter: String,
    pub kind: ListKind,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MakeFolderParams {
    pub path: String,
}

fn path_param(folder: bool) -> ParamDef {
    ParamDef::new("path", if folder { "Folder" } else { "File" }, ParamKind::Path { folder }).help(
        "A full path, or one inside the files folder in MacroMagi's data folder. A path wired in is used instead.",
    )
}

fn short(path: &str) -> String {
    let path = path.trim();
    if path.is_empty() {
        return "No file chosen".to_owned();
    }
    let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
    name.to_owned()
}

pub fn register(registry: &mut Registry) {
    registry.register(
        NodeDef::new(
            ids::READ_FILE,
            "Read File",
            Category::Files,
            "Reads a file: all its text, one line of it, or a picture. A picture can be wired into an \
             Image Detection block to look for it on screen, or into Write to File.",
        )
        .keywords(&["file", "read", "open", "load", "text", "line", "picture", "image", "csv", "txt"])
        .data_in("path", "Path", DataType::Text)
        .data_out("found", "Exists", DataType::Bool)
        .data_out("text", "Text", DataType::Text)
        .data_out("lines", "Line count", DataType::Number)
        .data_out("picture", "Picture", DataType::Image)
        .param(path_param(false))
        .param(ParamDef::new(
            "mode",
            "Read",
            ParamKind::choice(&[("text", "All the text"), ("line", "One line"), ("picture", "A picture")]),
        ))
        .param(ParamDef::new("line", "Line", ParamKind::int(1, 100_000_000, "")).show_if("mode", "line"))
        .defaults(&ReadFileParams::default())
        .summarize(|node, _| {
            let p: ReadFileParams = node.params_or_default();
            let what = match p.mode {
                ReadMode::Text => "All the text".to_owned(),
                ReadMode::Line => format!("Line {}", p.line),
                ReadMode::Picture => "A picture".to_owned(),
            };
            vec![short(&p.path), what]
        }),
    );

    registry.register(
        NodeDef::new(
            ids::WRITE_FILE,
            "Write to File",
            Category::Files,
            "Writes to a file: replaces what it holds, adds to its end, or changes one line. What is wired \
             into “Value” is written instead of the text setting; a picture (from Screenshot or Read File) \
             is saved as a PNG file, replacing one of the same name. Missing folders are made.",
        )
        .keywords(&["file", "write", "save", "append", "line", "log", "picture", "screenshot", "png"])
        .flow_in()
        .flow_out(ports::OUT, "")
        .data_in("value", "Value", DataType::Any)
        .data_in("path", "Path", DataType::Text)
        .param(path_param(false))
        .param(ParamDef::new(
            "mode",
            "Write",
            ParamKind::choice(&[
                ("replace", "Set the file to it"),
                ("append", "Add it to the end"),
                ("line", "Change one line"),
            ]),
        ))
        .param(ParamDef::new("line", "Line", ParamKind::int(1, 100_000_000, "")).show_if("mode", "line"))
        .param(ParamDef::new("new_line", "On a new line", ParamKind::Bool).show_if("mode", "append"))
        .param(ParamDef::new("text", "Text", ParamKind::Text { multiline: true }))
        .defaults(&WriteFileParams::default())
        .summarize(|node, _| {
            let p: WriteFileParams = node.params_or_default();
            let how = match p.mode {
                WriteMode::Replace => "Sets it".to_owned(),
                WriteMode::Append => "Adds to it".to_owned(),
                WriteMode::Line => format!("Changes line {}", p.line),
            };
            vec![short(&p.path), how]
        }),
    );

    registry.register(
        NodeDef::new(
            ids::FIND_TEXT,
            "Find in Text",
            Category::Files,
            "Looks for a word or phrase in the text wired in – say, a file from Read File – and says on \
             which line it first is, what that line says, and on how many lines it is.",
        )
        .keywords(&["find", "search", "text", "line", "keyword", "contains", "grep"])
        .data_in("text", "Text", DataType::Text)
        .data_out("found", "Found", DataType::Bool)
        .data_out("line", "Line number", DataType::Number)
        .data_out("line_text", "The line", DataType::Text)
        .data_out("count", "Lines with it", DataType::Number)
        .param(ParamDef::new("find", "Look for", ParamKind::Text { multiline: false }))
        .param(ParamDef::new("match_case", "Match upper and lower case", ParamKind::Bool))
        .defaults(&FindTextParams::default())
        .summarize(|node, _| {
            let p: FindTextParams = node.params_or_default();
            vec![if p.find.is_empty() { "Nothing to look for".to_owned() } else { format!("“{}”", p.find) }]
        }),
    );

    registry.register(
        NodeDef::new(
            ids::LIST_FOLDER,
            "Folder Contents",
            Category::Files,
            "Lists the files (or folders) in a folder, one name per line, how many there are, and the full \
             path of the first.",
        )
        .keywords(&["folder", "directory", "list", "files", "dir", "ls"])
        .data_in("path", "Path", DataType::Text)
        .data_out("found", "Exists", DataType::Bool)
        .data_out("names", "Names", DataType::Text)
        .data_out("count", "Count", DataType::Number)
        .data_out("first", "First", DataType::Text)
        .param(path_param(true))
        .param(
            ParamDef::new("filter", "Only names with", ParamKind::Text { multiline: false })
                .help("For example .png. Empty lists everything."),
        )
        .param(ParamDef::new(
            "kind",
            "List",
            ParamKind::choice(&[("files", "Files"), ("folders", "Folders"), ("both", "Files and folders")]),
        ))
        .defaults(&ListFolderParams::default())
        .summarize(|node, _| {
            let p: ListFolderParams = node.params_or_default();
            vec![short(&p.path)]
        }),
    );

    registry.register(
        NodeDef::new(ids::MAKE_FOLDER, "Make Folder", Category::Files, "Makes a folder, and any folders it is in.")
            .keywords(&["folder", "directory", "make", "create", "mkdir"])
            .flow_in()
            .flow_out(ports::OUT, "")
            .data_in("path", "Path", DataType::Text)
            .param(path_param(true))
            .defaults(&MakeFolderParams::default())
            .summarize(|node, _| {
                let p: MakeFolderParams = node.params_or_default();
                vec![short(&p.path)]
            }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_file_blocks_have_complete_defaults() {
        let registry = Registry::builtin();
        for id in [ids::READ_FILE, ids::WRITE_FILE, ids::FIND_TEXT, ids::LIST_FOLDER, ids::MAKE_FOLDER] {
            let def = registry.get(id).unwrap();
            assert_eq!(def.category, Category::Files);
            assert!(!def.default_params().is_empty(), "{id}");
        }
    }
}
