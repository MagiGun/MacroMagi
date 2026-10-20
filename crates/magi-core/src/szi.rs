//! `.szi`: the Magi model format.
//!
//! A Magi model is a folder, so it can hold more than weights: its native
//! tools above all. MacroMagi writes it when it compiles a model, and runs it;
//! CodeMagi will run it too.
//!
//! ```text
//! <Name>.szi            the description (JSON): format, name, architecture,
//!                       how it reads and writes, its tools, its memory settings
//! weights.safetensors   the weights (the native engine runs these)
//! model.onnx            the same model for ONNX runtimes (optional)
//! nt/<tool>/            its native tools: tool.json and the program
//! wiki/                 notes its wiki_search tool reads (optional)
//! README.md             what it is and how to run it
//! ```

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::magic::{MagicFile, MagicModel};

pub const FORMAT: &str = "magi.szi";
pub const VERSION: u32 = 1;
pub const EXTENSION: &str = "szi";
pub const WEIGHTS: &str = "weights.safetensors";
pub const ONNX: &str = "model.onnx";
pub const TOOLS: &str = "nt";

/// The `.szi` file: the model's description and where its files are.
#[derive(Serialize, Deserialize)]
struct Manifest {
    format: String,
    version: u32,
    #[serde(default)]
    app: String,
    /// `Magi-F` (tiny), `Magi-S` (mid-sized) or `Magi-T` (frontier) for Magi models.
    #[serde(default)]
    family: String,
    #[serde(default)]
    files: Value,
    #[serde(flatten)]
    model: MagicModel,
}

/// The family of a Magi model by its size: Magi-F up to 1B parameters, Magi-S up to 30B, Magi-T above.
pub fn family(parameters: u64) -> &'static str {
    match parameters {
        0..=1_000_000_000 => "Magi-F",
        1_000_000_001..=30_000_000_000 => "Magi-S",
        _ => "Magi-T",
    }
}

/// Writes a trained model as a `.szi` folder in `dir`; returns the `.szi` file.
pub fn write(file: &MagicFile, dir: &Path, stem: &str, with_onnx: bool) -> Result<PathBuf> {
    if file.weights.is_empty() && file.onnx.is_empty() {
        return Err(Error::format("the model has not been trained yet"));
    }
    std::fs::create_dir_all(dir)?;
    let mut model = file.model.clone();
    // What builds and trains the model stays in the workspace.
    model.studio = None;
    model.datasets.clear();
    model.training.clear();
    model.finetune.clear();
    model.package = None;
    if !file.weights.is_empty() {
        std::fs::write(dir.join(WEIGHTS), &file.weights)?;
    }
    if with_onnx && !file.onnx.is_empty() {
        std::fs::write(dir.join(ONNX), &file.onnx)?;
    }
    let tools = model.text.as_ref().map(|t| t.tools.clone()).unwrap_or_default();
    if tools.iter().any(|t| t.native) {
        crate::native_tools::write_folder(&tools, &file.data, &dir.join(TOOLS))
            .map_err(|e| Error::format(format!("the native tools could not be written: {e}")))?;
    }
    let parameters = model.trained.as_ref().map_or(0, |t| t.parameters);
    let manifest = Manifest {
        format: FORMAT.into(),
        version: VERSION,
        app: crate::VERSION.into(),
        family: family(parameters).into(),
        files: serde_json::json!({
            "weights": (!file.weights.is_empty()).then_some(WEIGHTS),
            "onnx": (with_onnx && !file.onnx.is_empty()).then_some(ONNX),
            "native_tools": tools.iter().any(|t| t.native).then_some(TOOLS),
        }),
        model,
    };
    let path = dir.join(format!("{stem}.{EXTENSION}"));
    std::fs::write(&path, serde_json::to_vec_pretty(&manifest)?)?;
    let native: Vec<&str> = tools.iter().filter(|t| t.native).map(|t| t.name.as_str()).collect();
    std::fs::write(
        dir.join("README.md"),
        format!(
            "# {name}\n\n{description}\n\nA {family} model in Magi's `.szi` format, made in MacroMagi {app}. Open \
             `{stem}.szi` in MacroMagi (File › Open) to use it, or add it to a macro.\n\n- `weights.safetensors` – \
             the weights, which MacroMagi's native engine runs with its key/value cache.\n- `model.onnx` – the same \
             model for ONNX runtimes, when it was written.\n- `nt/` – its native tools{tools}: programs it calls, \
             run on this computer.\n",
            name = manifest.model.name,
            description = manifest.model.description,
            family = manifest.family,
            app = crate::VERSION,
            tools = if native.is_empty() { String::new() } else { format!(" ({})", native.join(", ")) },
        ),
    )?;
    Ok(path)
}

/// Reads a `.szi` model (its description file, or its folder).
pub fn read(path: &Path) -> Result<MagicFile> {
    let file = if path.is_dir() {
        std::fs::read_dir(path)?
            .flatten()
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|e| e == EXTENSION))
            .ok_or_else(|| Error::format("this folder has no .szi file"))?
    } else {
        path.to_path_buf()
    };
    let dir = file.parent().unwrap_or(Path::new("."));
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(&file)?)?;
    if manifest.format != FORMAT {
        return Err(Error::format("this is not a Magi model (.szi) description"));
    }
    if manifest.version > VERSION {
        return Err(Error::format("this model was saved by a newer MacroMagi; please update"));
    }
    let read_if = |name: &str| std::fs::read(dir.join(name)).unwrap_or_default();
    let mut data = std::collections::BTreeMap::new();
    let tools = dir.join(TOOLS);
    if tools.is_dir() {
        collect(&tools, &tools, &mut data);
    }
    Ok(MagicFile { model: manifest.model, onnx: read_if(ONNX), weights: read_if(WEIGHTS), data })
}

/// Opens a `.magic` workspace or model, or a `.szi` model. A workspace whose
/// model is in a package gets the model (weights, ONNX, native tools) from it.
pub fn open(path: &Path) -> Result<MagicFile> {
    if path.is_dir() || path.extension().is_some_and(|e| e.eq_ignore_ascii_case(EXTENSION)) {
        return read(path);
    }
    let mut file = MagicFile::load(path)?;
    if let Some(package) = file.model.package.clone().filter(|_| file.onnx.is_empty() && file.weights.is_empty()) {
        // The package's path, or the same path beside the workspace when the folder moved.
        let stored = PathBuf::from(&package);
        let beside = path.parent().map(|dir| dir.join("szi").join(stored.file_name().unwrap_or_default()));
        if let Some(found) = [Some(stored), beside].into_iter().flatten().find(|p| p.is_file()) {
            let model = read(&found)?;
            file.onnx = model.onnx;
            file.weights = model.weights;
            for (name, bytes) in model.data {
                file.data.entry(name).or_insert(bytes);
            }
        }
    }
    Ok(file)
}

/// The files of a native tools folder, as `nt/<tool>/<path>`.
fn collect(root: &Path, dir: &Path, into: &mut std::collections::BTreeMap<String, Vec<u8>>) {
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(root, &path, into);
        } else if let (Ok(rest), Ok(bytes)) = (path.strip_prefix(root), std::fs::read(&path)) {
            let rest = rest.to_string_lossy().replace(std::path::MAIN_SEPARATOR, "/");
            if rest.contains('/') {
                into.insert(format!("{TOOLS}/{rest}"), bytes);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::magic::{ModelKind, TextSpec, ToolSpec, TrainingRecord};

    #[test]
    fn a_model_goes_into_a_szi_folder_and_comes_back() {
        let dir = std::env::temp_dir().join(format!("magi-szi-test-{}", std::process::id()));
        let file = MagicFile {
            model: MagicModel {
                kind: ModelKind::Text,
                name: "Magi-F1".into(),
                text: Some(TextSpec {
                    tools: vec![ToolSpec { name: "calculator".into(), native: true, ..Default::default() }],
                    ..TextSpec::default()
                }),
                trained: Some(TrainingRecord { parameters: 3_500_000, ..Default::default() }),
                ..MagicModel::default()
            },
            onnx: vec![1, 2, 3],
            weights: vec![4, 5, 6],
            data: Default::default(),
        };
        let path = write(&file, &dir, "Magi-F1", true).unwrap();
        assert!(dir.join("nt").join("calculator").join("run.py").is_file());
        let back = read(&dir).unwrap();
        assert_eq!((back.onnx, back.weights), (vec![1, 2, 3], vec![4, 5, 6]));
        assert_eq!(back.model.name, "Magi-F1");
        assert!(back.data.contains_key("nt/calculator/tool.json"));
        let manifest: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(manifest["family"], "Magi-F");
        let _ = std::fs::remove_dir_all(dir);
    }
}
