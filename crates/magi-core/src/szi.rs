//! `.szi`: the Magi model format.
//!
//! A Magi model is one file: download it, open it, and it runs –
//! with its native tools, which a model file of weights alone could not carry.
//! MacroMagi writes it when it compiles a model, and runs it; a workspace
//! exported as Python writes it too; CodeMagi will run it.
//!
//! The file is a zip archive (the weights stored as they are, not compressed,
//! so they can be read straight from the file):
//!
//! ```text
//! model.json            the description: format, name, architecture, how it
//!                       reads and writes, its tools, its memory settings
//! weights.safetensors   the weights (the native engine runs these)
//! model.onnx            the same model for ONNX runtimes (when asked for)
//! nt/<tool>/            its native tools: tool.json and the program
//! README.md             what it is and how to run it
//! ```
//!
//! Version 1 was a folder: a `.szi` description (JSON) beside those files. It
//! still opens.

use std::collections::BTreeMap;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use crate::error::{Error, Result};
use crate::magic::{MagicFile, MagicModel};

pub const FORMAT: &str = "magi.szi";
pub const VERSION: u32 = 2;
pub const EXTENSION: &str = "szi";
pub const MANIFEST: &str = "model.json";
pub const WEIGHTS: &str = "weights.safetensors";
pub const ONNX: &str = "model.onnx";
pub const TOOLS: &str = "nt";

/// The description: what the model is, and which files it has.
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

/// Writes a trained model as a `.szi` file at `path`.
pub fn write(file: &MagicFile, path: &Path, with_onnx: bool) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    // Written beside it first, so a failed write leaves the model that was there.
    let part = path.with_extension("szi.part");
    let result = std::fs::File::create(&part)
        .map_err(Error::from)
        .and_then(|out| write_to(file, std::io::BufWriter::new(out), with_onnx));
    match result {
        Ok(()) => Ok(std::fs::rename(&part, path)?),
        Err(error) => {
            let _ = std::fs::remove_file(&part);
            Err(error)
        }
    }
}

/// A trained model as the bytes of a `.szi` file.
pub fn to_bytes(file: &MagicFile, with_onnx: bool) -> Result<Vec<u8>> {
    let mut out = std::io::Cursor::new(Vec::new());
    write_to(file, &mut out, with_onnx)?;
    Ok(out.into_inner())
}

fn write_to<W: Write + Seek>(file: &MagicFile, writer: W, with_onnx: bool) -> Result<()> {
    if file.weights.is_empty() && file.onnx.is_empty() {
        return Err(Error::format("the model has not been trained yet"));
    }
    let mut model = file.model.clone();
    // What builds and trains the model stays in the workspace.
    model.studio = None;
    model.datasets.clear();
    model.training.clear();
    model.finetune.clear();
    model.package = None;
    let with_onnx = with_onnx && !file.onnx.is_empty();
    let tools = model.text.as_ref().map(|t| t.tools.clone()).unwrap_or_default();
    let native: Vec<&str> = tools.iter().filter(|t| t.native).map(|t| t.name.as_str()).collect();
    let parameters = model.trained.as_ref().map_or(0, |t| t.parameters);
    let manifest = Manifest {
        format: FORMAT.into(),
        version: VERSION,
        app: crate::VERSION.into(),
        family: family(parameters).into(),
        files: serde_json::json!({
            "weights": (!file.weights.is_empty()).then_some(WEIGHTS),
            "onnx": with_onnx.then_some(ONNX),
            "native_tools": (!native.is_empty()).then_some(TOOLS),
        }),
        model,
    };

    let mut zip = ZipWriter::new(writer);
    let text = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    // Large files go in as they are – no 4 GB limit – and stay readable in place.
    let stored = SimpleFileOptions::default().compression_method(CompressionMethod::Stored).large_file(true);
    zip.start_file(MANIFEST, text)?;
    zip.write_all(&serde_json::to_vec_pretty(&manifest)?)?;
    if !file.weights.is_empty() {
        zip.start_file(WEIGHTS, stored)?;
        zip.write_all(&file.weights)?;
    }
    if with_onnx {
        zip.start_file(ONNX, stored)?;
        zip.write_all(&file.onnx)?;
    }
    for tool in tools.iter().filter(|t| t.native) {
        for (path, bytes) in crate::native_tools::files(&tool.name, &file.data) {
            zip.start_file(format!("{TOOLS}/{}/{path}", tool.name), text)?;
            zip.write_all(&bytes)?;
        }
    }
    zip.start_file("README.md", text)?;
    zip.write_all(readme(&manifest, &native, with_onnx).as_bytes())?;
    zip.finish()?;
    Ok(())
}

fn readme(manifest: &Manifest, native: &[&str], with_onnx: bool) -> String {
    let mut text = format!(
        "# {name}\n\n{description}\n\nA {family} model in Magi's `.szi` format, made in MacroMagi {app}. Open the \
         `.szi` file in MacroMagi (File › Open, or the AI Agent window) to use it, or add it to a macro.\n\n\
         The `.szi` file is a zip archive:\n\n- `model.json` – what the model is: its architecture, how it reads \
         and writes, its tools.\n- `weights.safetensors` – the weights, which MacroMagi's native engine runs with \
         its key/value cache.\n",
        name = manifest.model.name,
        description = manifest.model.description,
        family = manifest.family,
        app = crate::VERSION,
    );
    if with_onnx {
        text.push_str("- `model.onnx` – the same model for ONNX runtimes.\n");
    }
    if !native.is_empty() {
        text.push_str(&format!(
            "- `nt/` – its native tools ({}): programs it calls, run on this computer.\n",
            native.join(", ")
        ));
    }
    text
}

/// Reads a `.szi` model: the file, or a version 1 folder or its description file.
pub fn read(path: &Path) -> Result<MagicFile> {
    if path.is_dir() {
        let description = std::fs::read_dir(path)?
            .flatten()
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|e| e == EXTENSION))
            .ok_or_else(|| Error::format("this folder has no .szi file"))?;
        return read_folder(&description);
    }
    let mut start = [0u8; 4];
    let mut handle = std::fs::File::open(path)?;
    let is_zip = handle.read_exact(&mut start).is_ok() && start == *b"PK\x03\x04";
    if !is_zip {
        return read_folder(path);
    }
    handle.rewind()?;
    read_from(std::io::BufReader::new(handle))
}

/// A `.szi` file from its bytes.
pub fn from_bytes(bytes: &[u8]) -> Result<MagicFile> {
    read_from(std::io::Cursor::new(bytes))
}

fn read_from<R: Read + Seek>(reader: R) -> Result<MagicFile> {
    let mut zip = ZipArchive::new(reader).map_err(|_| Error::format("this is not a Magi model (.szi) file"))?;
    let manifest: Manifest = serde_json::from_slice(&entry(&mut zip, MANIFEST)?)?;
    check(&manifest)?;
    let mut data = BTreeMap::new();
    let names: Vec<String> = zip
        .file_names()
        .filter_map(|name| name.ok().map(|n| n.into_owned()))
        .filter(|n| n.starts_with("nt/") && !n.ends_with('/'))
        .collect();
    for name in names {
        let bytes = entry(&mut zip, &name)?;
        data.insert(name, bytes);
    }
    let optional = |zip: &mut ZipArchive<R>, name: &str| {
        if zip.index_for_name(name).is_some() { entry(zip, name) } else { Ok(Vec::new()) }
    };
    let weights = optional(&mut zip, WEIGHTS)?;
    let onnx = optional(&mut zip, ONNX)?;
    Ok(MagicFile { model: manifest.model, onnx, weights, data })
}

fn entry<R: Read + Seek>(zip: &mut ZipArchive<R>, name: &str) -> Result<Vec<u8>> {
    let mut file = zip.by_name(name).map_err(|_| Error::format(format!("the model has no {name}")))?;
    let mut bytes = Vec::with_capacity(file.size() as usize);
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn check(manifest: &Manifest) -> Result<()> {
    if manifest.format != FORMAT {
        return Err(Error::format("this is not a Magi model (.szi) description"));
    }
    if manifest.version > VERSION {
        return Err(Error::format("this model was saved by a newer MacroMagi; please update"));
    }
    Ok(())
}

/// A version 1 model: its description file, with the files beside it.
fn read_folder(description: &Path) -> Result<MagicFile> {
    let dir = description.parent().unwrap_or(Path::new("."));
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(description)?)?;
    check(&manifest)?;
    let read_if = |name: &str| std::fs::read(dir.join(name)).unwrap_or_default();
    let mut data = BTreeMap::new();
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
        // The package's path, or the same name beside the workspace (or in its szi/ folder, as
        // version 1 wrote it) when the folder moved.
        let stored = PathBuf::from(&package);
        let name = stored.file_name().unwrap_or_default().to_owned();
        let beside = path.parent().map(|dir| [dir.join(&name), dir.join("szi").join(&name)]);
        if let Some(found) = std::iter::once(stored).chain(beside.into_iter().flatten()).find(|p| p.is_file()) {
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
fn collect(root: &Path, dir: &Path, into: &mut BTreeMap<String, Vec<u8>>) {
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

    fn example() -> MagicFile {
        MagicFile {
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
        }
    }

    #[test]
    fn a_model_goes_into_one_szi_file_and_comes_back() {
        let dir = std::env::temp_dir().join(format!("magi-szi-test-{}", std::process::id()));
        let path = dir.join("Magi-F1.szi");
        write(&example(), &path, true).unwrap();
        assert!(path.is_file() && !dir.join("Magi-F1.szi.part").exists());
        let back = read(&path).unwrap();
        assert_eq!((back.onnx, back.weights), (vec![1, 2, 3], vec![4, 5, 6]));
        assert_eq!(back.model.name, "Magi-F1");
        assert!(back.data.contains_key("nt/calculator/tool.json") && back.data.contains_key("nt/calculator/run.py"));
        // Without ONNX, and from bytes as a download gives them.
        let bytes = to_bytes(&example(), false).unwrap();
        let back = from_bytes(&bytes).unwrap();
        assert!(back.onnx.is_empty() && back.weights == vec![4, 5, 6]);
        let mut zip = ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
        let manifest: Value = serde_json::from_slice(&entry(&mut zip, MANIFEST).unwrap()).unwrap();
        assert_eq!((manifest["family"].as_str(), manifest["version"].as_u64()), (Some("Magi-F"), Some(2)));
        assert_eq!(zip.by_name(WEIGHTS).unwrap().compression(), CompressionMethod::Stored);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A `.szi` file written elsewhere – by a workspace exported as Python – opens:
    /// `MAGI_TEST_SZI=path cargo test -p magi-core opens_a_given -- --ignored`.
    #[test]
    #[ignore = "needs a .szi file (MAGI_TEST_SZI)"]
    fn opens_a_given_szi_file() {
        let path = std::env::var_os("MAGI_TEST_SZI").expect("MAGI_TEST_SZI");
        let file = read(Path::new(&path)).unwrap();
        assert!(!file.weights.is_empty(), "its weights");
        assert!(file.model.trained.is_some(), "what training made");
        assert!(file.data.keys().any(|k| k.starts_with("nt/")), "its native tools");
    }

    #[test]
    fn a_version_1_folder_still_opens() {
        let dir = std::env::temp_dir().join(format!("magi-szi-v1-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("nt").join("calculator")).unwrap();
        let manifest = serde_json::json!({ "format": FORMAT, "version": 1, "kind": "text", "name": "Old" });
        std::fs::write(dir.join("Old.szi"), serde_json::to_vec(&manifest).unwrap()).unwrap();
        std::fs::write(dir.join(WEIGHTS), [7, 8]).unwrap();
        std::fs::write(dir.join("nt").join("calculator").join("tool.json"), b"{}").unwrap();
        for path in [dir.clone(), dir.join("Old.szi")] {
            let back = read(&path).unwrap();
            assert_eq!((back.model.name.as_str(), back.weights.as_slice()), ("Old", &[7u8, 8][..]));
            assert!(back.data.contains_key("nt/calculator/tool.json"));
        }
        let _ = std::fs::remove_dir_all(dir);
    }
}
