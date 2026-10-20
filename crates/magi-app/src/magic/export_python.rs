//! File › Export as Python: the workspace as a folder of Python that does what
//! the canvas does – one `run.py` to execute, the plan the canvas makes, the
//! datasets, the trainer, the model's native tools and a README.

use std::path::{Path, PathBuf};

use magi_core::magic::{InputSpec, Layout, ModelKind};
use magi_core::nodes::studio::{self, PlanFiles};

use super::{MagicDoc, file_stem};

const RUN_PY: &str = r#""""{name} – trains the model of a MacroMagi Magic workspace, as MacroMagi does.

    python run.py              train it, then write the Magi package (szi/)
    python run.py --package    only write the package (after training)

Needs Python 3.10+ with PyTorch and onnx (pip install -r requirements.txt; for an
NVIDIA graphics card, PyTorch's CUDA build from pytorch.org). It prints its
progress as one JSON object per line, as MacroMagi reads it.
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.join(HERE, "runtime"))


def train():
    from magi_train import train as trainer
    return trainer.main("plan.json")


def family(parameters):
    return "Magi-F" if parameters <= 1_000_000_000 else "Magi-S" if parameters <= 30_000_000_000 else "Magi-T"


def package():
    """The Magi package: the .szi description, the weights, the ONNX model and the native tools."""
    model = json.load(open("model.json", encoding="utf-8"))
    report = json.load(open(os.path.join("out", "report.json"), encoding="utf-8"))
    model["trained"] = {k: report.get(k) for k in ("stages", "steps", "loss", "parameters")}
    target = os.path.join(HERE, "szi")
    os.makedirs(target, exist_ok=True)
    files = {}
    for name in ("weights.safetensors", "model.onnx"):
        source = os.path.join("out", name)
        if os.path.isfile(source):
            shutil.copy(source, os.path.join(target, name))
            files["weights" if name.endswith("safetensors") else "onnx"] = name
    if os.path.isdir("nt"):
        shutil.copytree("nt", os.path.join(target, "nt"), dirs_exist_ok=True)
        files["native_tools"] = "nt"
    manifest = {"format": "magi.szi", "version": 1, "app": "MacroMagi {version} (Python export)",
                "family": family(report.get("parameters") or 0), "files": files}
    manifest.update(model)
    path = os.path.join(target, "{stem}.szi")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(json.dumps({{"event": "log", "message": "Wrote " + path}}))


if __name__ == "__main__":
    if "--package" not in sys.argv:
        code = train()
        if code:
            sys.exit(code)
    package()
"#;

/// Writes the workspace as Python into a new folder in `into`; returns the folder.
pub fn export(doc: &MagicDoc, runtime: &Path, into: &Path) -> Result<PathBuf, String> {
    let pipeline = studio::pipeline(&doc.studio.graph)?;
    if !pipeline.trains() {
        return Err("The canvas does not train a model: add training stages first.".into());
    }
    let stem = file_stem(&doc.title());
    let dir = into.join(format!("{stem} (Python)"));
    let data = dir.join("data");
    std::fs::create_dir_all(&data).map_err(|e| format!("The folder could not be made: {e}"))?;
    let write = |path: &Path, bytes: &[u8]| std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()));

    // The datasets, by the paths the plan gives the trainer.
    let mut written = std::collections::BTreeMap::new();
    for (index, (name, bytes)) in doc.file.data.iter().enumerate() {
        if let Some(file) = name.strip_prefix("data/text/") {
            let relative = format!("data/{index}-{}", file.replace('/', "-"));
            write(&dir.join(&relative), bytes)?;
            written.insert(name.clone(), relative);
        }
    }
    // The weights to start from.
    let mut weights = None;
    if pipeline.entry.kind == studio::kinds::BASE {
        let params: studio::BaseParams = pipeline.entry.params_or_default();
        let bytes = if params.path.trim().is_empty() {
            doc.file.weights.clone()
        } else {
            magi_core::szi::open(Path::new(params.path.trim())).map_err(|e| e.to_string())?.weights
        };
        write(&data.join("base.safetensors"), &bytes)?;
        weights = Some("data/base.safetensors".to_owned());
    }
    // The pictures, prepared as the trainer reads them (vision).
    let kind = pipeline.kind(&doc.file.model);
    let mut pictures = None;
    if kind == ModelKind::Vision {
        let side = pipeline.architecture.context.clamp(8, 1024);
        let spec = InputSpec {
            width: side,
            height: side,
            layout: Layout::Nchw,
            softmax: true,
            ..doc.file.model.input.clone()
        };
        super::train::write_pictures(&doc.file, &spec, &data.join("pictures.bin"))?;
        pictures = Some("data/pictures.bin".to_owned());
    }
    let files = PlanFiles {
        dataset: &|name: &str| written.get(name).cloned(),
        weights,
        pictures,
        out_dir: "out".into(),
        devices: Vec::new(),
    };
    let mut plan = studio::plan(&pipeline, &doc.studio.graph, &doc.file.model, &files)?;
    // Whatever graphics cards the computer it runs on has.
    plan["devices"] = serde_json::json!("auto");
    write(&dir.join("plan.json"), &serde_json::to_vec_pretty(&plan).unwrap_or_default())?;

    // The model's description, as the package will carry it.
    let mut model = doc.file.model.clone();
    model.kind = kind;
    model.architecture = Some(pipeline.architecture.clone());
    model.studio = None;
    model.datasets.clear();
    model.training.clear();
    model.finetune.clear();
    model.package = None;
    write(&dir.join("model.json"), &serde_json::to_vec_pretty(&model).unwrap_or_default())?;
    if let Some(text) = &model.text
        && text.tools.iter().any(|t| t.native)
    {
        magi_core::native_tools::write_folder(&text.tools, &doc.file.data, &dir.join("nt"))
            .map_err(|e| e.to_string())?;
    }

    // The trainer.
    copy_dir(&runtime.join("magi_train"), &dir.join("runtime").join("magi_train"))?;
    let run = RUN_PY
        .replace("{name}", &doc.title())
        .replace("{stem}", &stem)
        .replace("{version}", magi_core::VERSION)
        .replace("{{", "{")
        .replace("}}", "}");
    write(&dir.join("run.py"), run.as_bytes())?;
    write(&dir.join("requirements.txt"), b"torch\nonnx\n")?;
    let stages: Vec<String> =
        pipeline.stages.iter().map(|s| s.kind.trim_start_matches("magic.").replace('_', " ")).collect();
    let readme = format!(
        "# {name} – the workspace as Python\n\n\
         This folder does what the Magic tab's canvas does, without MacroMagi: it trains **{name}** and writes \
         its Magi package.\n\n\
         ```bash\npip install -r requirements.txt\npython run.py\n```\n\n\
         For an NVIDIA graphics card, install PyTorch's CUDA build first (see pytorch.org). Progress is printed \
         as one JSON object per line.\n\n\
         ## What is here\n\n\
         | | |\n| --- | --- |\n\
         | `run.py` | Runs the training, then writes `szi/` – the model in Magi's `.szi` format, with its weights, \
         its ONNX model and its native tools. Open `szi/{stem}.szi` in MacroMagi to use it, or to export GGUF. |\n\
         | `plan.json` | The canvas as the trainer reads it: the architecture ({layers} layers, {width} wide, {context} \
         tokens of context), the stages ({stages}), their datasets and optimizers. |\n\
         | `model.json` | The model's description: how it reads and writes, its tools, thinking and effort. |\n\
         | `data/` | The datasets{pictures}. |\n\
         | `nt/` | The model's native tools, when it has some. |\n\
         | `runtime/magi_train/` | MacroMagi's trainer, unchanged. |\n\n\
         Change `plan.json` to change the training: the steps, the batch, the optimizer, the layers.\n",
        name = doc.title(),
        layers = pipeline.architecture.layers.len(),
        width = pipeline.architecture.width,
        context = pipeline.architecture.context,
        stages = stages.join(", "),
        pictures = if kind == ModelKind::Vision { " and the pictures, prepared" } else { "" },
    );
    write(&dir.join("README.md"), readme.as_bytes())?;
    Ok(dir)
}

fn copy_dir(from: &Path, to: &Path) -> Result<(), String> {
    std::fs::create_dir_all(to).map_err(|e| e.to_string())?;
    for entry in std::fs::read_dir(from).map_err(|e| format!("{}: {e}", from.display()))?.flatten() {
        let path = entry.path();
        let target = to.join(entry.file_name());
        if path.is_dir() {
            if entry.file_name() != "__pycache__" {
                copy_dir(&path, &target)?;
            }
        } else {
            std::fs::copy(&path, &target).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}
