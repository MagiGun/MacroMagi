//! Optional downloads: parts some features need that MacroMagi does not ship,
//! installed into (and removed from) the data folder – or, for Python
//! packages, the Python used for AI – on request. Settings › Runtime lists them.
//!
//! | Part | For | From |
//! | --- | --- | --- |
//! | PyTorch (CPU) / PyTorch for NVIDIA | training models in the Magic tab | pip (PyTorch's own index), into the AI Python |
//! | onnx | exporting trained models so MacroMagi runs them | pip, into the AI Python |
//! | ONNX Runtime | faster ONNX models on the CPU | Microsoft's GitHub releases |
//! | ONNX Runtime for graphics cards | ONNX models on any DirectX 12 graphics card (Windows) | Microsoft's NuGet packages |
//! | Python | the Python engine, without installing Python (Windows; managed on the Engine page) | python.org |
//! | Document tools | Magi's open_document and edit_document (.docx, .xlsx, .pptx, .pdf) | pip, into the AI Python |
//! | Image descriptions | Magi's view_image: SmolVLM-256M-Instruct describes pictures | pip and Hugging Face |
//! | MagiDB | Magi's library of how things are done, on this computer (faster, and offline) | its GitHub repository |
//!
//! Each downloaded part lives in its own folder under `data/components`. A
//! part in use cannot be deleted on Windows (its files are locked); it is then
//! marked and removed the next time MacroMagi starts.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

/// ONNX Runtime for the CPU, and the last release built for DirectML.
const ORT_VERSION: &str = "1.31.0";
const ORT_DIRECTML_VERSION: &str = "1.24.4";
const DIRECTML_VERSION: &str = "1.15.4";
const PYTHON_VERSION: &str = "3.13.14";
/// The packages Magi's document tools read and write files with.
const DOCUMENT_PACKAGES: [&str; 5] = ["python-docx", "openpyxl", "python-pptx", "pypdf", "pillow"];
/// The small vision model view_image describes pictures with (Apache-2.0, about 500 MB).
pub const VISION_MODEL: &str = "HuggingFaceTB/SmolVLM-256M-Instruct";
/// MagiDB's repository, as a zip of its main branch.
pub const MAGIDB_ZIP: &str = "https://github.com/MagiGun/MagiDB/archive/refs/heads/main.zip";
/// PyTorch's builds for NVIDIA graphics cards: CUDA 12.8 covers cards from the GTX 900s to the RTX 50s.
const TORCH_CUDA_INDEX: &str = "https://download.pytorch.org/whl/cu128";
const TORCH_CPU_INDEX: &str = "https://download.pytorch.org/whl/cpu";
/// Marks a folder to be deleted at the next start.
const REMOVE_MARK: &str = ".remove";
/// Holds the version of a part downloaded from a "latest" release.
const VERSION_FILE: &str = "VERSION";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Component {
    OnnxRuntime,
    OnnxRuntimeGpu,
    Python,
    /// PyTorch for the CPU.
    PyTorch,
    /// PyTorch for NVIDIA graphics cards (CUDA).
    PyTorchCuda,
    /// The onnx package, which exports trained models.
    Onnx,
    /// Packages Magi's document tools use.
    DocumentTools,
    /// The small vision model Magi's view_image describes pictures with.
    ImageDescriptions,
    /// MagiDB, downloaded.
    MagiDb,
}

/// How far an install has got: what it is doing, and the share done when known.
pub type Report<'a> = &'a dyn Fn(&str, Option<f32>);

impl Component {
    pub const ALL: [Component; 9] = [
        Component::OnnxRuntime,
        Component::OnnxRuntimeGpu,
        Component::Python,
        Component::PyTorch,
        Component::PyTorchCuda,
        Component::Onnx,
        Component::DocumentTools,
        Component::ImageDescriptions,
        Component::MagiDb,
    ];
    /// For Magi's tools.
    pub const TOOLS: [Component; 3] = [Component::MagiDb, Component::DocumentTools, Component::ImageDescriptions];
    /// For training, in the AI Python.
    pub const TRAINING: [Component; 3] = [Component::PyTorchCuda, Component::PyTorch, Component::Onnx];
    /// For running ONNX models.
    pub const ONNX: [Component; 2] = [Component::OnnxRuntime, Component::OnnxRuntimeGpu];

    pub fn name(self) -> &'static str {
        match self {
            Component::OnnxRuntime => "ONNX Runtime",
            Component::OnnxRuntimeGpu => "ONNX Runtime for graphics cards",
            Component::Python => "Embedded Python",
            Component::PyTorch => "PyTorch for the CPU",
            Component::PyTorchCuda => "PyTorch for NVIDIA graphics cards",
            Component::Onnx => "onnx",
            Component::DocumentTools => "Document tools",
            Component::ImageDescriptions => "Image descriptions",
            Component::MagiDb => "MagiDB",
        }
    }

    /// Engine (runs models) or framework (trains or exports them), as a badge.
    pub fn kind(self) -> &'static str {
        match self {
            Component::PyTorch | Component::PyTorchCuda | Component::Onnx => "Training",
            Component::Python => "Python",
            Component::DocumentTools | Component::ImageDescriptions | Component::MagiDb => "Tools",
            _ => "Engine",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Component::OnnxRuntime => {
                "Runs ONNX models faster on the CPU than the built-in engine. AI models work without it."
            }
            Component::OnnxRuntimeGpu => {
                "Runs ONNX models on a graphics card (any DirectX 12 card, through DirectML). Also does what ONNX \
                 Runtime does."
            }
            Component::Python => {
                "Python for the Python engine and Python Script blocks, on computers that do not have it."
            }
            Component::PyTorch => {
                "Trains models in the Magic tab on the processor. Smaller download; slower for anything but small \
                 models. Replaces the NVIDIA build if that is installed."
            }
            Component::PyTorchCuda => {
                "Trains models in the Magic tab on NVIDIA graphics cards (CUDA 12.8: GTX 900s to RTX 50s), many times \
                 faster than the CPU. Also trains on the CPU. Replaces the CPU build."
            }
            Component::Onnx => {
                "Exports trained models so MacroMagi can run them. Training needs it, whichever PyTorch is installed."
            }
            Component::DocumentTools => {
                "Lets Magi models open and change .docx, .xlsx and .pptx files and read .pdf (python-docx, \
                 openpyxl, python-pptx, pypdf), in the AI Python."
            }
            Component::ImageDescriptions => {
                "Lets Magi models see pictures: a small vision model (SmolVLM-256M-Instruct, Apache-2.0) describes \
                 them, and Windows reads their text. Needs PyTorch."
            }
            Component::MagiDb => {
                "Magi's library of how things are done, on this computer: faster, and it works offline. Without it, \
                 Magi models read the pages online."
            }
        }
    }

    /// About how much is downloaded.
    pub fn size(self) -> &'static str {
        match self {
            Component::OnnxRuntime => "about 15 MB",
            Component::OnnxRuntimeGpu => "about 40 MB",
            Component::Python => "about 11 MB",
            Component::PyTorch => "about 250 MB",
            Component::PyTorchCuda => "about 3 GB",
            Component::Onnx => "about 20 MB",
            Component::DocumentTools => "about 15 MB",
            Component::ImageDescriptions => "about 600 MB",
            Component::MagiDb => "a few MB",
        }
    }

    /// Why it cannot be downloaded on this system, if it cannot.
    pub fn unavailable(self) -> Option<&'static str> {
        match self {
            Component::OnnxRuntimeGpu if !cfg!(windows) => Some(if cfg!(target_os = "macos") {
                "On macOS, ONNX Runtime uses the graphics hardware by itself."
            } else {
                "On Linux, graphics card support needs CUDA; install ONNX Runtime with CUDA yourself."
            }),
            Component::OnnxRuntime if cfg!(all(target_os = "macos", target_arch = "x86_64")) => {
                Some("ONNX Runtime no longer offers builds for Intel Macs; the built-in engine is used.")
            }
            Component::Python if !cfg!(windows) => {
                Some("Python is part of most Linux and macOS systems; install it from python.org if it is missing.")
            }
            Component::PyTorchCuda if cfg!(target_os = "macos") => {
                Some("Macs have no NVIDIA graphics cards; PyTorch for the CPU uses Apple's graphics by itself.")
            }
            _ => None,
        }
    }

    fn folder(self) -> &'static str {
        match self {
            Component::OnnxRuntime => "onnxruntime",
            Component::OnnxRuntimeGpu => "onnxruntime-gpu",
            Component::Python => "python",
            Component::PyTorch | Component::PyTorchCuda | Component::Onnx | Component::DocumentTools => "pytorch",
            Component::ImageDescriptions => "vision",
            Component::MagiDb => "magidb",
        }
    }

    fn is_package(self) -> bool {
        matches!(self, Component::PyTorch | Component::PyTorchCuda | Component::Onnx | Component::DocumentTools)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    /// Downloaded by MacroMagi.
    Installed {
        bytes: u64,
        detail: String,
    },
    /// Present without MacroMagi's help.
    OnSystem(String),
    Missing,
    /// Removed, but some files were in use: gone after a restart.
    RemovedAfterRestart,
}

impl Status {
    pub fn present(&self) -> bool {
        matches!(self, Status::Installed { .. } | Status::OnSystem(_))
    }
}

/// What PyTorch in the AI Python is.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TorchInfo {
    pub version: String,
    /// The CUDA version of an NVIDIA build.
    pub cuda: Option<String>,
    /// The graphics cards it can train on, in CUDA order.
    pub devices: Vec<String>,
}

/// Where every part stands, with what PyTorch can use.
#[derive(Clone, Debug, Default)]
pub struct RuntimeReport {
    pub statuses: Vec<(Component, Status)>,
    pub torch: Option<TorchInfo>,
    pub onnx: Option<String>,
    /// Whether the document tools' packages are in the AI Python.
    pub documents: bool,
}

#[derive(Clone)]
pub struct Components {
    root: PathBuf,
    app_dir: PathBuf,
}

fn folder_size(path: &Path) -> u64 {
    std::fs::read_dir(path)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| {
                    let path = e.path();
                    if path.is_dir() { folder_size(&path) } else { e.metadata().map_or(0, |m| m.len()) }
                })
                .sum()
        })
        .unwrap_or(0)
}

/// Downloads a file, reporting how much has arrived.
fn download(url: &str, progress: Report<'_>) -> Result<Vec<u8>, String> {
    let mut response = ureq::get(url).call().map_err(|e| format!("could not download {url}: {e}"))?;
    let total: Option<u64> =
        response.headers().get("content-length").and_then(|v| v.to_str().ok()).and_then(|v| v.parse().ok());
    let mut reader = response.body_mut().as_reader();
    let mut bytes = Vec::with_capacity(total.unwrap_or(0) as usize);
    let mut chunk = vec![0u8; 256 * 1024];
    let mut last = 0;
    loop {
        let read = reader.read(&mut chunk).map_err(|e| format!("the download stopped: {e}"))?;
        if read == 0 {
            break;
        }
        bytes.extend_from_slice(&chunk[..read]);
        let megabytes = bytes.len() / (1024 * 1024);
        if megabytes != last {
            last = megabytes;
            match total {
                Some(total) => {
                    let share = bytes.len() as f32 / total.max(1) as f32;
                    progress(&format!("Downloading… {}%", (share * 100.0) as u32), Some(share));
                }
                None => progress(&format!("Downloading… {megabytes} MB"), None),
            }
        }
    }
    Ok(bytes)
}

/// Writes the files of a ZIP archive whose paths end with one of `wanted`
/// into `into`, under their own file name. Returns how many were written.
fn unzip_files(archive: &[u8], wanted: &[&str], into: &Path) -> Result<usize, String> {
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(archive)).map_err(|e| e.to_string())?;
    let mut written = 0;
    for index in 0..zip.len() {
        let mut entry = zip.by_index(index).map_err(|e| e.to_string())?;
        let name = entry.name().map_err(|e| e.to_string())?.replace('\\', "/");
        if let Some(wanted) = wanted.iter().find(|w| name.ends_with(*w)) {
            let file = wanted.rsplit('/').next().unwrap_or(wanted);
            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
            std::fs::write(into.join(file), bytes).map_err(|e| e.to_string())?;
            written += 1;
        }
    }
    Ok(written)
}

/// Unpacks a whole ZIP archive into `into`.
fn unzip_all(archive: &[u8], into: &Path) -> Result<(), String> {
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(archive)).map_err(|e| e.to_string())?;
    zip.extract(into).map_err(|e| e.to_string())
}

/// The file of a `.tgz` archive whose name starts with `prefix` (the real
/// library, not its links), saved as `save_as`.
#[cfg(not(windows))]
fn untar_library(archive: &[u8], prefix: &str, save_as: &Path) -> Result<(), String> {
    let decoder = flate2::read::GzDecoder::new(std::io::Cursor::new(archive));
    let mut tar = tar::Archive::new(decoder);
    for entry in tar.entries().map_err(|e| e.to_string())? {
        let mut entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path().map_err(|e| e.to_string())?.to_string_lossy().into_owned();
        let file = path.rsplit('/').next().unwrap_or(&path).to_owned();
        if entry.header().entry_type().is_file() && file.starts_with(prefix) {
            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
            return std::fs::write(save_as, bytes).map_err(|e| e.to_string());
        }
    }
    Err("the library was not found in the download".into())
}

/// Runs a Python command, returning what it printed.
fn python_output(python: &Path, args: &[&str]) -> Result<String, String> {
    let mut command = Command::new(python);
    command.args(args).env("PYTHONDONTWRITEBYTECODE", "1");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let output = command.output().map_err(|e| format!("could not run {}: {e}", python.display()))?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
    } else {
        let error = String::from_utf8_lossy(&output.stderr);
        let last = error.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("it failed");
        Err(last.trim().to_owned())
    }
}

/// Runs pip in a Python, saying what it does.
fn pip(python: &Path, args: &[&str], progress: Report<'_>, doing: &str) -> Result<String, String> {
    progress(doing, None);
    let mut all = vec!["-m", "pip"];
    all.extend_from_slice(args);
    all.extend_from_slice(&["--disable-pip-version-check", "--no-warn-script-location"]);
    python_output(python, &all)
}

impl Components {
    pub fn new(data_dir: &Path, app_dir: &Path) -> Self {
        Self { root: data_dir.join("components"), app_dir: app_dir.to_owned() }
    }

    fn dir(&self, component: Component) -> PathBuf {
        self.root.join(component.folder())
    }

    /// Deletes what was marked for removal while it was in use. Call at start,
    /// before anything is loaded.
    pub fn clean_up(&self) {
        for component in Component::ALL {
            let dir = self.dir(component);
            if dir.join(REMOVE_MARK).exists() {
                let _ = std::fs::remove_dir_all(&dir);
            }
        }
    }

    fn library_name() -> &'static str {
        if cfg!(windows) {
            "onnxruntime.dll"
        } else if cfg!(target_os = "macos") {
            "libonnxruntime.dylib"
        } else {
            "libonnxruntime.so"
        }
    }

    fn installed(&self, component: Component) -> bool {
        let dir = self.dir(component);
        if dir.join(REMOVE_MARK).exists() {
            return false;
        }
        match component {
            Component::OnnxRuntime | Component::OnnxRuntimeGpu => dir.join(Self::library_name()).is_file(),
            Component::Python => dir.join("python.exe").is_file(),
            Component::ImageDescriptions => dir.join("model").join("config.json").is_file(),
            Component::MagiDb => std::fs::read_dir(&dir).is_ok_and(|mut d| d.any(|e| e.is_ok())),
            _ => false,
        }
    }

    /// The ONNX Runtime library to use, and whether it can use graphics cards:
    /// the graphics build if it is installed, else the CPU one, else one next
    /// to MacroMagi.
    pub fn onnxruntime(&self) -> Option<(PathBuf, bool)> {
        if self.installed(Component::OnnxRuntimeGpu) {
            return Some((self.dir(Component::OnnxRuntimeGpu).join(Self::library_name()), true));
        }
        if self.installed(Component::OnnxRuntime) {
            return Some((self.dir(Component::OnnxRuntime).join(Self::library_name()), cfg!(target_os = "macos")));
        }
        let beside = self.app_dir.join(Self::library_name());
        beside.is_file().then_some((beside, false))
    }

    /// DirectML's library, which has to be loaded before ONNX Runtime asks for it.
    pub fn directml(&self) -> Option<PathBuf> {
        let path = self.dir(Component::OnnxRuntimeGpu).join("DirectML.dll");
        (self.installed(Component::OnnxRuntimeGpu) && path.is_file()).then_some(path)
    }

    /// The Python downloaded by MacroMagi.
    pub fn python(&self) -> Option<PathBuf> {
        let path = self.dir(Component::Python).join("python.exe");
        self.installed(Component::Python).then_some(path)
    }

    /// The image description model, if it is installed.
    pub fn vision_model(&self) -> Option<PathBuf> {
        self.installed(Component::ImageDescriptions).then(|| self.dir(Component::ImageDescriptions).join("model"))
    }

    /// MagiDB, if it was downloaded.
    pub fn magidb(&self) -> Option<PathBuf> {
        self.installed(Component::MagiDb).then(|| self.dir(Component::MagiDb))
    }

    /// Where every part stands. `python` is the AI Python, where PyTorch and onnx
    /// are looked for. Starts Python: call it off the UI thread.
    pub fn report(&self, python: Option<&Path>) -> RuntimeReport {
        let mut report = RuntimeReport::default();
        if let Some(python) = python {
            let probe = "import torch, json\n\
                         print(json.dumps({'version': torch.__version__, 'cuda': torch.version.cuda, \
                         'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] \
                         if torch.cuda.is_available() else []}))";
            if let Ok(text) = python_output(python, &["-c", probe])
                && let Ok(value) = serde_json::from_str::<serde_json::Value>(text.lines().last().unwrap_or_default())
            {
                report.torch = Some(TorchInfo {
                    version: value["version"].as_str().unwrap_or_default().to_owned(),
                    cuda: value["cuda"].as_str().map(str::to_owned),
                    devices: value["devices"]
                        .as_array()
                        .map(|d| d.iter().filter_map(|n| n.as_str().map(str::to_owned)).collect())
                        .unwrap_or_default(),
                });
            }
            report.onnx = python_output(python, &["-c", "import onnx; print(onnx.__version__)"]).ok();
            report.documents = python_output(python, &["-c", "import docx, openpyxl, pptx, pypdf"]).is_ok();
        }
        for component in Component::ALL {
            let status = self.status(component, python, &report);
            report.statuses.push((component, status));
        }
        report
    }

    fn status(&self, component: Component, python: Option<&Path>, report: &RuntimeReport) -> Status {
        let dir = self.dir(component);
        if !component.is_package() && dir.join(REMOVE_MARK).exists() {
            return Status::RemovedAfterRestart;
        }
        let in_python = |detail: String| {
            let ours = python.is_some_and(|p| self.python().is_some_and(|o| o == p));
            if ours { Status::Installed { bytes: 0, detail } } else { Status::OnSystem(detail) }
        };
        let where_ = python.map(|p| format!(" · {}", p.display())).unwrap_or_default();
        match component {
            Component::PyTorch => match &report.torch {
                Some(torch) if torch.cuda.is_none() => in_python(format!("{}{where_}", torch.version)),
                _ => Status::Missing,
            },
            Component::PyTorchCuda => match &report.torch {
                Some(torch) if torch.cuda.is_some() => {
                    let cards = match torch.devices.len() {
                        0 => " · no NVIDIA card found".to_owned(),
                        1 => format!(" · {}", torch.devices[0]),
                        n => format!(" · {n} cards"),
                    };
                    in_python(format!("{} (CUDA {}){cards}", torch.version, torch.cuda.as_deref().unwrap_or("?")))
                }
                _ => Status::Missing,
            },
            Component::Onnx => match &report.onnx {
                Some(version) => in_python(format!("{version}{where_}")),
                None => Status::Missing,
            },
            Component::DocumentTools if report.documents => in_python(format!("python-docx, openpyxl…{where_}")),
            Component::DocumentTools => Status::Missing,
            _ if self.installed(component) => {
                let detail = match component {
                    Component::OnnxRuntime => ORT_VERSION.to_string(),
                    Component::OnnxRuntimeGpu => format!("{ORT_DIRECTML_VERSION}, DirectML {DIRECTML_VERSION}"),
                    Component::Python => PYTHON_VERSION.to_string(),
                    Component::ImageDescriptions => VISION_MODEL.to_string(),
                    _ => std::fs::read_to_string(dir.join(VERSION_FILE)).unwrap_or_default().trim().to_owned(),
                };
                Status::Installed { bytes: folder_size(&dir), detail }
            }
            Component::OnnxRuntime => {
                let beside = self.app_dir.join(Self::library_name());
                if beside.is_file() {
                    Status::OnSystem(format!("next to MacroMagi: {}", beside.display()))
                } else {
                    Status::Missing
                }
            }
            _ => Status::Missing,
        }
    }

    /// Downloads and installs a part. Blocks: run it off the UI thread.
    pub fn install(&self, component: Component, python: Option<&Path>, progress: Report<'_>) -> Result<String, String> {
        if let Some(reason) = component.unavailable() {
            return Err(reason.to_owned());
        }
        let dir = self.dir(component);
        if !component.is_package() {
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        }
        let result = self.install_into(component, &dir, python, progress);
        if result.is_err() && !component.is_package() {
            let _ = std::fs::remove_dir_all(&dir);
        }
        result
    }

    /// pip, set up first in a Python downloaded by MacroMagi.
    fn ensure_pip(&self, python: &Path, progress: Report<'_>) -> Result<(), String> {
        if python_output(python, &["-m", "pip", "--version"]).is_ok() {
            return Ok(());
        }
        progress("Setting up pip…", None);
        let script = download("https://bootstrap.pypa.io/get-pip.py", progress)?;
        let path = self.root.join("get-pip.py");
        std::fs::create_dir_all(&self.root).map_err(|e| e.to_string())?;
        std::fs::write(&path, script).map_err(|e| e.to_string())?;
        let result = python_output(python, &[&path.to_string_lossy(), "--no-warn-script-location"]);
        let _ = std::fs::remove_file(&path);
        result.map(|_| ()).map_err(|e| format!("could not set up pip: {e}"))
    }

    fn install_into(
        &self,
        component: Component,
        dir: &Path,
        python: Option<&Path>,
        progress: Report<'_>,
    ) -> Result<String, String> {
        match component {
            Component::OnnxRuntime => {
                let base = format!("https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}");
                #[cfg(windows)]
                {
                    let archive = download(&format!("{base}/onnxruntime-win-x64-{ORT_VERSION}.zip"), progress)?;
                    progress("Unpacking…", None);
                    let found =
                        unzip_files(&archive, &["lib/onnxruntime.dll", "lib/onnxruntime_providers_shared.dll"], dir)?;
                    if found == 0 {
                        return Err("the download did not contain ONNX Runtime".into());
                    }
                }
                #[cfg(target_os = "linux")]
                {
                    let archive = download(&format!("{base}/onnxruntime-linux-x64-{ORT_VERSION}.tgz"), progress)?;
                    progress("Unpacking…", None);
                    untar_library(&archive, "libonnxruntime.so.", &dir.join(Self::library_name()))?;
                }
                #[cfg(target_os = "macos")]
                {
                    let archive = download(&format!("{base}/onnxruntime-osx-arm64-{ORT_VERSION}.tgz"), progress)?;
                    progress("Unpacking…", None);
                    untar_library(&archive, "libonnxruntime.", &dir.join(Self::library_name()))?;
                }
                let _ = base;
                Ok(format!("ONNX Runtime {ORT_VERSION} installed. Restart MacroMagi to use it."))
            }
            Component::OnnxRuntimeGpu => {
                let nuget = "https://api.nuget.org/v3-flatcontainer";
                let ort = download(
                    &format!(
                        "{nuget}/microsoft.ml.onnxruntime.directml/{ORT_DIRECTML_VERSION}/microsoft.ml.onnxruntime.directml.{ORT_DIRECTML_VERSION}.nupkg"
                    ),
                    progress,
                )?;
                progress("Unpacking…", None);
                let found = unzip_files(
                    &ort,
                    &[
                        "runtimes/win-x64/native/onnxruntime.dll",
                        "runtimes/win-x64/native/onnxruntime_providers_shared.dll",
                    ],
                    dir,
                )?;
                if found == 0 {
                    return Err("the download did not contain ONNX Runtime".into());
                }
                let directml = download(
                    &format!(
                        "{nuget}/microsoft.ai.directml/{DIRECTML_VERSION}/microsoft.ai.directml.{DIRECTML_VERSION}.nupkg"
                    ),
                    progress,
                )?;
                progress("Unpacking DirectML…", None);
                if unzip_files(&directml, &["bin/x64-win/DirectML.dll"], dir)? == 0 {
                    return Err("the download did not contain DirectML".into());
                }
                Ok("ONNX Runtime for graphics cards installed. Restart MacroMagi to use it.".into())
            }
            Component::Python => {
                let archive = download(
                    &format!(
                        "https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"
                    ),
                    progress,
                )?;
                progress("Unpacking…", None);
                unzip_all(&archive, dir)?;
                // Let the private Python find packages installed into it.
                for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())?.flatten() {
                    let path = entry.path();
                    if path.extension().is_some_and(|e| e == "_pth") {
                        let text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
                        let text = text.replace("#import site", "import site");
                        let text = if text.contains("Lib\\site-packages") {
                            text
                        } else {
                            format!("{text}\nLib\\site-packages\n")
                        };
                        std::fs::write(&path, text).map_err(|e| e.to_string())?;
                    }
                }
                Ok(format!(
                    "Python {PYTHON_VERSION} installed. Choose it under Settings › Engine, or leave the choice on automatic."
                ))
            }
            Component::PyTorch | Component::PyTorchCuda => {
                let python = python.ok_or(
                    "no Python was found to install PyTorch into. Choose or download one under Settings › Engine.",
                )?;
                self.ensure_pip(python, progress)?;
                // One build at a time: the other is taken out first.
                if python_output(python, &["-c", "import torch"]).is_ok() {
                    pip(python, &["uninstall", "-y", "torch"], progress, "Taking out the PyTorch that is there…")?;
                }
                let (index, what) = if component == Component::PyTorchCuda {
                    (
                        TORCH_CUDA_INDEX,
                        "Downloading and installing PyTorch for NVIDIA graphics cards (about 3 GB; this takes a while)…",
                    )
                } else {
                    (TORCH_CPU_INDEX, "Downloading and installing PyTorch (this takes a while)…")
                };
                pip(python, &["install", "torch", "--index-url", index], progress, what)
                    .map_err(|e| format!("pip could not install PyTorch: {e}"))?;
                if python_output(python, &["-c", "import onnx"]).is_err() {
                    pip(python, &["install", "onnx"], progress, "Installing onnx, which training exports models with…")
                        .map_err(|e| format!("pip could not install onnx: {e}"))?;
                }
                Ok(format!("{} installed into {}.", component.name(), python.display()))
            }
            Component::DocumentTools => {
                let python = python.ok_or(
                    "no Python was found to install the document tools into. Choose one under Settings › Engine.",
                )?;
                self.ensure_pip(python, progress)?;
                let mut args = vec!["install"];
                args.extend(DOCUMENT_PACKAGES);
                pip(python, &args, progress, "Installing python-docx, openpyxl, python-pptx and pypdf…")
                    .map_err(|e| format!("pip could not install the document tools: {e}"))?;
                Ok(format!("Document tools installed into {}.", python.display()))
            }
            Component::ImageDescriptions => {
                let python = python.ok_or(
                    "no Python was found to install image descriptions into. Choose one under Settings › Engine.",
                )?;
                if python_output(python, &["-c", "import torch"]).is_err() {
                    return Err("image descriptions need PyTorch: install it first (Training models, above)".into());
                }
                self.ensure_pip(python, progress)?;
                pip(
                    python,
                    &["install", "transformers", "pillow", "huggingface_hub"],
                    progress,
                    "Installing transformers…",
                )
                .map_err(|e| format!("pip could not install transformers: {e}"))?;
                progress("Downloading SmolVLM-256M-Instruct (about 500 MB)…", None);
                let target = dir.join("model");
                let script = format!(
                    "from huggingface_hub import snapshot_download; snapshot_download({VISION_MODEL:?}, local_dir={:?})",
                    target.display().to_string()
                );
                python_output(python, &["-c", &script])
                    .map_err(|e| format!("the model could not be downloaded: {e}"))?;
                Ok("Image descriptions installed: Magi models can look at pictures.".into())
            }
            Component::MagiDb => {
                let archive = download(MAGIDB_ZIP, progress)?;
                progress("Unpacking…", None);
                let mut zip = zip::ZipArchive::new(std::io::Cursor::new(archive)).map_err(|e| e.to_string())?;
                // The zip holds one folder (MagiDB-main/): its contents go into the part's folder.
                for index in 0..zip.len() {
                    let mut entry = zip.by_index(index).map_err(|e| e.to_string())?;
                    let Some(name) = entry.enclosed_name() else { continue };
                    let inner: PathBuf = name.components().skip(1).collect();
                    if inner.as_os_str().is_empty() || entry.is_dir() {
                        continue;
                    }
                    let target = dir.join(inner);
                    if let Some(parent) = target.parent() {
                        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                    }
                    let mut bytes = Vec::new();
                    entry.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
                    std::fs::write(target, bytes).map_err(|e| e.to_string())?;
                }
                Ok("MagiDB downloaded: Magi models read it on this computer.".into())
            }
            Component::Onnx => {
                let python =
                    python.ok_or("no Python was found to install onnx into. Choose one under Settings › Engine.")?;
                self.ensure_pip(python, progress)?;
                pip(python, &["install", "onnx"], progress, "Installing onnx…")
                    .map_err(|e| format!("pip could not install onnx: {e}"))?;
                Ok(format!("onnx installed into {}.", python.display()))
            }
        }
    }

    /// Removes a part. Returns a message to show.
    pub fn uninstall(&self, component: Component, python: Option<&Path>) -> Result<String, String> {
        if component.is_package() {
            let python = python.ok_or("there is no Python to remove it from")?;
            let packages: &[&str] = match component {
                Component::Onnx => &["onnx"],
                Component::DocumentTools => &DOCUMENT_PACKAGES[..4],
                _ => &["torch"],
            };
            let mut args = vec!["-m", "pip", "uninstall", "-y", "--disable-pip-version-check"];
            args.extend(packages);
            python_output(python, &args).map_err(|e| format!("pip could not remove {}: {e}", packages.join(", ")))?;
            return Ok(format!("{} removed from {}.", component.name(), python.display()));
        }
        let dir = self.dir(component);
        match std::fs::remove_dir_all(&dir) {
            Ok(()) => Ok(format!("{} removed.", component.name())),
            Err(_) if dir.exists() => {
                // In use: finish at the next start.
                std::fs::write(dir.join(REMOVE_MARK), b"").map_err(|e| e.to_string())?;
                Ok(format!("{} is in use; it is removed when MacroMagi restarts.", component.name()))
            }
            Err(_) => Ok(format!("{} removed.", component.name())),
        }
    }
}

/// A command that opens no console window.
fn quiet(program: &str) -> Command {
    let mut command = Command::new(program);
    command.stdin(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    command
}

/// Every Python interpreter that can be found, in the order the engine tries
/// them: the one next to MacroMagi, the downloaded one, then the ones
/// installed on the system. Starts programs: call it off the UI thread.
pub fn find_pythons(components: &Components, app_dir: &Path) -> Vec<(PathBuf, String)> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    let beside = if cfg!(windows) {
        app_dir.join("python").join("python.exe")
    } else {
        app_dir.join("python").join("bin").join("python3")
    };
    candidates.push(beside);
    if let Some(ours) = components.python() {
        candidates.push(ours);
    }
    #[cfg(windows)]
    {
        // The py launcher lists every installed Python, Microsoft Store ones included.
        if let Ok(output) = quiet("py").arg("-0p").output() {
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                if let Some(path) = line.split_whitespace().last().filter(|p| p.to_lowercase().ends_with(".exe")) {
                    candidates.push(PathBuf::from(path));
                }
            }
        }
    }
    let finder = if cfg!(windows) { "where" } else { "which" };
    for name in ["python", "python3"] {
        let mut command = quiet(finder);
        if !cfg!(windows) {
            command.arg("-a");
        }
        if let Ok(output) = command.arg(name).output() {
            candidates.extend(String::from_utf8_lossy(&output.stdout).lines().map(|l| PathBuf::from(l.trim())));
        }
    }
    let mut found: Vec<(PathBuf, String)> = Vec::new();
    for path in candidates {
        // Running it is the test: the Microsoft Store's stand-in for a missing
        // Python fails to run with arguments, a real one answers.
        if !path.is_file() {
            continue;
        }
        let key = path.canonicalize().unwrap_or_else(|_| path.clone());
        if found.iter().any(|(p, _)| p.canonicalize().unwrap_or_else(|_| p.clone()) == key) {
            continue;
        }
        if let Ok(version) = python_output(&path, &["-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])"]) {
            let ok = version.split('.').take(2).map(|n| n.parse::<u32>().unwrap_or(0)).collect::<Vec<_>>();
            if ok.len() == 2 && (ok[0], ok[1]) >= (3, 8) {
                found.push((path, version));
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use magi_core::magic::InputSpec;

    use super::*;

    /// Downloads ONNX Runtime for graphics cards and runs a model with it on
    /// the CPU and on the first graphics card. With `MAGI_TEST_MODEL` set to
    /// an image classifier, also compares the engines' speed on it.
    #[test]
    #[ignore = "downloads about 55 MB"]
    fn onnx_runtime_downloads_and_runs_models_on_the_cpu_and_a_graphics_card() {
        let dir = std::env::temp_dir().join(format!("magi-components-{}", std::process::id()));
        let components = Components::new(&dir, &dir);
        let message = components.install(Component::OnnxRuntimeGpu, None, &|_, _| {}).unwrap();
        assert!(message.contains("installed"));
        let (library, gpu) = components.onnxruntime().unwrap();
        assert!(gpu);
        if let Some(directml) = components.directml() {
            assert!(magi_platform::gpu::preload(&directml));
        }
        magi_ml::use_onnxruntime(&library, gpu).unwrap();

        let onnx = include_bytes!("../../magi-engine/tests/data/channel_means.onnx");
        let spec = InputSpec { width: 4, height: 4, mean: [0.0; 3], std: [1.0; 3], ..InputSpec::default() };
        let green: Vec<u8> = (0..64).flat_map(|_| [20u8, 230, 20, 255]).collect();
        for engine in
            [magi_ml::Engine::BuiltIn, magi_ml::Engine::OnnxRuntimeCpu, magi_ml::Engine::OnnxRuntimeGpu { index: 0 }]
        {
            let mut model = magi_ml::Model::load_on(onnx, &spec, engine).unwrap();
            let p = model.classify(&green, 8, 8).unwrap();
            assert!(p[1] > p[0] && p[1] > p[2], "{engine:?}: {p:?}");
        }

        if let Some(path) = std::env::var_os("MAGI_TEST_MODEL") {
            let onnx = std::fs::read(path).unwrap();
            let spec = magi_ml::inspect(&onnx).unwrap().suggested(InputSpec::default());
            let picture: Vec<u8> =
                (0..320 * 320).flat_map(|i| [(i % 251) as u8, (i % 13) as u8 * 19, 90, 255]).collect();
            for engine in [
                magi_ml::Engine::BuiltIn,
                magi_ml::Engine::OnnxRuntimeCpu,
                magi_ml::Engine::OnnxRuntimeGpu { index: 0 },
            ] {
                let mut model = magi_ml::Model::load_on(&onnx, &spec, engine).unwrap();
                let first = model.classify(&picture, 320, 320).unwrap();
                let start = std::time::Instant::now();
                for _ in 0..100 {
                    model.classify(&picture, 320, 320).unwrap();
                }
                let ms = start.elapsed().as_secs_f64() * 10.0;
                println!("{}: {ms:.2} ms per picture, first answers {:?}", engine.label(), &first[..3]);
            }
        }
        assert!(components.uninstall(Component::OnnxRuntimeGpu, None).unwrap().contains("restarts"), "in use");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
