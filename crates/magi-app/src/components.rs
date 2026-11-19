//! Optional downloads: parts some features need that MacroMagi does not ship,
//! installed into (and removed from) the data folder on request.
//!
//! | Part | For | From |
//! | --- | --- | --- |
//! | Python | the Python engine, without installing Python (Windows; managed on the Engine page) | python.org |
//!
//! Each downloaded part lives in its own folder under `data/components`. A
//! part in use cannot be deleted on Windows (its files are locked); it is then
//! marked and removed the next time MacroMagi starts.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

const PYTHON_VERSION: &str = "3.13.14";
/// Marks a folder to be deleted at the next start.
const REMOVE_MARK: &str = ".remove";
/// Folders of parts earlier versions downloaded for AI models, which nothing uses any more.
const RETIRED: [&str; 5] = ["onnxruntime", "onnxruntime-gpu", "pytorch", "vision", "magidb"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Component {
    Python,
}

/// How far an install has got: what it is doing, and the share done when known.
pub type Report<'a> = &'a dyn Fn(&str, Option<f32>);

impl Component {
    pub const ALL: [Component; 1] = [Component::Python];

    pub fn name(self) -> &'static str {
        match self {
            Component::Python => "Embedded Python",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Component::Python => {
                "Python for the Python engine and Python Script blocks, on computers that do not have it."
            }
        }
    }

    /// About how much is downloaded.
    pub fn size(self) -> &'static str {
        match self {
            Component::Python => "about 11 MB",
        }
    }

    /// Why it cannot be downloaded on this system, if it cannot.
    pub fn unavailable(self) -> Option<&'static str> {
        match self {
            Component::Python if !cfg!(windows) => {
                Some("Python is part of most Linux and macOS systems; install it from python.org if it is missing.")
            }
            _ => None,
        }
    }

    fn folder(self) -> &'static str {
        match self {
            Component::Python => "python",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    /// Downloaded by MacroMagi.
    Installed {
        bytes: u64,
        detail: String,
    },
    Missing,
    /// Removed, but some files were in use: gone after a restart.
    RemovedAfterRestart,
}

#[derive(Clone)]
pub struct Components {
    root: PathBuf,
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

/// Unpacks a whole ZIP archive into `into`.
fn unzip_all(archive: &[u8], into: &Path) -> Result<(), String> {
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(archive)).map_err(|e| e.to_string())?;
    zip.extract(into).map_err(|e| e.to_string())
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

impl Components {
    pub fn new(data_dir: &Path) -> Self {
        Self { root: data_dir.join("components") }
    }

    fn dir(&self, component: Component) -> PathBuf {
        self.root.join(component.folder())
    }

    /// Deletes what was marked for removal while it was in use, and what earlier
    /// versions downloaded for AI models. Call at start, before anything is loaded.
    pub fn clean_up(&self) {
        for component in Component::ALL {
            let dir = self.dir(component);
            if dir.join(REMOVE_MARK).exists() {
                let _ = std::fs::remove_dir_all(&dir);
            }
        }
        for folder in RETIRED {
            let _ = std::fs::remove_dir_all(self.root.join(folder));
        }
    }

    fn installed(&self, component: Component) -> bool {
        let dir = self.dir(component);
        if dir.join(REMOVE_MARK).exists() {
            return false;
        }
        match component {
            Component::Python => dir.join("python.exe").is_file(),
        }
    }

    /// The Python downloaded by MacroMagi.
    pub fn python(&self) -> Option<PathBuf> {
        let path = self.dir(Component::Python).join("python.exe");
        self.installed(Component::Python).then_some(path)
    }

    /// Where a part stands.
    pub fn status(&self, component: Component) -> Status {
        let dir = self.dir(component);
        if dir.join(REMOVE_MARK).exists() {
            return Status::RemovedAfterRestart;
        }
        if self.installed(component) {
            let detail = match component {
                Component::Python => PYTHON_VERSION.to_string(),
            };
            return Status::Installed { bytes: folder_size(&dir), detail };
        }
        Status::Missing
    }

    /// Downloads and installs a part. Blocks: run it off the UI thread.
    pub fn install(&self, component: Component, progress: Report<'_>) -> Result<String, String> {
        if let Some(reason) = component.unavailable() {
            return Err(reason.to_owned());
        }
        let dir = self.dir(component);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let result = self.install_into(component, &dir, progress);
        if result.is_err() {
            let _ = std::fs::remove_dir_all(&dir);
        }
        result
    }

    fn install_into(&self, component: Component, dir: &Path, progress: Report<'_>) -> Result<String, String> {
        match component {
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
        }
    }

    /// Removes a part. Returns a message to show.
    pub fn uninstall(&self, component: Component) -> Result<String, String> {
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
    use super::*;

    #[test]
    fn retired_ai_downloads_are_cleaned_up_and_python_is_kept() {
        let dir = std::env::temp_dir().join(format!("magi-components-{}", std::process::id()));
        let components = Components::new(&dir);
        for folder in ["onnxruntime", "vision", "magidb", "python"] {
            std::fs::create_dir_all(dir.join("components").join(folder)).unwrap();
        }
        std::fs::write(dir.join("components/python/python.exe"), b"").unwrap();
        components.clean_up();
        assert!(!dir.join("components/onnxruntime").exists() && !dir.join("components/magidb").exists());
        assert_eq!(components.python(), Some(dir.join("components/python/python.exe")));
        assert!(matches!(components.status(Component::Python), Status::Installed { .. }));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
