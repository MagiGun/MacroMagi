//! Native tools MacroMagi carries out itself (`"run": {"kind": "host"}`): what needs the system
//! rather than Python.
//!
//! `view_image` reads a picture for a Magi model, which never sees pixels: the text in it, with
//! the text recognition built into Windows (the same as the Text Scan block), and what it shows,
//! from a small vision model (`nt/view_image/describe.py`, installed under Settings › Runtime as
//! Image descriptions).

use magi_core::geom::Point;
use magi_core::host::Frame;
use magi_core::native_tools::{NativeTool, RunSpec};
use serde_json::{Value, json};

/// Registers the host tools with the agent loop.
pub fn register() {
    magi_ml::tools::set_host_tool("view_image", Box::new(view_image));
}

fn view_image(args: &Value) -> Result<Value, String> {
    let raw = args.get("path").and_then(Value::as_str).unwrap_or_default().trim();
    if raw.is_empty() {
        return Err("say which picture to look at".into());
    }
    let path = expand_home(raw);
    let picture = image::open(&path).map_err(|e| format!("could not open the picture {raw}: {e}"))?.to_rgba8();
    let (width, height) = picture.dimensions();
    let mut result = json!({ "path": raw, "width": width, "height": height });

    // The text in it, line by line.
    let frame = Frame::new(Point::default(), width, height, picture.into_raw());
    match magi_platform::ocr::read_text(&frame, "") {
        Ok(words) => {
            let mut lines: Vec<String> = Vec::new();
            for word in words {
                while lines.len() <= word.line {
                    lines.push(String::new());
                }
                let line = &mut lines[word.line];
                if !line.is_empty() {
                    line.push(' ');
                }
                line.push_str(&word.text);
            }
            let text = lines.join("\n");
            result["text"] = if text.trim().is_empty() { json!("(no text)") } else { json!(text) };
        }
        Err(error) => result["text_error"] = json!(error.0),
    }

    // What it shows, from the small vision model.
    let files = magi_core::native_tools::files("view_image", &Default::default());
    let describer = NativeTool {
        name: "view_image".into(),
        run: RunSpec { kind: "python".into(), entry: "describe.py".into(), timeout_ms: 170_000 },
        ..NativeTool::default()
    };
    match magi_ml::tools::run_native(&describer, &files, &json!({ "path": path.display().to_string() })) {
        Ok(found) => match found.get("description").and_then(Value::as_str) {
            Some(description) => result["description"] = json!(description),
            None => {
                result["description_error"] = found.get("error").cloned().unwrap_or(json!("no description"));
            }
        },
        Err(error) => result["description_error"] = json!(error),
    }
    Ok(result)
}

fn expand_home(path: &str) -> std::path::PathBuf {
    if let Some(rest) = path.strip_prefix("~/").or_else(|| path.strip_prefix("~\\"))
        && let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))
    {
        return std::path::PathBuf::from(home).join(rest);
    }
    std::path::PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_picture_is_read_and_its_size_given() {
        let dir = std::env::temp_dir().join(format!("magi-view-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("red.png");
        image::RgbaImage::from_pixel(8, 6, image::Rgba([200, 20, 20, 255])).save(&path).unwrap();
        let seen = view_image(&json!({ "path": path.display().to_string() })).unwrap();
        assert_eq!((seen["width"].as_u64(), seen["height"].as_u64()), (Some(8), Some(6)));
        assert!(seen.get("text").is_some() || seen.get("text_error").is_some());
        assert!(view_image(&json!({ "path": dir.join("missing.png").display().to_string() })).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
