//! Window nodes: bringing a program to the front.

use serde::{Deserialize, Serialize};

use super::{Category, DataType, NodeDef, ParamDef, ParamKind, Registry, ids, ports};
use crate::text::format_ms;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowMatch {
    /// The window's title contains the text.
    #[default]
    Title,
    /// The program's executable name contains the text (`game.exe`).
    Process,
}

/// How a window occupies the screen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScreenKind {
    /// Work it out from the window itself.
    #[default]
    Smart,
    /// An ordinary window with a frame.
    Windowed,
    /// A borderless window covering the whole screen ("fullscreen windowed").
    Borderless,
    /// Exclusive fullscreen: the program owns the display. Switching to it
    /// takes the longest, and it minimises when it loses focus.
    Exclusive,
}

impl ScreenKind {
    pub fn id(self) -> &'static str {
        match self {
            ScreenKind::Smart => "smart",
            ScreenKind::Windowed => "windowed",
            ScreenKind::Borderless => "borderless",
            ScreenKind::Exclusive => "exclusive",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ScreenKind::Smart => "Detect automatically",
            ScreenKind::Windowed => "Windowed",
            ScreenKind::Borderless => "Borderless fullscreen",
            ScreenKind::Exclusive => "Exclusive fullscreen",
        }
    }

    /// How long a switch to a window of this kind takes to settle.
    pub fn settle_ms(self) -> f64 {
        match self {
            ScreenKind::Smart | ScreenKind::Windowed => 150.0,
            ScreenKind::Borderless => 300.0,
            ScreenKind::Exclusive => 1500.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FocusWindowParams {
    /// Text to look for in the title or program name.
    pub target: String,
    pub match_by: WindowMatch,
    pub screen: ScreenKind,
    /// Keep looking for the window this long before giving up. Zero looks once.
    pub timeout_ms: f64,
}

impl Default for FocusWindowParams {
    fn default() -> Self {
        Self { target: String::new(), match_by: WindowMatch::Title, screen: ScreenKind::Smart, timeout_ms: 2000.0 }
    }
}

pub(super) fn register(registry: &mut Registry) {
    registry.register(
        NodeDef::new(
            ids::FOCUS_WINDOW,
            "Switch to Window",
            Category::Window,
            "Brings a program's window to the front (restoring it if it is minimised) and waits \
             until it is ready for input. Knows how windowed, borderless and exclusive fullscreen \
             programs behave, or works it out itself.",
        )
        .keywords(&["focus", "activate", "window", "switch", "game", "alt tab", "screen", "fullscreen"])
        .flow_in()
        .flow_out(ports::OUT, "Switched")
        .flow_out("missing", "Not found")
        .data_out("screen", "Screen type", DataType::Text)
        .param(
            ParamDef::new("target", "Window", ParamKind::Text { multiline: false })
                .help("Part of the window's title, or of the program's name, e.g. “Minecraft” or “game.exe”."),
        )
        .param(ParamDef::new(
            "match_by",
            "Find it by",
            ParamKind::choice(&[("title", "Window title"), ("process", "Program name")]),
        ))
        .param(
            ParamDef::new(
                "screen",
                "Screen type",
                ParamKind::choice(&[
                    ("smart", "Detect automatically"),
                    ("windowed", "Windowed"),
                    ("borderless", "Borderless fullscreen"),
                    ("exclusive", "Exclusive fullscreen"),
                ]),
            )
            .help("Exclusive fullscreen programs need longer to come back after a switch."),
        )
        .param(ParamDef::new("timeout_ms", "Keep looking for", ParamKind::Duration))
        .defaults(&FocusWindowParams::default())
        .summarize(|node, _| {
            let p: FocusWindowParams = node.params_or_default();
            if p.target.trim().is_empty() {
                return vec!["No window chosen".into()];
            }
            let mut lines = vec![format!("“{}”", p.target)];
            lines.push(p.screen.label().to_owned());
            if p.timeout_ms > 0.0 {
                lines.push(format!("Waits up to {}", format_ms(p.timeout_ms)));
            }
            lines
        }),
    );

    registry.register(
        NodeDef::new(
            ids::SWITCH_DISPLAY,
            "Switch to Display",
            Category::Window,
            "Moves the pointer to the middle of a display, and can click there so what is on that display \
             gets the keyboard – for games and programs spread over several displays.",
        )
        .keywords(&["display", "monitor", "screen", "switch", "move", "focus", "second screen"])
        .flow_in()
        .flow_out(ports::OUT, "")
        .param(ParamDef::new("display", "Display", ParamKind::Display))
        .param(
            ParamDef::new("click", "Click there", ParamKind::Bool)
                .help("A left click in the middle of the display, so the window there comes to the front."),
        )
        .defaults(&SwitchDisplayParams::default())
        .summarize(|node, _| {
            let p: SwitchDisplayParams = node.params_or_default();
            let mut lines = vec![format!("Display {}", p.display + 1)];
            if p.click {
                lines.push("Clicks in the middle".into());
            }
            lines
        }),
    );

    registry.register(
        NodeDef::new(
            ids::CURRENT_WINDOW,
            "Current Window",
            Category::Window,
            "The window in front: its title, its program, the display it is on and its middle – and the \
             display the pointer is on. Displays are numbered from 1.",
        )
        .keywords(&["window", "current", "active", "foreground", "title", "program", "display", "screen"])
        .data_out("title", "Title", DataType::Text)
        .data_out("program", "Program", DataType::Text)
        .data_out("display", "Its display", DataType::Number)
        .data_out("pointer_display", "Pointer's display", DataType::Number)
        .data_out("position", "Middle", DataType::Point)
        .defaults(&serde_json::Map::new())
        .summarize(|_, _| vec!["The window in front".into()]),
    );
}

/// Settings of a **Switch to Display** block.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SwitchDisplayParams {
    pub display: u8,
    pub click: bool,
}
