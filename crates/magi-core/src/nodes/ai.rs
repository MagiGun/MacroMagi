//! The AI block: **AI Vision**, which learns from example pictures of a few
//! groups which one the screen shows. It is a value block, evaluated whenever a
//! block reads it, like Image Detection.

use serde::{Deserialize, Serialize};

use super::{Category, DataType, NodeDef, ParamDef, ParamKind, Registry, ids};
use crate::geom::Rect;

/// One group an AI Vision block tells apart: its name and its example pictures.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VisionGroup {
    pub name: String,
    /// Ids of image assets.
    pub images: Vec<String>,
}

/// Settings of an **AI Vision** block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AiVisionParams {
    pub groups: Vec<VisionGroup>,
    /// Where to look. `None` looks at the whole of `display`.
    pub region: Option<Rect>,
    /// The display looked at when there is no area (0 = primary).
    pub display: u8,
    /// How sure, `0.0..=1.0`, the block must be of a group for "Found".
    pub threshold: f64,
}

impl Default for AiVisionParams {
    fn default() -> Self {
        Self {
            groups: vec![
                VisionGroup { name: "Group 1".into(), images: Vec::new() },
                VisionGroup { name: "Group 2".into(), images: Vec::new() },
            ],
            region: None,
            display: 0,
            threshold: 0.8,
        }
    }
}

impl AiVisionParams {
    /// The groups that have example pictures.
    pub fn trained_groups(&self) -> impl Iterator<Item = &VisionGroup> {
        self.groups.iter().filter(|g| !g.images.is_empty())
    }

    /// What is missing before it can tell the screen apart, if anything.
    pub fn problem(&self) -> Option<&'static str> {
        (self.trained_groups().count() < 2)
            .then_some("AI Vision needs at least two groups with example pictures to tell apart")
    }
}

pub(super) fn register(registry: &mut Registry) {
    registry.register(
        NodeDef::new(
            ids::AI_VISION,
            "AI Vision",
            Category::Ai,
            "Learns from example pictures which of a few groups the screen shows – a menu or the game, a \
             full bar or an empty one, whose turn it is – even where Image Detection cannot, because the \
             screen never looks exactly the same. Make two or more groups and add pictures of each: \
             captured from the screen or loaded from files. “Found” is true when it is at least as sure \
             as the confidence setting; “Group” says which one.",
        )
        .keywords(&[
            "ai",
            "vision",
            "recognise",
            "recognize",
            "classify",
            "learn",
            "train",
            "model",
            "see",
            "groups",
            "pictures",
            "machine learning",
            "detect",
            "state",
        ])
        .data_out("found", "Found", DataType::Bool)
        .data_out("group", "Group", DataType::Text)
        .data_out("confidence", "Confidence", DataType::Number)
        .data_out("likeliest", "Likeliest group", DataType::Text)
        .data_in("picture", "Picture", DataType::Image)
        .param(ParamDef::new("groups", "Groups", ParamKind::VisionGroups).help(
            "Each group is one thing to recognise. Give each a few pictures of it as it can look – \
                 more, and more varied, pictures make it surer. It learns from them by itself, in a \
                 moment, when the macro runs.",
        ))
        .param(
            ParamDef::new("region", "Area", ParamKind::Region)
                .help("Where to look – the same part of the screen the example pictures show."),
        )
        .param(
            ParamDef::new("display", "Display", ParamKind::Display)
                .help("The display looked at when there is no area."),
        )
        .param(
            ParamDef::new("threshold", "Confidence", ParamKind::Percent)
                .help("How sure it must be of a group for “Found”. Below it, “Group” is empty."),
        )
        .defaults(&AiVisionParams::default())
        .summarize(|node, _| {
            let p: AiVisionParams = node.params_or_default();
            let pictures: usize = p.groups.iter().map(|g| g.images.len()).sum();
            let names: Vec<&str> = p.groups.iter().map(|g| g.name.as_str()).collect();
            let headline = match p.problem() {
                Some(_) if pictures == 0 => "No pictures yet".to_owned(),
                Some(_) => "Needs pictures in two groups".to_owned(),
                None => format!("{} · {pictures} pictures", names.join(" / ")),
            };
            let area = match p.region {
                Some(r) => format!("In {}×{} at {}, {}", r.w, r.h, r.x, r.y),
                None if p.display == 0 => "Whole screen".into(),
                None => format!("Whole display {}", p.display + 1),
            };
            vec![headline, format!("Sure ≥ {:.0}%", p.threshold * 100.0), area]
        }),
    );
}
