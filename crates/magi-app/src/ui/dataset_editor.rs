//! The Text Dataset editor: the samples of one dataset file of a `.magic`
//! model, searched, filtered, checked and edited in place.

use egui::{RichText, Ui, vec2};
use egui_phosphor::regular as icon;
use magi_core::magic::TextSpec;
use magi_core::nodes::studio;

use super::widgets;
use crate::theme;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Filter {
    All,
    Problems,
}

pub struct DatasetEditor {
    /// The file's path inside the `.magic` file.
    pub file: String,
    samples: Vec<String>,
    search: String,
    filter: Filter,
    selected: Option<usize>,
    /// Changes not yet put back into the model.
    pub dirty: bool,
    pub open: bool,
}

pub enum DatasetEvent {
    /// Put the samples back into the model's file.
    Save(String, String),
}

const REQUEST: &str = r#"{"prompt": "What is the opposite of hot?", "response": "Cold."}"#;
const NEGATIVE: &str = r#"{"prompt": "What is 5 plus 7?", "rejected": "5 plus 7 is 5 plus 7 is"}"#;

/// Samples of each kind, to copy from.
const EXAMPLES: [(&str, &str); 7] = [
    ("Request and answer", REQUEST),
    (
        "Reasoning",
        r#"{"prompt": "I have 12 cookies for 4 friends. How many each?", "thinking": "Equal sharing is division: 12 / 4 = 3.", "response": "3 each.", "effort": "medium"}"#,
    ),
    (
        "Tool use",
        r#"{"tools": [{"name": "get_weather", "description": "Gets the current weather for a place.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}], "messages": [{"role": "user", "content": "Is it warm in Madrid?"}, {"role": "assistant", "content": "", "tool_calls": [{"name": "get_weather", "arguments": {"location": "Madrid"}}]}, {"role": "tool", "content": "{\"temperature_c\": 27, \"conditions\": \"sunny\"}"}, {"role": "assistant", "content": "Yes – sunny and 27 °C."}]}"#,
    ),
    (
        "Conversation",
        r#"{"messages": [{"role": "user", "content": "Count to three."}, {"role": "assistant", "content": "1, 2, 3."}, {"role": "user", "content": "Now backwards."}, {"role": "assistant", "content": "3, 2, 1."}]}"#,
    ),
    ("Answer not to give", NEGATIVE),
    (
        "Compaction",
        r#"{"context": "A long conversation about planning a trip …", "compact": "The user is planning a trip to Rome in May."}"#,
    ),
    ("Plain text", r#"{"text": "Text to learn the language from."}"#),
];

/// Samples to start a new file with, so the format is plain.
pub fn template(file: &str, negative: bool) -> String {
    if !studio::is_jsonl(file) {
        return "Plain text to learn the language from. Paragraphs are separated by blank lines.\n\n\
                Write anything the model should know how to say: how things are named, how sentences flow.\n"
            .into();
    }
    if negative {
        return format!("{NEGATIVE}\n");
    }
    format!("{REQUEST}\n")
}

impl DatasetEditor {
    pub fn new(file: &str, bytes: &[u8]) -> Self {
        Self {
            file: file.to_owned(),
            samples: studio::split_samples(file, &String::from_utf8_lossy(bytes)),
            search: String::new(),
            filter: Filter::All,
            selected: None,
            dirty: false,
            open: true,
        }
    }

    fn content(&self) -> String {
        studio::join_samples(&self.file, &self.samples)
    }

    pub fn show(&mut self, ctx: &egui::Context, spec: &TextSpec) -> Vec<DatasetEvent> {
        let mut events = Vec::new();
        let mut open = self.open;
        let title = format!(
            "{}  {}{}",
            icon::FILE_TEXT,
            self.file.rsplit('/').next().unwrap_or_default(),
            if self.dirty { " •" } else { "" }
        );
        egui::Window::new(title)
            .id(egui::Id::new("magic-dataset-editor"))
            .open(&mut open)
            .default_size(vec2(760.0, 520.0))
            .min_width(520.0)
            .resizable(true)
            .show(ctx, |ui| self.body(ui, spec, &mut events));
        if !open && self.dirty {
            // Closing keeps the changes.
            events.push(DatasetEvent::Save(self.file.clone(), self.content()));
            self.dirty = false;
        }
        self.open = open;
        events
    }

    fn body(&mut self, ui: &mut Ui, spec: &TextSpec, events: &mut Vec<DatasetEvent>) {
        let jsonl = studio::is_jsonl(&self.file);

        // The toolbar first: what it changes is checked below, in the same frame.
        widgets::button_row(ui, |ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.search)
                    .hint_text(format!("{}  Search the samples", icon::MAGNIFYING_GLASS))
                    .desired_width(200.0),
            );
            widgets::segmented(ui, &mut self.filter, &[(Filter::All, "All"), (Filter::Problems, "Problems only")]);
            if ui.button(format!("{}  Add sample", icon::PLUS)).on_hover_text("An empty sample at the end").clicked() {
                let sample = if jsonl { r#"{"prompt": "", "response": ""}"# } else { "" };
                self.samples.push(sample.to_owned());
                self.selected = Some(self.samples.len() - 1);
                self.filter = Filter::All;
                self.search.clear();
                self.dirty = true;
            }
            if jsonl {
                ui.menu_button(format!("{}  Add an example", icon::LIGHTBULB), |ui| {
                    for (name, sample) in EXAMPLES {
                        if ui.button(name).clicked() {
                            self.samples.push(sample.to_owned());
                            self.selected = Some(self.samples.len() - 1);
                            self.filter = Filter::All;
                            self.search.clear();
                            self.dirty = true;
                            ui.close();
                        }
                    }
                });
            }
            let save = egui::Button::new(format!("{}  Keep changes", icon::CHECK));
            if ui
                .add_enabled(self.dirty, save)
                .on_hover_text("Puts them into the model; save the model to keep them in its .magic file")
                .clicked()
            {
                events.push(DatasetEvent::Save(self.file.clone(), self.content()));
                self.dirty = false;
            }
        });

        let checks: Vec<Result<&'static str, String>> =
            self.samples.iter().map(|s| studio::check_sample(jsonl, s, spec)).collect();
        let problems = checks.iter().filter(|c| c.is_err()).count();
        ui.label(
            RichText::new(format!(
                "{} samples · {}{}",
                self.samples.len(),
                if jsonl { "one JSON object per line" } else { "paragraphs of plain text" },
                match problems {
                    0 => " · no problems".to_owned(),
                    1 => " · 1 with a problem".to_owned(),
                    n => format!(" · {n} with problems"),
                }
            ))
            .size(11.5)
            .color(if problems > 0 { theme::warning() } else { theme::text_dim() }),
        );
        if jsonl {
            widgets::note(
                ui,
                "Each line is one sample: {\"prompt\", \"response\"} for a request and its answer (with \"thinking\" \
                 for its reasoning, \"effort\" for its level, \"tools\" for the tools offered), {\"messages\": […]} for a \
                 conversation – every answer in it is learned, tool calls as \"tool_calls\", results as \"tool\" \
                 messages – {\"prompt\", \"rejected\"} for an answer not to give, {\"context\", \"compact\"} for \
                 compaction, or {\"text\"}. Add an example to see each.",
            );
        }
        ui.separator();

        let needle = self.search.trim().to_lowercase();
        let shown: Vec<usize> = (0..self.samples.len())
            .filter(|i| needle.is_empty() || self.samples[*i].to_lowercase().contains(&needle))
            .filter(|i| self.filter == Filter::All || checks[*i].is_err())
            .collect();

        let mut remove = None;
        let mut duplicate = None;
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            if shown.is_empty() {
                ui.weak(match (self.samples.is_empty(), self.filter) {
                    (true, _) => "No samples yet: add one, or add an example to change.",
                    (false, Filter::Problems) => "No sample has a problem.",
                    (false, Filter::All) => "No sample matches the search.",
                });
            }
            for index in shown {
                let selected = self.selected == Some(index);
                ui.push_id(index, |ui| {
                    egui::Frame::new()
                        .fill(if selected { theme::bg_widget() } else { theme::bg_raised() })
                        .corner_radius(theme::RADIUS)
                        .inner_margin(6)
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.horizontal(|ui| {
                                let (badge, color) = match &checks[index] {
                                    Ok(kind) => (format!("{}  {kind}", icon::CHECK_CIRCLE), theme::text_faint()),
                                    Err(_) => (format!("{}  Problem", icon::WARNING), theme::warning()),
                                };
                                ui.label(
                                    RichText::new(format!("#{}", index + 1)).size(11.0).color(theme::text_faint()),
                                );
                                ui.label(RichText::new(badge).size(11.0).color(color));
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if widgets::icon_button(ui, icon::TRASH, "Delete this sample").clicked() {
                                        remove = Some(index);
                                    }
                                    if widgets::icon_button(ui, icon::COPY, "Duplicate it").clicked() {
                                        duplicate = Some(index);
                                    }
                                    let edit = if selected { icon::CARET_UP } else { icon::PENCIL_SIMPLE };
                                    if widgets::icon_button(ui, edit, if selected { "Done" } else { "Edit" }).clicked()
                                    {
                                        self.selected = if selected { None } else { Some(index) };
                                    }
                                });
                            });
                            if selected {
                                let edit = ui.add(
                                    egui::TextEdit::multiline(&mut self.samples[index])
                                        .code_editor()
                                        .desired_rows(5)
                                        .desired_width(f32::INFINITY),
                                );
                                if edit.changed() {
                                    self.dirty = true;
                                }
                            } else {
                                let preview: String = self.samples[index].chars().take(240).collect();
                                let label = ui.add(
                                    egui::Label::new(RichText::new(preview).monospace().size(11.5))
                                        .wrap()
                                        .sense(egui::Sense::click()),
                                );
                                if label.on_hover_text("Click to edit").clicked() {
                                    self.selected = Some(index);
                                }
                            }
                            if let Err(problem) = &checks[index] {
                                ui.label(RichText::new(problem).size(11.0).color(theme::warning()));
                            }
                        });
                });
                ui.add_space(3.0);
            }
        });
        if let Some(index) = duplicate {
            let copy = self.samples[index].clone();
            self.samples.insert(index + 1, copy);
            self.selected = Some(index + 1);
            self.dirty = true;
        }
        if let Some(index) = remove {
            self.samples.remove(index);
            self.selected = None;
            self.dirty = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_example_is_a_valid_sample() {
        let spec =
            TextSpec { effort: Some(Default::default()), thinking: Some(Default::default()), ..TextSpec::default() };
        for (name, sample) in EXAMPLES {
            assert!(studio::check_sample(true, sample, &spec).is_ok(), "{name}: {sample}");
        }
        assert!(studio::check_sample(true, template("a.jsonl", true).trim(), &spec).is_ok());
    }
}
