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

/// Samples to start a new file with, so the format is plain.
pub fn template(file: &str, negative: bool) -> String {
    if !studio::is_jsonl(file) {
        return "Plain text to learn the language from. Paragraphs are separated by blank lines.\n".into();
    }
    if negative {
        return "{\"prompt\": \"Press the Enter key\", \"rejected\": \"<tool>{\\\"name\\\": \\\"press\\\", \\\"args\\\": {\\\"key\\\": \\\"Enter\\\"}}</tool><tool>{\\\"name\\\": \\\"press\\\", \\\"args\\\": {\\\"key\\\": \\\"Enter\\\"}}</tool>\"}\n".into();
    }
    "{\"prompt\": \"Press the Enter key\", \"response\": \"<think>One tap of Enter.</think><tool>{\\\"name\\\": \\\"press\\\", \\\"args\\\": {\\\"key\\\": \\\"Enter\\\"}}</tool>Done.\"}\n\
     {\"messages\": [{\"role\": \"user\", \"content\": \"Say hello\"}, {\"role\": \"assistant\", \"content\": \"Hello!\"}]}\n"
        .into()
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
        let checks: Vec<Result<&'static str, String>> =
            self.samples.iter().map(|s| studio::check_sample(jsonl, s, spec)).collect();
        let problems = checks.iter().filter(|c| c.is_err()).count();

        widgets::button_row(ui, |ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.search)
                    .hint_text(format!("{}  Search the samples", icon::MAGNIFYING_GLASS))
                    .desired_width(220.0),
            );
            widgets::segmented(
                ui,
                &mut self.filter,
                &[(Filter::All, "All"), (Filter::Problems, if problems > 0 { "With problems" } else { "Problems" })],
            );
            if ui.button(format!("{}  Add sample", icon::PLUS)).clicked() {
                let sample = if jsonl { "{\"prompt\": \"\", \"response\": \"\"}" } else { "" };
                self.samples.push(sample.to_owned());
                self.selected = Some(self.samples.len() - 1);
                self.dirty = true;
            }
            let save = egui::Button::new(format!("{}  Keep changes", icon::CHECK));
            if ui
                .add_enabled(self.dirty, save)
                .on_hover_text("Into the model; save it to keep them in the .magic file")
                .clicked()
            {
                events.push(DatasetEvent::Save(self.file.clone(), self.content()));
                self.dirty = false;
            }
        });
        ui.label(
            RichText::new(format!(
                "{} samples · {}{}",
                self.samples.len(),
                if jsonl { "one JSON object per line" } else { "paragraphs of plain text" },
                if problems > 0 { format!(" · {problems} with problems") } else { String::new() }
            ))
            .size(11.5)
            .color(if problems > 0 { theme::warning() } else { theme::text_dim() }),
        );
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
                ui.weak(if self.samples.is_empty() { "No samples yet." } else { "No sample matches." });
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
