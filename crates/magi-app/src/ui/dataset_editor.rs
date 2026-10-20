//! The Text Dataset editor: the samples of one dataset file of a `.magic`
//! model, shown as the chat they teach (and edited there), as the model reads
//! them in training, or as their JSON lines – searched, filtered, selected
//! together, given reasoning efforts and scanned for issues. It can pop out
//! into a window of its own.

use std::collections::{BTreeMap, BTreeSet};

use egui::{Align, Color32, Layout, RichText, Stroke, Ui, vec2};
use egui_phosphor::regular as icon;
use magi_core::dataset;
use magi_core::magic::{TextSpec, spaced_json};
use magi_core::nodes::studio;
use serde_json::{Map, Value, json};

use super::widgets;
use crate::theme;

/// Samples on one page of the list.
const PAGE: usize = 25;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Filter {
    All,
    Problems,
    Selected,
    /// The samples of one finding of the last scan.
    Finding(usize),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum View {
    /// As the chat it teaches, edited in place.
    Chat,
    /// As training writes it out for the model, chat tags and all.
    Model,
    /// The JSON lines themselves.
    Raw,
}

/// What each sample is (its kind), or what is wrong with it.
type Checks = Vec<Result<&'static str, String>>;

/// A change of effort for the selected samples.
enum EffortChange {
    /// Back to the model's default level.
    Default,
    /// One level added to them all, or (when they all have it) taken off.
    Toggle(String, bool),
}

/// What a scan found: issues, each with the samples it concerns, and the
/// words fine-tuning uses that pretraining never does.
struct Scan {
    revision: u64,
    findings: Vec<Finding>,
    unknown: Vec<(String, usize)>,
    notes: Vec<String>,
}

struct Finding {
    title: String,
    detail: String,
    samples: Vec<usize>,
}

pub struct DatasetEditor {
    /// The file's path inside the `.magic` file.
    pub file: String,
    samples: Vec<String>,
    search: String,
    filter: Filter,
    view: View,
    selected: BTreeSet<usize>,
    /// The sample clicked last, for selecting a range with Shift.
    anchor: Option<usize>,
    /// Samples shown as JSON while the rest are in the chat view.
    raw: BTreeSet<usize>,
    page: usize,
    /// Bumped with every change, so caches and the scan know they are old.
    revision: u64,
    checks: Option<(u64, TextSpec, Checks)>,
    scan: Option<Scan>,
    /// The sample to bring into view once it is drawn.
    reveal: Option<usize>,
    /// Shown in a window of its own, outside the main window.
    popped_out: bool,
    /// Changes not yet put back into the model.
    pub dirty: bool,
    pub open: bool,
}

pub enum DatasetEvent {
    /// Put the samples back into the model's file.
    Save(String, String),
}

/// What the editor knows of the rest of the model.
pub struct DatasetContext<'a> {
    pub spec: &'a TextSpec,
    /// Every dataset file of the model, by its path inside the `.magic` file.
    pub data: &'a BTreeMap<String, Vec<u8>>,
    /// The files each training stage learns from, with the stage's kind, in order.
    pub stages: Vec<(String, Vec<String>)>,
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

/// What a card asks of the list once it is drawn.
enum Action {
    Remove(usize),
    Duplicate(usize),
    Copy(usize),
    ToggleRaw(usize),
    Select(usize, bool),
}

impl DatasetEditor {
    pub fn new(file: &str, bytes: &[u8]) -> Self {
        Self {
            file: file.to_owned(),
            samples: studio::split_samples(file, &String::from_utf8_lossy(bytes)),
            search: String::new(),
            filter: Filter::All,
            view: View::Chat,
            selected: BTreeSet::new(),
            anchor: None,
            raw: BTreeSet::new(),
            page: 0,
            revision: 0,
            checks: None,
            scan: None,
            reveal: None,
            popped_out: false,
            dirty: false,
            open: true,
        }
    }

    /// Shows the samples as `chat`, `model` (Model's view) or `raw`.
    pub fn set_view(&mut self, view: &str) {
        self.view = match view {
            "model" => View::Model,
            "raw" => View::Raw,
            _ => View::Chat,
        };
    }

    fn content(&self) -> String {
        studio::join_samples(&self.file, &self.samples)
    }

    fn jsonl(&self) -> bool {
        studio::is_jsonl(&self.file)
    }

    fn changed(&mut self) {
        self.dirty = true;
        self.revision += 1;
    }

    pub fn show(&mut self, ctx: &egui::Context, cx: &DatasetContext<'_>) -> Vec<DatasetEvent> {
        let mut events = Vec::new();
        let name = self.file.rsplit('/').next().unwrap_or_default().to_owned();
        let mut open = self.open;
        if self.popped_out {
            let builder = egui::ViewportBuilder::default()
                .with_title(format!("{name} • MacroMagi"))
                .with_inner_size([1000.0, 720.0])
                .with_min_inner_size([560.0, 380.0]);
            let mut closed = false;
            ctx.show_viewport_immediate(
                egui::ViewportId::from_hash_of("magic-dataset-editor"),
                builder,
                |ui, class| {
                    match class {
                        egui::ViewportClass::EmbeddedWindow => self.body(ui, cx, &mut events),
                        _ => {
                            egui::CentralPanel::default()
                                .frame(egui::Frame::new().fill(theme::bg_panel().to_opaque()).inner_margin(12))
                                .show(ui, |ui| self.body(ui, cx, &mut events));
                        }
                    }
                    closed = ui.input(|i| i.viewport().close_requested());
                },
            );
            if closed {
                open = false;
            }
        } else {
            let title = format!("{}  {name}{}", icon::FILE_TEXT, if self.dirty { " •" } else { "" });
            egui::Window::new(title)
                .id(egui::Id::new("magic-dataset-editor"))
                .open(&mut open)
                .default_size(vec2(860.0, 600.0))
                .min_width(560.0)
                .resizable(true)
                .show(ctx, |ui| self.body(ui, cx, &mut events));
        }
        if !open && self.dirty {
            // Closing keeps the changes.
            events.push(DatasetEvent::Save(self.file.clone(), self.content()));
            self.dirty = false;
        }
        self.open = open;
        events
    }

    fn checks_fresh(&self) -> bool {
        self.checks.as_ref().is_some_and(|(revision, ..)| *revision == self.revision)
    }

    /// The checks of every sample, and the samples the filter and search show.
    fn shown(&mut self, spec: &TextSpec) -> (Checks, Vec<usize>) {
        let checks = self.checks(spec);
        let needle = self.search.trim().to_lowercase();
        let finding: Option<BTreeSet<usize>> = match self.filter {
            Filter::Finding(i) => {
                self.scan.as_ref().and_then(|s| s.findings.get(i)).map(|f| f.samples.iter().copied().collect())
            }
            _ => None,
        };
        let shown = (0..self.samples.len())
            .filter(|i| match self.filter {
                Filter::All => true,
                Filter::Problems => checks[*i].is_err(),
                Filter::Selected => self.selected.contains(i),
                Filter::Finding(_) => finding.as_ref().is_some_and(|f| f.contains(i)),
            })
            .filter(|i| needle.is_empty() || self.samples[*i].to_lowercase().contains(&needle))
            .collect();
        (checks, shown)
    }

    /// What each sample is, or what is wrong with it; checked again only after a change.
    fn checks(&mut self, spec: &TextSpec) -> Checks {
        let fresh =
            self.checks.as_ref().is_some_and(|(revision, checked, _)| *revision == self.revision && checked == spec);
        if !fresh {
            let jsonl = self.jsonl();
            let checks = self.samples.iter().map(|s| studio::check_sample(jsonl, s, spec)).collect();
            self.checks = Some((self.revision, spec.clone(), checks));
        }
        self.checks.as_ref().map(|(_, _, checks)| checks.clone()).unwrap_or_default()
    }

    fn body(&mut self, ui: &mut Ui, cx: &DatasetContext<'_>, events: &mut Vec<DatasetEvent>) {
        let jsonl = self.jsonl();
        let checks = self.checks(cx.spec);
        let problems = checks.iter().filter(|c| c.is_err()).count();
        if problems == 0 && self.filter == Filter::Problems {
            self.filter = Filter::All;
        }
        if self.selected.is_empty() && self.filter == Filter::Selected {
            self.filter = Filter::All;
        }

        self.toolbar(ui, cx, problems, events);
        if !jsonl {
            self.view = View::Chat;
        }

        // The samples shown: searched, filtered, a page at a time. The bars above the list
        // may change the samples, so what is shown is worked out again after them.
        let (checks, shown) = self.shown(cx.spec);
        self.status_line(ui, problems, &shown);
        if !self.selected.is_empty() {
            self.selection_bar(ui, cx);
        }
        self.scan_panel(ui);
        ui.separator();
        let (checks, shown) = if checks.len() == self.samples.len() && self.checks_fresh() {
            (checks, shown)
        } else {
            self.shown(cx.spec)
        };
        let pages = shown.len().div_ceil(PAGE).max(1);
        if let Some(reveal) = self.reveal
            && let Some(position) = shown.iter().position(|i| *i == reveal)
        {
            self.page = position / PAGE;
        }
        self.page = self.page.min(pages - 1);
        let page: Vec<usize> = shown.iter().skip(self.page * PAGE).take(PAGE).copied().collect();

        let mut actions = Vec::new();
        let footer = if pages > 1 { 34.0 } else { 0.0 };
        egui::ScrollArea::vertical()
            .id_salt(("dataset-samples", self.page, self.filter == Filter::All))
            .auto_shrink([false, false])
            .max_height((ui.available_height() - footer).max(120.0))
            .show(ui, |ui| {
                if page.is_empty() {
                    ui.add_space(12.0);
                    ui.weak(match (self.samples.is_empty(), self.filter) {
                        (true, _) => "No samples yet: add one, or add an example to change.",
                        (false, Filter::Problems) => "No sample has a problem.",
                        (false, _) => "No sample matches.",
                    });
                }
                for index in page {
                    let reveal = self.reveal == Some(index);
                    let response = ui.push_id(index, |ui| self.card(ui, cx, index, &checks[index], &mut actions));
                    if reveal {
                        response.response.scroll_to_me(Some(Align::Center));
                        self.reveal = None;
                    }
                    ui.add_space(6.0);
                }
            });
        if pages > 1 {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui.add_enabled(self.page > 0, egui::Button::new(format!("{}  Previous", icon::CARET_LEFT))).clicked()
                {
                    self.page -= 1;
                }
                ui.label(
                    RichText::new(format!("Page {} of {pages} · {} samples", self.page + 1, shown.len()))
                        .size(11.5)
                        .color(theme::text_dim()),
                );
                if ui
                    .add_enabled(self.page + 1 < pages, egui::Button::new(format!("Next  {}", icon::CARET_RIGHT)))
                    .clicked()
                {
                    self.page += 1;
                }
            });
        }
        self.apply(ui, actions, &shown);
    }

    fn toolbar(&mut self, ui: &mut Ui, cx: &DatasetContext<'_>, problems: usize, events: &mut Vec<DatasetEvent>) {
        let jsonl = self.jsonl();
        widgets::button_row(ui, |ui| {
            let search = ui.add(
                egui::TextEdit::singleline(&mut self.search)
                    .hint_text(format!("{}  Search", icon::MAGNIFYING_GLASS))
                    .desired_width(170.0),
            );
            if search.changed() {
                self.page = 0;
            }
            let mut filters = vec![(Filter::All, "All".to_owned())];
            if problems > 0 || self.filter == Filter::Problems {
                filters.push((Filter::Problems, format!("Problems ({problems})")));
            }
            if !self.selected.is_empty() {
                filters.push((Filter::Selected, format!("Selected ({})", self.selected.len())));
            }
            if let (Filter::Finding(i), Some(scan)) = (self.filter, &self.scan)
                && let Some(finding) = scan.findings.get(i)
            {
                filters.push((self.filter, finding.title.clone()));
            }
            let options: Vec<(Filter, &str)> = filters.iter().map(|(f, l)| (*f, l.as_str())).collect();
            if widgets::segmented(ui, &mut self.filter, &options) {
                self.page = 0;
            }
            if jsonl {
                ui.add_space(6.0);
                widgets::segmented(
                    ui,
                    &mut self.view,
                    &[(View::Chat, "Chat"), (View::Model, "Model's view"), (View::Raw, "Raw")],
                );
            }
        });
        widgets::button_row(ui, |ui| {
            if ui
                .button(format!("{}  Scan for issues", icon::MAGNIFYING_GLASS_PLUS))
                .on_hover_text(
                    "Looks through the samples for what training would skip, misread or learn badly: problems, \
                     repeats, samples longer than the context window, reasoning efforts without samples, and words \
                     these samples use that the pretraining text never does",
                )
                .clicked()
            {
                self.scan = Some(self.scan(cx));
            }
            ui.menu_button(format!("{}  Add", icon::PLUS), |ui| {
                let mut added = None;
                if jsonl {
                    if ui.button("Request and answer").clicked() {
                        added = Some(r#"{"prompt": "", "response": ""}"#.to_owned());
                    }
                    if ui.button("Conversation").clicked() {
                        added = Some(
                            r#"{"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]}"#
                                .to_owned(),
                        );
                    }
                    if ui.button("Tool use").clicked() {
                        let tool = cx.spec.tools.first().map_or("tool", |t| t.name.as_str());
                        let sample = json!({ "messages": [
                            { "role": "user", "content": "" },
                            { "role": "assistant", "content": "", "tool_calls": [{ "name": tool, "arguments": {} }] },
                            { "role": "tool", "content": "" },
                            { "role": "assistant", "content": "" },
                        ] });
                        added = sample.as_object().map(dataset::to_line);
                    }
                    if ui.button("Answer not to give").clicked() {
                        added = Some(r#"{"prompt": "", "rejected": ""}"#.to_owned());
                    }
                    ui.separator();
                    ui.menu_button(format!("{}  An example", icon::LIGHTBULB), |ui| {
                        for (name, sample) in EXAMPLES {
                            if ui.button(name).clicked() {
                                added = Some(sample.to_owned());
                            }
                        }
                    });
                } else if ui.button("Paragraph").clicked() {
                    added = Some("New paragraph.".to_owned());
                }
                if let Some(sample) = added {
                    self.samples.push(sample);
                    let index = self.samples.len() - 1;
                    self.raw.remove(&index);
                    self.reveal = Some(index);
                    self.filter = Filter::All;
                    self.search.clear();
                    self.changed();
                    ui.close();
                }
            });
            let save = egui::Button::new(format!("{}  Keep changes", icon::CHECK));
            if ui
                .add_enabled(self.dirty, save)
                .on_hover_text("Puts them into the model; save the workspace to keep them in its .magic file")
                .clicked()
            {
                events.push(DatasetEvent::Save(self.file.clone(), self.content()));
                self.dirty = false;
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let (glyph, hint) = if self.popped_out {
                    (icon::ARROW_SQUARE_IN, "Back into the main window")
                } else {
                    (icon::ARROW_SQUARE_OUT, "Into a window of its own")
                };
                if widgets::icon_button(ui, glyph, hint).clicked() {
                    self.popped_out = !self.popped_out;
                }
            });
        });
    }

    fn status_line(&mut self, ui: &mut Ui, problems: usize, shown: &[usize]) {
        let jsonl = self.jsonl();
        ui.horizontal(|ui| {
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
            if jsonl && !shown.is_empty() {
                if self.selected.is_empty() {
                    ui.label(
                        RichText::new("· tick samples to give them efforts, copy or remove them together")
                            .size(11.5)
                            .color(theme::text_faint()),
                    );
                }
                let all = shown.iter().all(|i| self.selected.contains(i));
                let label =
                    if all { "Unselect the shown".to_owned() } else { format!("Select the {} shown", shown.len()) };
                if ui.small_button(label).clicked() {
                    if all {
                        for i in shown {
                            self.selected.remove(i);
                        }
                    } else {
                        self.selected.extend(shown.iter().copied());
                    }
                }
            }
        });
    }

    /// What can be done to the selected samples at once.
    fn selection_bar(&mut self, ui: &mut Ui, cx: &DatasetContext<'_>) {
        let jsonl = self.jsonl();
        egui::Frame::new().fill(theme::bg_widget()).corner_radius(theme::RADIUS).inner_margin(6).show(ui, |ui| {
            ui.set_width(ui.available_width());
            widgets::button_row(ui, |ui| {
                ui.label(RichText::new(format!("{} selected", self.selected.len())).strong());
                if jsonl && let Some(effort) = &cx.spec.effort {
                    ui.add_space(6.0);
                    ui.label(RichText::new("Reasoning effort").size(11.5).color(theme::text_dim()));
                    let items: Vec<Option<Map<String, Value>>> =
                        self.selected.iter().map(|i| object(&self.samples[*i]).filter(dataset::is_request)).collect();
                    let has = |level: &str| {
                        items.iter().flatten().filter(|item| dataset::efforts(item).iter().any(|l| l == level)).count()
                    };
                    let requests = items.iter().flatten().count();
                    let defaults = items.iter().flatten().filter(|item| dataset::efforts(item).is_empty()).count();
                    let mut set = None;
                    if chip(ui, "Default", defaults == requests && requests > 0, defaults > 0 && defaults < requests)
                        .on_hover_text(format!("At the model's default level ({})", effort.default))
                        .clicked()
                    {
                        set = Some(EffortChange::Default);
                    }
                    for level in &effort.levels {
                        let count = has(level);
                        let all = count == requests && requests > 0;
                        if chip(ui, level, all, count > 0 && !all)
                            .on_hover_text(if all {
                                format!("Take “{level}” off them")
                            } else {
                                format!("Learn them at “{level}” too")
                            })
                            .clicked()
                        {
                            set = Some(EffortChange::Toggle(level.clone(), all));
                        }
                    }
                    if let Some(set) = set {
                        let order = &effort.levels;
                        for index in self.selected.clone() {
                            if let Some(mut item) = object(&self.samples[index]).filter(dataset::is_request) {
                                let mut levels = dataset::efforts(&item);
                                match &set {
                                    EffortChange::Default => levels.clear(),
                                    EffortChange::Toggle(level, true) => levels.retain(|l| l != level),
                                    EffortChange::Toggle(level, false) if !levels.contains(level) => {
                                        levels.push(level.clone());
                                    }
                                    EffortChange::Toggle(..) => {}
                                }
                                levels.sort_by_key(|l| order.iter().position(|o| o == l).unwrap_or(usize::MAX));
                                dataset::set_efforts(&mut item, &levels);
                                self.samples[index] = dataset::to_line(&item);
                            }
                        }
                        self.changed();
                    }
                }
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if widgets::icon_button(ui, icon::X, "Clear the selection").clicked() {
                        self.selected.clear();
                    }
                    if widgets::icon_button(ui, icon::TRASH, "Remove them").clicked() {
                        for index in self.selected.iter().rev() {
                            self.samples.remove(*index);
                        }
                        self.selected.clear();
                        self.raw.clear();
                        self.changed();
                    }
                    if widgets::icon_button(ui, icon::COPY_SIMPLE, "Copy them (as their lines)").clicked() {
                        let lines: Vec<&str> = self.selected.iter().map(|i| self.samples[*i].as_str()).collect();
                        ui.ctx().copy_text(lines.join(if jsonl { "\n" } else { "\n\n" }));
                    }
                    if widgets::icon_button(ui, icon::COPY, "Duplicate them, each after itself").clicked() {
                        let mut copies = BTreeSet::new();
                        for (shift, index) in self.selected.clone().into_iter().enumerate() {
                            let at = index + shift;
                            self.samples.insert(at + 1, self.samples[at].clone());
                            copies.insert(at + 1);
                        }
                        self.selected = copies;
                        self.raw.clear();
                        self.changed();
                    }
                });
            });
        });
    }

    /// The last scan's findings, with a way to show the samples of each.
    fn scan_panel(&mut self, ui: &mut Ui) {
        let Some(scan) = &self.scan else { return };
        let stale = scan.revision != self.revision;
        let mut show = None;
        let mut close = false;
        let mut again = false;
        egui::Frame::new()
            .fill(theme::bg_raised())
            .stroke(Stroke::new(1.0, theme::stroke()))
            .corner_radius(theme::RADIUS)
            .inner_margin(8)
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    let title = if scan.findings.is_empty() { "No issues found" } else { "Issues found" };
                    ui.label(RichText::new(title).strong());
                    if stale {
                        ui.label(RichText::new("· changed since").size(11.5).color(theme::text_faint()));
                        if ui.small_button("Scan again").clicked() {
                            again = true;
                        }
                    }
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if widgets::icon_button(ui, icon::X, "Close the scan").clicked() {
                            close = true;
                        }
                    });
                });
                egui::ScrollArea::vertical().id_salt("dataset-scan").max_height(200.0).show(ui, |ui| {
                    for (index, finding) in scan.findings.iter().enumerate() {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(
                                RichText::new(format!("{}  {}", icon::WARNING, finding.title)).color(theme::warning()),
                            );
                            if !finding.samples.is_empty()
                                && ui.small_button(format!("Show {}", finding.samples.len())).clicked()
                            {
                                show = Some(index);
                            }
                        });
                        widgets::note(ui, &finding.detail);
                        if finding.title.starts_with("Words") {
                            ui.horizontal_wrapped(|ui| {
                                ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
                                for (word, count) in scan.unknown.iter().take(80) {
                                    word_chip(ui, word, *count);
                                }
                                if scan.unknown.len() > 80 {
                                    ui.label(
                                        RichText::new(format!("and {} more", scan.unknown.len() - 80))
                                            .size(11.0)
                                            .color(theme::text_faint()),
                                    );
                                }
                            });
                            if ui.small_button(format!("{}  Copy the words", icon::COPY_SIMPLE)).clicked() {
                                let words: Vec<&str> = scan.unknown.iter().map(|(w, _)| w.as_str()).collect();
                                ui.ctx().copy_text(words.join("\n"));
                            }
                        }
                        ui.add_space(4.0);
                    }
                    for note in &scan.notes {
                        widgets::notice(ui, icon::INFO, note.clone(), theme::text_dim());
                    }
                });
            });
        if let Some(index) = show {
            self.filter = Filter::Finding(index);
            self.page = 0;
        }
        if again {
            self.scan = None;
            self.filter = Filter::All;
        }
        if close {
            self.scan = None;
            if matches!(self.filter, Filter::Finding(_)) {
                self.filter = Filter::All;
            }
        }
    }

    /// Looks through the samples (and the model's other datasets) for issues.
    fn scan(&mut self, cx: &DatasetContext<'_>) -> Scan {
        let jsonl = self.jsonl();
        let checks = self.checks(cx.spec);
        let mut findings = Vec::new();
        let mut notes = Vec::new();
        let mut finding = |title: String, detail: &str, samples: Vec<usize>| {
            findings.push(Finding { title, detail: detail.to_owned(), samples });
        };

        let broken: Vec<usize> = (0..checks.len()).filter(|i| checks[*i].is_err()).collect();
        if !broken.is_empty() {
            finding(
                plural(broken.len(), "sample with a problem", "samples with problems"),
                "Training skips or misreads them: each says what is wrong under it.",
                broken,
            );
        }

        let mut seen: BTreeMap<String, usize> = BTreeMap::new();
        let mut repeats = Vec::new();
        for (index, sample) in self.samples.iter().enumerate() {
            let key: String = sample.split_whitespace().collect::<Vec<_>>().join(" ");
            if seen.insert(key, index).is_some() {
                repeats.push(index);
            }
        }
        if !repeats.is_empty() {
            finding(
                plural(repeats.len(), "repeated sample", "repeated samples"),
                "The same sample more than once: it is learned more, and the rest less. Shown are the second and \
                 later ones.",
                repeats,
            );
        }

        // Longer than the context window: training cuts them (a token is a byte, or a tag).
        let context = cx.spec.context as usize;
        if context > 0 {
            let effort = cx.spec.effort.as_ref().map(|e| e.default.clone()).unwrap_or_default();
            let long: Vec<usize> = (0..self.samples.len())
                .filter(|i| {
                    let text = match object(&self.samples[*i]) {
                        Some(item) if jsonl => dataset::as_trained(cx.spec, &item, &effort)
                            .unwrap_or_else(|| dataset::texts(&self.samples[*i], true).join(" ")),
                        _ => self.samples[*i].clone(),
                    };
                    text.len() > context
                })
                .collect();
            if !long.is_empty() {
                finding(
                    plural(
                        long.len(),
                        "sample longer than the context window",
                        "samples longer than the context window",
                    ),
                    &format!(
                        "The model reads {context} tokens at once (a token is a letter or a tag here); training cuts \
                         what is longer. Shorten them, or give the model a longer context in Model Size."
                    ),
                    long,
                );
            }
        }

        // Reasoning the model will not learn: no Thinking block.
        if jsonl && cx.spec.thinking.is_none() {
            let reasoning: Vec<usize> = (0..self.samples.len())
                .filter(|i| {
                    object(&self.samples[*i]).is_some_and(|item| {
                        let has = |m: &Map<String, Value>| {
                            m.get("thinking").is_some_and(|t| t.as_str().is_some_and(|t| !t.trim().is_empty()))
                        };
                        has(&item)
                            || item
                                .get("messages")
                                .and_then(Value::as_array)
                                .is_some_and(|list| list.iter().filter_map(Value::as_object).any(has))
                    })
                })
                .collect();
            if !reasoning.is_empty() {
                finding(
                    plural(
                        reasoning.len(),
                        "sample with reasoning that is not learned",
                        "samples with reasoning that is not learned",
                    ),
                    "The Thinking block is not chained in, or is off, so training leaves their reasoning out and the \
                     model answers without thinking. Chain in the Thinking block to teach it.",
                    reasoning,
                );
            }
        }

        // Reasoning efforts no sample of the training set is written at.
        // Each file once, though several stages may read it.
        let files: BTreeSet<&String> = cx.stages.iter().flat_map(|(_, files)| files).collect();
        let mine = self.content();
        let contents: Vec<(&str, String)> = files
            .iter()
            .map(|f| {
                let content = if **f == self.file {
                    mine.clone()
                } else {
                    cx.data.get(*f).map(|b| String::from_utf8_lossy(b).into_owned()).unwrap_or_default()
                };
                (f.as_str(), content)
            })
            .collect();
        if let Some(effort) = &cx.spec.effort {
            let counts = dataset::effort_counts(effort, contents.iter().map(|(f, c)| (*f, c.as_str())));
            for (level, _) in &counts {
                if let Some(donor) = dataset::donor(&counts, level) {
                    finding(
                        format!("No sample at “{level}”"),
                        &format!(
                            "No dataset the training reads has a sample at reasoning effort “{level}”, so it learns \
                             from the “{donor}” samples and reasons like them. Tick samples and give them \
                             “{level}” (with reasoning written for it)."
                        ),
                        Vec::new(),
                    );
                }
            }
        }

        // Words fine-tuning and reinforcement learning use that pretraining never does.
        let pretraining: Vec<&String> =
            cx.stages.iter().filter(|(kind, _)| kind == studio::kinds::PRETRAIN).flat_map(|(_, files)| files).collect();
        let later = cx.stages.iter().any(|(kind, files)| kind != studio::kinds::PRETRAIN && files.contains(&self.file));
        if pretraining.is_empty() {
            notes.push("No pretraining stage reads a dataset, so there is no text to check the words against.".into());
        } else if pretraining.contains(&&self.file) && !later {
            notes.push(
                "This is pretraining text: the word check looks at the files fine-tuning and reinforcement learning \
                 read."
                    .into(),
            );
        } else {
            let known: Vec<String> = contents
                .iter()
                .filter(|(f, _)| pretraining.iter().any(|p| p.as_str() == *f))
                .flat_map(|(f, content)| {
                    let jsonl = studio::is_jsonl(f);
                    studio::split_samples(f, content).into_iter().flat_map(move |s| dataset::texts(&s, jsonl))
                })
                .collect();
            let texts: Vec<Vec<String>> = self.samples.iter().map(|s| dataset::texts(s, jsonl)).collect();
            let unknown =
                dataset::unknown_words(known.iter().map(String::as_str), texts.iter().flatten().map(String::as_str));
            if !unknown.is_empty() {
                let words: BTreeSet<&str> = unknown.iter().map(|(w, _)| w.as_str()).collect();
                let samples: Vec<usize> = (0..texts.len())
                    .filter(|i| texts[*i].iter().any(|t| dataset::words(t).any(|w| words.contains(w.as_str()))))
                    .collect();
                finding(
                    format!("Words pretraining never uses ({})", unknown.len()),
                    "These words are in this file but in none of the pretraining text, so the model meets them only \
                     here and learns them poorly. Add text that uses them to a pretraining dataset.",
                    samples,
                );
            }
            return Scan { revision: self.revision, findings, unknown, notes };
        }
        Scan { revision: self.revision, findings, unknown: Vec::new(), notes }
    }

    /// One sample: its header (selection, kind, efforts, actions), then the sample as the view shows it.
    fn card(
        &mut self,
        ui: &mut Ui,
        cx: &DatasetContext<'_>,
        index: usize,
        check: &Result<&'static str, String>,
        actions: &mut Vec<Action>,
    ) {
        let jsonl = self.jsonl();
        let selected = self.selected.contains(&index);
        let raw =
            self.view == View::Raw || self.raw.contains(&index) || (jsonl && object(&self.samples[index]).is_none());
        egui::Frame::new()
            .fill(if selected { theme::bg_widget() } else { theme::bg_raised() })
            .stroke(Stroke::new(1.0, if selected { theme::stroke_strong() } else { theme::stroke() }))
            .corner_radius(theme::RADIUS)
            .inner_margin(8)
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    let mut on = selected;
                    if ui.checkbox(&mut on, "").on_hover_text("Select (Shift: a range)").changed() {
                        actions.push(Action::Select(index, on));
                    }
                    ui.label(RichText::new(format!("#{}", index + 1)).size(11.0).color(theme::text_faint()));
                    let (badge, color) = match check {
                        Ok(kind) => (format!("{}  {kind}", kind_icon(kind)), theme::text_dim()),
                        Err(_) => (format!("{}  Problem", icon::WARNING), theme::warning()),
                    };
                    ui.label(RichText::new(badge).size(11.0).color(color));
                    if jsonl {
                        self.effort_chips(ui, cx, index);
                    }
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if widgets::icon_button(ui, icon::TRASH, "Remove this sample").clicked() {
                            actions.push(Action::Remove(index));
                        }
                        if widgets::icon_button(ui, icon::COPY, "Duplicate it").clicked() {
                            actions.push(Action::Duplicate(index));
                        }
                        if widgets::icon_button(ui, icon::COPY_SIMPLE, "Copy it (as its line)").clicked() {
                            actions.push(Action::Copy(index));
                        }
                        if jsonl && self.view != View::Raw {
                            let hint = if raw { "Back to the chat" } else { "Edit it as JSON" };
                            if widgets::icon_toggle(ui, icon::BRACKETS_CURLY, raw, hint).clicked() {
                                actions.push(Action::ToggleRaw(index));
                            }
                        }
                    });
                });
                if let Some(tools) = object(&self.samples[index]).as_ref().and_then(|item| item.get("tools").cloned()) {
                    tools_line(ui, &tools);
                }
                let changed = if raw {
                    raw_edit(ui, &mut self.samples[index])
                } else if !jsonl {
                    ui.add(
                        egui::TextEdit::multiline(&mut self.samples[index])
                            .desired_rows(2)
                            .desired_width(f32::INFINITY)
                            .hint_text("A paragraph of plain text"),
                    )
                    .changed()
                } else if self.view == View::Model {
                    model_view(ui, cx.spec, &self.samples[index]);
                    false
                } else {
                    chat_edit(ui, cx.spec, &mut self.samples[index])
                };
                if changed {
                    self.changed();
                }
                if let Err(problem) = check {
                    ui.label(RichText::new(format!("{}  {problem}", icon::WARNING)).size(11.0).color(theme::warning()));
                }
            });
    }

    /// The reasoning efforts of one sample, each a chip to turn on or off.
    fn effort_chips(&mut self, ui: &mut Ui, cx: &DatasetContext<'_>, index: usize) {
        let Some(mut item) = object(&self.samples[index]).filter(dataset::is_request) else { return };
        let levels = dataset::efforts(&item);
        ui.add_space(6.0);
        let Some(effort) = &cx.spec.effort else {
            // A model without levels learns no effort; the sample may still name one.
            if !levels.is_empty() {
                ui.label(RichText::new(format!("effort {}", levels.join(", "))).size(11.0).color(theme::text_faint()))
                    .on_hover_text("This model has no Reasoning Effort block, so it learns no effort");
            }
            return;
        };
        let mut next = None;
        ui.spacing_mut().item_spacing.x = 3.0;
        if chip(ui, "Default", levels.is_empty(), false)
            .on_hover_text(format!("At the model's default level ({})", effort.default))
            .clicked()
            && !levels.is_empty()
        {
            next = Some(Vec::new());
        }
        for level in &effort.levels {
            let on = levels.contains(level);
            if chip(ui, level, on, false)
                .on_hover_text(if on {
                    "Learned at this level: click to take it off"
                } else {
                    "Learn it at this level too"
                })
                .clicked()
            {
                let mut changed: Vec<String> = effort
                    .levels
                    .iter()
                    .filter(|l| levels.contains(*l) != (l.as_str() == level.as_str()))
                    .cloned()
                    .collect();
                // Levels the model does not have stay as they are.
                changed.extend(levels.iter().filter(|l| !effort.levels.contains(l)).cloned());
                next = Some(changed);
            }
        }
        if let Some(levels) = next {
            dataset::set_efforts(&mut item, &levels);
            self.samples[index] = dataset::to_line(&item);
            self.changed();
        }
    }

    fn apply(&mut self, ui: &Ui, actions: Vec<Action>, shown: &[usize]) {
        for action in actions {
            match action {
                Action::Remove(index) => {
                    self.samples.remove(index);
                    self.selected = self
                        .selected
                        .iter()
                        .filter(|i| **i != index)
                        .map(|i| if *i > index { i - 1 } else { *i })
                        .collect();
                    self.raw.clear();
                    self.changed();
                }
                Action::Duplicate(index) => {
                    self.samples.insert(index + 1, self.samples[index].clone());
                    self.selected = self.selected.iter().map(|i| if *i > index { i + 1 } else { *i }).collect();
                    self.raw.clear();
                    self.reveal = Some(index + 1);
                    self.changed();
                }
                Action::Copy(index) => ui.ctx().copy_text(self.samples[index].clone()),
                Action::ToggleRaw(index) => {
                    if !self.raw.remove(&index) {
                        self.raw.insert(index);
                    }
                }
                Action::Select(index, on) => {
                    let shift = ui.input(|i| i.modifiers.shift);
                    match (shift, self.anchor) {
                        (true, Some(anchor)) => {
                            let (a, b) = (
                                shown.iter().position(|i| *i == anchor).unwrap_or(0),
                                shown.iter().position(|i| *i == index).unwrap_or(0),
                            );
                            for i in &shown[a.min(b)..=a.max(b)] {
                                if on {
                                    self.selected.insert(*i);
                                } else {
                                    self.selected.remove(i);
                                }
                            }
                        }
                        _ if on => {
                            self.selected.insert(index);
                        }
                        _ => {
                            self.selected.remove(&index);
                        }
                    }
                    self.anchor = Some(index);
                }
            }
        }
    }
}

/// A sample's JSON object, when it is one.
fn object(sample: &str) -> Option<Map<String, Value>> {
    match serde_json::from_str::<Value>(sample) {
        Ok(Value::Object(item)) => Some(item),
        _ => None,
    }
}

fn plural(n: usize, one: &str, many: &str) -> String {
    if n == 1 { format!("1 {one}") } else { format!("{n} {many}") }
}

fn kind_icon(kind: &str) -> &'static str {
    match kind {
        "Tool use" => icon::WRENCH,
        "Conversation" => icon::CHATS,
        "Reasoning" => icon::BRAIN,
        "Not to answer" => icon::PROHIBIT,
        "Compaction" => icon::ARROWS_IN,
        "Text" => icon::TEXT_ALIGN_LEFT,
        _ => icon::CHAT_CIRCLE,
    }
}

/// A small rounded toggle: filled when on, outlined when some of a selection has it.
fn chip(ui: &mut Ui, label: &str, on: bool, some: bool) -> egui::Response {
    let (fill, text, stroke) = if on {
        (theme::accent(), theme::on_accent(), Stroke::NONE)
    } else if some {
        (Color32::TRANSPARENT, theme::text(), Stroke::new(1.0, theme::accent()))
    } else {
        (Color32::TRANSPARENT, theme::text_dim(), Stroke::new(1.0, theme::stroke()))
    };
    ui.add(
        egui::Button::new(RichText::new(label).size(11.0).color(text))
            .fill(fill)
            .stroke(stroke)
            .corner_radius(9)
            .min_size(vec2(0.0, 18.0)),
    )
}

fn word_chip(ui: &mut Ui, word: &str, count: usize) {
    egui::Frame::new().fill(theme::bg_input()).corner_radius(9).inner_margin(egui::Margin::symmetric(6, 1)).show(
        ui,
        |ui| {
            ui.label(RichText::new(format!("{word} ×{count}")).size(11.0).color(theme::text_dim()));
        },
    );
}

/// The tools a sample offers, by name.
fn tools_line(ui: &mut Ui, tools: &Value) {
    let Value::Array(list) = tools else { return };
    let names: Vec<String> = list
        .iter()
        .filter_map(|t| t.get("function").unwrap_or(t).get("name").and_then(Value::as_str).map(str::to_owned))
        .collect();
    let about: Vec<String> = list
        .iter()
        .map(|t| {
            let t = t.get("function").unwrap_or(t);
            format!(
                "{}: {}",
                t.get("name").and_then(Value::as_str).unwrap_or("?"),
                t.get("description").and_then(Value::as_str).unwrap_or("")
            )
        })
        .collect();
    ui.label(
        RichText::new(format!(
            "{}  Offers {}",
            icon::WRENCH,
            if names.is_empty() { "no tools".into() } else { names.join(" · ") }
        ))
        .size(11.0)
        .color(theme::text_faint()),
    )
    .on_hover_text(if about.is_empty() {
        "This sample offers no tools: the model learns to answer without them".to_owned()
    } else {
        format!("The tools this sample offers (edit them as JSON):\n{}", about.join("\n"))
    });
}

fn raw_edit(ui: &mut Ui, sample: &mut String) -> bool {
    ui.add(egui::TextEdit::multiline(sample).code_editor().desired_rows(3).desired_width(f32::INFINITY)).changed()
}

/// The sample as training writes it out for the model: tags faint, its text plain.
fn model_view(ui: &mut Ui, spec: &TextSpec, sample: &str) {
    let Some(item) = object(sample) else { return };
    let effort = spec.effort.as_ref().map(|e| dataset::learned_at(&item, e)).unwrap_or_default();
    let Some(text) = dataset::as_trained(spec, &item, effort.first().map_or("", String::as_str)) else {
        let texts = dataset::texts(sample, true).join("\n\n");
        ui.label(RichText::new(texts).monospace().size(11.5));
        return;
    };
    let mut tags: Vec<&str> = spec.specials.iter().map(String::as_str).filter(|t| !t.is_empty()).collect();
    tags.sort_by_key(|t| std::cmp::Reverse(t.len()));
    let mut job = egui::text::LayoutJob::default();
    let plain = egui::TextFormat { font_id: egui::FontId::monospace(11.5), color: theme::text(), ..Default::default() };
    let tag = egui::TextFormat { color: theme::text_faint(), background: theme::bg_input(), ..plain.clone() };
    let mut rest = text.as_str();
    while !rest.is_empty() {
        let next = tags
            .iter()
            .filter_map(|t| rest.find(t).map(|at| (at, *t)))
            .min_by_key(|(at, t)| (*at, std::cmp::Reverse(t.len())));
        match next {
            Some((at, found)) => {
                job.append(&rest[..at], 0.0, plain.clone());
                job.append(found, 0.0, tag.clone());
                // A new line after each turn, as it reads easier; the model reads none.
                if found == spec.end_tag.as_str() {
                    job.append("\n", 0.0, plain.clone());
                }
                rest = &rest[at + found.len()..];
            }
            None => {
                job.append(rest, 0.0, plain.clone());
                rest = "";
            }
        }
    }
    ui.add(egui::Label::new(job).wrap());
    if effort.len() > 1 {
        widgets::note(ui, format!("Shown at “{}”; it is learned at {} too.", effort[0], effort[1..].join(", ")));
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Who {
    System,
    User,
    Model,
    Avoid,
    Tool,
}

/// A chat bubble, the user's on the right. Returns whether its ✕ was clicked.
fn bubble(ui: &mut Ui, who: Who, removable: bool, body: impl FnOnce(&mut Ui)) -> bool {
    let full = ui.available_width();
    let width = if who == Who::System { full } else { (full * 0.86).max(240.0).min(full) };
    let (fill, stroke, label) = match who {
        Who::System => (theme::bg_input(), Stroke::NONE, "System"),
        Who::User => (theme::bg_widget_active(), Stroke::NONE, "Request"),
        Who::Model => (theme::bg_panel(), Stroke::new(1.0, theme::stroke()), "Model"),
        Who::Avoid => (theme::bg_panel(), Stroke::new(1.0, theme::danger()), "Not to give"),
        Who::Tool => (theme::bg_input(), Stroke::new(1.0, theme::stroke()), "Tool result"),
    };
    let mut remove = false;
    let layout = if who == Who::User { Layout::right_to_left(Align::Min) } else { Layout::left_to_right(Align::Min) };
    ui.allocate_ui_with_layout(vec2(full, 0.0), layout, |ui| {
        egui::Frame::new().fill(fill).stroke(stroke).corner_radius(theme::RADIUS_LARGE).inner_margin(8).show(
            ui,
            |ui| {
                ui.set_width(width - 18.0);
                ui.with_layout(Layout::top_down(Align::Min), |ui| {
                    ui.horizontal(|ui| {
                        let color = if who == Who::Avoid { theme::danger() } else { theme::text_faint() };
                        ui.label(RichText::new(label).size(10.5).color(color));
                        if removable {
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                if ui
                                    .add(egui::Button::new(RichText::new(icon::X).size(11.0)).frame(false))
                                    .on_hover_text("Remove this turn")
                                    .clicked()
                                {
                                    remove = true;
                                }
                            });
                        }
                    });
                    body(ui);
                });
            },
        );
    });
    ui.add_space(4.0);
    remove
}

/// Text inside a bubble, without a frame of its own.
fn bubble_text(ui: &mut Ui, text: &mut String, hint: &str, color: Option<Color32>, monospace: bool) -> bool {
    let mut edit = egui::TextEdit::multiline(text)
        .frame(egui::Frame::NONE)
        .desired_rows(1)
        .desired_width(f32::INFINITY)
        .hint_text(hint)
        .text_color_opt(color);
    if monospace {
        edit = edit.code_editor();
    }
    ui.add(edit).changed()
}

/// A string field of a JSON object, edited as text.
fn field_text(
    ui: &mut Ui,
    item: &mut Map<String, Value>,
    key: &str,
    hint: &str,
    color: Option<Color32>,
    monospace: bool,
) -> bool {
    let mut text = match item.get(key) {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    };
    if bubble_text(ui, &mut text, hint, color, monospace) {
        item.insert(key.into(), Value::String(text));
        return true;
    }
    false
}

/// A request's reasoning: faint, with a way to add it when there is none. It is
/// learned only with the Thinking block chained in and on.
fn thinking_field(ui: &mut Ui, item: &mut Map<String, Value>, spec: &TextSpec) -> bool {
    let on = spec.thinking.is_some();
    const OFF: &str = "Reasoning is learned only with the Thinking block chained into the start's Setup and turned on";
    if item.contains_key("thinking") {
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("{}  Reasoning", icon::BRAIN)).size(10.5).color(theme::text_faint()));
            if !on {
                ui.label(RichText::new("· not learned: the Thinking block is off").size(10.5).color(theme::warning()))
                    .on_hover_text(OFF);
            }
        });
        let color = if on { theme::text_dim() } else { theme::text_faint() };
        let changed = field_text(ui, item, "thinking", "How it reasons, before answering", Some(color), false);
        ui.separator();
        return changed;
    }
    let add = ui
        .add_enabled(on, egui::Button::new(format!("{}  Reasoning", icon::PLUS)).small())
        .on_hover_text("Add how it reasons before answering")
        .on_disabled_hover_text(OFF);
    if add.clicked() {
        item.insert("thinking".into(), Value::String(String::new()));
        return true;
    }
    false
}

/// A JSON value edited as one line; kept while it does not parse yet.
fn json_line(ui: &mut Ui, id: egui::Id, value: &mut Value) -> bool {
    let current = spaced_json(value);
    let mut buffer: String = ui.data(|d| d.get_temp::<String>(id)).unwrap_or_else(|| current.clone());
    let valid = serde_json::from_str::<Value>(&buffer).is_ok();
    let response = ui.add(
        egui::TextEdit::singleline(&mut buffer)
            .id(id)
            .code_editor()
            .desired_width(f32::INFINITY)
            .text_color_opt((!valid).then(theme::danger)),
    );
    let mut changed = false;
    if response.changed()
        && let Ok(parsed) = serde_json::from_str::<Value>(&buffer)
    {
        *value = parsed;
        changed = true;
    }
    if response.has_focus() {
        ui.data_mut(|d| d.insert_temp(id, buffer));
    } else {
        ui.data_mut(|d| d.remove::<String>(id));
    }
    changed
}

/// An assistant message's tool calls: a tool's name and its arguments, each.
fn tool_calls(ui: &mut Ui, message: &mut Map<String, Value>, removable: bool) -> bool {
    let mut changed = false;
    let Some(Value::Array(calls)) = message.get_mut("tool_calls") else { return false };
    let mut remove = None;
    for (n, call) in calls.iter_mut().enumerate() {
        let Some(call) = call.as_object_mut() else { continue };
        // OpenAI's {"function": {...}} form is edited where its fields are.
        let call = if call.get("function").is_some_and(Value::is_object) {
            call.get_mut("function").and_then(Value::as_object_mut).expect("checked")
        } else {
            call
        };
        egui::Frame::new().fill(theme::bg_input()).corner_radius(theme::RADIUS).inner_margin(6).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(RichText::new(icon::WRENCH).color(theme::accent_bright()))
                    .on_hover_text("A tool call: the model asks for a tool to be run");
                let mut name = call.get("name").and_then(Value::as_str).unwrap_or_default().to_owned();
                if ui
                    .add(egui::TextEdit::singleline(&mut name).code_editor().desired_width(180.0).hint_text("tool"))
                    .changed()
                {
                    call.insert("name".into(), Value::String(name));
                    changed = true;
                }
                let key =
                    if call.contains_key("arguments") || !call.contains_key("args") { "arguments" } else { "args" };
                let mut arguments = call.get(key).cloned().unwrap_or_else(|| json!({}));
                let id = ui.id().with(("call", n));
                if json_line(ui, id, &mut arguments) {
                    call.insert(key.into(), arguments);
                    changed = true;
                }
                if removable
                    && ui
                        .add(egui::Button::new(RichText::new(icon::X).size(11.0)).frame(false))
                        .on_hover_text("Remove this call")
                        .clicked()
                {
                    remove = Some(n);
                }
            });
        });
    }
    if let Some(n) = remove {
        calls.remove(n);
        changed = true;
    }
    changed
}

/// The sample as the chat it teaches, every part editable where it shows.
fn chat_edit(ui: &mut Ui, spec: &TextSpec, sample: &mut String) -> bool {
    let Some(mut item) = object(sample) else { return raw_edit(ui, sample) };
    let mut changed = false;
    if item.contains_key("system") {
        bubble(ui, Who::System, false, |ui| {
            changed |= field_text(
                ui,
                &mut item,
                "system",
                "The system prompt instead of the model's own",
                Some(theme::text_dim()),
                false,
            );
        });
    }
    if item.contains_key("messages") {
        changed |= conversation_edit(ui, spec, &mut item);
    } else if item.contains_key("context") && item.contains_key("compact") {
        bubble(ui, Who::Tool, false, |ui| {
            ui.label(RichText::new("Long context").size(10.5).color(theme::text_faint()));
            changed |= field_text(ui, &mut item, "context", "The long text", None, false);
        });
        bubble(ui, Who::Model, false, |ui| {
            ui.label(RichText::new("Its compact form").size(10.5).color(theme::text_faint()));
            changed |= field_text(ui, &mut item, "compact", "What to keep of it", None, false);
        });
    } else if item.contains_key("prompt") {
        bubble(ui, Who::User, false, |ui| {
            changed |= field_text(ui, &mut item, "prompt", "The request", None, false);
        });
        let avoid = item.contains_key("rejected") || item.contains_key("bad");
        bubble(ui, if avoid { Who::Avoid } else { Who::Model }, false, |ui| {
            if avoid {
                let key = if item.contains_key("rejected") { "rejected" } else { "bad" };
                changed |= field_text(ui, &mut item, key, "An answer the model should not give", None, false);
            } else {
                changed |= thinking_field(ui, &mut item, spec);
                changed |= field_text(ui, &mut item, "response", "The answer", None, false);
            }
        });
        if !avoid
            && ui
                .small_button(format!("{}  Continue as a conversation", icon::CHATS))
                .on_hover_text("Turns it into a conversation, to add more turns or tool calls")
                .clicked()
        {
            let mut answer = json!({ "role": "assistant", "content": item.remove("response").unwrap_or_default() });
            if let Some(thinking) = item.remove("thinking") {
                answer["thinking"] = thinking;
            }
            let user = json!({ "role": "user", "content": item.remove("prompt").unwrap_or_default() });
            item.insert(
                "messages".into(),
                json!([user, answer, { "role": "user", "content": "" }, { "role": "assistant", "content": "" }]),
            );
            changed = true;
        }
    } else if item.contains_key("text") {
        changed |= field_text(ui, &mut item, "text", "Plain text to learn the language from", None, false);
    } else {
        return raw_edit(ui, sample);
    }
    if changed {
        *sample = dataset::to_line(&item);
    }
    changed
}

fn conversation_edit(ui: &mut Ui, spec: &TextSpec, item: &mut Map<String, Value>) -> bool {
    let mut changed = false;
    let Some(Value::Array(messages)) = item.get_mut("messages") else { return false };
    let mut remove = None;
    for (n, message) in messages.iter_mut().enumerate() {
        let Some(message) = message.as_object_mut() else { continue };
        let role = message.get("role").and_then(Value::as_str).unwrap_or("user").to_owned();
        ui.push_id(n, |ui| {
            let who = match role.as_str() {
                "assistant" => Who::Model,
                "tool" => Who::Tool,
                "system" => Who::System,
                _ => Who::User,
            };
            if bubble(ui, who, true, |ui| {
                if who == Who::Model {
                    changed |= thinking_field(ui, message, spec);
                    let calls = message.get("tool_calls").is_some_and(|c| c.as_array().is_some_and(|c| !c.is_empty()));
                    let hint = if calls { "Text before its tool calls (often none)" } else { "The answer" };
                    changed |= field_text(ui, message, "content", hint, None, false);
                    changed |= tool_calls(ui, message, true);
                    if ui
                        .small_button(format!("{}  Tool call", icon::PLUS))
                        .on_hover_text("The model asks for a tool to be run")
                        .clicked()
                    {
                        let tool = spec.tools.first().map_or("tool", |t| t.name.as_str());
                        let calls = message.entry("tool_calls").or_insert_with(|| json!([]));
                        if let Some(calls) = calls.as_array_mut() {
                            calls.push(json!({ "name": tool, "arguments": {} }));
                        }
                        changed = true;
                    }
                } else {
                    let hint = match who {
                        Who::Tool => "What the tool gave back",
                        Who::System => "A system message",
                        _ => "The request",
                    };
                    changed |= field_text(ui, message, "content", hint, None, who == Who::Tool);
                }
            }) {
                remove = Some(n);
            }
        });
    }
    if let Some(n) = remove {
        messages.remove(n);
        changed = true;
    }
    widgets::button_row(ui, |ui| {
        ui.label(RichText::new("Add").size(11.0).color(theme::text_faint()));
        for (label, role) in [("Request", "user"), ("Answer", "assistant"), ("Tool result", "tool")] {
            if ui.small_button(label).clicked() {
                messages.push(json!({ "role": role, "content": "" }));
                changed = true;
            }
        }
    });
    changed
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

    #[test]
    fn the_model_view_writes_the_training_text() {
        let mut spec = TextSpec { effort: Some(Default::default()), ..TextSpec::default() };
        spec.ensure_specials();
        let item = object(EXAMPLES[2].1).unwrap();
        let text = dataset::as_trained(&spec, &item, "none").unwrap();
        assert!(text.starts_with("<|system|>"), "{text}");
        assert!(text.contains("Reasoning effort: none"));
        assert!(
            text.contains(
                "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Madrid\"}}</tool_call>"
            )
        );
        assert!(text.contains("<|tool|>{\"temperature_c\": 27"));
        assert!(text.ends_with("Yes – sunny and 27 °C.<|end|>"));
    }
}
