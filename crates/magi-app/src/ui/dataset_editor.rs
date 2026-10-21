//! The Text Dataset editor: the samples of one dataset file of a `.magic`
//! model, shown as the chat they teach (and edited there), as the model reads
//! them in training, or as their JSON lines – searched, filtered, selected
//! together, given reasoning efforts and scanned for issues. It can pop out
//! into a window of its own.

use std::collections::{BTreeMap, BTreeSet};

use egui::{Align, Color32, Layout, RichText, Stroke, Ui, vec2};
use egui_phosphor::regular as icon;
use magi_core::magic::{TextSpec, ToolSpec, spaced_json};
use magi_core::nodes::studio;
use magi_core::{bulk, dataset};

use crate::magic::generate::{self, GeneratorSettings};
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
    /// The JSON lines themselves.
    Raw,
}

/// A change to take back: one sample's text before it was edited, or every sample.
enum Undo {
    One(usize, String),
    All(Vec<String>),
}

/// How many changes can be taken back.
const UNDO_DEPTH: usize = 40;

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

/// What a bulk edit does.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BulkOp {
    Replace,
    Delete,
    Keep,
    DeleteLonger,
    Effort,
    Tools,
    StripThinking,
    Trim,
    Dedupe,
}

/// Which samples a bulk edit reaches.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BulkScope {
    Selected,
    File,
    AllFiles,
}

/// The Edit in bulk panel.
struct Bulk {
    op: BulkOp,
    find: String,
    with: String,
    regex: bool,
    case_sensitive: bool,
    whole_words: bool,
    part: bulk::Part,
    characters: usize,
    /// Empty: the model's default level.
    effort: String,
    no_tools: bool,
    scope: BulkScope,
    /// What it would do, as last previewed (cleared when anything changes).
    preview: Option<Result<String, String>>,
}

impl Default for Bulk {
    fn default() -> Self {
        Self {
            op: BulkOp::Replace,
            find: String::new(),
            with: String::new(),
            regex: false,
            case_sensitive: false,
            whole_words: false,
            part: bulk::Part::All,
            characters: 4000,
            effort: String::new(),
            no_tools: false,
            scope: BulkScope::File,
            preview: None,
        }
    }
}

impl Bulk {
    fn edit(&self) -> bulk::Edit {
        let pattern = bulk::Pattern {
            text: self.find.clone(),
            regex: self.regex,
            case_sensitive: self.case_sensitive,
            whole_words: self.whole_words,
        };
        match self.op {
            BulkOp::Replace => bulk::Edit::Replace { pattern, with: self.with.clone(), part: self.part },
            BulkOp::Delete => bulk::Edit::Delete { pattern, part: self.part, keep: false },
            BulkOp::Keep => bulk::Edit::Delete { pattern, part: self.part, keep: true },
            BulkOp::DeleteLonger => bulk::Edit::DeleteLonger { characters: self.characters },
            BulkOp::Effort => bulk::Edit::SetEffort((!self.effort.is_empty()).then(|| self.effort.clone())),
            BulkOp::Tools => bulk::Edit::SetTools { none: self.no_tools },
            BulkOp::StripThinking => bulk::Edit::StripThinking,
            BulkOp::Trim => bulk::Edit::Trim,
            BulkOp::Dedupe => bulk::Edit::Dedupe,
        }
    }
}

fn report_text(report: bulk::Report, files: usize) -> String {
    if report.is_empty() && report.skipped == 0 {
        return "Nothing would change.".into();
    }
    let mut parts = Vec::new();
    if report.changed > 0 {
        parts.push(format!("{} changed", plural(report.changed, "sample", "samples")));
    }
    if report.deleted > 0 {
        parts.push(format!("{} deleted", plural(report.deleted, "sample", "samples")));
    }
    if report.skipped > 0 {
        parts.push(format!(
            "{} left as they were (the edit would break them)",
            plural(report.skipped, "sample", "samples")
        ));
    }
    let mut text = parts.join(", ");
    if files > 1 {
        text.push_str(&format!(" in {files} files"));
    }
    text.push('.');
    text
}

/// The Generate with AI panel.
struct Generate {
    settings: GeneratorSettings,
    prompt: String,
    count: usize,
    kind: generate::Kind,
    /// The endpoint's models, once listed (or why they could not be).
    models: Option<Result<Vec<String>, String>>,
    job: Option<generate::Job>,
    /// How many came in, and why it stopped.
    added: usize,
    note: Option<Result<String, String>>,
}

pub struct DatasetEditor {
    /// The file's path inside the `.magic` file.
    pub file: String,
    /// The Generate with AI panel, when open.
    generate: Option<Generate>,
    /// The Edit in bulk panel, when open.
    bulk: Option<Bulk>,
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
    undo: Vec<Undo>,
    redo: Vec<Undo>,
    /// The sample typed in last, and when: typing in one sample is taken back at once.
    typing: Option<(usize, std::time::Instant)>,
    /// Changes not yet put back into the model.
    pub dirty: bool,
    pub open: bool,
}

pub enum DatasetEvent {
    /// Put the samples back into the model's file.
    Save(String, String),
    /// The generator's endpoint changed: keep it.
    Generator(GeneratorSettings),
}

/// What the editor knows of the rest of the model.
pub struct DatasetContext<'a> {
    pub spec: &'a TextSpec,
    /// Every dataset file of the model, by its path inside the `.magic` file.
    pub data: &'a BTreeMap<String, Vec<u8>>,
    /// The files each training stage learns from, with the stage's kind, in order.
    pub stages: Vec<(String, Vec<String>)>,
    /// Every tool MacroMagi knows – the model's, its native tools, the AI Agent's –
    /// and whether it is on for this model.
    pub tools: Vec<(ToolSpec, bool)>,
    /// Where samples are generated (None: generating is not offered here).
    pub generator: Option<&'a GeneratorSettings>,
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
            bulk: None,
            generate: None,
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
            undo: Vec::new(),
            redo: Vec::new(),
            typing: None,
            dirty: false,
            open: true,
        }
    }

    /// Shows the samples as `chat` or `raw`.
    pub fn set_view(&mut self, view: &str) {
        self.view = if view == "raw" { View::Raw } else { View::Chat };
        if view == "bulk" {
            self.bulk = Some(Bulk::default());
        }
    }

    /// Opens with a scan of the file already done, as Scan all files did it.
    pub fn with_scan(mut self, cx: &DatasetContext<'_>) -> Self {
        self.scan = Some(self.scan(cx));
        self
    }

    /// What a scan of a file finds, by title (Scan all files in the Text Dataset block).
    pub fn scan_titles(file: &str, bytes: &[u8], cx: &DatasetContext<'_>) -> Vec<String> {
        let mut editor = Self::new(file, bytes);
        editor.scan(cx).findings.into_iter().map(|f| f.title).collect()
    }

    /// Before a change to many samples: all of them, to take it back.
    fn remember_all(&mut self) {
        self.undo.push(Undo::All(self.samples.clone()));
        self.after_remember();
    }

    /// After one sample was edited: its text before, unless the same sample was typed in a moment ago.
    fn remember_one(&mut self, index: usize, before: String) {
        let now = std::time::Instant::now();
        let typing = self.typing.is_some_and(|(i, at)| i == index && now.duration_since(at).as_secs_f32() < 1.5);
        self.typing = Some((index, now));
        if typing && matches!(self.undo.last(), Some(Undo::One(i, _)) if *i == index) {
            self.redo.clear();
            return;
        }
        self.undo.push(Undo::One(index, before));
        self.after_remember();
        self.typing = Some((index, now));
    }

    fn after_remember(&mut self) {
        self.redo.clear();
        self.typing = None;
        if self.undo.len() > UNDO_DEPTH {
            self.undo.remove(0);
        }
    }

    /// Puts a remembered state back, and gives what it replaced (to redo or undo again).
    fn restore(&mut self, entry: Undo) -> Undo {
        self.raw.clear();
        self.selected.clear();
        self.typing = None;
        let back = match entry {
            Undo::One(index, text) if index < self.samples.len() => {
                self.reveal = Some(index);
                Undo::One(index, std::mem::replace(&mut self.samples[index], text))
            }
            Undo::One(..) => Undo::All(self.samples.clone()),
            Undo::All(samples) => Undo::All(std::mem::replace(&mut self.samples, samples)),
        };
        self.changed();
        back
    }

    fn undo(&mut self) {
        if let Some(entry) = self.undo.pop() {
            let back = self.restore(entry);
            self.redo.push(back);
        }
    }

    fn redo(&mut self) {
        if let Some(entry) = self.redo.pop() {
            let back = self.restore(entry);
            self.undo.push(back);
        }
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
        self.bulk_panel(ui, cx, events);
        self.generate_panel(ui, cx, events);
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
                widgets::segmented(ui, &mut self.view, &[(View::Chat, "Chat"), (View::Raw, "Raw")]);
            }
        });
        // Ctrl+Z and Ctrl+Y (or Ctrl+Shift+Z), when no text box is taking the keys.
        if !ui.ctx().egui_wants_keyboard_input() {
            let (undo, redo) = ui.input(|i| {
                let z = i.modifiers.command && i.key_pressed(egui::Key::Z);
                (
                    z && !i.modifiers.shift,
                    (z && i.modifiers.shift) || (i.modifiers.command && i.key_pressed(egui::Key::Y)),
                )
            });
            if undo {
                self.undo();
            } else if redo {
                self.redo();
            }
        }
        widgets::button_row(ui, |ui| {
            if ui
                .add_enabled(!self.undo.is_empty(), egui::Button::new(icon::ARROW_COUNTER_CLOCKWISE))
                .on_hover_text("Undo (Ctrl+Z)")
                .clicked()
            {
                self.undo();
            }
            if ui
                .add_enabled(!self.redo.is_empty(), egui::Button::new(icon::ARROW_CLOCKWISE))
                .on_hover_text("Redo (Ctrl+Y)")
                .clicked()
            {
                self.redo();
            }
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
            if ui
                .add(egui::Button::new(format!("{}  Edit in bulk", icon::PENCIL_LINE)).selected(self.bulk.is_some()))
                .on_hover_text(
                    "Find and replace (text or a regular expression) in chosen parts of the samples, delete the ones \
                     that match or are too long, set their effort or tools, take out their reasoning, trim, remove \
                     repeats – in this file or all of them",
                )
                .clicked()
            {
                self.bulk = if self.bulk.is_some() { None } else { Some(Bulk::default()) };
            }
            if jsonl
                && let Some(settings) = cx.generator
                && ui
                    .add(egui::Button::new(format!("{}  Generate", icon::SPARKLE)).selected(self.generate.is_some()))
                    .on_hover_text(
                        "Has another AI model write samples – one on this computer (LM Studio, Ollama…) or any \
                         OpenAI-compatible endpoint – each checked like one written by hand",
                    )
                    .clicked()
            {
                self.generate = match self.generate.take() {
                    Some(open) if open.job.is_none() => None,
                    Some(open) => Some(open),
                    None => Some(Generate {
                        settings: settings.clone(),
                        prompt: String::new(),
                        count: 50,
                        kind: generate::Kind::Answers,
                        models: None,
                        job: None,
                        added: 0,
                        note: None,
                    }),
                };
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
                    self.remember_all();
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

    fn generate_panel(&mut self, ui: &mut Ui, cx: &DatasetContext<'_>, events: &mut Vec<DatasetEvent>) {
        let Some(mut panel) = self.generate.take() else { return };
        // What the job made so far, into the file (one undo for the whole generation).
        if let Some(job) = &panel.job {
            let mut progress = job.progress.lock().unwrap_or_else(|p| p.into_inner());
            if !progress.made.is_empty() {
                let made = std::mem::take(&mut progress.made);
                drop(progress);
                if panel.added == 0 {
                    self.remember_all();
                }
                panel.added += made.len();
                self.samples.extend(made);
                self.changed();
                progress = job.progress.lock().unwrap_or_else(|p| p.into_inner());
            }
            if progress.done {
                panel.note = Some(match &progress.error {
                    Some(error) => Err(format!("{error} ({} samples added).", panel.added)),
                    None => Ok(format!(
                        "{} {} added{}.",
                        progress.status.trim_end_matches('.'),
                        plural(panel.added, "sample", "samples"),
                        if progress.rejected > 0 {
                            format!(", {} lines left out", progress.rejected)
                        } else {
                            String::new()
                        }
                    )),
                });
                drop(progress);
                panel.job = None;
            } else {
                let status = progress.status.clone();
                drop(progress);
                panel.note = Some(Ok(format!("{status} {} so far.", plural(panel.added, "sample", "samples"))));
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(300));
            }
        }
        let running = panel.job.is_some();
        let before = panel.settings.clone();
        let mut close = false;
        widgets::card(ui, "Generate with AI", |ui| {
            ui.add_enabled_ui(!running, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Endpoint").color(theme::text_dim()));
                    ui.add(egui::TextEdit::singleline(&mut panel.settings.url).desired_width(220.0));
                    ui.menu_button(icon::CARET_DOWN, |ui| {
                        for (name, url) in generate::ENDPOINTS {
                            if ui.button(format!("{name}  ({url})")).clicked() {
                                panel.settings.url = url.into();
                                panel.models = None;
                                ui.close();
                            }
                        }
                    });
                    ui.label(RichText::new("Model").color(theme::text_dim()));
                    let models = panel.models.as_ref().and_then(|m| m.as_ref().ok()).cloned().unwrap_or_default();
                    egui::ComboBox::from_id_salt("generate-model")
                        .width(200.0)
                        .truncate()
                        .selected_text(if panel.settings.model.is_empty() {
                            "the endpoint's own"
                        } else {
                            panel.settings.model.as_str()
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut panel.settings.model, String::new(), "the endpoint's own");
                            for model in models {
                                ui.selectable_value(&mut panel.settings.model, model.clone(), model);
                            }
                        });
                    if widgets::icon_button(ui, icon::ARROWS_CLOCKWISE, "List the endpoint's models").clicked() {
                        panel.models = Some(generate::models(&panel.settings));
                    }
                });
                if let Some(Err(error)) = &panel.models {
                    widgets::notice(ui, icon::WARNING_CIRCLE, error.clone(), theme::danger());
                }
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Thinking").color(theme::text_dim()));
                    for (value, label) in generate::THINKING {
                        if ui.selectable_label(panel.settings.thinking == value, label).clicked() {
                            panel.settings.thinking = value.into();
                        }
                    }
                });
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Key").color(theme::text_dim()));
                    ui.add(
                        egui::TextEdit::singleline(&mut panel.settings.key)
                            .password(true)
                            .hint_text("only if the endpoint needs one")
                            .desired_width(220.0),
                    );
                });
                ui.add(
                    egui::TextEdit::multiline(&mut panel.prompt)
                        .hint_text(
                            "What the samples are about – say, everyday cooking questions with short, exact answers",
                        )
                        .desired_rows(2)
                        .desired_width(f32::INFINITY),
                );
                ui.horizontal_wrapped(|ui| {
                    for (kind, label) in generate::Kind::ALL {
                        if ui.selectable_label(panel.kind == kind, label).clicked() {
                            panel.kind = kind;
                        }
                    }
                    ui.add_space(8.0);
                    ui.add(egui::DragValue::new(&mut panel.count).range(1..=5000).suffix(" samples"));
                });
            });
            ui.horizontal(|ui| {
                if running {
                    if ui.button(format!("{}  Stop", icon::STOP)).clicked()
                        && let Some(job) = &panel.job
                    {
                        job.stop();
                    }
                } else {
                    let ready = !panel.prompt.trim().is_empty() && !panel.settings.url.trim().is_empty();
                    if ui
                        .add_enabled(ready, egui::Button::new(format!("{}  Generate", icon::SPARKLE)))
                        .on_disabled_hover_text("Say what the samples are about")
                        .clicked()
                    {
                        let request = generate::Request {
                            settings: panel.settings.clone(),
                            prompt: panel.prompt.clone(),
                            count: panel.count,
                            kind: panel.kind,
                            spec: cx.spec.clone(),
                            existing: self.samples.clone(),
                        };
                        let ctx = ui.ctx().clone();
                        panel.added = 0;
                        panel.note = None;
                        panel.job = Some(generate::start(request, move || ctx.request_repaint()));
                    }
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                }
                if running {
                    ui.spinner();
                }
            });
            match &panel.note {
                Some(Ok(text)) => widgets::notice(ui, icon::INFO, text.clone(), theme::text_dim()),
                Some(Err(text)) => widgets::notice(ui, icon::WARNING_CIRCLE, text.clone(), theme::danger()),
                None => {}
            }
            widgets::note(
                ui,
                "Check what a model's terms allow: some services forbid training other models on what they write. \
                 Open models you run yourself usually do not.",
            );
        });
        if panel.settings != before {
            events.push(DatasetEvent::Generator(panel.settings.clone()));
        }
        if !close {
            self.generate = Some(panel);
        }
    }

    fn bulk_panel(&mut self, ui: &mut Ui, cx: &DatasetContext<'_>, events: &mut Vec<DatasetEvent>) {
        let Some(mut panel) = self.bulk.take() else { return };
        let jsonl = self.jsonl();
        let before = (
            panel.op,
            panel.find.clone(),
            panel.with.clone(),
            panel.regex,
            panel.case_sensitive,
            panel.whole_words,
            panel.part,
            panel.characters,
            panel.effort.clone(),
            panel.no_tools,
            panel.scope,
        );
        let mut apply = false;
        let mut close = false;
        widgets::card(ui, "Edit in bulk", |ui| {
            let mut ops = vec![
                (BulkOp::Replace, "Replace"),
                (BulkOp::Delete, "Delete matching"),
                (BulkOp::Keep, "Keep matching"),
                (BulkOp::DeleteLonger, "Delete long"),
                (BulkOp::Dedupe, "Remove repeats"),
                (BulkOp::Trim, "Trim"),
            ];
            if jsonl {
                ops.extend([
                    (BulkOp::Effort, "Effort"),
                    (BulkOp::Tools, "Tools"),
                    (BulkOp::StripThinking, "No thinking"),
                ]);
            }
            ui.horizontal_wrapped(|ui| {
                for (op, label) in &ops {
                    if ui.selectable_label(panel.op == *op, *label).clicked() {
                        panel.op = *op;
                    }
                }
            });
            ui.add_space(4.0);
            match panel.op {
                BulkOp::Replace | BulkOp::Delete | BulkOp::Keep => {
                    ui.horizontal(|ui| {
                        ui.add(egui::TextEdit::singleline(&mut panel.find).hint_text("Find").desired_width(220.0));
                        if panel.op == BulkOp::Replace {
                            ui.add(
                                egui::TextEdit::singleline(&mut panel.with)
                                    .hint_text(if panel.regex {
                                        "Replace with ($1 for a group)"
                                    } else {
                                        "Replace with"
                                    })
                                    .desired_width(220.0),
                            );
                        }
                    });
                    ui.horizontal_wrapped(|ui| {
                        ui.checkbox(&mut panel.regex, "Regular expression");
                        ui.checkbox(&mut panel.case_sensitive, "Match case");
                        ui.checkbox(&mut panel.whole_words, "Whole words");
                        if jsonl {
                            ui.label(RichText::new("in").color(theme::text_dim()));
                            egui::ComboBox::from_id_salt("bulk-part")
                                .selected_text(
                                    bulk::Part::ALL.iter().find(|(p, _)| *p == panel.part).map_or("", |(_, l)| *l),
                                )
                                .show_ui(ui, |ui| {
                                    for (part, label) in bulk::Part::ALL {
                                        ui.selectable_value(&mut panel.part, part, label);
                                    }
                                });
                        }
                    });
                }
                BulkOp::DeleteLonger => {
                    ui.horizontal(|ui| {
                        ui.label("Longer than");
                        ui.add(egui::DragValue::new(&mut panel.characters).range(1..=1_000_000).speed(50));
                        ui.label("characters (what the model reads and writes)");
                    });
                }
                BulkOp::Effort => {
                    ui.horizontal(|ui| {
                        ui.label("Set their reasoning effort to");
                        let levels = cx.spec.effort.as_ref().map(|e| e.levels.clone()).unwrap_or_default();
                        egui::ComboBox::from_id_salt("bulk-effort")
                            .selected_text(if panel.effort.is_empty() {
                                "the model's default".to_owned()
                            } else {
                                panel.effort.clone()
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut panel.effort, String::new(), "the model's default");
                                for level in levels {
                                    ui.selectable_value(&mut panel.effort, level.clone(), level);
                                }
                            });
                    });
                }
                BulkOp::Tools => {
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut panel.no_tools, false, "List the model's own tools");
                        ui.radio_value(&mut panel.no_tools, true, "List no tools");
                    });
                }
                BulkOp::StripThinking => widgets::note(ui, "Takes out their reasoning; the answers stay."),
                BulkOp::Trim => widgets::note(ui, "Takes spaces off the ends of every text."),
                BulkOp::Dedupe => {
                    widgets::note(ui, "Deletes samples that repeat an earlier one exactly (spacing aside).")
                }
            }
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Where").color(theme::text_dim()));
                if !self.selected.is_empty() {
                    ui.radio_value(
                        &mut panel.scope,
                        BulkScope::Selected,
                        format!("Selected ({})", self.selected.len()),
                    );
                }
                ui.radio_value(&mut panel.scope, BulkScope::File, "This file");
                ui.radio_value(&mut panel.scope, BulkScope::AllFiles, "Every dataset file");
            });
            if self.selected.is_empty() && panel.scope == BulkScope::Selected {
                panel.scope = BulkScope::File;
            }
            ui.horizontal(|ui| {
                if ui.button(format!("{}  Preview", icon::EYE)).clicked() {
                    panel.preview = Some(self.bulk_run(&panel, cx, false, events));
                }
                let ready = matches!(&panel.preview, Some(Ok(_)));
                if ui
                    .add_enabled(ready, egui::Button::new(format!("{}  Apply", icon::CHECK)))
                    .on_disabled_hover_text("Preview first")
                    .on_hover_text("This file's change can be taken back with Undo; other files' cannot")
                    .clicked()
                {
                    apply = true;
                }
                if ui.button("Close").clicked() {
                    close = true;
                }
            });
            match &panel.preview {
                Some(Ok(text)) => widgets::notice(ui, icon::INFO, text.clone(), theme::text_dim()),
                Some(Err(text)) => widgets::notice(ui, icon::WARNING_CIRCLE, text.clone(), theme::danger()),
                None => {}
            }
        });
        let after = (
            panel.op,
            panel.find.clone(),
            panel.with.clone(),
            panel.regex,
            panel.case_sensitive,
            panel.whole_words,
            panel.part,
            panel.characters,
            panel.effort.clone(),
            panel.no_tools,
            panel.scope,
        );
        if after != before {
            panel.preview = None;
        }
        if apply {
            let done = self.bulk_run(&panel, cx, true, events);
            panel.preview = Some(done.map(|t| format!("Done: {}", t.trim_end_matches('.').to_lowercase())));
        }
        if !close {
            self.bulk = Some(panel);
        }
    }

    /// Runs a bulk edit – for real, or only to say what it would do – on its samples.
    fn bulk_run(
        &mut self,
        panel: &Bulk,
        cx: &DatasetContext<'_>,
        real: bool,
        events: &mut Vec<DatasetEvent>,
    ) -> Result<String, String> {
        let edit = panel.edit();
        let mut total = bulk::Report::default();
        let mut files = 0;
        // This file: undoable.
        let only: Option<std::collections::HashSet<usize>> =
            (panel.scope == BulkScope::Selected).then(|| self.selected.iter().copied().collect());
        let mut mine = self.samples.clone();
        let report = bulk::apply(&mut mine, self.jsonl(), &edit, only.as_ref())?;
        if !report.is_empty() {
            files += 1;
            total.add(report);
            if real {
                self.remember_all();
                self.samples = mine;
                self.selected.clear();
                self.changed();
            }
        } else {
            total.add(report);
        }
        // The others, put back into the model as they are changed.
        if panel.scope == BulkScope::AllFiles {
            for (file, bytes) in cx.data.iter().filter(|(f, _)| f.starts_with("data/text/") && **f != self.file) {
                let mut samples = studio::split_samples(file, &String::from_utf8_lossy(bytes));
                let report = bulk::apply(&mut samples, studio::is_jsonl(file), &edit, None)?;
                total.add(report);
                if !report.is_empty() {
                    files += 1;
                    if real {
                        events.push(DatasetEvent::Save(file.clone(), studio::join_samples(file, &samples)));
                    }
                }
            }
        }
        Ok(report_text(total, files))
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
                        self.undo.push(Undo::All(self.samples.clone()));
                        self.after_remember();
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
                        self.remember_all();
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
                        self.remember_all();
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

        // Answers in a row: the model never writes a second one by itself – after an answer it
        // stops, and MacroMagi only lets it go on after a tool's result.
        if jsonl {
            let in_a_row: Vec<usize> = (0..self.samples.len())
                .filter(|i| {
                    object(&self.samples[*i])
                        .and_then(|item| item.get("messages").and_then(Value::as_array).cloned())
                        .is_some_and(|messages| {
                            messages.windows(2).any(|pair| {
                                let role =
                                    |m: &Value| m.get("role").and_then(Value::as_str).unwrap_or("user").to_owned();
                                role(&pair[0]) == "assistant" && role(&pair[1]) == "assistant"
                            })
                        })
                })
                .collect();
            if !in_a_row.is_empty() {
                finding(
                    plural(in_a_row.len(), "conversation with answers in a row", "conversations with answers in a row"),
                    "Two answers from the model with nothing between them. Each is learned, but the second never \
                     happens when the model runs: after an answer it stops, and it goes on only after a tool's \
                     result (a tool call, then a Tool result, then the next answer – that is the agent loop).",
                    in_a_row,
                );
            }
        }

        // Longer than the context window, in the model's own tokens: training cuts them.
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
                    // Bytes are at least as many as tokens: only long ones need counting.
                    text.len() > context && cx.spec.encode(&text).len() > context
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
                        "The model reads {context} tokens at once ({}); training cuts what is longer. Shorten \
                         them, or give the model a longer context in Model Size.",
                        if cx.spec.tokenizer.merges.is_empty() {
                            "a token is a letter or a tag"
                        } else {
                            "its learned tokens: a few letters each"
                        }
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
        let before = self.samples[index].clone();
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
                    let badge = ui.label(RichText::new(badge).size(11.0).color(color));
                    if jsonl {
                        // What training writes for the model, on hover.
                        let sample = self.samples[index].clone();
                        badge.on_hover_ui(|ui| {
                            ui.set_max_width(560.0);
                            ui.label(RichText::new("As the model reads it in training").size(11.0).strong());
                            model_view(ui, cx.spec, &sample);
                        });
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
                } else {
                    chat_edit(ui, cx, &mut self.samples[index])
                };
                if changed {
                    self.changed();
                }
                if let Err(problem) = check {
                    ui.label(RichText::new(format!("{}  {problem}", icon::WARNING)).size(11.0).color(theme::warning()));
                }
            });
        if index < self.samples.len() && self.samples[index] != before {
            self.remember_one(index, before);
        }
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
                    self.remember_all();
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
                    self.remember_all();
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
/// Whether a sample may call a tool: one it offers, or – offering none – one of the model's.
fn tool_on(cx: &DatasetContext<'_>, offered: Option<&[ToolSpec]>, name: &str) -> bool {
    match offered {
        Some(tools) => tools.iter().any(|t| t.name == name),
        None => cx.spec.tools.iter().any(|t| t.name == name),
    }
}

/// The schema of a tool by name: the sample's, the model's or any MacroMagi knows.
fn schema_of<'a>(cx: &'a DatasetContext<'_>, offered: Option<&'a [ToolSpec]>, name: &str) -> Option<&'a Value> {
    offered
        .into_iter()
        .flatten()
        .chain(&cx.spec.tools)
        .chain(cx.tools.iter().map(|(t, _)| t))
        .find(|t| t.name == name)
        .map(|t| &t.parameters)
}

/// Arguments to start a call with: each of the schema's properties, empty.
fn arguments_template(schema: &Value) -> Value {
    let mut out = Map::new();
    if let Some(Value::Object(properties)) = schema.get("properties") {
        for (key, property) in properties {
            let empty = match property.get("type").and_then(Value::as_str) {
                Some("number" | "integer") => json!(0),
                Some("boolean") => json!(false),
                Some("array") => json!([]),
                Some("object") => json!({}),
                _ => json!(""),
            };
            out.insert(key.clone(), empty);
        }
    }
    Value::Object(out)
}

/// "+ Tool call": a menu of the tools, with their arguments – the ones on for this
/// model (or this sample) first, then the rest MacroMagi has, marked.
fn tool_picker(ui: &mut Ui, cx: &DatasetContext<'_>, offered: Option<&[ToolSpec]>) -> Option<ToolSpec> {
    let mut chosen = None;
    ui.menu_button(format!("{}  Tool call", icon::PLUS), |ui| {
        ui.set_min_width(320.0);
        let mut seen = BTreeSet::new();
        let mut on: Vec<ToolSpec> = offered.map(<[ToolSpec]>::to_vec).unwrap_or_else(|| cx.spec.tools.clone());
        on.retain(|t| seen.insert(t.name.clone()));
        let off: Vec<ToolSpec> =
            cx.tools.iter().filter(|(t, _)| seen.insert(t.name.clone())).map(|(t, _)| t.clone()).collect();
        let label = if offered.is_some() { "Offered in this sample" } else { "On for this model" };
        ui.label(RichText::new(label).size(10.5).color(theme::text_faint()));
        if on.is_empty() {
            ui.weak("None");
        }
        for tool in &on {
            if tool_choice(ui, tool, true).clicked() {
                chosen = Some(tool.clone());
                ui.close();
            }
        }
        if !off.is_empty() {
            ui.separator();
            ui.label(RichText::new("Not on for this model").size(10.5).color(theme::warning()));
            for tool in &off {
                if tool_choice(ui, tool, false).clicked() {
                    chosen = Some(tool.clone());
                    ui.close();
                }
            }
        }
    })
    .response
    .on_hover_text("The model asks for a tool to be run: pick one");
    chosen
}

fn tool_choice(ui: &mut Ui, tool: &ToolSpec, on: bool) -> egui::Response {
    let arguments: Vec<String> = tool
        .parameters
        .get("properties")
        .and_then(Value::as_object)
        .map(|p| {
            p.iter().map(|(k, v)| format!("{k}: {}", v.get("type").and_then(Value::as_str).unwrap_or("any"))).collect()
        })
        .unwrap_or_default();
    let mut text = RichText::new(format!("{}({})", tool.name, arguments.join(", "))).monospace().size(12.0);
    if !on {
        text = text.color(theme::text_faint());
    }
    let response = ui.button(text);
    let about = if on {
        tool.description.clone()
    } else {
        format!(
            "{}\n\nNot on for this model: turn it on in the Tools or Native Tools block, or offer it in the sample's tools.",
            tool.description
        )
    };
    response.on_hover_text(about)
}

/// An argument's value, edited as its type is: text, a number, a switch, or JSON.
fn argument_edit(ui: &mut Ui, id: egui::Id, value: &mut Value, kind: Option<&str>) -> bool {
    match (kind, &mut *value) {
        (Some("boolean"), Value::Bool(on)) => ui.checkbox(on, "").changed(),
        (Some("number" | "integer"), Value::Number(_)) | (Some("number" | "integer"), Value::Null) => {
            let mut text = match value {
                Value::Number(n) => n.to_string(),
                _ => String::new(),
            };
            let response = ui.add(egui::TextEdit::singleline(&mut text).desired_width(120.0));
            if response.changed() {
                if let Ok(n) = text.trim().parse::<i64>() {
                    *value = json!(n);
                } else if let Some(n) = text.trim().parse::<f64>().ok().and_then(serde_json::Number::from_f64) {
                    *value = Value::Number(n);
                }
                return true;
            }
            false
        }
        (_, Value::String(text)) => {
            ui.add(egui::TextEdit::singleline(text).desired_width(f32::INFINITY).hint_text("…")).changed()
        }
        _ => json_line(ui, id, value),
    }
}

fn tool_calls(
    ui: &mut Ui,
    cx: &DatasetContext<'_>,
    offered: Option<&[ToolSpec]>,
    message: &mut Map<String, Value>,
    removable: bool,
) -> bool {
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
        let key = if call.contains_key("arguments") || !call.contains_key("args") { "arguments" } else { "args" };
        let id = ui.id().with(("call", n));
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
                let mut arguments = call.get(key).cloned().unwrap_or_else(|| json!({}));
                let name = call.get("name").and_then(Value::as_str).unwrap_or_default().to_owned();
                if !tool_on(cx, offered, &name) {
                    ui.label(
                        RichText::new(format!("{}  not on for this model", icon::WARNING))
                            .size(10.5)
                            .color(theme::warning()),
                    )
                    .on_hover_text(
                        "This model has no tool of that name (and the sample does not offer it): turn it on in \
                             the Tools or Native Tools block, or add it to the sample's tools",
                    );
                }
                let raw_id = id.with("raw");
                let schema = schema_of(cx, offered, &name).cloned();
                let mut raw = ui.data(|d| d.get_temp::<bool>(raw_id)).unwrap_or(schema.is_none());
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if removable
                        && ui
                            .add(egui::Button::new(RichText::new(icon::X).size(11.0)).frame(false))
                            .on_hover_text("Remove this call")
                            .clicked()
                    {
                        remove = Some(n);
                    }
                    if schema.is_some()
                        && widgets::icon_toggle(ui, icon::BRACKETS_CURLY, raw, "Edit the arguments as JSON").clicked()
                    {
                        raw = !raw;
                        ui.data_mut(|d| d.insert_temp(raw_id, raw));
                    }
                });
                if raw && json_line(ui, id, &mut arguments) {
                    call.insert(key.into(), arguments.clone());
                    changed = true;
                }
            });
            // One line per argument, as the tool's schema describes it.
            let properties = schema_of(cx, offered, call.get("name").and_then(Value::as_str).unwrap_or_default())
                .and_then(|s| s.get("properties"))
                .and_then(Value::as_object)
                .cloned();
            let raw = ui.data(|d| d.get_temp::<bool>(id.with("raw"))).unwrap_or(properties.is_none());
            if let (false, Some(properties), Some(Value::Object(arguments))) = (raw, properties, call.get_mut(key)) {
                egui::Grid::new(id.with("arguments")).num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
                    let mut keys: Vec<String> = properties.keys().cloned().collect();
                    keys.extend(arguments.keys().filter(|k| !properties.contains_key(*k)).cloned());
                    for key in keys {
                        let property = properties.get(&key);
                        let kind = property.and_then(|p| p.get("type")).and_then(Value::as_str);
                        let label = ui.label(RichText::new(&key).monospace().size(11.5).color(theme::text_dim()));
                        if let Some(about) = property.and_then(|p| p.get("description")).and_then(Value::as_str) {
                            label.on_hover_text(about);
                        }
                        let value = arguments.entry(key.clone()).or_insert_with(|| match kind {
                            Some("number" | "integer") => json!(0),
                            Some("boolean") => json!(false),
                            _ => json!(""),
                        });
                        if argument_edit(ui, id.with(("arg", &key)), value, kind) {
                            changed = true;
                        }
                        ui.end_row();
                    }
                });
            }
        });
    }
    if let Some(n) = remove {
        calls.remove(n);
        changed = true;
    }
    changed
}

/// The sample as the chat it teaches, every part editable where it shows.
fn chat_edit(ui: &mut Ui, cx: &DatasetContext<'_>, sample: &mut String) -> bool {
    let spec = cx.spec;
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
        changed |= conversation_edit(ui, cx, &mut item);
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

fn conversation_edit(ui: &mut Ui, cx: &DatasetContext<'_>, item: &mut Map<String, Value>) -> bool {
    let spec = cx.spec;
    let mut changed = false;
    // The tools this sample can call: its own ("tools"), or the model's.
    let offered: Option<Vec<ToolSpec>> = item.get("tools").and_then(Value::as_array).map(|list| {
        list.iter().filter_map(|t| serde_json::from_value(t.get("function").unwrap_or(t).clone()).ok()).collect()
    });
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
                    changed |= tool_calls(ui, cx, offered.as_deref(), message, true);
                    if let Some(tool) = tool_picker(ui, cx, offered.as_deref()) {
                        let calls = message.entry("tool_calls").or_insert_with(|| json!([]));
                        if let Some(calls) = calls.as_array_mut() {
                            calls.push(json!({ "name": tool.name, "arguments": arguments_template(&tool.parameters) }));
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
