//! The AI Agent window: a small chat app for the models MacroMagi makes and
//! runs – a preview of CodeMagi's. Models come from the Magic tab or from
//! `.szi` and `.magic` files; chats are kept on disk and named by
//! the model; each chat has its own settings (system prompt, sampling,
//! context, thinking, effort, native tools, compaction); files can be
//! attached to a message; sent messages can be edited, deleted or answered
//! again.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use egui::{Align, Layout, RichText, Ui, vec2};
use egui_phosphor::regular as icon;
use magi_core::magic::{MagicFile, ModelKind, TextSpec};
use serde::{Deserialize, Serialize};

use super::magic_tab::{fingerprint, load_engine};
use super::widgets;
use crate::theme;

/// What a model is asked to name a chat with (as the example datasets teach it).
const TITLE_ASK: &str = "Write a short title (2-5 words) for this conversation. Reply with the title only.";
/// Text files up to this size go into the message; bigger ones, pictures and documents are
/// named by their path, and the model reads them with its tools (read_file, view_image,
/// open_document) – as Magi's datasets teach it.
const ATTACH_INLINE: usize = 20_000;
/// Pictures, by their extension.
const PICTURES: [&str; 8] = ["png", "jpg", "jpeg", "gif", "bmp", "webp", "tif", "tiff"];
/// Documents its document tools read, by their extension.
const DOCUMENTS: [&str; 7] = ["docx", "xlsx", "xlsm", "pptx", "pdf", "csv", "html"];
/// The widths of the chat list, the settings and the column of messages.
const CHATS_WIDTH: f32 = 236.0;
const SETTINGS_WIDTH: f32 = 300.0;
const MESSAGES_WIDTH: f32 = 700.0;
/// The size of a message's text.
const BODY: f32 = 14.5;
/// The narrowest the messages get before the chat list is hidden.
const MESSAGES_MIN: f32 = 340.0;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Attachment {
    pub name: String,
    /// A small text file's text; empty when the model reads the file itself.
    pub text: String,
    /// Where the file is, for the model's tools.
    pub path: String,
    /// `picture`, `document` or `file` (text the model reads with read_file).
    pub kind: String,
}

impl Attachment {
    /// How the attachment reads in the message (as Magi's datasets write it).
    pub fn in_message(&self) -> String {
        if self.path.is_empty() || (!self.text.is_empty() && self.kind == "file") {
            format!("\n\nFile {}:\n{}", self.name, self.text)
        } else {
            format!("\n\nAttached: {} ({})", self.path, self.kind)
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatMessage {
    /// `user` or `assistant`.
    pub role: String,
    /// What shows: the request, or the answer without its reasoning.
    pub text: String,
    /// An answer as the model wrote it, without its reasoning: what it reads back.
    pub raw: String,
    pub thinking: String,
    pub attachments: Vec<Attachment>,
    /// The native tools it ran on the way, as chats before 0.1.7.2 kept them: each call and its result.
    pub steps: Vec<(String, String)>,
    /// The tools it called on the way: what it thought, each call, and what came back.
    pub tools: Vec<ToolStep>,
    pub error: Option<String>,
}

/// A tool an answer called: what the model thought first, the call, and what came back.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolStep {
    pub thinking: String,
    pub name: String,
    /// The arguments, as JSON.
    pub args: String,
    /// What came back, as JSON; empty: it did not run, and `note` says why.
    pub result: String,
    pub error: bool,
    pub note: String,
}

impl ToolStep {
    /// A step as chats before 0.1.7.2 kept it: `name(args)` and the result.
    fn from_old(call: &str, result: &str) -> Self {
        let (name, args) = call
            .split_once('(')
            .map_or((call.to_owned(), String::new()), |(n, a)| (n.to_owned(), a.trim_end_matches(')').to_owned()));
        Self { name, args, result: result.to_owned(), error: result.contains("\"error\""), ..Default::default() }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatSettings {
    /// Instead of the model's own system prompt; empty: the model's.
    pub system: String,
    /// Below 0: the model's.
    pub temperature: f32,
    /// 0: the model's.
    pub max_tokens: u32,
    /// 0: the model's context window.
    pub context: u32,
    /// `model`, `on` or `off`.
    pub thinking: String,
    /// Empty: the model's default level.
    pub effort: String,
    pub native_tools: bool,
    pub max_steps: u32,
    /// Write the earlier part short when the chat outgrows the context (models trained to compact).
    pub compact: bool,
}

impl Default for ChatSettings {
    fn default() -> Self {
        Self {
            system: String::new(),
            temperature: -1.0,
            max_tokens: 0,
            context: 0,
            thinking: "model".into(),
            effort: String::new(),
            native_tools: true,
            max_steps: 100,
            compact: true,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatSession {
    pub id: String,
    pub title: String,
    /// Named – by the model or by hand – so it is not named again.
    pub titled: bool,
    pub messages: Vec<ChatMessage>,
    pub settings: ChatSettings,
    /// The earlier part of the chat, written short by the model, and how many messages it holds.
    pub summary: String,
    pub compacted: usize,
    /// When it was last changed (seconds since 1970), for the order of the list.
    pub changed: u64,
}

/// A model to chat with.
struct ChatModel {
    name: String,
    file: Arc<MagicFile>,
    /// Where it came from; `None`: the Magic tab's model.
    path: Option<PathBuf>,
}

/// How a model is run.
#[derive(Clone, Copy, PartialEq)]
enum Way {
    /// MacroMagi's native engine (from its weights) or its ONNX model.
    BuiltIn { native: bool },
}

/// What a request came back with.
struct Reply {
    answer: Result<magi_ml::tools::AgentAnswer, String>,
    /// The earlier part written short, and how many messages that covers.
    summary: Option<(String, usize)>,
    title: Option<String>,
}

type Pending = Arc<Mutex<Option<Reply>>>;
type Loading = Arc<Mutex<Option<Result<(), String>>>>;
type Engine = Arc<Mutex<Option<(u64, magi_ml::TextModel)>>>;

#[derive(Default)]
pub struct AgentChat {
    pub open: bool,
    dir: Option<PathBuf>,
    chats: Vec<ChatSession>,
    current: usize,
    models: Vec<ChatModel>,
    model: usize,
    engine: Engine,
    loading: Option<Loading>,
    /// The chat a request is being answered for, and the answer when it comes.
    pending: Option<(String, Pending)>,
    input: String,
    attachments: Vec<Attachment>,
    editing: Option<(usize, String)>,
    renaming: Option<(usize, String)>,
    pub(crate) show_settings: bool,
    /// What the model thought, shown opened (for every chat; the app's settings keep it).
    pub(crate) open_thoughts: bool,
    error: Option<String>,
    /// The Magic tab's model as last offered: its weights' and ONNX model's sizes, its text spec
    /// and name – to tell when it changed without keeping a second copy of it to compare.
    offered: Option<(usize, usize, Option<TextSpec>, String)>,
    /// Models opened from files, let go of while the window is closed, and whether one was chosen.
    closed: Vec<(PathBuf, bool)>,
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

impl AgentChat {
    /// The chats folder, read once.
    pub fn set_dir(&mut self, dir: PathBuf) {
        if self.dir.as_ref() == Some(&dir) {
            return;
        }
        self.chats = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
            .filter_map(|e| serde_json::from_slice::<ChatSession>(&std::fs::read(e.path()).ok()?).ok())
            .collect();
        self.chats.sort_by_key(|c| std::cmp::Reverse(c.changed));
        self.dir = Some(dir);
        if self.chats.is_empty() {
            self.new_chat();
        }
        self.current = 0;
    }

    #[cfg(test)]
    pub fn chat_count(&self) -> usize {
        self.chats.len()
    }

    fn new_chat(&mut self) {
        let id = format!("chat-{}-{:04x}", now(), rand_bits());
        let settings = self.chats.get(self.current).map(|c| c.settings.clone()).unwrap_or_default();
        self.chats
            .insert(0, ChatSession { id, title: "New chat".into(), settings, changed: now(), ..Default::default() });
        self.current = 0;
        self.editing = None;
    }

    fn save(&self, index: usize) {
        let (Some(dir), Some(chat)) = (&self.dir, self.chats.get(index)) else { return };
        let _ = std::fs::create_dir_all(dir);
        if let Ok(bytes) = serde_json::to_vec_pretty(chat) {
            let _ = std::fs::write(dir.join(format!("{}.json", chat.id)), bytes);
        }
    }

    fn delete_chat(&mut self, index: usize) {
        if let (Some(dir), Some(chat)) = (&self.dir, self.chats.get(index)) {
            let _ = std::fs::remove_file(dir.join(format!("{}.json", chat.id)));
        }
        self.chats.remove(index);
        if self.chats.is_empty() {
            self.new_chat();
        }
        self.current = self.current.min(self.chats.len() - 1);
    }

    /// Offers the Magic tab's model (kept up to date with it) first in the list.
    pub fn offer_magic_model(&mut self, file: Option<&MagicFile>) {
        let usable = file.filter(|f| f.model.kind == ModelKind::Text && (!f.weights.is_empty() || !f.onnx.is_empty()));
        let first_is_magic = self.models.first().is_some_and(|m| m.path.is_none());
        match usable {
            Some(file) => {
                let signature = (file.weights.len(), file.onnx.len(), file.model.text.clone(), file.model.name.clone());
                let changed = !first_is_magic || self.offered.as_ref() != Some(&signature);
                if changed {
                    self.offered = Some(signature);
                    let model = ChatModel {
                        name: format!(
                            "{} (Magic tab)",
                            if file.model.name.is_empty() { "Model" } else { &file.model.name }
                        ),
                        file: Arc::new(chat_copy(file)),
                        path: None,
                    };
                    if first_is_magic {
                        self.models[0] = model;
                        if self.model == 0 {
                            *self.engine.lock().unwrap_or_else(|p| p.into_inner()) = None;
                        }
                    } else {
                        self.models.insert(0, model);
                        if !self.models.is_empty() && self.models.len() > 1 {
                            self.model += 1;
                        }
                    }
                }
            }
            None if first_is_magic => {
                self.models.remove(0);
                self.model = self.model.saturating_sub(1);
                self.offered = None;
            }
            None => {}
        }
    }

    /// While the window is closed it keeps nothing big: no copy of the Magic tab's model, and no
    /// model loaded (models opened from files stay listed, to load again when it opens).
    pub fn release(&mut self) {
        if self.models.is_empty() || self.pending.is_some() {
            return;
        }
        let chosen = self.model;
        for (index, model) in std::mem::take(&mut self.models).into_iter().enumerate() {
            if let Some(path) = model.path {
                self.closed.push((path, index == chosen));
            }
        }
        self.model = 0;
        self.offered = None;
        *self.engine.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }

    /// Opens again the models let go of when the window closed.
    fn reopen(&mut self) {
        let magic = usize::from(self.models.first().is_some_and(|m| m.path.is_none()));
        for (path, chosen) in std::mem::take(&mut self.closed) {
            if let Ok(model) = open_model(&path) {
                self.models.push(model);
                if chosen {
                    self.model = self.models.len() - 1;
                }
            }
        }
        if self.model >= self.models.len() {
            self.model = magic.min(self.models.len()).saturating_sub(1);
        }
    }

    fn load_file(&mut self) {
        let Some(path) = rfd::FileDialog::new().add_filter("AI model", &["szi", "magic"]).pick_file() else {
            return;
        };
        match open_model(&path) {
            Ok(model) => {
                self.models.push(model);
                self.model = self.models.len() - 1;
                *self.engine.lock().unwrap_or_else(|p| p.into_inner()) = None;
                self.error = None;
            }
            Err(error) => self.error = Some(error),
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open {
            return;
        }
        if !self.closed.is_empty() {
            self.reopen();
        }
        let builder = egui::ViewportBuilder::default()
            .with_title("AI Agent • MacroMagi")
            .with_inner_size([1040.0, 720.0])
            .with_min_inner_size([620.0, 420.0]);
        let mut closed = false;
        ctx.show_viewport_immediate(egui::ViewportId::from_hash_of("magi-agent-chat"), builder, |ui, class| {
            match class {
                egui::ViewportClass::EmbeddedWindow => self.contents(ui),
                _ => {
                    egui::CentralPanel::default()
                        .frame(egui::Frame::new().fill(theme::bg_app().to_opaque()))
                        .show(ui, |ui| self.contents(ui));
                }
            }
            closed = ui.input(|i| i.viewport().close_requested());
        });
        if closed {
            self.open = false;
        }
    }

    fn contents(&mut self, ui: &mut Ui) {
        self.poll(ui.ctx());
        // Fixed widths: the panels hold their size whatever is in them, and what is in them
        // fits (truncated or wrapped) instead of pushing them wider. A narrow window keeps the
        // messages readable: the chat list makes way first.
        let settings = if self.show_settings { SETTINGS_WIDTH } else { 0.0 };
        if ui.available_width() - settings >= CHATS_WIDTH + MESSAGES_MIN {
            egui::Panel::left("agent-chats")
                .resizable(false)
                .exact_size(CHATS_WIDTH)
                .frame(egui::Frame::new().fill(theme::bg_panel()).inner_margin(egui::Margin::symmetric(10, 12)))
                .show(ui, |ui| self.chat_list(ui));
        }
        if self.show_settings {
            egui::Panel::right("agent-settings")
                .resizable(false)
                .exact_size(SETTINGS_WIDTH)
                .frame(egui::Frame::new().fill(theme::bg_panel()).inner_margin(egui::Margin::symmetric(14, 12)))
                .show(ui, |ui| {
                    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                        ui.set_max_width(SETTINGS_WIDTH - 28.0);
                        self.settings_panel(ui);
                    });
                });
        }
        egui::Panel::top("agent-model")
            .resizable(false)
            .frame(
                egui::Frame::new()
                    .fill(theme::bg_panel())
                    .inner_margin(egui::Margin::symmetric(14, 8))
                    .stroke(egui::Stroke::new(1.0, theme::stroke())),
            )
            .show(ui, |ui| self.model_bar(ui));
        egui::Panel::bottom("agent-input")
            .resizable(false)
            .frame(egui::Frame::new().fill(theme::bg_app()).inner_margin(egui::Margin::symmetric(16, 12)))
            .show(ui, |ui| self.input_box(ui));
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(theme::bg_app()).inner_margin(egui::Margin::symmetric(16, 12)))
            .show(ui, |ui| self.messages(ui));
    }

    // ---- the list of chats ----

    fn chat_list(&mut self, ui: &mut Ui) {
        let busy = self.pending.is_some();
        ui.horizontal(|ui| {
            ui.label(RichText::new("Chats").size(14.0).strong());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let new = ui
                    .add_enabled(!busy, egui::Button::new(RichText::new(format!("{}  New", icon::PLUS)).size(12.5)))
                    .on_hover_text("A new chat, with this one's settings");
                if new.clicked() {
                    self.new_chat();
                }
            });
        });
        ui.add_space(8.0);
        let mut select = None;
        let mut delete = None;
        let mut renamed = None;
        let mut rename = None;
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            let width = ui.available_width();
            for (index, chat) in self.chats.iter().enumerate() {
                let current = index == self.current;
                if let Some((at, name)) = &mut self.renaming
                    && *at == index
                {
                    let edit = ui.add(egui::TextEdit::singleline(name).desired_width(width));
                    if edit.lost_focus() {
                        renamed = Some((index, name.clone()));
                    } else {
                        edit.request_focus();
                    }
                    ui.add_space(2.0);
                    continue;
                }
                // A row: the title, cut short to fit, and its buttons on hover.
                let (rect, response) = ui.allocate_exact_size(vec2(width, 30.0), egui::Sense::click());
                let hovered = response.hovered();
                let fill = if current {
                    theme::bg_widget_active()
                } else if hovered {
                    theme::bg_widget()
                } else {
                    egui::Color32::TRANSPARENT
                };
                ui.painter().rect_filled(rect, theme::RADIUS, fill);
                let show_buttons = (hovered || current) && !busy;
                let text_right = if show_buttons { rect.right() - 54.0 } else { rect.right() - 8.0 };
                let text_rect =
                    egui::Rect::from_min_max(rect.left_top() + vec2(10.0, 0.0), egui::pos2(text_right, rect.bottom()));
                let mut job = egui::text::LayoutJob::simple_singleline(
                    chat.title.clone(),
                    egui::FontId::proportional(12.5),
                    if current { theme::text() } else { theme::text_dim() },
                );
                job.wrap = egui::text::TextWrapping::truncate_at_width(text_rect.width().max(10.0));
                let galley = ui.painter().layout_job(job);
                ui.painter().galley(
                    egui::pos2(text_rect.left(), rect.center().y - galley.size().y / 2.0),
                    galley,
                    theme::text(),
                );
                if show_buttons {
                    let pencil = egui::Rect::from_center_size(
                        egui::pos2(rect.right() - 40.0, rect.center().y),
                        vec2(22.0, 22.0),
                    );
                    let trash = egui::Rect::from_center_size(
                        egui::pos2(rect.right() - 16.0, rect.center().y),
                        vec2(22.0, 22.0),
                    );
                    let small = |glyph: &str| egui::Button::new(RichText::new(glyph).size(13.0)).frame(false);
                    if ui.put(pencil, small(icon::PENCIL_SIMPLE)).on_hover_text("Rename").clicked() {
                        rename = Some((index, chat.title.clone()));
                    }
                    if ui.put(trash, small(icon::TRASH)).on_hover_text("Delete this chat").clicked() {
                        delete = Some(index);
                    }
                }
                if response.clicked() && !busy {
                    select = Some(index);
                }
                if response.double_clicked() {
                    rename = Some((index, chat.title.clone()));
                }
                ui.add_space(2.0);
            }
        });
        if let Some(name) = rename {
            self.renaming = Some(name);
        }
        if let Some((index, name)) = renamed {
            if let Some(chat) = self.chats.get_mut(index)
                && !name.trim().is_empty()
            {
                chat.title = name.trim().to_owned();
                chat.titled = true;
            }
            self.renaming = None;
            self.save(index);
        }
        if let Some(index) = select {
            self.current = index;
            self.editing = None;
        }
        if let Some(index) = delete {
            self.delete_chat(index);
        }
    }

    // ---- the model ----

    fn way(&self) -> Option<(Way, &ChatModel)> {
        let model = self.models.get(self.model)?;
        let file = &model.file;
        let own = file.model.architecture.is_some();
        let native = own
            && !file.weights.is_empty()
            && file.model.architecture.as_ref().is_some_and(|a| a.native_problem().is_none());
        let way = if native {
            Way::BuiltIn { native: true }
        } else if !file.onnx.is_empty() {
            Way::BuiltIn { native: false }
        } else {
            return None;
        };
        Some((way, model))
    }

    fn key(model: &ChatModel, way: Way) -> u64 {
        match way {
            Way::BuiltIn { native: true } => fingerprint(&model.file.weights),
            Way::BuiltIn { native: false } => fingerprint(&model.file.onnx),
        }
    }

    fn model_bar(&mut self, ui: &mut Ui) {
        let busy = self.pending.is_some() || self.loading.is_some();
        let way = self.way().map(|(way, model)| {
            let key = Self::key(model, way);
            let loaded = self.engine.try_lock().map_or(true, |s| s.as_ref().is_some_and(|(k, _)| *k == key));
            (way, loaded)
        });
        let mut load = false;
        let mut pick = None;
        let mut open_file = false;
        ui.horizontal(|ui| {
            ui.set_min_height(30.0);
            // From the right: the settings switch, loading, the state – so they never slide under a panel.
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if widgets::icon_toggle(ui, icon::SLIDERS_HORIZONTAL, self.show_settings, "This chat's settings")
                    .clicked()
                {
                    self.show_settings = !self.show_settings;
                }
                if let Some((way, loaded)) = &way {
                    ui.add_enabled_ui(!busy, |ui| {
                        if *loaded {
                            if widgets::icon_button(ui, icon::EJECT, "Unload: free its memory").clicked() {
                                *self.engine.lock().unwrap_or_else(|p| p.into_inner()) = None;
                            }
                        } else if widgets::icon_button(
                            ui,
                            icon::UPLOAD_SIMPLE,
                            "Load it now (else it loads with the first message)",
                        )
                        .clicked()
                        {
                            load = true;
                        }
                    });
                    let (text, color) = if self.loading.is_some() {
                        ("Loading…", theme::text_dim())
                    } else if *loaded {
                        ("Loaded", theme::success())
                    } else {
                        ("Not loaded", theme::text_faint())
                    };
                    let engine = match way {
                        Way::BuiltIn { native: true } => "native engine",
                        Way::BuiltIn { native: false } => "ONNX",
                    };
                    egui::Frame::new()
                        .fill(theme::bg_input())
                        .corner_radius(10)
                        .inner_margin(egui::Margin::symmetric(8, 3))
                        .show(ui, |ui| {
                            ui.label(RichText::new(format!("{}  {text}", icon::CIRCLE)).size(11.0).color(color))
                                .on_hover_text(format!("Runs on the {engine}"));
                        });
                }
                // Then, from the left, the model: as wide as there is room for.
                ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                    ui.label(RichText::new(icon::ROBOT).size(17.0).color(theme::text_dim()));
                    let current = self.models.get(self.model).map_or("No model".to_owned(), |m| m.name.clone());
                    let width = (ui.available_width() - 44.0).clamp(120.0, 320.0);
                    ui.add_enabled_ui(!busy, |ui| {
                        egui::ComboBox::from_id_salt("agent-model")
                            .width(width)
                            .truncate()
                            .selected_text(RichText::new(current).strong())
                            .show_ui(ui, |ui| {
                                for (index, model) in self.models.iter().enumerate() {
                                    let hint =
                                        model.path.as_ref().map_or("The model open on the Magic tab".to_owned(), |p| {
                                            p.display().to_string()
                                        });
                                    if ui
                                        .selectable_label(index == self.model, &model.name)
                                        .on_hover_text(hint)
                                        .clicked()
                                    {
                                        pick = Some(index);
                                    }
                                }
                            });
                        if widgets::icon_button(ui, icon::FOLDER_OPEN, "Open a model file: .szi or .magic").clicked() {
                            open_file = true;
                        }
                    });
                });
            });
        });
        if let Some(index) = pick.filter(|i| *i != self.model) {
            self.model = index;
            *self.engine.lock().unwrap_or_else(|p| p.into_inner()) = None;
        }
        if open_file {
            self.load_file();
        }
        if load && let Some((way, ..)) = way {
            self.load(ui.ctx(), way);
        }
        if way.is_none() {
            ui.label(
                RichText::new("Train a text model on the Magic tab, or open a model file.")
                    .size(11.5)
                    .color(theme::text_faint()),
            );
        }
        if let Some(error) = &self.error {
            ui.add(egui::Label::new(RichText::new(error).size(11.5).color(theme::danger())).wrap());
        }
    }

    fn load(&mut self, ctx: &egui::Context, way: Way) {
        let Some((_, model)) = self.way() else { return };
        let slot: Loading = Arc::new(Mutex::new(None));
        let done = Arc::clone(&slot);
        let ctx = ctx.clone();
        let file = Arc::clone(&model.file);
        let engine = Arc::clone(&self.engine);
        let key = Self::key(model, way);
        std::thread::spawn(move || {
            let spec = file.model.text.clone().unwrap_or_default();
            let Way::BuiltIn { native } = way;
            let architecture = file.model.architecture.clone().unwrap_or_default();
            let memory = file.model.memory.clone().unwrap_or_default();
            let bytes = if native { &file.weights } else { &file.onnx };
            let result = load_engine(native, &architecture, &memory, &spec, bytes).map(|model| {
                *engine.lock().unwrap_or_else(|p| p.into_inner()) = Some((key, model));
            });
            *done.lock().unwrap_or_else(|p| p.into_inner()) = Some(result.map_err(|e| e.0));
            ctx.request_repaint();
        });
        self.loading = Some(slot);
    }

    fn options(&self) -> magi_ml::AskOptions {
        let settings = self.chats.get(self.current).map(|c| c.settings.clone()).unwrap_or_default();
        magi_ml::AskOptions {
            max_tokens: settings.max_tokens as usize,
            temperature: settings.temperature,
            seed: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64),
            context: settings.context as usize,
            thinking: match settings.thinking.as_str() {
                "on" => Some(true),
                "off" => Some(false),
                _ => None,
            },
            effort: settings.effort.clone(),
        }
    }

    // ---- a chat's settings ----

    fn settings_panel(&mut self, ui: &mut Ui) {
        let spec = self.models.get(self.model).and_then(|m| m.file.model.text.clone()).unwrap_or_default();
        let compacts =
            self.models.get(self.model).is_some_and(|m| m.file.model.trained.as_ref().is_some_and(|t| t.can_compact()));
        let index = self.current;
        let Some(chat) = self.chats.get_mut(index) else { return };
        let before = chat.settings.clone();
        let s = &mut chat.settings;
        // A setting: its name on the left, the control on the right, in one row.
        let row = |ui: &mut Ui, label: &str, add: &mut dyn FnMut(&mut Ui)| {
            ui.horizontal(|ui| {
                ui.set_min_height(26.0);
                ui.label(RichText::new(label).color(theme::text_dim()));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| add(ui));
            });
        };
        ui.label(RichText::new("Chat settings").size(14.0).strong());
        ui.add_space(8.0);
        ui.label(RichText::new("System prompt").color(theme::text_dim()));
        ui.add(
            egui::TextEdit::multiline(&mut s.system)
                .hint_text(if spec.system.is_empty() { "None" } else { spec.system.as_str() })
                .desired_rows(4)
                .desired_width(f32::INFINITY),
        );
        widgets::note(ui, "Empty: the model's own (shown faintly).");
        ui.add_space(8.0);
        ui.separator();
        if let Some(thinking) = &spec.thinking {
            let mut on = match s.thinking.as_str() {
                "on" => true,
                "off" => false,
                _ => thinking.on_by_default,
            };
            row(ui, "Thinking", &mut |ui| {
                if widgets::toggle(ui, &mut on).changed() {
                    s.thinking = if on { "on" } else { "off" }.into();
                }
            });
        }
        if let Some(effort) = &spec.effort {
            row(ui, "Effort", &mut |ui| {
                let mut level = effort.level(&s.effort).to_owned();
                // Reversed: the row lays its controls out from the right.
                let options: Vec<(String, &str)> =
                    effort.levels.iter().rev().map(|l| (l.clone(), l.as_str())).collect();
                if widgets::segmented(ui, &mut level, &options) {
                    s.effort = level;
                }
            });
        }
        if spec.tools.iter().any(|t| t.native) {
            row(ui, "Native tools", &mut |ui| {
                widgets::toggle(ui, &mut s.native_tools).on_hover_text("Run the tools it calls on this computer");
            });
            if s.native_tools {
                row(ui, "Tool calls at most", &mut |ui| {
                    ui.add(egui::DragValue::new(&mut s.max_steps).range(1..=200));
                });
            }
        }
        row(ui, "Compact long chats", &mut |ui| {
            ui.add_enabled(compacts, egui::Checkbox::without_text(&mut s.compact))
                .on_hover_text("When the chat outgrows the context window, the model writes its earlier part short and goes on from that")
                .on_disabled_hover_text("This model did not learn compaction (compaction samples)");
        });
        ui.add_space(4.0);
        ui.separator();
        ui.label(RichText::new("Sampling").color(theme::text_dim()));
        let mut own = s.temperature < 0.0;
        row(ui, "Temperature", &mut |ui| {
            if !own {
                ui.add(egui::DragValue::new(&mut s.temperature).range(0.0..=2.0).speed(0.01).max_decimals(2));
            }
            if ui.checkbox(&mut own, format!("model's ({})", spec.temperature)).changed() {
                s.temperature = if own { -1.0 } else { spec.temperature };
            }
        });
        let mut own = s.max_tokens == 0;
        row(ui, "Longest answer", &mut |ui| {
            if !own {
                ui.add(egui::DragValue::new(&mut s.max_tokens).range(1..=65536));
            }
            if ui.checkbox(&mut own, format!("model's ({})", spec.max_tokens)).changed() {
                s.max_tokens = if own { 0 } else { spec.max_tokens.max(16) };
            }
        });
        let mut own = s.context == 0;
        row(ui, "Context", &mut |ui| {
            if !own {
                ui.add(egui::DragValue::new(&mut s.context).range(256..=1_048_576).speed(64));
            }
            if ui.checkbox(&mut own, format!("model's ({})", spec.context)).changed() {
                s.context = if own { 0 } else { spec.context.max(256) };
            }
        });
        let changed = chat.settings != before;
        if !chat.summary.is_empty() {
            ui.add_space(8.0);
            ui.separator();
            ui.label(RichText::new("The earlier part, written short").color(theme::text_dim()));
            widgets::note(ui, &chat.summary);
        }
        if changed {
            self.save(index);
        }
        ui.add_space(8.0);
        ui.separator();
        ui.label(RichText::new("Every chat").color(theme::text_dim()));
        row(ui, "Thinking shown opened", &mut |ui| {
            widgets::toggle(ui, &mut self.open_thoughts)
                .on_hover_text("What the model thought, shown in full instead of folded under Thought");
        });
    }

    // ---- the messages ----

    fn messages(&mut self, ui: &mut Ui) {
        let index = self.current;
        let open_thoughts = self.open_thoughts;
        let pending = self.pending.as_ref().is_some_and(|(id, _)| self.chats.get(index).is_some_and(|c| &c.id == id));
        let model_name = self.models.get(self.model).map_or("Model".to_owned(), |m| m.name.clone());
        let mut action: Option<(usize, &'static str)> = None;
        let mut saved_edit: Option<(usize, String)> = None;
        egui::ScrollArea::vertical().auto_shrink([false, false]).stick_to_bottom(true).show(ui, |ui| {
            let Some(chat) = self.chats.get(index) else { return };
            // One column, centred, as wide as reads well – as chat apps do.
            let full = ui.available_width();
            let column = (full - 24.0).clamp(240.0, MESSAGES_WIDTH);
            let margin = ((full - column) / 2.0).max(0.0);
            // Placed by its rectangle: a row of its own would centre it vertically and leave gaps.
            let in_column = |ui: &mut Ui, add: &mut dyn FnMut(&mut Ui)| {
                let top_left = egui::pos2(ui.max_rect().left() + margin, ui.cursor().top());
                let rect = egui::Rect::from_min_size(top_left, vec2(column, ui.available_height().max(1.0)));
                ui.scope_builder(egui::UiBuilder::new().max_rect(rect).layout(Layout::top_down(Align::Min)), |ui| {
                    ui.set_width(column);
                    add(ui);
                });
            };
            ui.add_space(16.0);
            if chat.messages.is_empty() {
                ui.add_space(90.0);
                ui.vertical_centered(|ui| {
                    ui.label(RichText::new(icon::SPARKLE).size(34.0).color(theme::text_faint()));
                    ui.add_space(8.0);
                    ui.label(RichText::new("What can I help with?").size(22.0).color(theme::text()));
                    ui.add_space(4.0);
                    ui.label(RichText::new(&model_name).size(12.5).color(theme::text_faint()));
                });
            }
            if chat.compacted > 0 {
                in_column(ui, &mut |ui| {
                    widgets::notice(
                        ui,
                        icon::ARROWS_IN,
                        format!(
                            "The first {} messages are written short for the model (chat settings show how).",
                            chat.compacted
                        ),
                        theme::text_faint(),
                    );
                });
                ui.add_space(8.0);
            }
            for (n, message) in chat.messages.iter().enumerate() {
                let user = message.role == "user";
                ui.push_id(n, |ui| {
                    in_column(ui, &mut |ui| {
                        if user {
                            // The request: a bubble on the right, as wide as its text, at most three quarters of the column.
                            let most = (column * 0.78).max(220.0);
                            let editing = self.editing.as_ref().is_some_and(|(at, _)| *at == n);
                            let painter = ui.painter().clone();
                            let natural = |text: &str, size: f32| {
                                text.lines()
                                    .map(|line| {
                                        painter
                                            .layout_no_wrap(
                                                line.to_owned(),
                                                egui::FontId::proportional(size),
                                                theme::text(),
                                            )
                                            .size()
                                            .x
                                    })
                                    .fold(0.0, f32::max)
                            };
                            let mut width = natural(&message.text, BODY);
                            for file in &message.attachments {
                                width = width.max(natural(&format!("{}  {}", icon::FILE_TEXT, file.name), 11.5));
                            }
                            let width = if editing { most } else { (width + 2.0).clamp(24.0, most) };
                            ui.allocate_ui_with_layout(vec2(column, 0.0), Layout::right_to_left(Align::Min), |ui| {
                                egui::Frame::new()
                                    .fill(theme::bg_widget_active())
                                    .corner_radius(18)
                                    .inner_margin(egui::Margin::symmetric(14, 10))
                                    .show(ui, |ui| {
                                        ui.set_width(width);
                                        ui.with_layout(Layout::top_down(Align::Min), |ui| {
                                            if let Some((at, text)) = &mut self.editing
                                                && *at == n
                                            {
                                                ui.add(
                                                    egui::TextEdit::multiline(text)
                                                        .desired_rows(2)
                                                        .desired_width(width),
                                                );
                                                ui.horizontal(|ui| {
                                                    if widgets::accent_button(
                                                        ui,
                                                        format!("{}  Send", icon::PAPER_PLANE_RIGHT),
                                                        theme::accent(),
                                                        vec2(0.0, 26.0),
                                                    )
                                                    .clicked()
                                                    {
                                                        saved_edit = Some((n, text.clone()));
                                                    }
                                                    if ui.button("Cancel").clicked() {
                                                        action = Some((n, "cancel"));
                                                    }
                                                });
                                                return;
                                            }
                                            for file in &message.attachments {
                                                ui.label(
                                                    RichText::new(format!("{}  {}", icon::FILE_TEXT, file.name))
                                                        .size(11.5)
                                                        .color(theme::text_dim()),
                                                );
                                            }
                                            ui.add(
                                                egui::Label::new(RichText::new(&message.text).size(BODY))
                                                    .wrap()
                                                    .selectable(true),
                                            );
                                        });
                                    });
                            });
                        } else {
                            // The answer: no bubble, the column's width – each tool it ran (with what it
                            // thought first), what it thought last, then the text.
                            let mut last_thought = String::new();
                            for (k, step) in message.tools.iter().enumerate() {
                                if !step.thinking.trim().is_empty() {
                                    thought(ui, ("step-thinking", n, k), "Thought", &step.thinking, open_thoughts);
                                    last_thought = step.thinking.trim().to_owned();
                                }
                                tool_step(ui, (n, k), step);
                            }
                            // Chats from before the steps were kept in full.
                            for (k, (call, result)) in
                                message.steps.iter().enumerate().filter(|_| message.tools.is_empty())
                            {
                                tool_step(ui, (n, k), &ToolStep::from_old(call, result));
                            }
                            if !message.thinking.trim().is_empty() && message.thinking.trim() != last_thought {
                                thought(ui, ("thinking", n), "Thought", &message.thinking, open_thoughts);
                            }
                            if !message.text.trim().is_empty() {
                                answer_text(ui, &message.text);
                            }
                            if let Some(error) = &message.error {
                                widgets::notice(ui, icon::WARNING_CIRCLE, error.clone(), theme::danger());
                            }
                        }
                        // Its buttons, under it, small and faint.
                        if !pending && self.editing.is_none() {
                            let layout = if user {
                                Layout::right_to_left(Align::Min)
                            } else {
                                Layout::left_to_right(Align::Min)
                            };
                            ui.allocate_ui_with_layout(vec2(column, 0.0), layout, |ui| {
                                // A short row: the theme's buttons are taller than these small icons.
                                ui.spacing_mut().item_spacing.x = 2.0;
                                ui.spacing_mut().interact_size.y = 18.0;
                                ui.spacing_mut().button_padding.y = 0.0;
                                let small = |ui: &mut Ui, glyph: &str, hint: &str| {
                                    ui.add(
                                        egui::Button::new(RichText::new(glyph).size(13.5).color(theme::text_faint()))
                                            .frame(false),
                                    )
                                    .on_hover_text(hint)
                                    .clicked()
                                };
                                if small(ui, icon::COPY_SIMPLE, "Copy") {
                                    ui.ctx().copy_text(message.text.clone());
                                }
                                if user
                                    && small(
                                        ui,
                                        icon::PENCIL_SIMPLE,
                                        "Edit, then send again (the messages after it go)",
                                    )
                                {
                                    action = Some((n, "edit"));
                                }
                                if !user
                                    && small(
                                        ui,
                                        icon::ARROW_CLOCKWISE,
                                        "Answer again (this answer and what follows go)",
                                    )
                                {
                                    action = Some((n, "again"));
                                }
                                if small(ui, icon::TRASH, "Delete this message") {
                                    action = Some((n, "delete"));
                                }
                            });
                        }
                    });
                    ui.add_space(6.0);
                });
            }
            if pending {
                in_column(ui, &mut |ui| {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(RichText::new("Thinking…").size(13.0).color(theme::text_dim()));
                    });
                });
            }
            ui.add_space(12.0);
        });
        let ctx = ui.ctx().clone();
        if let Some((n, text)) = saved_edit {
            if let Some(chat) = self.chats.get_mut(index) {
                chat.messages.truncate(n + 1);
                chat.messages[n].text = text;
                chat.compacted = chat.compacted.min(n);
            }
            self.editing = None;
            self.ask(&ctx);
        }
        let Some((n, what)) = action else { return };
        let Some(chat) = self.chats.get_mut(index) else { return };
        match what {
            "edit" => self.editing = Some((n, chat.messages[n].text.clone())),
            "cancel" => self.editing = None,
            "again" => {
                chat.messages.truncate(n);
                chat.compacted = chat.compacted.min(n);
                if chat.messages.last().is_some_and(|m| m.role == "user") {
                    self.ask(&ctx);
                }
            }
            "delete" => {
                chat.messages.remove(n);
                chat.compacted = chat.compacted.min(chat.messages.len());
                self.save(index);
            }
            _ => {}
        }
    }

    // ---- writing ----

    fn input_box(&mut self, ui: &mut Ui) {
        let busy = self.pending.is_some();
        let ready = self.way().is_some();
        let full = ui.available_width();
        let column = (full - 24.0).clamp(240.0, MESSAGES_WIDTH);
        let margin = ((full - column) / 2.0).max(0.0);
        ui.horizontal(|ui| {
            ui.add_space(margin);
            ui.allocate_ui_with_layout(vec2(column, 0.0), Layout::top_down(Align::Min), |ui| {
                ui.set_width(column);
                if !self.attachments.is_empty() {
                    let mut remove = None;
                    ui.horizontal_wrapped(|ui| {
                        for (n, file) in self.attachments.iter().enumerate() {
                            if ui
                                .small_button(format!("{}  {}  {}", icon::FILE_TEXT, file.name, icon::X))
                                .on_hover_text("Take it off")
                                .clicked()
                            {
                                remove = Some(n);
                            }
                        }
                    });
                    if let Some(n) = remove {
                        self.attachments.remove(n);
                    }
                    ui.add_space(4.0);
                }
                // The composer: one rounded box – attach, the text (growing to eight lines), send.
                egui::Frame::new()
                    .fill(theme::bg_input())
                    .stroke(egui::Stroke::new(1.0, theme::stroke()))
                    .corner_radius(22)
                    .inner_margin(egui::Margin { left: 8, right: 8, top: 8, bottom: 8 })
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let attach = widgets::icon_button(
                                ui,
                                icon::PAPERCLIP,
                                "Attach files: text, pictures (the model looks at them) or documents (it opens them)",
                            );
                            if attach.clicked()
                                && let Some(paths) = rfd::FileDialog::new().pick_files()
                            {
                                self.attach(paths);
                            }
                            let send_width = 40.0;
                            let response = ui.add(
                                egui::TextEdit::multiline(&mut self.input)
                                    .frame(egui::Frame::NONE)
                                    .font(egui::FontId::proportional(BODY))
                                    .hint_text(if ready { "Message the model" } else { "Open a model first" })
                                    .desired_rows(1)
                                    .clip_text(false)
                                    .desired_width(ui.available_width() - send_width),
                            );
                            let enter = response.has_focus()
                                && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
                            let can = ready && !busy && !self.input.trim().is_empty();
                            let send = ui.add_enabled_ui(can, |ui| {
                                widgets::accent_button(ui, icon::ARROW_UP, theme::accent(), vec2(32.0, 32.0))
                                    .on_hover_text("Send (Enter; Shift+Enter starts a new line)")
                                    .clicked()
                            });
                            if (send.inner || enter) && can {
                                // Enter adds a line break before it is read: take it back off.
                                let text = self.input.trim().to_owned();
                                self.input.clear();
                                let attachments = std::mem::take(&mut self.attachments);
                                if let Some(chat) = self.chats.get_mut(self.current) {
                                    chat.messages.push(ChatMessage {
                                        role: "user".into(),
                                        text,
                                        attachments,
                                        ..Default::default()
                                    });
                                }
                                self.editing = None;
                                self.ask(ui.ctx());
                            }
                        });
                    });
                ui.add_space(4.0);
                ui.vertical_centered(|ui| {
                    ui.label(
                        RichText::new("Small models get things wrong: check what matters.")
                            .size(10.5)
                            .color(theme::text_faint()),
                    );
                });
            });
        });
    }

    /// Files attached to the next message: small text files go in whole; pictures, documents and
    /// big files are named, for the model's tools to read.
    fn attach(&mut self, paths: Vec<PathBuf>) {
        for path in paths {
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let extension = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
            let shown = path.display().to_string().replace('\\', "/");
            let named = |kind: &str| Attachment {
                name: name.clone(),
                text: String::new(),
                path: shown.clone(),
                kind: kind.into(),
            };
            if PICTURES.contains(&extension.as_str()) {
                self.attachments.push(named("picture"));
                continue;
            }
            if DOCUMENTS.contains(&extension.as_str()) {
                self.attachments.push(named("document"));
                continue;
            }
            match std::fs::read(&path) {
                Ok(bytes) if std::str::from_utf8(&bytes[..bytes.len().min(4096)]).is_ok() => {
                    let mut attachment = named("file");
                    if bytes.len() <= ATTACH_INLINE {
                        attachment.text = String::from_utf8_lossy(&bytes).into_owned();
                    }
                    self.attachments.push(attachment);
                }
                Ok(_) => self.error = Some(format!("{} is neither text, a picture nor a document.", path.display())),
                Err(error) => self.error = Some(format!("Could not read {}: {error}", path.display())),
            }
        }
    }

    /// The chat's turns as the model reads them, from `from` on.
    fn turns(chat: &ChatSession, from: usize) -> Vec<(String, String)> {
        let mut turns = Vec::new();
        for message in chat.messages.iter().skip(from) {
            if message.role == "user" {
                let mut text = message.text.clone();
                for file in &message.attachments {
                    text.push_str(&file.in_message());
                }
                turns.push(("user".to_owned(), text));
            } else if message.error.is_none() {
                turns.push((
                    "assistant".to_owned(),
                    if message.raw.is_empty() { message.text.clone() } else { message.raw.clone() },
                ));
            }
        }
        turns
    }

    /// Answers the last message of the current chat, in the background.
    fn ask(&mut self, ctx: &egui::Context) {
        let Some((way, model)) = self.way() else { return };
        let Some(chat) = self.chats.get(self.current).cloned() else { return };
        let file = Arc::clone(&model.file);
        let options = self.options();
        let engine = Arc::clone(&self.engine);
        let key = Self::key(model, way);
        let slot: Pending = Arc::new(Mutex::new(None));
        let (done, ctx) = (Arc::clone(&slot), ctx.clone());
        self.pending = Some((chat.id.clone(), slot));
        std::thread::spawn(move || {
            let reply = answer(&file, way, key, &engine, &chat, &options);
            *done.lock().unwrap_or_else(|p| p.into_inner()) = Some(reply);
            ctx.request_repaint();
        });
    }

    /// Takes in what came back: an answer, a loaded model.
    fn poll(&mut self, ctx: &egui::Context) {
        if let Some(loading) = &self.loading {
            let done = loading.lock().unwrap_or_else(|p| p.into_inner()).take();
            match done {
                Some(result) => {
                    self.error = result.err();
                    self.loading = None;
                }
                None => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
            }
        }
        let Some((id, pending)) = &self.pending else { return };
        let Some(reply) = pending.lock().unwrap_or_else(|p| p.into_inner()).take() else {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
            return;
        };
        let id = id.clone();
        self.pending = None;
        let Some(index) = self.chats.iter().position(|c| c.id == id) else { return };
        let chat = &mut self.chats[index];
        if let Some((summary, count)) = reply.summary {
            chat.summary = summary;
            chat.compacted = count;
        }
        let native_tools = chat.settings.native_tools;
        let message = match reply.answer {
            Ok(agent) => {
                let mut tools: Vec<ToolStep> = agent
                    .steps
                    .iter()
                    .map(|s| ToolStep {
                        thinking: s.thinking.clone(),
                        name: s.call.name.clone(),
                        args: magi_core::magic::spaced_json(&s.call.args),
                        result: magi_core::magic::spaced_json(&s.result),
                        error: s.result.get("error").is_some(),
                        note: String::new(),
                    })
                    .collect();
                // Calls it made last that did not run: shown, with why.
                let note = if native_tools {
                    "Not run: it reached the most tool calls a message may make (chat settings)."
                } else {
                    "Not run: native tools are off in this chat's settings."
                };
                tools.extend(agent.answer.tools.iter().map(|call| ToolStep {
                    name: call.name.clone(),
                    args: magi_core::magic::spaced_json(&call.args),
                    note: note.into(),
                    ..Default::default()
                }));
                ChatMessage {
                    role: "assistant".into(),
                    text: agent.answer.text.clone(),
                    raw: agent.answer.raw.clone(),
                    thinking: agent.answer.thinking.clone(),
                    tools,
                    ..Default::default()
                }
            }
            Err(error) => ChatMessage { role: "assistant".into(), error: Some(error), ..Default::default() },
        };
        chat.messages.push(message);
        if let Some(title) = reply.title.filter(|_| !chat.titled) {
            chat.title = title;
            chat.titled = true;
        }
        chat.changed = now();
        self.save(index);
    }
}

/// A collapsible span of the model's reasoning, opened to begin with when `open`. Turning `open` on
/// or off opens or folds every one (each can still be folded or opened by hand).
fn thought(ui: &mut Ui, id: impl std::hash::Hash + std::fmt::Debug, label: &str, text: &str, open: bool) {
    egui::CollapsingHeader::new(
        RichText::new(format!("{}  {label}", icon::BRAIN)).size(12.5).color(theme::text_faint()),
    )
    .id_salt((id, open))
    .default_open(open)
    .show(ui, |ui| {
        ui.add(
            egui::Label::new(RichText::new(text.trim()).size(13.0).color(theme::text_dim())).wrap().selectable(true),
        );
    });
}

/// A tool the model called, as a row that opens to its arguments and what came back.
fn tool_step(ui: &mut Ui, id: impl std::hash::Hash + std::fmt::Debug, step: &ToolStep) {
    use egui::text::{LayoutJob, TextFormat};
    let (glyph, color, state) = if step.result.is_empty() {
        (icon::PROHIBIT, theme::text_faint(), "not run")
    } else if step.error {
        (icon::WARNING_CIRCLE, theme::warning(), "error")
    } else {
        (icon::CHECK_CIRCLE, theme::success(), "")
    };
    let format = |size: f32, color: egui::Color32| TextFormat {
        font_id: egui::FontId::proportional(size),
        color,
        ..Default::default()
    };
    let mut header = LayoutJob::default();
    header.append(&format!("{}  ", icon::WRENCH), 0.0, format(12.5, theme::text_faint()));
    header.append(&step.name, 0.0, format(12.5, theme::text()));
    header.append(&format!("  {}", short(&step.args, 56)), 0.0, format(12.0, theme::text_faint()));
    header.append(&format!("   {glyph} {state}"), 0.0, format(12.0, color));
    egui::Frame::new()
        .fill(theme::bg_panel())
        .stroke(egui::Stroke::new(1.0, theme::stroke()))
        .corner_radius(10)
        .inner_margin(egui::Margin::symmetric(10, 4))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            egui::CollapsingHeader::new(header).id_salt(id).show(ui, |ui| {
                ui.label(RichText::new("Arguments").size(11.0).color(theme::text_faint()));
                code_block(ui, &pretty(&step.args));
                if step.result.is_empty() {
                    widgets::notice(ui, icon::PROHIBIT, step.note.clone(), theme::text_faint());
                } else {
                    ui.label(RichText::new("Result").size(11.0).color(theme::text_faint()));
                    code_block(ui, &pretty(&step.result));
                }
            });
        });
    ui.add_space(4.0);
}

/// JSON laid out to read, cut when it is very long.
fn pretty(json: &str) -> String {
    let text = serde_json::from_str::<serde_json::Value>(json)
        .ok()
        .and_then(|v| serde_json::to_string_pretty(&v).ok())
        .unwrap_or_else(|| json.to_owned());
    short(&text, 4000)
}

/// Text cut to about `limit` characters.
fn short(text: &str, limit: usize) -> String {
    let text = text.trim();
    match text.char_indices().nth(limit) {
        Some((at, _)) => format!("{}…", &text[..at]),
        None => text.to_owned(),
    }
}

fn code_block(ui: &mut Ui, text: &str) {
    egui::Frame::new().fill(theme::bg_input()).corner_radius(8).inner_margin(egui::Margin::symmetric(10, 8)).show(
        ui,
        |ui| {
            ui.set_width(ui.available_width());
            ui.add(
                egui::Label::new(RichText::new(text).monospace().size(12.5).color(theme::text()))
                    .wrap()
                    .selectable(true),
            );
        },
    );
}

/// An answer's text, with the Markdown chat apps show: code blocks, `code`, **bold**, headings and lists.
fn answer_text(ui: &mut Ui, text: &str) {
    use egui::text::{LayoutJob, TextFormat};
    let body = theme::mix(theme::text(), theme::text_dim(), 0.15);
    for (k, part) in text.split("```").enumerate() {
        if k % 2 == 1 {
            // A code block; its first line may name the language.
            let code = match part.split_once('\n') {
                Some((first, rest)) if !first.trim().contains(' ') => rest,
                _ => part,
            };
            code_block(ui, code.trim_end());
            ui.add_space(4.0);
            continue;
        }
        for paragraph in part.split("\n\n").map(str::trim).filter(|p| !p.is_empty()) {
            let mut job = LayoutJob::default();
            for (i, line) in paragraph.lines().enumerate() {
                if i > 0 {
                    job.append("\n", 0.0, TextFormat::default());
                }
                let trimmed = line.trim_start();
                let level = trimmed.chars().take_while(|c| *c == '#').count();
                let (line, size, strong) = if (1..=4).contains(&level) && trimmed[level..].starts_with(' ') {
                    (trimmed[level..].trim(), BODY + 4.0 - level as f32, true)
                } else if let Some(rest) = trimmed.strip_prefix("- ").or_else(|| trimmed.strip_prefix("* ")) {
                    job.append(
                        "•  ",
                        0.0,
                        TextFormat { font_id: egui::FontId::proportional(BODY), color: body, ..Default::default() },
                    );
                    (rest, BODY, false)
                } else {
                    (line, BODY, false)
                };
                // **bold** and `code`, inline.
                let (mut bold, mut code) = (strong, false);
                let mut rest = line;
                while !rest.is_empty() {
                    let next = [rest.find("**").map(|i| (i, 2)), rest.find('`').map(|i| (i, 1))]
                        .into_iter()
                        .flatten()
                        .min_by_key(|(i, _)| *i);
                    let (end, mark) = next.unwrap_or((rest.len(), 0));
                    let format = if code {
                        TextFormat {
                            font_id: egui::FontId::monospace(size - 1.5),
                            color: theme::text(),
                            background: theme::bg_input(),
                            ..Default::default()
                        }
                    } else {
                        TextFormat {
                            font_id: egui::FontId::proportional(size),
                            color: if bold { theme::text() } else { body },
                            ..Default::default()
                        }
                    };
                    job.append(&rest[..end], 0.0, format);
                    match mark {
                        2 if !code => bold = !bold,
                        1 => code = !code,
                        2 => job.append("**", 0.0, TextFormat::default()),
                        _ => {}
                    }
                    rest = &rest[(end + mark).min(rest.len())..];
                }
            }
            ui.add(egui::Label::new(job).wrap().selectable(true));
            ui.add_space(6.0);
        }
    }
}

/// A model file to chat with: `.szi` or `.magic`.
fn open_model(path: &Path) -> Result<ChatModel, String> {
    match magi_core::szi::open(path).map_err(|e| e.to_string()) {
        Ok(file) if file.model.kind == ModelKind::Text => {
            let name = if file.model.name.is_empty() {
                path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
            } else {
                file.model.name.clone()
            };
            Ok(ChatModel { name, file: Arc::new(chat_copy(&file)), path: Some(path.to_path_buf()) })
        }
        Ok(_) => Err("That is a vision model; chats need a text model.".into()),
        Err(error) => Err(format!("Could not open the model: {error}")),
    }
}

/// What chatting needs of a model: its description, its weights (or its ONNX model when the
/// native engine cannot run it) and its native tools – not its datasets.
fn chat_copy(file: &MagicFile) -> MagicFile {
    let native =
        !file.weights.is_empty() && file.model.architecture.as_ref().is_some_and(|a| a.native_problem().is_none());
    MagicFile {
        model: file.model.clone(),
        weights: file.weights.clone(),
        onnx: if native { Vec::new() } else { file.onnx.clone() },
        data: file.data.iter().filter(|(p, _)| p.starts_with("nt/")).map(|(p, b)| (p.clone(), b.clone())).collect(),
    }
}

/// A random-enough number for a chat's name on disk.
fn rand_bits() -> u32 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.subsec_nanos()) & 0xffff
}

/// Answers a chat's last message (in the background): written short first if
/// it no longer fits, and named after its first answer.
fn answer(
    file: &MagicFile,
    way: Way,
    key: u64,
    engine: &Engine,
    chat: &ChatSession,
    options: &magi_ml::AskOptions,
) -> Reply {
    let spec: TextSpec = file.model.text.clone().unwrap_or_default();
    let settings = &chat.settings;
    let architecture = file.model.architecture.clone().unwrap_or_default();
    let memory = file.model.memory.clone().unwrap_or_default();
    let Way::BuiltIn { native } = way;
    let generate = |prompt: &str, options: &magi_ml::AskOptions| -> Result<String, magi_ml::MlError> {
        let mut slot = engine.lock().unwrap_or_else(|p| p.into_inner());
        if slot.as_ref().is_none_or(|(k, _)| *k != key) {
            *slot = None;
            let bytes = if native { &file.weights } else { &file.onnx };
            *slot = Some((key, load_engine(native, &architecture, &memory, &spec, bytes)?));
        }
        let (_, model) = slot.as_mut().expect("loaded above");
        magi_ml::complete(model, &spec, prompt, options)
    };

    // The system text: the chat's (or the model's), and the earlier part written short.
    let mut summary = (chat.summary.clone(), chat.compacted);
    let system_of = |summary: &str| {
        let base = if settings.system.trim().is_empty() { spec.system.clone() } else { settings.system.clone() };
        if summary.is_empty() { base } else { format!("{base}\n\nEarlier in this conversation: {summary}") }
    };
    let window = if options.context > 0 { options.context } else { spec.context as usize };
    let room =
        window.saturating_sub(if options.max_tokens > 0 { options.max_tokens } else { spec.max_tokens as usize });
    let compacts = settings.compact && file.model.trained.as_ref().is_some_and(|t| t.can_compact());
    let mut changed_summary = false;
    for _ in 0..3 {
        let prompt = spec.chat_prompt(
            &system_of(&summary.0),
            &AgentChat::turns(chat, summary.1),
            options.thinking,
            &options.effort,
        );
        let unfolded = chat.messages.len().saturating_sub(summary.1);
        if !compacts || spec.encode(&prompt).len() <= room || unfolded < 3 {
            break;
        }
        // The older half (keeping the last request) is written short, with what was written short before.
        let fold = (unfolded / 2).max(1).min(unfolded - 1);
        let mut context = String::new();
        if !summary.0.is_empty() {
            context.push_str(&format!("Earlier: {}\n", summary.0));
        }
        for message in &chat.messages[summary.1..summary.1 + fold] {
            let who = if message.role == "user" { "User" } else { "Magi" };
            context.push_str(&format!("{who}: {}\n", message.text.trim()));
        }
        let ask = format!("{}{}", spec.prompt_with(context.trim_end(), None, ""), spec.compact_tags.0);
        let quiet = magi_ml::AskOptions { temperature: 0.0, ..options.clone() };
        match generate(&ask, &quiet) {
            Ok(raw) => {
                let short = raw.split(spec.compact_tags.1.as_str()).next().unwrap_or_default().trim().to_owned();
                if short.is_empty() {
                    break;
                }
                summary = (short, summary.1 + fold);
                changed_summary = true;
            }
            Err(_) => break,
        }
    }

    let prompt =
        spec.chat_prompt(&system_of(&summary.0), &AgentChat::turns(chat, summary.1), options.thinking, &options.effort);
    let steps = if settings.native_tools { settings.max_steps as usize } else { 0 };
    let tools: std::collections::BTreeMap<String, Vec<u8>> =
        file.data.iter().filter(|(p, _)| p.starts_with("nt/")).map(|(p, b)| (p.clone(), b.clone())).collect();
    let names: Vec<&str> = spec.tools.iter().filter(|t| t.native).map(|t| t.name.as_str()).collect();
    let mut run = |call: &magi_core::magic::ToolCall| {
        magi_ml::tools::run_call(&spec, &tools, call).or_else(|| {
            let error = if spec.tools.iter().any(|t| t.name == call.name) {
                format!("{} runs only in a macro, not in this chat", call.name)
            } else {
                format!("there is no tool named {}; the tools here are: {}", call.name, names.join(", "))
            };
            Some(serde_json::json!({ "error": error }))
        })
    };
    let mut step = |prompt: &str| generate(prompt, options);
    let answer = magi_ml::tools::agent_from(&spec, prompt, options, steps, &mut step, &mut run).map_err(|e| e.0);

    // Named after its first answer, by the model, else by its first words.
    let mut title = None;
    if !chat.titled && chat.messages.iter().filter(|m| m.role == "user").count() == 1 {
        let mut turns = AgentChat::turns(chat, summary.1);
        if let Ok(agent) = &answer {
            turns.push(("assistant".into(), agent.answer.raw.clone()));
        }
        turns.push(("user".into(), TITLE_ASK.into()));
        let ask = spec.chat_prompt(&system_of(&summary.0), &turns, Some(false), "");
        let quiet = magi_ml::AskOptions { temperature: 0.0, max_tokens: 24, ..options.clone() };
        let named = generate(&ask, &quiet)
            .ok()
            .map(|raw| spec.visible(&raw))
            .map(|t| t.lines().next().unwrap_or_default().trim().trim_matches(['"', '\'', '.', '*']).to_owned())
            .filter(|t| !t.is_empty() && t.split_whitespace().count() <= 8 && t.len() <= 60);
        title = named.or_else(|| {
            let first = chat.messages.iter().find(|m| m.role == "user").map(|m| m.text.clone()).unwrap_or_default();
            let words: Vec<&str> = first.split_whitespace().take(6).collect();
            (!words.is_empty()).then(|| words.join(" "))
        });
    }
    Reply { answer, summary: changed_summary.then_some(summary), title }
}
