//! The Magic tab: a canvas for making, training, setting up and trying AI
//! models – text models that answer and call tools, and vision models that
//! watch the screen – and handing them to the Macro tab.
//!
//! The canvas works like the Macro tab's. A model travels along the `model`
//! wire: where it starts (from zero, from a base model, or as it is), its
//! architecture blocks, its training stages fed by datasets, and its package.
//! The model's setup – its name, screen area, answers, tools – is shown by
//! inspector blocks, which open their settings in the panel on the right.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use egui::{DragValue, RichText, Ui, vec2};
use egui_phosphor::regular as icon;
use magi_core::geom::{Displays, Rect};
use magi_core::graph::NodeId;
use magi_core::magic::{
    CaptureArea, ClassSpec, ColorOrder, IMAGENET_MEAN, IMAGENET_STD, Layout, MagicModel, ModelKind, ModelSetting,
    SettingKind, SettingTarget, TextSpec, ToolSpec,
};
use magi_core::nodes::Registry;
use magi_core::nodes::ai::AGENT_TOOLS;
use magi_core::nodes::studio::{self, RewardRule, RlParams, TextDataParams, kinds};

use super::dataset_editor::{self, DatasetEditor, DatasetEvent};
use super::graph::{GraphEnv, GraphEvent, GraphView, node_icon};
use super::inspector::{self, InspectorEnv};
use super::widgets::{self, card};
use crate::magic::train::{Level, TrainJob, stage_name};
use crate::magic::{MagicDoc, Tester};
use crate::theme;

pub enum MagicEvent {
    /// Close the model (to start another).
    New,
    /// A new model with a canvas ready to train it.
    NewModel {
        text: bool,
    },
    Open,
    /// Import a project folder, or a file (`.zip` or `.onnx`).
    Import {
        folder: bool,
    },
    Save,
    SaveAs,
    UseInMacro,
    ReplaceModel,
    /// Pick the fixed screen area the model looks at.
    PickArea,
    /// Add pictures of an answer – or, `negative`, of what it is not.
    AddPictures {
        class: usize,
        negative: bool,
    },
    /// Add text files to a Text Dataset block.
    AddDatasetFiles {
        node: NodeId,
        negative: bool,
    },
    /// Run the model on a picture file.
    TryPicture,
    StartTest,
    StopTest,
    Train,
    StopTraining,
    Toast(String),
}

pub struct MagicEnv<'a> {
    pub displays: &'a Displays,
    /// Where models run, as a sentence.
    pub engines: String,
    pub dry_run: bool,
}

/// The Magic tab's blocks.
pub fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(studio::registry)
}

type Pending = Arc<Mutex<Option<Result<magi_ml::Answer, String>>>>;

/// Trying a text model.
#[derive(Default)]
struct Chat {
    request: String,
    pending: Option<(String, Pending)>,
    /// Requests and what the model said, newest last.
    answers: Vec<(String, Result<magi_ml::Answer, String>)>,
}

#[derive(Default)]
pub struct MagicTab {
    pub doc: Option<MagicDoc>,
    pub tester: Option<Tester>,
    /// Whether the live test presses the model's key.
    pub press: bool,
    preview: Option<(u64, egui::TextureHandle)>,
    /// The result of "Try a picture": its name and the class probabilities.
    pub picture: Option<(String, Vec<f32>)>,
    pub view: GraphView,
    pub job: Option<TrainJob>,
    chat: Chat,
    dataset: Option<DatasetEditor>,
    thumbnails: HashMap<String, egui::TextureHandle>,
}

/// Normalisations people know by name.
const NORMALISATIONS: [(&str, [f32; 3], [f32; 3]); 3] = [
    ("ImageNet (most models)", IMAGENET_MEAN, IMAGENET_STD),
    ("0 to 1", [0.0; 3], [1.0; 3]),
    ("−1 to 1", [0.5; 3], [0.5; 3]),
];

impl MagicTab {
    /// Forgets everything about the model that was open.
    pub fn reset(&mut self) {
        self.tester = None;
        self.picture = None;
        self.preview = None;
        self.chat = Chat::default();
        self.dataset = None;
        self.thumbnails.clear();
        let palette_size = self.view.palette_size;
        self.view = GraphView::default();
        self.view.palette_size = palette_size;
        if self.job.as_ref().is_some_and(TrainJob::is_running) {
            self.job = None;
        }
    }

    pub fn training(&self) -> bool {
        self.job.as_ref().is_some_and(TrainJob::is_running)
    }

    pub fn show(&mut self, ui: &mut Ui, env: &MagicEnv<'_>) -> Vec<MagicEvent> {
        let mut events = Vec::new();
        self.take_trained();
        egui::Panel::top("magic-toolbar")
            .frame(egui::Frame::new().fill(theme::bg_panel()).inner_margin(egui::Margin::symmetric(10, 6)))
            .show(ui, |ui| self.toolbar(ui, env, &mut events));

        if self.doc.is_none() {
            egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::bg_panel())).show(ui, |ui| {
                empty_state(ui, &mut events);
            });
            return events;
        }

        egui::Panel::bottom("magic-console")
            .resizable(true)
            .default_size(150.0)
            .size_range(90.0..=420.0)
            .frame(egui::Frame::new().fill(theme::bg_panel()).inner_margin(egui::Margin::symmetric(10, 6)))
            .show(ui, |ui| self.console(ui, &mut events));

        egui::Panel::right("magic-block")
            .resizable(true)
            .default_size(360.0)
            .size_range(280.0..=620.0)
            .frame(egui::Frame::new().fill(theme::bg_panel()).inner_margin(egui::Margin::same(12)))
            .show(ui, |ui| {
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    self.block_panel(ui, env, &mut events);
                });
            });

        egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::bg_canvas())).show(ui, |ui| {
            let doc = self.doc.as_mut().expect("checked above");
            let activity = HashMap::new();
            let graph_env = GraphEnv {
                registry: registry(),
                activity: &activity,
                running: &[],
                error_node: None,
                show_grid: true,
                show_activity: false,
                python_missing: false,
            };
            for event in self.view.show(ui, &mut doc.studio, &graph_env) {
                match event {
                    GraphEvent::Inspect(node) => {
                        self.view.selection = [node].into();
                    }
                    GraphEvent::Toast(text) => events.push(MagicEvent::Toast(text)),
                    _ => {}
                }
            }
        });

        let doc = self.doc.as_mut().expect("checked above");
        doc.sync_canvas();

        if let Some(editor) = &mut self.dataset {
            let spec = doc.file.model.text.clone().unwrap_or_default();
            for event in editor.show(ui.ctx(), &spec) {
                match event {
                    DatasetEvent::Save(file, content) => {
                        doc.file.data.insert(file, content.into_bytes());
                        doc.files_changed();
                    }
                }
            }
            if !editor.open {
                self.dataset = None;
            }
        }

        // Changes take effect in a running test at once.
        if let (Some(tester), Some(doc)) = (&self.tester, &self.doc) {
            let mut config = tester.config.lock().unwrap_or_else(|p| p.into_inner());
            if *config != doc.file.model {
                *config = doc.file.model.clone();
            }
        }
        events
    }

    /// Puts what a finished training run made into the model.
    fn take_trained(&mut self) {
        let Some(job) = &self.job else { return };
        let trained = job.state.lock().unwrap_or_else(|p| p.into_inner()).trained.take();
        if let (Some(trained), Some(doc)) = (trained, &mut self.doc) {
            doc.apply_trained(trained);
            self.tester = None;
            self.picture = None;
        }
    }

    fn toolbar(&mut self, ui: &mut Ui, env: &MagicEnv<'_>, events: &mut Vec<MagicEvent>) {
        widgets::button_row(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            ui.menu_button(format!("{}  New", icon::FILE_PLUS), |ui| {
                ui.set_width(230.0);
                if ui.button(format!("{}  Text model (agent)…", icon::ROBOT)).clicked() {
                    events.push(MagicEvent::NewModel { text: true });
                    ui.close();
                }
                if ui.button(format!("{}  Vision model…", icon::EYE)).clicked() {
                    events.push(MagicEvent::NewModel { text: false });
                    ui.close();
                }
                if self.doc.is_some() && ui.button(format!("{}  Close this model", icon::X)).clicked() {
                    events.push(MagicEvent::New);
                    ui.close();
                }
            });
            ui.menu_button(format!("{}  Import", icon::DOWNLOAD_SIMPLE), |ui| {
                ui.set_width(250.0);
                if ui.button(format!("{}  A project folder…", icon::FOLDER_OPEN)).clicked() {
                    events.push(MagicEvent::Import { folder: true });
                    ui.close();
                }
                if ui.button(format!("{}  A .zip or .onnx file…", icon::FILE_ZIP)).clicked() {
                    events.push(MagicEvent::Import { folder: false });
                    ui.close();
                }
                widgets::note(ui, "The model is read, and its settings worked out from the project's files.");
            })
            .response
            .on_hover_text("Import a model made elsewhere");
            if ui.button(format!("{}  Open", icon::FOLDER_OPEN)).on_hover_text("Open a .magic file").clicked() {
                events.push(MagicEvent::Open);
            }
            let has_doc = self.doc.is_some();
            if ui.add_enabled(has_doc, egui::Button::new(format!("{}  Save", icon::FLOPPY_DISK))).clicked() {
                events.push(MagicEvent::Save);
            }
            if ui.add_enabled(has_doc, egui::Button::new(format!("{}  Save as…", icon::FLOPPY_DISK_BACK))).clicked() {
                events.push(MagicEvent::SaveAs);
            }
            if has_doc
                && ui.button(format!("{}  Add block", icon::PLUS)).on_hover_text("Or right-click the canvas").clicked()
            {
                self.view.open_palette_in_view();
            }
            ui.add_space(6.0);
            if ui.available_size_before_wrap().x < 260.0 {
                ui.end_row();
            }
            if let Some(doc) = &self.doc {
                if self.training() {
                    if widgets::accent_button(
                        ui,
                        format!("{}  Stop training", icon::STOP),
                        theme::danger(),
                        vec2(0.0, 28.0),
                    )
                    .clicked()
                    {
                        events.push(MagicEvent::StopTraining);
                    }
                } else {
                    let problem = doc.training_problem();
                    let hint =
                        problem.clone().unwrap_or_else(|| "Run the training stages on the canvas, in order".into());
                    ui.add_enabled_ui(problem.is_none(), |ui| {
                        if widgets::accent_button(
                            ui,
                            format!("{}  Train", icon::PLAY),
                            theme::accent(),
                            vec2(0.0, 28.0),
                        )
                        .on_hover_text(&hint)
                        .on_disabled_hover_text(&hint)
                        .clicked()
                        {
                            events.push(MagicEvent::Train);
                        }
                    });
                }
                let ready = !doc.file.onnx.is_empty();
                ui.add_enabled_ui(ready, |ui| {
                    if ui
                        .add(
                            egui::Button::new(format!("{}  Use in Macro", icon::TREE_STRUCTURE))
                                .min_size(vec2(0.0, 28.0)),
                        )
                        .on_hover_text("Put the model into the open macro, with blocks that run it")
                        .on_disabled_hover_text("Train the model (or import one) first")
                        .clicked()
                    {
                        events.push(MagicEvent::UseInMacro);
                    }
                });
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    RichText::new(format!("{}  {}", icon::CPU, env.engines)).size(11.5).color(theme::text_faint()),
                )
                .on_hover_text("Where models run. Change it under Settings › AI.");
                if let Some(doc) = &self.doc {
                    let name = if doc.is_dirty() { format!("{} •", doc.title()) } else { doc.title() };
                    ui.add(egui::Label::new(RichText::new(name).color(theme::text_dim())).truncate());
                }
            });
        });
    }

    // ---- the console --------------------------------------------------------

    fn console(&mut self, ui: &mut Ui, events: &mut Vec<MagicEvent>) {
        let Some(job) = &self.job else {
            ui.label(RichText::new("CONSOLE").size(10.5).color(theme::text_faint()));
            widgets::note(
                ui,
                "Training reports here: each stage, its progress and loss. Build the pipeline on the canvas – a \
                 start, architecture blocks, training stages with datasets – then press Train.",
            );
            return;
        };
        let state = job.state.lock().unwrap_or_else(|p| p.into_inner());
        let running = state.outcome.is_none();
        let mut clear = false;
        ui.horizontal(|ui| {
            ui.label(RichText::new("CONSOLE").size(10.5).color(theme::text_faint()));
            if running {
                ui.spinner();
                let stage = stage_name(&state.stage);
                let text = match state.progress {
                    Some((step, steps)) => format!("{stage} · step {step} of {steps}"),
                    None if stage.is_empty() => "Starting…".to_owned(),
                    None => stage.to_owned(),
                };
                ui.label(RichText::new(text).color(theme::text_dim()));
                if let Some((step, steps)) = state.progress
                    && steps > 0
                {
                    ui.add(
                        egui::ProgressBar::new(step as f32 / steps as f32).desired_width(160.0).desired_height(10.0),
                    );
                }
            }
            let seconds = job.started.elapsed().as_secs();
            if running {
                ui.label(RichText::new(format!("{}:{:02}", seconds / 60, seconds % 60)).color(theme::text_faint()));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if !running && widgets::icon_button(ui, icon::BROOM, "Clear the console").clicked() {
                    clear = true;
                }
                if let Some(last) = state.losses.last() {
                    ui.label(RichText::new(format!("loss {:.4}", last[1])).monospace().color(theme::text_dim()));
                }
                if let Some(last) = state.rewards.last() {
                    ui.label(RichText::new(format!("reward {:.2}", last[1])).monospace().color(theme::text_dim()));
                }
            });
        });
        ui.horizontal_top(|ui| {
            let chart_width = (ui.available_width() * 0.35).clamp(160.0, 340.0);
            let log_width = ui.available_width() - chart_width - 10.0;
            ui.allocate_ui(vec2(log_width, ui.available_height()), |ui| {
                egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
                    for (level, text) in &state.lines {
                        let (glyph, color) = match level {
                            Level::Info => (icon::DOT_OUTLINE, theme::text_dim()),
                            Level::Stage => (icon::CARET_RIGHT, theme::text()),
                            Level::Warning => (icon::WARNING, theme::warning()),
                            Level::Error => (icon::X_CIRCLE, theme::danger()),
                            Level::Done => (icon::CHECK_CIRCLE, theme::success()),
                        };
                        ui.add(
                            egui::Label::new(RichText::new(format!("{glyph}  {text}")).size(12.0).color(color)).wrap(),
                        );
                    }
                });
            });
            loss_chart(ui, &state.losses, vec2(chart_width, ui.available_height().max(60.0)));
        });
        drop(state);
        if clear {
            self.job = None;
        }
        let _ = events;
    }

    // ---- the panel of the selected block ----------------------------------

    fn block_panel(&mut self, ui: &mut Ui, env: &MagicEnv<'_>, events: &mut Vec<MagicEvent>) {
        let selected = (self.view.selection.len() == 1).then(|| *self.view.selection.first().expect("one"));
        let doc = self.doc.as_mut().expect("checked by the caller");
        let Some(node_id) = selected.filter(|id| doc.studio.graph.contains(*id)) else {
            overview(ui, doc);
            return;
        };
        let kind = doc.studio.graph.node(node_id).map(|n| n.kind.clone()).unwrap_or_default();
        let custom = kinds::INSPECTORS.contains(&kind.as_str())
            || [kinds::TEXT_DATA, kinds::IMAGE_DATA, kinds::COMPILE, kinds::USE].contains(&kind.as_str());
        if custom {
            block_header(ui, &kind);
        }
        match kind.as_str() {
            kinds::INFO => info_card(ui, doc, events),
            kinds::INPUT => capture_card(ui, &mut doc.file.model, env, events),
            kinds::ANSWERS => classes_card(ui, &mut doc.file.model),
            kinds::ACTION => action_card(ui, &mut doc.file.model),
            kinds::SETTINGS => settings_card(ui, &mut doc.file.model),
            kinds::PREP => input_card(ui, &mut doc.file.model),
            kinds::TEXT => text_card(ui, &mut doc.file.model),
            kinds::TRY => {
                if doc.file.model.kind == ModelKind::Text {
                    self.chat_panel(ui);
                } else {
                    self.test_panel(ui, env, events);
                }
            }
            kinds::TEXT_DATA => {
                if let Some(open) = text_data_card(ui, doc, node_id, events) {
                    let bytes = doc.file.data.get(&open).cloned().unwrap_or_default();
                    self.dataset = Some(DatasetEditor::new(&open, &bytes));
                }
            }
            kinds::IMAGE_DATA => pictures_card(ui, doc, &mut self.thumbnails, events),
            kinds::COMPILE => compile_card(ui, doc, events),
            kinds::USE => use_card(ui, doc, events),
            _ => {
                let live = serde_json::Map::new();
                let mut inspector_env = InspectorEnv {
                    registry: registry(),
                    images: &mut self.view.images,
                    capturing: None,
                    recording: false,
                    displays: env.displays,
                    live: &live,
                };
                // Their settings are plain values: nothing needs the rest of the app.
                let _ = inspector::show(ui, &mut doc.studio, node_id, &mut inspector_env);
                if kind == kinds::RL
                    && let Some(node) = doc.studio.graph.node_mut(node_id)
                {
                    ui.separator();
                    let mut params: RlParams = node.params_or_default();
                    if rules_card(ui, &mut params.rules) {
                        node.set_param("rules", &params.rules);
                    }
                }
                if kind == kinds::BASE {
                    widgets::note(
                        ui,
                        "Fine-tuning needs the weights of a model trained in MacroMagi. Models imported from \
                         elsewhere (ONNX only) can be set up and used, but not trained further.",
                    );
                }
            }
        }
    }

    fn chat_panel(&mut self, ui: &mut Ui) {
        let doc = self.doc.as_ref().expect("checked by the caller");
        if let Some((request, pending)) = &self.chat.pending {
            let done = pending.lock().unwrap_or_else(|p| p.into_inner()).take();
            if let Some(result) = done {
                self.chat.answers.push((request.clone(), result));
                self.chat.pending = None;
            } else {
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
            }
        }
        if doc.file.onnx.is_empty() {
            widgets::notice(ui, icon::INFO, "Train the model first: there is nothing to ask yet.", theme::text_faint());
        }
        ui.add(
            egui::TextEdit::multiline(&mut self.chat.request)
                .hint_text("Ask it something, or give it a task")
                .desired_rows(3)
                .desired_width(f32::INFINITY),
        );
        let busy = self.chat.pending.is_some();
        widgets::button_row(ui, |ui| {
            let can = !busy && !doc.file.onnx.is_empty() && !self.chat.request.trim().is_empty();
            ui.add_enabled_ui(can, |ui| {
                if widgets::accent_button(
                    ui,
                    format!("{}  Ask", icon::PAPER_PLANE_RIGHT),
                    theme::accent(),
                    vec2(0.0, 28.0),
                )
                .clicked()
                {
                    let spec = doc.file.model.text.clone().unwrap_or_default();
                    let onnx = doc.file.onnx.clone();
                    let request = self.chat.request.trim().to_owned();
                    let pending: Pending = Arc::new(Mutex::new(None));
                    let (slot, ctx, asked) = (Arc::clone(&pending), ui.ctx().clone(), request.clone());
                    std::thread::spawn(move || {
                        let seed = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map_or(1, |d| d.as_nanos() as u64);
                        let result = magi_ml::TextModel::load(&onnx, spec.context as usize)
                            .and_then(|mut model| magi_ml::ask(&mut model, &spec, &asked, 0, -1.0, seed))
                            .map_err(|e| e.0);
                        *slot.lock().unwrap_or_else(|p| p.into_inner()) = Some(result);
                        ctx.request_repaint();
                    });
                    self.chat.pending = Some((request, pending));
                }
            });
            if busy {
                ui.spinner();
                ui.label(RichText::new("Thinking…").color(theme::text_dim()));
            }
            if !self.chat.answers.is_empty() && ui.small_button("Clear").clicked() {
                self.chat.answers.clear();
            }
        });
        ui.add_space(6.0);
        for (index, (request, answer)) in self.chat.answers.iter().enumerate().rev() {
            ui.push_id(index, |ui| {
                egui::Frame::new().fill(theme::bg_raised()).corner_radius(theme::RADIUS).inner_margin(8).show(
                    ui,
                    |ui| {
                        ui.set_width(ui.available_width());
                        ui.label(RichText::new(format!("{}  {request}", icon::USER)).color(theme::text_dim()));
                        match answer {
                            Err(error) => {
                                ui.label(RichText::new(error).color(theme::danger()));
                            }
                            Ok(answer) => {
                                let text =
                                    if answer.text.is_empty() { "(nothing to say)" } else { answer.text.as_str() };
                                ui.add(
                                    egui::Label::new(RichText::new(format!("{}  {text}", icon::ROBOT)).strong()).wrap(),
                                );
                                for call in &answer.tools {
                                    ui.label(
                                        RichText::new(format!("{}  {}  {}", icon::WRENCH, call.name, call.args))
                                            .monospace()
                                            .size(11.5)
                                            .color(theme::accent()),
                                    );
                                }
                                if !answer.thinking.is_empty() {
                                    egui::CollapsingHeader::new(
                                        RichText::new("Thinking").size(11.0).color(theme::text_faint()),
                                    )
                                    .show(ui, |ui| {
                                        ui.add(
                                            egui::Label::new(
                                                RichText::new(&answer.thinking).size(11.5).color(theme::text_dim()),
                                            )
                                            .wrap(),
                                        );
                                    });
                                }
                                egui::CollapsingHeader::new(
                                    RichText::new("Everything it wrote").size(11.0).color(theme::text_faint()),
                                )
                                .show(ui, |ui| {
                                    ui.add(egui::Label::new(RichText::new(&answer.raw).monospace().size(11.0)).wrap());
                                });
                            }
                        }
                    },
                );
            });
            ui.add_space(4.0);
        }
    }

    fn test_panel(&mut self, ui: &mut Ui, env: &MagicEnv<'_>, events: &mut Vec<MagicEvent>) {
        let doc = self.doc.as_ref().expect("checked by the caller");
        let running = self.tester.is_some();
        widgets::button_row(ui, |ui| {
            let (text, color) = if running {
                (format!("{}  Stop", icon::STOP), theme::danger())
            } else {
                (format!("{}  Watch the screen", icon::PLAY), theme::accent())
            };
            ui.add_enabled_ui(!doc.file.onnx.is_empty(), |ui| {
                if widgets::accent_button(ui, text, color, vec2(0.0, 28.0)).clicked() {
                    events.push(if running { MagicEvent::StopTest } else { MagicEvent::StartTest });
                }
            });
            if ui
                .button(format!("{}  Picture…", icon::IMAGE))
                .on_hover_text("Run the model on a picture file")
                .clicked()
            {
                events.push(MagicEvent::TryPicture);
            }
        });
        ui.add_enabled_ui(!running, |ui| {
            widgets::toggle_row(
                ui,
                "Press the key when it acts",
                "Off: the test only shows what the model would do. On: it presses the key, as a macro would.",
                &mut self.press,
            );
        });
        if self.press && env.dry_run {
            widgets::notice(ui, icon::INFO, "Dry run is on (Options): nothing is pressed.", theme::text_faint());
        }

        let mut shown_probabilities: Option<Vec<f32>> = None;
        if let Some(tester) = &self.tester {
            let state = tester.state.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(error) = &state.error {
                widgets::notice(ui, icon::WARNING, error.clone(), theme::danger());
            }
            ui.label(
                RichText::new(format!(
                    "{} · {:.0} looks per second · acted {} times",
                    state.engine, state.looks_per_second, state.acted
                ))
                .size(11.5)
                .color(theme::text_dim()),
            );
            if let Some((rgba, w, h)) = &state.picture
                && self.preview.as_ref().is_none_or(|(frame, _)| *frame != state.frame)
            {
                let image = egui::ColorImage::from_rgba_unmultiplied([*w as usize, *h as usize], rgba);
                let texture = ui.ctx().load_texture("magic-preview", image, egui::TextureOptions::LINEAR);
                self.preview = Some((state.frame, texture));
            }
            if let Some((_, texture)) = &self.preview {
                let side = ui.available_width().min(240.0);
                ui.vertical_centered(|ui| {
                    ui.image((
                        texture.id(),
                        vec2(side, side * texture.size()[1] as f32 / texture.size()[0].max(1) as f32),
                    ));
                });
                ui.label(RichText::new("What the model sees").size(10.5).color(theme::text_faint()));
            }
            if let Some(decision) = &state.decision {
                let (text, color) = if decision.act {
                    (
                        format!("{}  {} – act ({:.0}%)", icon::LIGHTNING, decision.label, decision.confidence * 100.0),
                        theme::success(),
                    )
                } else {
                    (format!("{} ({:.0}%)", decision.label, decision.confidence * 100.0), theme::text())
                };
                ui.label(RichText::new(text).strong().color(color));
            }
            shown_probabilities = Some(state.probabilities.clone());
        } else if let Some((name, probabilities)) = &self.picture {
            ui.label(RichText::new(format!("{}  {name}", icon::IMAGE)).color(theme::text_dim()));
            let decision = doc.file.model.decide(probabilities);
            ui.label(
                RichText::new(format!(
                    "{} ({:.0}%){}",
                    decision.label,
                    decision.confidence * 100.0,
                    if decision.act { " – act" } else { "" }
                ))
                .strong(),
            );
            shown_probabilities = Some(probabilities.clone());
        } else {
            widgets::note(
                ui,
                "Watch the screen to see what the model sees and answers, live – or try it on a picture file.",
            );
        }

        if let Some(probabilities) = shown_probabilities {
            ui.add_space(4.0);
            let best = probabilities.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i);
            for (class, p) in probabilities.iter().enumerate() {
                let label = doc.file.model.label(class);
                let acts = doc.file.model.classes.get(class).is_some_and(|c| c.act);
                let bar = egui::ProgressBar::new(*p)
                    .desired_height(14.0)
                    .text(RichText::new(format!("{label}  {:.0}%", p * 100.0)).size(11.0))
                    .fill(if Some(class) == best { theme::accent() } else { theme::bg_widget_active() });
                ui.add(bar).on_hover_text(if acts { "Acting on this class is on" } else { "Not acted on" });
            }
        }
    }
}

/// The name, category and description of a block, above its settings.
fn block_header(ui: &mut Ui, kind: &str) {
    let Some(def) = registry().get(kind) else { return };
    let accent = theme::category_color(def.category);
    ui.horizontal(|ui| {
        ui.label(RichText::new(node_icon(&def.id)).size(20.0).color(accent));
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 1.0;
            ui.label(RichText::new(&def.name).strong().size(15.0));
            ui.label(RichText::new(def.category.label()).size(11.0).color(accent));
        });
    });
    ui.add(egui::Label::new(RichText::new(&def.description).size(12.0).color(theme::text_dim())).wrap());
    ui.separator();
}

/// With no block selected: what the canvas makes.
fn overview(ui: &mut Ui, doc: &MagicDoc) {
    let model = &doc.file.model;
    ui.label(RichText::new(doc.title()).strong().size(15.0));
    ui.label(RichText::new(model.kind.label()).size(11.5).color(theme::text_dim()));
    ui.add_space(6.0);
    match studio::pipeline(&doc.studio.graph) {
        Ok(pipeline) => {
            card(ui, "The pipeline", |ui| {
                let start = registry().get(&pipeline.entry.kind).map(|d| d.name.clone()).unwrap_or_default();
                ui.label(format!("{}  {start}", icon::FLAG));
                for block in &pipeline.architecture.blocks {
                    ui.label(
                        RichText::new(format!("{}  {}", icon::CUBE, block.kind.replace('_', " ")))
                            .color(theme::text_dim()),
                    );
                }
                for stage in &pipeline.stages {
                    let name = registry().get(&stage.kind).map(|d| d.name.clone()).unwrap_or_default();
                    let data =
                        if stage.data.is_empty() && stage.kind != kinds::VISION_TRAIN { " – no dataset" } else { "" };
                    ui.label(format!("{}  {name}{data}", icon::CARET_RIGHT));
                }
                if pipeline.compile.is_none() {
                    widgets::note(ui, "Add Compile .magic Package at the end to say what the file includes.");
                }
                if !pipeline.trains() {
                    widgets::note(ui, "Nothing to train: the model is used as it is.");
                }
            });
        }
        Err(problem) => widgets::notice(ui, icon::WARNING, problem, theme::warning()),
    }
    if let Some(trained) = &model.trained {
        ui.add_space(6.0);
        card(ui, "Trained", |ui| trained_facts(ui, trained));
    }
    if !doc.notes.is_empty() {
        ui.add_space(6.0);
        card(ui, "What the import found", |ui| {
            for note in &doc.notes {
                ui.add(
                    egui::Label::new(RichText::new(format!("•  {note}")).size(11.5).color(theme::text_dim())).wrap(),
                );
            }
        });
    }
    ui.add_space(8.0);
    widgets::note(
        ui,
        "Select a block to see and change it. Right-click the canvas (or press Add block) for more blocks: \
         architecture, datasets, training stages.",
    );
}

fn trained_facts(ui: &mut Ui, trained: &magi_core::magic::TrainingRecord) {
    let stages: Vec<&str> = trained.stages.iter().map(|s| stage_name(s)).collect();
    ui.label(RichText::new(stages.join(" → ")).color(theme::text_dim()));
    ui.label(
        RichText::new(format!(
            "{} steps · loss {} · {:.2}M parameters",
            trained.steps,
            if trained.loss.is_finite() { format!("{:.4}", trained.loss) } else { "–".into() },
            trained.parameters as f64 / 1e6
        ))
        .size(11.5)
        .color(theme::text_faint()),
    );
}

/// The loss over the run, as a line.
fn loss_chart(ui: &mut Ui, points: &[[f64; 2]], size: egui::Vec2) {
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, theme::RADIUS, theme::bg_raised());
    if points.len() < 2 {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "Loss",
            egui::FontId::proportional(11.0),
            theme::text_faint(),
        );
        return;
    }
    let (x0, x1) = (points[0][0], points[points.len() - 1][0].max(points[0][0] + 1.0));
    let (lo, hi) = points.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| (lo.min(p[1]), hi.max(p[1])));
    let span = (hi - lo).max(1e-9);
    let inner = rect.shrink(8.0);
    let line: Vec<egui::Pos2> = points
        .iter()
        .map(|p| {
            egui::pos2(
                inner.left() + ((p[0] - x0) / (x1 - x0)) as f32 * inner.width(),
                inner.bottom() - ((p[1] - lo) / span) as f32 * inner.height(),
            )
        })
        .collect();
    painter.add(egui::Shape::line(line, egui::Stroke::new(1.5, theme::accent())));
    let font = egui::FontId::monospace(10.0);
    painter.text(inner.left_top(), egui::Align2::LEFT_TOP, format!("{hi:.3}"), font.clone(), theme::text_faint());
    painter.text(inner.left_bottom(), egui::Align2::LEFT_BOTTOM, format!("{lo:.3}"), font, theme::text_faint());
}

fn empty_state(ui: &mut Ui, events: &mut Vec<MagicEvent>) {
    ui.vertical_centered(|ui| {
        ui.add_space(ui.available_height() * 0.16);
        ui.label(RichText::new(icon::MAGIC_WAND).size(46.0).color(theme::accent()));
        ui.label(RichText::new("Magic").size(22.0).strong());
        ui.add_space(4.0);
        ui.set_max_width(ui.available_width().min(500.0));
        ui.add(
            egui::Label::new(
                RichText::new(
                    "Build, train and try AI models for your macros on a canvas: text models that answer and call \
                     tools, and vision models that watch the screen. Start a new one, open a .magic file, or import \
                     a model made elsewhere.",
                )
                .color(theme::text_dim()),
            )
            .wrap(),
        );
        ui.add_space(12.0);
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
        let row = |ui: &mut Ui, events: &mut Vec<MagicEvent>| {
            let text = format!("{}  New text model", icon::ROBOT);
            if widgets::accent_button(ui, text, theme::accent(), vec2(0.0, 32.0)).clicked() {
                events.push(MagicEvent::NewModel { text: true });
            }
            let vision = egui::Button::new(format!("{}  New vision model", icon::EYE));
            if ui.add(vision.min_size(vec2(0.0, 32.0))).clicked() {
                events.push(MagicEvent::NewModel { text: false });
            }
        };
        let row2 = |ui: &mut Ui, events: &mut Vec<MagicEvent>| {
            let open = egui::Button::new(format!("{}  Open .magic…", icon::FOLDER_OPEN));
            if ui.add(open.min_size(vec2(0.0, 32.0))).clicked() {
                events.push(MagicEvent::Open);
            }
            let folder = egui::Button::new(format!("{}  Import a project folder…", icon::FOLDER_OPEN));
            if ui.add(folder.min_size(vec2(0.0, 32.0))).clicked() {
                events.push(MagicEvent::Import { folder: true });
            }
            let file = egui::Button::new(format!("{}  Import a .zip or .onnx…", icon::FILE_ZIP));
            if ui.add(file.min_size(vec2(0.0, 32.0))).clicked() {
                events.push(MagicEvent::Import { folder: false });
            }
        };
        if ui.available_width() >= 600.0 {
            ui.horizontal(|ui| {
                ui.add_space((ui.available_width() - 330.0).max(0.0) / 2.0);
                row(ui, events);
            });
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.add_space((ui.available_width() - 560.0).max(0.0) / 2.0);
                row2(ui, events);
            });
        } else {
            row(ui, events);
            row2(ui, events);
        }
        ui.add_space(10.0);
        widgets::note(ui, "Training needs PyTorch, which can be installed under Settings › AI.");
    });
}

// ---- the inspector blocks -----------------------------------------------------

fn info_card(ui: &mut Ui, doc: &mut MagicDoc, events: &mut Vec<MagicEvent>) {
    let size_mb = doc.file.onnx.len() as f64 / (1024.0 * 1024.0);
    let classes = doc.shape.as_ref().and_then(|s| s.classes());
    let model = &mut doc.file.model;
    ui.horizontal(|ui| {
        widgets::caption(ui, "Name");
        ui.add(egui::TextEdit::singleline(&mut model.name).desired_width(f32::INFINITY));
    });
    ui.add(
        egui::TextEdit::multiline(&mut model.description)
            .hint_text("What it does")
            .desired_rows(2)
            .desired_width(f32::INFINITY),
    );
    ui.horizontal(|ui| {
        widgets::caption(ui, "Author");
        ui.add(egui::TextEdit::singleline(&mut model.author).desired_width(f32::INFINITY));
    });
    ui.horizontal(|ui| {
        widgets::caption(ui, "Licence");
        ui.add(egui::TextEdit::singleline(&mut model.license).hint_text("Such as MIT").desired_width(f32::INFINITY));
    });
    let facts = if size_mb > 0.0 && model.kind == ModelKind::Vision {
        format!(
            "Vision · reads {}×{} pictures · {} answers · {size_mb:.1} MB",
            model.input.width,
            model.input.height,
            classes.map_or_else(|| "?".to_owned(), |n| n.to_string())
        )
    } else if size_mb > 0.0 {
        format!(
            "Text · {} tokens of context · {size_mb:.1} MB{}",
            model.text.as_ref().map_or(0, |t| t.context),
            if doc.file.weights.is_empty() { "" } else { " · with weights" }
        )
    } else {
        format!("{} · not trained yet", model.kind.label())
    };
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(facts).size(11.5).color(theme::text_dim()));
        if model.kind == ModelKind::Vision
            && ui.small_button(format!("{}  Replace model file…", icon::ARROWS_CLOCKWISE)).clicked()
        {
            events.push(MagicEvent::ReplaceModel);
        }
    });
    if !model.source.is_empty() {
        ui.label(RichText::new(&model.source).size(11.0).color(theme::text_faint()));
    }
    if let Some(trained) = &model.trained {
        ui.add_space(4.0);
        trained_facts(ui, trained);
    }
}

fn capture_card(ui: &mut Ui, model: &mut MagicModel, env: &MagicEnv<'_>, events: &mut Vec<MagicEvent>) {
    let mut centred = matches!(model.capture.area, CaptureArea::Center { .. });
    if widgets::segmented(ui, &mut centred, &[(true, "Middle of the display"), (false, "A fixed area")]) {
        model.capture.area = if centred {
            CaptureArea::Center { height_share: f64::from(model.input.height) / 1080.0 }
        } else {
            let rect = model.capture.rect(env.displays, None).unwrap_or(Rect::new(
                0,
                0,
                model.input.width,
                model.input.height,
            ));
            CaptureArea::Region { rect: env.displays.rect_to_local(rect) }
        };
    }
    match &mut model.capture.area {
        CaptureArea::Center { height_share } => {
            ui.horizontal(|ui| {
                ui.label(RichText::new("A square of").color(theme::text_dim()));
                let mut pixels = *height_share * 1080.0;
                if ui.add(DragValue::new(&mut pixels).range(8.0..=1080.0).max_decimals(0).suffix(" px")).changed() {
                    *height_share = pixels / 1080.0;
                }
                ui.label(RichText::new("on a 1080-pixel-high display").color(theme::text_dim()));
            });
            ui.horizontal(|ui| {
                ui.label(RichText::new("On").color(theme::text_dim()));
                let mut display = Some(model.capture.display);
                if widgets::display_picker(ui, "magic-display", &mut display, env.displays, false) {
                    model.capture.display = display.unwrap_or(0);
                }
            });
            if let Some(rect) = model.capture.rect(env.displays, None) {
                widgets::note(
                    ui,
                    format!(
                        "Here: {}×{} pixels at {}, {}. It grows and shrinks with the display.",
                        rect.w, rect.h, rect.x, rect.y
                    ),
                );
            }
        }
        CaptureArea::Region { rect } => {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                ui.add(DragValue::new(&mut rect.x).prefix("X "));
                ui.add(DragValue::new(&mut rect.y).prefix("Y "));
                ui.add(DragValue::new(&mut rect.w).range(1..=100_000).prefix("W "));
                ui.add(DragValue::new(&mut rect.h).range(1..=100_000).prefix("H "));
                let mut display = rect.display;
                if widgets::display_picker(ui, "magic-area-display", &mut display, env.displays, true) {
                    *rect = widgets::rect_on(*rect, display, env.displays);
                }
                if ui.button(format!("{}  Select area…", icon::SELECTION)).clicked() {
                    events.push(MagicEvent::PickArea);
                }
            });
        }
    }
}

fn classes_card(ui: &mut Ui, model: &mut MagicModel) {
    widgets::note(ui, "What each answer of the model means, and whether to act on it.");
    if model.classes.is_empty() {
        ui.weak("The model's answers are not known yet.");
    }
    egui::Grid::new("magic-classes").num_columns(4).spacing([8.0, 4.0]).show(ui, |ui| {
        ui.label(RichText::new("#").size(11.0).color(theme::text_faint()));
        ui.label(RichText::new("Name").size(11.0).color(theme::text_faint()));
        ui.label(RichText::new("Act").size(11.0).color(theme::text_faint()));
        ui.label(RichText::new("Sure enough at · then wait").size(11.0).color(theme::text_faint()));
        ui.end_row();
        for (index, class) in model.classes.iter_mut().enumerate() {
            ui.label(RichText::new(index.to_string()).color(theme::text_faint()));
            // Sized outright: a grid cell only offers the width its column had last frame.
            ui.add_sized([130.0, 22.0], egui::TextEdit::singleline(&mut class.label));
            ui.checkbox(&mut class.act, "").on_hover_text("Act when the model gives this answer");
            ui.horizontal(|ui| {
                ui.add_enabled_ui(class.act, |ui| {
                    let mut percent = class.threshold * 100.0;
                    if ui.add(DragValue::new(&mut percent).range(0.0..=100.0).max_decimals(0).suffix("%")).changed() {
                        class.threshold = percent / 100.0;
                    }
                    widgets::duration_edit(ui, &mut class.delay_ms);
                });
            });
            ui.end_row();
        }
    });
    widgets::button_row(ui, |ui| {
        if ui.small_button(format!("{}  Add answer", icon::PLUS)).clicked() {
            let n = model.classes.len();
            model.classes.push(ClassSpec { label: format!("Class {n}"), threshold: 0.5, ..ClassSpec::default() });
        }
        if !model.classes.is_empty() && ui.small_button(format!("{}  Remove the last", icon::MINUS)).clicked() {
            model.classes.pop();
        }
    });
}

fn action_card(ui: &mut Ui, model: &mut MagicModel) {
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("Press").color(theme::text_dim()), 120.0);
        if let Some(picked) = widgets::input_picker(ui, "magic-key", model.action.key, true) {
            model.action.key = picked;
        }
    });
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("Hold for").color(theme::text_dim()), 120.0);
        widgets::duration_edit(ui, &mut model.action.hold_ms);
    });
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("Then ignore it for").color(theme::text_dim()), 120.0)
            .on_hover_text("So one event is not acted on twice");
        widgets::duration_edit(ui, &mut model.action.cooldown_ms);
    });
}

fn settings_card(ui: &mut Ui, model: &mut MagicModel) {
    widgets::note(
        ui,
        "Sliders for the values worth adjusting. Each changes one or more of the model's values, and every \
         block that uses the model can set them its own way.",
    );
    let labels: Vec<String> = model.classes.iter().map(|c| c.label.clone()).collect();
    let mut remove = None;
    for (index, setting) in model.settings.iter_mut().enumerate() {
        ui.push_id(("setting", index), |ui| {
            egui::Frame::new().fill(theme::bg_raised()).corner_radius(theme::RADIUS).inner_margin(6).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&setting.label).strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if widgets::icon_button(ui, icon::TRASH, "Remove this setting").clicked() {
                            remove = Some(index);
                        }
                        if widgets::reset_button(ui, setting.value != setting.default) {
                            setting.value = setting.default;
                        }
                    });
                });
                let suffix = match setting.kind {
                    SettingKind::Percent => "%",
                    SettingKind::Milliseconds => " ms",
                };
                let (min, max) = (setting.min.min(setting.max), setting.max.max(setting.min));
                ui.add(egui::Slider::new(&mut setting.value, min..=max).max_decimals(0).suffix(suffix));
                egui::CollapsingHeader::new(RichText::new("Change what it does").size(11.0).color(theme::text_faint()))
                    .id_salt(("edit", index))
                    .show(ui, |ui| edit_setting(ui, setting, &labels));
            });
        });
        ui.add_space(3.0);
    }
    if let Some(index) = remove {
        model.settings.remove(index);
    }
    if ui.button(format!("{}  Add a setting", icon::PLUS)).clicked() {
        let n = model.settings.len() + 1;
        model.settings.push(ModelSetting {
            key: format!("setting_{n}"),
            label: format!("Setting {n}"),
            kind: SettingKind::Percent,
            min: 0.0,
            max: 100.0,
            default: 50.0,
            value: 50.0,
            targets: Vec::new(),
            ..ModelSetting::default()
        });
    }
    // The values follow the sliders.
    *model = model.configured(&serde_json::Map::new());
}

fn edit_setting(ui: &mut Ui, setting: &mut ModelSetting, labels: &[String]) {
    ui.horizontal(|ui| {
        widgets::caption(ui, "Name");
        ui.add(egui::TextEdit::singleline(&mut setting.label).desired_width(f32::INFINITY));
    });
    ui.horizontal(|ui| {
        widgets::caption(ui, "Kind");
        let mut kind = setting.kind;
        if widgets::segmented(
            ui,
            &mut kind,
            &[(SettingKind::Percent, "Confidence %"), (SettingKind::Milliseconds, "Time (ms)")],
        ) {
            setting.kind = kind;
            setting.targets.clear();
        }
    });
    ui.horizontal(|ui| {
        widgets::caption(ui, "From");
        ui.add(DragValue::new(&mut setting.min).max_decimals(0));
        widgets::caption(ui, "to");
        ui.add(DragValue::new(&mut setting.max).max_decimals(0));
        widgets::caption(ui, "default");
        ui.add(DragValue::new(&mut setting.default).max_decimals(0));
    });
    widgets::caption(ui, "Changes");
    let mut toggle = |ui: &mut Ui, target: SettingTarget, text: String| {
        let mut on = setting.targets.contains(&target);
        if ui.checkbox(&mut on, text).changed() {
            if on {
                setting.targets.push(target);
            } else {
                setting.targets.retain(|t| *t != target);
            }
        }
    };
    match setting.kind {
        SettingKind::Percent => {
            for (class, label) in labels.iter().enumerate() {
                toggle(ui, SettingTarget::Threshold { class }, format!("How sure “{label}” must be"));
            }
        }
        SettingKind::Milliseconds => {
            for (class, label) in labels.iter().enumerate() {
                toggle(ui, SettingTarget::Delay { class }, format!("The wait before acting on “{label}”"));
            }
            toggle(ui, SettingTarget::Cooldown, "The time ignored after acting".into());
            toggle(ui, SettingTarget::Hold, "How long the key is held".into());
        }
    }
}

fn input_card(ui: &mut Ui, model: &mut MagicModel) {
    let input = &mut model.input;
    widgets::note(
        ui,
        "How a picture of the screen is turned into what the model reads. Importing and training work this \
         out; change it only if you know the model needs something else.",
    );
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("Size").color(theme::text_dim()), 110.0);
        ui.add(DragValue::new(&mut input.width).range(1..=4096).suffix(" px"));
        ui.label("×");
        ui.add(DragValue::new(&mut input.height).range(1..=4096).suffix(" px"));
    });
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("Layout").color(theme::text_dim()), 110.0);
        widgets::segmented(ui, &mut input.layout, &[(Layout::Nchw, "Channels first"), (Layout::Nhwc, "Channels last")]);
    });
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("Colors").color(theme::text_dim()), 110.0);
        widgets::segmented(ui, &mut input.color, &[(ColorOrder::Rgb, "RGB"), (ColorOrder::Bgr, "BGR")]);
    });
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("Normalisation").color(theme::text_dim()), 110.0);
        let current = NORMALISATIONS.iter().find(|(_, m, s)| *m == input.mean && *s == input.std).map(|(n, ..)| *n);
        widgets::combo("normalisation").selected_text(current.unwrap_or("Custom")).show_ui(ui, |ui| {
            let needle = widgets::combo_search(ui);
            for (name, mean, std) in NORMALISATIONS {
                if widgets::option(ui, &needle, current == Some(name), name) {
                    (input.mean, input.std) = (mean, std);
                }
            }
        });
    });
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("Mean").color(theme::text_dim()), 110.0);
        for value in &mut input.mean {
            ui.add(DragValue::new(value).speed(0.001).range(-10.0..=10.0).max_decimals(4));
        }
    });
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("Spread").color(theme::text_dim()), 110.0);
        for value in &mut input.std {
            ui.add(DragValue::new(value).speed(0.001).range(0.0001..=10.0).max_decimals(4));
        }
    });
    ui.checkbox(&mut input.softmax, "Turn the model's scores into probabilities")
        .on_hover_text("On for most classifiers. Off when the model already gives probabilities.");
}

/// A text model's chat tags, tools, personas and way of writing.
fn text_card(ui: &mut Ui, model: &mut MagicModel) {
    let trained = model.trained.is_some();
    let text = model.text.get_or_insert_with(TextSpec::default);
    widgets::caption(ui, "System prompt");
    ui.add(
        egui::TextEdit::multiline(&mut text.system)
            .hint_text("What the model is told before every request")
            .desired_rows(3)
            .desired_width(f32::INFINITY),
    );
    ui.add_space(4.0);
    card(ui, "How it writes", |ui| {
        ui.horizontal(|ui| {
            widgets::fixed_label(ui, RichText::new("Temperature").color(theme::text_dim()), 110.0);
            ui.add(egui::Slider::new(&mut text.temperature, 0.0..=2.0).max_decimals(2))
                .on_hover_text("0 always writes the likeliest; higher is freer");
        });
        ui.horizontal(|ui| {
            widgets::fixed_label(ui, RichText::new("Choose among").color(theme::text_dim()), 110.0);
            ui.add(DragValue::new(&mut text.top_k).range(0..=1000).suffix(" likeliest"));
        });
        ui.horizontal(|ui| {
            widgets::fixed_label(ui, RichText::new("Longest answer").color(theme::text_dim()), 110.0);
            ui.add(DragValue::new(&mut text.max_tokens).range(1..=8192).suffix(" tokens"));
        });
        ui.label(
            RichText::new(format!("Context: {} tokens (the Model Size block)", text.context))
                .size(11.5)
                .color(theme::text_faint()),
        );
    });
    ui.add_space(4.0);
    card(ui, "Tools it may call", |ui| {
        widgets::note(
            ui,
            format!(
                "Every tool call is written one way: {}{{\"name\": …, \"args\": {{…}}}}{}. The AI Agent block \
                 carries out press, type, click, move, wait and log itself; the macro handles the others.",
                text.tool_tags.0, text.tool_tags.1
            ),
        );
        let mut remove = None;
        for (index, tool) in text.tools.iter_mut().enumerate() {
            ui.push_id(("tool", index), |ui| {
                egui::Frame::new().fill(theme::bg_raised()).corner_radius(theme::RADIUS).inner_margin(6).show(
                    ui,
                    |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.add(egui::TextEdit::singleline(&mut tool.name).hint_text("name").desired_width(120.0));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if widgets::icon_button(ui, icon::TRASH, "Remove this tool").clicked() {
                                    remove = Some(index);
                                }
                            });
                        });
                        ui.add(
                            egui::TextEdit::singleline(&mut tool.description)
                                .hint_text("What it does")
                                .desired_width(f32::INFINITY),
                        );
                        json_edit(ui, &mut tool.parameters, "Arguments, as a JSON Schema");
                    },
                );
            });
            ui.add_space(3.0);
        }
        if let Some(index) = remove {
            text.tools.remove(index);
        }
        widgets::button_row(ui, |ui| {
            if ui.small_button(format!("{}  Add a tool", icon::PLUS)).clicked() {
                text.tools.push(ToolSpec {
                    name: format!("tool_{}", text.tools.len() + 1),
                    parameters: serde_json::json!({ "type": "object", "properties": {} }),
                    ..ToolSpec::default()
                });
            }
            let missing: Vec<_> =
                AGENT_TOOLS.iter().filter(|(name, _)| !text.tools.iter().any(|t| t.name == *name)).collect();
            if !missing.is_empty()
                && ui
                    .small_button(format!("{}  Add MacroMagi's tools", icon::WRENCH))
                    .on_hover_text("press, type, click, move, wait and log")
                    .clicked()
            {
                for (name, about) in missing {
                    text.tools.push(agent_tool(name, about));
                }
            }
        });
    });
    ui.add_space(4.0);
    card(ui, "Personas", |ui| {
        if text.personas.is_empty() {
            widgets::note(ui, "None. The Dual-Persona Router block gives the model two, each with its own tag.");
        }
        for persona in &text.personas {
            ui.label(format!("{}  {} – {}", persona.tag, persona.name, persona.role));
        }
    });
    ui.add_space(4.0);
    egui::CollapsingHeader::new(RichText::new("CHAT TAGS").size(10.5).color(theme::text_faint())).show(ui, |ui| {
        if trained {
            widgets::notice(
                ui,
                icon::WARNING,
                "The model learned these tags. Changing them means training it again.",
                theme::warning(),
            );
        }
        let mut changed = false;
        for (label, value) in [
            ("System", &mut text.system_tag),
            ("Request", &mut text.user_tag),
            ("Answer", &mut text.assistant_tag),
            ("End of a turn", &mut text.end_tag),
            ("Thinking opens", &mut text.think_tags.0),
            ("Thinking closes", &mut text.think_tags.1),
            ("Tool call opens", &mut text.tool_tags.0),
            ("Tool call closes", &mut text.tool_tags.1),
            ("Compact opens", &mut text.compact_tags.0),
            ("Compact closes", &mut text.compact_tags.1),
        ] {
            ui.horizontal(|ui| {
                widgets::fixed_label(ui, RichText::new(label).color(theme::text_dim()), 120.0);
                changed |= ui.add(egui::TextEdit::singleline(value).code_editor().desired_width(160.0)).lost_focus();
            });
        }
        if changed {
            text.ensure_specials();
        }
        ui.label(
            RichText::new(format!("{} tokens: 256 bytes and {} tags", text.vocab_size(), text.specials.len()))
                .size(11.0)
                .color(theme::text_faint()),
        );
    });
}

/// One of the tools the AI Agent block carries out, as a tool of the model.
fn agent_tool(name: &str, about: &str) -> ToolSpec {
    let property = |ty: &str| serde_json::json!({ "type": ty });
    let properties = match name {
        "press" => serde_json::json!({ "key": property("string") }),
        "type" | "log" => serde_json::json!({ "text": property("string") }),
        "click" => {
            serde_json::json!({ "x": property("number"), "y": property("number"), "button": property("string") })
        }
        "move" => serde_json::json!({ "x": property("number"), "y": property("number") }),
        "wait" => serde_json::json!({ "ms": property("number") }),
        _ => serde_json::json!({}),
    };
    ToolSpec {
        name: name.to_owned(),
        description: about.split(" – ").nth(1).unwrap_or(about).to_owned(),
        parameters: serde_json::json!({ "type": "object", "properties": properties }),
    }
}

/// A JSON value edited as text; it changes once the text is valid JSON.
fn json_edit(ui: &mut Ui, value: &mut serde_json::Value, hint: &str) {
    let id = ui.id().with("json");
    let mut text =
        ui.data_mut(|d| d.get_temp::<String>(id)).unwrap_or_else(|| serde_json::to_string(value).unwrap_or_default());
    let edit = ui.add(
        egui::TextEdit::multiline(&mut text).code_editor().hint_text(hint).desired_rows(2).desired_width(f32::INFINITY),
    );
    let parsed = serde_json::from_str::<serde_json::Value>(&text);
    if edit.changed()
        && let Ok(parsed) = &parsed
    {
        *value = parsed.clone();
    }
    if parsed.is_err() && !text.trim().is_empty() {
        ui.label(RichText::new("Not valid JSON yet").size(11.0).color(theme::warning()));
    }
    if edit.has_focus() {
        ui.data_mut(|d| d.insert_temp(id, text));
    } else {
        ui.data_mut(|d| d.remove::<String>(id));
    }
}

/// The reward rules of a Reinforcement Learning block. True when changed.
fn rules_card(ui: &mut Ui, rules: &mut Vec<RewardRule>) -> bool {
    let mut changed = false;
    ui.label(RichText::new("REWARD RULES").size(10.5).color(theme::text_faint()));
    widgets::note(
        ui,
        "Each answer scores the sum of the rules it meets: reward what is right and fast, take points off for \
         loops and broken tool calls.",
    );
    let mut remove = None;
    for (index, rule) in rules.iter_mut().enumerate() {
        ui.push_id(("rule", index), |ui| {
            ui.horizontal_wrapped(|ui| {
                let current = RewardRule::KINDS.iter().find(|(k, _)| *k == rule.kind).map_or("?", |(_, l)| *l);
                widgets::combo("rule-kind").selected_text(current).width(170.0).show_ui(ui, |ui| {
                    let needle = widgets::combo_search(ui);
                    for (kind, label) in RewardRule::KINDS {
                        if widgets::option(ui, &needle, rule.kind == kind, label) {
                            rule.kind = kind.to_owned();
                            changed = true;
                        }
                    }
                });
                if ["contains", "not_contains", "regex"].contains(&rule.kind.as_str()) {
                    changed |= ui.add(egui::TextEdit::singleline(&mut rule.text).desired_width(110.0)).changed();
                }
                if rule.kind == "short" {
                    changed |= ui
                        .add(DragValue::new(&mut rule.tokens).range(1..=4096).prefix("≤ ").suffix(" tokens"))
                        .changed();
                }
                if rule.kind == "loop" {
                    changed |= ui.add(DragValue::new(&mut rule.tokens).range(1..=64).suffix("-token run")).changed();
                    changed |= ui.add(DragValue::new(&mut rule.repeats).range(2..=64).suffix("× repeated")).changed();
                }
                changed |= ui.add(DragValue::new(&mut rule.score).speed(0.1).prefix("score ")).changed();
                if widgets::icon_button(ui, icon::TRASH, "Remove this rule").clicked() {
                    remove = Some(index);
                }
            });
        });
    }
    if let Some(index) = remove {
        rules.remove(index);
        changed = true;
    }
    if ui.small_button(format!("{}  Add a rule", icon::PLUS)).clicked() {
        rules.push(RewardRule::default());
        changed = true;
    }
    changed
}

// ---- datasets -------------------------------------------------------------------

/// A Text Dataset block's files. Returns a file to open in the editor.
fn text_data_card(ui: &mut Ui, doc: &mut MagicDoc, node_id: NodeId, events: &mut Vec<MagicEvent>) -> Option<String> {
    let mut params: TextDataParams = doc.studio.graph.node(node_id)?.params_or_default();
    let spec = doc.file.model.text.clone().unwrap_or_default();
    let mut open = None;
    let mut changed = false;
    for negative in [false, true] {
        let (title, about) = if negative {
            (
                "What not to do",
                "Answers to avoid – wrong tool calls, loops – as {\"prompt\", \"rejected\"} samples, or answers to unlearn.",
            )
        } else {
            (
                "What to do",
                "Requests and answers, conversations, tool traces ({\"prompt\", \"response\"}, {\"messages\"}), or .md and .txt text to pretrain on.",
            )
        };
        card(ui, title, |ui| {
            widgets::note(ui, about);
            let list = if negative { &mut params.negative } else { &mut params.positive };
            let mut remove = None;
            for (index, file) in list.iter().enumerate() {
                let bytes = doc.file.data.get(file);
                let samples =
                    bytes.map(|b| studio::split_samples(file, &String::from_utf8_lossy(b))).unwrap_or_default();
                let jsonl = studio::is_jsonl(file);
                let problems = samples.iter().filter(|s| studio::check_sample(jsonl, s, &spec).is_err()).count();
                ui.horizontal(|ui| {
                    ui.label(format!("{}  {}", icon::FILE_TEXT, file.rsplit('/').next().unwrap_or_default()));
                    let detail = match (bytes.is_some(), problems) {
                        (false, _) => "missing".to_owned(),
                        (true, 0) => format!("{} samples", samples.len()),
                        (true, n) => format!("{} samples · {n} with problems", samples.len()),
                    };
                    ui.label(RichText::new(detail).size(11.0).color(if problems > 0 {
                        theme::warning()
                    } else {
                        theme::text_faint()
                    }));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if widgets::icon_button(ui, icon::TRASH, "Take it out of the dataset").clicked() {
                            remove = Some(index);
                        }
                        if bytes.is_some()
                            && widgets::icon_button(ui, icon::PENCIL_SIMPLE, "Edit the samples").clicked()
                        {
                            open = Some(file.clone());
                        }
                    });
                });
            }
            if let Some(index) = remove {
                list.remove(index);
                changed = true;
            }
            widgets::button_row(ui, |ui| {
                if ui.small_button(format!("{}  Add files…", icon::PLUS)).clicked() {
                    events.push(MagicEvent::AddDatasetFiles { node: node_id, negative });
                }
                let stem = if negative { "not-to-do" } else { "to-do" };
                let new = |data: &std::collections::BTreeMap<String, Vec<u8>>, ext: &str| {
                    (1..)
                        .map(|n| format!("data/text/{stem}-{n}.{ext}"))
                        .find(|f| !data.contains_key(f))
                        .expect("a free name")
                };
                if ui.small_button(format!("{}  New .jsonl", icon::FILE_PLUS)).clicked() {
                    let file = new(&doc.file.data, "jsonl");
                    doc.file.data.insert(file.clone(), dataset_editor::template(&file, negative).into_bytes());
                    list.push(file.clone());
                    open = Some(file);
                    changed = true;
                }
                if !negative && ui.small_button(format!("{}  New .md", icon::FILE_PLUS)).clicked() {
                    let file = new(&doc.file.data, "md");
                    doc.file.data.insert(file.clone(), dataset_editor::template(&file, false).into_bytes());
                    list.push(file.clone());
                    open = Some(file);
                    changed = true;
                }
            });
        });
        ui.add_space(6.0);
    }
    if changed && let Some(node) = doc.studio.graph.node_mut(node_id) {
        node.set_params(&params);
    }
    open
}

/// A small picture of a dataset picture, made once.
fn thumbnail(
    ui: &Ui,
    cache: &mut HashMap<String, egui::TextureHandle>,
    file: &str,
    bytes: &[u8],
) -> Option<egui::TextureHandle> {
    if let Some(texture) = cache.get(file) {
        return Some(texture.clone());
    }
    let bitmap = magi_vision::decode(bytes).ok()?;
    let scale = 72.0 / bitmap.width.max(bitmap.height).max(1) as f32;
    let (w, h) =
        (((bitmap.width as f32 * scale).round() as u32).max(1), ((bitmap.height as f32 * scale).round() as u32).max(1));
    let rgb = magi_ml::resize_rgb(&bitmap.rgba, bitmap.width, bitmap.height, w, h);
    let rgba: Vec<u8> = rgb
        .chunks(3)
        .flat_map(|p| [p[0].clamp(0.0, 255.0) as u8, p[1].clamp(0.0, 255.0) as u8, p[2].clamp(0.0, 255.0) as u8, 255])
        .collect();
    let image = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
    let texture = ui.ctx().load_texture(format!("magic-thumb-{file}"), image, egui::TextureOptions::LINEAR);
    cache.insert(file.to_owned(), texture.clone());
    Some(texture)
}

/// The Vision Dataset block: pictures of each answer, and of what it is not.
fn pictures_card(
    ui: &mut Ui,
    doc: &mut MagicDoc,
    thumbnails: &mut HashMap<String, egui::TextureHandle>,
    events: &mut Vec<MagicEvent>,
) {
    if doc.file.model.classes.is_empty() {
        widgets::notice(ui, icon::INFO, "Give the model its answers first (the Answers block).", theme::text_faint());
        return;
    }
    let mut remove: Option<String> = None;
    let labels: Vec<String> = doc.file.model.classes.iter().map(|c| c.label.clone()).collect();
    for (class, label) in labels.iter().enumerate() {
        card(ui, label, |ui| {
            for negative in [false, true] {
                let items: Vec<String> = doc
                    .file
                    .model
                    .training
                    .iter()
                    .chain(&doc.file.model.finetune)
                    .filter(|d| d.class == class && d.negative == negative)
                    .map(|d| d.file.clone())
                    .collect();
                ui.horizontal(|ui| {
                    let title = if negative { format!("Not “{label}”") } else { format!("“{label}”") };
                    ui.label(RichText::new(format!("{title} · {}", items.len())).color(theme::text_dim()));
                    let hint = if negative {
                        "Pictures the model mistook for this answer, to unlearn"
                    } else {
                        "Pictures that show this answer"
                    };
                    if ui.small_button(format!("{}  Add…", icon::PLUS)).on_hover_text(hint).clicked() {
                        events.push(MagicEvent::AddPictures { class, negative });
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
                    for file in items.iter().take(60) {
                        let Some(bytes) = doc.file.data.get(file) else { continue };
                        let Some(texture) = thumbnail(ui, thumbnails, file, bytes) else { continue };
                        let size = texture.size_vec2() * (56.0 / texture.size_vec2().max_elem().max(1.0));
                        let response =
                            ui.add(egui::Image::new((texture.id(), size)).sense(egui::Sense::click())).on_hover_text(
                                format!("{}\nRight-click to remove", file.rsplit('/').next().unwrap_or_default()),
                            );
                        response.context_menu(|ui| {
                            if ui.button(format!("{}  Remove", icon::TRASH)).clicked() {
                                remove = Some(file.clone());
                                ui.close();
                            }
                        });
                    }
                    if items.len() > 60 {
                        ui.label(RichText::new(format!("and {} more", items.len() - 60)).color(theme::text_faint()));
                    }
                });
            }
        });
        ui.add_space(4.0);
    }
    if let Some(file) = remove {
        doc.file.model.training.retain(|d| d.file != file);
        doc.file.model.finetune.retain(|d| d.file != file);
        doc.file.data.remove(&file);
        thumbnails.remove(&file);
    }
}

// ---- packaging --------------------------------------------------------------------

fn compile_card(ui: &mut Ui, doc: &mut MagicDoc, events: &mut Vec<MagicEvent>) {
    let pictures = doc.file.model.training.len() + doc.file.model.finetune.len();
    let datasets = doc.file.model.datasets.len();
    let mb = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
    let include = &mut doc.file.model.include;
    ui.checkbox(&mut include.model, format!("The model ({:.1} MB)", mb(doc.file.onnx.len())))
        .on_hover_text("Without it, the file is only a configuration");
    ui.add_enabled_ui(!doc.file.weights.is_empty(), |ui| {
        ui.checkbox(
            &mut include.weights,
            format!("Its weights, to train further ({:.1} MB)", mb(doc.file.weights.len())),
        );
    });
    ui.checkbox(&mut include.datasets, format!("Text datasets ({datasets})"));
    let mut both = include.training && include.finetune;
    if ui.checkbox(&mut both, format!("Pictures ({pictures})")).changed() {
        (include.training, include.finetune) = (both, both);
    }
    ui.checkbox(&mut include.blocks, "Blocks to use it with");
    widgets::note(
        ui,
        "Always included: the canvas, the architecture, the tokens and chat tags, the tool schemas, the \
         answers and thresholds. Macros keep the model and its setup, never the datasets.",
    );
    widgets::button_row(ui, |ui| {
        if widgets::accent_button(ui, format!("{}  Save", icon::FLOPPY_DISK), theme::accent(), vec2(0.0, 28.0))
            .clicked()
        {
            events.push(MagicEvent::Save);
        }
        if ui.button(format!("{}  Save as…", icon::FLOPPY_DISK_BACK)).clicked() {
            events.push(MagicEvent::SaveAs);
        }
    });
}

fn use_card(ui: &mut Ui, doc: &MagicDoc, events: &mut Vec<MagicEvent>) {
    let text = doc.file.model.kind == ModelKind::Text;
    widgets::note(
        ui,
        if text {
            "Adds the model to the open macro with a Start block, an AI Agent block that asks it, and a Log \
             block for its answer. The AI Agent block can carry out its tool calls."
        } else {
            "Adds the model to the open macro with a Start block and an AI Watch block that runs it."
        },
    );
    ui.add_enabled_ui(!doc.file.onnx.is_empty(), |ui| {
        if widgets::accent_button(
            ui,
            format!("{}  Use in Macro", icon::TREE_STRUCTURE),
            theme::accent(),
            vec2(0.0, 28.0),
        )
        .on_disabled_hover_text("Train the model (or import one) first")
        .clicked()
        {
            events.push(MagicEvent::UseInMacro);
        }
    });
}
