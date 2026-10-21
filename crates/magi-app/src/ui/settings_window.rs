//! The settings window: movable and resizable, one page per topic.

use egui::{Color32, RichText, Ui, vec2};
use egui_phosphor::regular as icon;
use magi_core::graph::NodeId;
use magi_core::hotkey::Hotkey;
use magi_core::nodes::flow::StartMode;
use magi_core::project::EnginePreference;
use magi_core::recording::MouseRecording;
use magi_engine::PythonStatus;

use super::widgets::{self, HotkeyAction};
use super::{HotkeyTarget, UiRequest};
use crate::components::{Component, Status as ComponentStatus};
use crate::settings::{Settings, SettingsPage, UI_SCALE_RANGE};
use crate::theme::{self, ThemeMode};

/// Theme colors offered as swatches; any other can be picked.
const SWATCHES: [[u8; 3]; 8] = [
    [0x9B, 0x6D, 0xFF],
    [0x4D, 0xA3, 0xFF],
    [0x2D, 0xD4, 0xC0],
    [0x5B, 0xD1, 0x6B],
    [0xFF, 0xB5, 0x4A],
    [0xFF, 0x7A, 0x45],
    [0xF0, 0x5D, 0x6A],
    [0xFF, 0x7A, 0xB8],
];
/// The interface size is applied this long after the slider stops moving.
const SCALE_DELAY: f64 = 0.35;

/// A Start block of the open macro, for the hotkeys page.
pub struct StartHotkey {
    pub node: NodeId,
    pub name: String,
    pub hotkey: Hotkey,
}

pub struct SettingsEnv<'a> {
    pub capturing: Option<(&'a HotkeyTarget, &'a Hotkey)>,
    pub python: &'a PythonStatus,
    pub starts: &'a [StartHotkey],
    /// Pictures already in the backgrounds folder.
    pub backgrounds: &'a [String],
    /// A background is being downloaded.
    pub downloading: bool,
    /// Python interpreters found by the last check; `None` before the first.
    pub pythons: Option<&'a [(std::path::PathBuf, String)]>,
    /// The interpreter "automatic" stood for at the last check.
    pub python_auto: Option<&'a (std::path::PathBuf, String)>,
    /// A check for Python is under way.
    pub looking_for_python: bool,
    /// The Python downloaded from this page, if there is one.
    pub python_downloaded: Option<std::path::PathBuf>,
    pub gpus: &'a [magi_platform::gpu::Gpu],
    pub machine: &'a magi_platform::system::Machine,
    /// Where the optional parts stand, and what PyTorch sees (`None` until worked out).
    pub runtime: Option<&'a crate::components::RuntimeReport>,
    pub checking_runtime: bool,
    /// The Python AI uses: the one macros use.
    pub ai_python: Option<&'a (std::path::PathBuf, String)>,
    /// The part being installed or removed, what it is doing, and the share done when known.
    pub installing: Option<(Component, String, Option<f32>)>,
}

pub enum SettingsEvent {
    Request(UiRequest),
    /// Look for Pythons, and try to start the one macros use.
    CheckPython,
    ChooseBackgroundFile,
    DownloadBackground(String),
    RemoveBackground(String),
    ResetHotkeys,
    ResetAll,
    /// A Start block's hotkey was changed or cleared here.
    SetStartHotkey(NodeId, Hotkey),
    /// Work out the optional downloads' statuses.
    CheckComponents,
    /// Look for Pythons without starting the engine (the AI page needs to know where PyTorch would be).
    FindPythons,
    Install(Component),
    Remove(Component),
}

#[derive(Default)]
pub struct SettingsWindow {
    pub open: bool,
    url: String,
    confirm_reset: bool,
}

fn section(ui: &mut Ui, title: &str) {
    ui.add_space(10.0);
    ui.label(RichText::new(title.to_uppercase()).size(10.5).color(theme::text_faint()));
    ui.add_space(2.0);
}

fn hotkey_row(
    ui: &mut Ui,
    label: &str,
    hint: &str,
    hotkey: &mut Hotkey,
    target: HotkeyTarget,
    env: &SettingsEnv<'_>,
    events: &mut Vec<SettingsEvent>,
) -> Option<Hotkey> {
    let mut set = None;
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, label, 150.0).on_hover_text(hint);
        let pending = env.capturing.filter(|(active, _)| **active == target).map(|(_, pending)| pending);
        match widgets::hotkey_button(ui, hotkey, pending) {
            HotkeyAction::Capture => events.push(SettingsEvent::Request(UiRequest::CaptureHotkey(target))),
            HotkeyAction::Cancel => events.push(SettingsEvent::Request(UiRequest::CancelCapture)),
            HotkeyAction::Clear => {
                *hotkey = Hotkey::none();
                set = Some(Hotkey::none());
            }
            HotkeyAction::Set(new) => {
                *hotkey = new.clone();
                set = Some(new);
            }
            HotkeyAction::None => {}
        }
    });
    set
}

impl SettingsWindow {
    pub fn show(&mut self, ctx: &egui::Context, settings: &mut Settings, env: &SettingsEnv<'_>) -> Vec<SettingsEvent> {
        let mut events = Vec::new();
        if !self.open {
            return events;
        }
        let mut open = true;
        let screen = ctx.content_rect();
        let default = settings.settings_size.map_or(vec2(600.0, 620.0), |[w, h]| vec2(w, h));
        let response = egui::Window::new(format!("{}  Settings", icon::GEAR))
            .id(egui::Id::new("settings-window"))
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .movable(true)
            .constrain(true)
            .default_size(default.min(screen.size() - vec2(20.0, 20.0)))
            .min_size(vec2(300.0, 240.0))
            .default_pos(screen.center() - default / 2.0)
            .show(ctx, |ui| self.contents(ui, settings, env, &mut events));
        if let Some(response) = response {
            let size = response.response.rect.size();
            settings.settings_size = Some([size.x.round(), size.y.round()]);
        }
        self.open = open;
        events
    }

    fn contents(
        &mut self,
        ui: &mut Ui,
        settings: &mut Settings,
        env: &SettingsEnv<'_>,
        events: &mut Vec<SettingsEvent>,
    ) {
        let narrow = ui.available_width() < 430.0;
        let mut page_ui = |ui: &mut Ui, settings: &mut Settings| {
            // Scrolls both ways: a narrow window shows less, it does not grow.
            egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
                match settings.settings_page {
                    SettingsPage::General => self.general(ui, settings, events),
                    SettingsPage::Appearance => self.appearance(ui, settings, env, events),
                    SettingsPage::Hotkeys => hotkeys(ui, settings, env, events),
                    SettingsPage::Recording => recording(ui, settings),
                    SettingsPage::Engine => engine(ui, settings, env, events),
                    SettingsPage::Runtime => runtime(ui, settings, env, events),
                    SettingsPage::Hardware => hardware(ui, settings, env, events),
                }
                ui.add_space(8.0);
            });
        };
        if narrow {
            widgets::combo("settings-page")
                .width(ui.available_width())
                .selected_text(settings.settings_page.label())
                .show_ui(ui, |ui| {
                    let needle = widgets::combo_search(ui);
                    for page in SettingsPage::ALL {
                        if widgets::option(ui, &needle, settings.settings_page == page, page.label()) {
                            settings.settings_page = page;
                        }
                    }
                });
            ui.separator();
            page_ui(ui, settings);
        } else {
            ui.horizontal_top(|ui| {
                ui.vertical(|ui| {
                    ui.set_width(118.0);
                    for page in SettingsPage::ALL {
                        let glyph = match page {
                            SettingsPage::General => icon::SLIDERS_HORIZONTAL,
                            SettingsPage::Appearance => icon::PALETTE,
                            SettingsPage::Hotkeys => icon::KEYBOARD,
                            SettingsPage::Recording => icon::RECORD,
                            SettingsPage::Engine => icon::LIGHTNING,
                            SettingsPage::Runtime => icon::PULSE,
                            SettingsPage::Hardware => icon::CPU,
                        };
                        let selected = settings.settings_page == page;
                        let text = RichText::new(format!("{glyph}  {}", page.label())).color(if selected {
                            theme::text()
                        } else {
                            theme::text_dim()
                        });
                        let button =
                            egui::Button::new(text).selected(selected).frame(selected).min_size(vec2(118.0, 26.0));
                        if ui.add(button).clicked() {
                            settings.settings_page = page;
                        }
                    }
                });
                ui.separator();
                ui.vertical(|ui| page_ui(ui, settings));
            });
        }
    }

    fn general(&mut self, ui: &mut Ui, settings: &mut Settings, events: &mut Vec<SettingsEvent>) {
        section(ui, "Window");
        widgets::toggle_row(ui, "Keep MacroMagi on top of other windows", "", &mut settings.always_on_top);
        widgets::toggle_row(
            ui,
            "Hide MacroMagi from screen captures",
            "Keeps this window out of what macros see, and out of images you capture for them. It also hides the \
             window from screenshots and screen sharing – turn it off to record a tutorial.",
            &mut settings.hide_from_capture,
        );
        ui_scale(ui, settings);

        section(ui, "Exporting");
        widgets::note(ui, "What the Export dialog includes unless you change it there.");
        widgets::toggle_row(
            ui,
            "My presets",
            "Bundle your preset library into the file, so the presets travel with it.",
            &mut settings.export.presets,
        );
        widgets::toggle_row(
            ui,
            "Options",
            "Stop hotkeys, program rules and the other Options, so they apply wherever the file is opened.",
            &mut settings.export.options,
        );
        widgets::toggle_row(ui, "Auto Clicker settings", "", &mut settings.export.clicker);

        section(ui, "Safety");
        widgets::toggle_row(
            ui,
            "Dry run: do not send any input",
            "Macros run normally but nothing is pressed or clicked. The console lists what would have been sent.",
            &mut settings.dry_run,
        );

        section(ui, "Reset");
        if self.confirm_reset {
            widgets::notice(
                ui,
                icon::WARNING,
                "Every setting goes back to its default, hotkeys included.",
                theme::warning(),
            );
            ui.horizontal(|ui| {
                if widgets::accent_button(ui, "Reset everything", theme::danger(), vec2(130.0, 26.0)).clicked() {
                    events.push(SettingsEvent::ResetAll);
                    self.confirm_reset = false;
                }
                if ui.button("Cancel").clicked() {
                    self.confirm_reset = false;
                }
            });
        } else if ui.button(format!("{}  Reset all settings…", icon::ARROW_COUNTER_CLOCKWISE)).clicked() {
            self.confirm_reset = true;
        }

        section(ui, "About");
        about(ui);
    }

    fn appearance(
        &mut self,
        ui: &mut Ui,
        settings: &mut Settings,
        env: &SettingsEnv<'_>,
        events: &mut Vec<SettingsEvent>,
    ) {
        let look = &mut settings.appearance;
        section(ui, "Theme");
        widgets::segmented(
            ui,
            &mut look.theme,
            &[(ThemeMode::System, ThemeMode::System.label()), (ThemeMode::Light, "Light"), (ThemeMode::Dark, "Dark")],
        );

        section(ui, "Theme color");
        widgets::note(ui, "Colors every part of MacroMagi, icons and blocks included.");
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(6.0, 6.0);
            // The rings reach past the circles; keep them clear of the edge.
            ui.add_space(4.0);
            let mono = look.accent.is_none();
            if swatch(ui, None, mono).on_hover_text("Monochrome").clicked() {
                look.accent = None;
            }
            for rgb in SWATCHES {
                if swatch(ui, Some(rgb), look.accent == Some(rgb)).clicked() {
                    look.accent = Some(rgb);
                }
            }
            let mut custom = look.accent.unwrap_or([0x9B, 0x6D, 0xFF]);
            if ui.color_edit_button_srgb(&mut custom).on_hover_text("Any color").changed() {
                look.accent = Some(custom);
            }
        });

        section(ui, "Background");
        ui.horizontal_wrapped(|ui| {
            let current = look.background.clone().unwrap_or_else(|| "None".to_owned());
            widgets::combo("background").width(170.0).selected_text(current).show_ui(ui, |ui| {
                let needle = widgets::combo_search(ui);
                if widgets::option(ui, &needle, look.background.is_none(), "None") {
                    look.background = None;
                }
                for name in env.backgrounds {
                    ui.horizontal(|ui| {
                        if widgets::option(ui, &needle, look.background.as_ref() == Some(name), name) {
                            look.background = Some(name.clone());
                        }
                        if widgets::icon_button(ui, icon::TRASH, "Delete this picture").clicked() {
                            events.push(SettingsEvent::RemoveBackground(name.clone()));
                        }
                    });
                }
            });
            if ui.button(format!("{}  File…", icon::FOLDER_OPEN)).on_hover_text("PNG, JPEG, GIF, WebP or BMP").clicked()
            {
                events.push(SettingsEvent::ChooseBackgroundFile);
            }
        });
        ui.horizontal(|ui| {
            let field = ui.add_enabled(
                !env.downloading,
                egui::TextEdit::singleline(&mut self.url)
                    .hint_text("https://… picture or GIF address")
                    .desired_width(ui.available_width() - 90.0),
            );
            let entered = field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            let label =
                if env.downloading { "Loading…".to_owned() } else { format!("{}  Add", icon::DOWNLOAD_SIMPLE) };
            let ready = !env.downloading && !self.url.trim().is_empty();
            if (ui.add_enabled(ready, egui::Button::new(label)).clicked() || (entered && ready)) && ready {
                events.push(SettingsEvent::DownloadBackground(self.url.trim().to_owned()));
                self.url.clear();
            }
        });
        ui.add_enabled_ui(look.background.is_some(), |ui| {
            percent_slider(ui, "Picture", "How strongly the picture shows.", &mut look.background_opacity);
            ui.horizontal(|ui| {
                widgets::fixed_label(ui, RichText::new("Blur").color(theme::text_dim()), 110.0);
                ui.add(egui::Slider::new(&mut look.background_blur, 0.0..=40.0).suffix(" px").fixed_decimals(0));
            });
            percent_slider(
                ui,
                "Darkness",
                "Darkens the picture (lightens it in light mode) so text stays readable.",
                &mut look.background_darkness,
            );
            percent_slider(ui, "Panel opacity", "How solid the panels over the picture are.", &mut look.panel_opacity);
        });
    }
}

fn percent_slider(ui: &mut Ui, label: &str, hint: &str, value: &mut f32) {
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new(label).color(theme::text_dim()), 110.0).on_hover_text(hint);
        let mut percent = *value * 100.0;
        if ui.add(egui::Slider::new(&mut percent, 0.0..=100.0).suffix("%").fixed_decimals(0)).changed() {
            *value = percent / 100.0;
        }
    });
}

/// A round color swatch; `None` is the monochrome swatch (half black, half white).
fn swatch(ui: &mut Ui, rgb: Option<[u8; 3]>, selected: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(vec2(22.0, 22.0), egui::Sense::click());
    let painter = ui.painter();
    let center = rect.center();
    match rgb {
        Some([r, g, b]) => {
            painter.circle_filled(center, 9.0, Color32::from_rgb(r, g, b));
        }
        None => {
            painter.circle_filled(center, 9.0, Color32::BLACK);
            let half: Vec<egui::Pos2> = (0..=16)
                .map(|i| {
                    let angle = -std::f32::consts::FRAC_PI_2 + std::f32::consts::PI * i as f32 / 16.0;
                    center + 9.0 * vec2(angle.cos(), angle.sin())
                })
                .collect();
            painter.add(egui::Shape::convex_polygon(half, Color32::WHITE, egui::Stroke::NONE));
        }
    }
    let ring = if selected {
        theme::text()
    } else if response.hovered() {
        theme::text_dim()
    } else {
        theme::stroke_strong()
    };
    painter.circle_stroke(center, 10.5, egui::Stroke::new(if selected { 2.0 } else { 1.0 }, ring));
    response
}

/// The interface size: a slider that applies a moment after it stops moving,
/// and a value that can be typed as a percentage.
fn ui_scale(ui: &mut Ui, settings: &mut Settings) {
    let id = egui::Id::new("ui-scale-pending");
    let now = ui.input(|i| i.time);
    let (mut pending, mut changed_at): (f32, f64) =
        ui.data(|d| d.get_temp(id)).unwrap_or((settings.ui_scale, f64::NEG_INFINITY));
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("Interface size").color(theme::text_dim()), 110.0);
        let slider = egui::Slider::new(&mut pending, UI_SCALE_RANGE)
            .step_by(0.05)
            .update_while_editing(false)
            .custom_formatter(|v, _| format!("{:.0}%", v * 100.0))
            .custom_parser(|text| {
                let number: f64 = text.trim().trim_end_matches('%').trim().replace(',', ".").parse().ok()?;
                // "125" and "125%" mean 125%; "1.25" means the same.
                Some(if number > 5.0 { number / 100.0 } else { number })
            });
        let response = ui.add(slider);
        if response.changed() {
            changed_at = now;
        }
        if widgets::reset_button(ui, (pending - 1.0).abs() > 1e-4) {
            pending = 1.0;
            changed_at = f64::NEG_INFINITY;
        }
        let dragging = response.dragged() || ui.input(|i| i.pointer.any_down());
        if (pending - settings.ui_scale).abs() > 1e-4 {
            if !dragging && now - changed_at >= SCALE_DELAY {
                settings.ui_scale = pending;
            } else {
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(60));
            }
        } else {
            // Follows changes made elsewhere (reset, another window).
            pending = settings.ui_scale;
        }
    });
    ui.data_mut(|d| d.insert_temp(id, (pending, changed_at)));
}

fn hotkeys(ui: &mut Ui, settings: &mut Settings, env: &SettingsEnv<'_>, events: &mut Vec<SettingsEvent>) {
    widgets::note(ui, "Any key, key combination, mouse button or gamepad button. Hotkeys work in every program.");

    section(ui, "Auto Clicker");
    hotkey_row(
        ui,
        "Start / stop",
        "Starts and stops the auto clicker while the Auto Clicker tab is open.",
        &mut settings.clicker.hotkey,
        HotkeyTarget::Clicker,
        env,
        events,
    );
    ui.horizontal(|ui| {
        ui.add_space(154.0);
        widgets::segmented(
            ui,
            &mut settings.clicker.activation,
            &[(StartMode::Toggle, "Toggle"), (StartMode::Hold, "Hold")],
        );
    });

    section(ui, "Macro");
    hotkey_row(
        ui,
        "Run / stop",
        "Like the Run button: starts the macro, or stops it.",
        &mut settings.run_hotkey,
        HotkeyTarget::Run,
        env,
        events,
    );
    hotkey_row(
        ui,
        "Record",
        "Starts and stops recording your input.",
        &mut settings.record_hotkey,
        HotkeyTarget::Record,
        env,
        events,
    );
    for start in env.starts {
        let mut hotkey = start.hotkey.clone();
        if let Some(new) = hotkey_row(
            ui,
            &start.name,
            "A Start block of the open macro",
            &mut hotkey,
            HotkeyTarget::Node(start.node, "hotkey".into()),
            env,
            events,
        ) {
            events.push(SettingsEvent::SetStartHotkey(start.node, new));
        }
    }

    section(ui, "Stopping");
    hotkey_row(
        ui,
        "Stop everything",
        "Stops the auto clicker and every running macro, whatever has focus.",
        &mut settings.stop_hotkey,
        HotkeyTarget::StopEverything,
        env,
        events,
    );
    let mut remove = None;
    for (index, hotkey) in settings.options.stop_hotkeys.iter_mut().enumerate() {
        ui.push_id(("extra-stop", index), |ui| {
            if let Some(new) = hotkey_row(
                ui,
                "Also stops",
                "An extra stop hotkey",
                hotkey,
                HotkeyTarget::ExtraStop(Some(index)),
                env,
                events,
            ) && new.is_none()
            {
                remove = Some(index);
            }
        });
    }
    if let Some(index) = remove {
        settings.options.stop_hotkeys.remove(index);
    }
    let adding = HotkeyTarget::ExtraStop(None);
    if env.capturing.is_some_and(|(t, _)| *t == adding) {
        let mut empty = Hotkey::none();
        hotkey_row(ui, "Also stops", "", &mut empty, adding, env, events);
    } else if ui.button(format!("{}  Add another stop hotkey", icon::PLUS)).clicked() {
        events.push(SettingsEvent::Request(UiRequest::CaptureHotkey(adding)));
    }
    widgets::toggle_row(ui, "Stop on task switch (Alt+Tab, Win+Tab)", "", &mut settings.options.stop_on_task_switch);
    widgets::toggle_row(
        ui,
        "Strict hotkey modifiers",
        "On: a hotkey works only when exactly its modifier keys are held. Off: extra held modifiers are ignored.",
        &mut settings.options.strict_modifiers,
    );

    let conflicts = conflicts(settings, env.starts);
    if !conflicts.is_empty() {
        ui.add_space(6.0);
        for hotkey in conflicts {
            widgets::notice(ui, icon::WARNING, format!("{} is used more than once.", hotkey.label()), theme::warning());
        }
    }

    ui.add_space(10.0);
    if ui
        .button(format!("{}  Reset all hotkeys", icon::ARROW_COUNTER_CLOCKWISE))
        .on_hover_text("Puts every hotkey above back to its default. Start blocks keep theirs.")
        .clicked()
    {
        events.push(SettingsEvent::ResetHotkeys);
    }
}

/// Hotkeys bound to more than one thing that can be active at the same time.
fn conflicts(settings: &Settings, starts: &[StartHotkey]) -> Vec<Hotkey> {
    let global = [&settings.stop_hotkey, &settings.record_hotkey];
    let check = |tab_specific: Vec<&Hotkey>| {
        let mut all: Vec<&Hotkey> =
            global.iter().copied().chain(settings.options.stop_hotkeys.iter()).chain(tab_specific).collect();
        all.retain(|h| !h.is_none());
        let mut seen: Vec<&Hotkey> = Vec::new();
        let mut doubled = Vec::new();
        for hotkey in all {
            if seen.contains(&hotkey) {
                if !doubled.contains(hotkey) {
                    doubled.push(hotkey.clone());
                }
            } else {
                seen.push(hotkey);
            }
        }
        doubled
    };
    let mut found = check(vec![&settings.clicker.hotkey]);
    let macro_keys: Vec<&Hotkey> =
        std::iter::once(&settings.run_hotkey).chain(starts.iter().map(|s| &s.hotkey)).collect();
    for hotkey in check(macro_keys) {
        if !found.contains(&hotkey) {
            found.push(hotkey);
        }
    }
    found
}

fn recording(ui: &mut Ui, settings: &mut Settings) {
    section(ui, "Record");
    ui.horizontal_wrapped(|ui| {
        ui.checkbox(&mut settings.record.keyboard, "Keyboard");
        ui.checkbox(&mut settings.record.mouse_buttons, "Mouse buttons");
        ui.checkbox(&mut settings.record.mouse_moves, "Mouse movement");
        ui.checkbox(&mut settings.record.scroll, "Scroll wheel");
        ui.checkbox(&mut settings.record.gamepad, "Gamepad");
    });

    section(ui, "Mouse movement");
    for mode in MouseRecording::ALL {
        ui.radio_value(&mut settings.record.mouse_mode, mode, mode.label());
    }
    widgets::note(
        ui,
        "Games that lock the pointer (first-person cameras) keep moving it back to the middle. Smart recording \
         notices this and records how far the mouse moved instead, so the camera turns when the recording plays.",
    );

    section(ui, "After recording");
    widgets::toggle_row(ui, "Skip the pause before the first input", "", &mut settings.record.trim_start);
    widgets::skip_switches(ui, &mut settings.record.skip_switches);
    widgets::toggle_row(
        ui,
        "Automatically delete unassigned recordings when saving",
        "Recordings that no Play Recording block plays are left out of the file when it is saved.",
        &mut settings.delete_unused_recordings,
    );
    widgets::toggle_row(
        ui,
        "Add a Play Recording block when a recording ends",
        "",
        &mut settings.auto_insert_recording,
    );
    ui.add_space(4.0);
    widgets::notice(
        ui,
        icon::WARNING,
        "A recording contains everything you type while it runs, passwords included.",
        theme::warning(),
    );
}

fn engine(ui: &mut Ui, settings: &mut Settings, env: &SettingsEnv<'_>, events: &mut Vec<SettingsEvent>) {
    section(ui, "Engine");
    widgets::segmented(
        ui,
        &mut settings.engine,
        &[
            (EnginePreference::Auto, "Automatic"),
            (EnginePreference::Python, "Python"),
            (EnginePreference::Native, "Native"),
        ],
    );
    widgets::note(
        ui,
        match settings.engine {
            EnginePreference::Auto => "Uses Python when it is installed, otherwise the built-in native engine.",
            EnginePreference::Python => "Always uses Python. Needed for Python Script blocks.",
            EnginePreference::Native => "Always uses the built-in Rust engine. Needs nothing installed.",
        },
    );

    section(ui, "Python");
    ui.horizontal_wrapped(|ui| {
        let (text, color) = match env.python {
            PythonStatus::Running { version, command } => (format!("Python {version} ({command})"), theme::success()),
            PythonStatus::Unknown => ("Python has not been started yet".to_owned(), theme::text_dim()),
            PythonStatus::Unavailable(reason) => (reason.clone(), theme::warning()),
        };
        let busy = env.looking_for_python;
        let check = egui::Button::new(if busy {
            "Checking…".to_owned()
        } else {
            format!("{}  Check", icon::ARROWS_CLOCKWISE)
        });
        if ui
            .add_enabled(!busy, check)
            .on_hover_text("Look for Pythons on this computer, and start the one macros use to see whether it works")
            .clicked()
        {
            events.push(SettingsEvent::CheckPython);
        }
        ui.add(egui::Label::new(RichText::new(text).size(11.5).color(color)).wrap());
    });
    // Which Python: whatever "automatic" finds, one of those found, or any file.
    let found = env.pythons.unwrap_or_default();
    let describe = |path: &std::path::Path| match found.iter().find(|(p, _)| p == path) {
        Some((_, version)) => format!("Python {version} – {}", path.display()),
        None => path.display().to_string(),
    };
    let auto = match (env.python_auto, env.pythons) {
        (Some((path, version)), _) => format!("(auto) Python {version} – {}", path.display()),
        (None, Some(_)) => "(auto) no Python found".to_owned(),
        (None, None) => "(auto) not checked yet – press Check".to_owned(),
    };
    let shown = settings.python_path.as_deref().map_or_else(|| auto.clone(), describe);
    widgets::combo("python-choice").width(ui.available_width() - 8.0).truncate().selected_text(shown).show_ui(
        ui,
        |ui| {
            let needle = widgets::combo_search(ui);
            if widgets::option(ui, &needle, settings.python_path.is_none(), &auto) {
                settings.python_path = None;
            }
            for (path, version) in found {
                let chosen = settings.python_path.as_deref() == Some(path.as_path());
                if widgets::option(ui, &needle, chosen, format!("Python {version} – {}", path.display())) {
                    settings.python_path = Some(path.clone());
                }
            }
            if ui.button(format!("{}  Another file…", icon::FOLDER_OPEN)).clicked()
                && let Some(file) = rfd::FileDialog::new().set_title("Choose the Python interpreter").pick_file()
            {
                settings.python_path = Some(file);
            }
        },
    );
    match env.pythons {
        None => widgets::note(ui, "Press Check to look for Python. It is also looked for the first time a macro runs."),
        Some([]) => widgets::notice(ui, icon::WARNING, "No Python was found on this computer.", theme::warning()),
        Some(found) => widgets::note(
            ui,
            if found.len() == 1 {
                "1 Python installation found.".to_owned()
            } else {
                format!("{} Python installations found.", found.len())
            },
        ),
    }

    section(ui, "Embedded Python");
    widgets::note(
        ui,
        "A Python kept in MacroMagi's own folder, apart from any Python on this computer. \"Not installed\" here is about this one only: a Python found above works just as well.",
    );
    match Component::Python.unavailable() {
        Some(_) => {
            widgets::note(
                ui,
                "Python comes with most Linux and macOS systems; install it from python.org if it is missing.",
            );
            if ui.button(format!("{}  Get Python from python.org", icon::ARROW_SQUARE_OUT)).clicked() {
                ui.ctx().open_url(egui::OpenUrl::new_tab("https://www.python.org/downloads/"));
            }
        }
        None => {
            // Without the full check (which the AI page runs), whether the folder is there is enough.
            let known = env
                .runtime
                .and_then(|r| r.statuses.iter().find(|(c, _)| *c == Component::Python))
                .map(|(_, s)| s.clone());
            let status = known.unwrap_or_else(|| match &env.python_downloaded {
                Some(path) => ComponentStatus::Installed { bytes: 0, detail: path.display().to_string() },
                None => ComponentStatus::Missing,
            });
            component_card(ui, Component::Python, Some(&status), env, events);
        }
    }
    widgets::note(
        ui,
        "An unused Python process is stopped after a few minutes to save memory; it starts again on the next run.",
    );
}

/// One optional download: what it is, where it stands, and Install or Uninstall.
fn component_card(
    ui: &mut Ui,
    component: Component,
    status: Option<&ComponentStatus>,
    env: &SettingsEnv<'_>,
    events: &mut Vec<SettingsEvent>,
) {
    egui::Frame::new().fill(theme::bg_raised()).corner_radius(theme::RADIUS).inner_margin(8).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.label(RichText::new(component.name()).strong());
            let (text, color) = match status {
                Some(ComponentStatus::Installed { bytes, detail }) if *bytes > 0 => {
                    (format!("Installed · {detail} · {} MB", bytes / (1024 * 1024)), theme::success())
                }
                Some(ComponentStatus::Installed { detail, .. }) => (format!("Installed · {detail}"), theme::success()),
                Some(ComponentStatus::OnSystem(detail)) => (format!("Found · {detail}"), theme::success()),
                Some(ComponentStatus::Missing) => ("Not installed".to_owned(), theme::text_faint()),
                Some(ComponentStatus::RemovedAfterRestart) => {
                    ("Uninstalled when MacroMagi restarts".to_owned(), theme::warning())
                }
                None => ("…".to_owned(), theme::text_faint()),
            };
            ui.add(egui::Label::new(RichText::new(text).size(11.5).color(color)).truncate());
        });
        widgets::note(ui, component.description());
        match (&env.installing, component.unavailable()) {
            (Some((busy, message, share)), _) if *busy == component => {
                widgets::progress_bar(ui, *share);
                ui.label(RichText::new(message).size(11.0).color(theme::text_dim()));
            }
            (_, Some(reason)) => widgets::note(ui, reason),
            (installing, None) => {
                let idle = installing.is_none();
                ui.horizontal_wrapped(|ui| {
                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                    let installed = matches!(status, Some(ComponentStatus::Installed { .. }));
                    let found = matches!(status, Some(ComponentStatus::OnSystem(_)));
                    if !installed && !found {
                        let label = format!("{}  Install ({})", icon::DOWNLOAD_SIMPLE, component.size());
                        if ui.add_enabled(idle, egui::Button::new(label)).clicked() {
                            events.push(SettingsEvent::Install(component));
                        }
                    }
                    if installed
                        && ui.add_enabled(idle, egui::Button::new(format!("{}  Uninstall", icon::TRASH))).clicked()
                    {
                        events.push(SettingsEvent::Remove(component));
                    }
                });
            }
        }
    });
}

/// The Python AI training uses: the one macros use.
fn ai_python_card(ui: &mut Ui, settings: &mut Settings, env: &SettingsEnv<'_>, events: &mut Vec<SettingsEvent>) {
    egui::Frame::new().fill(theme::bg_raised()).corner_radius(theme::RADIUS).inner_margin(8).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("{}  Python for AI", icon::FILE_PY)).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let busy = env.checking_runtime || env.looking_for_python;
                if ui
                    .add_enabled(!busy, egui::Button::new(format!("{}  Check again", icon::ARROWS_CLOCKWISE)))
                    .clicked()
                {
                    events.push(SettingsEvent::CheckComponents);
                }
                if ui.button("Change…").on_hover_text("Settings › Engine: the Python macros and training use").clicked()
                {
                    settings.settings_page = SettingsPage::Engine;
                }
                if busy {
                    ui.spinner();
                }
            });
        });
        let text = match env.ai_python {
            Some((path, version)) => format!("Python {version} – {}", path.display()),
            None if env.looking_for_python => "Looking for Python…".to_owned(),
            None => "No Python found. Download one under Settings › Engine.".to_owned(),
        };
        ui.add(egui::Label::new(RichText::new(text).size(11.5).color(theme::text_dim())).wrap());
        widgets::note(
            ui,
            "Training runs in this Python, and PyTorch and onnx below are installed into it – the same Python \
             macros use. Change it on the Engine page.",
        );
    });
}

/// One part, in the style of a list of engines: what it is, its version, and Install or Uninstall.
fn runtime_row(
    ui: &mut Ui,
    component: Component,
    env: &SettingsEnv<'_>,
    recommended: bool,
    events: &mut Vec<SettingsEvent>,
) {
    let status = env.runtime.and_then(|r| r.statuses.iter().find(|(c, _)| *c == component)).map(|(_, s)| s);
    egui::Frame::new().fill(theme::bg_raised()).corner_radius(theme::RADIUS).inner_margin(8).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal_wrapped(|ui| {
            let glyph = if component.kind() == "Engine" { icon::PULSE } else { icon::CUBE };
            ui.label(RichText::new(glyph).color(theme::text_dim()));
            ui.label(RichText::new(component.name()).strong());
            let version = match status {
                Some(ComponentStatus::Installed { detail, .. }) | Some(ComponentStatus::OnSystem(detail)) => {
                    detail.split(" · ").next().unwrap_or_default().to_owned()
                }
                _ => String::new(),
            };
            if !version.is_empty() {
                egui::Frame::new()
                    .fill(theme::bg_widget())
                    .corner_radius(4)
                    .inner_margin(egui::Margin::symmetric(5, 1))
                    .show(ui, |ui| {
                        ui.label(RichText::new(version).monospace().size(10.5).color(theme::text_dim()));
                    });
            }
            if recommended && !status.is_some_and(ComponentStatus::present) {
                ui.label(RichText::new("Recommended here").size(10.5).color(theme::accent()));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let idle = env.installing.is_none();
                match (&env.installing, component.unavailable(), status) {
                    (Some((busy, ..)), _, _) if *busy == component => {
                        ui.spinner();
                    }
                    (_, Some(_), _) => {
                        ui.label(RichText::new("Not offered here").size(11.0).color(theme::text_faint()));
                    }
                    (_, None, Some(ComponentStatus::RemovedAfterRestart)) => {
                        ui.label(RichText::new("Removed at the next start").size(11.0).color(theme::warning()));
                    }
                    (_, None, Some(s)) if s.present() => {
                        // Right to left: the buttons, then what is installed.
                        if ui
                            .add_enabled(idle, egui::Button::new(format!("{}  Uninstall", icon::TRASH)))
                            .on_hover_text("Removes it from this computer")
                            .clicked()
                        {
                            events.push(SettingsEvent::Remove(component));
                        }
                        if ui
                            .add_enabled(idle, egui::Button::new(icon::ARROWS_CLOCKWISE))
                            .on_hover_text("Install again (the latest)")
                            .clicked()
                        {
                            events.push(SettingsEvent::Install(component));
                        }
                        ui.label(
                            RichText::new(format!("{}  Installed", icon::CHECK)).size(11.5).color(theme::success()),
                        );
                    }
                    (_, None, None) if env.runtime.is_none() => {
                        ui.label(RichText::new("…").color(theme::text_faint()));
                    }
                    _ => {
                        let label = format!("{}  Install ({})", icon::DOWNLOAD_SIMPLE, component.size());
                        if ui.add_enabled(idle, egui::Button::new(label)).clicked() {
                            events.push(SettingsEvent::Install(component));
                        }
                    }
                }
            });
        });
        ui.add(egui::Label::new(RichText::new(component.description()).size(11.5).color(theme::text_dim())).wrap());
        if let Some(Some((busy, message, share))) = Some(&env.installing)
            && *busy == component
        {
            widgets::progress_bar(ui, *share);
            ui.label(RichText::new(message).size(11.0).color(theme::text_dim()));
        }
        if let Some(reason) = component.unavailable() {
            widgets::note(ui, reason);
        }
    });
    ui.add_space(4.0);
}

fn runtime(ui: &mut Ui, settings: &mut Settings, env: &SettingsEnv<'_>, events: &mut Vec<SettingsEvent>) {
    // Rows fit the window: their text wraps rather than running off to the side.
    ui.set_max_width((ui.clip_rect().width() - 16.0).max(260.0));
    // The packages live in the AI Python: find it first (the statuses follow).
    if env.pythons.is_none() && settings.python_path.is_none() {
        if !env.looking_for_python {
            events.push(SettingsEvent::FindPythons);
        }
    } else if env.runtime.is_none() && !env.checking_runtime {
        events.push(SettingsEvent::CheckComponents);
    }
    ui.horizontal(|ui| {
        section(ui, "Runtime");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Max), |ui| {
            let busy = env.checking_runtime || env.looking_for_python;
            let label = if busy { "Checking…".to_owned() } else { format!("{}  Check again", icon::ARROWS_CLOCKWISE) };
            if ui
                .add_enabled(!busy, egui::Button::new(label))
                .on_hover_text("Looks again at what is installed. MacroMagi also does this when it starts, and when something that needs these fails.")
                .clicked()
            {
                events.push(SettingsEvent::CheckComponents);
            }
        });
    });
    widgets::note(
        ui,
        "What trains and runs AI models. Install what you need; uninstall it any time. Everything comes from its \
         publisher: PyTorch's and Python's package indexes, and Microsoft.",
    );
    ai_python_card(ui, settings, env, events);
    let nvidia = env.gpus.iter().find(|g| g.name.to_lowercase().contains("nvidia")).map(|g| g.name.clone());

    section(ui, "Training models");
    for component in Component::TRAINING {
        let recommended = match component {
            Component::PyTorchCuda => nvidia.is_some(),
            Component::PyTorch => nvidia.is_none(),
            _ => true,
        };
        runtime_row(ui, component, env, recommended, events);
    }
    if let Some(torch) = env.runtime.and_then(|r| r.torch.as_ref())
        && torch.cuda.is_none()
        && let Some(card) = &nvidia
    {
        widgets::note(
            ui,
            format!(
                "PyTorch here is the CPU build. With PyTorch for NVIDIA graphics cards, training uses your {card}."
            ),
        );
    }

    section(ui, "Magi's tools");
    widgets::note(
        ui,
        "What Magi models' native tools use: MagiDB on this computer, documents, pictures. Tools that change files or          run programs always ask you first.",
    );
    for component in Component::TOOLS {
        runtime_row(ui, component, env, component == Component::DocumentTools, events);
    }
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("MagiDB online").color(theme::text_dim()), 140.0)
            .on_hover_text("Where MagiDB's pages are read from when they are not on this computer (raw files)");
        ui.add(
            egui::TextEdit::singleline(&mut settings.magidb_url)
                .hint_text("github.com/MacroMagi/MagiDB")
                .desired_width(240.0),
        );
    });
    ui.horizontal(|ui| {
        widgets::fixed_label(ui, RichText::new("MagiDB folder").color(theme::text_dim()), 140.0)
            .on_hover_text("A folder of MagiDB pages to read instead of the downloaded copy – one you are writing");
        let shown =
            settings.magidb_dir.as_ref().map_or_else(|| "The downloaded copy".to_owned(), |d| d.display().to_string());
        ui.label(RichText::new(shown).color(theme::text_dim()));
        if ui.small_button(icon::FOLDER_OPEN).on_hover_text("Choose a folder").clicked()
            && let Some(folder) = rfd::FileDialog::new().set_title("A folder of MagiDB pages").pick_folder()
        {
            settings.magidb_dir = Some(folder);
        }
        if settings.magidb_dir.is_some() && ui.small_button(icon::X).on_hover_text("Use the downloaded copy").clicked()
        {
            settings.magidb_dir = None;
        }
    });

    section(ui, "Running ONNX models");
    widgets::note(
        ui,
        "Models trained here and imported ONNX models run without these, on the built-in engine; these make them faster.",
    );
    for component in Component::ONNX {
        runtime_row(ui, component, env, false, events);
    }
}

fn hardware(ui: &mut Ui, settings: &mut Settings, env: &SettingsEnv<'_>, events: &mut Vec<SettingsEvent>) {
    ui.set_max_width((ui.clip_rect().width() - 16.0).max(260.0));
    if env.runtime.is_none() && !env.checking_runtime && (env.pythons.is_some() || settings.python_path.is_some()) {
        events.push(SettingsEvent::CheckComponents);
    }
    ui.horizontal(|ui| {
        section(ui, "This computer");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Max), |ui| {
            let busy = env.checking_runtime;
            let label =
                if busy { "Checking…".to_owned() } else { format!("{}  Check again", icon::ARROWS_CLOCKWISE) };
            if ui.add_enabled(!busy, egui::Button::new(label)).clicked() {
                events.push(SettingsEvent::CheckComponents);
            }
        });
    });
    egui::Frame::new().fill(theme::bg_raised()).corner_radius(theme::RADIUS).inner_margin(8).show(ui, |ui| {
        ui.set_width(ui.available_width());
        egui::Grid::new("machine").num_columns(2).spacing([16.0, 4.0]).show(ui, |ui| {
            ui.label(RichText::new("Processor").color(theme::text_dim()));
            ui.label(format!("{} · {} threads", env.machine.cpu, env.machine.threads));
            ui.end_row();
            ui.label(RichText::new("Memory").color(theme::text_dim()));
            ui.label(format!("{:.1} GB", env.machine.memory_mb as f64 / 1024.0));
            ui.end_row();
            for gpu in env.gpus {
                ui.label(RichText::new("Graphics card").color(theme::text_dim()));
                ui.label(format!("{} · {:.1} GB", gpu.name, gpu.memory_mb as f64 / 1024.0));
                ui.end_row();
            }
            if env.gpus.is_empty() {
                ui.label(RichText::new("Graphics card").color(theme::text_dim()));
                ui.label(RichText::new("None found").color(theme::text_faint()));
                ui.end_row();
            }
        });
    });

    section(ui, "Training");
    let torch = env.runtime.and_then(|r| r.torch.as_ref());
    let cards: Vec<String> = torch.map(|t| t.devices.clone()).unwrap_or_default();
    match torch {
        None if env.checking_runtime => widgets::note(ui, "Looking at PyTorch…"),
        None => widgets::notice(
            ui,
            icon::INFO,
            "Install PyTorch under Settings › Runtime to train models.",
            theme::text_faint(),
        ),
        Some(t) if t.cuda.is_none() => widgets::notice(
            ui,
            icon::INFO,
            "PyTorch here is the CPU build: training runs on the processor. Install PyTorch for NVIDIA graphics cards \
             (Runtime) to train on a graphics card.",
            theme::text_faint(),
        ),
        Some(_) if cards.is_empty() => widgets::notice(
            ui,
            icon::WARNING,
            "PyTorch for NVIDIA is installed, but sees no NVIDIA graphics card.",
            theme::warning(),
        ),
        Some(_) => {}
    }
    let mut mode = match &settings.train_gpus {
        None => 0,
        Some(list) if list.is_empty() => 1,
        Some(_) => 2,
    };
    let before = mode;
    widgets::segmented(ui, &mut mode, &[(0, "Automatic"), (1, "The CPU"), (2, "Chosen cards")]);
    if mode != before {
        settings.train_gpus = match mode {
            0 => None,
            1 => Some(Vec::new()),
            _ => Some(if cards.is_empty() { Vec::new() } else { vec![0] }),
        };
    }
    match &mut settings.train_gpus {
        None => widgets::note(
            ui,
            match cards.first() {
                Some(card) => format!("Trains on the {card}."),
                None => "Trains on a graphics card when PyTorch can use one, else on the processor.".to_owned(),
            },
        ),
        Some(list) if mode == 1 => {
            list.clear();
            widgets::note(ui, "Trains on the processor, even with a graphics card.");
        }
        Some(list) => {
            if cards.is_empty() {
                widgets::note(ui, "No graphics card PyTorch can use.");
            }
            for (index, card) in cards.iter().enumerate() {
                let mut on = list.contains(&(index as u32));
                if ui.checkbox(&mut on, format!("CUDA {index}: {card}")).changed() {
                    if on {
                        list.push(index as u32);
                        list.sort_unstable();
                    } else {
                        list.retain(|i| *i != index as u32);
                    }
                }
            }
            if list.len() > 1 {
                widgets::note(ui, "Several cards share each batch (data parallel): bigger batches train faster.");
            }
        }
    }

    section(ui, "ONNX models");
    let runtime = magi_ml::onnxruntime();
    let gpu_ready = runtime.as_ref().is_some_and(|(_, gpu)| *gpu);
    let label = |device: magi_ml::Device| match device {
        magi_ml::Device::Auto => "Automatic – whatever is fastest".to_owned(),
        magi_ml::Device::Cpu => "The CPU".to_owned(),
        magi_ml::Device::Gpu { index } => env
            .gpus
            .iter()
            .find(|g| g.index == index)
            .map_or_else(|| format!("Graphics card {}", index + 1), |g| format!("{} ({} MB)", g.name, g.memory_mb)),
    };
    widgets::combo("ai-device")
        .width(ui.available_width() - 8.0)
        .truncate()
        .selected_text(label(settings.ai_device))
        .show_ui(ui, |ui| {
            let needle = widgets::combo_search(ui);
            if widgets::option(ui, &needle, settings.ai_device == magi_ml::Device::Auto, label(magi_ml::Device::Auto)) {
                settings.ai_device = magi_ml::Device::Auto;
            }
            if widgets::option(ui, &needle, settings.ai_device == magi_ml::Device::Cpu, label(magi_ml::Device::Cpu)) {
                settings.ai_device = magi_ml::Device::Cpu;
            }
            for gpu in env.gpus {
                let device = magi_ml::Device::Gpu { index: gpu.index };
                ui.add_enabled_ui(gpu_ready, |ui| {
                    if widgets::option(ui, &needle, settings.ai_device == device, label(device)) {
                        settings.ai_device = device;
                    }
                });
            }
        });
    widgets::note(
        ui,
        match &runtime {
            Some((path, true)) => format!("Vision models and ONNX text models run with ONNX Runtime for graphics cards ({}).", path.display()),
            Some((path, false)) => format!("ONNX models run with ONNX Runtime on the CPU ({}).", path.display()),
            None => "ONNX models run on the built-in engine (CPU). ONNX Runtime (Runtime page) makes them faster and can use a graphics card.".to_owned(),
        },
    );
}

// Who makes MacroMagi, with links to them.
pub fn about(ui: &mut Ui) {
    ui.label(RichText::new(format!("MacroMagi {}", magi_core::VERSION)).strong());
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        ui.label("Developed and created by");
        ui.hyperlink_to("MagiGun", "https://github.com/MagiGun");
        ui.label("and");
        ui.hyperlink_to("uvxszi", "https://github.com/uvxszi");
    });
}

#[cfg(test)]
mod tests {
    use magi_core::input::Key;

    use super::*;

    #[test]
    fn conflicts_are_found_only_between_hotkeys_active_together() {
        let mut settings = Settings::default();
        assert!(conflicts(&settings, &[]).is_empty(), "the defaults do not clash");

        // The clicker hotkey and the run hotkey are never active at the same time.
        settings.run_hotkey = settings.clicker.hotkey.clone();
        assert!(conflicts(&settings, &[]).is_empty());

        let start = StartHotkey { node: 1, name: "Start".into(), hotkey: Hotkey::key(Key::F8) };
        assert_eq!(conflicts(&settings, &[start]), [Hotkey::key(Key::F8)], "F8 is also stop everything");
    }
}
