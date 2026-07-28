use bevy::{
    feathers::{
        controls::FeathersButton,
        theme::{ThemeBackgroundColor, ThemedText},
        tokens,
    },
    input_focus::AutoFocus,
    prelude::*,
    ui_widgets::Activate,
};

use crate::{
    screens::Screen,
    tooltip::{TooltipPlugin, *},
};
pub struct MainScreenPlugin;

impl Plugin for MainScreenPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(TooltipPlugin);
        app.add_systems(OnEnter(Screen::Main), setup_ui);
        app.add_systems(OnEnter(Screen::Help), setup_help);
    }
}

fn setup_ui(mut commands: Commands) {
    commands.spawn_scene(main_root());
}

/// 3 Buttons:
/// * Play
/// * Help
/// * Quit
fn main_root() -> impl Scene {
    bsn! {
        DespawnOnExit<Screen>(Screen::Main)
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            width: percent(100),
            height: percent(100),
            row_gap: px(10),
        }
        ThemeBackgroundColor(tokens::WINDOW_BG)
        Children[
            (
                @FeathersButton{
                    @caption: bsn! {Text("Play!") ThemedText}
                }
                on(go_to_play)
                AutoFocus
            ),
            (
                @FeathersButton{
                    @caption: bsn! {Text("Help") ThemedText}
                }
                on(go_to_help)
                AutoFocus
            ),
            (
                @FeathersButton{
                    @caption: bsn! {Text("Quit") ThemedText}
                }
                on(quit)
                AutoFocus
            )
        ]
    }
}

fn go_to_help(_: On<Activate>, mut next: ResMut<NextState<Screen>>) {
    next.set(Screen::Help);
}

fn go_to_play(_: On<Activate>, mut next: ResMut<NextState<Screen>>) {
    next.set(Screen::Gameplay);
}

fn setup_help(
    commands: Commands,
    asset_server: Res<AssetServer>,
    known_toolips: Res<TooltipMap>,
    mut stack: ResMut<TooltipStack>,
) {
    spawn_tooltip(
        commands,
        asset_server,
        &known_toolips.tooltips,
        &mut stack.entities,
        "Some text containing clickable words, and non clickable words\nand a line break",
        (px(0), px(0)),
        false,
    );
}

fn quit(_: On<Activate>, mut commands: Commands) {
    commands.write_message(AppExit::Success);
}
