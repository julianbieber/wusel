use bevy::{feathers::FeathersPlugins, log::LogPlugin, prelude::*};

use crate::{camera::CameraPlugin, control::ControlPlugin, screens::ScreenPlugin};

mod camera;
mod control;
mod gameplay;
mod main_screen;
mod screens;
mod tooltip;

fn main() -> AppExit {
    App::new()
        .add_plugins((
            DefaultPlugins
                .set(ImagePlugin::default_nearest())
                // The log layer has to exist before the logger does, and `LogPlugin` is
                // built long before `ControlPlugin`, so this is wired here rather than
                // inside the plugin that reads it. It installs nothing unless the game is
                // being driven.
                .set(LogPlugin {
                    custom_layer: control::log_layer,
                    ..default()
                }),
            FeathersPlugins,
            CameraPlugin,
            ScreenPlugin,
            // Inert unless `WUSEL_CONTROL` names a socket, so the shipped game is
            // unaffected by its presence.
            ControlPlugin,
        ))
        .run()
}
