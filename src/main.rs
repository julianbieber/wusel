use bevy::{feathers::FeathersPlugins, prelude::*};

use crate::screens::ScreenPlugin;

mod gameplay;
mod main_screen;
mod screens;
mod tooltip;

fn main() -> AppExit {
    App::new()
        .add_plugins((
            DefaultPlugins.set(ImagePlugin::default_nearest()),
            FeathersPlugins,
            ScreenPlugin,
        ))
        .run()
}
