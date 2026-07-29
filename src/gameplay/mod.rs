use bevy::prelude::*;

mod city;
mod noise;
mod plan;
mod river;
mod road;
mod terrain;
mod weather;
mod world;

pub use world::world_half_extent;

pub struct GameplayPlugin;

impl Plugin for GameplayPlugin {
    fn build(&self, app: &mut App) {
        // Weather is not world state — it reads no tiles and edits none — so it is
        // its own plugin here rather than part of the world's.
        app.add_plugins((world::WorldPlugin, weather::WeatherPlugin));
    }
}
