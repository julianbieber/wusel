use bevy::prelude::*;

mod city;
mod noise;
mod plan;
mod road;
mod terrain;
mod world;

pub use world::world_half_extent;

pub struct GameplayPlugin;

impl Plugin for GameplayPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(world::WorldPlugin);
    }
}
