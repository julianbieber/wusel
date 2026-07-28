use bevy::prelude::*;

mod noise;
mod terrain;
mod world;

pub use world::world_half_extent;

pub struct GameplayPlugin;

impl Plugin for GameplayPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(world::WorldPlugin);
    }
}
