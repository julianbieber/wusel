use bevy::prelude::*;

mod biome;
mod city;
mod drainage;
mod noise;
mod plan;
mod river;
mod road;
mod terrain;
mod tint;
mod weather;
mod world;

pub use world::world_half_extent;

/// The order the full-screen passes composite in.
///
/// Both of them ping-pong the same view target, so the second reads what the first
/// wrote and the order is the result — it cannot be left to whichever plugin was
/// added first. The tint is the terrain's own shading and goes *under* the weather,
/// so that a cloud shadow darkens shaded ground rather than the shading brightening
/// a cloud.
///
/// A set rather than `.before(weather_pass)`, because a system is only usable as an
/// ordering label where its parameter types are visible, and the weather's are its
/// own business.
#[derive(SystemSet, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ScreenEffectSystems {
    /// The terrain's shading, over the world and under everything else.
    Tint,
    /// Clouds, their shadows and rain.
    Weather,
}

pub struct GameplayPlugin;

impl Plugin for GameplayPlugin {
    fn build(&self, app: &mut App) {
        // Neither the weather nor the tint is world state — they read no tiles and
        // edit none — so they are their own plugins here rather than part of the
        // world's. The tint's *input* comes from the world, but only as a queue the
        // world already fills; nothing it does can change a tile.
        app.add_plugins((
            world::WorldPlugin,
            tint::TerrainTintPlugin,
            weather::WeatherPlugin,
        ));
    }
}
