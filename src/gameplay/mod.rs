use bevy::prelude::*;

mod biome;
pub(crate) mod city;
mod city_panel;
mod drainage;
pub(crate) mod growth;
mod noise;
pub(crate) mod plan;
mod river;
pub(crate) mod road;
mod sun;
pub(crate) mod terrain;
mod tint;
pub(crate) mod weather;
pub(crate) mod world;

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
        // Neither the weather nor the tint nor the sun is world state — they read no
        // tiles and edit none — so they are their own plugins here rather than part
        // of the world's. The tint's *input* comes from the world, but only as a
        // queue the world already fills; nothing it does can change a tile.
        //
        // The stats panel is a sibling for the same reason and a different one: it edits
        // no tile either, but unlike those two it *reads* world state. So the rule this
        // list follows is "edits tiles → inside the world's plugin", and reading implies
        // neither — the panel asks `CityMap` a question and takes the answer away.
        //
        // The sun goes in after the two overlays it writes to, because its sync
        // system expects both of them on the camera.
        app.add_plugins((
            world::WorldPlugin,
            tint::TerrainTintPlugin,
            weather::WeatherPlugin,
            city_panel::CityPanelPlugin,
            sun::SunPlugin,
        ));
    }
}
