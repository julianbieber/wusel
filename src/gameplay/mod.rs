use bevy::prelude::*;

mod biome;
pub(crate) mod city;
mod city_panel;
pub(crate) mod deposit;
pub(crate) mod document;
mod drainage;
pub(crate) mod ground;
pub(crate) mod growth;
pub(crate) mod industry;
pub(crate) mod inspect;
pub(crate) mod market;
mod noise;
pub(crate) mod plan;
pub(crate) mod prospect;
mod river;
pub(crate) mod road;
mod screen;
pub(crate) mod sun;
pub(crate) mod terrain;
mod tint;
pub(crate) mod trade;
pub(crate) mod weather;
pub(crate) mod world;

pub use world::world_half_extent;

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
        // `screen` is the odd one out and is none of those things: it owns no model at
        // all, only the one post-process pass the tint, the sun and the weather draw
        // through. There used to be two passes, and the order they composited in was an
        // ordering between systems that had to be stated here; now it is the order of
        // the lines in one fragment function, so this list is back to being a list.
        //
        // The order in it is *not* load-bearing: every contributor writes the overlay
        // in `Update` or in an `OnEnter` explicitly ordered after the attach, both of
        // which run after the whole `OnEnter` set this list feeds.
        app.add_plugins((
            world::WorldPlugin,
            screen::ScreenEffectPlugin,
            tint::TerrainTintPlugin,
            weather::WeatherPlugin,
            ground::GroundPlugin,
            inspect::InspectPlugin,
            prospect::ProspectPlugin,
            city_panel::CityPanelPlugin,
            sun::SunPlugin,
            // A sibling by the same rule as the panel, and for the same reason it is
            // not obvious: the traders edit no tile — a caravan is an entity, exactly
            // as a seam is — but they *read and write* the cities the world's plugin
            // grew. The ordering that needs is a set on the world's spine, not
            // membership of its plugin.
            trade::TradePlugin,
        ));
    }
}
