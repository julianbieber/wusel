//! Ask the running world a question and get JSON back.
//!
//! Everything here is read-only. An observation that changed the world would make a
//! scenario's own measurements part of what it measures.
//!
//! Adding a topic is a Rust change; adding a *scenario* is a data file. That asymmetry
//! is deliberate — it is what keeps writing a scenario per feature cheap.

use bevy::prelude::*;
use serde_json::{Value, json};

use super::log::LogBuffer;
use crate::{
    camera::{WorldCamera, orthographic_scale, visible_half_extent},
    gameplay::{
        city::{City, CitySize},
        ground::{ClimateMaps, GroundConfig, GroundCover, temperature_offset},
        growth::CityGrowth,
        plan::WorldPlan,
        road::RoadNetwork,
        sun::{PlanetConfig, Sun},
        terrain::TerrainKind,
        world::{BackgroundGeneration, ChunkCoord, WORLD_CHUNKS, WorldMap, tile_position_at},
    },
    screens::Screen,
};

/// How much cover counts as "there is snow here" for the world fractions. A tenth,
/// because the pass draws a tile from about there and a threshold that only counted
/// deep snow would report a bare world through a whole snowfall.
const COVER_THRESHOLD: f32 = 0.1;

pub(super) enum Topic {
    Terrain,
    Plan,
    Camera,
    Cities,
    Ground,
    Sun,
    Screen,
    Log,
}

impl Topic {
    pub(super) fn parse(word: &str) -> Result<Self, String> {
        match word {
            "terrain" => Ok(Self::Terrain),
            "plan" => Ok(Self::Plan),
            "camera" => Ok(Self::Camera),
            "cities" => Ok(Self::Cities),
            "ground" => Ok(Self::Ground),
            "sun" => Ok(Self::Sun),
            "screen" => Ok(Self::Screen),
            "log" => Ok(Self::Log),
            other => Err(format!(
                "unknown observation: {other} \
                 (terrain, plan, camera, cities, ground, sun, screen, log)"
            )),
        }
    }
}

pub(super) fn run(world: &mut World, topic: &Topic) -> Value {
    match topic {
        Topic::Terrain => terrain(world),
        Topic::Plan => plan(world),
        Topic::Camera => camera(world),
        Topic::Cities => cities(world),
        Topic::Ground => ground(world),
        Topic::Sun => sun(world),
        Topic::Screen => screen(world),
        Topic::Log => log(world),
    }
}

/// What the weather has left on the ground: here, and over the world.
///
/// Both, because neither alone says what a scenario wants to know. The world
/// fractions are how you assert that it snowed at all; the reading at the view centre
/// is how you tell whether the thing in the capture is the thing in the numbers.
fn ground(world: &mut World) -> Value {
    let Some(cover) = world.get_resource::<GroundCover>() else {
        return json!({ "running": false });
    };
    let (snowed, wet) = cover.fractions_over(COVER_THRESHOLD);

    // The camera's own `Transform`, for the reason `city_panel.rs` gives at the
    // matching conversion: the pan writes it in `Update` and propagation runs in
    // `PostUpdate`, so the global one is a frame behind.
    let centre = {
        let mut cameras = world.query_filtered::<&Transform, With<WorldCamera>>();
        cameras
            .iter(world)
            .next()
            .map(|transform| tile_position_at(transform.translation.truncate()))
    };
    let here = centre.map(|tile| {
        world
            .get_resource::<GroundCover>()
            .map(|cover| cover.at(tile))
    });

    // The climate is absent until its bake lands, and that absence is what makes the
    // world dry — so saying so beats reporting a temperature nobody is standing in.
    let climate = centre.and_then(|tile| {
        let maps = world.get_resource::<ClimateMaps>()?;
        let config = world.get_resource::<GroundConfig>()?;
        let planet = world.get_resource::<PlanetConfig>()?;
        let sun = world.get_resource::<Sun>()?;
        let cell = maps.at(tile);
        let offset = temperature_offset(config, planet, sun);
        Some(json!({
            "normal_celsius": cell.normal_celsius,
            "diurnal_amplitude_celsius": cell.diurnal_amplitude_celsius,
            "humidity": cell.humidity,
            "temperature_celsius": cell.temperature(offset),
            "swing": offset.swing,
            "seasonal_celsius": offset.seasonal_celsius,
        }))
    });

    json!({
        "running": true,
        "climate_baked": climate.is_some(),
        "threshold": COVER_THRESHOLD,
        "snowed_fraction": snowed,
        "wet_fraction": wet,
        "centre_tile": centre.map(|tile| [tile.x, tile.y]),
        "snow": here.flatten().map(|cell| cell.snow),
        "wetness": here.flatten().map(|cell| cell.wetness),
        "climate": climate,
    })
}

/// Where the planet has turned to, and what that delivers.
///
/// The gap gh-26 left: the sun was the first thing in the crate a capture could not
/// settle, because "is that shadow long because it is early or because the ridge is
/// tall" is not a question a PNG answers.
fn sun(world: &mut World) -> Value {
    let Some(sun) = world.get_resource::<Sun>() else {
        return json!({ "running": false });
    };
    let orbit_phase = world
        .get_resource::<PlanetConfig>()
        .map(|planet| planet.orbit_phase);

    json!({
        "running": true,
        "rotation": sun.rotation,
        "hour": sun.hour(),
        "is_up": sun.is_up(),
        "altitude_degrees": sun.position.altitude.to_degrees(),
        "bearing": [sun.position.bearing.x, sun.position.bearing.y],
        "ray_slope": sun.position.ray_slope,
        "light_level": sun.light_level(),
        "declination_degrees": sun.declination.to_degrees(),
        "orbit_phase": orbit_phase,
    })
}

/// The warnings and errors since the last time anyone asked.
///
/// Absent unless the game was started with a control socket — the layer is only installed
/// then, because a buffer with no reader is a slow leak. Saying so beats reporting an
/// empty log, which would read as "nothing went wrong".
fn log(world: &mut World) -> Value {
    match world.get_resource::<LogBuffer>() {
        Some(buffer) => buffer.drain(),
        None => json!({ "available": false, "reason": "log capture is not installed" }),
    }
}

/// How much of the world exists, and what it is made of.
///
/// The histogram is over generated chunks only, so it is meaningful before the
/// background pass finishes — with `generated` beside it saying how much of the world
/// the proportions were taken from.
fn terrain(world: &mut World) -> Value {
    let Some(map) = world.get_resource::<WorldMap>() else {
        return json!({ "loaded": false });
    };

    let mut histogram = [0u64; 16];
    let mut generated = 0usize;
    for chunk in map.generated() {
        generated += 1;
        for kind in chunk.iter() {
            histogram[*kind as usize] += 1;
        }
    }

    let tiles: u64 = histogram.iter().sum();
    let kinds: serde_json::Map<String, Value> = histogram
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .map(|(index, count)| (kind_name(index), json!(count)))
        .collect();

    let remaining = world
        .get_resource::<BackgroundGeneration>()
        .map(BackgroundGeneration::remaining);

    json!({
        "loaded": true,
        "generated_chunks": generated,
        "total_chunks": (WORLD_CHUNKS.x * WORLD_CHUNKS.y) as usize,
        "pending_chunks": remaining,
        "tiles": tiles,
        "kinds": kinds,
    })
}

fn plan(world: &mut World) -> Value {
    let stage = match world.get_resource::<WorldPlan>() {
        Some(plan) => plan_stage(plan),
        None => return json!({ "running": false }),
    };

    let roads = world
        .get_resource::<RoadNetwork>()
        .map(|network| network.links.len());

    let mut sizes = [0u32; 4];
    let mut population = 0.0f32;
    let mut query = world.query::<(&City, Option<&CityGrowth>)>();
    let mut count = 0usize;
    for (city, growth) in query.iter(world) {
        count += 1;
        sizes[size_index(city.size)] += 1;
        if let Some(growth) = growth {
            population += growth.population;
        }
    }

    json!({
        "running": true,
        "stage": stage,
        "cities": count,
        "city_sizes": {
            "hamlet": sizes[0],
            "village": sizes[1],
            "borough": sizes[2],
            "metropolis": sizes[3],
        },
        "population": population,
        "roads": roads,
    })
}

/// Where the camera is and how much world it can see.
///
/// `visible_half_extent` rather than the window size, because the viewport in logical
/// pixels only equals world units at scale 1 — and it is the same answer the pan clamp
/// and the chunk streamer take, so a scenario reads what the game acted on.
fn camera(world: &mut World) -> Value {
    // Scoped so the camera's borrow of the world ends before the chunk count needs its
    // own query — everything wanted from it is `Copy`.
    let Some((translation, visible, scale)) = ({
        let mut query =
            world.query_filtered::<(&Camera, &Transform, &Projection), With<WorldCamera>>();
        query
            .iter(world)
            .next()
            .map(|(camera, transform, projection)| {
                (
                    transform.translation.truncate(),
                    visible_half_extent(camera, projection),
                    orthographic_scale(projection),
                )
            })
    }) else {
        return json!({ "present": false });
    };

    let tile = tile_position_at(translation);

    let mut chunks = world.query::<&ChunkCoord>();
    let resident = chunks.iter(world).count();

    json!({
        "present": true,
        "translation": [translation.x, translation.y],
        "centre_tile": [tile.x, tile.y],
        "scale": scale,
        "visible_half_extent": [visible.x, visible.y],
        "resident_chunks": resident,
    })
}

fn cities(world: &mut World) -> Value {
    let mut query = world.query::<(&City, Option<&CityGrowth>)>();
    let mut list: Vec<Value> = query
        .iter(world)
        .map(|(city, growth)| {
            json!({
                "id": city.id,
                "centre": [city.centre.x, city.centre.y],
                "size": format!("{:?}", city.size),
                "radius": city.radius,
                "population": growth.map(|growth| growth.population),
                "food": growth.map(|growth| growth.food),
                "capacity": growth.map(|growth| growth.capacity),
            })
        })
        .collect();

    // Sorted by id so two runs of the same scenario produce comparable output; query
    // iteration order is an ECS implementation detail and would make a diff noise.
    list.sort_by_key(|city| city["id"].as_u64().unwrap_or_default());
    json!({ "count": list.len(), "cities": list })
}

fn screen(world: &mut World) -> Value {
    match world.get_resource::<State<Screen>>() {
        Some(state) => json!({ "screen": format!("{:?}", state.get()) }),
        None => json!({ "screen": null }),
    }
}

/// Also used by a `wait plan` timeout, which has to say how far it got.
pub(super) fn plan_stage(plan: &WorldPlan) -> &'static str {
    match plan {
        WorldPlan::WaitingForTerrain => "waiting-for-terrain",
        WorldPlan::Rivers(_) => "rivers",
        WorldPlan::Drainage(_) => "drainage",
        WorldPlan::Cities(_) => "cities",
        WorldPlan::Roads(_) => "roads",
        WorldPlan::Done => "done",
    }
}

fn size_index(size: CitySize) -> usize {
    match size {
        CitySize::Hamlet => 0,
        CitySize::Village => 1,
        CitySize::Borough => 2,
        CitySize::Metropolis => 3,
    }
}

/// The atlas index is the enum discriminant, so this table is in the same order as
/// `TerrainKind` and `assets/textures/terrain.png`.
fn kind_name(index: usize) -> String {
    const KINDS: [TerrainKind; 16] = [
        TerrainKind::Forest,
        TerrainKind::ShallowWater,
        TerrainKind::Grass,
        TerrainKind::Town,
        TerrainKind::Mountain,
        TerrainKind::DeepWater,
        TerrainKind::Road,
        TerrainKind::River,
        TerrainKind::Sand,
        TerrainKind::Snow,
        TerrainKind::Rock,
        TerrainKind::Marsh,
        TerrainKind::Scrub,
        TerrainKind::Gravel,
        TerrainKind::Reed,
        TerrainKind::Farmland,
    ];
    match KINDS.get(index) {
        Some(kind) => format!("{kind:?}"),
        None => format!("unknown-{index}"),
    }
}
