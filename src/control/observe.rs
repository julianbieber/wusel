//! Ask the running world a question and get JSON back.
//!
//! Everything here is read-only. An observation that changed the world would make a
//! scenario's own measurements part of what it measures.
//!
//! Adding a topic is a Rust change; adding a *scenario* is a data file. That asymmetry
//! is deliberate — it is what keeps writing a scenario per feature cheap.

use bevy::{platform::collections::HashMap, prelude::*};
use serde_json::{Value, json};

use super::log::LogBuffer;
use crate::{
    camera::{WorldCamera, orthographic_scale, visible_half_extent},
    gameplay::{
        city::{City, CitySize},
        deposit::{Deposit, Resource},
        ground::{ClimateMaps, GroundConfig, GroundCover, temperature_offset},
        growth::{CityGrowth, GrowthConfig},
        industry::{CityIndustry, IndustryConfig},
        inspect::{ActiveOverlay, FieldSources},
        market::{CityTreasury, MarketConfig, prices},
        plan::WorldPlan,
        prospect::ProspectMaps,
        road::RoadNetwork,
        sun::{PlanetConfig, Sun},
        terrain::TerrainKind,
        trade::{Caravan, Errand, Trader},
        weather::SkySampler,
        world::{
            BackgroundGeneration, ChunkCoord, TerrainBake, WORLD_CHUNKS, WorldMap, tile_position_at,
        },
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
    Deposits,
    Ground,
    Sun,
    Overlay,
    Screen,
    Traders,
    Log,
}

impl Topic {
    pub(super) fn parse(word: &str) -> Result<Self, String> {
        match word {
            "terrain" => Ok(Self::Terrain),
            "plan" => Ok(Self::Plan),
            "camera" => Ok(Self::Camera),
            "cities" => Ok(Self::Cities),
            "deposits" => Ok(Self::Deposits),
            "ground" => Ok(Self::Ground),
            "sun" => Ok(Self::Sun),
            "overlay" => Ok(Self::Overlay),
            "screen" => Ok(Self::Screen),
            "traders" => Ok(Self::Traders),
            "log" => Ok(Self::Log),
            other => Err(format!(
                "unknown observation: {other} \
                 (terrain, plan, camera, cities, deposits, ground, sun, overlay, \
                 screen, traders, log)"
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
        Topic::Deposits => deposits(world),
        Topic::Ground => ground(world),
        Topic::Sun => sun(world),
        Topic::Overlay => overlay(world),
        Topic::Screen => screen(world),
        Topic::Traders => traders(world),
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

    // The bake reported beside the chunks rather than as a topic of its own, because a
    // scenario asking "is there a world yet" is asking one question: the fields have to
    // be baked *and* the chunks cut from them. Reporting only the chunks would say
    // "0 generated, 4096 pending" throughout the bake and give no clue why.
    let bake = world.get_resource::<TerrainBake>().map(|bake| {
        let (done, total) = bake.progress();
        json!({
            "complete": bake.is_complete(),
            "fields_baked": done,
            "fields_total": total,
            "last_field": bake.last_field(),
        })
    });

    json!({
        "loaded": true,
        "generated_chunks": generated,
        "total_chunks": (WORLD_CHUNKS.x * WORLD_CHUNKS.y) as usize,
        "pending_chunks": remaining,
        "bake": bake,
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
    // The three configs a price is quoted against, taken before the query borrows the
    // world. Cloned rather than held, because a `&World` and a `QueryState` cannot both
    // be live — and they are knobs, so a clone is a handful of floats.
    let market = world.get_resource::<MarketConfig>().cloned();
    let industry_config = world.get_resource::<IndustryConfig>().cloned();
    let growth_config = world.get_resource::<GrowthConfig>().cloned();

    let mut query = world.query::<(
        &City,
        Option<&CityGrowth>,
        Option<&CityIndustry>,
        Option<&CityTreasury>,
    )>();
    let mut list: Vec<Value> = query
        .iter(world)
        .map(|(city, growth, industry, treasury)| {
            // Both maps are built by walking `Resource::ALL`, never by a hand-written
            // list of six — so a seventh resource reaches this topic as a table row in
            // `deposit.rs` and nothing here.
            let by_resource = |read: &dyn Fn(&CityIndustry, Resource) -> f32| -> Option<Value> {
                industry.map(|industry| {
                    Value::Object(
                        Resource::ALL
                            .iter()
                            .map(|resource| {
                                (
                                    resource.label().to_string(),
                                    json!(read(industry, *resource)),
                                )
                            })
                            .collect(),
                    )
                })
            };

            json!({
                "id": city.id,
                "centre": [city.centre.x, city.centre.y],
                "size": format!("{:?}", city.size),
                "radius": city.radius,
                "population": growth.map(|growth| growth.population),
                "food": growth.map(|growth| growth.food),
                "capacity": growth.map(|growth| growth.capacity),
                "stocks": by_resource(&|industry, resource| industry.stock(resource)),
                // Keyed by profession rather than by resource, because that is what a
                // hand *is* — and the two lists cannot drift, since a profession is a
                // label on `Resource` rather than an enum of its own.
                "hands": industry.map(|industry| {
                    Value::Object(
                        Resource::ALL
                            .iter()
                            .map(|resource| {
                                (
                                    resource.profession().to_string(),
                                    json!(industry.hands(*resource)),
                                )
                            })
                            .collect(),
                    )
                }),
                "idle": industry.map(|industry| industry.idle()),
                // What the land asks for in all. Above the population the city is
                // labour-stretched, which is the branch `industry::effort` documents as
                // the one the knobs keep it off — so it is worth being able to see.
                "hands_wanted": industry.map(|industry| industry.total_hands_wanted()),
                "happiness": industry.map(|industry| industry.happiness()),
                "seams": industry.map(|industry| industry.seams()),
                // gh-7's harvest, as its two halves: what the last cut works out to per
                // step — which is what sizes the population — and what is standing in
                // the fields waiting for the next one. Neither is derivable from the
                // other or from `static_yield`, since a crop depends on a season that is
                // over and on whether the barn had room for it.
                "harvest_rate": industry.map(CityIndustry::harvest_rate),
                "ripening": industry.map(CityIndustry::ripening),
                // gh-7. The purse and what this city will pay for one unit of each
                // resource *right now* — which is the number a caravan compares, so a
                // scenario asserting that goods moved the right way reads the same
                // figure the decision was taken on.
                "treasury": treasury.map(CityTreasury::money),
                "prices": match (&market, &industry_config, &growth_config, industry, growth) {
                    (Some(market), Some(config), Some(growth_config), Some(industry), Some(growth)) => {
                        let quoted = prices(market, config, growth_config, industry, growth);
                        Some(Value::Object(
                            Resource::ALL
                                .iter()
                                .map(|resource| {
                                    (resource.label().to_string(), json!(quoted[resource.index()]))
                                })
                                .collect(),
                        ))
                    }
                    _ => None,
                },
            })
        })
        .collect();

    // Sorted by id so two runs of the same scenario produce comparable output; query
    // iteration order is an ECS implementation detail and would make a diff noise.
    list.sort_by_key(|city| city["id"].as_u64().unwrap_or_default());
    json!({ "count": list.len(), "cities": list })
}

/// Every trader's purse, and every wagon's errand and load.
///
/// The one observation gh-7 cannot do without, because a caravan is a coloured dot: a
/// capture proves it is *on* a road and says nothing about what it is carrying, what
/// it paid, or whether it is going anywhere on purpose. Cities are reported by
/// `City.id` here for the same reason `observe deposits` reports its owners that way —
/// an id is stable across runs and an `Entity` is not, so nothing in a scenario may
/// key on one.
fn traders(world: &mut World) -> Value {
    let ids: Vec<(Entity, u32)> = world
        .query::<(Entity, &City)>()
        .iter(world)
        .map(|(entity, city)| (entity, city.id))
        .collect();
    let city_id = |entity: Entity| {
        ids.iter()
            .find(|(candidate, _)| *candidate == entity)
            .map(|(_, id)| *id)
    };

    let purses: Vec<(Entity, u32, f32)> = world
        .query::<(Entity, &Trader)>()
        .iter(world)
        .map(|(entity, trader)| (entity, trader.id, trader.money()))
        .collect();

    // What each wagon is standing on, read before the query below borrows the world.
    // "Is it on a road" is the assertion the whole feature turns on, and a capture
    // cannot settle it — so the answer is reported rather than left to the author.
    let ground: HashMap<IVec2, String> = {
        let places: Vec<IVec2> = world
            .query::<(&Caravan, &Transform)>()
            .iter(world)
            .map(|(_, transform)| {
                tile_position_at(transform.translation.truncate())
                    .floor()
                    .as_ivec2()
            })
            .collect();
        match world.get_resource::<WorldMap>() {
            Some(map) => places
                .into_iter()
                .filter_map(|tile| Some((tile, format!("{:?}", map.tile(tile)?))))
                .collect(),
            None => HashMap::default(),
        }
    };

    let mut wagons: Vec<Value> = world
        .query::<(&Caravan, &Transform)>()
        .iter(world)
        .map(|(caravan, transform)| {
            let trader = purses
                .iter()
                .find(|(entity, ..)| *entity == caravan.trader)
                .map(|(_, id, _)| *id);
            let errand = match caravan.errand {
                Errand::Resting { city, .. } => json!({
                    "state": "resting",
                    "city": city_id(city),
                }),
                Errand::Travelling { leg, to } => json!({
                    "state": "travelling",
                    "to": city_id(to),
                    "travelled_tiles": leg.travelled_tiles,
                    "length_tiles": leg.length_tiles,
                }),
            };
            // Where it actually is, in global tile space. The one thing a capture cannot
            // tell you and the whole point of the feature: a scenario asserts a wagon is
            // *on the road* by checking this tile against `WorldMap`, which no amount of
            // looking at a coloured dot can do.
            let tile = tile_position_at(transform.translation.truncate());
            json!({
                "trader": trader,
                "errand": errand,
                "tile": [tile.x, tile.y],
                "on": ground.get(&tile.floor().as_ivec2()).cloned(),
                "carried": caravan.carried(),
                // Keyed by resource, with what the load cost beside it — the two
                // together are what says whether a sale would be a profit, which is the
                // rule that decides whether cargo travels on.
                "cargo": Value::Object(
                    caravan
                        .cargo()
                        .iter()
                        .map(|lot| {
                            (
                                lot.resource.label().to_string(),
                                json!({ "units": lot.units, "paid_per_unit": lot.paid_per_unit }),
                            )
                        })
                        .collect(),
                ),
            })
        })
        .collect();

    // Sorted for a clean diff between runs, on the trader id and then the load, since
    // a trader's own wagons are otherwise indistinguishable in the output.
    wagons.sort_by(|a, b| {
        let key = |v: &Value| {
            (
                v["trader"].as_u64().unwrap_or_default(),
                v["carried"].as_f64().unwrap_or_default().to_bits(),
            )
        };
        key(a).cmp(&key(b))
    });

    let mut list: Vec<Value> = purses
        .iter()
        .map(|(_, id, money)| json!({ "id": id, "money": money }))
        .collect();
    list.sort_by_key(|trader| trader["id"].as_u64().unwrap_or_default());

    json!({
        "running": !list.is_empty(),
        "traders": list,
        "caravans": wagons,
    })
}

/// The seams, and who works each one.
///
/// The owning city is reported as its `City.id` and **never as an `Entity`**: an id is
/// stable across runs and an entity is not, and the cities topic is already keyed that
/// way. Present but empty before the Deposits stage has run, and absent outside a
/// session — which is the same shape every other world topic has.
fn deposits(world: &mut World) -> Value {
    // Read in two passes because the owner is an entity on one component and an id on
    // another; collecting the seams first keeps the borrow of each query short.
    let seams: Vec<Deposit> = world.query::<&Deposit>().iter(world).copied().collect();
    let mut owners = world.query::<(Entity, &City)>();
    let owner_ids: Vec<(Entity, u32)> = owners.iter(world).map(|(e, city)| (e, city.id)).collect();

    let mut list: Vec<Value> = seams
        .iter()
        .map(|seam| {
            json!({
                "resource": seam.resource.label(),
                "tile": [seam.tile.x, seam.tile.y],
                "richness": seam.richness,
                "owner": seam.owner.and_then(|entity| {
                    owner_ids.iter().find(|(e, _)| *e == entity).map(|(_, id)| *id)
                }),
            })
        })
        .collect();

    // Sorted by tile, so two runs of a scenario diff cleanly — the same rule the
    // cities list is sorted under, and for the same reason.
    list.sort_by_key(|seam| {
        let tile = &seam["tile"];
        (
            tile[1].as_i64().unwrap_or_default(),
            tile[0].as_i64().unwrap_or_default(),
        )
    });
    json!({ "count": list.len(), "deposits": list })
}

/// What the inspection overlay is drawing, and what it is drawing at the middle of
/// the screen.
///
/// The field and its range come from `ActiveOverlay`, which the overlay's own sync
/// wrote this frame — **not** re-derived here. With a range fitted to what is on
/// screen, deriving it a second time would be a second answer to "what is the player
/// looking at", and the two would part company the moment the camera moved.
///
/// The value beside it *is* read afresh, from the CPU's own copy of the field rather
/// than from the map the shader samples — so the pair is a genuine cross-check.
/// `position` is where that value lands on the ramp, through the same `normalize` the
/// shader transcribes: 0 is the low end, 1 the high, and for a diverging field 0.5 is
/// exactly the freezing point.
fn overlay(world: &mut World) -> Value {
    let Some(active) = world.get_resource::<ActiveOverlay>().copied() else {
        return json!({ "running": false });
    };
    let Some(range) = active.range else {
        return json!({ "running": true, "field": active.field.label() });
    };

    // Scoped so the query's borrow ends before the resources are read; the tile is
    // `Copy`. The camera's own `Transform`, for the reason `city_panel.rs` gives at
    // the matching conversion.
    let centre = {
        let mut cameras = world.query_filtered::<&Transform, With<WorldCamera>>();
        cameras
            .iter(world)
            .next()
            .map(|transform| tile_position_at(transform.translation.truncate()))
    };

    let value = centre.and_then(|tile| {
        let offset = temperature_offset(
            world.get_resource::<GroundConfig>()?,
            world.get_resource::<PlanetConfig>()?,
            world.get_resource::<Sun>()?,
        );
        FieldSources {
            world: world.get_resource::<WorldMap>(),
            climate: world.get_resource::<ClimateMaps>(),
            cover: world.get_resource::<GroundCover>(),
            sky: world.get_resource::<SkySampler>(),
            prospect: world.get_resource::<ProspectMaps>(),
            offset,
        }
        .value(active.field, tile)
    });

    json!({
        "running": true,
        "field": active.field.label(),
        "low": range.low,
        "mid": range.mid,
        "high": range.high,
        "diverging": range.diverging,
        "unit": range.unit,
        "centre_tile": centre.map(|tile| [tile.x, tile.y]),
        "value": value,
        "position": value.map(|value| range.normalize(value)),
    })
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
        WorldPlan::Deposits(_) => "deposits",
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
