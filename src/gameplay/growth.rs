//! Cities that live: food off the land they work, and a population that rises or
//! falls against it.
//!
//! This is the first thing in the crate that is a *simulation* rather than a
//! generator. Everything before it is a pure function of the seed evaluated once;
//! this has state that advances, and most of what is written below is about that
//! difference.
//!
//! **Reproducibility ends here, deliberately.** The harvest is modulated by the
//! live sky, which drifts on the frame clock, so two runs of one seed diverge from
//! the first step. The terrain, the rivers, the cities as founded and the roads are
//! all still identical — the seed decides the world, it no longer decides what
//! becomes of it. The consequence is that nothing here can be tested by reproducing
//! a world, which is why [`step_city`] takes the sky as an argument instead of
//! reading it: with a fixed [`Sky`] every property below is an ordinary unit test.
//!
//! **A tile has one owner, and no code here arranges that.** A claim requires the
//! tile to be habitable, and nothing a city stamps is habitable, so a field one city
//! has taken cannot be taken by its neighbour. The competition between crowded
//! cities is that single existing predicate, not a partition of the land.
//!
//! **The ledger remembers claims, never the ground under them.** A released tile
//! takes its kind from its neighbours, so a ring of fields dissolves back into the
//! country it was cut out of, and there is no second copy of the map to keep true.

use bevy::{platform::collections::HashSet, prelude::*};

use crate::{
    gameplay::{
        city::{City, CityMap, CitySize, MAX_CITY_RADIUS},
        deposit::{Deposit, DepositMap},
        industry::{
            CityIndustry, EstateOffsets, IndustryConfig, Labour, seed_industry, step_industry,
        },
        plan::WorldPlan,
        terrain::{TerrainConfig, TerrainKind, TerrainSampler},
        weather::SkySampler,
        world::{
            DirtyChunks, TileEdit, WorldMap, WorldSystems, chunk_index_of_tile, tile_in_world,
        },
    },
    screens::Screen,
};

/// Knobs for the simulation. Configuration rather than world state, so like
/// [`TerrainConfig`] and `WeatherConfig` this outlives a session.
#[derive(Resource, Clone)]
pub struct GrowthConfig {
    /// Seconds of play per step. The frame's delta is accumulated into this, so the
    /// *rate* of growth is the same on every machine even though its outcome is not
    /// reproducible.
    pub step_seconds: f32,
    /// Steps one frame may run. The accumulator is clamped to this afterwards, so a
    /// stall's backlog is **discarded rather than owed** — without the clamp a
    /// machine that hitches once runs at the cap every frame from then on, never
    /// repays, and quietly simulates slower than the clock it claims to keep.
    pub max_steps_per_frame: u32,
    /// Logistic rate per step. This and `step_seconds` are not independent: the
    /// real-time rate is `growth_rate / step_seconds`.
    pub growth_rate: f32,
    /// Food one head eats per step. With the yields below this is what sets how much
    /// land a city of a given size needs.
    pub food_per_person: f32,
    /// How many people one tile of town holds.
    ///
    /// Must stay well above `farm_yield_grass / food_per_person`, and that is a
    /// stability condition rather than taste: a town tile is built *on* a field, so
    /// if it housed fewer people than the field it destroyed could feed, growing
    /// would be net-negative food and a city would oscillate to nothing.
    /// `a_town_tile_houses_more_than_the_field_it_replaces_feeds` guards it.
    ///
    /// It is also the lever from population onto the *tier*, since the radius is read
    /// back off the town's tile count. At the default the world settles at populations
    /// of 114 (min), 6940 (median) and 13362 (max), which spans Hamlet to Borough.
    ///
    /// **No city reaches `Metropolis` at these defaults**, and that is a real outcome
    /// rather than a mis-set knob: the plan founds two on settlement score, the land
    /// cannot feed them, and they come down to boroughs. Lowering this to 25 does put
    /// 8 cities in the top tier — but it costs the spread, taking Boroughs from 45 of
    /// 92 to 59, which is the homogenisation this whole design is trying to avoid. The
    /// tier that is out of reach is worth less than the distribution.
    pub town_people_per_tile: f32,
    /// What a field yields, by the ground that was cleared for it. Grass beats
    /// forest because the field on cleared woodland is the marginal one, and scrub is
    /// worst — it is the bare rung of the cover ladder, and gh-14 made it habitable,
    /// so without a value of its own it would have been silently priced as woodland.
    ///
    /// The gap between them is doing real work now that `min_field_fertility` cuts
    /// against it: at these values grass clears the floor across most of its humidity
    /// range and forest only in the wet half, so a wood is farmed where it is damp and
    /// left standing where it is not.
    ///
    /// With the rest of the defaults the world settles at **70976 tiles of farmland,
    /// 0.423% of it** — against 0.24% for roads, so the fields read as a feature of the
    /// map rather than as speckle. The towns go from the 6178 tiles the plan founded
    /// them with to 15986, 0.095% of the world.
    pub farm_yield_grass: f32,
    pub farm_yield_forest: f32,
    pub farm_yield_scrub: f32,
    /// How a field's yield splits between what is always true of *that tile* and what
    /// the sky is doing over the city now. `base` is the floor and `humidity` is the
    /// climate, both baked into the tile's own `base_yield` when it is claimed; `rain`
    /// is the visible transient, applied to the whole harvest.
    ///
    /// The climate is sampled **per tile**, not once per city, and that is what gives
    /// a city's fields their shape: the humidity field's wavelength is ~50 tiles
    /// against a 56-tile reach, so one side of a city is measurably wetter than the
    /// other and the good ground is not a ring.
    pub yield_base_weight: f32,
    pub yield_humidity_weight: f32,
    pub yield_rain_weight: f32,
    /// The least fertile ground a city will break, as a fraction of the best ground
    /// there is (well-watered grass).
    ///
    /// **This is what stops the fields being a disc.** Without it a city claims every
    /// habitable tile the cursor reaches, so its farmland is whatever circle the reach
    /// allows minus the coastline — and since almost every city can reach enough land,
    /// almost every city ends up the same size. With it, dry ground and thin woodland
    /// are simply left alone: the fields follow the good country, spreading along a
    /// wet valley and stopping at a dry ridge, and a city in poor country stays small
    /// because the land is poor rather than because a radius says so.
    ///
    /// At the default, with the yields below: grass is worth breaking above humidity
    /// 0.16, forest only above 0.62. Both ends matter — a floor that admits dry
    /// forest is barely a filter, and one that excludes damp grass leaves the world's
    /// cities with nothing to eat.
    pub min_field_fertility: f32,
    /// How much more food a city lays fields for than it currently eats. This is
    /// what makes growth possible at all: a city sized to exactly its own demand has
    /// no surplus, and the logistic term that feeds on surplus is zero.
    pub growth_headroom: f32,
    /// How far past the target the fields must run before any are released, as a
    /// fraction. Without a band a city that has just balanced releases a field, goes
    /// hungry, claims it back, and spends the session flickering one ring in and out
    /// of the map.
    pub farm_hysteresis: f32,
    /// How far from the centre a field may be. It bounds the offset table, so it also
    /// prices the simulation.
    ///
    /// A bound rather than the thing that decides a city's size. Measured on the
    /// default world, a city holds this share of the tiles within its reach: median
    /// **39%**, max 72%, min 1%. So no city fills its circle — what it farms is the
    /// good ground inside the circle, and `min_field_fertility` is what chooses it.
    ///
    /// It was not always so. On an earlier cut with no fertility floor the median was
    /// **97%** and the max 100%: every city took everything it could reach, the reach
    /// was therefore the only thing setting a city's size, and 85 of 107 came out
    /// Boroughs whatever they were founded as. That is the failure this number can
    /// still cause if it is set low enough to bind before the land does.
    ///
    /// gh-14's three-rung cover ladder pushed the median down again, from 57% to 39%:
    /// `Scrub` is habitable and therefore claimable, but poor enough that most of it
    /// is under the floor, so a city now has more reachable land and takes less of it.
    pub farm_max_reach_tiles: u32,
    /// Tiles one city may claim, release or re-stamp per step. The founding rings
    /// are laid unbudgeted; this governs everything after, so that growth spreads
    /// across the map the way the rivers fill in rather than arriving in one frame.
    ///
    /// The unbudgeted part is 25147 tiles across 92 cities in ~36 ms, once, in the
    /// frame the plan reaches `Done`. After that a whole step of the world costs
    /// **0.53 ms** — 2000 steps of 92 cities in 1.06 s — so this is a knob on how fast
    /// the map changes rather than on what the simulation costs.
    ///
    /// Nearly all of that 0.36 ms is one city's cursor rescan, which re-samples the
    /// humidity under every habitable tile in its reach. Only the one city whose turn
    /// it is pays it, so the cost is flat in the number of cities; a step where nobody
    /// rescans is free, because an exhausted cursor claims nothing.
    pub claims_per_step: u32,
    /// A city never falls below this. Without it the logistic drags a starving city
    /// to zero and leaves a town with nobody in it and no rule for what happens next.
    pub min_population: f32,
}

impl Default for GrowthConfig {
    fn default() -> Self {
        Self {
            step_seconds: 0.5,
            max_steps_per_frame: 4,
            growth_rate: 0.02,
            food_per_person: 1.0,
            town_people_per_tile: 40.0,
            farm_yield_grass: 12.0,
            farm_yield_forest: 9.0,
            farm_yield_scrub: 6.0,
            yield_base_weight: 0.55,
            yield_humidity_weight: 0.45,
            yield_rain_weight: 0.35,
            min_field_fertility: 0.62,
            growth_headroom: 0.25,
            farm_hysteresis: 0.20,
            farm_max_reach_tiles: 28,
            claims_per_step: 8,
            min_population: 20.0,
        }
    }
}

/// The session's clock. World state, so it goes on `OnExit` — a new session starts
/// at step 0 rather than inheriting the last world's age.
#[derive(Resource, Default)]
pub struct GrowthClock {
    step: u64,
    carry_seconds: f32,
}

/// The terrain sampler, built once for the session.
///
/// Constructing one costs six noise fields and a biome map, so it is hoisted here
/// for the same reason the weather bake hoists its own out of its loop. After a city
/// is seeded the simulation needs it only for restoring released tiles.
#[derive(Resource)]
struct GrowthFields(TerrainSampler);

/// Every offset a city may claim, sorted by distance from the centre.
///
/// One table shared by every city, built once. It is what makes claiming and
/// releasing O(1): a city keeps a cursor into it, so "the nearest unclaimed tile" is
/// the next entry rather than a rescan of the whole annulus — which was the cost
/// that fell hardest on exactly the hemmed-in cities that never find anything.
#[derive(Resource)]
struct ClaimOffsets(Vec<IVec2>);

impl ClaimOffsets {
    fn build(reach: u32) -> Self {
        let reach = reach as i32;
        let mut offsets: Vec<IVec2> = (-reach..=reach)
            .flat_map(|dy| (-reach..=reach).map(move |dx| IVec2::new(dx, dy)))
            .filter(|offset| offset.length_squared() <= reach * reach)
            .collect();
        // Ties broken by coordinate so the table is one fixed order rather than
        // whatever the flat_map happened to produce.
        offsets.sort_unstable_by_key(|offset| (offset.length_squared(), offset.y, offset.x));
        Self(offsets)
    }
}

/// One tile a city holds.
///
/// There is no record of what was under it. That is the point: a released tile is
/// restored from its neighbours, so the ledger never becomes a second copy of the
/// map that has to be kept true.
///
/// `base_yield` is the static half of what the tile produces — its cleared ground and
/// the humidity over it, neither of which changes. Computed once while the ground is
/// still visible and kept as a number, which is what lets a step multiply one sum
/// instead of walking 150k tiles.
#[derive(Clone, Copy, Debug)]
struct Claim {
    tile: IVec2,
    base_yield: f32,
}

/// A city's population, its fields, and the land it holds.
///
/// The town is the **prefix** of the claims, and the fields after it are kept
/// **sorted by distance from the centre**. That one ordering is what makes every
/// operation here cheap: claiming appends, releasing pops the outermost, and the
/// town grows by taking the nearest field — so no rule ever searches for a tile.
///
/// The *whole* ledger is not sorted, and cannot be. A town grows outward past land
/// it never took — a bay, a neighbour's fields — and a later rescan can turn that
/// land up as a field nearer to the centre than tiles the town has since built on.
/// Such a field is filed among the fields, never into the town's prefix: the town is
/// what the city has built on, not what it happens to own nearest.
#[derive(Component)]
pub struct CityGrowth {
    pub population: f32,
    /// What the fields yielded last step, and what the population needed. Kept
    /// rather than recomputed: together they are the whole explanation of why a city
    /// is growing, and nothing else can reconstruct them afterwards.
    pub food: f32,
    pub demand: f32,
    /// The population the current harvest supports — the logistic's K.
    pub capacity: f32,
    /// Humidity at the centre, sampled once. It never changes, and caching it is
    /// what keeps [`TerrainSampler`] off the per-step path.
    humidity: f32,
    /// The sum of every field's `base_yield`, maintained as land is taken and given
    /// up. A harvest multiplies this; nothing walks the claims to find it.
    static_yield: f32,
    /// How far into [`ClaimOffsets`] this city has looked.
    frontier: usize,
    /// Distance-sorted. `..town_claims` is town, the rest is fields.
    claims: Vec<Claim>,
    town_claims: usize,
    /// Chunks this city's tiles touch, so [`CityMap`] is told about a chunk once
    /// rather than every step.
    chunks: HashSet<usize>,
}

impl CityGrowth {
    /// Tiles of field the city holds — what the measurements count, and what
    /// [`crate::gameplay::city_panel`] shows beside the population.
    pub fn fields(&self) -> usize {
        self.claims.len() - self.town_claims
    }

    /// The harvest with the weather *and* the labour taken out: what this city's
    /// fields are worth if every one of them is worked.
    ///
    /// Read by [`crate::gameplay::industry`], which is the module that decides how
    /// many of them are. Exposed rather than recomputed there, because two answers to
    /// "what is this land worth" is exactly what the incremental sum exists to
    /// prevent.
    pub fn static_yield(&self) -> f32 {
        self.static_yield
    }

    /// Tiles the city has *built* on, as opposed to farms. Reported rather than
    /// derived from `City::radius`, which is the area rounded into a circle: a city
    /// clipped by a coast holds fewer tiles than its radius suggests, and the count
    /// is the honest half of that pair.
    pub fn town(&self) -> usize {
        self.town_claims
    }
}

/// What the sky is doing over a city this step.
///
/// An argument rather than something the step reads, and that is the whole test
/// seam: the rain is the one irreproducible input, so injecting it makes every
/// property of the simulation assertable without an app, a GPU or a clock.
///
/// Rain only. The *climate* used to be here too, and is not, because it belongs to
/// each field rather than to the city: a cloud is far wider than a city, but the
/// humidity that decides whether a tile is worth farming varies across one.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sky {
    pub rain: f32,
}

pub struct CityGrowthPlugin;

impl Plugin for CityGrowthPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GrowthConfig>();
        app.add_systems(OnEnter(Screen::Gameplay), start_simulation);
        app.add_systems(OnExit(Screen::Gameplay), tear_down_simulation);
        app.add_systems(
            Update,
            (seed_cities, simulate_cities)
                .chain()
                .in_set(WorldSystems::Growth),
        );
    }
}

fn start_simulation(
    mut commands: Commands,
    terrain: Res<TerrainConfig>,
    config: Res<GrowthConfig>,
) {
    commands.insert_resource(GrowthClock::default());
    commands.insert_resource(GrowthFields(terrain.sampler()));
    commands.insert_resource(ClaimOffsets::build(config.farm_max_reach_tiles));
}

/// Drops the session's simulation. The cities go with their own entities, which
/// carry `DespawnOnExit`, so nothing here has to unwind a ledger.
fn tear_down_simulation(mut commands: Commands) {
    commands.remove_resource::<GrowthClock>();
    commands.remove_resource::<GrowthFields>();
    commands.remove_resource::<ClaimOffsets>();
}

/// Gives every city its starting population and its founding fields.
///
/// Seeded here rather than in `apply_city_plan` so that the plan never depends on
/// the simulation's types, and keyed on the *absence* of [`CityGrowth`] rather than
/// on `Added<City>`, whose window is long gone by the time the plan reaches `Done`.
///
/// The founding ring is laid **unbudgeted**, the way `OnEnter(Screen::Gameplay)`
/// hands the streamer unlimited budgets. A city that has stood for years already has
/// its fields; making it claim them at the per-step rate would have every city in the
/// world starving for the couple of hundred steps its ring took to fill, which is a
/// world-wide collapse rather than an opening state.
fn seed_cities(
    mut commands: Commands,
    mut map: ResMut<WorldMap>,
    mut dirty: ResMut<DirtyChunks>,
    mut cities: ResMut<CityMap>,
    mut seams: Query<&mut Deposit>,
    config: Res<GrowthConfig>,
    industry_config: Res<IndustryConfig>,
    fields: Res<GrowthFields>,
    offsets: Res<ClaimOffsets>,
    estate: Res<EstateOffsets>,
    deposits: Res<DepositMap>,
    plan: Res<WorldPlan>,
    unseeded: Query<(Entity, &City), Without<CityGrowth>>,
) {
    if !matches!(*plan, WorldPlan::Done) || unseeded.is_empty() {
        return;
    }

    // In id order, because seeding claims land: which of two neighbours takes a
    // contested tile has to be one fixed answer rather than the query's order.
    let mut pending: Vec<(Entity, City)> = unseeded.iter().map(|(e, c)| (e, *c)).collect();
    pending.sort_unstable_by_key(|(_, city)| city.id);

    // Two cities' founding discs can come within a few tiles of each other, and both
    // are `Town`, so the habitability test cannot separate them the way it separates
    // fields. This does, for the one pass where it is needed.
    let mut taken: HashSet<IVec2> = HashSet::new();

    for (entity, city) in pending {
        let mut edits = Vec::new();
        let growth = seed_city(
            &config,
            &map,
            &offsets.0,
            &city,
            entity,
            &fields.0,
            &mut taken,
            &mut cities,
            &mut edits,
        );
        map.apply_edits(&edits, &mut dirty);

        // In the same id order and in the same breath, so which of two neighbours
        // takes a contested seam is one fixed answer — and so the seam's `owner` and
        // the city's list are written from one act rather than derived from each
        // other later.
        let industry = seed_industry(
            &industry_config,
            &map,
            estate.offsets(),
            &city,
            &growth,
            &deposits,
            |seam| {
                seams
                    .get(seam)
                    .ok()
                    .filter(|deposit| deposit.owner.is_none())
                    .map(|deposit| deposit.tile)
            },
        );
        for &seam in industry.seam_entities() {
            if let Ok(mut deposit) = seams.get_mut(seam) {
                deposit.owner = Some(entity);
            }
        }

        commands.entity(entity).insert((growth, industry));
    }
}

/// One city's starting state: the town it was founded with, read back off the map,
/// and the fields that population needs.
///
/// A free function rather than a loop body so that a test can seed one city into a
/// world it made up, with no app and no plan.
#[allow(clippy::too_many_arguments)]
fn seed_city(
    config: &GrowthConfig,
    map: &WorldMap,
    offsets: &[IVec2],
    city: &City,
    entity: Entity,
    sampler: &TerrainSampler,
    taken: &mut HashSet<IVec2>,
    cities: &mut CityMap,
    edits: &mut Vec<TileEdit>,
) -> CityGrowth {
    let mut growth = CityGrowth {
        population: 0.0,
        food: 0.0,
        demand: 0.0,
        capacity: 0.0,
        // Only the sky reads this — a cloud is far wider than a city, so where it is
        // raining is one question per city. What a *field* is worth is asked per tile.
        humidity: sampler.humidity(city.centre.x as f32, city.centre.y as f32),
        static_yield: 0.0,
        frontier: 0,
        claims: Vec::new(),
        town_claims: 0,
        chunks: HashSet::new(),
    };

    // The town it was founded with is whatever `plan_cities` actually stamped, read
    // back off the map. Counting the tiles rather than trusting the radius is what
    // makes a coastal city's clipped disc the smaller town it really is.
    let town_reach = (city.radius as f32 * 1.5).ceil() as i32;
    let town_reach2 = town_reach * town_reach;
    while growth.frontier < offsets.len() {
        let offset = offsets[growth.frontier];
        if offset.length_squared() > town_reach2 {
            break;
        }
        growth.frontier += 1;

        let tile = city.centre + offset;
        if map.tile(tile) == Some(TerrainKind::Town) && taken.insert(tile) {
            growth.note_chunk(entity, tile, cities);
            growth.claims.push(Claim {
                tile,
                base_yield: 0.0,
            });
            growth.town_claims += 1;
        }
    }

    growth.population =
        (growth.town_claims as f32 * config.town_people_per_tile).max(config.min_population);

    let want = growth.wanted_yield(config);
    claim_towards(
        &mut growth,
        entity,
        city,
        want,
        // Founding predates the industry: the city has not been told who works what
        // yet, so the ring it is founded with is the one gh-6 laid.
        1.0,
        usize::MAX,
        map,
        sampler,
        offsets,
        config,
        cities,
        edits,
    );

    growth
}

/// Runs whole steps out of the frame's elapsed time.
fn simulate_cities(
    mut map: ResMut<WorldMap>,
    mut dirty: ResMut<DirtyChunks>,
    mut clock: ResMut<GrowthClock>,
    mut cities: ResMut<CityMap>,
    mut order: Local<Vec<Entity>>,
    mut query: Query<(Entity, &mut City, &mut CityGrowth, &mut CityIndustry)>,
    seams: Query<&Deposit>,
    config: Res<GrowthConfig>,
    industry_config: Res<IndustryConfig>,
    fields: Res<GrowthFields>,
    offsets: Res<ClaimOffsets>,
    estate: Res<EstateOffsets>,
    sky: Option<Res<SkySampler>>,
    plan: Res<WorldPlan>,
    time: Res<Time>,
) {
    if !matches!(*plan, WorldPlan::Done) {
        return;
    }

    let step_seconds = config.step_seconds.max(f32::EPSILON);
    clock.carry_seconds += time.delta_secs();
    let mut steps = (clock.carry_seconds / step_seconds) as u32;
    if steps == 0 {
        return;
    }
    // Clamped rather than carried: the backlog of a stall is dropped, so the
    // simulation cannot spend the rest of the session running flat out to repay it.
    steps = steps.min(config.max_steps_per_frame.max(1));
    // Whatever is left over after the cap is *dropped*, not owed: capping the steps
    // while letting the accumulator keep the debt would leave a machine that hitched
    // once running flat out every frame from then on and never repaying it.
    clock.carry_seconds = (clock.carry_seconds - steps as f32 * step_seconds).min(step_seconds);

    if order.len() != query.iter().len() {
        let mut ordered: Vec<(u32, Entity)> =
            query.iter().map(|(e, city, _, _)| (city.id, e)).collect();
        ordered.sort_unstable();
        *order = ordered.into_iter().map(|(_, e)| e).collect();
    }

    let mut edits = Vec::new();
    for _ in 0..steps {
        clock.step += 1;
        let sweep = (clock.step as usize).checked_rem(order.len()).unwrap_or(0);

        for (index, &entity) in order.iter().enumerate() {
            let Ok((_, mut city, mut growth, mut industry)) = query.get_mut(entity) else {
                continue;
            };

            let centre = city.centre.as_vec2();
            let rain = sky
                .as_ref()
                .map_or(0.0, |sky| sky.rain_at(centre, growth.humidity));
            let swept = index == sweep;

            // Hoisted out of `step_city` so that the industry step below reads the
            // *same* `static_yield` the growth step will. It still runs before
            // anything is stamped, which is the ordering that made it load-bearing.
            if swept {
                growth.resum(&map);
            }

            // One loop, one clock: the industry step, then the growth step. They have
            // to interleave per step rather than per frame, which is why this is a
            // call rather than a system of its own.
            let labour = step_industry(
                &industry_config,
                &config,
                &map,
                estate.offsets(),
                Sky { rain },
                swept,
                &city,
                &growth,
                &mut industry,
                |seam| {
                    seams
                        .get(seam)
                        .ok()
                        .map(|deposit| (deposit.resource, deposit.richness))
                },
            );

            edits.clear();
            step_city(
                &config,
                &map,
                &fields.0,
                &offsets.0,
                Sky { rain },
                labour,
                swept,
                entity,
                &mut city,
                &mut growth,
                &mut cities,
                &mut edits,
            );

            // Applied per city rather than per step: the next city's claims have to
            // see this one's, or two neighbours would both find the same tile free.
            if !edits.is_empty() {
                map.apply_edits(&edits, &mut dirty);
            }
        }
    }
}

/// One city, one step: harvest, grow, resize the town, resize the fields.
///
/// Pure but for the map it reads and the edits it appends. The sky and the [`Labour`]
/// both arrive as arguments — which is what makes every property of the simulation
/// testable with no app and no GPU. `Labour::default()` is gh-6's loop exactly.
///
/// **`resum` is the caller's now, not this function's.** The industry step reads
/// `static_yield` to size its harvest, so it has to see the same number this does;
/// with the sweep in here the two would disagree on exactly the steps where the
/// ledger changed. The ordering that made it load-bearing is unchanged — it still
/// runs before anything is stamped, because this step's edits do not reach the map
/// until the caller applies them and a sweep after them would see every converted
/// tile as stale and drop it.
#[allow(clippy::too_many_arguments)]
fn step_city(
    config: &GrowthConfig,
    map: &WorldMap,
    sampler: &TerrainSampler,
    offsets: &[IVec2],
    sky: Sky,
    labour: Labour,
    swept: bool,
    entity: Entity,
    city: &mut City,
    growth: &mut CityGrowth,
    cities: &mut CityMap,
    edits: &mut Vec<TileEdit>,
) {
    // Reported as itself, so the panel's Harvest row keeps meaning what it meant: the
    // food off the fields, with the hands that worked them and the sky over them.
    growth.food = growth.static_yield * labour.farmer_share * harvest_multiplier(config, sky);
    growth.demand = growth.population * config.food_per_person;
    // The granary's release enters here and nowhere else. It is capped at what demand
    // is short of the harvest, so a full store can only ever stop the ceiling falling
    // — it can never push K above the land.
    growth.capacity =
        (growth.food + labour.granary_release) / config.food_per_person.max(f32::EPSILON);
    growth.population = grow(
        config,
        growth.population,
        growth.capacity,
        rate(config, labour),
    );

    let budget = config.claims_per_step.max(1) as usize;
    resize_town(
        config,
        growth,
        city,
        map,
        sampler,
        budget,
        labour.build_allowance,
        edits,
    );

    // The fields are sized against the rain-free, labour-limited yield. **Neither the
    // rain nor the granary is in this comparison**, and for the same reason: a city
    // that released a ring because its store was full would starve the step the store
    // emptied, and the round trip is lossy twice over — a neighbour can take the
    // freed tile, and re-claiming re-reads the yield from whatever it was restored to.
    let workable = growth.static_yield * labour.farmer_share;
    let want = growth.wanted_yield(config);
    if workable < want {
        // A city that has run out of table and still wants land gets one rescan,
        // and only when its turn comes round: the cursor never goes back on its own,
        // so a tile skipped because a neighbour held it would otherwise be lost for
        // good even after that neighbour gave it up.
        if swept && growth.frontier >= offsets.len() {
            growth.frontier = 0;
        }
        claim_towards(
            growth,
            entity,
            city,
            want,
            labour.farmer_share,
            budget,
            map,
            sampler,
            offsets,
            config,
            cities,
            edits,
        );
    } else if workable > want * (1.0 + config.farm_hysteresis) {
        release_towards(
            growth,
            want,
            labour.farmer_share,
            budget,
            map,
            sampler,
            edits,
        );
    }
}

/// The logistic's rate for this step: the base rate scaled by how well the city is
/// supplied.
///
/// The *scale* rather than the happiness arrives in the [`Labour`], because the knobs
/// that turn one into the other — the swing, the neutral point and the floor — belong
/// to `IndustryConfig`, and this module has no business knowing what a basket is. At
/// a scale of 1 this is `growth_rate` exactly, which is gh-6.
///
/// It may go negative, which is the point — see [`grow`] for what happens then, which
/// is deliberately *not* a negative logistic rate.
fn rate(config: &GrowthConfig, labour: Labour) -> f32 {
    config.growth_rate * labour.growth_scale
}

/// What this step does to a harvest.
///
/// Rain, and nothing else: each field's climate is already baked into its own
/// `base_yield` at the moment it was claimed, so `static_yield` *is* the steady
/// harvest and this only ever adds to it.
pub fn harvest_multiplier(config: &GrowthConfig, sky: Sky) -> f32 {
    1.0 + config.yield_rain_weight * sky.rain
}

/// The least a tile may yield and still be worth breaking.
///
/// Measured against the best ground there is — well-watered grass — so the knob reads
/// as "how much worse than the best land will a city settle for" rather than as a
/// number in yield units that moves whenever the yields do.
fn fertility_floor(config: &GrowthConfig) -> f32 {
    config.min_field_fertility
        * config.farm_yield_grass
        * (config.yield_base_weight + config.yield_humidity_weight)
}

/// The logistic step, in closed form, plus the bleed an unhappy city suffers.
///
/// Closed form rather than `p + r·p·(1 - p/K)` for two reasons. The Euler form
/// oscillates and then diverges once the rate is large, which makes `growth_rate` and
/// `step_seconds` unsafe to tune together; and it divides by K, which is **zero for
/// every city on its first step** — inf, then NaN, and a population of NaN gives a
/// radius of 0 and releases the whole city. Here K is floored instead, so a city with
/// no fields decays smoothly to the minimum rather than falling off a cliff.
///
/// **The unhappiness is a separate decay and not a negative rate**, and that is a
/// correction to how gh-24's spec described it. Feeding a negative `r` to the closed
/// form does not model decline: for `p > K` the logistic's own derivative
/// `r·p·(1 - p/K)` is a negative times a negative, so an unhappy city *grows* — and
/// past `p = K/(1 - e^r)` the denominator goes through zero as well. Flooring the
/// rate fixes neither. So the logistic runs at the non-negative part of the rate and
/// the negative part is applied as an exponential bleed, which is monotone in the
/// rate, total for every population, and exactly gh-6's loop whenever the rate is
/// positive.
fn grow(config: &GrowthConfig, population: f32, capacity: f32, rate: f32) -> f32 {
    let carrying = capacity.max(config.min_population);
    let growth = rate.max(0.0).exp();
    let next = carrying * population * growth / (carrying + population * (growth - 1.0));
    // An unhappy city bleeds people while its fields still feed them.
    (next * rate.min(0.0).exp()).max(config.min_population)
}

impl CityGrowth {
    /// The field yield the city wants: enough for its population plus the headroom
    /// that leaves it a surplus to grow on.
    ///
    /// Compared against `static_yield`, which is the harvest **with the weather taken
    /// out** — that is the whole anti-ratchet guarantee, and here it is structural
    /// rather than arranged: rain is not in either side of the comparison, so no cloud
    /// can make a city give up a field that the next dry step wants back.
    fn wanted_yield(&self, config: &GrowthConfig) -> f32 {
        self.population * (1.0 + config.growth_headroom) * config.food_per_person
    }

    /// Files a new field in the ledger, keeping it ordered by distance.
    ///
    /// Almost always an append — the frontier walks outward, so a new claim is the
    /// farthest thing the city holds. The exception is the rescan after the cursor is
    /// reset, which starts again at the centre and can turn up a tile nearer than
    /// ones already held; without this that one case would quietly unsort the ledger,
    /// and every rule below rests on it being sorted.
    ///
    /// Never inserted into the town's prefix, however near the tile is: the town is
    /// what the city has *built* on, not merely what it owns nearest.
    fn take_claim(&mut self, claim: Claim, centre: IVec2) {
        let distance = (claim.tile - centre).length_squared();
        let at = self.town_claims
            + self.claims[self.town_claims..]
                .partition_point(|held| (held.tile - centre).length_squared() <= distance);

        self.claims.insert(at, claim);
        self.static_yield += claim.base_yield;
    }

    fn note_chunk(&mut self, entity: Entity, tile: IVec2, cities: &mut CityMap) {
        let chunk = chunk_index_of_tile(tile);
        if self.chunks.insert(chunk) {
            cities.insert(chunk, entity);
        }
    }

    /// Recomputes the running sum from the ledger, dropping claims that something
    /// else has taken.
    ///
    /// One city per step, round-robin, which is a few microseconds — where validating
    /// every ledger every step would be 150k tile reads and a megabyte of memmove,
    /// and would falsify the whole reason the sum is kept incrementally. It also
    /// bounds the float drift of thousands of add-and-subtract cycles.
    fn resum(&mut self, map: &WorldMap) {
        let mut town = 0usize;
        let mut sum = 0.0;
        let mut kept = Vec::with_capacity(self.claims.len());

        for (index, claim) in self.claims.iter().enumerate() {
            let expected = if index < self.town_claims {
                TerrainKind::Town
            } else {
                TerrainKind::Farmland
            };
            // Nothing can currently take a claimed tile — the router finishes before
            // the simulation starts. This is what keeps that from being an assumption
            // the moment trade, re-founding or a second writer arrives.
            if map.tile(claim.tile) != Some(expected) {
                continue;
            }
            if index < self.town_claims {
                town += 1;
            } else {
                sum += claim.base_yield;
            }
            kept.push(*claim);
        }

        self.claims = kept;
        self.town_claims = town;
        self.static_yield = sum;
    }
}

#[cfg(test)]
impl CityGrowth {
    /// A ledger of the given shape and nothing else.
    ///
    /// [`crate::gameplay::industry`] reads exactly four numbers off a city —
    /// population, static yield, field count and town count — so this constructor is
    /// the whole of the coupling between the two modules, written down. If it ever
    /// needs a fifth argument, that is the signal that the industry has started
    /// reading the ledger rather than its summary.
    pub(crate) fn for_test(population: f32, static_yield: f32, fields: usize, town: usize) -> Self {
        Self {
            population,
            food: 0.0,
            demand: 0.0,
            capacity: 0.0,
            humidity: 0.0,
            static_yield,
            frontier: 0,
            claims: vec![
                Claim {
                    tile: IVec2::ZERO,
                    base_yield: 0.0,
                };
                town + fields
            ],
            town_claims: town,
            chunks: HashSet::new(),
        }
    }
}

/// Moves the town's edge to where the population puts it.
///
/// The town is driven by *tile count* rather than by radius, which is what keeps a
/// clipped city honest: a coastal town whose disc is half sea holds half as many
/// people, and its radius is then reported from the tiles it actually has rather than
/// from a circle it never filled.
/// How many tiles of town a population of this size wants.
///
/// Pulled out and made public because [`crate::gameplay::industry`] has to spend on
/// the tiles this step is about to add *before* they are added — a city builds out of
/// its stores, so the decision is taken where the stores are. Two answers to "how big
/// should this town be" would let a city pay for tiles it never lays.
pub fn town_target(config: &GrowthConfig, population: f32) -> usize {
    let cap = (std::f32::consts::PI * (MAX_CITY_RADIUS * MAX_CITY_RADIUS) as f32) as usize;
    ((population / config.town_people_per_tile.max(f32::EPSILON)).round() as usize).clamp(1, cap)
}

fn resize_town(
    config: &GrowthConfig,
    growth: &mut CityGrowth,
    city: &mut City,
    map: &WorldMap,
    sampler: &TerrainSampler,
    budget: usize,
    // `allowance` is what the building spend paid for. Growing *inward* is free — a
    // city that gives a town tile back is not building anything — so it bounds only
    // the outward move.
    allowance: usize,
    edits: &mut Vec<TileEdit>,
) {
    let target = town_target(config, growth.population);

    let mut moved = 0;
    let mut built = 0;
    // Outward: the nearest field becomes town. It leaves the ledger's field half, so
    // its yield stops counting — which is the cost of building, and the reason
    // `town_people_per_tile` has to exceed what a field feeds.
    while growth.town_claims < target
        && growth.town_claims < growth.claims.len()
        && moved < budget
        && built < allowance
    {
        built += 1;
        let claim = growth.claims[growth.town_claims];
        growth.static_yield -= claim.base_yield;
        growth.town_claims += 1;
        moved += 1;
        edits.push(TileEdit {
            tile: claim.tile,
            kind: TerrainKind::Town,
        });
    }
    // Inward: the outermost town tile goes back to being worked. The city still owns
    // the land — it has only stopped building on it — so this is a re-stamp and not a
    // release, and the tile keeps its place in the ledger.
    while growth.town_claims > target && growth.town_claims > 0 && moved < budget {
        growth.town_claims -= 1;
        moved += 1;
        let tile = growth.claims[growth.town_claims].tile;
        let ground = restore_kind(map, sampler, tile);
        let yield_ = field_yield(
            config,
            ground,
            sampler.humidity(tile.x as f32, tile.y as f32),
        );
        growth.claims[growth.town_claims].base_yield = yield_;
        growth.static_yield += yield_;
        edits.push(TileEdit {
            tile,
            kind: TerrainKind::Farmland,
        });
    }

    // The radius is now a *report* of the area rather than a shape imposed on it, and
    // the tier follows it — so a hamlet that thrives becomes a borough by growing
    // into one.
    city.radius = ((growth.town_claims as f32 / std::f32::consts::PI)
        .sqrt()
        .round() as u32)
        .clamp(1, MAX_CITY_RADIUS);
    city.size = CitySize::from_radius(city.radius);
}

/// Takes land outward from the centre until the city has the yield it wants.
#[allow(clippy::too_many_arguments)]
fn claim_towards(
    growth: &mut CityGrowth,
    entity: Entity,
    city: &City,
    want: f32,
    // The share of the fields the city has the hands to work. A city that has sent
    // its people to a seam is sized against what those hands can actually bring in.
    share: f32,
    budget: usize,
    map: &WorldMap,
    sampler: &TerrainSampler,
    offsets: &[IVec2],
    config: &GrowthConfig,
    cities: &mut CityMap,
    edits: &mut Vec<TileEdit>,
) {
    let floor = fertility_floor(config);
    let mut taken = 0;
    while growth.static_yield * share < want && taken < budget && growth.frontier < offsets.len() {
        let tile = city.centre + offsets[growth.frontier];
        growth.frontier += 1;

        if !tile_in_world(tile) {
            continue;
        }
        // The one test that makes a claim exclusive: `Farmland` and `Town` are not
        // habitable, so a tile another city holds is refused here without this code
        // knowing that other city exists.
        match map.tile(tile) {
            Some(kind) if kind.is_habitable() => {
                // The climate of this tile, not of the city — which is what makes the
                // fields follow the good ground instead of filling a circle.
                let base_yield =
                    field_yield(config, kind, sampler.humidity(tile.x as f32, tile.y as f32));
                if base_yield < floor {
                    continue;
                }

                growth.note_chunk(entity, tile, cities);
                growth.take_claim(Claim { tile, base_yield }, city.centre);
                taken += 1;
                edits.push(TileEdit {
                    tile,
                    kind: TerrainKind::Farmland,
                });
            }
            _ => continue,
        }
    }
}

/// Gives land back, outermost first — which is what guarantees the released tile has
/// unclaimed country beside it to take its kind from.
fn release_towards(
    growth: &mut CityGrowth,
    want: f32,
    share: f32,
    budget: usize,
    map: &WorldMap,
    sampler: &TerrainSampler,
    edits: &mut Vec<TileEdit>,
) {
    let mut given = 0;
    while growth.claims.len() > growth.town_claims
        && given < budget
        && (growth.static_yield - growth.claims[growth.claims.len() - 1].base_yield) * share >= want
    {
        let claim = growth.claims.pop().expect("checked non-empty");
        growth.static_yield -= claim.base_yield;
        given += 1;
        edits.push(TileEdit {
            tile: claim.tile,
            kind: restore_kind(map, sampler, claim.tile),
        });
    }
}

/// What a field cleared from this ground is worth, before the sky is applied.
///
/// The ground it was cleared from times the climate *at that tile*, which is the only
/// place the climate enters — the harvest multiplier carries rain and nothing else, so
/// nothing is counted twice.
fn field_yield(config: &GrowthConfig, ground: TerrainKind, humidity: f32) -> f32 {
    // Every habitable kind is named. A catch-all would price whatever the next
    // habitable kind turns out to be as woodland without anyone noticing — which is
    // exactly what would have happened to `Scrub`.
    let ground = match ground {
        TerrainKind::Grass => config.farm_yield_grass,
        TerrainKind::Forest => config.farm_yield_forest,
        _ => config.farm_yield_scrub,
    };
    ground * (config.yield_base_weight + config.yield_humidity_weight * humidity)
}

/// What a released tile becomes: the commonest habitable kind around it.
///
/// Voting only among *habitable* kinds is what makes this safe rather than merely
/// plausible. The tile was habitable when it was claimed, so a habitable kind is
/// always a right answer, and no release can put water, rock, river or road where a
/// city had a field however strange its surroundings.
///
/// The fallback is doing real work, not covering an impossible case: a footprint
/// clipped into lobes by a coast or a river can leave a field whose every neighbour
/// is sea. Falling back to the *biome's* own wet kind rather than to a hardcoded
/// `Grass` is what stops a released field on a Highland coast becoming meadow — and
/// every biome's wet kind is habitable, which is what keeps the invariant true.
fn restore_kind(map: &WorldMap, sampler: &TerrainSampler, tile: IVec2) -> TerrainKind {
    for radius in 1..=2 {
        let mut votes: [(TerrainKind, u32); 3] = [
            (TerrainKind::Grass, 0),
            (TerrainKind::Forest, 0),
            (TerrainKind::Scrub, 0),
        ];
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let Some(kind) = map.tile(tile + IVec2::new(dx, dy)) else {
                    continue;
                };
                if let Some(vote) = votes.iter_mut().find(|(candidate, _)| *candidate == kind) {
                    vote.1 += 1;
                }
            }
        }
        // Strictly greater, so a tie goes to the *earlier* entry and the array's own
        // order breaks it: a tile between equal grass and scrub comes back as the
        // better ground. `max_by_key` would take the last instead, which is the
        // poorer one.
        let mut best: Option<(TerrainKind, u32)> = None;
        for (kind, count) in votes {
            if count > 0 && best.is_none_or(|(_, seen)| count > seen) {
                best = Some((kind, count));
            }
        }
        if let Some((kind, _)) = best {
            return kind;
        }
    }

    sampler
        .sample(tile.x as f32, tile.y as f32)
        .cover
        .kinds()
        .lush
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gameplay::biome::Biome;

    /// Near the middle of the world, so a city's reach is never clipped by the edge.
    const CENTRE: IVec2 = IVec2::new(2048, 2048);

    /// One city in a world of the test's choosing, stepped by hand.
    ///
    /// The whole point of the split between [`step_city`] and its system is that this
    /// exists: no app, no GPU, no clock, and the sky arrives as a number.
    struct Sim {
        config: GrowthConfig,
        map: WorldMap,
        dirty: DirtyChunks,
        cities: CityMap,
        offsets: Vec<IVec2>,
        sampler: TerrainSampler,
        city: City,
        growth: CityGrowth,
        /// Injected, exactly as the sky is, and defaulting to "no industry at all" —
        /// which is gh-6's loop. A test that wants to see what sending hands to a
        /// seam costs sets it and steps.
        labour: Labour,
    }

    impl Sim {
        /// A city founded on a disc of `radius` Town tiles, with `around` for country.
        ///
        /// The fertility filter is off here: these are properties of the population
        /// loop, and a floor would make them turn on whatever the real humidity field
        /// happens to be under `CENTRE`.
        /// `the_fields_refuse_ground_that_is_not_worth_breaking` tests the filter.
        fn found(radius: u32, around: TerrainKind) -> Self {
            Self::found_with(
                GrowthConfig {
                    min_field_fertility: 0.0,
                    ..GrowthConfig::default()
                },
                radius,
                around,
            )
        }

        /// The same, with the caller's own knobs.
        fn found_with(config: GrowthConfig, radius: u32, around: TerrainKind) -> Self {
            let map = WorldMap::from_fn(move |tile| {
                if (tile - CENTRE).length_squared() <= (radius * radius) as i32 {
                    TerrainKind::Town
                } else {
                    around
                }
            });
            Self::found_in(map, config, radius)
        }

        fn found_in(mut map: WorldMap, config: GrowthConfig, radius: u32) -> Self {
            let offsets = ClaimOffsets::build(config.farm_max_reach_tiles).0;
            let city = City {
                id: 0,
                centre: CENTRE,
                size: CitySize::from_radius(radius),
                radius,
            };
            let mut dirty = DirtyChunks::default();
            let mut cities = CityMap::default();
            let sampler = TerrainConfig::default().sampler();
            let mut edits = Vec::new();
            let growth = seed_city(
                &config,
                &map,
                &offsets,
                &city,
                Entity::PLACEHOLDER,
                &sampler,
                &mut HashSet::new(),
                &mut cities,
                &mut edits,
            );
            map.apply_edits(&edits, &mut dirty);

            Self {
                config,
                map,
                dirty,
                cities,
                offsets,
                sampler,
                city,
                growth,
                labour: Labour::default(),
            }
        }

        fn step(&mut self, rain: f32) -> Vec<TileEdit> {
            let mut edits = Vec::new();
            // The sweep is what `simulate_cities` hoisted out, so the harness has to
            // do it too — and in the same place, before anything is stamped.
            self.growth.resum(&self.map);
            step_city(
                &self.config,
                &self.map,
                &self.sampler,
                &self.offsets,
                Sky { rain },
                self.labour,
                true,
                Entity::PLACEHOLDER,
                &mut self.city,
                &mut self.growth,
                &mut self.cities,
                &mut edits,
            );
            self.map.apply_edits(&edits, &mut self.dirty);
            edits
        }

        fn run(&mut self, steps: usize, rain: f32) {
            for _ in 0..steps {
                self.step(rain);
            }
        }
    }

    /// Every city in the world has an empty ledger on its first step, so the K of the
    /// logistic is zero for all of them at once. Dividing by it is inf, then NaN, and
    /// a population of NaN gives a radius of 0 and releases the city.
    #[test]
    fn a_city_with_no_fields_falls_to_the_floor_rather_than_to_nan() {
        let config = GrowthConfig::default();
        let mut population = 5000.0;
        for _ in 0..200 {
            population = grow(&config, population, 0.0, config.growth_rate);
            assert!(population.is_finite(), "population went to {population}");
            assert!(population >= config.min_population);
        }
        // Decays onto the floor rather than snapping to it, so a city that loses its
        // land empties out over a while instead of in one step.
        assert!(
            (population - config.min_population).abs() < 1.0,
            "{population}"
        );
    }

    #[test]
    fn population_settles_at_the_capacity_its_fields_support() {
        let config = GrowthConfig::default();
        let capacity = 1200.0;

        let mut from_below = config.min_population;
        let mut from_above = capacity * 3.0;
        for _ in 0..4000 {
            from_below = grow(&config, from_below, capacity, config.growth_rate);
            from_above = grow(&config, from_above, capacity, config.growth_rate);
        }

        assert!((from_below - capacity).abs() < 1.0, "{from_below}");
        assert!((from_above - capacity).abs() < 1.0, "{from_above}");
    }

    /// An unhappy city loses people while its fields still feed them — and that is
    /// what a negative rate has to mean.
    ///
    /// The obvious implementation is a negative `r` in the logistic, and it is wrong
    /// in two ways at once. `r·p·(1 - p/K)` above K is a negative times a negative, so
    /// an unhappy city over its capacity *grows*; and past `p = K/(1 - e^r)` the
    /// closed form's denominator goes through zero, which flooring the rate does not
    /// fix. Hence the bleed is applied outside the logistic, and this is the test that
    /// says so — it runs the case a naive floor would have blown up on.
    #[test]
    fn an_unhappy_city_bleeds_people_whatever_its_capacity_is() {
        let config = GrowthConfig::default();

        for capacity in [0.0f32, 100.0, 5000.0] {
            // Deliberately far above capacity, which is exactly where a negative
            // logistic rate misbehaves.
            let mut population = 20_000.0;
            for _ in 0..200 {
                let next = grow(&config, population, capacity, -0.02);
                assert!(next.is_finite(), "population went to {next}");
                assert!(
                    next <= population + 1e-3,
                    "an unhappy city grew from {population} to {next} at capacity {capacity}"
                );
                population = next;
            }
            assert!(
                population < 20_000.0,
                "the city never lost anyone at capacity {capacity}"
            );
        }
    }

    /// The whole domain, not just the shipped defaults: no rate, population or
    /// capacity may produce a NaN, an infinity or a negative population.
    #[test]
    fn no_rate_takes_the_population_to_nan_or_through_zero() {
        let config = GrowthConfig::default();

        for rate in [-1.0f32, -0.2, -0.02, 0.0, 0.02, 0.5, 2.0] {
            for capacity in [0.0f32, 1.0, 20.0, 5000.0, 1e6] {
                for start in [0.0f32, config.min_population, 1000.0, 1e6] {
                    let mut population = start;
                    for _ in 0..500 {
                        population = grow(&config, population, capacity, rate);
                        assert!(
                            population.is_finite() && population >= config.min_population,
                            "rate {rate}, capacity {capacity}, from {start}: {population}"
                        );
                    }
                }
            }
        }
    }

    /// The acceptance criterion gh-24 turns on, stated as an invariant: **food still
    /// sizes the population**. Nothing that is not food enters the logistic's K, so a
    /// city handed a full warehouse of everything else has exactly the capacity its
    /// fields give it — happiness is on the rate alone.
    #[test]
    fn nothing_but_food_enters_the_capacity() {
        let mut plain = Sim::found(5, TerrainKind::Grass);
        let mut happy = Sim::found(5, TerrainKind::Grass);
        happy.labour = Labour {
            growth_scale: 3.0,
            ..Labour::default()
        };

        plain.step(0.0);
        happy.step(0.0);

        assert_eq!(
            plain.growth.capacity, happy.growth.capacity,
            "happiness moved the capacity"
        );
        assert!(
            happy.growth.population > plain.growth.population,
            "happiness did not move the rate either, so it does nothing at all"
        );
    }

    /// The granary can only ever stop the ceiling *falling*: what it releases enters
    /// the capacity, and a city living off its stores is one whose Supports row
    /// exceeds its Harvest row.
    #[test]
    fn the_granarys_release_holds_the_ceiling_up() {
        let mut fed = Sim::found(5, TerrainKind::Grass);
        let mut living_off_stores = Sim::found(5, TerrainKind::Grass);
        living_off_stores.labour = Labour {
            granary_release: 500.0,
            ..Labour::default()
        };

        fed.step(0.0);
        living_off_stores.step(0.0);

        assert_eq!(
            fed.growth.food, living_off_stores.growth.food,
            "the harvest moved"
        );
        assert!(
            living_off_stores.growth.capacity > fed.growth.capacity,
            "the release did not reach the capacity"
        );
    }

    /// A city that has sent its people elsewhere brings in less food off the same
    /// fields — which is what makes a seam cost something.
    #[test]
    fn fewer_farmers_means_a_smaller_harvest_off_the_same_land() {
        let mut whole = Sim::found(5, TerrainKind::Grass);
        let mut halved = Sim::found(5, TerrainKind::Grass);
        halved.labour = Labour {
            farmer_share: 0.5,
            ..Labour::default()
        };

        whole.step(0.0);
        halved.step(0.0);

        assert!(
            (halved.growth.food - whole.growth.food * 0.5).abs() < 1e-2,
            "{} against half of {}",
            halved.growth.food,
            whole.growth.food
        );

        whole.run(400, 0.0);
        halved.run(400, 0.0);
        assert!(
            halved.growth.population < whole.growth.population,
            "sending half the hands away cost the city nothing: {} against {}",
            halved.growth.population,
            whole.growth.population
        );
    }

    /// A city that cannot afford stone does not stop growing in *people*; it stops
    /// building, and grows crowded instead.
    #[test]
    fn a_city_that_cannot_pay_for_stone_stops_building_not_growing() {
        let mut building = Sim::found(4, TerrainKind::Grass);
        let mut broke = Sim::found(4, TerrainKind::Grass);
        broke.labour = Labour {
            build_allowance: 0,
            ..Labour::default()
        };

        building.run(300, 0.0);
        broke.run(300, 0.0);

        assert!(
            broke.growth.town() < building.growth.town(),
            "the allowance did not stop the town growing: {} against {}",
            broke.growth.town(),
            building.growth.town()
        );
        // But the people are still there — crowded rather than absent.
        assert!(
            broke.growth.population > broke.config.min_population * 2.0,
            "the city emptied out instead of growing crowded"
        );
    }

    /// A town tile is built *on* a field. If it housed fewer people than that field
    /// could feed, growing would cost a city more food than it gained and every city
    /// in the world would oscillate down to the floor.
    #[test]
    fn a_town_tile_houses_more_than_the_field_it_replaces_feeds() {
        let config = GrowthConfig::default();
        let best_field = config.farm_yield_grass.max(config.farm_yield_forest)
            * harvest_multiplier(&config, Sky { rain: 1.0 });
        assert!(
            config.town_people_per_tile > best_field / config.food_per_person,
            "a town tile holds {} people and destroys a field feeding {}",
            config.town_people_per_tile,
            best_field / config.food_per_person
        );
    }

    /// A city that has stood for years already has its fields — it does not spend the
    /// first few hundred steps starving while it lays them.
    #[test]
    fn a_founded_city_starts_with_the_fields_its_people_need() {
        let sim = Sim::found(6, TerrainKind::Grass);

        assert!(sim.growth.town_claims > 90, "{}", sim.growth.town_claims);
        assert!(
            sim.growth.fields() > 0,
            "the city was founded with no fields"
        );
        assert!(
            sim.growth.static_yield >= sim.growth.wanted_yield(&sim.config),
            "the founding ring does not feed the founding population"
        );
    }

    /// The point of the whole feature: how much land a city can reach is what decides
    /// how big it gets.
    #[test]
    fn a_city_with_more_land_around_it_grows_bigger() {
        let mut open = Sim::found(6, TerrainKind::Grass);
        let mut hemmed = Sim::found(6, TerrainKind::Rock);

        open.run(600, 0.0);
        hemmed.run(600, 0.0);

        assert!(
            open.growth.population > hemmed.growth.population * 3.0,
            "open {} against hemmed in {}",
            open.growth.population,
            hemmed.growth.population
        );
        assert!(open.city.radius > hemmed.city.radius);
    }

    /// Growth has to be reversible, and the ledger is what makes it so — without the
    /// ground beneath a claim ever being recorded.
    #[test]
    fn a_footprint_that_grew_can_shrink_back() {
        let mut sim = Sim::found(6, TerrainKind::Grass);
        sim.run(400, 0.0);
        let grown = sim.growth.fields();
        assert!(grown > 0);

        // A famine, imposed rather than simulated: the fields are worth nothing this
        // step and the city has to give them up.
        sim.growth.population = sim.config.min_population;
        for _ in 0..400 {
            let edits = sim.step(0.0);
            for edit in edits {
                assert!(
                    edit.kind != TerrainKind::DeepWater && edit.kind != TerrainKind::ShallowWater,
                    "shrinking put water at {}",
                    edit.tile
                );
            }
        }

        assert!(
            sim.growth.fields() < grown,
            "{} fields before, {} after",
            grown,
            sim.growth.fields()
        );
    }

    /// The invariant the neighbour vote rests on: a released tile was habitable when
    /// it was claimed, so it must be habitable again — whatever it is surrounded by.
    #[test]
    fn a_released_tile_is_always_habitable() {
        let sampler = TerrainConfig::default().sampler();

        // A field on a spit, with nothing but sea and rock around it. The vote finds
        // nothing and the fallback is what answers.
        let hostile = WorldMap::from_fn(|tile| {
            if tile == CENTRE {
                TerrainKind::Farmland
            } else if tile.x % 2 == 0 {
                TerrainKind::DeepWater
            } else {
                TerrainKind::Rock
            }
        });
        assert!(restore_kind(&hostile, &sampler, CENTRE).is_habitable());

        // And where there *is* country, it takes its kind from it rather than from
        // the fallback: a field cut out of a wood goes back to being wood.
        let wooded = WorldMap::from_fn(|tile| {
            if tile == CENTRE {
                TerrainKind::Farmland
            } else {
                TerrainKind::Forest
            }
        });
        assert_eq!(restore_kind(&wooded, &sampler, CENTRE), TerrainKind::Forest);
    }

    /// Every biome's wet kind is habitable, which is what makes it a safe fallback.
    /// A seventh biome whose wet kind was Snow would break `restore_kind` silently.
    #[test]
    fn every_biome_can_restore_a_released_tile() {
        for biome in Biome::ALL {
            assert!(
                biome.kinds().lush.is_habitable(),
                "{biome:?}'s lush kind is not habitable, so a field cannot be restored to it"
            );
        }
    }

    /// Rain must never *cost* a city land.
    ///
    /// This is the anti-ratchet guarantee, and it holds structurally: `static_yield`
    /// is the harvest with the weather taken out, and it is both sides of the
    /// claim/release comparison. If a wet step
    /// could release a ring that the next dry step re-claimed, each flap would be
    /// lossy twice over: the released tile is briefly habitable, so a neighbour can
    /// take it for good, and re-claiming re-reads the yield from whatever the tile
    /// was restored to — which is how a world ratchets its forests into grass.
    ///
    /// Rain does still reach the footprint, but only the long way round: it raises
    /// the population, the population raises what the city wants, and the logistic
    /// integrates it far too slowly for one cloud to matter. What it cannot do is
    /// take anything away.
    #[test]
    fn a_wet_step_never_costs_a_city_a_field() {
        let mut sim = Sim::found(5, TerrainKind::Forest);
        sim.run(400, 0.0);

        let mut held = sim.growth.claims.len();
        for step in 0..200 {
            // Alternating, so the city is rained on and then not — the pattern that
            // would flap a footprint sized against the harvest.
            sim.step(if step % 2 == 0 { 1.0 } else { 0.0 });
            assert!(
                sim.growth.claims.len() >= held,
                "the rain cost the city land: {} then {}",
                held,
                sim.growth.claims.len()
            );
            held = sim.growth.claims.len();
        }
    }

    /// The competition, and the thing no code arranges: `Farmland` is not habitable,
    /// so a claim is refused on a tile a neighbour already holds.
    #[test]
    fn two_neighbouring_cities_never_hold_the_same_tile() {
        let config = GrowthConfig::default();
        let offsets = ClaimOffsets::build(config.farm_max_reach_tiles).0;
        let sampler = TerrainConfig::default().sampler();

        // Close enough that their reaches overlap heavily: 30 tiles apart with a
        // 28-tile reach each.
        let centres = [CENTRE, CENTRE + IVec2::new(30, 0)];
        let mut map = WorldMap::from_fn(move |tile| {
            if centres.iter().any(|c| (tile - *c).length_squared() <= 25) {
                TerrainKind::Town
            } else {
                TerrainKind::Grass
            }
        });

        let mut dirty = DirtyChunks::default();
        let mut cities = CityMap::default();
        let mut taken = HashSet::new();
        let mut state: Vec<(City, CityGrowth)> = Vec::new();

        for (id, centre) in centres.iter().enumerate() {
            let city = City {
                id: id as u32,
                centre: *centre,
                size: CitySize::Village,
                radius: 5,
            };
            let mut edits = Vec::new();
            let growth = seed_city(
                &config,
                &map,
                &offsets,
                &city,
                Entity::PLACEHOLDER,
                &sampler,
                &mut taken,
                &mut cities,
                &mut edits,
            );
            map.apply_edits(&edits, &mut dirty);
            state.push((city, growth));
        }

        for _ in 0..400 {
            for (city, growth) in state.iter_mut() {
                let mut edits = Vec::new();
                growth.resum(&map);
                step_city(
                    &config,
                    &map,
                    &sampler,
                    &offsets,
                    Sky { rain: 0.0 },
                    Labour::default(),
                    true,
                    Entity::PLACEHOLDER,
                    city,
                    growth,
                    &mut cities,
                    &mut edits,
                );
                map.apply_edits(&edits, &mut dirty);
            }
        }

        let first: HashSet<IVec2> = state[0].1.claims.iter().map(|c| c.tile).collect();
        let clash = state[1]
            .1
            .claims
            .iter()
            .find(|claim| first.contains(&claim.tile));
        assert!(clash.is_none(), "both cities hold {clash:?}");

        // And both actually grew into the contested ground, or the test proves
        // nothing about competition.
        assert!(state[0].1.fields() > 100 && state[1].1.fields() > 100);
    }

    /// The fields are not a disc: ground that is not worth breaking is left alone.
    ///
    /// Two identical cities, one on grass and one on forest, at the same place — so
    /// the humidity under them is the same and the only difference is what the ground
    /// is worth. With the floor set between the two, one farms and the other cannot.
    #[test]
    fn the_fields_refuse_ground_that_is_not_worth_breaking() {
        let config = GrowthConfig::default();
        let climate = TerrainConfig::default()
            .sampler()
            .humidity(CENTRE.x as f32, CENTRE.y as f32);
        let grass = field_yield(&config, TerrainKind::Grass, climate);
        let forest = field_yield(&config, TerrainKind::Forest, climate);
        assert!(forest < grass, "the test needs the two to differ");

        // A floor the grass clears and the forest does not.
        let floor = (grass + forest) / 2.0;
        let config = GrowthConfig {
            min_field_fertility: floor
                / (config.farm_yield_grass
                    * (config.yield_base_weight + config.yield_humidity_weight)),
            ..config
        };

        let on_grass = Sim::found_with(config.clone(), 5, TerrainKind::Grass);
        let on_forest = Sim::found_with(config, 5, TerrainKind::Forest);

        assert!(
            on_grass.growth.fields() > 0,
            "nothing was worth farming on good ground"
        );
        assert_eq!(
            on_forest.growth.fields(),
            0,
            "ground below the floor was broken anyway"
        );
    }

    /// Releasing pops the outermost field and the town grows by taking the nearest
    /// one. Both rest on the fields being ordered by distance, and nothing re-sorts
    /// them — a rescan has to file its finds in the right place instead.
    #[test]
    fn the_fields_stay_sorted_by_distance_from_the_centre() {
        let mut sim = Sim::found(6, TerrainKind::Grass);
        sim.run(300, 0.2);

        assert!(sim.growth.town_claims <= sim.growth.claims.len());
        assert!(sim.growth.fields() > 0);

        let distances: Vec<i32> = sim.growth.claims[sim.growth.town_claims..]
            .iter()
            .map(|claim| (claim.tile - sim.city.centre).length_squared())
            .collect();
        assert!(
            distances.windows(2).all(|pair| pair[0] <= pair[1]),
            "the fields are out of order"
        );
    }
}

#[cfg(test)]
mod measurements {
    use super::*;
    use crate::gameplay::{
        biome::Biome,
        city::plan_cities,
        deposit::{Resource, plan_deposits},
        drainage::plan_drainage,
        industry::EstateOffsets,
        plan::WorldPlanConfig,
        river::plan_rivers,
        world::{WORLD_TILES, WorldSnapshot, chunk_index_of_tile},
    };
    use bevy::platform::collections::HashMap;

    /// Where the figures in [`GrowthConfig`]'s doc comments come from.
    ///
    /// The whole world, planned and then simulated — the only test that can say the
    /// defaults grow a world worth looking at, since every other test here works on
    /// terrain it made up. Ignored because it generates all 4096 chunks:
    /// `cargo test --release -- --ignored --nocapture`, and run it *alone*.
    ///
    /// The roads are deliberately not laid: they cost a couple of seconds and the
    /// only thing they change here is a handful of tiles a city may not claim.
    ///
    /// The sky is held at **no rain**, which is what the great majority of the world
    /// is at any moment, and it is also the only way to measure a simulation that has
    /// given up being reproducible.
    #[test]
    #[ignore = "generates the whole 4096x4096 world"]
    fn the_default_config_grows_the_world_into_a_steady_state() {
        let terrain = TerrainConfig::default();
        let plan_config = WorldPlanConfig::default();
        let config = GrowthConfig::default();

        let industry_config = IndustryConfig::default();

        let base = WorldSnapshot::generated(&terrain);
        let river_edits: Vec<TileEdit> = plan_rivers(&terrain, &plan_config, &base)
            .by_chunk
            .iter()
            .flatten()
            .copied()
            .collect();
        let watered = base.with_edits(&river_edits);
        // The drainage stage is not optional now that the industry reads the ground: a
        // wadi moves tiles across the habitable line and onto the salt recipe's list,
        // so a world without it is not the world the game plans against.
        let drain_edits: Vec<TileEdit> = plan_drainage(&terrain, &plan_config, &watered)
            .by_chunk
            .iter()
            .flatten()
            .copied()
            .collect();
        let planned_world = watered.with_edits(&drain_edits);
        let sites = plan_deposits(&terrain, &plan_config, &planned_world);
        let planned = plan_cities(&terrain, &plan_config, &planned_world);

        let mut map = WorldMap::from_fn(|tile| planned_world.tile(tile).expect("inside the world"));
        let mut dirty = DirtyChunks::default();
        for city in &planned {
            map.apply_edits(&city.edits, &mut dirty);
        }

        let sampler = terrain.sampler();
        let offsets = ClaimOffsets::build(config.farm_max_reach_tiles).0;
        let mut cities = CityMap::default();
        let mut taken = HashSet::new();

        // The seams, indexed the way the plan indexes them. The entity values stand
        // for nothing and nothing keys on them — the lookup below answers from a list
        // where the app answers from a query, which is the same shape.
        let mut seam_map = DepositMap::default();
        let mut seam_index: HashMap<Entity, usize> = HashMap::new();
        for (index, site) in sites.iter().enumerate() {
            let entity = Entity::from_raw_u32(index as u32 + 1).expect("nonzero");
            seam_map.insert(chunk_index_of_tile(site.tile), entity);
            seam_index.insert(entity, index);
        }
        let mut seam_owner: Vec<Option<usize>> = vec![None; sites.len()];
        let of_seam = |entity: Entity| seam_index.get(&entity).map(|&index| (index, sites[index]));
        let estate = EstateOffsets::build(industry_config.estate_reach_tiles);

        let started = std::time::Instant::now();
        let mut state: Vec<(City, CityGrowth, CityIndustry, CitySize)> = Vec::new();
        for (order, planned_city) in planned.iter().enumerate() {
            let city = planned_city.city;
            let mut edits = Vec::new();
            let growth = seed_city(
                &config,
                &map,
                &offsets,
                &city,
                Entity::PLACEHOLDER,
                &sampler,
                &mut taken,
                &mut cities,
                &mut edits,
            );
            map.apply_edits(&edits, &mut dirty);

            let industry = seed_industry(
                &industry_config,
                &map,
                estate.offsets(),
                &city,
                &growth,
                &seam_map,
                |entity| {
                    of_seam(entity)
                        .and_then(|(index, site)| seam_owner[index].is_none().then_some(site.tile))
                },
            );
            for &seam in industry.seam_entities() {
                if let Some((index, _)) = of_seam(seam) {
                    seam_owner[index] = Some(order);
                }
            }

            state.push((city, growth, industry, city.size));
        }
        let seeding = started.elapsed();

        let founding_fields: usize = state.iter().map(|(_, g, _, _)| g.fields()).sum();
        println!(
            "\n{} cities seeded in {:.0} ms — {founding_fields} tiles of founding field",
            state.len(),
            seeding.as_secs_f64() * 1000.0
        );
        assert!(!state.is_empty(), "the world has no cities at all");
        assert!(founding_fields > 0, "not one city was founded with a field");

        const STEPS: usize = 2000;
        let started = std::time::Instant::now();
        for step in 0..STEPS {
            let sweep = step % state.len();
            for (index, (city, growth, industry, _)) in state.iter_mut().enumerate() {
                let mut edits = Vec::new();
                if index == sweep {
                    growth.resum(&map);
                }
                let labour = step_industry(
                    &industry_config,
                    &config,
                    &map,
                    estate.offsets(),
                    Sky { rain: 0.0 },
                    index == sweep,
                    city,
                    growth,
                    industry,
                    |entity| of_seam(entity).map(|(_, site)| (site.resource, site.richness)),
                );
                step_city(
                    &config,
                    &map,
                    &sampler,
                    &offsets,
                    Sky { rain: 0.0 },
                    labour,
                    index == sweep,
                    Entity::PLACEHOLDER,
                    city,
                    growth,
                    &mut cities,
                    &mut edits,
                );
                map.apply_edits(&edits, &mut dirty);
            }
        }
        let elapsed = started.elapsed();

        let mut populations: Vec<f32> = state.iter().map(|(_, g, _, _)| g.population).collect();
        populations.sort_by(f32::total_cmp);
        let fields: usize = state.iter().map(|(_, g, _, _)| g.fields()).sum();
        let town: usize = state.iter().map(|(_, g, _, _)| g.town_claims).sum();
        let floored = populations
            .iter()
            .filter(|p| **p <= config.min_population * 1.01)
            .count();
        // How much of its reach a city actually holds. The cursor running out says
        // nothing — every city wants headroom beyond what it has, so every cursor
        // reaches the end. This is the number that separates a city hemmed in by its
        // land from one that has taken everything within reach.
        let mut held: Vec<f64> = state
            .iter()
            .map(|(_, g, _, _)| g.claims.len() as f64 / offsets.len() as f64)
            .collect();
        held.sort_by(f64::total_cmp);
        let world_tiles = (WORLD_TILES.x as u64 * WORLD_TILES.y as u64) as f64;

        println!(
            "{STEPS} steps in {:.2} s — {:.3} ms per step for {} cities",
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() * 1000.0 / STEPS as f64,
            state.len()
        );
        println!(
            "  population   min {:.0}  median {:.0}  max {:.0}   ({floored} at the floor)",
            populations[0],
            populations[populations.len() / 2],
            populations[populations.len() - 1]
        );
        println!(
            "  farmland     {fields} tiles, {:.3}% of the world",
            fields as f64 / world_tiles * 100.0
        );
        println!(
            "  town         {town} tiles, {:.3}% of the world (founded with {})",
            town as f64 / world_tiles * 100.0,
            planned.iter().map(|c| c.edits.len()).sum::<usize>()
        );
        println!(
            "  reach held   min {:.0}%  median {:.0}%  max {:.0}%",
            held[0] * 100.0,
            held[held.len() / 2] * 100.0,
            held[held.len() - 1] * 100.0
        );

        for size in [
            CitySize::Hamlet,
            CitySize::Village,
            CitySize::Borough,
            CitySize::Metropolis,
        ] {
            let founded = state.iter().filter(|(_, _, _, was)| *was == size).count();
            let now = state.iter().filter(|(c, _, _, _)| c.size == size).count();
            let mean: f32 = {
                let of_size: Vec<f32> = state
                    .iter()
                    .filter(|(_, _, _, was)| *was == size)
                    .map(|(_, g, _, _)| g.population)
                    .collect();
                if of_size.is_empty() {
                    0.0
                } else {
                    of_size.iter().sum::<f32>() / of_size.len() as f32
                }
            };
            println!("  {size:?}: founded {founded}, now {now}, mean population {mean:.0}");
        }

        // The point of the whole thing: the world is no longer the one that was
        // planned. If every city sat where it was founded, none of this would be
        // doing anything.
        let moved = state
            .iter()
            .filter(|(city, _, _, was)| city.size != *was)
            .count();
        println!("  {moved} cities are no longer the size they were founded at\n");
        assert!(moved > 0, "not one city changed size in {STEPS} steps");

        // And a default that grew everything to the cap, or starved everything, would
        // be as dead as one that did nothing.
        assert!(
            floored < state.len() / 2,
            "{floored} of {} cities are at the floor",
            state.len()
        );
        // A world that grew every city to the cap, or starved every city to the floor,
        // would be as dead as one where nothing moved. Deliberately *not* the plan
        // test's "every size exists" — which tier the top of the world reaches is a
        // fact about the terrain and about `CitySize::radius`, not about this module,
        // and asserting it here would make an unrelated terrain change fail in
        // growth's tests.
        let tiers = [
            CitySize::Hamlet,
            CitySize::Village,
            CitySize::Borough,
            CitySize::Metropolis,
        ]
        .iter()
        .filter(|size| state.iter().any(|(city, _, _, _)| city.size == **size))
        .count();
        assert!(
            tiers >= 3,
            "the world's cities collapsed into {tiers} tier(s)"
        );

        // **The acceptance criterion gh-24 turns on**, and it is a measurement rather
        // than an assertion about any one city: two cities in different biomes end up
        // holding different resources.
        //
        // Reported by the *dominant biome under the city's centre* rather than by
        // anything the industry knows — nothing in this module has ever learned what
        // a biome is, so the correlation being there at all is the ground showing
        // through the simulation.
        println!("\n  what a city holds, by the region its centre sits in:");
        println!(
            "  {:<10}{:>7}{:>10}{:>9}{:>9}{:>9}{:>9}{:>8}",
            "biome", "cities", "wood", "stone", "iron", "copper", "salt", "happy"
        );
        let mut profiles: Vec<(Biome, [f32; 5])> = Vec::new();
        for biome in Biome::ALL {
            let here: Vec<&(City, CityGrowth, CityIndustry, CitySize)> = state
                .iter()
                .filter(|(city, _, _, _)| {
                    sampler
                        .sample(city.centre.x as f32, city.centre.y as f32)
                        .dominant
                        == biome
                })
                .collect();
            if here.is_empty() {
                continue;
            }

            let mean = |read: &dyn Fn(&CityIndustry) -> f32| {
                here.iter().map(|(_, _, i, _)| read(i)).sum::<f32>() / here.len() as f32
            };
            // Per head, so a region of big cities does not simply out-hold a region of
            // small ones and the columns compare what the *land* gave them.
            let per_head = |resource: Resource| {
                here.iter()
                    .map(|(_, g, i, _)| i.stock(resource) / g.population.max(1.0))
                    .sum::<f32>()
                    / here.len() as f32
            };

            let profile = [
                per_head(Resource::Wood),
                per_head(Resource::Stone),
                per_head(Resource::Iron),
                per_head(Resource::Copper),
                per_head(Resource::Salt),
            ];
            println!(
                "  {:<10}{:>7}{:>10.3}{:>9.3}{:>9.3}{:>9.3}{:>9.3}{:>8.2}",
                format!("{biome:?}"),
                here.len(),
                profile[0],
                profile[1],
                profile[2],
                profile[3],
                profile[4],
                mean(&|industry| industry.happiness()),
            );
            profiles.push((biome, profile));
        }

        let seams_worked: usize = state.iter().map(|(_, _, i, _)| i.seams()).sum();
        let mining = state.iter().filter(|(_, _, i, _)| i.seams() > 0).count();
        println!(
            "  {seams_worked} of {} seams are worked, by {mining} of {} cities",
            sites.len(),
            state.len()
        );

        // **The stability condition**, and the reason `effort` is a share of the land
        // rather than a staffing ratio. Where a city has fewer people than its land
        // offers work, `staffing` is `population / total`, which puts the population
        // back into the capacity and leaves the logistic with no stable equilibrium.
        // The knobs have to keep essentially every city off that branch.
        let stretched = state
            .iter()
            .filter(|(_, g, i, _)| i.total_hands_wanted() > g.population)
            .count();
        println!(
            "  {stretched} of {} cities want more hands than they have",
            state.len()
        );
        assert!(
            stretched * 4 < state.len(),
            "{stretched} of {} cities are labour-stretched — the capacity is \
             proportional to the population for all of them, which is the collapse \
             `effort` exists to prevent",
            state.len()
        );

        // Two regions have to differ, or the whole feature is decoration. Compared as
        // the largest gap in any one resource between any two regions, per head — a
        // world where every region held the same basket would score ~0.
        let mut widest: (f32, &str) = (0.0, "");
        for (index, (a_biome, a)) in profiles.iter().enumerate() {
            for (b_biome, b) in &profiles[index + 1..] {
                for (slot, name) in ["wood", "stone", "iron", "copper", "salt"]
                    .iter()
                    .enumerate()
                {
                    let gap = (a[slot] - b[slot]).abs() / a[slot].max(b[slot]).max(f32::EPSILON);
                    if gap > widest.0 {
                        widest = (gap, name);
                        println!(
                            "    {a_biome:?} against {b_biome:?}: {name} differs by {:.0}%",
                            gap * 100.0
                        );
                    }
                }
            }
        }
        assert!(
            widest.0 > 0.25,
            "no two regions differ by more than {:.0}% in any resource — \
             the cities are interchangeable after all",
            widest.0 * 100.0
        );

        // And it has to move in both directions, or the "simulation" is a ratchet.
        let grew = state
            .iter()
            .filter(|(city, _, _, was)| city.radius > was.radius())
            .count();
        let shrank = state
            .iter()
            .filter(|(city, _, _, was)| city.radius < was.radius())
            .count();
        println!("  {grew} cities grew, {shrank} shrank");
        assert!(grew > 0 && shrank > 0, "{grew} grew and {shrank} shrank");
    }
}
