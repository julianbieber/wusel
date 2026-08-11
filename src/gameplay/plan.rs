//! Laying cities and roads over a world that has already been generated.
//!
//! Nothing here decides a tile from its neighbourhood — that is what
//! [`crate::gameplay::terrain`] does, and why it needs no chunk margin. A city
//! covers a disc and a road spans hundreds of tiles, so both are planned once
//! the *whole* world exists and then written back over it as tile edits.
//!
//! The plan runs while the player is already walking around, so the cities and
//! the roads appear chunk by chunk as each edit lands. It is a state machine over
//! one session: wait for the terrain, lay the seams, plan the cities, then route
//! the roads.
//!
//! The order is not arbitrary. Each stage is stamped into `WorldMap` before the
//! next one is planned, and each takes its snapshot afterwards, so a road sees the
//! towns it must not pave.
//!
//! TODO(jb-doc): that the water used to be the first two stages here and no longer
//! is, why it had to be a stage at all, and what changed so that it need not be.

use crate::gameplay::world::WorldSampler;
use bevy::{
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};

use crate::{
    gameplay::{
        city::{City, CityMap, PlannedCity, plan_cities},
        deposit::{Deposit, DepositMap, chunk_of_deposit, plan_deposits},
        prospect::start_prospect_bake,
        road::{RoadNetwork, RoutedRoad, choose_pairs, route_road},
        terrain::TerrainConfig,
        world::{
            BackgroundGeneration, DirtyChunks, WorldMap, WorldSnapshot, WorldSystems,
            tile_translation,
        },
    },
    screens::Screen,
};

/// Knobs for the plan. Configuration rather than world state, so this is one of
/// the two resources that outlives a session.
#[derive(Resource, Clone)]
pub struct WorldPlanConfig {
    /// The world is cut into squares this wide, each proposing at most one city.
    /// This is the main lever on how many cities the world has.
    pub region_size_tiles: u32,
    /// How far a city's outline may stray from a circle, as a fraction of its
    /// radius. At 0 a city is a drawn disc; much above 0.3 it stops reading as
    /// round at all.
    pub city_wobble: f32,
    pub city_min_gap_tiles: u32,
    pub road_max_distance_tiles: u32,
    /// How far apart two roads must leave the same city, in degrees. Below this
    /// they read as one road drawn twice, so the longer of the two is dropped.
    ///
    /// Measured on the default world, this fires rarely: of 200 pairs the
    /// Gabriel pass produces, 0 are closer than 10 degrees, 1 is closer than 20
    /// and 2 are closer than 30. Above ~45 it starts cutting roads that point
    /// genuinely different ways (13 of them at 45, 24 at 60), which thins the
    /// network rather than deduplicating it — so 30 buys the real duplicates
    /// without costing anything else.
    pub road_min_separation_degrees: f32,
    /// Tiles between lattice nodes when routing. The cost of a route is
    /// quadratic in this, so it is the lever that keeps road planning affordable.
    pub road_node_stride: u32,
    /// What a unit of elevation change costs, in tiles of detour. High enough
    /// that a road will go a long way around a ridge rather than over it.
    pub road_elevation_penalty: f32,
    /// What a step costs when it runs on road that is already there, as a
    /// fraction of what it would cost over open ground. This is what makes two
    /// roads to the same city merge into one rather than run side by side: below
    /// it, a route will detour to pick up an existing road. At 1.0 there is no
    /// discount and every road is routed as if it were the only one.
    ///
    /// Measured on the default world, in tiles of road laid for the same 189
    /// roads: 44572 at 1.0, 42004 at 0.5, 39884 at 0.25, 39760 at 0.1, 39654 at
    /// 0.02. So the merging is all but exhausted by 0.25, and going lower only
    /// flattens the search heuristic — which costs time and buys a tenth of a
    /// percent.
    pub road_reuse_discount: f32,
    /// How far outside the two cities' bounding box a route may detour. This is
    /// what bounds the work per road, and it is measured rather than guessed:
    /// on the default world 48 routed 132 of 200 pairs and 192 routes 158, with
    /// 384 finding nothing more — so below ~192 the box, not the water, is what
    /// is turning pairs down.
    pub route_padding_tiles: u32,
    /// What crossing one tile of river costs a road, in tiles of detour. A road
    /// may cross a river where it may not cross the sea, because a river runs
    /// from the mountains down to the sea and refusing it outright would cut the
    /// continent into pieces the network cannot span.
    ///
    /// A crossing lays a `Road` tile like any other step, so the next route this
    /// way sees road rather than river and pays the reuse discount instead of
    /// this — which is what makes roads converge on the same crossings rather
    /// than each fording the river wherever it happens to meet it.
    pub road_river_crossing_penalty: f32,
    /// The world is cut into squares this wide, each proposing at most one seam.
    ///
    /// With `deposit_threshold` this is the lever on how many mines the world has,
    /// and the yardstick is unlike every other density here: a seam is not measured
    /// as a share of the *map* but as a share of the *cities*, because a seam nobody
    /// can reach is one the simulation never sees.
    ///
    /// **The two populations are anti-correlated, and that is the whole difficulty.**
    /// A city sits on habitable ground and a seam sits on bare, high or dry ground —
    /// `Mountain`, `Rock`, `Gravel`, `Sand`, `Marsh`, none of them habitable — so a
    /// city reaches a seam far less often than a uniform scatter of the same density
    /// would suggest. At `estate_reach_tiles` 56 a city's disc is 9852 tiles of a
    /// 16.7 M-tile world and 520 seams should give one to a quarter of them; the
    /// measured figure is 5%. Reading the density off the map area is the mistake
    /// this note exists to prevent.
    ///
    /// Measured on the default world, as the share of cities holding no seam, exactly
    /// one resource, and two or more (`the_default_config_lays_seams_of_every_resource`):
    ///
    /// ```text
    ///                reach 56          reach 80          reach 112
    ///   cell  seams   0    1   2+     0    1   2+      0    1   2+
    ///    128    110  100%  0%   0%   96%   3%   0%    91%   7%   1%
    ///     64    520   94%  5%   0%   89%  10%   0%    78%  20%   1%
    ///     48    879   93%  4%   2%   82%  15%   2%    72%  20%   6%
    ///     32   2111   90%  4%   5%   77%  11%  10%    60%  23%  15%
    ///     24   3591   83% 13%   3%   75%  15%   9%    58%  28%  13%
    /// ```
    ///
    /// 64 against a reach of 112 is the default, and the column that matters is the
    /// middle one rather than the first: **one city in five works a seam, and nearly
    /// every one of those works a single kind.** Going denser buys more mining cities
    /// and spends the differentiation — at cell 32 fifteen percent of cities hold two
    /// resources or more, and a world where the big cities all have everything is the
    /// homogenisation this feature exists to undo.
    pub deposit_cell_tiles: u32,
    /// How far across a cell a seam's candidate may fall. Below `deposit_cell_tiles`
    /// it insets the candidate, so two neighbouring seams are at least
    /// `deposit_cell_tiles - deposit_jitter_tiles` apart and the layout needs no
    /// spacing pass of its own.
    pub deposit_jitter_tiles: u32,
    /// How well a candidate's ground must suit a recipe before there is a seam
    /// there. What it clears this by is the seam's `richness`, so it is also the
    /// zero of that scale — and it is what
    /// [`crate::gameplay::inspect`]'s prospectivity overlay puts its neutral band
    /// on, read from here rather than restated.
    pub deposit_threshold: f32,
}

impl Default for WorldPlanConfig {
    fn default() -> Self {
        Self {
            region_size_tiles: 128,
            city_wobble: 0.25,
            city_min_gap_tiles: 4,
            road_max_distance_tiles: 256,
            road_min_separation_degrees: 30.0,
            road_node_stride: 8,
            road_elevation_penalty: 400.0,
            road_reuse_discount: 0.25,
            route_padding_tiles: 192,
            road_river_crossing_penalty: 60.0,
            deposit_cell_tiles: 64,
            deposit_jitter_tiles: 48,
            deposit_threshold: 0.5,
        }
    }
}

/// How far the plan has got, and the work still in flight to get it further.
///
/// The tasks live inside the state on purpose: dropping this resource on leaving
/// gameplay drops them with it, so a route planned for one world can never land
/// in the next.
#[derive(Resource)]
pub enum WorldPlan {
    /// TODO(jb-doc): what this waits for now that the water is not one of the stages,
    /// and what the world it is waiting for already contains.
    WaitingForTerrain,
    /// The seams, and now the first stage that plans anything. Before the cities, so
    /// that a later change can let a settlement score read what is under the site;
    /// nothing does yet, and siting is unchanged by this stage.
    ///
    /// It stamps no tile, so there is nothing to spread across frames — the task
    /// lands and every seam is spawned at once.
    Deposits(Task<Vec<Deposit>>),
    Cities(Task<Vec<PlannedCity>>),
    Roads(RoadPlanning),
    Done,
}

/// The road stage's working set.
///
/// Only one route runs at a time. That is the price of letting roads merge: a
/// route has to see the roads already on the ground to reuse them, so it needs a
/// snapshot taken after the last one landed. Routing them all at once against
/// one snapshot would mean none of them could ever see another.
///
/// It costs less than it sounds. A route takes a millisecond or two, so the
/// whole network is laid in a couple of seconds of wall clock — and since it is
/// laid one road per frame while the player is already walking around, the
/// network drawing itself in is something you watch rather than wait for.
pub struct RoadPlanning {
    /// Retaken after each road lands, so the next route can reuse it.
    world: WorldSnapshot,
    queue: Option<RoadQueue>,
    in_flight: Option<Task<Option<RoutedRoad>>>,
}

struct RoadQueue {
    cities: Vec<City>,
    /// Sorted so that `pop` yields the *longest* road first. Order is
    /// load-bearing now that roads reuse each other: the long routes go down
    /// first as trunks, and the short local ones snap onto them, rather than a
    /// long route having to thread its way along a chain of short ones.
    pending: Vec<(usize, usize)>,
}

pub struct WorldPlanPlugin;

impl Plugin for WorldPlanPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldPlanConfig>();
        app.add_systems(OnEnter(Screen::Gameplay), start_plan);
        app.add_systems(OnExit(Screen::Gameplay), tear_down_plan);
        app.add_systems(
            Update,
            (
                start_deposit_plan,
                apply_deposit_plan,
                apply_city_plan,
                drive_road_plan,
            )
                .chain()
                .in_set(WorldSystems::Planning),
        );
    }
}

fn start_plan(mut commands: Commands) {
    commands.insert_resource(WorldPlan::WaitingForTerrain);
    commands.insert_resource(CityMap::default());
    commands.insert_resource(DepositMap::default());
    commands.insert_resource(RoadNetwork::default());
}

/// The seams themselves go with the screen, because every deposit entity carries
/// `DespawnOnExit(Screen::Gameplay)` — the crate's only lifecycle mechanism. All
/// that is dropped here is the index into them, on the same transition, so a seam
/// can never outlive the world it was read off.
fn tear_down_plan(mut commands: Commands) {
    commands.remove_resource::<WorldPlan>();
    commands.remove_resource::<CityMap>();
    commands.remove_resource::<DepositMap>();
    commands.remove_resource::<RoadNetwork>();
}

/// Hands the finished world to the first planning stage, once there is a finished
/// world to hand over. This is what the whole plan waits on.
///
/// TODO(jb-doc): that the world handed over already has its water in it, and why that
/// leaves a seam reading the same ground the drainage stage used to change for it.
fn start_deposit_plan(
    mut plan: ResMut<WorldPlan>,
    map: Res<WorldMap>,
    generation: Res<BackgroundGeneration>,
    sampler: Option<Res<WorldSampler>>,
    config: Res<WorldPlanConfig>,
) {
    if !matches!(*plan, WorldPlan::WaitingForTerrain) || !generation.is_complete() {
        return;
    }

    // A snapshot rather than a copy: the map is ~16 MB and nothing writes it
    // until the plan comes back.
    let world = map
        .snapshot()
        .expect("the background pass reported every chunk generated");
    // TODO(jb-comment): why this `expect` is sound rather than optimistic.
    let sampler = sampler
        .expect("the plan only advances once the terrain is baked")
        .0
        .clone();
    let config = config.clone();

    let task =
        AsyncComputeTaskPool::get().spawn(async move { plan_deposits(&sampler, &config, &world) });
    *plan = WorldPlan::Deposits(task);
}

/// Spawns a seam per site, indexes it, and opens the city stage.
///
/// Nothing is stamped here, so unlike every stage above there is no edit list and no
/// chunk to refresh — a deposit is a record, and the map it was read off is
/// untouched. The city stage therefore plans against exactly the world the drainage
/// stage left behind.
fn apply_deposit_plan(
    mut commands: Commands,
    mut plan: ResMut<WorldPlan>,
    mut deposits: ResMut<DepositMap>,
    map: Res<WorldMap>,
    terrain: Res<TerrainConfig>,
    sampler: Option<Res<WorldSampler>>,
    config: Res<WorldPlanConfig>,
) {
    let WorldPlan::Deposits(task) = &mut *plan else {
        return;
    };
    let Some(planned) = block_on(poll_once(task)) else {
        return;
    };

    // In cell order, which is the order `plan_deposits` produced them in, so the
    // index rows are one fixed order across runs. The `Entity` values are not, and
    // nothing may key on them.
    for deposit in &planned {
        let entity = commands
            .spawn((*deposit, DespawnOnExit(Screen::Gameplay)))
            .id();
        deposits.insert(chunk_of_deposit(deposit), entity);
    }

    // Sound rather than optimistic: the first stage waits on `generation.is_complete()`,
    // and no chunk can generate before the bake has published the sampler. Every later
    // stage is gated on the plan already being past that point.
    let sampler = sampler
        .expect("the plan only advances once the terrain is baked")
        .0
        .clone();

    let world = map
        .snapshot()
        .expect("the world was complete when the plan started");

    // The prospectivity map is baked from **this** snapshot and this seam list, which
    // is the whole reason the bake is opened from here rather than by a system of its
    // own: the map and the seams on it can never disagree about the world they read.
    // It does not gate the plan — the city stage opens below whatever the bake is
    // doing.
    start_prospect_bake(&mut commands, &sampler, world.clone(), planned);

    let terrain = terrain.clone();
    let config = config.clone();
    let task = AsyncComputeTaskPool::get()
        .spawn(async move { plan_cities(&sampler, &terrain, &config, &world) });
    *plan = WorldPlan::Cities(task);
}

/// Takes the finished city plan: stamps the tiles, spawns an entity per city,
/// and opens the road stage.
fn apply_city_plan(
    mut commands: Commands,
    mut plan: ResMut<WorldPlan>,
    mut map: ResMut<WorldMap>,
    mut dirty: ResMut<DirtyChunks>,
    mut cities: ResMut<CityMap>,
) {
    let WorldPlan::Cities(task) = &mut *plan else {
        return;
    };
    let Some(planned) = block_on(poll_once(task)) else {
        return;
    };

    for city in &planned {
        map.apply_edits(&city.edits, &mut dirty);

        let entity = commands
            .spawn((
                city.city,
                Transform::from_translation(tile_translation(city.city.centre).extend(1.0)),
                DespawnOnExit(Screen::Gameplay),
            ))
            .id();
        for chunk in city.chunks() {
            cities.insert(chunk, entity);
        }
    }

    // Taken after the stamping, so a route can see the Town tiles it must not
    // pave over.
    let world = map
        .snapshot()
        .expect("the world was complete when the plan started");
    *plan = WorldPlan::Roads(RoadPlanning {
        world,
        queue: None,
        in_flight: None,
    });
}

/// Keeps routes in flight and writes each one's tiles as it lands.
fn drive_road_plan(
    mut plan: ResMut<WorldPlan>,
    mut map: ResMut<WorldMap>,
    mut dirty: ResMut<DirtyChunks>,
    mut network: ResMut<RoadNetwork>,
    terrain: Res<TerrainConfig>,
    sampler: Option<Res<WorldSampler>>,
    config: Res<WorldPlanConfig>,
    cities: Query<&City>,
) {
    let WorldPlan::Roads(state) = &mut *plan else {
        return;
    };

    let queue = state.queue.get_or_insert_with(|| {
        // Query iteration order is not guaranteed, and the same seed has to give
        // the same roads, so the cities go back into id order before pairing.
        let mut cities: Vec<City> = cities.iter().copied().collect();
        cities.sort_unstable_by_key(|city| city.id);
        let pending = choose_pairs(&cities, &config);
        RoadQueue { cities, pending }
    });

    if let Some(task) = &mut state.in_flight {
        let Some(routed) = block_on(poll_once(task)) else {
            return;
        };
        state.in_flight = None;

        // A pair with no route is simply left unconnected: water separates them,
        // and an island is allowed to have no roads.
        if let Some(road) = routed {
            map.apply_edits(&road.edits, &mut dirty);
            network.links.push(road.link);
            // Retaken so the next route can see this road and merge onto it.
            state.world = map
                .snapshot()
                .expect("the world was complete when the plan started");
        }
    }

    let Some((from, to)) = queue.pending.pop() else {
        *plan = WorldPlan::Done;
        return;
    };

    let (from, to) = (queue.cities[from], queue.cities[to]);
    let world = state.world.clone();
    let terrain = terrain.clone();
    // Sound rather than optimistic: the first stage waits on `generation.is_complete()`,
    // and no chunk can generate before the bake has published the sampler. Every later
    // stage is gated on the plan already being past that point.
    let sampler = sampler
        .expect("the plan only advances once the terrain is baked")
        .0
        .clone();
    let config = config.clone();
    state.in_flight = Some(
        AsyncComputeTaskPool::get()
            .spawn(async move { route_road(&sampler, &terrain, &config, &world, &from, &to) }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::gameplay::terrain::shared_test_sampler;
    use crate::gameplay::world::WORLD_TILES;
    use crate::gameplay::{city::CitySize, terrain::TerrainKind};

    /// The whole plan, against the world the game actually generates.
    ///
    /// This is the only test that can say the defaults produce a world worth
    /// looking at — every other test here works on terrain it made up. It
    /// generates all 4096 chunks, so it is ignored by default: run it with
    /// `cargo test -- --ignored --nocapture` after touching the planner.
    #[test]
    #[ignore = "generates the whole 4096x4096 world"]
    fn the_default_config_lays_out_cities_of_every_size_and_roads_between_them() {
        let terrain = TerrainConfig::default();
        let config = WorldPlanConfig::default();
        // TODO(jb-comment): why the cities are planned against the generated world
        // directly, and that this is not the plan skipping a step.
        let world = WorldSnapshot::generated(&terrain, shared_test_sampler());
        report_water(&world);

        let planned = plan_cities(shared_test_sampler(), &terrain, &config, &world);
        let cities: Vec<City> = planned.iter().map(|p| p.city).collect();
        println!("{} cities", cities.len());
        assert!(!cities.is_empty(), "the world has no cities at all");

        // TODO(jb-comment): why a tier count alone cannot diagnose a missing tier, and
        // what this reports instead.
        let cut = |excess: f32| terrain.town_threshold + excess * (1.0 - terrain.town_threshold);
        let mut best = 0.0f32;
        for y in (0..WORLD_TILES.y).step_by(16) {
            for x in (0..WORLD_TILES.x).step_by(16) {
                best = best.max(shared_test_sampler().settlement(x as f32, y as f32));
            }
        }
        println!(
            "the settlement field peaks at {best:.4}; with the {:.2} coast bonus that is \
             {:.4}, against a Borough cut of {:.4} and a Metropolis cut of {:.4}",
            terrain.town_coast_bonus,
            best + terrain.town_coast_bonus,
            cut(0.55),
            cut(0.78),
        );

        for size in [
            CitySize::Hamlet,
            CitySize::Village,
            CitySize::Borough,
            CitySize::Metropolis,
        ] {
            let count = cities.iter().filter(|c| c.size == size).count();
            println!("  {size:?}: {count}");
            assert!(count > 0, "no {size:?} anywhere in the world");
        }

        for (index, a) in cities.iter().enumerate() {
            for b in &cities[index + 1..] {
                let gap = a.centre.as_vec2().distance(b.centre.as_vec2());
                assert!(
                    gap >= (a.radius + b.radius) as f32,
                    "cities {} and {} overlap",
                    a.id,
                    b.id
                );
            }
        }

        for city in &planned {
            for edit in &city.edits {
                assert!(
                    world
                        .tile(edit.tile)
                        .expect("inside the world")
                        .is_habitable(),
                    "city {} paved {} ",
                    city.city.id,
                    edit.tile
                );
                assert!(
                    edit.tile.as_vec2().distance(city.city.centre.as_vec2())
                        <= city.city.radius as f32 * (1.0 + config.city_wobble),
                    "city {} claimed a tile outside its disc",
                    city.city.id
                );
            }
        }

        let pairs = choose_pairs(&cities, &config);
        println!("{} city pairs to route", pairs.len());
        assert!(!pairs.is_empty(), "no two cities are close enough to link");

        // Pruning a road must never be what cuts a city off the map.
        for city in &cities {
            let reachable = cities.iter().any(|other| {
                other.id != city.id
                    && other.centre.as_vec2().distance(city.centre.as_vec2())
                        <= config.road_max_distance_tiles as f32
            });
            let linked = pairs
                .iter()
                .any(|&(a, b)| cities[a].id == city.id || cities[b].id == city.id);
            assert!(
                linked || !reachable,
                "city {} has a neighbour in range but no road",
                city.id
            );
        }

        let network = lay_roads(&terrain, &config, &world, &cities, &pairs);
        println!(
            "{} of them got a road, {} tiles of road in all, bridging a river {} times",
            network.roads, network.tiles, network.crossings
        );
        assert!(network.roads > 0, "not one pair could be joined over land");

        // Reuse is the whole point of the discount: without it every road is
        // routed as if it were alone, and the same network costs more tiles
        // because roads run alongside each other instead of merging.
        let alone = lay_roads(
            &terrain,
            &WorldPlanConfig {
                road_reuse_discount: 1.0,
                ..config.clone()
            },
            &world,
            &cities,
            &pairs,
        );
        println!(
            "{} tiles if no road may reuse another — {} more",
            alone.tiles,
            alone.tiles - network.tiles
        );
        assert!(
            network.tiles < alone.tiles,
            "the reuse discount saved nothing"
        );
    }

    /// What the flow field left in the generated world, and the checks that only
    /// mean anything against the world the game actually generates.
    ///
    /// TODO(jb-doc): why this counts tiles where its predecessor counted edits.
    fn report_water(world: &WorldSnapshot) {
        let mut river = 0usize;
        let mut water = 0usize;
        let mut tiles = 0usize;
        for y in 0..WORLD_TILES.y as i32 {
            for x in 0..WORLD_TILES.x as i32 {
                let kind = world
                    .tile(IVec2::new(x, y))
                    .expect("the whole world is inside the world");
                tiles += 1;
                match kind {
                    TerrainKind::River => river += 1,
                    kind if kind.is_water() => water += 1,
                    _ => {}
                }
            }
        }
        let share = |count: usize| 100.0 * count as f64 / tiles as f64;
        println!(
            "{river} tiles of river ({:.3}% of the world) against {water} of every other \
             water ({:.2}%)",
            share(river),
            share(water),
        );
        assert!(river > 0, "the world has no rivers at all");
    }

    struct Network {
        roads: usize,
        tiles: usize,
        /// Tiles of road laid over a river — every one of them a bridge, and the
        /// only place `road_river_crossing_penalty` shows up in the result.
        crossings: usize,
    }

    /// Walks the road stage the way the driver does — one road at a time, each
    /// routed against the world the last one left behind — and reports how much
    /// road it took. Asserts the water rule on every road as it goes.
    fn lay_roads(
        terrain: &TerrainConfig,
        config: &WorldPlanConfig,
        world: &WorldSnapshot,
        cities: &[City],
        pairs: &[(usize, usize)],
    ) -> Network {
        let mut world = world.clone();
        let mut network = Network {
            roads: 0,
            tiles: 0,
            crossings: 0,
        };

        // From the back, which is the longest first — the same order the queue
        // is popped in, and the order that decides which roads become trunks.
        for &(a, b) in pairs.iter().rev() {
            let Some(road) = route_road(
                shared_test_sampler(),
                terrain,
                config,
                &world,
                &cities[a],
                &cities[b],
            ) else {
                continue;
            };
            for edit in &road.edits {
                let under = world.tile(edit.tile).expect("inside the world");
                // The sea and a lake are refused outright; a river is bridged.
                assert!(
                    !under.is_water(),
                    "road {}-{} crosses water at {}",
                    road.link.from,
                    road.link.to,
                    edit.tile
                );
                if under == TerrainKind::River {
                    network.crossings += 1;
                }
            }
            network.roads += 1;
            network.tiles += road.edits.len();
            world = world.with_edits(&road.edits);
        }

        network
    }
}
