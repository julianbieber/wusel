//! Laying cities and roads over a world that has already been generated.
//!
//! Nothing here decides a tile from its neighbourhood — that is what
//! [`crate::gameplay::terrain`] does, and why it needs no chunk margin. A city
//! covers a disc and a road spans hundreds of tiles, so both are planned once
//! the *whole* world exists and then written back over it as tile edits.
//!
//! The plan runs while the player is already walking around, so rivers, cities
//! and roads appear chunk by chunk as each edit lands. It is a state machine
//! over one session: wait for the terrain, cut the rivers, plan the cities, then
//! route the roads.
//!
//! The order is not arbitrary. Each stage is stamped into `WorldMap` before the
//! next one is planned, and each takes its snapshot afterwards, so a city sees
//! the rivers it must not pave and a road sees both.

use crate::gameplay::world::WorldSampler;
use bevy::{
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};

use crate::{
    gameplay::{
        city::{City, CityMap, PlannedCity, plan_cities},
        deposit::{Deposit, DepositMap, chunk_of_deposit, plan_deposits},
        drainage::{DrainagePlan, plan_drainage},
        prospect::start_prospect_bake,
        river::{RiverPlan, plan_rivers},
        road::{RoadNetwork, RoutedRoad, choose_pairs, route_road},
        terrain::TerrainConfig,
        world::{
            BackgroundGeneration, DirtyChunks, TileEdit, WorldMap, WorldSnapshot, WorldSystems,
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
    /// The world is cut into squares this wide, each proposing at most one
    /// spring. With `TerrainConfig::river_source_threshold` this is the lever on
    /// how many rivers the world has, and it has to be pushed harder than it
    /// looks: the elevation field's longest wavelength is about 25 tiles, so a
    /// descent reaches water within a few steps and one spring per mountain
    /// leaves the map bare.
    ///
    /// Measured on the default world, in share of tiles that end up river: 48
    /// gives 0.011%, 24 gives 0.042%, 16 gives 0.090%, 12 gives 0.229% and 8
    /// gives 0.360%. Roads cover 0.24%, which is the mark for "reads as a
    /// feature of the map rather than as speckle".
    pub river_source_cell_tiles: u32,
    /// Tiles between lattice nodes when a particle descends. Anchored on the
    /// world origin, so this is also the resolution at which two rivers merge
    /// instead of running alongside each other.
    pub river_step_tiles: u32,
    /// How far a particle may walk before it is abandoned. A backstop against a
    /// descent the terrain has talked into wandering, not a shape control.
    pub river_max_steps: u32,
    /// How many particles must cross a tile for its channel to widen by one.
    ///
    /// Low, and it has to be. Flow only accumulates where two descents meet, and
    /// on this terrain they hardly ever do: at the default spacing the busiest
    /// segment in the whole world carries 3 particles, so anything above 2 would
    /// mean every river in the world came out one tile wide. Of ~11k segments,
    /// 2 gives about 350 that are two tiles across and none wider — the width
    /// machinery is right, but the landscape rarely feeds it. Long rivers with a
    /// real hierarchy of tributaries would need an elevation field with
    /// something longer than a 25-tile wavelength in it.
    pub river_flow_per_width: u32,
    /// The drop per tile at which the terrain outvotes the shape rules.
    ///
    /// The first of six knobs that are **shape** controls, which is a category
    /// this config did not have: until now a river's course was whatever
    /// steepest descent produced, and every knob here was a density, a budget or
    /// a backstop.
    ///
    /// A step's descent is scored against this rather than against the best
    /// available step, and that choice is the whole reason the rule behaves
    /// differently in different country. On a steep slope the drops are large
    /// multiples of it, descent swamps the other two terms, and the river runs
    /// near the fall line — which is what a river in steep ground does. On
    /// gentle ground every drop is a fraction of it, the term goes quiet, and
    /// the heading and meander terms decide. Score against the best candidate
    /// instead and the terms keep the same proportions everywhere, so a
    /// mountain torrent meanders exactly as hard as a lowland one.
    ///
    /// So it has to be read off the terrain rather than picked. The drop per tile
    /// along a course on the default world runs 0.0008 at p10, 0.0049 at the
    /// median and 0.0123 at p90, and the default is that p90 — nine steps in ten
    /// are gentle enough for the shape terms to have a say, and the steepest
    /// tenth is left to the hill. Measured against mean excursion, which is the
    /// number `the_shape_of_the_worlds_rivers` exists to print: 0.002 gives
    /// 0.202, 0.004 gives 0.205, 0.012 gives 0.228, 0.020 gives 0.249 and 0.040
    /// gives 0.266. It keeps paying past p90 — but past there the descent term is
    /// a rounding error, and a river that ignores the ground it is on except to
    /// avoid climbing is not a river.
    pub river_reference_drop: f32,
    /// What continuing in the same direction is worth, against a drop of
    /// `river_reference_drop`.
    ///
    /// This is the term that kills the staircase. Where the true fall direction
    /// falls between two of the eight lattice directions, steepest descent flips
    /// between them every node and lays a zigzag with 4-tile teeth; a particle
    /// that pays to turn picks one and holds it.
    pub river_heading_weight: f32,
    /// What leaning to the side the meander field points is worth, on the same
    /// scale as `river_heading_weight`.
    ///
    /// Against the heading term this is the sinuosity dial: heading alone gives
    /// straighter rivers than steepest descent, meander alone gives a course
    /// that wanders without committing, and the ratio between them is what makes
    /// a bend a bend.
    pub river_meander_weight: f32,
    /// Noise scale of the meander field, so 1 / this is the wavelength in tiles
    /// over which the water changes which way it leans — half a wavelength is
    /// one bend.
    ///
    /// The default is ~50 tiles, which is about a screen at zoom 1: a bend you
    /// can see the whole of without it reading as a wobble in a straight line.
    pub river_meander_scale: f32,
    /// How many level steps in a row are wandering rather than pooling.
    ///
    /// A particle may now take a step that does not descend, which is what lets
    /// a river meander across a flood plain instead of flooding it — but only
    /// this many in a row, or a broad flat would swallow the course entirely and
    /// leave it stopping in the middle of nowhere at the step cap. Past this the
    /// water is declared to be standing and the basin is flooded from where the
    /// particle stands, which is the old behaviour arrived at late.
    pub river_flat_run_nodes: u32,
    /// Fewest points a channel segment's curve is sampled at.
    ///
    /// A floor, not a count: the sampling is dense enough to leave no gaps on
    /// its own, and this only matters for a segment short enough that the
    /// gap-free density would be one or two points.
    pub river_curve_samples: u32,
    /// A basin that spills before it holds this much water leaves no lake at
    /// all. Load-bearing: fbm at this stride is full of dips a tile or two deep,
    /// and a pond at every one of them would turn each river into a string of
    /// beads.
    ///
    /// **Since `Lattice::spill` this is also the lever on how long a river is**,
    /// and it is by a distance the strongest one. A basin under it is not merely
    /// undrawn — the channel is drawn straight across it — so raising this
    /// converts lakes into crossings, and every lake converted is one that is no
    /// longer chopping a course in two. Measured on the default world:
    ///
    /// ```text
    ///   min   courses>=8   p90 course   excursion   river     lake
    ///    64          453      64 tiles       0.228   36418   234714
    ///   128          560      81 tiles       0.256   40987   218674
    ///   256          680     108 tiles       0.295   46842   190722
    ///   512          779     151 tiles       0.358   57167   127460
    /// ```
    ///
    /// It keeps paying, and past 512 it runs out of road: no basin can exceed
    /// `river_lake_max_tiles`, so above that nothing is ever drawn and the world
    /// has no inland water at all. 256 is the default because river coverage
    /// lands at 0.28% against roads' 0.24% — the yardstick every other density
    /// here was chosen against — where 512 gives 0.34%.
    ///
    /// What it costs is honesty about the water: a basin under this is *filled*,
    /// so the river crosses standing water that is not drawn. At 256 that is a
    /// hollow up to 16 tiles across, which reads as a river running over a damp
    /// flat; at 512 it is 23 and starting to be a pond that is missing.
    pub river_lake_min_tiles: u32,
    /// A basin that has not found a way out by this size is a closed lake, and
    /// the river feeding it ends there. This is what stops one unlucky basin
    /// from flooding half a continent.
    pub river_lake_max_tiles: u32,
    /// How many chunks of river are stamped into the world per frame. Doing the
    /// whole world in one frame would be a visible stall; spread out, it is the
    /// rivers filling in across the map, which is worth watching.
    pub river_chunks_stamped_per_frame: u32,
    /// The world is cut into squares this wide, each proposing at most one valley
    /// head. Coarser than the river spacing on purpose: these are the trunk
    /// valleys a landscape reads by, not every rill in it.
    ///
    /// With `drain_min_flow`, the lever on how much of the network you see.
    /// Measured on the default world as a share of tiles, at a step of 16:
    ///
    /// ```text
    ///   cell   floor 2   floor 3
    ///     48    0.063%    0.014%
    ///     32    0.227%    0.081%
    ///     24    0.452%    0.199%
    ///     16    1.224%    0.633%
    /// ```
    ///
    /// 24 against a floor of 3 gives 0.199%, next to roads at 0.24% — which is the
    /// same yardstick `river_source_cell_tiles` was chosen against. Note the two
    /// columns are different pictures at the same density and not a free choice:
    /// tightening the cell adds *heads* and so lengthens the branching network,
    /// while dropping the floor draws paths that fewer descents agreed on, which
    /// adds isolated rills. See `the_drainage_density_against_its_two_knobs`.
    pub drain_source_cell_tiles: u32,
    /// Tiles between lattice nodes when a drainage particle descends.
    ///
    /// Four times the river stride, and that is what makes the stage work at all.
    /// A drainage particle never floods, so it stops at the first node with nothing
    /// lower beside it — and at the river's 4-tile stride the relief layer's own
    /// fine octaves put a local minimum every few nodes, so the first cut of this
    /// laid **404 tiles in the entire world** because no two descents ever met. The
    /// pits are a property of the sampling scale, not of the landscape: at 16 tiles
    /// the walk sees the broad fall of the ground, particles run for hundreds of
    /// tiles, and their paths coincide often enough for flow to mean something.
    ///
    /// The precision is not missed. A valley is a broad feature, and the lattice is
    /// still anchored on the world origin, which is the property that actually
    /// matters — two particles crossing the same ground step between the same
    /// nodes. It is also 16x less scratch than the river lattice.
    pub drain_step_tiles: u32,
    /// How many particles must agree on a node before it is drawn at all.
    ///
    /// The floor is what stops this reintroducing the speckle gh-14 is about: every
    /// valley head walks a path, and drawing all of them would put a one-tile
    /// squiggle through every square of the world. Only where descents *converge*
    /// is there a valley worth seeing.
    pub drain_min_flow: u32,
    /// How wide a heavily used channel gets. Small — this is a treeline, not a
    /// river, and a wide one would read as a road.
    pub drain_max_width: u32,
    /// How far a particle may walk before it is abandoned. A backstop only: every
    /// step is strictly downhill, so a walk terminates on the terrain long before
    /// this.
    pub drain_max_steps: u32,
    /// How many chunks of dry valley are stamped per frame, on the same terms as
    /// the rivers'.
    pub drain_chunks_stamped_per_frame: u32,
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
            river_source_cell_tiles: 12,
            river_step_tiles: 4,
            river_max_steps: 2048,
            river_flow_per_width: 2,
            river_reference_drop: 0.012,
            river_heading_weight: 0.6,
            river_meander_weight: 1.0,
            river_meander_scale: 0.02,
            river_flat_run_nodes: 24,
            river_curve_samples: 8,
            river_lake_min_tiles: 256,
            river_lake_max_tiles: 512,
            river_chunks_stamped_per_frame: 64,
            drain_source_cell_tiles: 24,
            drain_step_tiles: 16,
            drain_min_flow: 3,
            drain_max_width: 2,
            drain_max_steps: 512,
            drain_chunks_stamped_per_frame: 64,
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
    WaitingForTerrain,
    Rivers(RiverStamping),
    /// The dry valleys, between the rivers and the cities. It has to be after the
    /// rivers so a channel ends where the water starts rather than crossing it, and
    /// before the cities because it moves tiles onto and off the habitable list —
    /// a wadi through a desert lays down settleable ground, and a city stage that
    /// had already run would never see it.
    Drainage(DrainageStamping),
    /// The seams, between the drainage and the cities. After the drainage, because a
    /// wadi changes the ground a salt pan is read off; before the cities, so that a
    /// later change can let a settlement score read what is under the site. Nothing
    /// does yet, and siting is unchanged by this stage.
    ///
    /// It stamps no tile, so unlike the two stages before it there is nothing to
    /// spread across frames — the task lands and every seam is spawned at once.
    Deposits(Task<Vec<Deposit>>),
    Cities(Task<Vec<PlannedCity>>),
    Roads(RoadPlanning),
    Done,
}

/// The river stage's working set.
///
/// One task cuts every river in the world at once — unlike a road, a river needs
/// nothing from the river before it, since the particles share their flow
/// through the lattice rather than through the map. What cannot be done at once
/// is the *stamping*: a world of rivers is on the order of 10^5 edits, so they
/// are written a batch of chunks at a time and the map fills in over about a
/// second of play.
pub struct RiverStamping {
    in_flight: Option<Task<RiverPlan>>,
    /// Chunks still to stamp. Reversed on arrival so that `pop` yields them in
    /// chunk order, and the fill sweeps the world one way rather than jumping
    /// about.
    pending: Vec<Vec<TileEdit>>,
}

/// The drainage stage's working set — the river stage's shape exactly, and for the
/// same reasons: one task cuts every valley in the world at once because the
/// particles share their flow through the lattice rather than through the map, and
/// the stamping is spread because the edit list is large enough that `apply_edits`
/// would be a visible stall in one frame.
pub struct DrainageStamping {
    in_flight: Option<Task<DrainagePlan>>,
    /// Reversed on arrival so that `pop` yields chunks in order and the fill sweeps
    /// the world one way rather than jumping about.
    pending: Vec<Vec<TileEdit>>,
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
                start_river_plan,
                apply_river_plan,
                apply_drainage_plan,
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

/// Hands the finished world to the river planner, once there is a finished world
/// to hand over. This is the first stage, so it is what the whole plan waits on.
fn start_river_plan(
    mut plan: ResMut<WorldPlan>,
    map: Res<WorldMap>,
    generation: Res<BackgroundGeneration>,
    terrain: Res<TerrainConfig>,
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
    let terrain = terrain.clone();
    // Sound rather than optimistic: the first stage waits on `generation.is_complete()`,
    // and no chunk can generate before the bake has published the sampler. Every later
    // stage is gated on the plan already being past that point.
    let sampler = sampler
        .expect("the plan only advances once the terrain is baked")
        .0
        .clone();
    let config = config.clone();

    let task = AsyncComputeTaskPool::get()
        .spawn(async move { plan_rivers(&sampler, &terrain, &config, &world) });
    *plan = WorldPlan::Rivers(RiverStamping {
        in_flight: Some(task),
        pending: Vec::new(),
    });
}

/// Stamps the rivers a batch of chunks at a time, and opens the city stage once
/// the last of them is down.
///
/// The city stage starts from here rather than from a system of its own, because
/// the snapshot it plans against has to be the one *with* the rivers in it — a
/// city must be clipped by a river the same way it is clipped by a coast.
fn apply_river_plan(
    mut plan: ResMut<WorldPlan>,
    mut map: ResMut<WorldMap>,
    mut dirty: ResMut<DirtyChunks>,
    terrain: Res<TerrainConfig>,
    sampler: Option<Res<WorldSampler>>,
    config: Res<WorldPlanConfig>,
) {
    let WorldPlan::Rivers(state) = &mut *plan else {
        return;
    };

    if let Some(task) = &mut state.in_flight {
        let Some(planned) = block_on(poll_once(task)) else {
            return;
        };
        state.in_flight = None;
        state.pending = planned.by_chunk;
        // Popped from the back, so reversing here is what makes the fill sweep
        // the world in chunk order instead of backwards.
        state.pending.reverse();
    }

    for _ in 0..config.river_chunks_stamped_per_frame.max(1) {
        let Some(edits) = state.pending.pop() else {
            break;
        };
        map.apply_edits(&edits, &mut dirty);
    }

    if !state.pending.is_empty() {
        return;
    }

    let world = map
        .snapshot()
        .expect("the world was complete when the plan started");
    let terrain = terrain.clone();
    // Sound rather than optimistic: the first stage waits on `generation.is_complete()`,
    // and no chunk can generate before the bake has published the sampler. Every later
    // stage is gated on the plan already being past that point.
    let sampler = sampler
        .expect("the plan only advances once the terrain is baked")
        .0
        .clone();
    let config = config.clone();
    let task = AsyncComputeTaskPool::get()
        .spawn(async move { plan_drainage(&sampler, &terrain, &config, &world) });
    *plan = WorldPlan::Drainage(DrainageStamping {
        in_flight: Some(task),
        pending: Vec::new(),
    });
}

/// Stamps the dry valleys a batch of chunks at a time, and opens the deposit stage
/// once the last of them is down.
///
/// The next stage starts from here for the reason the city stage used to: the
/// snapshot it reads has to be the one *with* the valleys in it. This stage moves
/// tiles across the habitable line in both directions — a wadi turns desert `Sand`
/// into settleable `Scrub` — so a layout taken before it would be reading a
/// different world from the one on screen. A salt pan is read off exactly the ground
/// a wadi changes, which is why the seams come after the valleys and not before.
fn apply_drainage_plan(
    mut plan: ResMut<WorldPlan>,
    mut map: ResMut<WorldMap>,
    mut dirty: ResMut<DirtyChunks>,
    terrain: Res<TerrainConfig>,
    sampler: Option<Res<WorldSampler>>,
    config: Res<WorldPlanConfig>,
) {
    let WorldPlan::Drainage(state) = &mut *plan else {
        return;
    };

    if let Some(task) = &mut state.in_flight {
        let Some(planned) = block_on(poll_once(task)) else {
            return;
        };
        state.in_flight = None;
        state.pending = planned.by_chunk;
        state.pending.reverse();
    }

    for _ in 0..config.drain_chunks_stamped_per_frame.max(1) {
        let Some(edits) = state.pending.pop() else {
            break;
        };
        map.apply_edits(&edits, &mut dirty);
    }

    if !state.pending.is_empty() {
        return;
    }

    let world = map
        .snapshot()
        .expect("the world was complete when the plan started");
    let _terrain = terrain.clone();
    // Sound rather than optimistic: the first stage waits on `generation.is_complete()`,
    // and no chunk can generate before the bake has published the sampler. Every later
    // stage is gated on the plan already being past that point.
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
        let base = WorldSnapshot::generated(&terrain, shared_test_sampler());

        // The stages in the order the plan runs them: the cities are laid out
        // over a world that already has its rivers, because that is the world
        // the game plans them against.
        let rivers = plan_rivers(shared_test_sampler(), &terrain, &config, &base);
        let river_edits: Vec<TileEdit> = rivers.by_chunk.iter().flatten().copied().collect();
        report_rivers(&base, &rivers, &river_edits);
        let watered = base.with_edits(&river_edits);

        // The drainage stage sits between the rivers and the cities, and it has to
        // be here rather than skipped: it moves tiles across the habitable line, so
        // a city plan taken against `watered` would be planning a different world
        // from the one the game shows.
        let drainage = plan_drainage(shared_test_sampler(), &terrain, &config, &watered);
        let drain_edits: Vec<TileEdit> = drainage.by_chunk.iter().flatten().copied().collect();
        println!(
            "{} tiles of dry valley across {} chunks",
            drain_edits.len(),
            drainage.by_chunk.len()
        );
        assert!(
            !drain_edits.is_empty(),
            "the world has no dry valleys at all"
        );
        for edit in &drain_edits {
            assert!(
                !edit.kind.is_water(),
                "the drainage stage laid {:?} at {}",
                edit.kind,
                edit.tile
            );
        }
        let world = watered.with_edits(&drain_edits);

        let planned = plan_cities(shared_test_sampler(), &terrain, &config, &world);
        let cities: Vec<City> = planned.iter().map(|p| p.city).collect();
        println!("{} cities", cities.len());
        assert!(!cities.is_empty(), "the world has no cities at all");

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

    /// What the river stage produced, and the checks that only mean anything
    /// against the world the game actually generates.
    fn report_rivers(world: &WorldSnapshot, plan: &RiverPlan, edits: &[TileEdit]) {
        let river = edits
            .iter()
            .filter(|edit| edit.kind == TerrainKind::River)
            .count();
        let lake = edits
            .iter()
            .filter(|edit| edit.kind == TerrainKind::ShallowWater)
            .count();
        println!(
            "{river} tiles of river and {lake} of lake, across {} chunks — {} frames of stamping",
            plan.by_chunk.len(),
            plan.by_chunk
                .len()
                .div_ceil(WorldPlanConfig::default().river_chunks_stamped_per_frame as usize),
        );
        assert!(river > 0, "the world has no rivers at all");

        // Water is cut into land: a channel over the sea would run a river
        // through the middle of the ocean it is supposed to end at.
        for edit in edits {
            assert!(
                !world.tile(edit.tile).expect("inside the world").is_water(),
                "{} was already water",
                edit.tile
            );
        }
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
