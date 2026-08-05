//! Who works a city's land, and what becomes of what they take.
//!
//! Four decisions carry the design, each ruling out a simpler one:
//!
//! - **A profession *is* the resource its hands produce.** There is no `Profession`
//!   enum: farmer, woodcutter, quarrier, miner and salter are labels on
//!   [`Resource`], so the two lists cannot drift. The day a refiner arrives — a
//!   smith eating iron and producing tools — the 1:1 breaks and a profession becomes
//!   an enum of its own. Refining is out of scope here.
//! - **The split follows the land, not a policy.** Hands are allocated in proportion
//!   to the work the city's own estate offers, capped by it. So two cities on the
//!   same ground allocate the same hands whatever is in their stores, there is no
//!   target to overshoot, and nothing here can oscillate. The cost is that a city
//!   cannot *react* to a shortage — it can only be short.
//! - **Farming is one of those professions**, which is what gives the split a cost:
//!   a city sitting on iron sends hands to the seam, harvests less, and is smaller
//!   than the same site without the seam. Food still sizes the population — nothing
//!   but food enters the logistic's K — but how much food there is now depends on
//!   who is farming.
//! - **Happiness scales the growth *rate*, never the capacity.** A well-supplied
//!   city fills toward its food ceiling faster; a badly supplied one bleeds people
//!   while its fields still feed them. Keeping it out of K is what keeps gh-6's
//!   "food remains the thing that sizes population" literally true.
//!
//! **Food is a stock like the rest of them, and the granary is what a city eats
//! through a bad step.** gh-6 compared a flow against a flow —
//! `capacity = harvest / food_per_person` — so a city was exactly as big as its
//! worst step. Now the harvest goes into a store, the population eats out of it, and
//! the capacity is measured against what is *on the table*: the harvest, topped up
//! from the granary by as much as the city is short. Two properties make that safe
//! rather than a boom-bust oscillator:
//!
//! - **The store may only ever make up a shortfall, never raise the ceiling.** It
//!   tops the harvest up to what the city already wants and not one grain further,
//!   so a full granary cannot push K above the land and cannot start a cycle. What
//!   it *can* do is stop K falling, which is the whole point of having one.
//! - **The granary is not in the claim/release comparison.** The fields are still
//!   sized against the rain-free, labour-limited `static_yield`, for exactly the
//!   reason rain is kept out of it: a city that released a ring because its store was
//!   full would starve the moment the store emptied, and re-claiming is lossy twice
//!   over.
//!
//! **The test seam is the one gh-6 already built.** `step_city` takes the sky as an
//! argument rather than reading it, which is what makes an irreproducible simulation
//! testable; it now takes a [`Labour`] beside it, on exactly the same terms.
//!
//! **One thing here is not what gh-24 specified, and it is the load-bearing one.**
//! The spec scales the harvest by "hands on the fields over hands the fields want".
//! That ratio contains the population, so the capacity does, so the logistic's K is
//! proportional to its own p and has no stable non-zero equilibrium — the world
//! measured out at a median population of 22 with eighty of ninety-two cities
//! shrinking. [`effort`] is a share of the *land* instead, with the population
//! entering only through a `min` that binds for 15 cities of 92. The headline
//! properties are unchanged: a seam still costs a city its harvest, and the split is
//! still a pure function of the estate — more literally so than before.
//!
//! Defaults carry their measurements, from the `#[ignore]`d
//! `the_default_config_grows_the_world_into_a_steady_state` in
//! [`crate::gameplay::growth`]. At the shipped values the world settles at a median
//! population of **5640** against gh-6's 6940 — the fifth of every city's effort that
//! is no longer farming — with 74132 tiles of farmland, 12889 of town, and a step
//! costing **0.678 ms** for 92 cities against gh-6's 0.53.

use bevy::prelude::*;

use crate::{
    gameplay::{
        city::City,
        deposit::{DepositMap, RESOURCE_COUNT, Resource},
        growth::{CityGrowth, GrowthConfig, Sky, harvest_multiplier, town_target},
        terrain::TerrainKind,
        world::{WORLD_CHUNKS, WORLD_TILES, WorldMap, chunk_of_tile, tile_in_world},
    },
    screens::Screen,
};

/// Knobs for the industry. Configuration rather than world state, so like
/// [`GrowthConfig`] this outlives a session.
#[derive(Resource, Clone)]
pub struct IndustryConfig {
    /// How far a city reaches for a seam and for its wood and stone, in tiles.
    ///
    /// Four times `farm_max_reach_tiles`, and it has to be that much larger. A city
    /// works a mine it would never plough to, and more to the point a seam sits on
    /// ground a city cannot be built on — so the reach is what decides whether the
    /// mining half of this module does anything at all. Measured on the default
    /// world, the share of cities holding at least one seam is 6% at a reach of 56,
    /// 11% at 80 and **22% at 112**; the sweep in `deposit_cell_tiles` has the whole
    /// table.
    ///
    /// It is also what prices the round-robin sweep, quadratically — but the walk is
    /// only `WorldMap::tile` per offset, with no field sampled, so 39408 tiles here
    /// costs well under gh-6's 2464-tile claim rescan, which pays for a humidity
    /// sample on every one of them.
    pub estate_reach_tiles: u32,
    /// Hands one tile of field asks for.
    ///
    /// Against the wood, stone and seam demands below, this is what sets the share of
    /// a city's effort that goes to farming — and so, through [`effort`], how big its
    /// harvest is. Raising it makes the world's cities larger and less specialised;
    /// lowering it hands more of every city's effort to the other five resources.
    ///
    /// **It has a ceiling, and it is a stability condition rather than taste.** A
    /// city's total demand for hands must stay under its population, or `effort`'s
    /// staffing term engages and the capacity becomes proportional to the population
    /// — see [`effort`] for what that does. Fields scale with population, so this is
    /// the term that decides it. At 4 the shipped world has **15 of 92** cities
    /// stretched; the whole-world measurement asserts it stays under a quarter.
    pub hands_per_field: f32,
    /// Hands one tile of wood or stone ground asks for.
    ///
    /// Small, because the disc is large: a city's reach holds thousands of forest and
    /// rock tiles against a few hundred fields, so at anything near `hands_per_field`
    /// the woods would take the whole workforce and the fields would go unworked.
    pub hands_per_tile: f32,
    /// Hands one unit of a seam's richness asks for.
    pub hands_per_richness: f32,
    /// What each of those yields per step when it is fully worked.
    pub wood_per_tile: f32,
    pub stone_per_tile: f32,
    pub yield_per_richness: f32,
    /// Per head per step, of **each** resource in the upkeep and comfort baskets.
    ///
    /// They are also the weights: satisfaction is the two baskets' shares averaged in
    /// proportion to what each demanded, so "how much does comfort matter" is this
    /// pair's ratio rather than a third knob that could disagree with them.
    pub upkeep_per_person: f32,
    pub comfort_per_person: f32,
    /// What one new tile of town costs. A city that cannot afford stone does not stop
    /// growing in *people* — it stops building, and grows crowded instead.
    pub build_cost_wood: f32,
    pub build_cost_stone: f32,
    /// How much of each resource one tile of town can store. The whole cap, so a
    /// hamlet cannot hold a metropolis's hoard and the overflow is simply lost.
    pub store_per_town_tile: f32,
    /// Food's own cap, because grain is bulk and a granary is not a strongroom.
    ///
    /// It is also the knob that says how long a bad spell a city can eat through,
    /// which is `granary_per_town_tile / (town_people_per_tile * food_per_person)`
    /// steps of a *total* crop failure — 10 at the defaults. That is the visible
    /// difference the store exists to make: a city with a full granary shrugs off a
    /// dry spell and one with an empty granary collapses in the same step.
    pub granary_per_town_tile: f32,
    /// How far happiness moves toward the step's satisfaction. Smoothed because a
    /// single dry step must not swing the growth rate.
    pub happiness_inertia: f32,
    /// Where satisfaction stops helping and starts hurting, and how hard.
    ///
    /// The swing has to exceed `1 / happiness_neutral` or no city can ever decline
    /// from unhappiness: below that, `1 + swing * (0 - neutral)` is still positive
    /// even for a city getting nothing at all, and the "bleeds people" half of the
    /// design does not exist. `the_swing_is_hard_enough_for_a_city_to_actually_decline`
    /// is the guard.
    ///
    /// The neutral point is set near the world's *measured* median rather than at a
    /// round number, so the knob reads as "better or worse supplied than usual"
    /// rather than putting every city in the world on one side of it. At the defaults
    /// the shipped world runs Plains 0.28, Ocean 0.31, Desert 0.36, Forest 0.37 and
    /// Wetland 0.42 — so the wooded and the mining regions grow and the open plains
    /// stagnate, which is the ground showing through.
    pub happiness_swing: f32,
    pub happiness_neutral: f32,
    /// The floor under the effective growth rate — how fast the unhappiest possible
    /// city may bleed. A backstop rather than a working knob: at the defaults the
    /// swing bottoms out at -0.0045, well above it.
    pub min_growth_rate: f32,
}

impl Default for IndustryConfig {
    fn default() -> Self {
        Self {
            estate_reach_tiles: 112,
            hands_per_field: 4.0,
            hands_per_tile: 0.05,
            hands_per_richness: 1600.0,
            wood_per_tile: 0.18,
            stone_per_tile: 0.18,
            yield_per_richness: 2500.0,
            upkeep_per_person: 0.02,
            comfort_per_person: 0.01,
            build_cost_wood: 12.0,
            build_cost_stone: 8.0,
            store_per_town_tile: 40.0,
            granary_per_town_tile: 400.0,
            happiness_inertia: 0.05,
            happiness_swing: 3.5,
            happiness_neutral: 0.35,
            min_growth_rate: -0.02,
        }
    }
}

/// What a city needs an iron or a wood *for*.
///
/// A city eats before it keeps working, works before it builds, and builds before it
/// enjoys itself — and that order is the design rather than an implementation
/// detail, because each spend takes what the store has and the one after it gets the
/// remainder.
///
/// Food is deliberately in **neither** basket: it already sizes the population, and
/// counting it twice would have one bad step both shrink a city and embitter it.
const UPKEEP_BASKET: [Resource; 2] = [Resource::Iron, Resource::Wood];
const COMFORT_BASKET: [Resource; 2] = [Resource::Salt, Resource::Copper];

/// Every offset within a city's estate reach, so the sweep walks a table rather than
/// a bounding box. One table shared by every city, built once — the same trick
/// `ClaimOffsets` plays, and for the same reason.
#[derive(Resource)]
pub struct EstateOffsets(Vec<IVec2>);

impl EstateOffsets {
    pub fn offsets(&self) -> &[IVec2] {
        &self.0
    }

    pub(crate) fn build(reach: u32) -> Self {
        let reach = reach as i32;
        Self(
            (-reach..=reach)
                .flat_map(|dy| (-reach..=reach).map(move |dx| IVec2::new(dx, dy)))
                .filter(|offset| offset.length_squared() <= reach * reach)
                .collect(),
        )
    }
}

/// What a city's land offers, measured rather than chosen: for each resource, what it
/// could produce with enough hands, and how many hands that is.
///
/// Both are re-measured on the round-robin sweep and not every step — except `Food`,
/// which is read from the city's own `static_yield` and its field count and so is
/// free.
#[derive(Clone, Copy, Debug, Default)]
pub struct Estate {
    potential: [f32; RESOURCE_COUNT],
    hands_wanted: [f32; RESOURCE_COUNT],
}

/// What a city holds and who is working.
///
/// A second component beside [`CityGrowth`] rather than more fields on it, and the
/// justification is ownership rather than taste: `growth.rs` writes the population and
/// the fields, this writes the stores and the hands, and the two never touch each
/// other's state.
#[derive(Component, Debug)]
pub struct CityIndustry {
    stocks: [f32; RESOURCE_COUNT],
    hands: [f32; RESOURCE_COUNT],
    /// People with nothing to work. They eat and they want comfort like anyone else,
    /// and a city with more people than land has them by construction.
    idle: f32,
    happiness: f32,
    estate: Estate,
    /// The seams this city works, as the entities themselves. Claimed once, never
    /// given back, and the same handle the seam's own `owner` points back with — so
    /// the two directions cannot describe different sets.
    deposits: Vec<Entity>,
    wood_tiles: u32,
    stone_tiles: u32,
}

impl CityIndustry {
    pub fn stock(&self, resource: Resource) -> f32 {
        self.stocks[resource.index()]
    }

    pub fn hands(&self, resource: Resource) -> f32 {
        self.hands[resource.index()]
    }

    pub fn idle(&self) -> f32 {
        self.idle
    }

    pub fn happiness(&self) -> f32 {
        self.happiness
    }

    /// What the city's whole estate asks for in hands. Above the population, the
    /// city is stretched and `effort`'s staffing term bites — which is the branch the
    /// knobs are chosen to keep it off. See `effort`.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn total_hands_wanted(&self) -> f32 {
        self.estate.hands_wanted.iter().sum()
    }

    /// How many seams the city works — what the panel and `observe cities` report
    /// without either of them having to hold an `Entity`.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn seams(&self) -> usize {
        self.deposits.len()
    }

    /// The seams themselves, so the caller that seeded this city can write the other
    /// direction of the claim.
    pub fn seam_entities(&self) -> &[Entity] {
        &self.deposits
    }
}

/// What the industry step hands the growth step.
///
/// The second argument on the seam [`Sky`] opened, and for the same reason: injected
/// rather than read, so every property of the growth loop stays an ordinary unit test.
#[derive(Clone, Copy, Debug)]
pub struct Labour {
    /// Hands on the fields over hands the fields want, on 0..1. The harvest is scaled
    /// by it, and so is the yield the fields are *sized* against — a city that has
    /// sent its people to the seam gives back the fields it can no longer work.
    pub farmer_share: f32,
    /// What the logistic's rate is multiplied by this step.
    ///
    /// The scale rather than the spec's raw `happiness`, and deliberately: the knobs
    /// that turn one into the other are `happiness_swing`, `happiness_neutral` and
    /// `min_growth_rate`, all of which live here, and handing `growth.rs` a raw
    /// happiness would mean handing it this config too. The raw value is on
    /// [`CityIndustry`], which is where the panel reads it from anyway.
    ///
    /// 1.0 is the base rate untouched, which is gh-6.
    pub growth_scale: f32,
    /// What the granary put on the table this step because the harvest fell short.
    /// Zero in a good step, and it enters the capacity and nothing else — in
    /// particular it never reaches the comparison the fields are sized by.
    pub granary_release: f32,
    /// How many tiles of town the building spend paid for this step.
    ///
    /// Not in the spec's field list, and it has to be: "the town may only grow by as
    /// many tiles as the building spend paid for" is a decision taken here — the
    /// stores are here — and applied there, so it needs a channel. Without one the
    /// growth step would either build for free or have to read the stores itself.
    pub build_allowance: usize,
}

impl Default for Labour {
    /// A world with no industry: everyone farms, nobody is unhappy, the granary is
    /// empty and building is unconstrained. This is gh-6's loop exactly, which is
    /// what keeps `population_settles_at_the_capacity_its_fields_support` meaningful.
    fn default() -> Self {
        Self {
            farmer_share: 1.0,
            growth_scale: 1.0,
            granary_release: 0.0,
            build_allowance: usize::MAX,
        }
    }
}

/// What the logistic's rate is multiplied by, from how well the last step went.
///
/// The swing has to exceed `1 / happiness_neutral` for this to reach zero at all —
/// below that a city getting nothing still grows, just slowly, and the "bleeds
/// people" half of the design does not exist.
///
/// Floored so that `growth_rate * scale` cannot fall below `min_growth_rate`, which
/// is what bounds how fast the unhappiest possible city may empty out.
fn growth_scale(config: &IndustryConfig, growth_config: &GrowthConfig, happiness: f32) -> f32 {
    let scale = 1.0 + config.happiness_swing * (happiness - config.happiness_neutral);
    let floor = config.min_growth_rate / growth_config.growth_rate.max(f32::EPSILON);
    scale.max(floor)
}

pub struct IndustryPlugin;

impl Plugin for IndustryPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<IndustryConfig>();
        app.add_systems(OnEnter(Screen::Gameplay), start_industry);
        app.add_systems(OnExit(Screen::Gameplay), tear_down_industry);
    }
}

/// No system of its own beyond these two: the industry step runs inside growth's
/// existing loop, because the two have to interleave per *step* and there is only one
/// clock.
fn start_industry(mut commands: Commands, config: Res<IndustryConfig>) {
    commands.insert_resource(EstateOffsets::build(config.estate_reach_tiles));
}

fn tear_down_industry(mut commands: Commands) {
    commands.remove_resource::<EstateOffsets>();
}

/// One city's opening industry: the seams it claims, the ground it can reach, and a
/// full warehouse.
///
/// The stores open **full**, on the same argument the founding fields are laid
/// unbudgeted under: a city that has stood for years is not starting from an empty
/// warehouse, and starting from one would have every city in the world unable to
/// build for hundreds of steps. It matters more for the granary than for the rest —
/// an empty one on step 1 would put every city in the world one bad step from decline
/// at once.
/// `unclaimed` answers "is this seam still going spare, and where is it" and takes
/// nothing: the caller writes the seam's `owner` from the list this returns, in the
/// same breath, so the two directions of the claim are one act and neither is derived
/// from the other later.
#[allow(clippy::too_many_arguments)]
pub fn seed_industry(
    config: &IndustryConfig,
    map: &WorldMap,
    offsets: &[IVec2],
    city: &City,
    growth: &CityGrowth,
    deposits: &DepositMap,
    unclaimed: impl Fn(Entity) -> Option<IVec2>,
) -> CityIndustry {
    let mut industry = CityIndustry {
        stocks: [0.0; RESOURCE_COUNT],
        hands: [0.0; RESOURCE_COUNT],
        idle: 0.0,
        // Neutral is the honest opening: nothing has been supplied or gone short yet,
        // so the city grows at its base rate until a step says otherwise.
        happiness: config.happiness_neutral,
        estate: Estate::default(),
        deposits: Vec::new(),
        wood_tiles: 0,
        stone_tiles: 0,
    };

    // First come, in id order, and a claim is for the session: a city never gives a
    // seam back, however small it shrinks. Two neighbours competing for one seam is
    // settled once rather than every step.
    let reach = config.estate_reach_tiles as i32;
    for chunk in chunks_within(city.centre, reach) {
        for &seam in deposits.in_chunk(chunk) {
            let Some(tile) = unclaimed(seam) else {
                continue;
            };
            // The chunk rows are over-inclusive — a chunk overlapping the bounding
            // box can hold a seam outside the disc — so a candidate that fails this
            // is expected rather than a bug, exactly as `CityMap`'s rows are.
            if (tile - city.centre).length_squared() <= reach * reach {
                industry.deposits.push(seam);
            }
        }
    }

    count_ground(&mut industry, map, offsets, city.centre);

    let town = growth.town() as f32;
    for resource in Resource::ALL {
        industry.stocks[resource.index()] = town * store_cap_per_tile(config, resource);
    }

    industry
}

/// One city's industry step, run before its growth step.
///
/// Pure but for the map it reads: the sky and the seams' richness arrive as
/// arguments, which is what keeps every property below an ordinary unit test.
#[allow(clippy::too_many_arguments)]
pub fn step_industry(
    config: &IndustryConfig,
    growth_config: &GrowthConfig,
    map: &WorldMap,
    offsets: &[IVec2],
    sky: Sky,
    swept: bool,
    city: &City,
    growth: &CityGrowth,
    industry: &mut CityIndustry,
    // `worked` is what one seam is — its resource and its richness, or `None` if it
    // is no longer there. Through a closure rather than a query so the step stays a
    // free function: the caller reads the `Deposit` components and a test hands it a
    // table.
    worked: impl Fn(Entity) -> Option<(Resource, f32)>,
) -> Labour {
    // The tiles and the seams are re-counted only on the round-robin sweep, which is
    // the same amortisation `resum` already runs on: walking the reach every step for
    // every city is exactly the cost the incremental sum exists to avoid.
    if swept {
        count_ground(industry, map, offsets, city.centre);
    }
    measure_estate(config, industry, growth, &worked);
    allocate_hands(industry, growth.population);

    let output = extraction(industry, growth.population);
    let farmer_share = effort(industry, growth.population, Resource::Food);

    // The harvest is the real one, sky and all — the growth step recomputes exactly
    // this from the same three numbers so its Harvest row keeps meaning what it meant.
    // Computing the granary release against a rain-free harvest instead would have the
    // store make up a shortfall that the rain had already covered.
    let harvest = growth.static_yield() * farmer_share * harvest_multiplier(growth_config, sky);
    let demand = growth.population * growth_config.food_per_person;

    let granary = &mut industry.stocks[Resource::Food.index()];
    let available = *granary + harvest;
    let eaten = demand.min(available);
    // Whatever of that the harvest could not cover, which is zero in any step where
    // the fields fed the city on their own. Capped at the shortfall by construction,
    // so a full store cannot push K above the land and no boom-bust cycle exists.
    let granary_release = (eaten - harvest).max(0.0);
    let granary_cap = growth.town() as f32 * config.granary_per_town_tile;
    *granary = (available - eaten).clamp(0.0, granary_cap);

    // Everything but food, which the granary block above has already settled.
    for resource in Resource::ALL {
        if resource != Resource::Food {
            industry.stocks[resource.index()] += output[resource.index()];
        }
    }

    let upkeep = spend(
        industry,
        &UPKEEP_BASKET,
        config.upkeep_per_person,
        growth.population,
    );
    let build_allowance = build(config, growth_config, industry, growth);
    let comfort = spend(
        industry,
        &COMFORT_BASKET,
        config.comfort_per_person,
        growth.population,
    );

    // Weighted by what each basket demanded, so the two per-person knobs are also the
    // weighting and there is no third number that could disagree with them.
    let upkeep_weight = config.upkeep_per_person * UPKEEP_BASKET.len() as f32;
    let comfort_weight = config.comfort_per_person * COMFORT_BASKET.len() as f32;
    let satisfaction = (upkeep * upkeep_weight + comfort * comfort_weight)
        / (upkeep_weight + comfort_weight).max(f32::EPSILON);
    industry.happiness +=
        (satisfaction - industry.happiness) * config.happiness_inertia.clamp(0.0, 1.0);

    // The overflow is lost rather than owed, so a town that shrinks loses stores it
    // was holding.
    let town = growth.town() as f32;
    for resource in Resource::ALL {
        let cap = town * store_cap_per_tile(config, resource);
        industry.stocks[resource.index()] = industry.stocks[resource.index()].clamp(0.0, cap);
    }

    Labour {
        farmer_share,
        growth_scale: growth_scale(config, growth_config, industry.happiness),
        granary_release,
        build_allowance,
    }
}

/// What one tile of town can hold of a given resource. Food has its own cap because
/// grain is bulk and a granary is not a strongroom.
fn store_cap_per_tile(config: &IndustryConfig, resource: Resource) -> f32 {
    match resource {
        Resource::Food => config.granary_per_town_tile,
        _ => config.store_per_town_tile,
    }
}

/// Counts the wood and stone ground inside the city's reach.
///
/// **Not exclusive, deliberately**: two cities whose reaches overlap both work the
/// same hillside, because standing timber and an outcrop are not consumed by being
/// worked. Nothing is claimed and no tile is marked.
fn count_ground(industry: &mut CityIndustry, map: &WorldMap, offsets: &[IVec2], centre: IVec2) {
    let (mut wood, mut stone) = (0u32, 0u32);
    for offset in offsets {
        let tile = centre + *offset;
        if !tile_in_world(tile) {
            continue;
        }
        match map.tile(tile) {
            Some(TerrainKind::Forest) => wood += 1,
            Some(TerrainKind::Rock | TerrainKind::Mountain | TerrainKind::Gravel) => stone += 1,
            _ => {}
        }
    }
    industry.wood_tiles = wood;
    industry.stone_tiles = stone;
}

/// What the land could produce with enough hands, and how many hands that is.
fn measure_estate(
    config: &IndustryConfig,
    industry: &mut CityIndustry,
    growth: &CityGrowth,
    worked: &impl Fn(Entity) -> Option<(Resource, f32)>,
) {
    let mut estate = Estate::default();

    // Read every step, because the fields move every step and both numbers are
    // already to hand.
    estate.potential[Resource::Food.index()] = growth.static_yield();
    estate.hands_wanted[Resource::Food.index()] = growth.fields() as f32 * config.hands_per_field;

    estate.potential[Resource::Wood.index()] = industry.wood_tiles as f32 * config.wood_per_tile;
    estate.hands_wanted[Resource::Wood.index()] =
        industry.wood_tiles as f32 * config.hands_per_tile;
    estate.potential[Resource::Stone.index()] = industry.stone_tiles as f32 * config.stone_per_tile;
    estate.hands_wanted[Resource::Stone.index()] =
        industry.stone_tiles as f32 * config.hands_per_tile;

    // Re-read rather than cached, even though ownership never moves, so the two
    // directions of the claim cannot describe different sets: a seam that is no
    // longer there simply stops contributing.
    for &seam in &industry.deposits {
        let Some((resource, amount)) = worked(seam) else {
            continue;
        };
        estate.potential[resource.index()] += amount * config.yield_per_richness;
        estate.hands_wanted[resource.index()] += amount * config.hands_per_richness;
    }

    industry.estate = estate;
}

/// Splits the population across the professions, and this is the whole of the model.
///
/// **No target, no policy, no hysteresis**: the split is a function of the land, so
/// it is the same on the first step as on the thousandth and cannot flap. A resource
/// the land does not offer is never produced — `hands_wanted` is zero, the share is
/// zero, and no branch anywhere special-cases the empty case.
fn allocate_hands(industry: &mut CityIndustry, population: f32) {
    let total: f32 = industry.estate.hands_wanted.iter().sum();
    if total <= 0.0 {
        industry.hands = [0.0; RESOURCE_COUNT];
        industry.idle = population.max(0.0);
        return;
    }

    let mut working = 0.0;
    for resource in Resource::ALL {
        let wanted = industry.estate.hands_wanted[resource.index()];
        // Capped by the work there is, which is what leaves a city with more people
        // than land visibly idle rather than quietly over-producing.
        let hands = (population * wanted / total).min(wanted);
        industry.hands[resource.index()] = hands;
        working += hands;
    }
    industry.idle = (population - working).max(0.0);
}

/// One step's output: each resource's potential, scaled by the hands on it.
fn extraction(industry: &CityIndustry, population: f32) -> [f32; RESOURCE_COUNT] {
    let mut output = [0.0; RESOURCE_COUNT];
    for resource in Resource::ALL {
        output[resource.index()] =
            industry.estate.potential[resource.index()] * effort(industry, population, resource);
    }
    output
}

/// The share of a city's whole effort that goes to one resource — a **pure function
/// of the land**, with no population in it, times how well staffed the city is
/// overall.
///
/// **The obvious formula is `hands[r] / hands_wanted[r]`, and it collapses the
/// world.** Where a city has fewer people than its land offers work, that ratio is
/// `population / total_hands_wanted`, so the harvest becomes proportional to the
/// population, so `capacity` does, so the logistic's K is proportional to its own p —
/// which has no stable non-zero equilibrium at all. Whichever side of 1 the constant
/// falls on, every city in the world either runs away or decays to the floor
/// together. Measured on the default world before this was fixed: median population
/// **22**, forty cities on the floor, and eighty of ninety-two shrinking.
///
/// The share of *effort* has no such term. A city whose land offers a fifth of its
/// work as mining farms four fifths as hard, whatever its population — which is also
/// a more literal reading of "the split follows the land, not a policy" than the
/// staffing ratio ever was. Capacity is back to being a property of the ground, which
/// is what gh-6's stability rested on, and a seam still costs a city its harvest
/// because it is still a share of the same effort.
///
/// Staffing is the one place the population enters, and only through a `min` that is
/// 1 for every city in the shipped world — a city with more people than work has them
/// idle rather than over-producing, and one with fewer is stretched. It is deliberately
/// *not* allowed to make the food case proportional again: see
/// `total_hands_wanted_stays_under_what_a_city_can_staff`.
fn effort(industry: &CityIndustry, population: f32, resource: Resource) -> f32 {
    let total: f32 = industry.estate.hands_wanted.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let staffing = (population / total).min(1.0);
    industry.estate.hands_wanted[resource.index()] / total * staffing
}

/// Takes a basket's demand out of the stores and reports the share it got.
///
/// Each spend takes what the store has, so a basket the city cannot meet leaves the
/// store empty rather than negative — and the *next* spend in the order gets what is
/// left, which is what makes the order the design.
fn spend(
    industry: &mut CityIndustry,
    basket: &[Resource],
    per_person: f32,
    population: f32,
) -> f32 {
    let want = population * per_person;
    if want <= 0.0 || basket.is_empty() {
        return 1.0;
    }

    let mut met = 0.0;
    for resource in basket {
        let stock = &mut industry.stocks[resource.index()];
        let taken = want.min(*stock);
        *stock -= taken;
        met += taken / want;
    }
    met / basket.len() as f32
}

/// Spends on the town tiles the growth step is about to add, and reports how many the
/// spend paid for.
///
/// A city that cannot afford stone does not stop growing in *people*; it stops
/// building, and grows crowded instead.
fn build(
    config: &IndustryConfig,
    growth_config: &GrowthConfig,
    industry: &mut CityIndustry,
    growth: &CityGrowth,
) -> usize {
    let target = town_target(growth_config, growth.population);
    let wanted = target.saturating_sub(growth.town());
    if wanted == 0 {
        // Shrinking, or already the right size. The growth step is free to give tiles
        // back — that costs nothing and refunds nothing.
        return usize::MAX;
    }

    let wanted = wanted.min(growth_config.claims_per_step.max(1) as usize);
    let afford = |stock: f32, cost: f32| {
        if cost <= 0.0 {
            usize::MAX
        } else {
            (stock / cost) as usize
        }
    };
    let affordable = wanted
        .min(afford(
            industry.stocks[Resource::Wood.index()],
            config.build_cost_wood,
        ))
        .min(afford(
            industry.stocks[Resource::Stone.index()],
            config.build_cost_stone,
        ));

    industry.stocks[Resource::Wood.index()] -= affordable as f32 * config.build_cost_wood;
    industry.stocks[Resource::Stone.index()] -= affordable as f32 * config.build_cost_stone;
    affordable
}

/// The chunks a disc of `reach` tiles about `centre` touches.
///
/// Deposits are indexed by chunk exactly as cities are, so a city finds the seams it
/// can reach without scanning every seam in the world.
fn chunks_within(centre: IVec2, reach: i32) -> impl Iterator<Item = usize> {
    let last = WORLD_TILES.as_ivec2() - IVec2::ONE;
    let low = chunk_of_tile((centre - IVec2::splat(reach)).clamp(IVec2::ZERO, last));
    let high = chunk_of_tile((centre + IVec2::splat(reach)).clamp(IVec2::ZERO, last));
    (low.y..=high.y)
        .flat_map(move |cy| (low.x..=high.x).map(move |cx| (cy * WORLD_CHUNKS.x + cx) as usize))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gameplay::city::CitySize;

    const CENTRE: IVec2 = IVec2::new(2048, 2048);

    fn city() -> City {
        City {
            id: 0,
            centre: CENTRE,
            size: CitySize::Village,
            radius: 5,
        }
    }

    /// A world of one kind, so nothing the estate counts is an accident of the
    /// terrain. Only the swept path reads it.
    fn bare_map() -> WorldMap {
        WorldMap::from_fn(|_| TerrainKind::Grass)
    }

    /// One seam the tests can hand to the step: an entity that stands for nothing
    /// and the two numbers the step actually reads off it.
    #[derive(Clone, Copy)]
    struct Seam {
        entity: Entity,
        resource: Resource,
        richness: f32,
    }

    /// The entity values are arbitrary and nothing keys on them — the real caller
    /// answers from a query and this answers from a list, which is the same shape.
    fn seams(of: &[(Resource, f32)]) -> Vec<Seam> {
        of.iter()
            .enumerate()
            .map(|(index, &(resource, richness))| Seam {
                entity: Entity::from_raw_u32(index as u32 + 1).expect("nonzero"),
                resource,
                richness,
            })
            .collect()
    }

    /// A city holding `wood` and `stone` tiles and the given seams, with full stores
    /// for a town of `town` tiles.
    fn holding(
        config: &IndustryConfig,
        wood: u32,
        stone: u32,
        seams: &[Seam],
        town: usize,
    ) -> CityIndustry {
        let mut industry = CityIndustry {
            stocks: [0.0; RESOURCE_COUNT],
            hands: [0.0; RESOURCE_COUNT],
            idle: 0.0,
            happiness: config.happiness_neutral,
            estate: Estate::default(),
            deposits: seams.iter().map(|seam| seam.entity).collect(),
            wood_tiles: wood,
            stone_tiles: stone,
        };
        for resource in Resource::ALL {
            industry.stocks[resource.index()] = town as f32 * store_cap_per_tile(config, resource);
        }
        industry
    }

    /// Runs one industry step against a fixed ledger.
    fn step(
        config: &IndustryConfig,
        growth_config: &GrowthConfig,
        industry: &mut CityIndustry,
        seams: &[Seam],
        growth: &CityGrowth,
        rain: f32,
    ) -> Labour {
        step_industry(
            config,
            growth_config,
            &bare_map(),
            &[],
            Sky { rain },
            false,
            &city(),
            growth,
            industry,
            |entity| {
                seams
                    .iter()
                    .find(|seam| seam.entity == entity)
                    .map(|seam| (seam.resource, seam.richness))
            },
        )
    }

    /// The whole of the profession model: the split follows the land. Two cities on
    /// the same ground allocate the same hands whatever is in their stores, so
    /// nothing here can oscillate and no hysteresis band is needed anywhere.
    #[test]
    fn the_split_is_a_pure_function_of_the_estate() {
        let config = IndustryConfig::default();
        let growth_config = GrowthConfig::default();
        let growth = CityGrowth::for_test(4000.0, 6000.0, 500, 100);
        let seam = seams(&[(Resource::Iron, 0.5)]);

        let mut full = holding(&config, 1200, 400, &seam, 100);
        let mut empty = holding(&config, 1200, 400, &seam, 100);
        empty.stocks = [0.0; RESOURCE_COUNT];

        step(&config, &growth_config, &mut full, &seam, &growth, 0.0);
        step(&config, &growth_config, &mut empty, &seam, &growth, 0.0);

        for resource in Resource::ALL {
            assert_eq!(
                full.hands(resource),
                empty.hands(resource),
                "a full store changed how many {} the city has",
                resource.profession()
            );
        }
    }

    /// Hands sum to at most the population, no one works two professions, and the
    /// remainder is idle. Idle hands still eat and still want comfort.
    #[test]
    fn hands_never_exceed_the_population_and_the_rest_is_idle() {
        let config = IndustryConfig::default();
        let growth_config = GrowthConfig::default();
        let seam = seams(&[(Resource::Salt, 0.8)]);

        // Both regimes: more people than work, and more work than people.
        for population in [200.0f32, 4000.0, 200_000.0] {
            let growth = CityGrowth::for_test(population, 6000.0, 500, 100);
            let mut industry = holding(&config, 1200, 400, &seam, 100);
            step(&config, &growth_config, &mut industry, &seam, &growth, 0.0);

            let working: f32 = Resource::ALL.iter().map(|r| industry.hands(*r)).sum();
            assert!(
                working <= population + 1e-2,
                "{working} hands out of {population} people"
            );
            assert!(
                (working + industry.idle() - population).abs() < 1e-2,
                "{working} working and {} idle is not {population}",
                industry.idle()
            );
        }
    }

    /// A resource the land does not offer is never produced: `hands_wanted` is zero,
    /// the share is zero, and no branch anywhere special-cases the empty case.
    #[test]
    fn a_resource_the_land_does_not_offer_is_never_produced() {
        let config = IndustryConfig::default();
        let growth_config = GrowthConfig::default();
        let growth = CityGrowth::for_test(4000.0, 6000.0, 500, 100);

        // No wood, no stone, no seams — a city on bare open ground.
        let mut industry = holding(&config, 0, 0, &[], 100);
        industry.stocks = [0.0; RESOURCE_COUNT];
        step(&config, &growth_config, &mut industry, &[], &growth, 0.0);

        for resource in [
            Resource::Wood,
            Resource::Stone,
            Resource::Iron,
            Resource::Copper,
            Resource::Salt,
        ] {
            assert_eq!(industry.hands(resource), 0.0, "{resource:?} got hands");
            assert_eq!(industry.stock(resource), 0.0, "{resource:?} was produced");
        }
        // And everyone who is not farming is idle rather than lost.
        assert!(industry.idle() >= 0.0);
    }

    /// A stock is never negative and never exceeds town tiles times its own cap, so a
    /// town that shrinks loses stores it was holding and the overflow is lost rather
    /// than owed.
    #[test]
    fn a_stock_stays_between_nothing_and_what_the_town_can_hold() {
        let config = IndustryConfig::default();
        let growth_config = GrowthConfig::default();
        let seam = seams(&[(Resource::Iron, 1.0), (Resource::Salt, 1.0)]);
        let mut industry = holding(&config, 4000, 4000, &seam, 300);

        // The town shrinks under the stores it was holding, which is the case the cap
        // has to be re-applied for.
        for town in [300usize, 200, 50, 10] {
            let growth = CityGrowth::for_test(2000.0, 6000.0, 500, town);
            for _ in 0..50 {
                step(&config, &growth_config, &mut industry, &seam, &growth, 0.0);
                for resource in Resource::ALL {
                    let cap = town as f32 * store_cap_per_tile(&config, resource);
                    let stock = industry.stock(resource);
                    assert!(
                        (0.0..=cap + 1e-2).contains(&stock),
                        "{resource:?} holds {stock} against a cap of {cap}"
                    );
                }
            }
        }
    }

    /// The granary may only ever make up a shortfall, never raise the ceiling — so a
    /// full store cannot push K above the land and no boom-bust cycle exists.
    #[test]
    fn the_granary_only_ever_makes_up_a_shortfall() {
        let config = IndustryConfig::default();
        let growth_config = GrowthConfig::default();
        // A city whose fields comfortably feed it. The store is full and must put
        // nothing on the table.
        let fed = CityGrowth::for_test(1000.0, 100_000.0, 500, 100);
        let mut industry = holding(&config, 0, 0, &[], 100);
        let labour = step(&config, &growth_config, &mut industry, &[], &fed, 0.0);
        assert_eq!(
            labour.granary_release, 0.0,
            "the store fed a city its fields already fed"
        );

        // And one whose fields feed it nothing at all: the release is exactly the
        // shortfall and not one grain more.
        let starved = CityGrowth::for_test(1000.0, 0.0, 0, 100);
        let mut industry = holding(&config, 0, 0, &[], 100);
        let labour = step(&config, &growth_config, &mut industry, &[], &starved, 0.0);
        let demand = starved.population * growth_config.food_per_person;
        assert!(
            (labour.granary_release - demand).abs() < 1e-2,
            "released {} against a demand of {demand}",
            labour.granary_release
        );
    }

    /// A city can outlive a total crop failure for as many steps as its store holds
    /// and not one more, and that number is a knob rather than an accident.
    #[test]
    fn a_full_granary_buys_the_steps_its_knob_says_it_does() {
        let config = IndustryConfig::default();
        let growth_config = GrowthConfig::default();
        let town = 100usize;
        // Population pinned at what the town houses, so the arithmetic in the doc
        // comment is the arithmetic under test.
        let population = town as f32 * growth_config.town_people_per_tile;
        let growth = CityGrowth::for_test(population, 0.0, 0, town);
        let mut industry = holding(&config, 0, 0, &[], town);

        let expected = (config.granary_per_town_tile
            / (growth_config.town_people_per_tile * growth_config.food_per_person))
            .round() as usize;

        let mut fed = 0;
        for _ in 0..expected * 4 {
            let labour = step(&config, &growth_config, &mut industry, &[], &growth, 0.0);
            if labour.granary_release <= 0.0 {
                break;
            }
            fed += 1;
        }

        assert_eq!(
            fed, expected,
            "the granary fed the city for {fed} steps of total failure, not {expected}"
        );
    }

    /// The cost of the split, and the point of the whole feature: a city sitting on a
    /// seam sends hands to it and harvests less than the same site without one.
    #[test]
    fn a_city_on_a_seam_farms_with_fewer_hands_than_one_without() {
        let config = IndustryConfig::default();
        let growth_config = GrowthConfig::default();
        let growth = CityGrowth::for_test(4000.0, 6000.0, 500, 100);

        let with = seams(&[(Resource::Iron, 0.8)]);
        let mut mining = holding(&config, 1200, 400, &with, 100);
        let mut farming = holding(&config, 1200, 400, &[], 100);

        let mined = step(&config, &growth_config, &mut mining, &with, &growth, 0.0);
        let farmed = step(&config, &growth_config, &mut farming, &[], &growth, 0.0);

        assert!(
            mined.farmer_share < farmed.farmer_share,
            "the seam cost nothing: {} against {}",
            mined.farmer_share,
            farmed.farmer_share
        );
        assert!(
            mining.hands(Resource::Iron) > 0.0,
            "nobody went to the seam"
        );
        assert_eq!(farming.hands(Resource::Iron), 0.0);
    }

    /// Happiness is smoothed, so a single bad step cannot swing the growth rate — and
    /// a well-supplied city ends up growing faster than a badly supplied one.
    #[test]
    fn being_supplied_speeds_a_city_up_and_going_short_slows_it_down() {
        let config = IndustryConfig::default();
        let growth_config = GrowthConfig::default();
        let growth = CityGrowth::for_test(4000.0, 6000.0, 500, 100);
        let rich = seams(&[
            (Resource::Iron, 1.0),
            (Resource::Salt, 1.0),
            (Resource::Copper, 1.0),
        ]);

        let mut supplied = holding(&config, 4000, 2000, &rich, 100);
        let mut short = holding(&config, 0, 0, &[], 100);
        short.stocks = [0.0; RESOURCE_COUNT];

        let mut supplied_scale = 0.0;
        let mut short_scale = 0.0;
        for _ in 0..200 {
            supplied_scale =
                step(&config, &growth_config, &mut supplied, &rich, &growth, 0.0).growth_scale;
            short_scale = step(&config, &growth_config, &mut short, &[], &growth, 0.0).growth_scale;
        }

        assert!(
            supplied_scale > 1.0,
            "a well-supplied city grows no faster than the base rate: {supplied_scale}"
        );
        assert!(
            short_scale < 0.0,
            "a city getting nothing at all still grows: {short_scale}"
        );
        assert!(
            short_scale >= config.min_growth_rate / growth_config.growth_rate,
            "the floor under the rate did not hold"
        );
    }

    /// The swing has to clear this or the "bleeds people" half of the design does not
    /// exist at all: a city getting nothing would still grow, only slower.
    #[test]
    fn the_swing_is_hard_enough_for_a_city_to_actually_decline() {
        let config = IndustryConfig::default();
        assert!(
            config.happiness_swing * config.happiness_neutral > 1.0,
            "at swing {} and neutral {} no city can ever decline from unhappiness",
            config.happiness_swing,
            config.happiness_neutral
        );
    }

    /// The chunk rows a seeding city consults have to cover its whole reach, or a seam
    /// inside the disc would never be found.
    #[test]
    fn the_chunk_scan_covers_the_whole_estate_reach() {
        let reach = IndustryConfig::default().estate_reach_tiles as i32;
        for centre in [CENTRE, IVec2::new(64, 64), IVec2::new(4000, 100)] {
            let rows: Vec<usize> = chunks_within(centre, reach).collect();
            for dy in [-reach, 0, reach] {
                for dx in [-reach, 0, reach] {
                    let tile = centre + IVec2::new(dx, dy);
                    if !tile_in_world(tile) {
                        continue;
                    }
                    assert!(
                        rows.contains(&crate::gameplay::world::chunk_index_of_tile(tile)),
                        "the scan around {centre} misses {tile}"
                    );
                }
            }
        }
    }
}
