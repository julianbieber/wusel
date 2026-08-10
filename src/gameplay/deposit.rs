//! What the land offers beyond food: the seams a city can work.
//!
//! **A seam is a place, a wood is an area.** Iron, copper and salt are discrete
//! sites laid out once from the finished world — the same idiom as a river spring
//! or a city candidate, one jittered candidate per lattice cell kept if the ground
//! matches — and a site has exactly one owner. Wood and stone are not: they are
//! read off the `Forest` and the `Rock`/`Mountain`/`Gravel` tiles inside a city's
//! reach, so two cities whose reaches overlap both log the same hillside. That is
//! the one place the crate's "a tile has one owner" principle deliberately does not
//! extend, and the reason is that standing timber and an outcrop are not consumed
//! by being worked, while a seam is somewhere you either have or do not.
//!
//! **A deposit is an entity**, the way a city is. The entity carrying [`Deposit`]
//! *is* the record of the seam existing, which buys three things a row in a table
//! would not. Ownership becomes a pointer between two entities rather than an id
//! looked up in a side list. `DespawnOnExit(Screen::Gameplay)` is the whole of its
//! lifetime, so a seam cannot outlive the world it was read off. And [`DepositMap`]
//! is not a store but an index of entities by chunk, exactly as
//! [`crate::gameplay::city::CityMap`] is, so there is never a second copy of a seam
//! to keep true.
//!
//! **Nothing here stamps a tile.** A deposit is a record, so this stage costs no
//! chunk refresh, no height upload, and cannot put a kind on the map that
//! `the_terrain_never_produces_a_kind_the_plan_stamps` would have to learn about.
//! The cost is that a mine is invisible until the panel says so.

use bevy::{platform::collections::HashMap, prelude::*};

use crate::gameplay::{
    biome::Biome,
    noise::hash2,
    plan::WorldPlanConfig,
    terrain::{TerrainKind, TerrainSampler},
    world::{WORLD_TILES, WorldSnapshot, chunk_index_of_tile},
};

/// The six things a city can hold.
///
/// Food is in the list even though it has no deposit at all: it is what every other
/// part of the feature is measured against, and leaving it out would mean two
/// parallel lists of "things a city holds" that could drift.
///
/// The discriminant is an index into the per-resource arrays
/// [`crate::gameplay::industry`] keeps, on the same terms `TerrainKind`'s
/// discriminant is its tileset column: the enum and the array length cannot drift,
/// because the array is a fixed-length array over [`RESOURCE_COUNT`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum Resource {
    Food = 0,
    Wood = 1,
    Stone = 2,
    Iron = 3,
    Copper = 4,
    Salt = 5,
}

pub const RESOURCE_COUNT: usize = 6;

impl Resource {
    /// In discriminant order, because everything that walks the resources indexes
    /// an array with what it finds — the panel's rows, the observation's fields and
    /// the hand allocation all rely on the two agreeing.
    pub const ALL: [Resource; RESOURCE_COUNT] = [
        Resource::Food,
        Resource::Wood,
        Resource::Stone,
        Resource::Iron,
        Resource::Copper,
        Resource::Salt,
    ];

    pub fn index(self) -> usize {
        self as usize
    }

    /// What the panel and the ctl call it. Lower case, because the ctl matches on
    /// it and a verb a player types should not carry capitals.
    pub fn label(self) -> &'static str {
        match self {
            Resource::Food => "food",
            Resource::Wood => "wood",
            Resource::Stone => "stone",
            Resource::Iron => "iron",
            Resource::Copper => "copper",
            Resource::Salt => "salt",
        }
    }

    /// What the hands working it are called. A **profession is the resource its
    /// hands produce** — there is no `Profession` enum, so the two lists cannot
    /// drift. The day a refiner arrives, eating iron and producing tools, the 1:1
    /// breaks and this becomes an enum of its own.
    ///
    /// Its only reader is `observe cities`, which vanishes on wasm with the rest of
    /// `control/` — the same reason `WorldMap::generated` carries this allow.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn profession(self) -> &'static str {
        match self {
            Resource::Food => "farmers",
            Resource::Wood => "woodcutters",
            Resource::Stone => "quarriers",
            Resource::Iron => "miners",
            Resource::Copper => "smelters",
            Resource::Salt => "salters",
        }
    }
}

/// What ground a resource is found in.
///
/// Numbers and lists only, and **no rule anywhere names a `Resource`** — so adding
/// a seventh is a row in [`RECIPES`] rather than a branch. The lists are static
/// slices rather than the spec's `Vec`, because the table is a constant and a
/// constant that allocates would be built once per call.
pub struct DepositRecipe {
    pub resource: Resource,
    /// The tile kinds this seam is found in. Nothing in here is water, a river, or
    /// a kind the plan stamps, which is what makes "a deposit is never on water"
    /// true by omission rather than by a check that could fall out of step.
    pub ground: &'static [TerrainKind],
    /// The regions it belongs to. Read against `TileSample::dominant` — "which
    /// region is this" — rather than against `cover`, because a recipe naming
    /// `Highland` is naming the region's geology and not the tile's vegetation.
    pub biomes: &'static [Biome],
    pub min_elevation: f32,
    pub max_elevation: f32,
    /// How rich this resource's best ground is, before the threshold. The lever on
    /// how common one seam is against another, and the only per-resource number
    /// here that is a preference rather than a fact about the map.
    pub weight: f32,
}

impl DepositRecipe {
    /// How well this ground suits the recipe, on 0..`weight` — and exactly zero for
    /// ground the recipe does not name at all, which is what makes the ground and
    /// biome lists a gate rather than one term among several.
    ///
    /// Triangular across the elevation band: a seam is richest in the middle of the
    /// country it belongs to and peters out at either edge, so the band's own width
    /// is what decides how rare a good site is. A tile outside the band clamps to an
    /// edge and scores zero, so no separate range test is needed.
    pub fn score(&self, kind: TerrainKind, biome: Biome, elevation: f32) -> f32 {
        if !self.ground.contains(&kind) || !self.biomes.contains(&biome) {
            return 0.0;
        }
        let span = self.max_elevation - self.min_elevation;
        if span <= 0.0 {
            return 0.0;
        }
        let t = ((elevation - self.min_elevation) / span).clamp(0.0, 1.0);
        self.weight * (1.0 - (2.0 * t - 1.0).abs())
    }
}

/// The whole table. Three rows, because `Food`, `Wood` and `Stone` are area
/// resources read off the tiles a city can reach rather than sites laid out here.
///
/// The bands are read against `TerrainConfig`'s own: 0.42 is the water line, 0.72
/// the foot of the mountains, 0.78 where mountain becomes scree and 0.88 the snow
/// line.
pub const RECIPES: [DepositRecipe; 3] = [
    // High and cold: the mountain band and the scree above it, in the one region
    // that has any. Iron is the upkeep basket's expensive half, so a city with a
    // seam is meaningfully different from one without.
    DepositRecipe {
        resource: Resource::Iron,
        ground: &[TerrainKind::Mountain, TerrainKind::Rock],
        biomes: &[Biome::Highland],
        min_elevation: 0.72,
        max_elevation: 0.92,
        weight: 1.0,
    },
    // The stripped bedrock of the lowland band — `Rock` where the lithology layer
    // is hard and `Gravel` where it is soft — plus the bottom of the scree. Two
    // regions rather than one, so copper is the commonest of the three.
    DepositRecipe {
        resource: Resource::Copper,
        ground: &[TerrainKind::Rock, TerrainKind::Gravel],
        biomes: &[Biome::Highland, Biome::Desert],
        min_elevation: 0.45,
        max_elevation: 0.79,
        weight: 1.0,
    },
    // Just above the water line, where a desert pan dries out and a marsh does not
    // drain. The narrowest band of the three, which is what makes salt the seam a
    // city is most likely to be short of.
    DepositRecipe {
        resource: Resource::Salt,
        ground: &[TerrainKind::Sand, TerrainKind::Marsh],
        biomes: &[Biome::Desert, Biome::Wetland],
        min_elevation: 0.42,
        max_elevation: 0.60,
        weight: 1.0,
    },
];

/// One worked place.
///
/// The entity carrying this *is* the record of the seam existing, the way a `City`
/// entity is the record of a city, and it is spawned with
/// `DespawnOnExit(Screen::Gameplay)` — there is no cleanup system anywhere in the
/// crate, so a seam without that tag would leak into the next world.
///
/// `richness` is how far the site cleared its recipe's threshold, on 0..1 — the same
/// trick `CitySize::from_excess` plays with a settlement score, and it is what keeps
/// two iron seams from being interchangeable.
///
/// `owner` points at the city entity working it. Settled once, at seeding, in city
/// id order, and it never moves after. A pointer rather than an id, so the reverse —
/// a city's own list of seams — is the same handle read the other way and neither
/// side can name something that has been despawned.
#[derive(Component, Clone, Copy, Debug)]
pub struct Deposit {
    pub resource: Resource,
    pub tile: IVec2,
    pub richness: f32,
    pub owner: Option<Entity>,
}

/// Which deposits touch which chunk.
///
/// **An index into the deposit entities and nothing more**, exactly as `CityMap` is
/// an index into the city entities: the seam itself lives on its own entity, so
/// there is no second copy to keep true. A seam is a single tile, so unlike a city
/// it appears in exactly one chunk's row.
#[derive(Resource, Default)]
pub struct DepositMap {
    by_chunk: HashMap<usize, Vec<Entity>>,
}

impl DepositMap {
    pub fn insert(&mut self, chunk: usize, deposit: Entity) {
        self.by_chunk.entry(chunk).or_default().push(deposit);
    }

    pub fn in_chunk(&self, chunk: usize) -> &[Entity] {
        self.by_chunk.get(&chunk).map_or(&[], Vec::as_slice)
    }
}

/// Salt for the per-cell jitter, so a seam is not the corner of its cell.
const DEPOSIT_SITE_SALT: i32 = 0x6b21_9d4fu32 as i32;

/// Lays out every deposit in the world.
///
/// A pure function of `(TerrainConfig, WorldPlanConfig, WorldSnapshot)`, so the
/// terrain, the rivers, the cities as founded, the roads **and now the seams** are
/// all still identical on every platform. Reproducibility ends where it already
/// ended, at the first simulation step.
///
/// The world is cut into square cells and each one proposes at most one site, which
/// is what bounds the count and stops a region being paved with mines. Unlike the
/// city stage there is no spacing pass: a cell already keeps its neighbours a
/// `deposit_cell_tiles - deposit_jitter_tiles` inset apart, and two seams of
/// different resources sitting near each other is a mining district rather than a
/// defect.
pub fn plan_deposits(
    sampler: &TerrainSampler,
    config: &WorldPlanConfig,
    world: &WorldSnapshot,
) -> Vec<Deposit> {
    let cell = config.deposit_cell_tiles.max(1) as i32;
    let cells = IVec2::new(WORLD_TILES.x as i32 / cell, WORLD_TILES.y as i32 / cell);

    let mut sites = Vec::new();
    // Row-major, which is cell order — so the index rows are in one fixed order and
    // a scenario's output diffs cleanly across runs. What is reproducible is the
    // layout and never the `Entity` values: nothing may key a decision or an
    // assertion on an id, only on the tile it sits at.
    for cy in 0..cells.y {
        for cx in 0..cells.x {
            if let Some(deposit) = candidate_for_cell(sampler, world, config, cell, cx, cy) {
                sites.push(deposit);
            }
        }
    }
    sites
}

/// The one seam a cell proposes, or `None` if its candidate tile is ordinary ground.
fn candidate_for_cell(
    sampler: &TerrainSampler,
    world: &WorldSnapshot,
    config: &WorldPlanConfig,
    cell: i32,
    cx: i32,
    cy: i32,
) -> Option<Deposit> {
    let jitter = (config.deposit_jitter_tiles.max(1) as i32).min(cell);
    // Centred in the cell, so the inset is what guarantees two neighbouring seams
    // are at least `cell - jitter` tiles apart without a spacing pass.
    let inset = (cell - jitter) / 2;
    let h = hash2(cx ^ DEPOSIT_SITE_SALT, cy);
    let offset = IVec2::new(
        inset + (h & 0xffff) as i32 % jitter,
        inset + ((h >> 16) & 0xffff) as i32 % jitter,
    );
    let tile = IVec2::new(cx, cy) * cell + offset;

    // The kind comes off the finished map — after the rivers, the lakes and the dry
    // valleys — so a wadi that turned desert sand into scrub has already had its say
    // on whether a salt pan is here.
    let kind = world.tile(tile)?;
    let sample = sampler.sample(tile.x as f32, tile.y as f32);

    let (resource, score) = RECIPES
        .iter()
        .map(|recipe| {
            (
                recipe.resource,
                recipe.score(kind, sample.dominant, sample.elevation),
            )
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))?;

    let threshold = config.deposit_threshold;
    if score < threshold {
        return None;
    }

    Some(Deposit {
        resource,
        tile,
        richness: ((score - threshold) / (1.0 - threshold).max(f32::EPSILON)).clamp(0.0, 1.0),
        owner: None,
    })
}

/// Every chunk a seam falls in — one, since a seam is a tile.
pub fn chunk_of_deposit(deposit: &Deposit) -> usize {
    chunk_index_of_tile(deposit.tile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gameplay::terrain::TerrainConfig;

    use crate::gameplay::terrain::shared_test_sampler;
    use crate::gameplay::world::WorldMap;

    fn config() -> WorldPlanConfig {
        WorldPlanConfig::default()
    }

    /// The invariant that holds by *omission* rather than by a check: no recipe names
    /// water, a river, or a kind the plan stamps, so the layout cannot put a seam on
    /// one however the scoring is retuned.
    #[test]
    fn no_recipe_can_put_a_seam_on_water_or_on_a_tile_the_plan_stamps() {
        for recipe in &RECIPES {
            for kind in recipe.ground {
                assert!(
                    !kind.is_water(),
                    "{:?} is found in {kind:?}, which is water",
                    recipe.resource
                );
                assert!(
                    !matches!(
                        kind,
                        TerrainKind::River
                            | TerrainKind::Town
                            | TerrainKind::Road
                            | TerrainKind::Farmland
                    ),
                    "{:?} is found in {kind:?}, which the plan stamps",
                    recipe.resource
                );
            }
        }
    }

    /// `Food`, `Wood` and `Stone` are area resources read off the tiles a city can
    /// reach, so they must never appear as a site.
    #[test]
    fn only_the_seam_resources_have_a_recipe() {
        for recipe in &RECIPES {
            assert!(
                matches!(
                    recipe.resource,
                    Resource::Iron | Resource::Copper | Resource::Salt
                ),
                "{:?} has a deposit recipe but is an area resource",
                recipe.resource
            );
        }
        for resource in [Resource::Food, Resource::Wood, Resource::Stone] {
            assert!(
                !RECIPES.iter().any(|r| r.resource == resource),
                "{resource:?} has a deposit recipe"
            );
        }
    }

    /// The ground and biome lists are a gate: ground the recipe does not name scores
    /// nothing at all, whatever the elevation is.
    #[test]
    fn a_recipe_scores_nothing_outside_its_own_ground() {
        let iron = &RECIPES[0];
        let mid = (iron.min_elevation + iron.max_elevation) / 2.0;

        assert!(iron.score(TerrainKind::Mountain, Biome::Highland, mid) > 0.0);
        assert_eq!(iron.score(TerrainKind::Grass, Biome::Highland, mid), 0.0);
        assert_eq!(iron.score(TerrainKind::Mountain, Biome::Plains, mid), 0.0);
        // And outside the band, where the triangle clamps to an edge.
        assert_eq!(
            iron.score(TerrainKind::Mountain, Biome::Highland, iron.min_elevation),
            0.0
        );
        assert_eq!(
            iron.score(TerrainKind::Mountain, Biome::Highland, 0.05),
            0.0,
            "ground far below the band scored anyway"
        );
    }

    /// Two seams of the same resource are not interchangeable: the one nearer the
    /// middle of its band is the richer.
    #[test]
    fn a_site_deeper_in_its_band_scores_higher() {
        let copper = &RECIPES[1];
        let mid = (copper.min_elevation + copper.max_elevation) / 2.0;
        let edge = copper.min_elevation + (copper.max_elevation - copper.min_elevation) * 0.1;

        let centre = copper.score(TerrainKind::Gravel, Biome::Desert, mid);
        let margin = copper.score(TerrainKind::Gravel, Biome::Desert, edge);
        assert!(centre > margin, "{centre} against {margin}");
        assert!(centre <= copper.weight, "the score ran past its own weight");
    }

    /// One site per cell, so no region can be paved with mines — and the inset keeps
    /// two neighbouring seams apart without a spacing pass.
    #[test]
    fn at_most_one_seam_per_cell_and_never_two_on_one_tile() {
        let _terrain = TerrainConfig::default();
        let config = config();
        // A patch of the world rather than the whole thing: `WorldMap::from_fn` is
        // cheap, and the layout only ever reads a tile it picked itself.
        let world = WorldMap::from_fn(|tile| {
            if (tile.x / 7 + tile.y / 5) % 3 == 0 {
                TerrainKind::Rock
            } else {
                TerrainKind::Gravel
            }
        })
        .snapshot()
        .expect("from_fn fills every chunk");

        let sites = plan_deposits(shared_test_sampler(), &config, &world);
        assert!(
            !sites.is_empty(),
            "no seam anywhere in a world of bare rock"
        );

        let cell = config.deposit_cell_tiles as i32;
        let mut seen = std::collections::HashSet::new();
        for site in &sites {
            let cell_of = IVec2::new(site.tile.x / cell, site.tile.y / cell);
            assert!(
                seen.insert(cell_of),
                "two seams proposed by cell {cell_of} at {}",
                site.tile
            );
        }
    }

    /// The layout is a pure function of its inputs, which is what keeps the world
    /// identical across runs and platforms.
    #[test]
    fn the_same_world_lays_out_the_same_seams() {
        let _terrain = TerrainConfig::default();
        let config = config();
        let world = WorldMap::from_fn(|tile| {
            if tile.x % 11 < 4 {
                TerrainKind::Sand
            } else {
                TerrainKind::Marsh
            }
        })
        .snapshot()
        .expect("from_fn fills every chunk");

        let first = plan_deposits(shared_test_sampler(), &config, &world);
        let second = plan_deposits(shared_test_sampler(), &config, &world);

        assert_eq!(first.len(), second.len());
        for (a, b) in first.iter().zip(&second) {
            assert_eq!(a.tile, b.tile);
            assert_eq!(a.resource, b.resource);
            assert_eq!(a.richness, b.richness);
        }
    }

    /// The discriminant is an index, and everything that walks the resources relies
    /// on the two agreeing.
    #[test]
    fn every_resource_indexes_its_own_slot() {
        for (index, resource) in Resource::ALL.iter().enumerate() {
            assert_eq!(resource.index(), index, "{resource:?} is out of order");
        }
        assert_eq!(Resource::ALL.len(), RESOURCE_COUNT);
    }
}

#[cfg(test)]
mod measurements {
    use super::*;
    use crate::gameplay::terrain::TerrainConfig;
    use crate::gameplay::terrain::shared_test_sampler;
    use crate::gameplay::{
        city::plan_cities,
        drainage::plan_drainage,
        industry::IndustryConfig,
        river::plan_rivers,
        world::{TileEdit, WorldSnapshot},
    };

    /// Where the figures in `WorldPlanConfig`'s deposit doc comments come from.
    ///
    /// The whole world, planned as far as the seams — the only test that can say the
    /// defaults lay out a world worth mining, since every other test here works on
    /// terrain it made up. Ignored because it generates all 4096 chunks:
    /// `cargo test --release -- --ignored --nocapture`, and run it *alone*.
    ///
    /// The drainage stage is not optional here for the reason the plan runs it: a
    /// wadi turns desert `Sand` into `Scrub` and a marsh channel into `Reed`, both of
    /// which move ground onto and off the salt recipe's list.
    #[test]
    #[ignore = "generates the whole 4096x4096 world"]
    fn the_default_config_lays_seams_of_every_resource() {
        let terrain = TerrainConfig::default();
        let config = WorldPlanConfig::default();

        let base = WorldSnapshot::generated(&terrain, shared_test_sampler());
        let river_edits: Vec<TileEdit> =
            plan_rivers(shared_test_sampler(), &terrain, &config, &base)
                .by_chunk
                .iter()
                .flatten()
                .copied()
                .collect();
        let watered = base.with_edits(&river_edits);
        let drain_edits: Vec<TileEdit> =
            plan_drainage(shared_test_sampler(), &terrain, &config, &watered)
                .by_chunk
                .iter()
                .flatten()
                .copied()
                .collect();
        let world = watered.with_edits(&drain_edits);

        // The cities, so the sweep can report the figure the knob is actually chosen
        // against: a seam nobody can reach is one the simulation never sees, so what
        // matters is the share of *cities* that hold one, not the share of the map.
        let cities = plan_cities(shared_test_sampler(), &terrain, &config, &world);
        let shipped_reach = IndustryConfig::default().estate_reach_tiles as i32;

        // The sweep the `deposit_cell_tiles` doc comment reports. One world, many
        // layouts — the generation is what costs, and the layout itself is cheap.
        //
        // The figure it is chosen against is the **share of cities holding a seam**,
        // not the share of the map, and the two are only loosely related: cities sit
        // on habitable ground and seams sit on bare, high or dry ground, so the two
        // populations are anti-correlated and a city reaches a seam far less often
        // than a uniform scatter of the same density would suggest.
        println!(
            "\n{} cities in the world, shipping an estate reach of {shipped_reach}",
            cities.len()
        );
        for reach in [56i32, 80, 112] {
            println!(
                "\n  estate reach {reach} ({} tiles in a city's disc)\n  \
                 cell   seams    iron  copper    salt   richness    0 seams   1 kind   2+ kinds",
                (std::f32::consts::PI * (reach * reach) as f32) as u32
            );
            for cell in [128u32, 64, 48, 32, 24] {
                let sites = plan_deposits(
                    shared_test_sampler(),
                    &WorldPlanConfig {
                        deposit_cell_tiles: cell,
                        ..config.clone()
                    },
                    &world,
                );
                let count = |resource| {
                    sites
                        .iter()
                        .filter(|site| site.resource == resource)
                        .count()
                };
                let mean = if sites.is_empty() {
                    0.0
                } else {
                    sites.iter().map(|s| s.richness).sum::<f32>() / sites.len() as f32
                };

                // First come, in id order, exactly as the seeding claims them — a seam
                // has one owner, so two cities in range of the same one do not both
                // count it.
                let mut owned = vec![false; sites.len()];
                let mut kinds_held: Vec<usize> = Vec::with_capacity(cities.len());
                for planned in &cities {
                    let mut held = [false; RESOURCE_COUNT];
                    for (index, site) in sites.iter().enumerate() {
                        if !owned[index]
                            && (site.tile - planned.city.centre).length_squared() <= reach * reach
                        {
                            owned[index] = true;
                            held[site.resource.index()] = true;
                        }
                    }
                    kinds_held.push(held.iter().filter(|h| **h).count());
                }
                let share = |predicate: fn(usize) -> bool| {
                    kinds_held.iter().filter(|k| predicate(**k)).count() * 100 / cities.len().max(1)
                };

                println!(
                    "  {cell:>4}   {:>5}   {:>5}   {:>5}   {:>5}      {mean:.3}       {:>3}%     {:>3}%      {:>3}%",
                    sites.len(),
                    count(Resource::Iron),
                    count(Resource::Copper),
                    count(Resource::Salt),
                    share(|k| k == 0),
                    share(|k| k == 1),
                    share(|k| k >= 2),
                );
            }
        }

        let sites = plan_deposits(shared_test_sampler(), &config, &world);
        assert!(!sites.is_empty(), "the world has no seams at all");

        // Every resource has to exist somewhere, or a recipe's band is set past what
        // the terrain produces and one third of the feature is dead.
        for resource in [Resource::Iron, Resource::Copper, Resource::Salt] {
            let count = sites.iter().filter(|s| s.resource == resource).count();
            assert!(count > 0, "not one seam of {resource:?} in the whole world");
        }

        // And a seam is on ground its own recipe names, which is what makes "never on
        // water, in a river, or on a tile the plan stamped" a fact about this world
        // rather than only about the table.
        for site in &sites {
            let kind = world.tile(site.tile).expect("inside the world");
            let recipe = RECIPES
                .iter()
                .find(|r| r.resource == site.resource)
                .expect("every laid seam has a recipe");
            assert!(
                recipe.ground.contains(&kind),
                "{:?} at {} sits on {kind:?}",
                site.resource,
                site.tile
            );
            assert!((0.0..=1.0).contains(&site.richness));
        }
    }
}
