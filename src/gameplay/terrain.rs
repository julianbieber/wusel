//! Turns a position in the world into a terrain kind.
//!
//! Everything here is a pure function of `(TerrainConfig, global tile position)`,
//! and — since cities and roads moved out to [`crate::gameplay::plan`] — a pure
//! function of *that tile alone*. No rule here looks at a neighbouring tile, so
//! a chunk needs no padding and a tile cannot depend on where the chunk boundary
//! fell. The biome lookup reads neighbouring *cells*, which are themselves a
//! function of their own coordinates, so that still holds.
//!
//! The height is built in layers rather than sampled from one field, because one
//! field was the problem: at `relief_scale` the longest wavelength is ~25 tiles,
//! which gives a mottled archipelago with no continents and no interiors, and no
//! threshold can cut regions out of a field that has no structure at the scale a
//! player moves at. So:
//!
//! - **continent** — an order of magnitude coarser, and what makes land masses;
//! - **relief** — the old elevation field, demoted to the bumps on top;
//! - **ridged** — one-sided and crease-shaped, which is what makes a mountain
//!   *range* instead of a field of separate lumps. Only [`Biome::Highland`] leans
//!   on it, so it is skipped wherever the blended weight is zero — most of the
//!   world.
//!
//! How those three combine is not fixed: it is the [`HeightRecipe`] that
//! [`crate::gameplay::biome`] blends for the tile. That is where the variety comes
//! from.
//!
//! [`TerrainSampler`] is the **only** implementation of "how high, how wet is it
//! here". That matters outside this module: [`crate::gameplay::river`] walks its
//! particles downhill and [`crate::gameplay::road`] costs its steps against the
//! same answer, so a second implementation would put rivers running up the visible
//! hills. It is also why the config hands out a sampler rather than a bare
//! `NoiseField` as it used to.

use bevy::prelude::*;

use crate::gameplay::biome::{Biome, BiomeMap, HeightRecipe};
use crate::gameplay::noise::{NoiseField, RidgedNoiseField};

/// The twelve tiles of `assets/textures/terrain.png`, in atlas column order — the
/// discriminant *is* the tileset index, so the two can never drift apart.
///
/// `Town`, `Road` and `River` are never produced here: they are stamped over the
/// base terrain once the whole world exists, by [`crate::gameplay::plan`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum TerrainKind {
    Forest = 0,
    ShallowWater = 1,
    Grass = 2,
    Town = 3,
    Mountain = 4,
    DeepWater = 5,
    Road = 6,
    River = 7,
    Sand = 8,
    Snow = 9,
    Rock = 10,
    Marsh = 11,
}

/// Number of layers the terrain atlas is split into.
pub const TERRAIN_KIND_COUNT: u32 = 12;

impl TerrainKind {
    pub fn tileset_index(self) -> u16 {
        self as u16
    }

    /// Only these two kinds can be built on — a city is clipped by coast,
    /// mountain and river rather than paving them.
    ///
    /// The four kinds the biomes added are deliberately *not* habitable, and that
    /// is most of what makes a desert or a marsh feel different to walk into: no
    /// city is founded on sand, marsh, scree or snow, so those regions are empty of
    /// everything the plan would otherwise put there.
    pub fn is_habitable(self) -> bool {
        matches!(self, TerrainKind::Forest | TerrainKind::Grass)
    }

    /// What a road may never cross. `River` is deliberately not in here: a river
    /// runs from the mountains to the sea, so refusing it outright would cut the
    /// continent into pieces the road network cannot span. A road crosses one at
    /// a price instead — see `road_river_crossing_penalty`.
    ///
    /// A lake is `ShallowWater` and so is covered by this without lakes being a
    /// kind of their own.
    pub fn is_water(self) -> bool {
        matches!(self, TerrainKind::ShallowWater | TerrainKind::DeepWater)
    }
}

/// Thresholds and noise scales that decide what a tile becomes.
///
/// The settlement and humidity figures are not read here at all — nothing in
/// this module samples those fields. They live here because they describe the
/// same landscape as the rest; [`crate::gameplay::city`],
/// [`crate::gameplay::river`] and [`crate::gameplay::weather`] are their readers.
#[derive(Resource, Clone)]
pub struct TerrainConfig {
    pub seed: u32,
    /// The low-frequency layer that makes land masses. An order of magnitude
    /// coarser than `relief_scale`, which is the whole point: at 0.0015 the
    /// longest wavelength is ~670 tiles, so a continent is something you cross
    /// rather than something you can see the whole of.
    pub continent_scale: f32,
    /// How far the continent layer displaces a recipe's `base_height`, either way.
    /// This is the knob on how much of the world is sea: at the default it moves a
    /// `Plains` interior (0.52) down to 0.35 at its lowest, which is under water.
    pub continent_relief: f32,
    /// The old `elevation_scale`, demoted from "the terrain" to the bumps on top of
    /// it. Unchanged at 0.04 — as *relief* a 25-tile wavelength is right; it was
    /// only ever wrong as the whole landscape.
    pub relief_scale: f32,
    /// The ridged layer, which only a `Highland` recipe draws on. ~83 tiles between
    /// creases, so a range is a few ridges wide rather than one.
    pub ridge_scale: f32,
    pub vegetation_scale: f32,
    pub settlement_scale: f32,
    pub humidity_scale: f32,
    /// The Voronoi lattice the biomes are drawn on. 384 tiles is six chunks: two or
    /// three regions across the screen at `MAX_ZOOM_SCALE`, and a walk of about a
    /// minute to cross one.
    pub biome_cell_tiles: u32,
    /// How wide the band is in which two recipes mix. Everything further than this
    /// from a boundary carries exactly one recipe, which is what gives a region an
    /// interior instead of making the whole world an average of all six.
    pub biome_blend_tiles: u32,
    /// How far the biome lookup position is warped before the nearest-site test, so
    /// a region has an organic outline rather than a polygon's.
    ///
    /// A Voronoi edge is a perpendicular bisector — a straight line — and this is the
    /// only thing that bends one, which makes it the knob on "the biome edges look
    /// ruled". It trades outline crookedness against how big a region feels, and
    /// `the_warp_trades_region_size_for_a_crooked_outline` measures both:
    ///
    /// ```text
    ///   warp    crooked   mean region run
    ///      0      1.00x         406 tiles
    ///    160      1.46x         257 tiles
    ///    260      2.00x         182 tiles
    ///    440      3.20x         119 tiles
    /// ```
    ///
    /// 160 is a visibly crooked outline while a region is still ~4 chunks across.
    /// Note the amplitude alone is not enough — a warp only bends an edge where it
    /// has content at a wavelength shorter than that edge, which is what
    /// `biome::WARP_CELLS` and `WARP_OCTAVES` are for.
    pub biome_warp_tiles: f32,
    /// Elevation bands, in ascending order; anything above `lowland_max` is mountain.
    pub deep_water_max: f32,
    pub shallow_water_max: f32,
    pub lowland_max: f32,
    /// Above the mountain band the ground goes bare and then white. Both are only
    /// reached where a `Highland` recipe's ridged layer piles up, so they mark the
    /// ranges rather than appearing wherever the land happens to be high.
    pub scree_min: f32,
    pub snow_min: f32,
    /// Vegetation at or above this turns a lowland tile from its recipe's dry kind
    /// to its wet one.
    pub forest_threshold: f32,
    /// A candidate city site must clear this settlement score to be founded.
    pub town_threshold: f32,
    pub town_coast_bonus: f32,
    pub coast_radius: u32,
    /// How wet a mountain must be for a river to rise there. Together with
    /// `WorldPlanConfig::river_source_cell_tiles` this is the lever on how many
    /// rivers the world has: at the default spacing 0.6 gives 3236 springs, 0.55
    /// gives 4466 and 0.5 gives 5826.
    pub river_source_threshold: f32,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            seed: 0x5eed,
            continent_scale: 0.0015,
            continent_relief: 0.34,
            relief_scale: 0.04,
            ridge_scale: 0.012,
            vegetation_scale: 0.09,
            settlement_scale: 0.12,
            // Coarser than the vegetation field: weather covers more ground than
            // a wood does, so a whole range is wet rather than one peak in it.
            humidity_scale: 0.02,
            biome_cell_tiles: 384,
            biome_blend_tiles: 48,
            biome_warp_tiles: 160.0,
            deep_water_max: 0.32,
            shallow_water_max: 0.42,
            lowland_max: 0.72,
            scree_min: 0.78,
            snow_min: 0.88,
            forest_threshold: 0.5,
            town_threshold: 0.62,
            town_coast_bonus: 0.06,
            coast_radius: 2,
            river_source_threshold: 0.55,
        }
    }
}

/// Salts that give each field its own patch of the noise lattice.
const ELEVATION_SALT: u32 = 0x0000_0001;
const VEGETATION_SALT: u32 = 0x9e37_79b9;
const SETTLEMENT_SALT: u32 = 0x85eb_ca6b;
const HUMIDITY_SALT: u32 = 0xc2b2_ae35;
const CONTINENT_SALT: u32 = 0x27d4_eb2d;
const RIDGE_SALT: u32 = 0x1656_67b1;

/// The continent layer is there for its longest wavelength, so octaves finer than
/// the relief layer already provides are paid for on every tile and then buried
/// under it.
const CONTINENT_OCTAVES: u32 = 3;

/// Enough to shape a range without the crease pattern turning into noise.
const RIDGE_OCTAVES: u32 = 4;

/// Below this blended ridge weight the ridged layer is not sampled at all. It is
/// the expensive layer and only one recipe draws on it, so most of the world skips
/// it — that is what pays for the two layers this rework added.
const RIDGE_EPSILON: f32 = 1e-3;

impl TerrainConfig {
    /// What makes one habitable tile a likelier city site than another.
    ///
    /// The one field still handed out raw: [`crate::gameplay::city`] compares
    /// settlement scores between candidate sites and nothing biome-dependent enters
    /// into it.
    pub fn settlement_field(&self) -> NoiseField {
        NoiseField::new(self.seed, SETTLEMENT_SALT, self.settlement_scale)
    }

    /// The sampler every reader of the landscape has to go through.
    pub fn sampler(&self) -> TerrainSampler {
        TerrainSampler::new(self)
    }
}

/// What one tile turned out to be, before the bands cut it into a kind.
///
/// The blended recipe rides along because the bands need its `beach_width` and its
/// kind pair — the classification is biome-dependent, not just elevation-dependent.
pub struct TileSample {
    pub elevation: f32,
    pub vegetation: f32,
    /// Which region the tile is in. Not what the bands read — that is `cover` — so
    /// nothing outside the coverage and outline measurements has a use for it, and
    /// the allow is scoped to say exactly that rather than to silence the lint
    /// generally.
    #[cfg_attr(not(test), allow(dead_code))]
    pub dominant: Biome,
    /// Which biome supplies the tile's ground cover, drawn from the blend weights.
    /// Equal to `dominant` everywhere except inside a blend band.
    pub cover: Biome,
    pub recipe: HeightRecipe,
}

/// The one answer to "how high, how green, how wet is it here".
///
/// Built once and sampled many times: constructing it builds six noise fields and a
/// biome map, which is why callers hoist it out of their loops.
pub struct TerrainSampler {
    biomes: BiomeMap,
    continent_field: NoiseField,
    relief_field: NoiseField,
    ridge_field: RidgedNoiseField,
    vegetation_field: NoiseField,
    humidity_field: NoiseField,
    continent_relief: f32,
}

impl TerrainSampler {
    pub fn new(config: &TerrainConfig) -> Self {
        Self {
            biomes: BiomeMap::new(
                config.seed,
                config.biome_cell_tiles,
                config.biome_blend_tiles,
                config.biome_warp_tiles,
            ),
            continent_field: NoiseField::with_octaves(
                config.seed,
                CONTINENT_SALT,
                config.continent_scale,
                CONTINENT_OCTAVES,
            ),
            relief_field: NoiseField::new(config.seed, ELEVATION_SALT, config.relief_scale),
            ridge_field: RidgedNoiseField::new(
                config.seed,
                RIDGE_SALT,
                config.ridge_scale,
                RIDGE_OCTAVES,
            ),
            vegetation_field: NoiseField::new(
                config.seed,
                VEGETATION_SALT,
                config.vegetation_scale,
            ),
            humidity_field: NoiseField::new(config.seed, HUMIDITY_SALT, config.humidity_scale),
            continent_relief: config.continent_relief,
        }
    }

    /// The three layers, combined the way this tile's recipe says to.
    ///
    /// The continent and relief layers are displacements about zero, so they can
    /// lower the ground as well as raise it; the ridged layer is one-sided, so a
    /// range only ever builds up out of the terrain it sits on.
    fn height(&self, recipe: &HeightRecipe, x: f32, y: f32) -> f32 {
        let continent = self.continent_field.sample(x, y) - 0.5;
        let relief = self.relief_field.sample(x, y) - 0.5;

        let mut height =
            recipe.base_height + continent * self.continent_relief + relief * recipe.relief;

        if recipe.ridge > RIDGE_EPSILON {
            height += self.ridge_field.sample(x, y) * recipe.ridge;
        }

        height.clamp(0.0, 1.0)
    }

    /// How high the terrain is at a tile. Read by [`crate::gameplay::river`], whose
    /// particles walk downhill, and by [`crate::gameplay::road`], which costs a step
    /// by how much it climbs — `WorldMap` records only which band a tile fell in,
    /// not how high it is.
    pub fn elevation(&self, x: f32, y: f32) -> f32 {
        self.height(&self.biomes.blend(x, y).recipe, x, y)
    }

    /// How much rain falls at a tile, biased by the biome. Read by
    /// [`crate::gameplay::river`], to decide which mountains are wet enough for a
    /// river to rise in them, and by [`crate::gameplay::weather`], which bakes it
    /// into the map that says where the clouds are — so it rains over the country
    /// the rivers rise in, and a desert gets neither.
    pub fn humidity(&self, x: f32, y: f32) -> f32 {
        let bias = self.biomes.blend(x, y).recipe.humidity_bias;
        (self.humidity_field.sample(x, y) + bias).clamp(0.0, 1.0)
    }

    /// Everything about a tile, for the one caller that needs all of it.
    pub fn sample(&self, x: f32, y: f32) -> TileSample {
        let blended = self.biomes.blend(x, y);

        TileSample {
            elevation: self.height(&blended.recipe, x, y),
            vegetation: (self.vegetation_field.sample(x, y) + blended.recipe.vegetation_bias)
                .clamp(0.0, 1.0),
            dominant: blended.dominant,
            cover: blended.cover,
            recipe: blended.recipe,
        }
    }
}

/// Cuts a sampled tile into a kind.
///
/// Elevation alone still decides water from land from mountain — the bands are the
/// same four they were. What the biome changes is *what fills a band*: the lowland
/// pair comes from the recipe rather than being Grass and Forest everywhere, and
/// the sand band above the water line is as wide as the recipe says, which is zero
/// for a `Highland` coast and so gives a cliff instead of a beach.
fn classify(config: &TerrainConfig, sample: &TileSample) -> TerrainKind {
    let elevation = sample.elevation;

    if elevation < config.deep_water_max {
        TerrainKind::DeepWater
    } else if elevation < config.shallow_water_max {
        TerrainKind::ShallowWater
    } else if elevation < config.shallow_water_max + sample.recipe.beach_width {
        TerrainKind::Sand
    } else if elevation < config.lowland_max {
        let (dry, wet) = sample.cover.kinds();
        if sample.vegetation >= config.forest_threshold {
            wet
        } else {
            dry
        }
    } else if elevation < config.scree_min {
        TerrainKind::Mountain
    } else if elevation < config.snow_min {
        TerrainKind::Rock
    } else {
        TerrainKind::Snow
    }
}

/// Generates the kind of every tile in the chunk whose lower-left tile sits at
/// the global tile coordinate `origin`, in row-major order from that corner.
///
/// Every sample is taken in global tile space, never chunk-locally — that, and
/// the fact that no rule here reads a neighbouring tile, is what lets the world
/// be cut into chunks at all.
///
/// This is the only expensive call in the crate, and this rework made it **2.1x**
/// dearer: measured on one machine, the old two-field pipeline was 1.75 ms per 64x64
/// chunk against 3.70 ms now, which scales the ~4 ms the rest of the crate's notes
/// were written against to ~8.5 ms, and the whole-world background pass from ~16 s
/// to ~34 s. `the_default_config_produces_recognisably_different_regions` is how
/// that is taken. It is why [`crate::gameplay::world`] keeps this off the main thread
/// wherever it can, and why the blocking budget there had to come down.
pub fn generate_chunk(
    config: &TerrainConfig,
    origin: IVec2,
    chunk_size: UVec2,
) -> Box<[TerrainKind]> {
    // Hoisted: building one costs six noise fields and a biome map, and every tile
    // in the chunk wants the same one.
    let sampler = config.sampler();

    (0..chunk_size.element_product())
        .map(|i| {
            let x = (origin.x + (i % chunk_size.x) as i32) as f32;
            let y = (origin.y + (i / chunk_size.x) as i32) as f32;
            classify(config, &sampler.sample(x, y))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::gameplay::world::WORLD_TILES;

    const CHUNK: UVec2 = UVec2::splat(64);
    /// Somewhere in the middle of the world rather than at the origin, so the
    /// tests exercise the same coordinate magnitudes the game actually uses.
    const ORIGIN: IVec2 = IVec2::new(2048, 2048);

    fn kinds(config: &TerrainConfig) -> Box<[TerrainKind]> {
        generate_chunk(config, ORIGIN, CHUNK)
    }

    #[test]
    fn every_tile_gets_an_index_within_the_atlas() {
        let tiles = kinds(&TerrainConfig::default());
        assert_eq!(tiles.len() as u32, CHUNK.element_product());
        assert!(
            tiles
                .iter()
                .all(|k| (k.tileset_index() as u32) < TERRAIN_KIND_COUNT)
        );
    }

    #[test]
    fn generation_is_a_pure_function_of_position_and_config() {
        let config = TerrainConfig::default();
        assert_eq!(kinds(&config), kinds(&config));
    }

    #[test]
    fn a_different_seed_yields_a_different_map() {
        let config = TerrainConfig::default();
        let other = TerrainConfig {
            seed: config.seed ^ 0xabcd,
            ..config.clone()
        };
        assert_ne!(kinds(&config), kinds(&other));
    }

    /// The load-bearing property for a chunked world: two chunks that overlap a
    /// region must agree on it, which they only do because the samples are taken
    /// in global space and padded past the chunk border.
    #[test]
    fn a_tile_does_not_depend_on_where_the_chunk_boundary_falls() {
        let config = TerrainConfig::default();
        let shift = IVec2::new(37, 11);

        let base = generate_chunk(&config, ORIGIN, CHUNK);
        let shifted = generate_chunk(&config, ORIGIN + shift, CHUNK);

        for y in 0..CHUNK.y as i32 - shift.y {
            for x in 0..CHUNK.x as i32 - shift.x {
                let in_base = (y + shift.y) * CHUNK.x as i32 + (x + shift.x);
                let in_shifted = y * CHUNK.x as i32 + x;
                assert_eq!(
                    base[in_base as usize],
                    shifted[in_shifted as usize],
                    "tile ({}, {}) disagrees between the two chunks covering it",
                    ORIGIN.x + x + shift.x,
                    ORIGIN.y + y + shift.y,
                );
            }
        }
    }

    /// The thresholds are only useful if the default config actually produces a
    /// mixed map — a single-biome world would pass every other test here.
    ///
    /// **World scale, not one chunk.** This used to sample the chunk at `ORIGIN`,
    /// and it cannot any more: with a 384-tile biome cell a single 64x64 chunk is
    /// *supposed* to be nearly one biome, so a chunk containing all seven of these
    /// would mean the regions had not worked. That property is now its own test,
    /// `a_chunk_away_from_a_boundary_is_nearly_one_biome`.
    #[test]
    fn the_default_config_produces_every_base_kind() {
        let config = TerrainConfig::default();
        let counts = world_kind_counts(&config);

        for kind in [
            TerrainKind::Forest,
            TerrainKind::ShallowWater,
            TerrainKind::Grass,
            TerrainKind::Mountain,
            TerrainKind::DeepWater,
            TerrainKind::Sand,
            TerrainKind::Snow,
            TerrainKind::Rock,
            TerrainKind::Marsh,
        ] {
            assert!(
                counts[kind.tileset_index() as usize] > 0,
                "no {kind:?} anywhere in the world"
            );
        }
    }

    /// A tile drawn for a biome no recipe can reach is a tile drawn for nothing.
    /// Four were added for this rework and all four have to earn their column.
    #[test]
    fn each_kind_the_biomes_added_covers_a_real_share_of_the_world() {
        let config = TerrainConfig::default();
        let counts = world_kind_counts(&config);
        let total: u64 = counts.iter().sum();

        for kind in [
            TerrainKind::Sand,
            TerrainKind::Snow,
            TerrainKind::Rock,
            TerrainKind::Marsh,
        ] {
            let share = counts[kind.tileset_index() as usize] as f64 / total as f64;
            assert!(
                share > 0.001,
                "{kind:?} covers {:.4}% of the world, which is not enough to see",
                share * 100.0
            );
        }
    }

    /// Six biomes are only worth having if the world actually contains six. The
    /// upper bound is loose on purpose: `Ocean` is the sea and is *meant* to be the
    /// largest single region — it measures 32.9% — so the cap is there to catch one
    /// biome swallowing the world, not to hold the sea to a share.
    #[test]
    fn the_world_contains_every_biome_and_is_dominated_by_none() {
        let config = TerrainConfig::default();
        let sampler = config.sampler();

        let mut counts = [0u32; 6];
        let side = WORLD_TILES.x / 32;
        for y in 0..side {
            for x in 0..side {
                let dominant = sampler.sample((x * 32) as f32, (y * 32) as f32).dominant;
                let slot = Biome::ALL.iter().position(|b| *b == dominant).unwrap();
                counts[slot] += 1;
            }
        }

        let total: u32 = counts.iter().sum();
        for (slot, count) in counts.iter().enumerate() {
            let share = *count as f32 / total as f32;
            assert!(
                (0.02..0.40).contains(&share),
                "{:?} covers {:.2}% of the world",
                Biome::ALL[slot],
                share * 100.0
            );
        }
    }

    /// The other half of "distinct regions": a chunk in the interior of a region
    /// must be dominated by that region's own kinds. A world that scored well on
    /// variety by mixing every kind into every chunk would be the mottle this
    /// rework replaced.
    #[test]
    fn a_chunk_away_from_a_boundary_is_nearly_one_biome() {
        let config = TerrainConfig::default();
        let sampler = config.sampler();

        // Chunks spread across the world; each is scored only if its own centre is
        // deep enough inside a region for the claim to be about an interior.
        let mut interiors = 0;
        for step in 0..64 {
            let origin = IVec2::new((step % 8) * 512 + 96, (step / 8) * 512 + 96);
            let centre = origin + IVec2::splat(32);
            let here = sampler.sample(centre.x as f32, centre.y as f32).dominant;

            let tiles = generate_chunk(&config, origin, CHUNK);
            let mut agreeing = 0;
            for i in 0..tiles.len() {
                let tile =
                    origin + IVec2::new((i as u32 % CHUNK.x) as i32, (i as u32 / CHUNK.x) as i32);
                if sampler.sample(tile.x as f32, tile.y as f32).dominant == here {
                    agreeing += 1;
                }
            }

            if agreeing == tiles.len() {
                interiors += 1;
            }
        }

        assert!(
            interiors >= 32,
            "only {interiors}/64 sampled chunks fall wholly inside one region"
        );
    }

    /// Where `biome_warp_tiles` comes from: outline crookedness against the price
    /// paid for it, which is how big a region feels.
    ///
    /// The mean run length is the guard, and it had to replace the obvious one — the
    /// interior fraction sits at ~56% across this entire range, because warping the
    /// query is locally structure-preserving and so cannot destroy an interior no
    /// matter how hard it is pushed. It looks like a fragmentation guard and is not
    /// one. Run length is: at zero warp it reads 406 tiles against a 384-tile cell,
    /// which is the check that it measures what it claims to.
    ///
    /// `cargo test --release -- --ignored --nocapture the_warp_trades`
    #[test]
    #[ignore]
    fn the_warp_trades_region_size_for_a_crooked_outline() {
        let base = TerrainConfig::default();
        let straight = region_perimeter(&TerrainConfig {
            biome_warp_tiles: 0.0,
            ..base.clone()
        });

        println!("\n  warp  crooked  interior  stipple  run(tiles)");
        for warp in [0.0f32, 96.0, 128.0, 160.0, 200.0, 260.0, 340.0, 440.0] {
            let config = TerrainConfig {
                biome_warp_tiles: warp,
                ..base.clone()
            };
            let sampler = config.sampler();

            let mut interior = 0;
            let mut stipple = 0;
            let n = 16384;
            for i in 0..n {
                let (x, y) = ((i * 13 % 4096) as f32, (i * 211 % 4096) as f32);
                let sample = sampler.sample(x, y);
                if (sample.recipe.base_height - sample.dominant.recipe().base_height).abs() < 1e-4 {
                    interior += 1;
                }
                if sample.cover != sample.dominant {
                    stipple += 1;
                }
            }

            // Mean length of a same-region run along a scanline. This is the
            // fragmentation guard the interior fraction turned out not to be: warping
            // is locally structure-preserving, so interiors survive folding, but
            // regions shattering into speckle would collapse the run length.
            let step = 8;
            let side = WORLD_TILES.x / step;
            let mut runs = 1u32;
            let mut cells = 0u32;
            for y in (0..side).step_by(16) {
                let mut previous = sampler.sample(0.0, (y * step) as f32).dominant;
                for x in 1..side {
                    let here = sampler
                        .sample((x * step) as f32, (y * step) as f32)
                        .dominant;
                    if here != previous {
                        runs += 1;
                        previous = here;
                    }
                    cells += 1;
                }
            }

            println!(
                "  {warp:>4.0}  {:>7.2}x  {:>7.1}%  {:>6.1}%  {:>9.0}",
                region_perimeter(&config) as f64 / straight as f64,
                interior as f64 / n as f64 * 100.0,
                stipple as f64 / n as f64 * 100.0,
                cells as f64 * step as f64 / runs as f64,
            );
        }
        println!();
    }

    /// Total length of every region outline in the world, in boundary steps: pairs of
    /// adjacent sample points whose region differs.
    ///
    /// Sampled on a 16-tile grid, so it measures the outline of the *regions* and not
    /// the per-tile stipple the cover dither adds inside a band — `dominant`, never
    /// `cover`, for exactly that reason.
    fn region_perimeter(config: &TerrainConfig) -> u32 {
        let sampler = config.sampler();
        let step = 16;
        let side = WORLD_TILES.x / step;

        // Both axes, which is the whole point: counting only the horizontal pairs
        // measures how many times a walk to the right *crosses* a boundary, and a
        // vertical edge is crossed exactly once per row however much it wiggles. The
        // vertical pairs are what a wiggle actually shows up in.
        let mut steps = 0;
        for y in 0..side - 1 {
            for x in 0..side - 1 {
                let here = sampler
                    .sample((x * step) as f32, (y * step) as f32)
                    .dominant;
                let right = sampler
                    .sample(((x + 1) * step) as f32, (y * step) as f32)
                    .dominant;
                let below = sampler
                    .sample((x * step) as f32, ((y + 1) * step) as f32)
                    .dominant;
                if here != right {
                    steps += 1;
                }
                if here != below {
                    steps += 1;
                }
            }
        }
        steps
    }

    /// The whole world's kind histogram, indexed by tileset index.
    ///
    /// Sampled every 8th tile on both axes rather than exhaustively: 262k samples
    /// is enough for a coverage claim and runs in a debug test, where all 16 M
    /// would not.
    fn world_kind_counts(config: &TerrainConfig) -> [u64; TERRAIN_KIND_COUNT as usize] {
        let sampler = config.sampler();
        let mut counts = [0u64; TERRAIN_KIND_COUNT as usize];

        let side = WORLD_TILES.x / 8;
        for y in 0..side {
            for x in 0..side {
                let (tx, ty) = ((x * 8) as f32, (y * 8) as f32);
                let kind = classify(config, &sampler.sample(tx, ty));
                counts[kind.tileset_index() as usize] += 1;
            }
        }

        counts
    }

    /// The three stamped kinds belong to the plan, not to the terrain — if one
    /// ever came out of here, a chunk's contents would depend on its neighbours
    /// again.
    #[test]
    fn the_terrain_never_produces_a_town_a_road_or_a_river() {
        let tiles = kinds(&TerrainConfig::default());
        assert!(!tiles.contains(&TerrainKind::Town));
        assert!(!tiles.contains(&TerrainKind::Road));
        assert!(!tiles.contains(&TerrainKind::River));
    }

    /// Rivers rise where it rains, so the humidity field has to be its own
    /// landscape rather than a second view of the elevation it is sampled
    /// alongside.
    #[test]
    fn humidity_is_independent_of_the_other_fields() {
        let config = TerrainConfig::default();
        let sampler = config.sampler();

        let differs = (0..64).filter(|i| {
            let (x, y) = ((ORIGIN.x + i * 7) as f32, (ORIGIN.y + i * 13) as f32);
            let sample = sampler.sample(x, y);
            let humidity = sampler.humidity(x, y);
            (humidity - sample.elevation).abs() > 0.05
                && (humidity - sample.vegetation).abs() > 0.05
        });
        assert!(differs.count() > 48, "humidity tracks another field");
    }

    /// Elevation is the layer the rivers and the roads read, so its continuity is
    /// theirs too: a particle that walks downhill across a biome boundary must not
    /// find a cliff there that the drawn terrain does not have.
    #[test]
    fn elevation_does_not_step_at_a_biome_boundary() {
        let config = TerrainConfig::default();
        let sampler = config.sampler();

        // The relief layer alone moves by at most this much between adjacent tiles:
        // one tile is `relief_scale` of a noise cell, and the field is gain-stretched.
        let relief_step = config.relief_scale * 2.6;
        let mut worst = 0.0f32;

        for i in 0..8192 {
            let (x, y) = ((i * 13 % 4096) as f32, (i * 211 % 4096) as f32);
            let here = sampler.elevation(x, y);
            worst = worst.max((here - sampler.elevation(x + 1.0, y)).abs());
            worst = worst.max((here - sampler.elevation(x, y + 1.0)).abs());
        }

        assert!(
            worst < relief_step,
            "elevation steps by {worst} between adjacent tiles, over the {relief_step} the relief layer alone allows"
        );
    }

    /// Where the numbers in the config doc comments come from. Not a check — it
    /// prints what the default world is actually made of, which is how the recipe
    /// table was tuned.
    ///
    /// `cargo test --release -- --ignored --nocapture the_default_config_produces_recognisably`
    #[test]
    #[ignore]
    fn the_default_config_produces_recognisably_different_regions() {
        let config = TerrainConfig::default();
        let counts = world_kind_counts(&config);
        let total: u64 = counts.iter().sum();

        println!("\ntile coverage over the whole world:");
        for (index, count) in counts.iter().enumerate() {
            if *count == 0 {
                continue;
            }
            println!(
                "  {:>13?}  {:>6.2}%",
                KIND_BY_INDEX[index],
                *count as f64 / total as f64 * 100.0
            );
        }

        let water = counts[TerrainKind::DeepWater.tileset_index() as usize]
            + counts[TerrainKind::ShallowWater.tileset_index() as usize];
        println!(
            "  water total  {:>6.2}%",
            water as f64 / total as f64 * 100.0
        );

        let sampler = config.sampler();
        let mut biomes = [0u64; Biome::ALL.len()];
        let side = WORLD_TILES.x / 16;
        for y in 0..side {
            for x in 0..side {
                let dominant = sampler.sample((x * 16) as f32, (y * 16) as f32).dominant;
                let slot = Biome::ALL.iter().position(|b| *b == dominant).unwrap();
                biomes[slot] += 1;
            }
        }
        let biome_total: u64 = biomes.iter().sum();
        println!("biome coverage:");
        for (slot, count) in biomes.iter().enumerate() {
            println!(
                "  {:>10?}  {:>6.2}%",
                Biome::ALL[slot],
                *count as f64 / biome_total as f64 * 100.0
            );
        }

        // How crooked the region outlines are. A straight edge is the shortest path
        // between two points, so for a fixed set of regions a longer total perimeter
        // is a more crooked one — which makes this the objective version of "the
        // edges look too straight". Compared against the same world with the warp
        // switched off, which is pure Voronoi and therefore perfectly straight
        // segments, so the ratio is what the warp is buying.
        let straight = TerrainConfig {
            biome_warp_tiles: 0.0,
            ..config.clone()
        };
        let warped_perimeter = region_perimeter(&config);
        let straight_perimeter = region_perimeter(&straight);
        println!(
            "region outline: {straight_perimeter} boundary steps unwarped, {warped_perimeter} warped \
             — {:.2}x as crooked",
            warped_perimeter as f64 / straight_perimeter as f64
        );

        // Spread over the whole world rather than along a diagonal, so the figure
        // includes the highland chunks that pay for the ridged layer. The kinds are
        // summed and printed for one reason only: without a use for the result the
        // optimiser is free to drop the generation and time an allocation.
        let started = std::time::Instant::now();
        let chunks = 64;
        let mut checksum = 0u64;
        for i in 0..chunks {
            let origin = IVec2::new((i % 8) * 512, (i / 8) * 512);
            for kind in generate_chunk(&config, origin, CHUNK).iter() {
                checksum += kind.tileset_index() as u64;
            }
        }
        println!(
            "\ngenerate_chunk: {:.2} ms per 64x64 chunk (checksum {checksum})\n",
            started.elapsed().as_secs_f64() * 1000.0 / chunks as f64
        );
    }

    /// Atlas column order, for printing a histogram by index.
    const KIND_BY_INDEX: [TerrainKind; TERRAIN_KIND_COUNT as usize] = [
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
    ];
}
