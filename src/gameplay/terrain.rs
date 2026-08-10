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

use std::sync::Arc;

use bevy::prelude::*;
use watershed::Terrain;

use crate::gameplay::biome::{BIOME_TABLE, Biome, HeightRecipe};
use crate::gameplay::document;
use crate::gameplay::world::WORLD_TILES;

/// The fifteen tiles of `assets/textures/terrain.png`, in atlas column order — the
/// discriminant *is* the tileset index, so the two can never drift apart.
///
/// `Town`, `Road` and `River` are never produced here: they are stamped over the
/// base terrain once the whole world exists, by [`crate::gameplay::plan`].
///
/// Columns 12..14 were appended for gh-14, so nothing existing moved. `Scrub` is
/// the rung between bare ground and grass that lets the lowland band be a
/// three-step ladder instead of a binary; `Gravel` is warm deflated hardpan and
/// deliberately not `Rock`, which is cold alpine scree and reads wrong at sea
/// level; `Reed` is the wetland's middle step, and what a damp channel promotes
/// `Marsh` into.
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
    Scrub = 12,
    Gravel = 13,
    Reed = 14,
    /// Worked land around a city. Stamped by [`crate::gameplay::growth`], never
    /// generated — the same standing rule `Town` and `Road` are under.
    Farmland = 15,
}

/// Number of layers the terrain atlas is split into, which is also the number of
/// columns in `assets/textures/terrain.png` and the `width_in_tiles` its
/// `terrain.atlas.json` sidecar reports. The three have to agree: the strip is
/// divided by this count, so a mismatch does not fail — it silently slices every
/// tile at the wrong offset. `the_atlas_has_a_column_for_every_terrain_kind`
/// checks the PNG's own width against this.
pub const TERRAIN_KIND_COUNT: u32 = 16;

impl TerrainKind {
    pub fn tileset_index(self) -> u16 {
        self as u16
    }

    /// Only these three kinds can be built on — a city is clipped by coast,
    /// mountain and river rather than paving them.
    ///
    /// `Sand`, `Marsh`, `Rock`, `Snow`, `Gravel` and `Reed` are deliberately *not*
    /// habitable, and that is most of what makes a desert or a marsh feel different
    /// to walk into: no city is founded on any of them, so those regions are empty
    /// of everything the plan would otherwise put there.
    ///
    /// `Scrub` **is**, and that is a decision rather than an oversight. It is the
    /// bare rung of the `Plains` and `Forest` ladders, so making it uninhabitable
    /// would have cut city sites out of ordinary grassland — a much bigger loss
    /// than the one it buys. The visible consequence is at the other end: a wadi
    /// promotes desert `Sand` to `Scrub` @ [`crate::gameplay::drainage`], so towns
    /// do appear strung along desert drainage lines, which is where real ones are.
    ///
    /// `Farmland` is deliberately *not* here either, and that one omission is the
    /// whole of the competition between cities: a claim requires the tile to be
    /// habitable, so a field one city has taken cannot be taken again by its
    /// neighbour, and no partition of the land between them is computed anywhere.
    pub fn is_habitable(self) -> bool {
        matches!(
            self,
            TerrainKind::Forest | TerrainKind::Grass | TerrainKind::Scrub
        )
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
    /// The fine driver of the cover ladder. ~50 tiles, up from the 11 it was.
    ///
    /// This is the other half of gh-14, and the less obvious half. Adding coarse
    /// layers does not by itself make a region patchy: the ladder is a sum, and
    /// while its *fine* term dominates, splitting a binary into three rungs only
    /// buys more transitions at the fine term's own wavelength — the world gets
    /// more mixed and *less* coherent, which reads as speckle rather than as
    /// landscape. Both are monotony; they just fail in opposite directions.
    ///
    /// Measured against `the_default_config_measures_the_structure_gap`'s coherence
    /// figure for Plains, at a fixed ~4.2 effective kinds per region:
    ///
    /// ```text
    ///   scale  wavelength   coherence@8  coherence@32   mean run
    ///   0.090      11           18%          13%         4.1
    ///   0.040      25           33%          15%         6.0
    ///   0.020      50           47%          17%         7.0
    ///   0.015      67           53%          21%         7.3
    /// ```
    ///
    /// The pre-gh-14 world sat at 36% / 32% with a 6.5-tile run and only **2.8**
    /// effective kinds — coherent, but coherent about one kind, which is the
    /// complaint. 0.02 keeps that patch size while nearly halving the long-range
    /// uniformity and putting four kinds in a region instead of two.
    ///
    /// Fine per-tile texture is deliberately *not* this layer's job — that is
    /// gh-13's dither, and doing it here costs a kind boundary rather than a shade.
    pub vegetation_scale: f32,
    pub settlement_scale: f32,
    pub humidity_scale: f32,
    /// The climate normal at sea level, in degrees Celsius, before the lapse rate
    /// takes any of it back. Read only by [`TerrainSampler::temperature`], the way
    /// the settlement figures are read only by `city.rs`.
    ///
    /// **Degrees rather than the crate's usual dimensionless 0..1**, because this is
    /// the one field with a threshold that has to mean something: water freezes at a
    /// particular number, and no amount of remapping makes 0.42 that number.
    pub sea_level_celsius: f32,
    /// How much colder the top of the height range is than the bottom. The whole
    /// range, not a real km-per-degree figure — the heightmap has no vertical scale
    /// and `PlanetConfig::relief_tiles`, which does, is a lighting number nothing
    /// else may read.
    ///
    /// Together with `sea_level_celsius` this places the **transient snow band**: the
    /// ground that freezes overnight and thaws by afternoon is where the normal is
    /// within one diurnal amplitude of freezing, so the band is
    /// `2 * amplitude / lapse_celsius` of the height range. At 26/34 with a ~7-degree
    /// swing that is elevation 0.55 to 0.97 — nearly all the land above the middle of
    /// the lowland band, and the reason the loop is visible in one 300-second day
    /// rather than only in a configured winter.
    pub lapse_celsius: f32,
    /// The wavelength of the regional temperature anomaly, coarser than the humidity
    /// field's ~50 tiles: which country is having a cold spell is a bigger thing than
    /// which valley is wet.
    pub temperature_scale: f32,
    /// How wide that anomaly swings, peak to peak. Small against the lapse rate, so
    /// it moves the snow line about rather than deciding where it is.
    pub temperature_noise_celsius: f32,
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
    /// The lithology layer: hard and soft bedrock in bands, on one strike for the
    /// whole world the way a real fold belt is.
    ///
    /// This is the layer that actually fills gh-14's spectral gap, and its value is
    /// that it is **uncorrelated with the biome map**. Everything else about a tile
    /// — its recipe, its cover triple, its beach — is a function of which Voronoi
    /// region it fell in, so the whole world agrees with one partition and changes
    /// only where that partition does. Hardness is a second partition cutting
    /// across the first at an angle, which is what stops a region's interior from
    /// being self-similar all the way out.
    ///
    /// The scale is across-strike, so 0.006 is a ~167-tile wavelength and the three
    /// octaves take the finest detail to ~42 tiles: bands 40..160 tiles wide, which
    /// is the middle of the band the measurement says is empty.
    pub lithology_scale: f32,
    /// How far the domain is stretched along strike. A band is only a band if it is
    /// much longer than it is wide; at 1.0 this layer is blobs and buys nothing the
    /// relief layer does not already have.
    pub lithology_aspect: f32,
    /// The strike, in degrees. One angle for the world — deliberately not a field,
    /// because a regional structure that changes direction every few hundred tiles
    /// is not a regional structure.
    pub lithology_strike_degrees: f32,
    /// How far a hard band stands proud and a soft one weathers down, in elevation
    /// units. Small on purpose: this is a hogback, not a range, and it is
    /// displacing the height that [`crate::gameplay::river`] descends and
    /// [`crate::gameplay::road`] prices.
    pub lithology_relief: f32,
    /// Hardness at or above which exposed bedrock reads as `Rock` rather than
    /// `Gravel` — the one place the lithology layer picks a tile directly.
    pub lithology_rock_min: f32,
    /// How much soil hard bedrock takes away, at full hardness. Resistant rock
    /// weathers to less regolith, which is what makes a hard band thin its cover
    /// while the soft band beside it stays wooded.
    ///
    /// This is the amplitude knob on the whole lithology layer, and the one that
    /// decides how much of gh-14's 32..256-tile gap actually gets filled: the slope
    /// term feeding soil is a *gradient*, and differentiating amplifies the fine
    /// octaves, so hardness is the only clean mid-band contributor soil has.
    pub lithology_soil_strip: f32,
    /// How far apart the two gradient samples are taken, in tiles. Sized to the
    /// relief layer's ~25-tile wavelength so it measures the slope a player walks
    /// up rather than the per-tile noise, which at one tile would be nearly all
    /// finest-octave and would read as static.
    pub soil_slope_tiles: f32,
    /// The slope at which the ground is stripped bare. The knob on how much bedrock
    /// the world shows; at the default a `Plains` slope reaches it rarely and a
    /// `Highland` shoulder reaches it often, which is the intended difference.
    pub soil_slope_falloff: f32,
    /// Below this soil depth the bedrock shows through the lowland band, and the
    /// tile stops taking its kind from the biome's cover triple at all.
    pub bedrock_max: f32,
    /// How far soil depth moves a tile along its biome's cover ladder.
    ///
    /// **This is the knob that closes gh-14's gap**, and the first cut of this work
    /// did not have it at all. Letting soil decide only whether bedrock breaks
    /// through leaves every tile that *has* soil — most of the world, and nearly
    /// all of `Plains` — picking its kind from the vegetation field alone at an
    /// 11-tile wavelength, exactly as before. The measured symptom was a Plains
    /// agreement curve still flat from lag 32 out to 256 while `Highland` and
    /// `Desert`, where bedrock does break through, had already picked up structure.
    ///
    /// Reading soil here is what carries the slope and the lithology bands into the
    /// cover of ground that is nowhere near bare: a hollow grows a wood, the
    /// shoulder above it grows grass, and the hard band crossing both thins them by
    /// a rung.
    pub soil_vegetation_gain: f32,
    /// The dune layer: transverse aeolian crests, weighted per recipe so only a
    /// `Desert` pays for the sample. The scale is across-wind, so 0.025 puts ~40
    /// tiles between crests.
    pub dune_scale: f32,
    /// How far the dune domain is stretched along the wind. Lower than the
    /// lithology aspect — a dune ridge is long, but not as long as a fold belt.
    pub dune_aspect: f32,
    /// Which way the wind blows, in degrees. Crests run perpendicular to it.
    pub dune_wind_degrees: f32,
    /// How high a crest stands. This is what gives the tint pass something to
    /// shade, so a dune field reads as relief and not as a paint job — the whole
    /// reason the dune layer touches the height at all.
    pub dune_relief: f32,
    /// Elevation bands, in ascending order; anything above `lowland_max` is mountain.
    pub deep_water_max: f32,
    pub shallow_water_max: f32,
    pub lowland_max: f32,
    /// Above the mountain band the ground goes bare and then white. Both are only
    /// reached where a `Highland` recipe's ridged layer piles up, so they mark the
    /// ranges rather than appearing wherever the land happens to be high.
    pub scree_min: f32,
    pub snow_min: f32,
    /// The lower of the two vegetation cuts: below it a lowland tile takes its
    /// cover triple's `bare` kind, above it the `mid` kind.
    ///
    /// The cut this one was added beside is why gh-14 happened. With one threshold
    /// a `vegetation_bias` does not move a region along a ladder, it pins the whole
    /// region to one side — `Desert`'s -0.30 put 92.5% of its tiles below the cut
    /// and made it 80% Sand, and `Forest`'s +0.13 put 72.6% above it.
    pub scrub_threshold: f32,
    /// The upper cut: vegetation at or above this takes the triple's `lush` kind.
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
            vegetation_scale: 0.02,
            settlement_scale: 0.12,
            // Coarser than the vegetation field: weather covers more ground than
            // a wood does, so a whole range is wet rather than one peak in it.
            humidity_scale: 0.02,
            sea_level_celsius: 26.0,
            lapse_celsius: 34.0,
            // ~125 tiles, against the humidity field's ~50.
            temperature_scale: 0.008,
            temperature_noise_celsius: 4.0,
            biome_cell_tiles: 384,
            biome_blend_tiles: 48,
            biome_warp_tiles: 160.0,
            lithology_scale: 0.006,
            lithology_aspect: 6.0,
            // Off every axis, so the banding never lines up with the tile grid or
            // with a chunk edge — either would read as an artefact rather than as
            // geology.
            lithology_strike_degrees: 24.0,
            lithology_relief: 0.05,
            lithology_rock_min: 0.60,
            lithology_soil_strip: 0.60,
            // Twelve rather than the four this started at, and the difference is
            // the whole measurement: a finite difference is a high-pass filter, so
            // a short baseline reports the relief field's *finest* octaves and puts
            // fresh 2-tile noise into soil. Twelve is about half the relief layer's
            // ~25-tile wavelength, which is where the difference is most sensitive
            // to the slope a player actually walks up.
            soil_slope_tiles: 12.0,
            soil_slope_falloff: 0.011,
            bedrock_max: 0.22,
            soil_vegetation_gain: 0.60,
            dune_scale: 0.025,
            dune_aspect: 3.0,
            // Deliberately not the lithology strike: two anisotropic layers on the
            // same angle would compound into one stripe pattern.
            dune_wind_degrees: 65.0,
            dune_relief: 0.03,
            deep_water_max: 0.32,
            shallow_water_max: 0.42,
            lowland_max: 0.72,
            scree_min: 0.78,
            snow_min: 0.88,
            scrub_threshold: 0.34,
            forest_threshold: 0.5,
            town_threshold: 0.62,
            town_coast_bonus: 0.06,
            coast_radius: 2,
            river_source_threshold: 0.55,
        }
    }
}

/// Salts that give each field its own patch of the noise lattice.
pub const ELEVATION_SALT: u32 = 0x0000_0001;
pub const VEGETATION_SALT: u32 = 0x9e37_79b9;
pub const SETTLEMENT_SALT: u32 = 0x85eb_ca6b;
pub const HUMIDITY_SALT: u32 = 0xc2b2_ae35;
pub const CONTINENT_SALT: u32 = 0x27d4_eb2d;
pub const RIDGE_SALT: u32 = 0x1656_67b1;
pub const LITHOLOGY_SALT: u32 = 0x3b9a_ca07;
pub const DUNE_SALT: u32 = 0x6f4e_2b13;
pub const TEMPERATURE_SALT: u32 = 0x4d2b_7f11;

/// The continent layer is there for its longest wavelength, so octaves finer than
/// the relief layer already provides are paid for on every tile and then buried
/// under it.
pub const CONTINENT_OCTAVES: u32 = 3;

/// Enough to shape a range without the crease pattern turning into noise.
pub const RIDGE_OCTAVES: u32 = 4;

/// Enough octaves for a fold belt to have detail without the bands dissolving.
/// Three, matching the continent layer's reasoning: an octave finer than the
/// feature the layer exists to make is paid for on every tile and then buried.
pub const LITHOLOGY_OCTAVES: u32 = 3;

/// A dune field wants long clean crests, so fewer octaves than a mountain range.
pub const DUNE_OCTAVES: u32 = 3;

/// How much loose material a dune crest piles up. A working dune *is* soil depth,
/// so a crest climbs a rung of its biome's ladder — for a desert, off the hardpan
/// and onto sand — while the deflated trough beside it stays on the bare rung.
pub const DUNE_SOIL_GAIN: f32 = 0.55;

/// How much a dune crest suppresses vegetation. Small, and deliberately smaller
/// than the soil gain above: it is only there to stop a live sand face reaching
/// the ladder's *lush* rung, not to drive the crest down to bedrock. Together
/// those two are the whole of how dunes reach the tileset — no rule anywhere
/// names `Desert`.
pub const DUNE_VEGETATION_BITE: f32 = 0.12;

impl TerrainConfig {
    /// The document this config describes, unbaked.
    ///
    /// The numbers only — a few dozen kilobytes of noise specs and a region table, which
    /// is the whole of what a world is before anything evaluates it.
    pub fn document(&self, size: UVec2) -> Terrain {
        document::build(self, size)
    }

    /// The sampler every reader of the landscape has to go through, over a world of
    /// `size` tiles.
    ///
    /// **This bakes**, where the sampler it replaced only built noise fields. That is the
    /// whole change in cost model: the old one answered any coordinate analytically and
    /// so was free to construct and dear to ask, this one is dear to construct and nearly
    /// free to ask. A caller that used to make one per loop must now be handed one.
    pub fn sampler_over(&self, size: UVec2) -> TerrainSampler {
        let mut terrain = self.document(size);
        terrain
            .bake()
            .expect("a document this module builds has to bake");
        TerrainSampler::new(Arc::new(terrain))
    }

    /// The sampler over the whole world.
    ///
    /// Costs a whole-world bake — seconds and hundreds of megabytes. At run time there is
    /// exactly one and [`crate::gameplay::world`] owns it; anything else wanting one is a
    /// test, and a test that does not need the whole world should say so with
    /// [`Self::sampler_over`].
    pub fn sampler(&self) -> TerrainSampler {
        self.sampler_over(WORLD_TILES)
    }
}

/// What one tile turned out to be, before the bands cut it into a kind.
///
/// The blended recipe rides along because the bands need its `beach_width` and its
/// kind pair — the classification is biome-dependent, not just elevation-dependent.
pub struct TileSample {
    pub elevation: f32,
    pub vegetation: f32,
    /// How much loose material sits on the bedrock here, in the unit range. Below
    /// `bedrock_max` the ladder stops asking the biome what to lay down and shows
    /// the rock instead.
    ///
    /// This is the number that converts the height into *cover*. Elevation has
    /// structure at every scale out to the width of a continent, and before gh-14
    /// the lowland band discarded all of it into one vegetation test — which is
    /// why the mountains, where the height does pick the kind, were the only part
    /// of the world that read as landscape.
    pub soil: f32,
    /// How resistant the bedrock is, in the unit range. Uncorrelated with the
    /// biome map by construction, which is the point of it.
    pub hardness: f32,
    /// Which region the tile is in. Not what the bands read — that is `cover` — so
    /// nothing in generation has a use for it.
    ///
    /// [`crate::gameplay::deposit`] is the one thing outside the measurements that
    /// does, and it wants exactly this rather than `cover`: a recipe naming
    /// `Highland` is naming the region's geology, and inside a blend band the tile's
    /// *vegetation* may well have been drawn from the neighbour.
    pub dominant: Biome,
    /// Which biome supplies the tile's ground cover, drawn from the blend weights.
    /// Equal to `dominant` everywhere except inside a blend band.
    pub cover: Biome,
    pub recipe: HeightRecipe,
}

/// The one answer to "how high, how green, how wet is it here".
///
/// **A reader of a baked document, and nothing else.** Every number it hands back was
/// evaluated by `watershed` out of the specs [`crate::gameplay::document`] wrote, so this
/// module no longer knows how a landscape is made — only how to ask. What is left here is
/// the part that is wusel's and cannot move to a library: [`classify`], which cuts a
/// sample into a `TerrainKind`, and the [`Biome`] table those kinds hang off.
///
/// The field handles are resolved once, at construction. A document holds its fields in a
/// `Vec` and finds one by string compare, which is nothing per bake and a great deal per
/// tile — so the ids are looked up here and the hot path indexes.
///
/// **A tile is read at the centre of its cell.** A document cell stands for the tile it
/// covers, and the bake evaluates it at `i + 0.5`; asking at the integer corner would
/// land halfway between two texels and interpolate. That is the half-tile the whole
/// translation turns on — see `the_document_lays_down_the_world_the_sampler_used_to`.
#[derive(Clone)]
pub struct TerrainSampler {
    terrain: Arc<Terrain>,
    height: usize,
    soil: usize,
    vegetation: usize,
    humidity: usize,
    temperature: usize,
    hardness: usize,
    region_id: usize,
    cover_class: usize,
    settlement: usize,
    /// The blended recipe columns, in [`document::COLUMNS`] order.
    columns: [usize; document::COLUMN_COUNT],
}

impl TerrainSampler {
    /// Resolve a baked document into the handles the hot path indexes by.
    ///
    /// Every id here is one [`crate::gameplay::document`] declares, so a missing one is
    /// this crate disagreeing with itself rather than a document being wrong — hence the
    /// panic rather than a `Result` no caller could act on.
    pub fn new(terrain: Arc<Terrain>) -> Self {
        let index = |id: &str| {
            terrain
                .fields
                .iter()
                .position(|field| field.id.as_str() == id)
                .unwrap_or_else(|| panic!("the document is missing the `{id}` field"))
        };
        let mut columns = [0usize; document::COLUMN_COUNT];
        for (slot, column) in document::COLUMNS.iter().enumerate() {
            columns[slot] = index(&document::column_field(column));
        }
        Self {
            height: index(document::HEIGHT),
            soil: index(document::SOIL),
            vegetation: index(document::VEGETATION),
            humidity: index(document::HUMIDITY),
            temperature: index(document::TEMPERATURE),
            hardness: index(document::HARDNESS),
            region_id: index(document::REGION_ID),
            cover_class: index(document::COVER_CLASS),
            settlement: index(document::SETTLEMENT),
            columns,
            terrain,
        }
    }

    /// One field at a global tile position, read at the centre of the tile's cell.
    fn at(&self, field: usize, x: f32, y: f32) -> f32 {
        self.terrain.fields[field].sample(x + 0.5, y + 0.5)
    }

    /// The biome a categorical field names at a tile.
    ///
    /// The field is categorical, so the library already read it at its nearest texel and
    /// no rounding here can land between two regions.
    fn biome(&self, field: usize, x: f32, y: f32) -> Biome {
        let index = self.at(field, x, y).round().max(0.0) as usize;
        BIOME_TABLE
            .get(index)
            .map_or(BIOME_TABLE[0].0, |(biome, _)| *biome)
    }

    /// How high the terrain is at a tile. Read by [`crate::gameplay::river`], whose
    /// particles walk downhill, and by [`crate::gameplay::road`], which costs a step
    /// by how much it climbs — `WorldMap` records only which band a tile fell in,
    /// not how high it is.
    pub fn elevation(&self, x: f32, y: f32) -> f32 {
        self.at(self.height, x, y)
    }

    /// How much rain falls at a tile, biased by the region. Read by
    /// [`crate::gameplay::river`], to decide which mountains are wet enough for a
    /// river to rise in them, and by [`crate::gameplay::weather`], which bakes it
    /// into the map that says where the clouds are — so it rains over the country
    /// the rivers rise in, and a desert gets neither.
    pub fn humidity(&self, x: f32, y: f32) -> f32 {
        self.at(self.humidity, x, y)
    }

    /// The **climate normal** at a tile, in degrees Celsius: how warm it is here on
    /// an average day, before the sun has moved and before any weather.
    ///
    /// Degrees rather than the crate's usual dimensionless 0..1, because a freezing
    /// point has to mean something. Read by [`crate::gameplay::ground`], which adds
    /// the day's swing to it and decides whether what falls is rain or snow.
    ///
    /// **It is deliberately not part of [`TileSample`], and `classify` may never read
    /// it**, on exactly the terms `PlanetConfig::relief_tiles` may only be read by the
    /// lighting: a tile's *kind* must not start depending on the weather. That it is now
    /// a field of the document rather than a method here does not loosen the rule — the
    /// document offers it and `classify` still may not ask.
    pub fn temperature(&self, x: f32, y: f32) -> f32 {
        self.at(self.temperature, x, y)
    }

    /// The blended recipe at a tile.
    ///
    /// Nine reads where the sampler this replaced did one Voronoi blend — cheaper, since
    /// each is a raster lookup and the blend was the expensive half of a tile.
    fn recipe(&self, x: f32, y: f32) -> HeightRecipe {
        let column = |slot: usize| self.at(self.columns[slot], x, y);
        HeightRecipe {
            base_height: column(document::COL_BASE_HEIGHT_INDEX),
            relief: column(document::COL_RELIEF_INDEX),
            ridge: column(document::COL_RIDGE_INDEX),
            dune: column(document::COL_DUNE_INDEX),
            soil_bias: column(document::COL_SOIL_BIAS_INDEX),
            vegetation_bias: column(document::COL_VEGETATION_BIAS_INDEX),
            humidity_bias: column(document::COL_HUMIDITY_BIAS_INDEX),
            temperature_bias: column(document::COL_TEMPERATURE_BIAS_INDEX),
            beach_width: column(document::COL_BEACH_WIDTH_INDEX),
        }
    }

    /// What makes one habitable tile a likelier city site than another.
    ///
    /// Nothing biome-dependent enters into it — [`crate::gameplay::city`] only compares
    /// scores between candidate sites — which is why it was the last field in the crate
    /// handed out as raw noise rather than through the sampler. Now that the document
    /// carries it there is no reason for a second evaluation to exist.
    pub fn settlement(&self, x: f32, y: f32) -> f32 {
        self.at(self.settlement, x, y)
    }

    /// Everything about a tile, for the one caller that needs all of it.
    pub fn sample(&self, x: f32, y: f32) -> TileSample {
        TileSample {
            elevation: self.at(self.height, x, y),
            vegetation: self.at(self.vegetation, x, y),
            soil: self.at(self.soil, x, y),
            hardness: self.at(self.hardness, x, y),
            dominant: self.biome(self.region_id, x, y),
            cover: self.biome(self.cover_class, x, y),
            recipe: self.recipe(x, y),
        }
    }
}

/// The one baked world the tests share.
///
/// **A test cannot afford its own.** The sampler used to be analytic, so `config.sampler()`
/// was free and every test made one; it now bakes, and a whole world is seconds and
/// hundreds of megabytes. Baking one per test would be minutes and — since tests run in
/// parallel in one process — many times the memory of the world it is checking.
///
/// So the default config gets exactly one, built on first use and shared. A test that
/// changes the config cannot use it and must say what it needs with
/// [`TerrainConfig::sampler_over`]: a window big enough for the coordinates it touches,
/// which for most is a chunk or two.
#[cfg(test)]
pub(crate) fn shared_test_sampler() -> &'static TerrainSampler {
    use std::sync::OnceLock;
    static SHARED: OnceLock<TerrainSampler> = OnceLock::new();
    SHARED.get_or_init(|| TerrainConfig::default().sampler())
}

/// Cuts a sampled tile into a kind.
///
/// Elevation alone still decides water from land from mountain — the bands are the
/// same four they were. What the biome changes is *what fills a band*: the lowland
/// triple comes from the recipe rather than being Grass and Forest everywhere, and
/// the sand band above the water line is as wide as the recipe says, which is zero
/// for a `Highland` coast and so gives a cliff instead of a beach.
///
/// The lowland band asks two questions rather than one, and that is gh-14's fix.
/// **How deep is the soil** comes first, because bare rock is not a kind of
/// vegetation — below `bedrock_max` the biome does not get a say at all, which is
/// what puts outcrops on the steep ground and in the hard bands regardless of what
/// is growing nearby. Only where there is soil does the triple's ladder apply.
fn classify(config: &TerrainConfig, sample: &TileSample) -> TerrainKind {
    let elevation = sample.elevation;

    if elevation < config.deep_water_max {
        TerrainKind::DeepWater
    } else if elevation < config.shallow_water_max {
        TerrainKind::ShallowWater
    } else if elevation < config.shallow_water_max + sample.recipe.beach_width {
        TerrainKind::Sand
    } else if elevation < config.lowland_max {
        if sample.soil < config.bedrock_max {
            // The rock the slope stripped down to. Which rock it is comes from the
            // lithology layer, so a hard band outcrops as scree and a soft one
            // crumbles to hardpan — the one place hardness picks a tile directly.
            if sample.hardness >= config.lithology_rock_min {
                TerrainKind::Rock
            } else {
                TerrainKind::Gravel
            }
        } else {
            // Where on the ladder this tile sits: how green it is, moved by how
            // much soil there is to be green in. The second term is what gives the
            // lowland band any structure above the vegetation field's ~11 tiles.
            let ladder = sample.vegetation + (sample.soil - 0.5) * config.soil_vegetation_gain;
            let triple = sample.cover.kinds();
            if ladder >= config.forest_threshold {
                triple.lush
            } else if ladder >= config.scrub_threshold {
                triple.mid
            } else {
                triple.bare
            }
        }
    } else if elevation < config.scree_min {
        TerrainKind::Mountain
    } else if elevation < config.snow_min {
        TerrainKind::Rock
    } else {
        TerrainKind::Snow
    }
}

/// One chunk's tiles: what each one is, and how high it was.
///
/// The two arrays are the same length and the same order, so index `i` is one
/// tile's kind and that same tile's height. Keeping the height here rather than
/// letting [`generate_chunk`] drop it is what makes the shading and the tile under
/// it come from a single evaluation of the terrain — see [`height_byte`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ChunkTerrain {
    pub kinds: Box<[TerrainKind]>,
    pub heights: Box<[u8]>,
}

/// An elevation as the byte the heightmap stores.
///
/// `TerrainSampler` already clamps its output to `0.0..=1.0`, so this is a
/// quantization and not a guard. One byte is 1/255 of the height range, which is
/// well under one 8-bit colour level once a tint ramp has scaled it down — so the
/// quantization itself cannot band.
pub fn height_byte(elevation: f32) -> u8 {
    (elevation.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Generates the kind and the height of every tile in the chunk whose lower-left
/// tile sits at the global tile coordinate `origin`, in row-major order from that
/// corner.
///
/// Every sample is taken in global tile space, never chunk-locally — that, and
/// the fact that no rule here reads a neighbouring tile, is what lets the world
/// be cut into chunks at all.
///
/// The height is the one `classify` cut the tile with, not a second sampling: the
/// `TileSample` is bound rather than passed straight through, so a tile's kind and
/// its shading cannot disagree about how high it is, and no second implementation
/// of "how high is it here" enters the crate.
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
    sampler: &TerrainSampler,
    origin: IVec2,
    chunk_size: UVec2,
) -> ChunkTerrain {
    let (kinds, heights): (Vec<TerrainKind>, Vec<u8>) = (0..chunk_size.element_product())
        .map(|i| {
            let x = (origin.x + (i % chunk_size.x) as i32) as f32;
            let y = (origin.y + (i / chunk_size.x) as i32) as f32;
            let sample = sampler.sample(x, y);
            (classify(config, &sample), height_byte(sample.elevation))
        })
        .unzip();

    ChunkTerrain {
        kinds: kinds.into(),
        heights: heights.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gameplay::terrain::shared_test_sampler;

    use crate::gameplay::world::WORLD_TILES;

    const CHUNK: UVec2 = UVec2::splat(64);
    /// Big enough to hold `ORIGIN + CHUNK` and the slope reach beyond it, and no bigger.
    /// A document is anchored on the world origin, so a window is always a corner of the
    /// real world rather than a world of its own — the tiles it holds are the tiles the
    /// game would generate there.
    const WINDOW: UVec2 = UVec2::splat(2176);
    /// Somewhere in the middle of the world rather than at the origin, so the
    /// tests exercise the same coordinate magnitudes the game actually uses.
    const ORIGIN: IVec2 = IVec2::new(2048, 2048);

    /// A chunk's kinds, baking a window around `ORIGIN` rather than the whole world.
    ///
    /// The window is what keeps this cheap: a document is anchored on the world origin,
    /// so covering `ORIGIN + CHUNK` means baking out to there — but at a fraction of the
    /// 4096 the game bakes, and these tests only ever read the one chunk.
    fn kinds(config: &TerrainConfig) -> Box<[TerrainKind]> {
        let sampler = config.sampler_over(WINDOW);
        generate_chunk(config, &sampler, ORIGIN, CHUNK).kinds
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
    ///
    /// Both halves of a tile, since the height is now kept as well: a heightmap
    /// that depended on the chunking would show the chunk grid as a shading grid.
    #[test]
    fn a_tile_does_not_depend_on_where_the_chunk_boundary_falls() {
        let config = TerrainConfig::default();
        let shift = IVec2::new(37, 11);

        let sampler = config.sampler_over(WINDOW);
        let base = generate_chunk(&config, &sampler, ORIGIN, CHUNK);
        let shifted = generate_chunk(&config, &sampler, ORIGIN + shift, CHUNK);

        for y in 0..CHUNK.y as i32 - shift.y {
            for x in 0..CHUNK.x as i32 - shift.x {
                let in_base = ((y + shift.y) * CHUNK.x as i32 + (x + shift.x)) as usize;
                let in_shifted = (y * CHUNK.x as i32 + x) as usize;
                assert_eq!(
                    (base.kinds[in_base], base.heights[in_base]),
                    (shifted.kinds[in_shifted], shifted.heights[in_shifted]),
                    "tile ({}, {}) disagrees between the two chunks covering it",
                    ORIGIN.x + x + shift.x,
                    ORIGIN.y + y + shift.y,
                );
            }
        }
    }

    /// The height kept beside a tile is the one it was cut with, so the shading and
    /// the tile under it cannot disagree and no second implementation of "how high
    /// is it here" enters the crate.
    #[test]
    fn every_tile_keeps_the_height_it_was_classified_from() {
        let config = TerrainConfig::default();
        let sampler = config.sampler();
        let chunk = generate_chunk(&config, &sampler, ORIGIN, CHUNK);

        for i in 0..chunk.kinds.len() {
            let x = (ORIGIN.x + (i as u32 % CHUNK.x) as i32) as f32;
            let y = (ORIGIN.y + (i as u32 / CHUNK.x) as i32) as f32;
            let sample = sampler.sample(x, y);

            assert_eq!(chunk.kinds[i], classify(&config, &sample));
            assert_eq!(chunk.heights[i], height_byte(sample.elevation));
        }
    }

    /// The heightmap carries the water line with it, so the tint pass can leave the
    /// sea alone without knowing what a `TerrainKind` is.
    ///
    /// The byte *on* the line is deliberately unconstrained, and has to be:
    /// `107/255` is 0.4196 and `108/255` is 0.4235, so no byte lands on 0.42 and the
    /// quantization cannot resolve which side of it a tile sat. That sliver is one
    /// tile wide and falls on the coastline, where the tileset changes anyway.
    #[test]
    fn water_is_exactly_what_falls_below_the_tint_water_line() {
        let config = TerrainConfig::default();
        let line = height_byte(config.shallow_water_max);

        let sampler = shared_test_sampler();
        for step in 0..16 {
            let origin = IVec2::new((step % 4) * 1024 + 128, (step / 4) * 1024 + 128);
            let chunk = generate_chunk(&config, sampler, origin, CHUNK);

            for (kind, &height) in chunk.kinds.iter().zip(chunk.heights.iter()) {
                if height < line {
                    assert!(
                        kind.is_water(),
                        "{kind:?} at height {height} is under the sea"
                    );
                } else if height > line {
                    assert!(
                        !kind.is_water(),
                        "{kind:?} at height {height} is above the sea"
                    );
                }
            }
        }
    }

    /// The lapse rate is the whole reason the field is worth sampling per tile: a
    /// mountain has to be colder than the plain beside it, or the snow line is not a
    /// line and the ground cover is a flat sheet.
    ///
    /// Checked as a *rank* rather than as a difference, over pairs sharing a place —
    /// so it is the height doing this and not the regional anomaly.
    #[test]
    fn the_temperature_field_falls_with_height() {
        let config = TerrainConfig::default();
        let sampler = shared_test_sampler();

        // The lapse rate acting alone, at one place: everything else about the tile
        // is held fixed, so the drop is the height and only the height. The two
        // constants come off the config rather than the sampler, which since the
        // document carries fields rather than knobs is the only place they live.
        let recipe = Biome::Plains.recipe();
        let at = |height: f32| {
            config.sea_level_celsius - config.lapse_celsius * height + recipe.temperature_bias
        };
        assert!(
            at(0.8) < at(0.5),
            "a world that warms with height has no snow line"
        );

        // And over the real world, where the anomaly and the biome bias are also in
        // play: the high ground still has to come out the colder end.
        let mut low = Vec::new();
        let mut high = Vec::new();
        for i in 0..8000u32 {
            let x = (i.wrapping_mul(2654435761) % 4000) as f32;
            let y = (i.wrapping_mul(40503) % 4000) as f32;
            let elevation = sampler.elevation(x, y);
            let temperature = sampler.temperature(x, y);
            if elevation < 0.5 {
                low.push(temperature);
            } else if elevation > 0.8 {
                high.push(temperature);
            }
        }

        assert!(
            low.len() > 100 && high.len() > 100,
            "not enough of either to compare: {} low, {} high",
            low.len(),
            high.len()
        );
        let mean = |values: &[f32]| values.iter().sum::<f32>() / values.len() as f32;
        let (low, high) = (mean(&low), mean(&high));
        assert!(
            high < low - 5.0,
            "the high ground averages {high:.1} C against the lowland's {low:.1} C, \
             which is not enough of a drop to put a snow line anywhere"
        );
    }

    /// The temperature must not be a second view of a landscape already drawn. Its
    /// own salt is what makes a cold spell cross a valley rather than follow it —
    /// the shape of `hardness_is_uncorrelated_with_the_biome_map`, applied to the
    /// field this one shares its position with.
    #[test]
    fn temperature_is_independent_of_the_other_fields() {
        let config = TerrainConfig::default();
        let sampler = shared_test_sampler();

        // The anomaly alone: the normal with the height and the biome taken out, so
        // what is left is the field's own contribution. Both subtractions now come off
        // the sampler's own published answers rather than out of its insides — the
        // height is `elevation` and the bias is the sample's blended recipe.
        let mut anomaly = Vec::new();
        let mut humidity = Vec::new();
        for i in 0..8000u32 {
            let x = (i.wrapping_mul(2654435761) % 4000) as f32;
            let y = (i.wrapping_mul(40503) % 4000) as f32;
            let sample = sampler.sample(x, y);
            anomaly.push(
                sampler.temperature(x, y) - config.sea_level_celsius
                    + config.lapse_celsius * sample.elevation
                    - sample.recipe.temperature_bias,
            );
            humidity.push(sampler.humidity(x, y));
        }

        let correlation = correlation(&anomaly, &humidity);
        assert!(
            correlation.abs() < 0.15,
            "the temperature anomaly correlates {correlation:.3} with humidity, so it is \
             the same landscape read twice rather than a field of its own"
        );
    }

    /// The `produces_every_base_kind` of this feature: a default that quietly gives a
    /// world which never freezes has no snow in it, and one that never thaws has snow
    /// that never goes away. Both are the same defect — a transient effect that is
    /// not transient — and neither is visible in any other test here.
    ///
    /// Measured against the *normal alone*, without the day's swing, so it is a claim
    /// about the climate rather than about how the ground module happens to be tuned.
    #[test]
    fn the_default_config_leaves_the_world_both_freezing_and_thawed() {
        let config = TerrainConfig::default();
        let sampler = config.sampler();

        let mut land = 0u32;
        let mut cold = 0u32;
        for i in 0..20000u32 {
            let x = (i.wrapping_mul(2654435761) % 4000) as f32;
            let y = (i.wrapping_mul(40503) % 4000) as f32;
            // The sea is excluded: nothing lies on it, and the pass that draws the
            // cover already knows that.
            if sampler.elevation(x, y) <= config.shallow_water_max {
                continue;
            }
            land += 1;
            // Within a plausible night's swing of freezing is what "it snows here
            // sometimes" means. Ten degrees is generous on purpose — this guards
            // against a world tens of degrees off, not against a retune.
            if sampler.temperature(x, y) < 10.0 {
                cold += 1;
            }
        }

        let share = cold as f32 / land as f32;
        assert!(
            (0.05..0.9).contains(&share),
            "{:.1}% of the land is within a night's swing of freezing: at one end it \
             never snows, at the other it never thaws",
            share * 100.0
        );
    }

    /// Pearson's, for the independence check above.
    fn correlation(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len() as f32;
        let mean_a = a.iter().sum::<f32>() / n;
        let mean_b = b.iter().sum::<f32>() / n;
        let mut covariance = 0.0;
        let mut variance_a = 0.0;
        let mut variance_b = 0.0;
        for (a, b) in a.iter().zip(b) {
            let (da, db) = (a - mean_a, b - mean_b);
            covariance += da * db;
            variance_a += da * da;
            variance_b += db * db;
        }
        covariance / (variance_a.sqrt() * variance_b.sqrt()).max(f32::EPSILON)
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
            TerrainKind::Scrub,
            TerrainKind::Gravel,
            TerrainKind::Reed,
        ] {
            assert!(
                counts[kind.tileset_index() as usize] > 0,
                "no {kind:?} anywhere in the world"
            );
        }
        // Town, Road, River and Farmland are deliberately absent from that list:
        // they are stamped over finished terrain, so this test must keep *not*
        // seeing them. `the_terrain_never_produces_a_kind_the_plan_stamps` is the
        // other half of the same rule.
    }

    /// A tile drawn for a biome no recipe can reach is a tile drawn for nothing.
    /// Four were added by the biome rework and three more by gh-14, and all seven
    /// have to earn their column.
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
            TerrainKind::Scrub,
            TerrainKind::Gravel,
            TerrainKind::Reed,
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

            let tiles = generate_chunk(&config, &sampler, origin, CHUNK).kinds;
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

    /// The stamped kinds belong to the plan and to the simulation, not to the
    /// terrain — if one ever came out of here, a chunk's contents would depend on
    /// its neighbours again.
    ///
    /// Driven off a slice rather than enumerated in the name, because a name that
    /// lists the kinds is wrong every time another one is added.
    #[test]
    fn the_terrain_never_produces_a_kind_the_plan_stamps() {
        let tiles = kinds(&TerrainConfig::default());
        for kind in STAMPED_KINDS {
            assert!(!tiles.contains(&kind), "the terrain generated {kind:?}");
        }
    }

    /// The kinds nothing in [`classify`] may ever return.
    const STAMPED_KINDS: [TerrainKind; 4] = [
        TerrainKind::Town,
        TerrainKind::Road,
        TerrainKind::River,
        TerrainKind::Farmland,
    ];

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

    /// What gh-14 is, measured: the agreement curve per biome, and what each
    /// region is made of now that the ladder has three rungs.
    ///
    /// `cargo test --release -- --ignored --nocapture the_default_config_measures_the_structure`
    #[test]
    #[ignore]
    fn the_default_config_measures_the_structure_gap() {
        let config = TerrainConfig::default();
        let agreement = cover_agreement(&config);

        let before = cover_agreement(&without_substrate(&config));

        println!("\nstructure in the cover, as excess agreement over chance (0 = none):");
        print!("  lag              ");
        for lag in LAGS {
            print!("{lag:>7}");
        }
        println!();
        for (slot, biome) in Biome::ALL.iter().enumerate() {
            if agreement[slot][0] == 0.0 && before[slot][0] == 0.0 {
                continue;
            }
            for (label, row) in [("before", &before), ("after", &agreement)] {
                print!("  {biome:>9?} {label:>6}  ");
                for value in row[slot].iter().take(LAGS.len()) {
                    print!("{value:>6.0}%");
                }
                println!();
            }
        }

        let sampler = config.sampler();
        let mut runs = [0u64; Biome::ALL.len()];
        let mut tiles = [0u64; Biome::ALL.len()];
        let mut kinds = [[0u64; TERRAIN_KIND_COUNT as usize]; Biome::ALL.len()];
        let rows = 48;
        for r in 0..rows {
            let y = (r * (WORLD_TILES.y / rows)) as f32;
            let mut previous: Option<(TerrainKind, usize)> = None;
            for x in 0..WORLD_TILES.x {
                let sample = sampler.sample(x as f32, y);
                let slot = Biome::ALL
                    .iter()
                    .position(|b| *b == sample.dominant)
                    .unwrap();
                let kind = classify(&config, &sample);
                kinds[slot][kind.tileset_index() as usize] += 1;
                tiles[slot] += 1;
                if previous != Some((kind, slot)) {
                    runs[slot] += 1;
                    previous = Some((kind, slot));
                }
            }
        }

        // Soil is the layer everything else acts through, so its distribution is
        // what the constants above are actually tuned against.
        let mut soil = Vec::new();
        let mut hardness = Vec::new();
        for i in 0..20000u32 {
            let x = (i.wrapping_mul(2654435761) % 4000) as f32;
            let y = (i.wrapping_mul(40503) % 4000) as f32;
            let s = sampler.sample(x, y);
            if s.elevation <= config.shallow_water_max || s.elevation >= config.lowland_max {
                continue;
            }
            soil.push(s.soil);
            hardness.push(s.hardness);
        }
        soil.sort_by(f32::total_cmp);
        hardness.sort_by(f32::total_cmp);
        let percentile = |v: &[f32], p: f64| v[((v.len() - 1) as f64 * p) as usize];
        println!(
            "\nlowland soil    p05 {:.2}  p25 {:.2}  p50 {:.2}  p75 {:.2}  p95 {:.2}  \
             — {:.1}% below bedrock_max {:.2}",
            percentile(&soil, 0.05),
            percentile(&soil, 0.25),
            percentile(&soil, 0.50),
            percentile(&soil, 0.75),
            percentile(&soil, 0.95),
            soil.iter().filter(|s| **s < config.bedrock_max).count() as f64 / soil.len() as f64
                * 100.0,
            config.bedrock_max,
        );
        println!(
            "lowland hardness p05 {:.2}  p50 {:.2}  p95 {:.2}  — {:.1}% at or over \
             lithology_rock_min {:.2}",
            percentile(&hardness, 0.05),
            percentile(&hardness, 0.50),
            percentile(&hardness, 0.95),
            hardness
                .iter()
                .filter(|h| **h >= config.lithology_rock_min)
                .count() as f64
                / hardness.len() as f64
                * 100.0,
            config.lithology_rock_min,
        );

        // Where each biome sits on its own ladder. The two cuts are at
        // scrub_threshold and forest_threshold, so a biome whose whole
        // distribution falls on one side of both is pinned to one rung — which is
        // the defect this change exists to remove, and the thing that is easiest
        // to reintroduce while tuning the biases.
        let mut ladders: Vec<Vec<f32>> = vec![Vec::new(); Biome::ALL.len()];
        for i in 0..30000u32 {
            let x = (i.wrapping_mul(2654435761) % 4000) as f32;
            let y = (i.wrapping_mul(40503) % 4000) as f32;
            let s = sampler.sample(x, y);
            if s.elevation <= config.shallow_water_max + s.recipe.beach_width
                || s.elevation >= config.lowland_max
                || s.soil < config.bedrock_max
            {
                continue;
            }
            let slot = Biome::ALL.iter().position(|b| *b == s.cover).unwrap();
            ladders[slot].push(s.vegetation + (s.soil - 0.5) * config.soil_vegetation_gain);
        }
        println!(
            "\nwhere each biome sits on its ladder (cuts at {:.2} and {:.2}):",
            config.scrub_threshold, config.forest_threshold
        );
        for (slot, biome) in Biome::ALL.iter().enumerate() {
            let v = &mut ladders[slot];
            if v.len() < 50 {
                continue;
            }
            v.sort_by(f32::total_cmp);
            let p = |q: f64| v[((v.len() - 1) as f64 * q) as usize];
            let triple = biome.kinds();
            println!(
                "  {biome:>9?}  p10 {:.2}  p50 {:.2}  p90 {:.2}  |  {:>3.0}% {:?} / {:>3.0}% {:?} / {:>3.0}% {:?}",
                p(0.10),
                p(0.50),
                p(0.90),
                v.iter().filter(|l| **l < config.scrub_threshold).count() as f64 / v.len() as f64
                    * 100.0,
                triple.bare,
                v.iter()
                    .filter(|l| **l >= config.scrub_threshold && **l < config.forest_threshold)
                    .count() as f64
                    / v.len() as f64
                    * 100.0,
                triple.mid,
                v.iter().filter(|l| **l >= config.forest_threshold).count() as f64 / v.len() as f64
                    * 100.0,
                triple.lush,
            );
        }

        println!("\nwhat a region is made of, and how far a kind runs:");
        for (slot, biome) in Biome::ALL.iter().enumerate() {
            let n = tiles[slot].max(1);
            let mut histogram: Vec<(f64, TerrainKind)> = KIND_BY_INDEX
                .iter()
                .enumerate()
                .filter(|(i, _)| kinds[slot][*i] > 0)
                .map(|(i, k)| (kinds[slot][i] as f64 / n as f64 * 100.0, *k))
                .collect();
            histogram.sort_by(|a, b| b.0.total_cmp(&a.0));
            let listed: Vec<String> = histogram
                .iter()
                .take(5)
                .map(|(share, kind)| format!("{kind:?} {share:.0}%"))
                .collect();
            println!(
                "  {biome:>9?}  run {:>5.1}  {}",
                tiles[slot] as f64 / runs[slot].max(1) as f64,
                listed.join(", "),
            );
        }
        println!();
    }

    /// gh-14's acceptance criterion, and the one test that would have caught the
    /// world it was opened about.
    ///
    /// The complaint was large sections of a single tile, so the measurement is how
    /// many kinds a region is *effectively* made of — the inverse Simpson index,
    /// which counts a 1% tail kind as the 1% it is rather than as a whole kind. At
    /// the time gh-14 was written Desert scored 1.5 (80% Sand) and Plains 2.8;
    /// nothing in the suite objected, because every other test asks whether a kind
    /// exists somewhere and none asks whether a region is made of only one.
    #[test]
    fn no_region_is_built_out_of_one_or_two_kinds() {
        let config = TerrainConfig::default();
        let sampler = config.sampler();

        let mut counts = [[0u64; TERRAIN_KIND_COUNT as usize]; Biome::ALL.len()];
        for i in 0..40000u32 {
            let x = (i.wrapping_mul(2654435761) % 4000) as f32;
            let y = (i.wrapping_mul(40503) % 4000) as f32;
            let sample = sampler.sample(x, y);
            let slot = Biome::ALL
                .iter()
                .position(|b| *b == sample.dominant)
                .unwrap();
            counts[slot][classify(&config, &sample).tileset_index() as usize] += 1;
        }

        for (slot, biome) in Biome::ALL.iter().enumerate() {
            // Ocean is the sea and is *meant* to be mostly one kind — the point of
            // it is that it is water. Its shores are Plains' and Desert's problem.
            if *biome == Biome::Ocean {
                continue;
            }
            let total: u64 = counts[slot].iter().sum();
            let simpson: f64 = counts[slot]
                .iter()
                .map(|n| {
                    let share = *n as f64 / total.max(1) as f64;
                    share * share
                })
                .sum();
            let effective = 1.0 / simpson;
            assert!(
                effective >= 3.0,
                "a {biome:?} region is effectively made of {effective:.2} kinds"
            );
        }
    }

    /// The lithology layer is only worth its cost if it is a *second* partition of
    /// the world. If hardness tracked the biome map it would restate a boundary
    /// that is already drawn, and every region would still be internally uniform —
    /// which is the whole failure this change exists to fix.
    #[test]
    fn hardness_is_uncorrelated_with_the_biome_map() {
        let config = TerrainConfig::default();
        let sampler = config.sampler();

        let mut sums = [0.0f64; Biome::ALL.len()];
        let mut counts = [0u64; Biome::ALL.len()];
        for i in 0..20000u32 {
            let x = (i.wrapping_mul(2654435761) % 4000) as f32;
            let y = (i.wrapping_mul(40503) % 4000) as f32;
            let sample = sampler.sample(x, y);
            let slot = Biome::ALL
                .iter()
                .position(|b| *b == sample.dominant)
                .unwrap();
            sums[slot] += sample.hardness as f64;
            counts[slot] += 1;
        }

        let world: f64 = sums.iter().sum::<f64>() / counts.iter().sum::<u64>() as f64;
        for (slot, biome) in Biome::ALL.iter().enumerate() {
            if counts[slot] < 500 {
                continue;
            }
            let mean = sums[slot] / counts[slot] as f64;
            assert!(
                (mean - world).abs() < 0.05,
                "{biome:?} has mean hardness {mean:.3} against the world's {world:.3}, \
                 so the strata are tracking the regions instead of crossing them"
            );
        }
    }

    /// Where `vegetation_scale` comes from: the ladder's fine driver against its
    /// coarse ones, which is the trade between speckle and a uniform sheet.
    ///
    /// Both ends are monotony. Short wavelengths give a region four kinds that
    /// change every four tiles, which reads as noise; long ones give big coherent
    /// patches. The effective kind count stays flat at ~4.2 across the whole range,
    /// which is what makes this a free choice of patch size rather than a trade
    /// against variety.
    ///
    /// `cargo test --release -- --ignored --nocapture the_ladders_fine_driver`
    #[test]
    #[ignore]
    fn the_ladders_fine_driver_trades_speckle_for_patch_size() {
        let base = TerrainConfig::default();
        println!("\n  veg_scale  wavelength   biome    k8   k32  k128   run   kinds");
        for scale in [0.09f32, 0.06, 0.04, 0.025, 0.015] {
            let config = TerrainConfig {
                vegetation_scale: scale,
                ..base.clone()
            };
            let k = cover_agreement(&config);
            let sampler = config.sampler();

            for target in [Biome::Plains, Biome::Forest, Biome::Desert] {
                let slot = Biome::ALL.iter().position(|b| *b == target).unwrap();
                let mut runs = 0u64;
                let mut tiles = 0u64;
                let mut counts = [0u64; TERRAIN_KIND_COUNT as usize];
                for r in 0..24 {
                    let y = (r * (WORLD_TILES.y / 24)) as f32;
                    let mut previous = None;
                    for x in 0..WORLD_TILES.x {
                        let s = sampler.sample(x as f32, y);
                        if s.dominant != target {
                            previous = None;
                            continue;
                        }
                        let kind = classify(&config, &s);
                        counts[kind.tileset_index() as usize] += 1;
                        tiles += 1;
                        if previous != Some(kind) {
                            runs += 1;
                            previous = Some(kind);
                        }
                    }
                }
                // How many kinds the region is effectively made of: the inverse
                // Simpson index, so "80% one kind" scores near 1 however many
                // kinds appear in the tail.
                let total: u64 = counts.iter().sum();
                let simpson: f64 = counts
                    .iter()
                    .map(|n| {
                        let s = *n as f64 / total.max(1) as f64;
                        s * s
                    })
                    .sum();
                println!(
                    "  {scale:>9.3}  {:>10.0}  {target:>8?}  {:>4.0} {:>5.0} {:>5.0}  {:>4.1}  {:>5.2}",
                    1.0 / scale,
                    k[slot][3],
                    k[slot][5],
                    k[slot][7],
                    tiles as f64 / runs.max(1) as f64,
                    1.0 / simpson,
                );
            }
        }
        println!();
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
            let chunk = generate_chunk(&config, &sampler, origin, CHUNK);
            for (kind, height) in chunk.kinds.iter().zip(chunk.heights.iter()) {
                checksum += kind.tileset_index() as u64 + *height as u64;
            }
        }
        println!(
            "\ngenerate_chunk: {:.2} ms per 64x64 chunk (checksum {checksum})\n",
            started.elapsed().as_secs_f64() * 1000.0 / chunks as f64
        );
    }

    /// The lags the structure measurements are taken at, in tiles.
    const LAGS: [u32; 10] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

    /// The world as it was before gh-14, for the comparisons that have to mean
    /// something: every layer this change added turned off, and the three-rung
    /// ladder collapsed back to the binary it replaced by putting both cuts in the
    /// same place.
    ///
    /// Not a config anyone would ship — it is the control, and the only honest way
    /// to say what the new layers actually bought.
    fn without_substrate(config: &TerrainConfig) -> TerrainConfig {
        TerrainConfig {
            soil_vegetation_gain: 0.0,
            lithology_soil_strip: 0.0,
            lithology_relief: 0.0,
            dune_relief: 0.0,
            // Nothing can fall through to the bedrock branch.
            bedrock_max: 0.0,
            // Both cuts together, so the `mid` rung is unreachable and the ladder
            // is the dry/wet pair again.
            scrub_threshold: config.forest_threshold,
            ..config.clone()
        }
    }

    /// How much structure the cover has at each lag, per biome, as a percentage.
    /// Index matches [`LAGS`].
    ///
    /// This is *the* gh-14 measurement, and it is **excess agreement over chance**,
    /// not raw agreement. Two tiles of a one-kind region agree 100% of the time
    /// while carrying no structure whatever, so raw agreement rewards exactly the
    /// monotony this is meant to detect — and since the fix changes how many kinds
    /// a region has, a raw before/after comparison would be measuring the
    /// composition change rather than the structure. Normalising against the
    /// region's own kind marginals, `(agree - chance) / (1 - chance)`, is what makes
    /// the two worlds comparable: 0 means "no more alike than two tiles picked at
    /// random from this region", 100 means "identical".
    ///
    /// A world whose cover has structure decays smoothly toward 0. The world before
    /// this change fell to its floor by lag 8-16 and then ran dead flat for six
    /// octaves — and a flat plateau is what the eye integrates into "uniform green
    /// texture". Restricted to lowland tiles of one region on both ends, or it would
    /// be measuring band edges and region boundaries rather than the interiors that
    /// are the problem.
    fn cover_agreement(config: &TerrainConfig) -> [[f64; 10]; Biome::ALL.len()] {
        let sampler = config.sampler();
        let mut same = [[0u64; 10]; Biome::ALL.len()];
        let mut pairs = [[0u64; 10]; Biome::ALL.len()];
        // The marginal kind distribution per biome, for the chance floor.
        let mut marginal = [[0u64; TERRAIN_KIND_COUNT as usize]; Biome::ALL.len()];

        let lowland = |k: TerrainKind| {
            !k.is_water()
                && !matches!(
                    k,
                    TerrainKind::Mountain | TerrainKind::Rock | TerrainKind::Snow
                )
        };

        for i in 0..24000u32 {
            let x = (i.wrapping_mul(2654435761) % 3000) as f32;
            let y = (i.wrapping_mul(40503) % 3000) as f32;
            let here = sampler.sample(x, y);
            let slot = Biome::ALL.iter().position(|b| *b == here.dominant).unwrap();
            let here_kind = classify(config, &here);
            if !lowland(here_kind) {
                continue;
            }
            marginal[slot][here_kind.tileset_index() as usize] += 1;

            for (l, lag) in LAGS.iter().enumerate() {
                let there = sampler.sample(x + *lag as f32, y);
                if there.dominant != here.dominant {
                    continue;
                }
                let there_kind = classify(config, &there);
                if !lowland(there_kind) {
                    continue;
                }
                pairs[slot][l] += 1;
                if here_kind == there_kind {
                    same[slot][l] += 1;
                }
            }
        }

        let mut out = [[0.0f64; 10]; Biome::ALL.len()];
        for slot in 0..Biome::ALL.len() {
            // Chance that two tiles drawn independently from this region's own kind
            // mix happen to match: the sum of the squared shares.
            let total: u64 = marginal[slot].iter().sum();
            if total == 0 {
                continue;
            }
            let chance: f64 = marginal[slot]
                .iter()
                .map(|n| {
                    let share = *n as f64 / total as f64;
                    share * share
                })
                .sum();

            for l in 0..LAGS.len() {
                let agree = same[slot][l] as f64 / pairs[slot][l].max(1) as f64;
                out[slot][l] = ((agree - chance) / (1.0 - chance).max(1e-9) * 100.0).max(0.0);
            }
        }
        out
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
        TerrainKind::Scrub,
        TerrainKind::Gravel,
        TerrainKind::Reed,
        TerrainKind::Farmland,
    ];
}
