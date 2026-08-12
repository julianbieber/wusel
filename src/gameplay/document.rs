// TODO(jb-doc): module docs — that this is the one place wusel's world is written down
// as numbers a `watershed::Terrain` can carry, that nothing here knows what a tile looks
// like, and which direction the dependency runs.

use bevy::prelude::*;
use watershed::layer::{Blend, Layer, LayerOp, Mask, Remap, SlopeMode};
use watershed::noise::{NoiseKind, NoiseSpec, SampleTransform, WarpSpec};
use watershed::regions::{Region, RegionOutput, RegionSpec};
use watershed::{Field, Terrain, WaterSpec};

use crate::gameplay::biome::{
    BIOME_TABLE, HeightRecipe, WARP_CELLS, WARP_OCTAVES, WARP_X_SALT, WARP_Y_SALT,
};
use crate::gameplay::terrain::{
    CONTINENT_OCTAVES, CONTINENT_SALT, DUNE_OCTAVES, DUNE_SALT, DUNE_SOIL_GAIN,
    DUNE_VEGETATION_BITE, ELEVATION_SALT, HUMIDITY_SALT, LITHOLOGY_OCTAVES, LITHOLOGY_SALT,
    RIDGE_OCTAVES, RIDGE_SALT, SETTLEMENT_SALT, TEMPERATURE_SALT, TerrainConfig, VEGETATION_SALT,
};

/// The field names the rest of the crate reads a world by.
///
/// TODO(jb-doc): why these are constants rather than string literals at the call sites,
/// and what a rename costs a document already saved to disk.
pub const HEIGHT: &str = "height";
pub const SOIL: &str = "soil";
pub const VEGETATION: &str = "vegetation";
pub const HUMIDITY: &str = "humidity";
pub const TEMPERATURE: &str = "temperature";
pub const HARDNESS: &str = "hardness";
pub const REGION_ID: &str = "region_id";
pub const COVER_CLASS: &str = "cover_class";
pub const SETTLEMENT: &str = "settlement";

/// The intermediates. Named because a layer stack can only refer to a field by name,
/// not because anything outside this module reads them.
const CONTINENT: &str = "continent";
const RELIEF_RAW: &str = "relief_raw";
const RIDGE_RAW: &str = "ridge_raw";
const DUNE_RAW: &str = "dune_raw";
const DUNE: &str = "dune";
const RELIEF_HEIGHT: &str = "relief_height";
const SOIL_BASE: &str = "soil_base";
const VEGETATION_RAW: &str = "vegetation_raw";
const HUMIDITY_RAW: &str = "humidity_raw";
const TEMPERATURE_RAW: &str = "temperature_raw";

/// The recipe columns, in the order [`recipe_row`] writes them.
///
/// TODO(jb-comment): why the column list is written out here rather than derived from
/// `HeightRecipe`, and what the compiler does and does not catch if the two drift.
const COL_BASE_HEIGHT: &str = "base_height";
const COL_RELIEF: &str = "relief";
const COL_RIDGE: &str = "ridge";
const COL_DUNE: &str = "dune";
const COL_SOIL_BIAS: &str = "soil_bias";
const COL_VEGETATION_BIAS: &str = "vegetation_bias";
const COL_HUMIDITY_BIAS: &str = "humidity_bias";
const COL_TEMPERATURE_BIAS: &str = "temperature_bias";
const COL_BEACH_WIDTH: &str = "beach_width";

/// How many numbers a region's row carries. Named so the sampler can size an array of
/// handles by it rather than by a literal that could fall out of step.
pub const COLUMN_COUNT: usize = 9;

/// Where each column sits in [`COLUMNS`], so a reader can index rather than search.
///
/// TODO(jb-comment): why these are indices beside the names rather than an enum, and what
/// the compiler does and does not check about the pairing.
pub const COL_BASE_HEIGHT_INDEX: usize = 0;
pub const COL_RELIEF_INDEX: usize = 1;
pub const COL_RIDGE_INDEX: usize = 2;
pub const COL_DUNE_INDEX: usize = 3;
pub const COL_SOIL_BIAS_INDEX: usize = 4;
pub const COL_VEGETATION_BIAS_INDEX: usize = 5;
pub const COL_HUMIDITY_BIAS_INDEX: usize = 6;
pub const COL_TEMPERATURE_BIAS_INDEX: usize = 7;
pub const COL_BEACH_WIDTH_INDEX: usize = 8;

pub const COLUMNS: [&str; COLUMN_COUNT] = [
    COL_BASE_HEIGHT,
    COL_RELIEF,
    COL_RIDGE,
    COL_DUNE,
    COL_SOIL_BIAS,
    COL_VEGETATION_BIAS,
    COL_HUMIDITY_BIAS,
    COL_TEMPERATURE_BIAS,
    COL_BEACH_WIDTH,
];

/// The field a blended recipe column is published as, so a layer can mask itself by it.
pub fn column_field(column: &str) -> String {
    format!("recipe_{column}")
}

/// A recipe as the row of numbers a region carries.
///
/// The order is [`COLUMNS`], and nothing checks that but this function.
fn recipe_row(recipe: HeightRecipe) -> [f32; 9] {
    [
        recipe.base_height,
        recipe.relief,
        recipe.ridge,
        recipe.dune,
        recipe.soil_bias,
        recipe.vegetation_bias,
        recipe.humidity_bias,
        recipe.temperature_bias,
        recipe.beach_width,
    ]
}

/// The bands a blended column can reach, so a field carrying one does not clamp it.
///
/// A column is a weighted mean of the table's rows, so it never leaves the interval
/// between that column's smallest and largest entry — but a `Field` clamps to its
/// declared range, and the default range is the unit one, which would silently floor
/// every negative bias in the table at zero.
fn column_range(column: &str) -> (f32, f32) {
    let values = BIOME_TABLE
        .iter()
        .map(|(biome, _)| recipe_row(biome.recipe()));
    let index = COLUMNS
        .iter()
        .position(|name| *name == column)
        .expect("a column the document builds has to be one it declared");
    let mut low = f32::MAX;
    let mut high = f32::MIN;
    for row in values {
        low = low.min(row[index]);
        high = high.max(row[index]);
    }
    (low, high)
}

/// The Voronoi lattice, the blend band, the warp and the table — the whole of what
/// `biome.rs` used to build a `BiomeMap` out of.
fn region_spec(config: &TerrainConfig) -> RegionSpec {
    let mut spec = RegionSpec::new(
        config.seed,
        config.biome_cell_tiles,
        config.biome_blend_tiles,
        COLUMNS,
    )
    .with_warp(WarpSpec {
        seed: config.seed,
        amplitude: config.biome_warp_tiles,
        scale: 1.0 / (config.biome_cell_tiles.max(1) as f32 * WARP_CELLS),
        octaves: WARP_OCTAVES,
        // The two component fields are salted off the one world seed, which is what
        // `BiomeMap` did and what the library's own `sub_seed` cannot reproduce.
        salts: Some((WARP_X_SALT, WARP_Y_SALT)),
    });
    for (biome, weight) in BIOME_TABLE {
        spec = spec.with_region(Region::new(weight, recipe_row(biome.recipe())));
    }
    spec
}

fn noise(salt: u32, kind: NoiseKind, seed: u32, scale: f32, octaves: u32) -> NoiseSpec {
    NoiseSpec::new(seed, kind, scale)
        .with_salt(salt)
        .with_octaves(octaves)
}

/// How many tiles one texel of a field stands for, as a power of two.
///
/// **This is the knob the whole document costs memory on**, and the numbers are not
/// taste: at 4096 a shift-0 field is 64 MB and every step up quarters it. All 28 fields
/// at shift 0 is 1792 MB, which is not a thing wusel can hold.
///
/// **The rule is four samples across the field's finest detail.** An fbm of `n` octaves
/// at `scale` has its shortest wavelength at `1 / (scale * 2^(n-1))`, and a texel bigger
/// than a quarter of that aliases — bilinear reconstruction needs several samples per
/// wave, not the two Nyquist allows. Getting this wrong does not look like noise, it
/// looks like *geometry*: the first cut had `ridge` and `dune` at 2.5 samples per
/// wavelength and the cover came out with visibly axis-aligned block edges, which is the
/// interpolation grid showing through a threshold. It was caught by looking at the game,
/// not by a test.
///
/// A field whose finest detail is near a single tile therefore cannot be coarsened at
/// all — and that is not a compromise, since shift 0 is exactly what the analytic sampler
/// did when it evaluated once per tile.
///
/// Three fields are pinned by a *rule* rather than by their wavelength:
///
/// - **`height`** — a tile boundary has to be exact, and the library enforces the same
///   for whatever field a `WaterSpec` names.
/// - **`region_id` and `cover_class`** — categorical. The cover dither is per tile by
///   design, and a texel spanning sixteen of them would speckle in blocks.
fn shift_of(id: &str) -> u8 {
    match id {
        // Pinned by the rules above, or read per tile by `classify`, whose whole gh-14
        // argument is that the cover ladder carries fine structure.
        HEIGHT | REGION_ID | COVER_CLASS | SOIL | VEGETATION | HARDNESS => 0,
        // 5 octaves at 0.04 puts the relief's finest detail at 1.6 tiles, and the ladder
        // is built out of it. Nothing coarser can carry that.
        RELIEF_RAW | RELIEF_HEIGHT | SOIL_BASE | VEGETATION_RAW => 0,
        // ~8 tiles between the finest settlement bumps, and `city.rs` compares scores
        // between sites a few tiles apart.
        SETTLEMENT => 0,
        // 3.1 tiles, so it cannot be coarsened either — and it is read per tile in two
        // places that matter: `river.rs` tests a spring candidate against
        // the channel cut, and `growth.rs` a farm, the latter precisely because
        // the field's wavelength is shorter than a city's reach. An earlier cut at shift
        // 4 came out 0.045 off the analytic sampler on average, against a threshold of
        // 0.55.
        HUMIDITY | HUMIDITY_RAW => 0,
        // Temperature lapses `height`, so it inherits the height's own resolution
        // whatever its noise does; at shift 4 it was 0.44 C out on average, which walks
        // the snow line.
        TEMPERATURE => 0,
        // Its own anomaly is ~7.8 tiles at its finest and deliberately coarser than the
        // humidity field.
        TEMPERATURE_RAW => 1,
        // Ridged at 0.012 over 4 octaves is 10.4 tiles, the dune layers 10.0 across-wind.
        // A quarter of either is ~2.5, so 2 tiles a texel — this is the pair that aliased
        // at 4.
        RIDGE_RAW | DUNE_RAW | DUNE => 1,
        // ~670 tiles across with its finest octave at 167, which is the coarsest thing in
        // the world and the one field that can afford a 32-tile texel.
        CONTINENT => 5,
        _ => COLUMN_SHIFT,
    }
}

/// The shift every blended recipe column takes.
///
/// A column is the one thing here whose resolution is not a question about a wavelength.
/// A recipe is **constant in a region's interior and varies only across the blend band**,
/// which is `biome_blend_tiles * 2` = 96 tiles wide — so a texel every 4 tiles puts 24 of
/// them across the band and the bilinear read between them is very nearly the blend
/// itself. It is also nine fields, so it is where the memory would otherwise be: 9 x 64 MB
/// at shift 0 against 9 x 4 MB here.
///
/// What it may **not** do is smooth the *cover* — that is `cover_class`, categorical and
/// pinned at zero, and it is a separate field precisely so this one can be coarse.
const COLUMN_SHIFT: u8 = 2;

fn noise_field(id: &str, spec: NoiseSpec, range: (f32, f32)) -> Field {
    Field::new(id)
        .with_shift(shift_of(id))
        .with_range(range)
        .with_layer(Layer::new(LayerOp::Noise(spec)).with_blend(Blend::Replace))
}

/// A layer adding `of`, weighted per tile by a blended recipe column.
///
/// A mask is a lerp weight rather than a multiplier, so this is only a multiply
/// because the blend is `Add`: the texel goes from `under` to `under + value` by
/// `weight`, which lands on `under + value * weight`.
fn weighted_by(of: &str, column: &str) -> Layer {
    Layer::new(LayerOp::FieldRef(of.into()))
        .with_mask(Mask::Field(column_field(column).into(), Remap::IDENTITY))
}

/// A constant, weighted by a blended recipe column the same way.
fn constant_weighted_by(value: f32, column: &str) -> Layer {
    Layer::new(LayerOp::Constant(value))
        .with_mask(Mask::Field(column_field(column).into(), Remap::IDENTITY))
}

fn field_ref(of: &str, amplitude: f32) -> Layer {
    Layer::new(LayerOp::FieldRef(of.into())).with_amplitude(amplitude)
}

/// Everything `TerrainSampler` used to evaluate, written down as a document.
///
/// The fields come out in dependency order by construction, but nothing here relies on
/// that — the bake sorts them itself and fails on a cycle.
pub fn build(config: &TerrainConfig, size: UVec2) -> Terrain {
    let spec = region_spec(config);
    let seed = config.seed;

    let mut terrain = Terrain::new(size);

    // The recipe columns. Each is the same region op read for a different column, which
    // is what makes a boundary a structure rather than a rule about a pair of biomes.
    for column in COLUMNS {
        terrain = terrain.with_field(
            Field::new(column_field(column))
                .with_shift(COLUMN_SHIFT)
                .with_range(column_range(column))
                .with_layer(
                    Layer::new(LayerOp::Regions {
                        spec: spec.clone(),
                        output: RegionOutput::Blended(column.to_owned()),
                    })
                    .with_blend(Blend::Replace),
                ),
        );
    }

    // Which region a tile is in, and which supplies its cover. Categorical, so the
    // library reads them at their nearest texel with nothing here saying so.
    let region_count = BIOME_TABLE.len() as f32;
    for (id, output) in [
        (REGION_ID, RegionOutput::RegionId),
        (COVER_CLASS, RegionOutput::CoverClass),
    ] {
        terrain = terrain.with_field(
            Field::new(id)
                .with_shift(shift_of(id))
                .with_range((0.0, region_count))
                .with_layer(
                    Layer::new(LayerOp::Regions {
                        spec: spec.clone(),
                        output,
                    })
                    .with_blend(Blend::Replace),
                ),
        );
    }

    let raw = [
        (
            CONTINENT,
            noise(
                CONTINENT_SALT,
                NoiseKind::Fbm,
                seed,
                config.continent_scale,
                CONTINENT_OCTAVES,
            ),
        ),
        (
            RELIEF_RAW,
            noise(
                ELEVATION_SALT,
                NoiseKind::Fbm,
                seed,
                config.relief_scale,
                DEFAULT_OCTAVES,
            ),
        ),
        (
            RIDGE_RAW,
            noise(
                RIDGE_SALT,
                NoiseKind::Ridged,
                seed,
                config.ridge_scale,
                RIDGE_OCTAVES,
            ),
        ),
        (
            VEGETATION_RAW,
            noise(
                VEGETATION_SALT,
                NoiseKind::Fbm,
                seed,
                config.vegetation_scale,
                DEFAULT_OCTAVES,
            ),
        ),
        (
            HUMIDITY_RAW,
            noise(
                HUMIDITY_SALT,
                NoiseKind::Fbm,
                seed,
                config.humidity_scale,
                DEFAULT_OCTAVES,
            ),
        ),
        (
            TEMPERATURE_RAW,
            noise(
                TEMPERATURE_SALT,
                NoiseKind::Fbm,
                seed,
                config.temperature_scale,
                DEFAULT_OCTAVES,
            ),
        ),
        (
            SETTLEMENT,
            noise(
                SETTLEMENT_SALT,
                NoiseKind::Fbm,
                seed,
                config.settlement_scale,
                DEFAULT_OCTAVES,
            ),
        ),
    ];
    for (id, spec) in raw {
        terrain = terrain.with_field(noise_field(id, spec, (0.0, 1.0)));
    }

    // The two anisotropic layers. Both are the same fbm read on a world strike, and the
    // strikes differ on purpose — two of them on one angle compound into a single
    // stripe pattern.
    terrain = terrain.with_field(noise_field(
        HARDNESS,
        noise(
            LITHOLOGY_SALT,
            NoiseKind::Fbm,
            seed,
            config.lithology_scale,
            LITHOLOGY_OCTAVES,
        )
        .with_transform(SampleTransform {
            strike_degrees: config.lithology_strike_degrees,
            aspect: config.lithology_aspect,
        }),
        (0.0, 1.0),
    ));
    terrain = terrain.with_field(noise_field(
        DUNE_RAW,
        noise(
            DUNE_SALT,
            NoiseKind::Ridged,
            seed,
            config.dune_scale,
            DUNE_OCTAVES,
        )
        .with_transform(SampleTransform {
            strike_degrees: config.dune_wind_degrees,
            aspect: config.dune_aspect,
        }),
        (0.0, 1.0),
    ));

    // How high a crest stands here, which is zero everywhere no recipe weighs the layer.
    terrain = terrain.with_field(
        Field::new(DUNE)
            .with_shift(shift_of(DUNE))
            .with_range((0.0, 1.0))
            .with_layer(weighted_by(DUNE_RAW, COL_DUNE)),
    );

    // Just the layers that vary at the scale a slope is measured over — the term the
    // height and the soil both read, and the reason the soil costs one field rather
    // than a second evaluation of the whole stack.
    terrain = terrain.with_field(
        Field::new(RELIEF_HEIGHT)
            .with_shift(shift_of(RELIEF_HEIGHT))
            .with_range((-1.0, 1.0))
            .with_layer(constant_weighted_by(-0.5, COL_RELIEF))
            .with_layer(weighted_by(RELIEF_RAW, COL_RELIEF))
            .with_layer(weighted_by(RIDGE_RAW, COL_RIDGE)),
    );

    // The continent and relief layers are displacements about zero, so they lower the
    // ground as well as raise it; the ridged and dune layers are one-sided.
    terrain = terrain.with_field(
        Field::new(HEIGHT)
            .with_shift(shift_of(HEIGHT))
            .with_range((0.0, 1.0))
            .with_layer(field_ref(&column_field(COL_BASE_HEIGHT), 1.0))
            .with_layer(Layer::new(LayerOp::Constant(
                -0.5 * config.continent_relief,
            )))
            .with_layer(field_ref(CONTINENT, config.continent_relief))
            .with_layer(field_ref(RELIEF_HEIGHT, 1.0))
            .with_layer(Layer::new(LayerOp::Constant(
                -0.5 * config.lithology_relief,
            )))
            .with_layer(field_ref(HARDNESS, config.lithology_relief))
            .with_layer(field_ref(DUNE, config.dune_relief)),
    );

    // How much of the slope has stripped the ground bare, on its own so the clamp at
    // zero happens before the biome's own bias is added to it.
    terrain = terrain.with_field(
        Field::new(SOIL_BASE)
            .with_shift(shift_of(SOIL_BASE))
            .with_range((0.0, 1.0))
            .with_layer(Layer::new(LayerOp::Constant(1.0)))
            .with_layer(
                Layer::new(LayerOp::Slope {
                    of: RELIEF_HEIGHT.into(),
                    sample_tiles: config.soil_slope_tiles.max(1.0),
                    // The steepest axis rather than the gradient, and the difference is
                    // not cosmetic: measured against the sampler it moved soil by 0.109
                    // on average, against a `bedrock_max` of 0.22. A gradient is the
                    // better estimate of the surface; this is the climb a thing on the
                    // lattice actually makes, which is the question soil was always
                    // asking.
                    mode: SlopeMode::SteepestAxis,
                })
                .with_amplitude(-1.0 / config.soil_slope_falloff.max(1e-6)),
            ),
    );

    terrain = terrain.with_field(
        Field::new(SOIL)
            .with_shift(shift_of(SOIL))
            .with_range((0.0, 1.0))
            .with_layer(field_ref(SOIL_BASE, 1.0))
            .with_layer(field_ref(HARDNESS, -config.lithology_soil_strip))
            .with_layer(field_ref(&column_field(COL_SOIL_BIAS), 1.0))
            .with_layer(field_ref(DUNE, DUNE_SOIL_GAIN)),
    );

    // A live sand face carries nothing, which is the whole of how a dune reaches the
    // tileset: through the cover ladder rather than through a rule naming a desert.
    terrain = terrain.with_field(
        Field::new(VEGETATION)
            .with_shift(shift_of(VEGETATION))
            .with_range((0.0, 1.0))
            .with_layer(field_ref(VEGETATION_RAW, 1.0))
            .with_layer(field_ref(&column_field(COL_VEGETATION_BIAS), 1.0))
            .with_layer(field_ref(DUNE, -DUNE_VEGETATION_BITE)),
    );

    terrain = terrain.with_field(
        Field::new(HUMIDITY)
            .with_shift(shift_of(HUMIDITY))
            .with_range((0.0, 1.0))
            .with_layer(field_ref(HUMIDITY_RAW, 1.0))
            .with_layer(field_ref(&column_field(COL_HUMIDITY_BIAS), 1.0)),
    );

    // Degrees Celsius, so the range is a physical band rather than the crate's usual
    // unit one — a freezing point has to mean something.
    terrain = terrain.with_field(
        Field::new(TEMPERATURE)
            .with_shift(shift_of(TEMPERATURE))
            .with_range(TEMPERATURE_RANGE)
            .with_layer(Layer::new(LayerOp::Constant(config.sea_level_celsius)))
            .with_layer(field_ref(HEIGHT, -config.lapse_celsius))
            .with_layer(field_ref(&column_field(COL_TEMPERATURE_BIAS), 1.0))
            .with_layer(Layer::new(LayerOp::Constant(
                -0.5 * config.temperature_noise_celsius,
            )))
            .with_layer(field_ref(TEMPERATURE_RAW, config.temperature_noise_celsius)),
    );

    terrain
}

/// Where the world's water comes from, as the recipe `watershed` solves it by.
///
/// TODO(jb-doc): what naming a moisture field buys, and which coupling it replaces;
/// and why the height field named here has to be the shift-0 one.
pub fn water_spec(config: &TerrainConfig) -> WaterSpec {
    WaterSpec::new(HEIGHT)
        .with_moisture(HUMIDITY)
        .with_lake_min_cells(config.lake_min_tiles)
}

/// The fields something outside this module reads once the world is generated.
///
/// Everything else in the document is scaffolding: `classify` consumes `soil`,
/// `vegetation`, `hardness` and the two categorical fields once per tile and never again,
/// and `city.rs` reads `settlement` only while it is choosing sites. Naming the survivors
/// is what lets a staged bake drop the rest as it goes — see [`stages`].
#[cfg_attr(not(test), allow(dead_code))]
pub const LIVE: [&str; 3] = [HEIGHT, HUMIDITY, TEMPERATURE];

/// The fields that have to survive a whole bake, because `classify` reads them per tile
/// and `city.rs` reads the settlement score per candidate site.
///
/// This is what a staged bake is told to keep; everything not named here is scaffolding
/// and is dropped as soon as the last field reading it is done.
pub const GENERATION: [&str; 9] = [
    HEIGHT,
    SOIL,
    VEGETATION,
    HUMIDITY,
    TEMPERATURE,
    HARDNESS,
    REGION_ID,
    COVER_CLASS,
    SETTLEMENT,
];

/// The fields `generate_chunk` and the planning stages read, in the order a bake must
/// produce them, each paired with what may be dropped once it is done.
///
/// **The releases are the point.** A whole-world document is 754 MB with every field
/// resident and nothing outside this module ever wants most of them; releasing an
/// intermediate the moment the last field that reads it has been baked turns the peak
/// into the widest live set instead of the sum. The schedule is derived rather than
/// written down — a field is dropped after the last stage that names it as a dependency —
/// so it cannot fall out of step with the layer stacks above.
///
/// `keep` is what survives the whole bake: [`LIVE`] plus whatever the caller says it
/// still needs, which during generation is every field `classify` reads.
pub fn stages(terrain: &Terrain, keep: &[&str]) -> Result<Vec<Stage>, watershed::BakeError> {
    let order = terrain.bake_order()?;

    // The last stage that reads each field, so a field can be dropped after it.
    let mut last_read: Vec<(String, usize)> = Vec::new();
    for (index, id) in order.iter().enumerate() {
        let Some(field) = terrain.field(id.as_str()) else {
            continue;
        };
        for dependency in field.dependencies() {
            let name = dependency.to_string();
            match last_read.iter_mut().find(|(seen, _)| *seen == name) {
                Some((_, at)) => *at = index,
                None => last_read.push((name, index)),
            }
        }
    }

    Ok(order
        .iter()
        .enumerate()
        .map(|(index, id)| Stage {
            field: id.to_string(),
            release: last_read
                .iter()
                .filter(|(_, at)| *at == index)
                .map(|(name, _)| name.clone())
                .filter(|name| !keep.contains(&name.as_str()))
                .collect(),
        })
        .collect())
}

/// One step of a staged bake: the field to bake, and what dies with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage {
    pub field: String,
    pub release: Vec<String>,
}

/// The octave count a field takes when it does not ask for one.
const DEFAULT_OCTAVES: u32 = 5;

/// Wide enough that no configuration of the lapse rate and the biases clips against it.
const TEMPERATURE_RANGE: (f32, f32) = (-100.0, 100.0);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gameplay::biome::Biome;
    use crate::gameplay::terrain::shared_test_sampler;

    /// A corner of the real world rather than a world of its own: every field here is
    /// anchored on the world origin, so tiles 0..N are the tiles 0..N of the shipped
    /// map and no scaling enters the comparison.
    const SIZE: u32 = 256;

    /// How far in from the edge a comparison starts.
    ///
    /// **A window is exact, but not right up to its edge**, and the reach is set by the
    /// *coarsest* field rather than by the finest. Two things push it in:
    ///
    /// - A raster read clamps past the last texel instead of the terrain continuing, so
    ///   a field of texel size `t` is wrong within `t` of the border. `continent` is at
    ///   shift 5 — 32 tiles a texel — and `height` reads it, so every field derived from
    ///   the height inherits that.
    /// - `soil` reads a slope over `soil_slope_tiles`, which is 12 more.
    ///
    /// 64 covers both with room to spare. Getting this wrong does not make the test
    /// flaky, it makes it *wrong*: at 16 it reported a 6.5e-3 temperature difference that
    /// was entirely the continent field clamping, and nothing about the translation.
    const MARGIN: u32 = 64;

    /// What a bake of the whole world would cost, in time and in memory, extrapolated
    /// from a tile of it.
    ///
    /// The extrapolation is honest for the *texels* — every field is anchored on the
    /// world origin and costs the same per texel wherever it is — and optimistic for the
    /// region ops by a constant: their cell table is built once per bake and covers the
    /// document, so a 4096 bake pays for it once rather than 64 times.
    ///
    /// `cargo test --release -- --ignored --nocapture the_document_measures_a_whole_world`
    #[test]
    #[ignore]
    fn the_document_measures_a_whole_world() {
        use std::time::Instant;

        const SAMPLE: u32 = 512;
        const WORLD: u32 = 4096;
        let scale = (WORLD as f64 / SAMPLE as f64).powi(2);

        let config = TerrainConfig::default();
        let mut terrain = build(&config, UVec2::splat(SAMPLE));
        let started = Instant::now();
        terrain.bake().expect("the document has to bake");
        let elapsed = started.elapsed().as_secs_f64();

        let mut bytes = 0usize;
        let mut rows = Vec::new();
        for field in &terrain.fields {
            let resolution = field.resolution(UVec2::splat(WORLD));
            let field_bytes = resolution.x as usize * resolution.y as usize * 4;
            bytes += field_bytes;
            rows.push((
                field.id.to_string(),
                field.shift,
                field_bytes as f64 / (1 << 20) as f64,
            ));
        }
        rows.sort_by(|a, b| b.2.total_cmp(&a.2));

        // What a shipped asset would weigh: only the fields something outside this
        // module reads, and only their bakes.
        let runtime: Vec<&str> = vec![
            HEIGHT,
            SOIL,
            VEGETATION,
            HUMIDITY,
            TEMPERATURE,
            HARDNESS,
            REGION_ID,
            COVER_CLASS,
            SETTLEMENT,
        ];
        let runtime_bytes: usize = runtime
            .iter()
            .filter_map(|id| terrain.field(id))
            .map(|field| {
                let resolution = field.resolution(UVec2::splat(WORLD));
                resolution.x as usize * resolution.y as usize * 4
            })
            .sum();

        let path = std::env::temp_dir().join("wusel-document-measure.watershed");
        let weigh = |options| {
            terrain
                .save_to_path(&path, options)
                .expect("a document has to save");
            let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            let _ = std::fs::remove_file(&path);
            bytes
        };
        let on_disk = weigh(watershed::SaveOptions::full());
        // The shipped shape: the numbers a bake needs and nothing a bake can produce.
        let regenerable = weigh(watershed::SaveOptions::layers_only());

        // What a staged bake that drops its scaffolding actually has to hold. Everything
        // `classify` reads is kept, since generation is the widest moment.
        let generation: Vec<&str> = vec![
            HEIGHT,
            SOIL,
            VEGETATION,
            HUMIDITY,
            TEMPERATURE,
            HARDNESS,
            REGION_ID,
            COVER_CLASS,
            SETTLEMENT,
        ];
        let mut staged = build(&config, UVec2::splat(SAMPLE));
        let schedule = stages(&staged, &generation).expect("the document has to plan");
        let mut peak = 0usize;
        let started = Instant::now();
        for stage in &schedule {
            staged
                .bake_field(&stage.field)
                .expect("a stage has to bake");
            for id in &stage.release {
                staged.release(id);
            }
            peak = peak.max(staged.baked_bytes());
        }
        let staged_elapsed = started.elapsed().as_secs_f64();
        let settled = staged.baked_bytes();

        println!(
            "\n{} fields, bake of {SAMPLE} took {elapsed:.2}s",
            rows.len()
        );
        println!("a {WORLD} bake extrapolates to {:.0}s", elapsed * scale);
        println!("\nfield                     shift      MB");
        for (id, shift, megabytes) in &rows {
            println!("{id:<26}{shift:>4}{megabytes:>9.1}");
        }
        println!(
            "{:<26}{:>4}{:>9.1}",
            "TOTAL",
            "",
            bytes as f64 / (1 << 20) as f64
        );
        println!(
            "\nthe {} fields something outside this module reads: {:.0} MB at {WORLD}",
            runtime.len(),
            runtime_bytes as f64 / (1 << 20) as f64
        );
        println!(
            "a full {SAMPLE} document on disk: {:.1} MB, so ~{:.0} MB at {WORLD}",
            on_disk as f64 / (1 << 20) as f64,
            on_disk as f64 * scale / (1 << 20) as f64
        );
        println!("`WorldMap` holds 32 MB of kinds and heights today");
        println!(
            "the same document as specs alone: {regenerable} bytes, and it does not grow with the world"
        );
        println!(
            "\n{} stages, releasing as they go, took {staged_elapsed:.2}s at {SAMPLE}",
            schedule.len()
        );
        println!(
            "peak {:.0} MB at {WORLD}, settling to {:.0} MB with generation's fields kept",
            peak as f64 * scale / (1 << 20) as f64,
            settled as f64 * scale / (1 << 20) as f64
        );
        let live: usize = LIVE
            .iter()
            .filter_map(|id| terrain.field(id))
            .map(|field| {
                let resolution = field.resolution(UVec2::splat(WORLD));
                resolution.x as usize * resolution.y as usize * 4
            })
            .sum();
        println!(
            "and {:.1} MB once generation is over and only {LIVE:?} is live",
            live as f64 / (1 << 20) as f64
        );
    }

    /// Where the wait after pressing play actually goes, stage by stage, at the size the
    /// game bakes.
    ///
    /// `cargo test --release -- --ignored --nocapture the_bake_measures_where_the_wait_goes`
    #[test]
    #[ignore]
    fn the_bake_measures_where_the_wait_goes() {
        use std::time::Instant;

        let config = TerrainConfig::default();
        let mut terrain = config.document(UVec2::splat(4096));
        let schedule = stages(&terrain, &GENERATION).expect("the document has to plan");

        // What `classify` needs before a single chunk can be cut. Everything after the
        // last of these is a field the *streamer* does not wait on.
        let classify: [&str; 6] = [HEIGHT, SOIL, VEGETATION, HARDNESS, REGION_ID, COVER_CLASS];

        let mut elapsed = Vec::new();
        let started = Instant::now();
        let mut classify_ready = None;
        for stage in &schedule {
            let at = Instant::now();
            terrain
                .bake_field(&stage.field)
                .expect("a stage has to bake");
            for id in &stage.release {
                terrain.release(id);
            }
            elapsed.push((stage.field.clone(), at.elapsed().as_secs_f64()));
            if classify_ready.is_none()
                && classify
                    .iter()
                    .all(|id| terrain.field(id).is_some_and(|f| !f.baked().is_empty()))
            {
                classify_ready = Some(started.elapsed().as_secs_f64());
            }
        }
        let total = started.elapsed().as_secs_f64();

        let mut ranked = elapsed.clone();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!("\nthe ten dearest stages of {}:", schedule.len());
        for (id, seconds) in ranked.iter().take(10) {
            println!("  {id:<26}{seconds:6.2}s  {:4.1}%", 100.0 * seconds / total);
        }
        println!("\nwhole bake {total:.2}s");
        match classify_ready {
            Some(at) => println!(
                "everything `classify` reads is done at {at:.2}s — {:.0}% of the wait, \
                 and the {:.2}s after it is field nothing the streamer waits on",
                100.0 * at / total,
                total - at
            ),
            None => println!("classify's fields were never all baked"),
        }
    }

    fn baked(config: &TerrainConfig) -> Terrain {
        let mut terrain = build(config, UVec2::splat(SIZE));
        terrain.bake().expect("the document has to bake");
        terrain
    }

    /// A window of the world holds the same ground as the whole of it.
    ///
    /// **The successor to the translation guard, and the reason that guard could go.**
    /// While `TerrainSampler` was analytic there was a second implementation to check the
    /// document against, and it was checked: hardness and humidity came out *exact*,
    /// height agreed to a mean of 1.8e-7 and a worst of 2.3e-5, vegetation to 6.4e-7, and
    /// soil differed only where the document reads the relief actually next door where
    /// the sampler re-used the centre tile's recipe to dodge a second blend. Those
    /// figures are the record that the translation was faithful. The comparison itself
    /// cannot be repeated, because there is now exactly one answer to "how high is it
    /// here" — which was the point of the whole stage.
    ///
    /// What is still checkable, and what every test baking a small document rests on, is
    /// that a document is a **corner of the real world rather than a world of its own**:
    /// every field is anchored on the world origin, so tiles 0..N of a window are tiles
    /// 0..N of the shipped map. That is the document-level form of
    /// `a_tile_does_not_depend_on_where_the_chunk_boundary_falls`, and it is exact — a
    /// window is not an approximation of the world, it is a piece of it.
    ///
    /// The margin is not optional: within one `soil_slope_tiles` of its edge a window's
    /// slope-reading fields see a clamped raster rather than the terrain continuing, so
    /// they are wrong there and known to be.
    #[test]
    fn a_window_of_the_world_holds_the_same_ground_as_the_whole_of_it() {
        let config = TerrainConfig::default();
        let window = baked(&config);
        let world = shared_test_sampler();

        // Both sides are asked the same question the same way. `TerrainSampler` adds the
        // half-tile that lands a read on its cell's centre, so its caller passes the
        // integer tile; the document is asked for that cell directly.
        let at = |id: &str, i: u32, j: u32| {
            window
                .sample(id, i as f32 + 0.5, j as f32 + 0.5)
                .expect("a field the document declares has to be sampleable")
        };

        let fields = [HEIGHT, SOIL, VEGETATION, HARDNESS, HUMIDITY, TEMPERATURE];
        let mut worst = [0.0f32; 6];
        for j in MARGIN..SIZE - MARGIN {
            for i in MARGIN..SIZE - MARGIN {
                let (x, y) = (i as f32, j as f32);
                let sample = world.sample(x, y);
                let differences = [
                    (at(HEIGHT, i, j) - sample.elevation).abs(),
                    (at(SOIL, i, j) - sample.soil).abs(),
                    (at(VEGETATION, i, j) - sample.vegetation).abs(),
                    (at(HARDNESS, i, j) - sample.hardness).abs(),
                    (at(HUMIDITY, i, j) - world.humidity(x, y)).abs(),
                    (at(TEMPERATURE, i, j) - world.temperature(x, y)).abs(),
                ];
                for (slot, difference) in differences.iter().enumerate() {
                    worst[slot] = worst[slot].max(*difference);
                }
            }
        }

        for (id, difference) in fields.iter().zip(worst) {
            assert_eq!(
                difference, 0.0,
                "{id} differs by {difference} between a window and the whole world"
            );
        }
    }

    /// The region map is the load-bearing piece: if a tile landed in a different region
    /// in a window than in the world, every test that bakes a small document would be
    /// quietly measuring somewhere else.
    #[test]
    fn every_tile_lands_in_the_region_the_biome_map_put_it_in() {
        let config = TerrainConfig::default();
        let window = baked(&config);
        let world = shared_test_sampler();

        let index_of = |biome: Biome| {
            BIOME_TABLE
                .iter()
                .position(|(candidate, _)| *candidate == biome)
                .expect("every biome is in the table") as f32
        };

        let mut regions = 0u32;
        let mut covers = 0u32;
        for j in MARGIN..SIZE - MARGIN {
            for i in MARGIN..SIZE - MARGIN {
                let (x, y) = (i as f32 + 0.5, j as f32 + 0.5);
                let sample = world.sample(i as f32, j as f32);
                if window.sample(REGION_ID, x, y) != Some(index_of(sample.dominant)) {
                    regions += 1;
                }
                if window.sample(COVER_CLASS, x, y) != Some(index_of(sample.cover)) {
                    covers += 1;
                }
            }
        }
        assert_eq!(regions, 0, "{regions} tiles land in a different region");
        assert_eq!(covers, 0, "{covers} tiles take their cover from elsewhere");
    }

    /// Not a translation guard but a construction one: a blended column is a weighted
    /// mean of the table's rows, so a field whose range clipped one would silently
    /// floor every negative bias at zero and no other test would notice.
    #[test]
    fn no_recipe_column_is_clipped_by_the_field_that_carries_it() {
        let config = TerrainConfig::default();
        let terrain = baked(&config);

        for column in COLUMNS {
            let (low, high) = column_range(column);
            let field = terrain
                .field(&column_field(column))
                .expect("every column has a field");
            assert_eq!(field.bounds(), (low.min(high), low.max(high)));

            let mut seen = f32::MAX;
            let mut largest = f32::MIN;
            for j in 0..SIZE {
                for i in 0..SIZE {
                    let value = field.sample(i as f32 + 0.5, j as f32 + 0.5);
                    seen = seen.min(value);
                    largest = largest.max(value);
                }
            }
            assert!(
                seen >= low.min(high) - 1e-6 && largest <= low.max(high) + 1e-6,
                "{column} reaches {seen}..{largest}, outside its table's {low}..{high}"
            );
        }
    }
}
