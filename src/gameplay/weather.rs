//! Weather: cloud patches drifting over the world, the shadow each one throws,
//! and rain in the thick of them.
//!
//! Two textures and one full-screen pass, and the split between them is the whole
//! design:
//!
//! * **Where** it is cloudy is the humidity field — the same field the river
//!   springs read — baked once per session over the whole world. That map never
//!   moves, so a wet range is reliably overcast and a dry one reliably is not.
//! * **What** a cloud looks like is a second map holding one *tiling* period of
//!   the same noise. The shader scrolls it, which is the entire animation.
//!
//! So the shader evaluates no noise: it samples two textures. That is not only
//! ~50x cheaper than an fbm per fragment (which at 4K is the whole frame budget),
//! it also keeps [`crate::gameplay::noise`] the only noise in the crate, which is
//! what the terrain's determinism tests rest on.
//!
//! **This module is the sky and nothing else.** The pass that composites it is
//! [`crate::gameplay::screen`]'s, shared with the terrain's shading and the sun;
//! this bakes the maps, advances the clock and hands the knobs over through
//! [`ScreenOverlay::set_sky`]. The field is anchored in *world* space, not screen
//! space, and the view centre is filled in over there at extract time — after the
//! whole main-world frame, so panning can never drag the clouds a frame behind the
//! terrain under them.
//!
//! Nothing here outlives [`Screen::Gameplay`] except the knobs.

use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{Extent3d, TextureDimension, TextureFormat},
    },
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};

use crate::{
    gameplay::{
        noise::TilingNoiseField,
        screen::ScreenOverlay,
        terrain::{TerrainConfig, TerrainSampler},
        world::WORLD_TILES,
    },
    screens::Screen,
};

/// Salt for the cloud-shape field, so the sky is not a second view of a landscape.
const CLOUD_SHAPE_SALT: u32 = 0xc10d_5eed;

/// Noise cells across one period of the shape map. With the default 256-tile period
/// this puts the coarsest cloud lump at 32 tiles and, over four octaves, the finest
/// at 4 — five texels of a 256px map, which is about as fine as a baked field can
/// carry. It is a power of two because that is what lets every octave's lattice wrap
/// exactly.
const SHAPE_LATTICE_PERIOD: u32 = 8;

/// Everything about the weather that is a knob rather than world state.
///
/// Flat, like [`TerrainConfig`] and `WorldPlanConfig`, and for the same reason: the
/// shader's uniform is flat regardless, and the couplings between these values —
/// `rain_cut` sitting above the cut, `shadow_offset_tiles` needing to clear one
/// cloud's width — read as neighbours here rather than across nested structs.
#[derive(Resource, Clone)]
pub struct WeatherConfig {
    /// Texels along each side of the probability map, covering the whole world: 512
    /// over 4096 tiles is one texel per 8 tiles. The humidity field's coarsest
    /// wavelength is ~50 tiles, so this oversamples it 6x; the finer octaves it
    /// misses carry ~10% of the amplitude, and a texel sits within 0.05 of the
    /// average over its own 8x8 tiles.
    pub probability_texels_per_side: u32,
    /// Texels along each side of the tiling shape map. 256 over a 256-tile period is
    /// one tile per texel.
    pub shape_texels_per_side: u32,
    /// How much world one repeat of the shape map covers, which sets how big a cloud
    /// is: the coarsest lump in the field is an eighth of it.
    ///
    /// This is the one real tension in the whole overlay. At 512 a cloud was 64 tiles
    /// across — half the screen at scale 1, so the sky read as fog banks rather than
    /// weather. At 256 a cloud is ~32 tiles and a handful fit on screen, at the cost
    /// of the period repeating ~4 times across the widest zoom-out; the second layer
    /// at a fractional scale is what keeps that from reading as a pattern.
    pub shape_period_tiles: f32,
    /// Octaves baked into the shape map. Four, because the fifth would land inside
    /// two texels and only alias.
    pub shape_octaves: u32,
    /// How fast the sky moves, in tiles per second. A chunk is 64 tiles, so 2
    /// crosses one every 32 seconds.
    pub wind_drift_tiles_per_second: Vec2,
    /// The second shape layer's frequency, relative to the first. Deliberately not
    /// a whole number: the two layers beat against each other instead of lining up
    /// every period.
    pub cloud_fine_scale: f32,
    /// How fast the second layer drifts, relative to the first. Below 1 so the
    /// layers shear rather than sliding as one sheet — this is what makes a cloud
    /// look like it is changing shape rather than merely moving.
    pub cloud_fine_drift: f32,
    /// The first layer's share of the shape. The rest is the second layer.
    pub cloud_coarse_weight: f32,
    /// The cut in the raw cloud field below which the sky is clear.
    ///
    /// A cut on a field, *not* a coverage fraction, and sharper than it looks:
    /// probability times shape has a mean of 0.250, so at 0.20 about 62% of the world
    /// is under cloud, at 0.28 it is 35.9% under cloud with 64.4% carrying some, at
    /// 0.40 about 12%, and anything above ~0.55 is a permanently clear sky.
    pub cloud_cut: f32,
    /// Width of the smooth step around `cloud_cut` — the softness of a cloud edge.
    pub cloud_softness: f32,
    /// How bright the cloud itself is. Under 1 because pure white over 8px pixel
    /// art reads as a hole in the world rather than as weather.
    pub cloud_brightness: f32,
    /// How opaque the thickest cloud is allowed to get. Well under 1: at full
    /// opacity a cloud replaces the world under it, and what you want to see is
    /// terrain *through* weather.
    pub cloud_opacity: f32,
    /// Where a cloud's shadow falls, in tiles: cloud altitude times sun angle. The
    /// shadow of a cloud lands `-this` from it, so the default puts it down and to
    /// the right, with the sun up and to the left.
    ///
    /// Has to be a decent fraction of a cloud's own width or the shadow hides
    /// underneath the cloud casting it, and the coarsest lump is
    /// `shape_period_tiles / 8` across.
    pub shadow_offset_tiles: Vec2,
    /// How much of the light a full cloud takes away.
    pub shadow_strength: f32,
    /// How thick a cloud has to be before it rains, as a cut on the raw field rather
    /// than on the density: the density saturates, so almost every cloud clears any
    /// cut placed on it, and cutting there rained on 28.6% of the world at once. At
    /// 0.46 some rain falls on 4.2% of it and rain worth looking at on 1.6%.
    ///
    /// The rain is still *multiplied* by the density, which is what keeps "no rain
    /// out of a clear sky" true for any setting of these knobs rather than only for
    /// ones where this sits above `cloud_cut`.
    pub rain_cut: f32,
    /// Width of the ramp above `rain_cut`, so a rain patch fades in from the edge of
    /// the cloud's thick part instead of arriving with a rim.
    pub rain_softness: f32,
    /// How hard the rain darkens and greys what is under it.
    pub rain_strength: f32,
    /// Streak periods per second. The streaks are drawn in screen space — in world
    /// space they would be 16x denser at one end of the zoom range than the other.
    pub rain_streak_speed: f32,
}

impl Default for WeatherConfig {
    fn default() -> Self {
        Self {
            probability_texels_per_side: 512,
            shape_texels_per_side: 256,
            shape_period_tiles: 256.0,
            shape_octaves: 4,
            wind_drift_tiles_per_second: Vec2::new(2.0, 0.6),
            cloud_fine_scale: 2.6,
            cloud_fine_drift: 0.55,
            cloud_coarse_weight: 0.65,
            cloud_cut: 0.30,
            cloud_softness: 0.08,
            cloud_brightness: 0.92,
            cloud_opacity: 0.55,
            shadow_offset_tiles: Vec2::new(-8.0, 6.0),
            shadow_strength: 0.38,
            rain_cut: 0.46,
            rain_softness: 0.18,
            rain_strength: 0.45,
            rain_streak_speed: 1.8,
        }
    }
}

/// Where the sky has drifted to. In map periods, wrapped to `0..1`, which is what
/// keeps a long session from quantizing: the offsets never grow.
#[derive(Resource, Default)]
struct WeatherClock {
    coarse_offset: Vec2,
    fine_offset: Vec2,
    streak_phase: f32,
}

/// The bake in flight. Dropping the resource cancels it, so maps baked for one
/// world can never land in the next.
#[derive(Resource)]
struct WeatherBake(Task<BakedMaps>);

struct BakedMaps {
    probability: Image,
    shape: Image,
}

/// The two maps, once they are on the GPU's side of the asset server.
///
/// Absent until the bake lands, and that absence is the feature: with no maps the
/// pass leaves the scene alone, so the first few frames of a session are a clear
/// sky rather than a stall.
#[derive(Resource, Clone)]
pub(crate) struct WeatherMaps {
    // Read by `screen.rs`, which binds them: this module decides what is in a map and
    // that one decides how it reaches a fragment.
    pub(super) probability: Handle<Image>,
    pub(super) shape: Handle<Image>,
}

impl ExtractResource for WeatherMaps {
    type Source = Self;

    fn extract_resource(source: &Self) -> Self {
        source.clone()
    }
}

pub struct WeatherPlugin;

impl Plugin for WeatherPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WeatherConfig>();
        app.add_plugins(ExtractResourcePlugin::<WeatherMaps>::default());
        app.add_systems(OnEnter(Screen::Gameplay), start_weather_bake);
        app.add_systems(OnExit(Screen::Gameplay), end_weather);
        app.add_systems(
            Update,
            (
                finish_weather_bake,
                // The clock is the only writer of its own state and the overlay
                // sync is the only reader, so this pair is the whole ordering the
                // main world needs.
                (advance_weather_clock, sync_weather_overlay).chain(),
            )
                .run_if(in_state(Screen::Gameplay)),
        );
    }
}

// -- The main world ----------------------------------------------------------

fn start_weather_bake(
    mut commands: Commands,
    terrain: Res<TerrainConfig>,
    config: Res<WeatherConfig>,
) {
    // At zero rather than wherever the last session left off: the sky is world
    // state, and a new world gets a new one.
    commands.insert_resource(WeatherClock::default());
    // The CPU-side sky, built once because its field costs the same to construct
    // as any other. It goes in beside the clock and out beside it, so nothing can
    // read one session's weather against the next session's world.
    commands.insert_resource(SkySampler::new(&terrain, &config));

    let terrain = terrain.clone();
    let config = config.clone();
    let task = AsyncComputeTaskPool::get().spawn(async move { bake_maps(&terrain, &config) });
    commands.insert_resource(WeatherBake(task));
}

/// Drops the session's state. A bake still in flight goes with it, so maps built for
/// one world can never land in the next.
///
/// The overlay needs no help here: it belongs to [`crate::gameplay::screen`], which
/// takes it off the camera on the same transition.
fn end_weather(mut commands: Commands) {
    commands.remove_resource::<WeatherMaps>();
    commands.remove_resource::<WeatherClock>();
    commands.remove_resource::<WeatherBake>();
    commands.remove_resource::<SkySampler>();
}

fn finish_weather_bake(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    bake: Option<ResMut<WeatherBake>>,
) {
    let Some(mut bake) = bake else {
        return;
    };
    let Some(baked) = block_on(poll_once(&mut bake.0)) else {
        return;
    };

    commands.insert_resource(WeatherMaps {
        probability: images.add(baked.probability),
        shape: images.add(baked.shape),
    });
    commands.remove_resource::<WeatherBake>();
}

fn advance_weather_clock(
    time: Res<Time>,
    config: Res<WeatherConfig>,
    mut clock: ResMut<WeatherClock>,
    mut sky: ResMut<SkySampler>,
) {
    let delta = time.delta_secs();

    // The sky drifts itself and the clock reads it back, so there is one copy of
    // where the weather is rather than two kept in step. `streak_phase` is the
    // clock's own: streaks are screen-space decoration and nothing off screen can be
    // rained on by them, so the sampler has no use for it.
    sky.drift(delta);
    (clock.coarse_offset, clock.fine_offset) = sky.offsets();
    clock.streak_phase = (clock.streak_phase + config.rain_streak_speed * delta).fract();
}

/// Hands this frame's sky to the one pass that draws it. The clock goes over as
/// three numbers rather than as itself, which is what keeps [`WeatherClock`] private
/// on the same terms [`SkySampler`] keeps the field salts private.
fn sync_weather_overlay(
    config: Res<WeatherConfig>,
    clock: Res<WeatherClock>,
    mut overlay: Single<&mut ScreenOverlay>,
) {
    overlay.set_sky(
        &config,
        clock.coarse_offset,
        clock.fine_offset,
        clock.streak_phase,
    );
}

// -- The bake ----------------------------------------------------------------

/// The probability that it is cloudy over a tile: the humidity the terrain sampler
/// reports there, unchanged. Rivers rise in the wet mountains and it rains over the
/// same country — and since the biome's `humidity_bias` is part of that answer, a
/// desert is reliably clear and a wetland reliably overcast without the sky knowing
/// what a biome is.
fn cloud_probability_at(sampler: &TerrainSampler, tile: Vec2) -> f32 {
    sampler.humidity(tile.x, tile.y)
}

/// How much cloud is over a tile: the density, from the raw field there.
///
/// Zero is clear sky and one is solid overcast, and the step between them is sharp
/// enough that most of a cloud is at one or the other — which is why the rain below
/// is cut on the raw field instead.
///
/// The shader is the implementation — the density has to be evaluated per fragment,
/// because it moves. This is the same arithmetic in Rust so that the properties the
/// sky is supposed to have can be measured over the whole world without a GPU; the
/// two are edited together, like the uniform and its wgsl struct.
///
/// No longer test-only: [`SkySampler`] is the simulation's reader, and it goes
/// through exactly these functions so that the crate holds two transcriptions of
/// the sky's arithmetic rather than three.
fn cloud_density(config: &WeatherConfig, field: f32) -> f32 {
    smoothstep(
        config.cloud_cut - config.cloud_softness,
        config.cloud_cut + config.cloud_softness,
        field,
    )
}

/// How hard it is raining under a cloud, from the raw field and the density it
/// produced. Ramped rather than switched, so a rain patch has no rim, and multiplied
/// by the density, so rain out of a clear sky is not unlikely but impossible.
fn rain_amount(config: &WeatherConfig, field: f32, density: f32) -> f32 {
    smoothstep(
        config.rain_cut,
        config.rain_cut + config.rain_softness,
        field,
    ) * density
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Where a tile falls in the shape map's lattice. The map holds one tiling period,
/// so this is the one conversion between world tiles and the field's own space —
/// written here once rather than at each of the three places that used to do it.
fn shape_cell(config: &WeatherConfig, tile: Vec2) -> Vec2 {
    tile / config.shape_period_tiles * SHAPE_LATTICE_PERIOD as f32
}

/// The two layers of shape the shader mixes, at one point, scrolled to where the
/// clock has drifted them.
///
/// The offsets are in map periods — the same units the shader adds them in — so
/// they are scaled by the lattice period to reach cell space.
fn shape_at(
    field: &TilingNoiseField,
    config: &WeatherConfig,
    cell: Vec2,
    coarse_offset: Vec2,
    fine_offset: Vec2,
) -> f32 {
    let lattice = SHAPE_LATTICE_PERIOD as f32;
    let coarse_at = cell + coarse_offset * lattice;
    let fine_at = cell * config.cloud_fine_scale + fine_offset * lattice;

    let coarse = field.sample(coarse_at.x, coarse_at.y);
    let fine = field.sample(fine_at.x, fine_at.y);
    coarse * config.cloud_coarse_weight + fine * (1.0 - config.cloud_coarse_weight)
}

/// The sky as anything outside this module reads it.
///
/// The overlay is drawn from baked maps on the GPU; this evaluates the same fields
/// directly on the CPU, so the two answers agree to within the shape map's byte
/// quantization and its bilinear filtering rather than exactly. Where they differ,
/// **this one is authoritative** — it is what decides whether a city's harvest was
/// rained on, and the overlay only has to look right.
///
/// Owned here so that [`WeatherClock`] and the field salts stay private: a reader
/// gets an answer, not the machinery. Its absence is a clear sky, the same fallback
/// an unbaked map gives the overlay, which is what lets the simulation run in a test
/// with no weather plugin at all.
/// `Clone` because [`crate::gameplay::ground`]'s step runs on the compute pool and
/// has to own everything it reads — the sampler is two `Vec2`s, a config and a noise
/// field's offset, so copying it per step is nothing beside the 65k evaluations it
/// is copied for.
#[derive(Resource, Clone)]
pub struct SkySampler {
    field: TilingNoiseField,
    config: WeatherConfig,
    coarse_offset: Vec2,
    fine_offset: Vec2,
}

impl SkySampler {
    pub(super) fn new(terrain: &TerrainConfig, config: &WeatherConfig) -> Self {
        Self {
            field: TilingNoiseField::new(
                terrain.seed,
                CLOUD_SHAPE_SALT,
                SHAPE_LATTICE_PERIOD,
                config.shape_octaves,
            ),
            config: config.clone(),
            coarse_offset: Vec2::ZERO,
            fine_offset: Vec2::ZERO,
        }
    }

    /// Drift the sky by `seconds` of game time.
    ///
    /// **The clock's arithmetic, and its only copy.** `advance_weather_clock` calls
    /// this and then reads the offsets back, so the sky the simulation asks and the
    /// sky the overlay draws can never be a frame apart about where the weather is.
    ///
    /// The offsets are wrapped rather than accumulated: an offset that grew all
    /// session would eventually quantize, and a map period is exactly where a wrap is
    /// invisible.
    pub(super) fn drift(&mut self, seconds: f32) {
        let drift =
            self.config.wind_drift_tiles_per_second / self.config.shape_period_tiles * seconds;
        self.coarse_offset = (self.coarse_offset + drift).fract();
        self.fine_offset = (self.fine_offset
            + drift * self.config.cloud_fine_drift * self.config.cloud_fine_scale)
            .fract();
    }

    /// Where it has drifted to, in map periods.
    pub(super) fn offsets(&self) -> (Vec2, Vec2) {
        (self.coarse_offset, self.fine_offset)
    }

    /// How much cloud is over a tile, on 0..1.
    ///
    /// The same density the overlay draws and the shadow is cut from, so the ctl can
    /// report what the screen is showing rather than an approximation of it.
    /// `probability` is the humidity there, passed in for the reason
    /// [`Self::rain_at`] gives.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn cloud_at(&self, tile: Vec2, probability: f32) -> f32 {
        cloud_density(&self.config, probability * self.shape_at(tile))
    }

    /// The two shape layers where the clock has drifted them, at a tile. Shared by
    /// both readings above so they cannot disagree about where the cloud is.
    fn shape_at(&self, tile: Vec2) -> f32 {
        shape_at(
            &self.field,
            &self.config,
            shape_cell(&self.config, tile),
            self.coarse_offset,
            self.fine_offset,
        )
    }

    /// How hard it is raining over a tile, on 0..1.
    ///
    /// `probability` is the humidity there — passed in rather than sampled, because
    /// every caller already holds it and a `TerrainSampler` lookup is ~190 ns.
    pub fn rain_at(&self, tile: Vec2, probability: f32) -> f32 {
        let field = probability * self.shape_at(tile);
        rain_amount(&self.config, field, cloud_density(&self.config, field))
    }
}

/// Bakes both maps. Called on the compute pool: this is ~260k fbm samples for the
/// probability map alone, which is ~50 ms — the same order as the chunk generation
/// the first gameplay frame is already doing, and no reason to add to it.
fn bake_maps(terrain: &TerrainConfig, config: &WeatherConfig) -> BakedMaps {
    BakedMaps {
        probability: bake_probability_map(terrain, config),
        shape: bake_shape_map(terrain, config),
    }
}

fn bake_probability_map(terrain: &TerrainConfig, config: &WeatherConfig) -> Image {
    let side = config.probability_texels_per_side.max(1);
    let tiles_per_texel = WORLD_TILES.x as f32 / side as f32;
    // Hoisted out of the loop: one sampler for the whole map, since building it
    // costs six noise fields and a biome map.
    let sampler = terrain.sampler();

    let mut texels = Vec::with_capacity((side * side) as usize);
    for y in 0..side {
        for x in 0..side {
            // The texel's centre, so the map is the humidity field at the points it
            // claims to sample rather than at their corners.
            let tile = (Vec2::new(x as f32, y as f32) + Vec2::splat(0.5)) * tiles_per_texel;
            texels.push(to_byte(cloud_probability_at(&sampler, tile)));
        }
    }

    // Clamped, so a view of the world's edge reads the edge of the map rather than
    // wrapping the far side of the world into shot.
    map_image(side, texels, ImageAddressMode::ClampToEdge)
}

fn bake_shape_map(terrain: &TerrainConfig, config: &WeatherConfig) -> Image {
    let side = config.shape_texels_per_side.max(1);
    let field = TilingNoiseField::new(
        terrain.seed,
        CLOUD_SHAPE_SALT,
        SHAPE_LATTICE_PERIOD,
        config.shape_octaves,
    );
    let cells_per_texel = SHAPE_LATTICE_PERIOD as f32 / side as f32;

    let mut texels = Vec::with_capacity((side * side) as usize);
    for y in 0..side {
        for x in 0..side {
            let cell = Vec2::new(x as f32, y as f32) * cells_per_texel;
            texels.push(to_byte(field.sample(cell.x, cell.y)));
        }
    }

    // Repeated, because scrolling it forever is the animation.
    map_image(side, texels, ImageAddressMode::Repeat)
}

fn to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// One byte per texel, sampled smoothly.
///
/// The sampler is set here rather than inherited: the app-wide default is *nearest*
/// for the 8px pixel art, and a nearest-sampled probability map would draw the sky
/// in visible 64px blocks.
fn map_image(side: u32, texels: Vec<u8>, address_mode: ImageAddressMode) -> Image {
    let mut image = Image::new(
        Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        texels,
        TextureFormat::R8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        min_filter: ImageFilterMode::Linear,
        mag_filter: ImageFilterMode::Linear,
        address_mode_u: address_mode,
        address_mode_v: address_mode,
        ..default()
    });
    image
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One sample of the sky: the humidity there, the raw cloud field, and the
    /// density that field produces.
    struct Sky {
        probability: f32,
        field: f32,
        density: f32,
    }

    /// A grid of them over the default world, coarse enough to run in a test and
    /// fine enough to see cloud edges: one sample per 16 tiles.
    fn sample_densities() -> Vec<Sky> {
        let terrain = TerrainConfig::default();
        let config = WeatherConfig::default();
        let field = TilingNoiseField::new(
            terrain.seed,
            CLOUD_SHAPE_SALT,
            SHAPE_LATTICE_PERIOD,
            config.shape_octaves,
        );
        let sampler = terrain.sampler();

        let step = 16.0;
        let steps = (WORLD_TILES.x as f32 / step) as u32;
        let mut samples = Vec::new();
        for y in 0..steps {
            for x in 0..steps {
                let tile = Vec2::new(x as f32, y as f32) * step;
                let probability = cloud_probability_at(&sampler, tile);
                let cell = shape_cell(&config, tile);
                let raw = probability * shape_at(&field, &config, cell, Vec2::ZERO, Vec2::ZERO);
                samples.push(Sky {
                    probability,
                    field: raw,
                    density: cloud_density(&config, raw),
                });
            }
        }
        samples
    }

    /// The weather equivalent of `the_default_config_produces_every_base_kind`: the
    /// failure this guards against is a default that quietly yields one sky — either
    /// a world under permanent overcast or one where it never clouds over at all.
    #[test]
    fn the_default_weather_config_leaves_both_clear_sky_and_cloud_in_the_world() {
        let samples = sample_densities();
        let cloudy = samples.iter().filter(|sky| sky.density > 0.5).count();
        let clear = samples.iter().filter(|sky| sky.density <= 0.0).count();
        let fraction = cloudy as f32 / samples.len() as f32;

        assert!(
            (0.05..0.7).contains(&fraction),
            "{:.1}% of the world is under cloud",
            fraction * 100.0
        );
        assert!(
            clear > samples.len() / 10,
            "the sky is never clear anywhere"
        );
    }

    /// Where the figures in `WeatherConfig`'s doc comments come from. Ignored
    /// because it is a measurement rather than an assertion:
    /// `cargo test --release -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn the_default_config_measures_the_sky() {
        let config = WeatherConfig::default();
        let samples = sample_densities();
        let total = samples.len() as f32;
        let mean = |values: Vec<f32>| values.iter().sum::<f32>() / total;
        let fraction = |count: usize| count as f32 / total * 100.0;

        let mean_humidity = mean(samples.iter().map(|sky| sky.probability).collect());
        let mean_field = mean(samples.iter().map(|sky| sky.field).collect());
        let any_cloud = fraction(samples.iter().filter(|sky| sky.density > 0.0).count());
        let under_cloud = fraction(samples.iter().filter(|sky| sky.density > 0.5).count());
        let any_rain = fraction(
            samples
                .iter()
                .filter(|sky| rain_amount(&config, sky.field, sky.density) > 0.0)
                .count(),
        );
        let real_rain = fraction(
            samples
                .iter()
                .filter(|sky| rain_amount(&config, sky.field, sky.density) > 0.25)
                .count(),
        );

        println!("{} samples, one per 16 tiles", samples.len());
        println!("mean humidity            {mean_humidity:.3}");
        println!("mean raw field           {mean_field:.3}");
        println!("any cloud at all         {any_cloud:.1}%");
        println!("under cloud (d > 0.5)    {under_cloud:.1}%");
        println!("any rain at all          {any_rain:.1}%");
        println!("rain worth seeing        {real_rain:.1}%");
    }

    /// The point of driving the sky from the humidity field: it has to rain over the
    /// wet country and not over the dry, or the map may as well not be sampled.
    #[test]
    fn a_dry_region_never_clouds_over_and_a_wet_one_mostly_does() {
        let samples = sample_densities();
        let mean_density = |lo: f32, hi: f32| {
            let band: Vec<f32> = samples
                .iter()
                .filter(|sky| sky.probability >= lo && sky.probability < hi)
                .map(|sky| sky.density)
                .collect();
            assert!(!band.is_empty(), "no samples with humidity in {lo}..{hi}");
            band.iter().sum::<f32>() / band.len() as f32
        };

        let dry = mean_density(0.0, 0.25);
        let wet = mean_density(0.75, 1.01);
        assert!(
            dry < 0.05,
            "the driest quarter of the world still clouds over ({dry})"
        );
        assert!(
            wet > 0.5,
            "the wettest quarter of the world barely clouds over ({wet})"
        );
    }

    /// Rain is the raw field's ramp *times* the density, so this holds for any
    /// config rather than only for ones whose cuts are in the right order — which is
    /// the whole reason for the multiply.
    #[test]
    fn rain_falls_only_inside_a_cloud() {
        // Deliberately perverse: a rain cut of zero, below the cloud cut, is how you
        // would ask for rain out of a clear sky.
        let config = WeatherConfig {
            rain_cut: 0.0,
            ..default()
        };

        for i in 0..=100 {
            let field = i as f32 / 100.0;
            let density = cloud_density(&config, field);
            if density <= 0.0 {
                assert_eq!(
                    rain_amount(&config, field, density),
                    0.0,
                    "rain with no cloud above it, at a field value of {field}"
                );
            }
        }
    }

    /// That a shadow *is* the cloud field one offset away is true by construction —
    /// the shader calls one function twice. What is not automatic is that the offset
    /// clears a cloud's own body: too short and every shadow hides under the cloud
    /// casting it, which looks like no shadows at all.
    #[test]
    fn a_shadow_falls_far_enough_from_its_cloud_to_be_seen() {
        let terrain = TerrainConfig::default();
        let config = WeatherConfig::default();
        let field = TilingNoiseField::new(
            terrain.seed,
            CLOUD_SHAPE_SALT,
            SHAPE_LATTICE_PERIOD,
            config.shape_octaves,
        );
        let sampler = terrain.sampler();
        let density_at = |tile: Vec2| {
            let cell = shape_cell(&config, tile);
            let raw = cloud_probability_at(&sampler, tile)
                * shape_at(&field, &config, cell, Vec2::ZERO, Vec2::ZERO);
            cloud_density(&config, raw)
        };

        let step = 16.0;
        let steps = (WORLD_TILES.x as f32 / step) as u32;
        let mut cloudy = 0;
        let mut in_the_open = 0;
        for y in 0..steps {
            for x in 0..steps {
                let tile = Vec2::new(x as f32, y as f32) * step;
                let cloud = density_at(tile);
                if cloud > 0.5 {
                    cloudy += 1;
                    // The tile this cloud shadows: is that tile itself in the clear?
                    if density_at(tile - config.shadow_offset_tiles) < 0.5 {
                        in_the_open += 1;
                    }
                }
            }
        }

        let visible = in_the_open as f32 / cloudy as f32;
        assert!(
            visible > 0.15,
            "only {:.1}% of shadows fall outside the cloud that casts them",
            visible * 100.0
        );
    }

    /// A long session must not quantize the sky. The offsets are wrapped, so the
    /// bound is on the wrap rather than on how long you play.
    #[test]
    fn a_long_session_does_not_quantize_the_weather_clock() {
        let terrain = TerrainConfig::default();
        let config = WeatherConfig::default();
        let mut sky = SkySampler::new(&terrain, &config);
        let mut clock = WeatherClock::default();
        let delta = 1.0 / 60.0;

        // Ten hours at 60 fps, through the same call the running game makes.
        for _ in 0..(60 * 60 * 60 * 10) {
            sky.drift(delta);
            clock.streak_phase = (clock.streak_phase + config.rain_streak_speed * delta).fract();
        }

        let (coarse, _) = sky.offsets();
        assert!(coarse.x.abs() < 1.0 && coarse.y.abs() < 1.0);
        assert!(clock.streak_phase.abs() < 1.0);
        // The step is still resolvable at the end of it, which is the thing an
        // unwrapped accumulator loses.
        sky.drift(delta);
        assert_ne!(sky.offsets().0, coarse);
    }
}
