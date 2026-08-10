//! What the weather leaves on the ground, and how it goes away again.
//!
//! Rain wets the ground and it dries; below freezing what falls lies as snow, and
//! the snow melts back into wetness. Neither writes a tile: `WorldMap` is untouched,
//! no chunk is ever marked dirty, and the alpine `Snow` *kind* keeps meaning what it
//! always did — a permanent property of the height band — with a transient snow line
//! moving around underneath it.
//!
//! **This is the first state the weather has ever had.** Everything in
//! [`crate::gameplay::weather`] is a pure function of `(place, clock)`; wetness and
//! snow are integrals of what has happened, so they have to be remembered. Three ways
//! were considered and the third is here:
//!
//! - **Per tile.** 16.7 M tiles x 2 bytes is 33 MB beside the heightmap's 16, and a
//!   step that touches every one of them. The resolution buys nothing — a rain patch
//!   is a cloud interior, tens of tiles across, and nothing varies inside one tile.
//! - **Stateless, as an upwind convolution.** Worth writing down because it almost
//!   works: the cloud field translates with the wind, so the rain history at a point
//!   is a line *upwind* of it and a dozen taps in the shader would give wet trails
//!   behind clouds with no state at all. It fails on two counts — the two shape
//!   layers drift at different speeds, so the field is only approximately a
//!   translation, and snow's time constant is a whole day, which is hundreds of taps.
//!   Still the right trick if wetness ever has to exist without a grid.
//! - **A coarse world grid**, which is what this is. One texel per
//!   `WORLD_TILES / cover_texels_per_side` tiles, 128 KB of state for the whole
//!   world. Bilinear on the way out, so the *envelope* is smooth; the per-tile look
//!   comes from the dither the pass thresholds against, not from the grid.
//!
//! **The grid does not know where the water is, and must not learn.** Snow
//! accumulating over a lake is harmless because the pass that draws it already
//! excludes everything at or below the water line — it had to, for the height ramp.
//! That is what keeps this module free of `WorldMap` by construction rather than by
//! discipline.
//!
//! The step reads the sky through the public [`SkySampler`] rather than through the
//! baked maps, so the crate still holds exactly **two** transcriptions of the sky's
//! arithmetic — Rust and wgsl — and not three. That costs real time, and the doc
//! comment on [`GroundConfig::cover_texels_per_side`] has the measurement.

use crate::gameplay::terrain::TerrainSampler;
use crate::gameplay::world::WorldSampler;
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
        screen::ScreenOverlay,
        sun::{PlanetConfig, Sun, warmth_at},
        terrain::TerrainConfig,
        weather::SkySampler,
        world::WORLD_TILES,
    },
    screens::Screen,
};

/// The window the climate map's first byte is quantized over. Wide enough that no
/// plausible `TerrainConfig` clips against either end, which is what lets the shader
/// read a temperature out of a byte without knowing anything about the terrain.
pub(super) const CLIMATE_MIN_CELSIUS: f32 = -40.0;
pub(super) const CLIMATE_MAX_CELSIUS: f32 = 60.0;

/// And the window the second byte — the day's swing — is quantized over. One-sided,
/// because an amplitude has no negative half.
pub(super) const CLIMATE_MAX_AMPLITUDE_CELSIUS: f32 = 30.0;

/// What the pass reads where no climate has been baked yet.
///
/// The one map whose absence cannot be *zero*: a zeroed climate byte decodes to
/// [`CLIMATE_MIN_CELSIUS`] and would put the entire world under falling snow for the
/// second the bake takes. Mild and unvarying instead, so an unbaked world rains —
/// which is what it did before this feature existed, and the same
/// absence-is-the-fallback the unbaked sky and the ungenerated heightmap have.
pub(super) const CLIMATE_FALLBACK_CELSIUS: f32 = 10.0;

/// Everything about the ground's transient state that is a knob rather than world
/// state.
///
/// Flat, like [`crate::gameplay::weather::WeatherConfig`] and for the same reason:
/// the shader's uniform is flat regardless, and the couplings between these values
/// read as neighbours here rather than across nested structs.
///
/// **What the defaults do to a day**, from
/// `the_default_config_measures_a_day_of_weather`:
///
/// ```text
///   hour   snowed   wet     snow line
///    4.5    8.3%    16.7%     0.592     the peak, just before dawn
///    9.2    6.4%    17.2%     0.627
///   13.2    4.1%    17.8%     0.725
///   17.2    3.4%    17.7%     0.769     the least, late afternoon
///   23.2    5.3%    17.4%     0.647
/// ```
///
/// So the snow line walks from elevation **0.59 to 0.77 and back** over one
/// 300-second day, and the snowed share of the land more than doubles across it.
/// That is the whole feature in two numbers, and it is what says the loop is visible
/// without a configured winter — the world is at an equinox throughout.
///
/// Two things about the shape are worth keeping. The peak lands at **04:30**, an hour
/// or so before dawn, which the thermal lag alone could not have produced: the lag is
/// a symmetric phase shift and puts the *temperature* trough at about 02:00. What
/// moves the snow peak later is that snow is an **integral** — it keeps accumulating
/// as long as it is below freezing and does not start going until the ground is above
/// it. And the wet share barely moves (16.6% to 18.0%), because wetness is fed by a
/// rain patch covering ~4% of the world at any moment and drained on a 90-second
/// constant: it is a steady state with weather passing through it, not a cycle.
#[derive(Resource, Clone)]
pub struct GroundConfig {
    /// Texels along each side of the state grid, covering the whole world. 256 over
    /// 4096 tiles is one texel per 16 tiles, 65k texels and 128 KB of state.
    ///
    /// **This is the cost knob, and the cost is the step rather than the memory.**
    /// Each texel evaluates the sky through [`SkySampler`], which is two tiling fbms
    /// of four octaves — sixteen lattice corners each, every one a hash and a
    /// `sin_cos`. One step measures **22.0 ms** at these defaults, against a
    /// `step_seconds` of 0.25: about 9% of one core, continuously, and thirteen times
    /// a 60 fps frame budget. That is why it runs on `AsyncComputeTaskPool` with one
    /// in flight rather than as an ordinary fixed-step system like `growth.rs`'s.
    ///
    /// If it ever needs to be cheaper, halving this to 128 is the first rung and
    /// quarters the bill: the envelope is smoothed by the dither on the way out
    /// anyway, so it costs less than it sounds like. The rungs after that are
    /// sampling the weather's *baked* texels instead of re-evaluating the field —
    /// which buys ~5x at the price of a third transcription of the sky — and then a
    /// compute shader, which is a product decision rather than a performance one: it
    /// does not exist under WebGL2, and the CPU could no longer read the field
    /// without an async readback.
    pub cover_texels_per_side: u32,
    /// How much game time one step advances. Small enough that a cloud does not cross
    /// a texel between steps: at the default wind a texel is 16 tiles and 8 seconds
    /// wide.
    ///
    /// It is a *minimum*, not a cadence — if the pool takes longer than this the next
    /// step simply integrates everything that has accumulated, because the model is
    /// an integral and integrating a longer interval is exactly right.
    pub step_seconds: f32,
    /// How fast rain saturates the ground, per unit of rain per second. At the
    /// default a patch of real rain (~0.5) soaks the ground under it in about 16 s.
    pub wetting_rate: f32,
    /// The e-folding time of drying at freezing point, in seconds. A cloud crosses a
    /// place in ~16 s, so this is what decides how long the wet trail behind it is.
    pub drying_seconds: f32,
    /// How much faster warm ground dries, per degree above freezing. This is the only
    /// reason a desert is dry a minute after the rain and a cold marsh stays dark.
    pub drying_warmth: f32,
    /// How fast snow builds, per unit of rain per second. Faster than the wetting
    /// rate because snow has to *read* as snow — a barely-white ground is a bug
    /// report rather than weather.
    pub snowfall_rate: f32,
    /// Degree-day melt: how much snow a degree above freezing removes in a second.
    ///
    /// The standard hydrological model, and it is one line. What it buys is that snow
    /// goes first where it is warmest — so the snow line ends up where it belongs and
    /// walks uphill through the morning without any code knowing what a snow line is.
    pub melt_per_degree_second: f32,
    /// How much of the melted snow arrives as wetness. Under 1 because some of it
    /// runs off, and above 0 because meltwater leaving no trace is the one thing that
    /// would make the whole cycle read as a texture swap.
    pub meltwater_gain: f32,
    /// Where water freezes, in degrees Celsius. Zero, obviously — it is a knob only
    /// so that the two places that compare against it read the same name.
    pub freezing_celsius: f32,
    /// Half the width of the ramp across it. A ramp rather than a switch, so sleet
    /// exists and no frame flips a whole region from rain to snow at once.
    pub freezing_softness_celsius: f32,
    /// How far the temperature swings between midnight and noon, before damping.
    pub diurnal_amplitude_celsius: f32,
    /// How much of that swing wet air takes away. **The amplitude is a field, not a
    /// scalar**, and this is what makes it one: dry air swings hard and wet air barely
    /// moves. A desert therefore freezes at night while a wetland does not, and it
    /// costs nothing — the humidity is already in the climate map, and nothing here
    /// learns what a desert is.
    pub diurnal_humidity_damping: f32,
    /// How far behind the sun the ground runs, in rotations. 0.08 is about two hours,
    /// which puts the warmest moment in the mid-afternoon rather than at noon.
    ///
    /// It is a phase shift, so it moves the cold end by the same two hours — the
    /// coldest moment lands in the small hours rather than just before dawn. Fixing
    /// *that* needs an asymmetric response, which is state; see [`warmth_at`].
    pub thermal_lag_rotations: f32,
    /// How much colder a solstice winter is than an equinox.
    ///
    /// **Exactly zero at the default `PlanetConfig::orbit_phase`**, because it is
    /// scaled by the declination over the axial tilt and an equinox declination is
    /// zero. Turning that knob is a winter; an orbit advancing it is seasons, and
    /// nothing in this module would change either way.
    pub seasonal_amplitude_celsius: f32,
    /// How much darker soaked ground is drawn than dry.
    pub wet_darkening: f32,
    /// And how much of its colour it loses. Small: wet earth goes darker and a little
    /// duller, where a full desaturation reads as fog rather than as rain.
    pub wet_desaturation: f32,
    /// How far a snowed tile is pulled toward white. Not all the way, because the
    /// relief factor is kept — a snowy slope still has to read as a slope.
    pub snow_lightening: f32,
    /// The width of the ramp the cover is thresholded against the dither over.
    ///
    /// Zero would make every tile snowed or bare with nothing between, which is the
    /// pixel-art look this is after — but it also makes the *edge* of a snowfield
    /// crawl one tile at a time with a hard step. A narrow ramp lets the boundary
    /// tiles come in half-covered first.
    pub snow_dither_softness: f32,
}

impl Default for GroundConfig {
    fn default() -> Self {
        Self {
            cover_texels_per_side: 256,
            step_seconds: 0.25,
            wetting_rate: 0.12,
            drying_seconds: 90.0,
            drying_warmth: 0.06,
            snowfall_rate: 0.20,
            melt_per_degree_second: 0.004,
            meltwater_gain: 0.6,
            freezing_celsius: 0.0,
            freezing_softness_celsius: 1.5,
            diurnal_amplitude_celsius: 9.0,
            diurnal_humidity_damping: 0.6,
            thermal_lag_rotations: 0.08,
            seasonal_amplitude_celsius: 8.0,
            wet_darkening: 0.22,
            wet_desaturation: 0.25,
            snow_lightening: 0.85,
            snow_dither_softness: 0.12,
        }
    }
}

/// What one texel of the state grid is carrying, both on 0..1.
#[derive(Clone, Copy, Default, Debug)]
pub struct GroundCell {
    pub wetness: f32,
    pub snow: f32,
}

/// What one texel of the baked climate says about the place under it.
///
/// The amplitude is a field rather than a scalar because dry air swings and wet air
/// does not — see [`GroundConfig::diurnal_humidity_damping`]. The humidity is kept
/// because the step needs it to ask the sky how hard it is raining, and re-deriving
/// it per texel per step would be a `TerrainSampler` lookup 65k times a step.
#[derive(Clone, Copy, Default, Debug)]
pub struct ClimateCell {
    pub normal_celsius: f32,
    pub diurnal_amplitude_celsius: f32,
    pub humidity: f32,
}

impl ClimateCell {
    /// What it is here right now: the normal, plus the day's swing scaled by this
    /// place's own amplitude, plus the season.
    pub fn temperature(&self, offset: TemperatureOffset) -> f32 {
        self.normal_celsius
            + self.diurnal_amplitude_celsius * offset.swing
            + offset.seasonal_celsius
    }
}

/// Everything about *when* it is, as far as the temperature is concerned. One value
/// for the whole world per step — the geometry that varies with place is already in
/// the climate map.
#[derive(Clone, Copy, Default, Debug)]
pub struct TemperatureOffset {
    /// The day's heating on -1..1, read as it stood `thermal_lag_rotations` ago.
    pub swing: f32,
    /// The season, in degrees, and exactly zero at an equinox.
    pub seasonal_celsius: f32,
}

/// Where the planet's turn has the temperature, this instant.
///
/// Public because the ctl reports it and the step consumes it, and both have to be
/// looking at the same number — an observation derived a second way would drift from
/// what the world actually did.
pub fn temperature_offset(
    config: &GroundConfig,
    planet: &PlanetConfig,
    sun: &Sun,
) -> TemperatureOffset {
    TemperatureOffset {
        swing: warmth_at(
            planet,
            (sun.rotation - config.thermal_lag_rotations).rem_euclid(1.0),
        ),
        // Scaled by the declination over the tilt, so it runs from -1 to 1 over a
        // year and is *identically* zero at an equinox. That is what lets an orbit
        // land later without touching this module.
        seasonal_celsius: config.seasonal_amplitude_celsius * sun.declination.sin()
            / planet
                .axial_tilt_degrees
                .to_radians()
                .sin()
                .max(f32::EPSILON),
    }
}

/// The climate bake in flight. Dropping the resource cancels it, so a map baked for
/// one world can never land in the next.
#[derive(Resource)]
struct ClimateBake(Task<Vec<ClimateCell>>);

/// The cover map's handle, and the only thing about the cover the render world gets.
///
/// **The maps themselves are not extracted, and that is deliberate.**
/// `ExtractResourcePlugin` deep-copies its source on every frame the source is
/// *touched* — and the step touches [`GroundCover`] every frame — so extracting the
/// cover itself would clone 65k cells across sixty times a second to deliver a handle
/// the pass could have had on its own. The cells stay where their readers are.
#[derive(Resource, Clone)]
pub struct GroundCoverTexture(pub(super) Handle<Image>);

impl ExtractResource for GroundCoverTexture {
    type Source = Self;

    fn extract_resource(source: &Self) -> Self {
        source.clone()
    }
}

/// The climate map's handle, on the same terms — though this one is written once, so
/// the saving is a single 786 KB copy rather than one a frame.
#[derive(Resource, Clone)]
pub struct ClimateTexture(pub(super) Handle<Image>);

impl ExtractResource for ClimateTexture {
    type Source = Self;

    fn extract_resource(source: &Self) -> Self {
        source.clone()
    }
}

/// The baked climate, once it has landed.
///
/// Absent until then, and that absence is the feature: with no climate the step does
/// nothing, so the first seconds of a session are dry ground rather than a stall.
#[derive(Resource)]
pub struct ClimateMaps {
    cells: Vec<ClimateCell>,
    side: u32,
}

impl ClimateMaps {
    /// The climate at a global tile position, from the texel it falls in.
    ///
    /// Read only by [`crate::control`], which does not exist on wasm — the same
    /// reason `WorldMap::generated` carries this attribute.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn at(&self, tile: Vec2) -> ClimateCell {
        self.cells[texel_index(self.side, tile)]
    }
}

/// The world's wetness and snow, and the step that advances them.
///
/// World state: built empty on entering gameplay and dropped on leaving, so a second
/// session never inherits the first one's snow.
#[derive(Resource)]
pub struct GroundCover {
    cells: Vec<GroundCell>,
    side: u32,
    /// The image the step rewrites. Only its *handle* reaches the render world, as a
    /// [`GroundCoverTexture`].
    texture: Handle<Image>,
    /// Game time waiting to be integrated. A step takes all of it rather than one
    /// `step_seconds` worth, so a slow step swallows its backlog instead of queueing
    /// one.
    carry_seconds: f32,
    step: Option<Task<Vec<GroundCell>>>,
}

impl GroundCover {
    /// How wet and how snowed a tile is.
    ///
    /// The seam anything outside this module reads the ground through, built on the
    /// terms `SkySampler` was: an answer, not the machinery. Its absence is dry
    /// ground, which is what lets the simulation run in a test with no ground plugin
    /// at all.
    ///
    /// Nearest texel rather than bilinear, deliberately: the overlay filters because
    /// it is drawing a smooth envelope, and a reader asking "is there snow here"
    /// wants the cell's own answer rather than one blended with its neighbours'.
    ///
    /// The seam has no reader on wasm yet, because the only one so far is
    /// [`crate::control`] and that does not exist there. The attribute is the same one
    /// `WorldMap::generated` carries, and it goes when the simulation reads this.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn at(&self, tile: Vec2) -> GroundCell {
        self.cells[texel_index(self.side, tile)]
    }

    /// The share of the world's texels carrying more than `threshold` of each, as
    /// `(snow, wetness)`. What the ctl reports, and how the measurement below tracks
    /// a day — so, like [`Self::at`], it has no reader on wasm.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn fractions_over(&self, threshold: f32) -> (f32, f32) {
        let total = self.cells.len() as f32;
        let snow = self.cells.iter().filter(|c| c.snow > threshold).count();
        let wet = self.cells.iter().filter(|c| c.wetness > threshold).count();
        (snow as f32 / total, wet as f32 / total)
    }
}

pub struct GroundPlugin;

impl Plugin for GroundPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GroundConfig>();
        app.add_plugins((
            ExtractResourcePlugin::<GroundCoverTexture>::default(),
            ExtractResourcePlugin::<ClimateTexture>::default(),
        ));
        app.add_systems(OnEnter(Screen::Gameplay), start_ground);
        app.add_systems(
            Update,
            start_climate_bake.run_if(resource_added::<WorldSampler>),
        );
        app.add_systems(OnExit(Screen::Gameplay), end_ground);
        // No `OnEnter` sync, unlike the tint's ramp: this one needs the `Sun` that
        // another plugin inserts in the same schedule, and there is no ordering
        // between them to hang it off. It costs nothing — `Update` runs before the
        // frame's extract, so the pass never sees the uniform's zeroed ground half.
        app.add_systems(
            Update,
            (
                finish_climate_bake,
                // The step is the only writer of the cover and the sync is what tells
                // the pass which day it is, so this pair is the whole ordering the
                // main world needs.
                (step_ground_cover, sync_ground_overlay).chain(),
            )
                .run_if(in_state(Screen::Gameplay)),
        );
    }
}

/// Opens the session dry and puts the climate bake on the pool.
///
/// Dry rather than "unknown": a world whose bake has not landed shows bare ground,
/// which is the same absence-is-the-fallback the unbaked sky gives the clouds.
fn start_ground(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    _terrain: Res<TerrainConfig>,
    config: Res<GroundConfig>,
) {
    let side = config.cover_texels_per_side.max(1);
    let cells = vec![GroundCell::default(); (side * side) as usize];
    let texture = images.add(cover_image(side, &cells));

    commands.insert_resource(GroundCoverTexture(texture.clone()));
    commands.insert_resource(GroundCover {
        texture,
        cells,
        side,
        carry_seconds: 0.0,
        step: None,
    });
}

/// Starts the climate bake, the frame the terrain's own bake finishes.
///
/// The same wait `weather.rs` makes and for the same reason: a climate normal is the
/// sampler's temperature over the whole world, and the sampler reads a baked document.
/// Until it lands the cover map is the blank one `start_ground` already inserted, which
/// is dry ground and no snow — the fallback this module documents.
fn start_climate_bake(
    mut commands: Commands,
    sampler: Res<WorldSampler>,
    config: Res<GroundConfig>,
) {
    let sampler = sampler.0.clone();
    let config = config.clone();
    let task = AsyncComputeTaskPool::get().spawn(async move { bake_climate(&sampler, &config) });
    commands.insert_resource(ClimateBake(task));
}

/// Drops the session's state. A bake or a step still in flight goes with it.
fn end_ground(mut commands: Commands) {
    commands.remove_resource::<GroundCover>();
    commands.remove_resource::<GroundCoverTexture>();
    commands.remove_resource::<ClimateMaps>();
    commands.remove_resource::<ClimateTexture>();
    commands.remove_resource::<ClimateBake>();
}

fn finish_climate_bake(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    config: Res<GroundConfig>,
    bake: Option<ResMut<ClimateBake>>,
) {
    let Some(mut bake) = bake else {
        return;
    };
    let Some(cells) = block_on(poll_once(&mut bake.0)) else {
        return;
    };

    let side = config.cover_texels_per_side.max(1);
    commands.insert_resource(ClimateTexture(images.add(climate_image(side, &cells))));
    commands.insert_resource(ClimateMaps { cells, side });
    commands.remove_resource::<ClimateBake>();
}

/// Advances the whole world's wetness and snow, one pooled step at a time.
fn step_ground_cover(
    time: Res<Time>,
    config: Res<GroundConfig>,
    planet: Res<PlanetConfig>,
    sun: Res<Sun>,
    sky: Option<Res<SkySampler>>,
    climate: Option<Res<ClimateMaps>>,
    mut cover: ResMut<GroundCover>,
    mut images: ResMut<Assets<Image>>,
) {
    // A step that has landed is taken before a new one is started, so the frame a
    // task finishes is also the frame its successor can leave.
    if let Some(task) = cover.step.as_mut()
        && let Some(cells) = block_on(poll_once(task))
    {
        cover.step = None;
        if let Some(mut image) = images.get_mut(&cover.texture) {
            write_cover(&mut image, &cells);
        }
        cover.cells = cells;
    }

    cover.carry_seconds += time.delta_secs();

    // Until the sky and the climate are both up there is nothing to integrate — and
    // the carry is *not* banked while waiting, because a step that swallowed the
    // whole bake would rain a session's worth of weather onto the first frame.
    let (Some(sky), Some(climate)) = (sky, climate) else {
        cover.carry_seconds = 0.0;
        return;
    };
    if cover.step.is_some() || cover.carry_seconds < config.step_seconds {
        return;
    }

    let dt = std::mem::take(&mut cover.carry_seconds);
    let offset = temperature_offset(&config, &planet, &sun);
    let tiles_per_texel = WORLD_TILES.x as f32 / cover.side as f32;

    // Everything the step needs is copied in, so the task owns it outright and
    // dropping the resource is the whole of cancelling it.
    let config = config.clone();
    let climate = climate.cells.clone();
    let cells = cover.cells.clone();
    let side = cover.side;
    let sky = sky.clone();
    cover.step = Some(AsyncComputeTaskPool::get().spawn(async move {
        let mut next = cells;
        for (index, cell) in next.iter_mut().enumerate() {
            let climate = climate[index];
            let tile = texel_centre(side, index, tiles_per_texel);
            let rain = sky.rain_at(tile, climate.humidity);
            *cell = advance(&config, *cell, rain, climate.temperature(offset), dt);
        }
        next
    }));
}

/// One texel, one step. The whole model, and it is deliberately readable in one
/// screen: everything else in this module exists to call it 65k times off the main
/// thread.
///
/// **It is stable for any `dt`, including a frame-long one and a pathological one.**
/// The two gains are clamped at the end, the melt is bounded by the snow there is to
/// melt, and the drying is an exponential of something that cannot be positive — so
/// no step length can drive either quantity outside 0..1.
pub(crate) fn advance(
    config: &GroundConfig,
    cell: GroundCell,
    rain: f32,
    temperature: f32,
    dt: f32,
) -> GroundCell {
    let above_freezing = (temperature - config.freezing_celsius).max(0.0);
    // Ramped rather than switched: sleet exists, and no frame flips a whole region
    // from rain to snow. Written as one minus the rising step rather than as a
    // falling one, because wgsl's `smoothstep` is undefined with its edges reversed
    // and `screen.wgsl` has to say this the same way.
    let snowing = 1.0
        - smoothstep(
            config.freezing_celsius - config.freezing_softness_celsius,
            config.freezing_celsius + config.freezing_softness_celsius,
            temperature,
        );

    let mut snow = cell.snow + rain * config.snowfall_rate * snowing * dt;
    let mut wetness = cell.wetness + rain * config.wetting_rate * (1.0 - snowing) * dt;

    // Degree-day melt, bounded by what is there. Meltwater becomes wetness rather
    // than vanishing, which is what stops the thaw reading as a texture swap.
    let melted = snow.min(above_freezing * config.melt_per_degree_second * dt);
    snow -= melted;
    wetness += melted * config.meltwater_gain;

    // Warm ground dries faster, and the exponential is what makes this unconditional
    // rather than a subtraction that could go negative.
    wetness *= (-dt / config.drying_seconds.max(f32::EPSILON)
        * (1.0 + config.drying_warmth * above_freezing))
        .exp();

    GroundCell {
        wetness: wetness.clamp(0.0, 1.0),
        snow: snow.clamp(0.0, 1.0),
    }
}

/// Hands the pass this step's knobs and this frame's temperature offset.
///
/// The offset goes over as two scalars because the shader needs to know what is
/// *falling* at a fragment, which means combining them with the climate map there —
/// the same arithmetic [`ClimateCell::temperature`] does on the CPU, and the reason
/// what falls always agrees with what lies.
fn sync_ground_overlay(
    config: Res<GroundConfig>,
    planet: Res<PlanetConfig>,
    sun: Res<Sun>,
    mut overlay: Single<&mut ScreenOverlay>,
) {
    overlay.set_ground(&config, temperature_offset(&config, &planet, &sun));
}

// -- The bake ----------------------------------------------------------------

/// The climate at every texel: what it is normally, how far it swings in a day, and
/// how wet it is.
///
/// On the compute pool because it is two `TerrainSampler` lookups per texel and a
/// lookup is a biome blend — the expensive half of the terrain. It lands beside the
/// weather's own bake, which is already doing the same thing for the same reason.
fn bake_climate(sampler: &TerrainSampler, config: &GroundConfig) -> Vec<ClimateCell> {
    let side = config.cover_texels_per_side.max(1);
    let tiles_per_texel = WORLD_TILES.x as f32 / side as f32;

    (0..(side * side) as usize)
        .map(|index| {
            let tile = texel_centre(side, index, tiles_per_texel);
            let humidity = sampler.humidity(tile.x, tile.y);
            ClimateCell {
                normal_celsius: sampler.temperature(tile.x, tile.y),
                // Damped by the humidity, which is what freezes a desert at night
                // and keeps a marsh mild — with nothing here learning what either is.
                diurnal_amplitude_celsius: config.diurnal_amplitude_celsius
                    * (1.0 - config.diurnal_humidity_damping * humidity).max(0.0),
                humidity,
            }
        })
        .collect()
}

// -- Grids and images --------------------------------------------------------

/// The centre of a texel, in global tile space — the point the map claims to sample,
/// rather than its corner.
fn texel_centre(side: u32, index: usize, tiles_per_texel: f32) -> Vec2 {
    let index = index as u32;
    (Vec2::new((index % side) as f32, (index / side) as f32) + Vec2::splat(0.5)) * tiles_per_texel
}

/// Which texel a global tile position falls in, clamped to the grid so a reader at
/// the world's edge gets the edge rather than a panic.
///
/// Only the two read seams above want this, and neither has a caller on wasm.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn texel_index(side: u32, tile: Vec2) -> usize {
    let scaled = tile / WORLD_TILES.as_vec2() * side as f32;
    let x = (scaled.x as i32).clamp(0, side as i32 - 1) as usize;
    let y = (scaled.y as i32).clamp(0, side as i32 - 1) as usize;
    y * side as usize + x
}

fn to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn cover_texels(cells: &[GroundCell]) -> Vec<u8> {
    cells
        .iter()
        .flat_map(|cell| [to_byte(cell.wetness), to_byte(cell.snow)])
        .collect()
}

fn cover_image(side: u32, cells: &[GroundCell]) -> Image {
    // `default()` rather than `RENDER_WORLD`, because unlike the weather's maps this
    // one is *rewritten* every step and so has to keep its main-world copy.
    map_image(
        side,
        cover_texels(cells),
        TextureFormat::Rg8Unorm,
        RenderAssetUsages::default(),
        ImageFilterMode::Linear,
    )
}

fn write_cover(image: &mut Image, cells: &[GroundCell]) {
    // 128 KB a step at the defaults, which is a fifth of what a single landed chunk
    // costs the heightmap — so unlike that one this can be an ordinary `Image` and
    // let bevy re-upload the whole thing.
    image.data = Some(cover_texels(cells));
}

fn climate_image(side: u32, cells: &[ClimateCell]) -> Image {
    let texels = cells
        .iter()
        .flat_map(|cell| {
            [
                to_byte(
                    (cell.normal_celsius - CLIMATE_MIN_CELSIUS)
                        / (CLIMATE_MAX_CELSIUS - CLIMATE_MIN_CELSIUS),
                ),
                to_byte(cell.diurnal_amplitude_celsius / CLIMATE_MAX_AMPLITUDE_CELSIUS),
            ]
        })
        .collect();
    map_image(
        side,
        texels,
        TextureFormat::Rg8Unorm,
        RenderAssetUsages::RENDER_WORLD,
        ImageFilterMode::Linear,
    )
}

/// Sampled smoothly and clamped at the world's edge, so a view of the border reads
/// the border rather than wrapping the far side of the world into shot.
///
/// The filtering is set here rather than inherited: the app-wide default is *nearest*
/// for the 8px pixel art, and a nearest-sampled cover map would draw the snow in
/// visible 16-tile blocks.
pub(super) fn map_image(
    side: u32,
    texels: Vec<u8>,
    format: TextureFormat,
    usages: RenderAssetUsages,
    filter: ImageFilterMode,
) -> Image {
    let mut image = Image::new(
        Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        texels,
        format,
        usages,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        min_filter: filter,
        mag_filter: filter,
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        ..default()
    });
    image
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::gameplay::terrain::shared_test_sampler;

    use crate::gameplay::{terrain::height_byte, weather::WeatherConfig};

    /// A cell in a place with a fixed temperature, stepped by a fixed amount of rain
    /// — the whole model is `advance`, so every property below is an ordinary unit
    /// test with no app, no pool and no GPU.
    fn step(cell: GroundCell, rain: f32, temperature: f32, dt: f32) -> GroundCell {
        advance(&GroundConfig::default(), cell, rain, temperature, dt)
    }

    /// The one thing this whole feature is: what falls is decided by how cold it is,
    /// and nothing else in the module decides it a second time.
    #[test]
    fn precipitation_lies_as_snow_below_freezing_and_wets_the_ground_above() {
        let dry = GroundCell::default();

        let frozen = step(dry, 0.6, -8.0, 4.0);
        assert!(frozen.snow > 0.0, "it did not snow below freezing");
        assert_eq!(
            frozen.wetness, 0.0,
            "rain fell as well as snow at -8 C: {frozen:?}"
        );

        let thawed = step(dry, 0.6, 12.0, 4.0);
        assert!(thawed.wetness > 0.0, "it did not wet the ground above zero");
        assert_eq!(thawed.snow, 0.0, "snow lay at 12 C: {thawed:?}");

        // And in between, both — which is what a ramp across the freezing point buys
        // and a switch would not: no frame flips a whole region at once.
        let sleet = step(dry, 0.6, 0.0, 4.0);
        assert!(
            sleet.snow > 0.0 && sleet.wetness > 0.0,
            "at the freezing point exactly it should be doing both: {sleet:?}"
        );
    }

    /// The other half of "transient": a shower has to leave, or the world ends up
    /// permanently dark under weather that passed an hour ago.
    #[test]
    fn a_step_with_no_rain_dries_the_ground() {
        let soaked = GroundCell {
            wetness: 1.0,
            snow: 0.0,
        };
        let mut cell = soaked;
        for _ in 0..40 {
            cell = step(cell, 0.0, 15.0, 4.0);
        }
        assert!(
            cell.wetness < 0.1,
            "the ground was still {:.2} wet after 160 dry seconds",
            cell.wetness
        );

        // And the drying is monotone in the warmth, which is the only reason a desert
        // and a marsh look different a minute after the same shower.
        let cold = step(soaked, 0.0, 0.0, 20.0);
        let warm = step(soaked, 0.0, 25.0, 20.0);
        assert!(
            warm.wetness < cold.wetness,
            "warm ground ({:.3}) should dry faster than cold ({:.3})",
            warm.wetness,
            cold.wetness
        );
    }

    /// Degree-day melt is the whole reason a snow line exists and moves: snow goes
    /// first where it is warmest, so the line ends up where it belongs without
    /// anything in the module knowing what a snow line is.
    #[test]
    fn snow_melts_faster_the_warmer_it_is() {
        let laid = GroundCell {
            wetness: 0.0,
            snow: 1.0,
        };

        let mut previous = laid.snow;
        for temperature in [1.0, 3.0, 6.0, 12.0, 25.0] {
            let melted = step(laid, 0.0, temperature, 10.0).snow;
            assert!(
                melted < previous,
                "snow at {temperature} C ({melted:.3}) did not go faster than the step below it \
                 ({previous:.3})"
            );
            previous = melted;
        }

        // Below freezing it does not melt at all — the degree-day term is clamped at
        // zero, so a cold night cannot take snow away.
        assert_eq!(step(laid, 0.0, -5.0, 100.0).snow, 1.0);
    }

    /// Meltwater that vanished would make the thaw a texture swap. It has to arrive
    /// as wetness, and then dry like any other water.
    #[test]
    fn melting_snow_leaves_the_ground_wet() {
        let laid = GroundCell {
            wetness: 0.0,
            snow: 0.5,
        };
        let thawing = step(laid, 0.0, 8.0, 4.0);

        assert!(thawing.snow < laid.snow, "nothing melted");
        assert!(
            thawing.wetness > 0.0,
            "the melt left the ground bone dry: {thawing:?}"
        );
        // And it is the melt that did it, not rain: there was none.
        let frozen = step(laid, 0.0, -8.0, 4.0);
        assert_eq!(frozen.wetness, 0.0);
    }

    /// The stability guard, and the reason `advance` is written the way it is: the
    /// step runs on a pool at whatever interval the frame rate leaves it, so `dt` is
    /// not a number this module gets to choose.
    #[test]
    fn cover_stays_between_none_and_full_under_any_step() {
        let config = GroundConfig::default();
        let mut cell = GroundCell::default();

        // A deterministic walk over the perverse end of every input, including a `dt`
        // a thousand times the nominal step and a temperature no planet has.
        for i in 0..20_000u32 {
            let spin = |salt: u32| (i.wrapping_mul(salt) % 1000) as f32 / 1000.0;
            let rain = spin(2_654_435_761);
            let temperature = spin(40_503) * 160.0 - 80.0;
            let dt = spin(2_246_822_519).powi(3) * 250.0;
            cell = advance(&config, cell, rain, temperature, dt);
            assert!(
                (0.0..=1.0).contains(&cell.wetness) && (0.0..=1.0).contains(&cell.snow),
                "step {i} with rain {rain}, {temperature} C over {dt} s left {cell:?}"
            );
        }
    }

    /// The issue's own acceptance criterion, as arithmetic: cover is *driven* by the
    /// weather, so a region the clouds never reach can never acquire any.
    #[test]
    fn the_ground_never_gains_snow_where_it_never_rains() {
        let mut cell = GroundCell {
            wetness: 0.4,
            snow: 0.0,
        };
        for _ in 0..500 {
            cell = step(cell, 0.0, -30.0, 1.0);
            assert_eq!(cell.snow, 0.0, "snow appeared out of a clear freezing sky");
        }
        // The wetness it started with is still there, because nothing above freezing
        // ever happened to dry it — the exponential still runs, but slowly.
        assert!(cell.wetness < 0.4);
    }

    /// The seasonal term is exactly zero at the shipped orbit phase, which is the
    /// claim that lets an orbit land later without touching this module at all.
    #[test]
    fn the_season_is_exactly_nothing_at_an_equinox() {
        let config = GroundConfig::default();
        let planet = PlanetConfig::default();
        assert_eq!(
            planet.orbit_phase, 0.0,
            "the shipped world sits at an equinox"
        );

        let offset = temperature_offset(&config, &planet, &Sun::default());
        assert_eq!(offset.seasonal_celsius, 0.0);
        assert!(
            config.seasonal_amplitude_celsius > 0.0,
            "and it is zero because the declination is, not because the knob is"
        );
    }

    /// A winter is one moving knob, and it has to actually move: the declination over
    /// the tilt runs to 1 at a solstice, so the amplitude means what it says.
    #[test]
    fn a_solstice_winter_is_the_configured_amount_colder() {
        let config = GroundConfig::default();
        let winter = PlanetConfig {
            orbit_phase: 0.75,
            ..default()
        };
        // The declination at a southward solstice is minus the tilt.
        let sun = Sun {
            declination: -winter.axial_tilt_degrees.to_radians(),
            ..default()
        };

        let seasonal = temperature_offset(&config, &winter, &sun).seasonal_celsius;
        assert!(
            (seasonal + config.seasonal_amplitude_celsius).abs() < 1.0e-4,
            "a solstice winter came out {seasonal} rather than \
             {}",
            -config.seasonal_amplitude_celsius
        );
    }

    /// What a day of the shipped weather actually does to the ground, hour by hour.
    ///
    /// Where the numbers in `GroundConfig`'s doc comments come from, and the only
    /// thing that says whether the defaults put the whole loop inside one 300-second
    /// day rather than only inside a configured winter.
    ///
    /// `cargo test --release -- --ignored --nocapture`, and run it alone: it bakes a
    /// climate map and then steps the whole world a few hundred times.
    #[test]
    #[ignore = "measurement, not a check"]
    fn the_default_config_measures_a_day_of_weather() {
        let terrain = TerrainConfig::default();
        let weather = WeatherConfig::default();
        let planet = PlanetConfig::default();
        let config = GroundConfig::default();

        let side = config.cover_texels_per_side;
        let tiles_per_texel = WORLD_TILES.x as f32 / side as f32;
        let climate = bake_climate(shared_test_sampler(), &config);
        let mut sky = SkySampler::new(&terrain, &weather);
        let mut cells = vec![GroundCell::default(); climate.len()];

        // The elevation under each texel, quantized exactly as the heightmap stores
        // it, so "the snow line is at 0.7" means the same thing here and on screen.
        // Also which texels are land: the pass never draws cover over water, so
        // counting it there would flatter every figure below.
        let sampler = terrain.sampler();
        let heights: Vec<f32> = (0..cells.len())
            .map(|index| {
                let tile = texel_centre(side, index, tiles_per_texel);
                height_byte(sampler.elevation(tile.x, tile.y)) as f32 / 255.0
            })
            .collect();
        let land: Vec<usize> = (0..cells.len())
            .filter(|index| heights[*index] > terrain.shallow_water_max)
            .collect();

        let dt = config.step_seconds;
        let steps = (planet.rotation_period_seconds / dt).round() as u32;
        // Two turns, and only the second is reported: the first is the world filling
        // in from bone dry, which is a transient of the measurement rather than of
        // the weather.
        let mut rotation = planet.start_rotation;
        let mut cost = std::time::Duration::ZERO;
        let mut readings = Vec::new();

        for turn in 0..2 {
            for _ in 0..steps {
                let sun = Sun {
                    rotation,
                    ..default()
                };
                let offset = temperature_offset(&config, &planet, &sun);

                let started = std::time::Instant::now();
                for (index, cell) in cells.iter_mut().enumerate() {
                    let climate = climate[index];
                    let tile = texel_centre(side, index, tiles_per_texel);
                    let rain = sky.rain_at(tile, climate.humidity);
                    *cell = advance(&config, *cell, rain, climate.temperature(offset), dt);
                }
                cost += started.elapsed();

                sky.drift(dt);
                rotation = (rotation + dt / planet.rotation_period_seconds).fract();

                if turn == 1 {
                    let snowed: Vec<usize> = land
                        .iter()
                        .copied()
                        .filter(|index| cells[*index].snow > 0.1)
                        .collect();
                    let wet = land.iter().filter(|i| cells[**i].wetness > 0.1).count();
                    // The snow line: the lowest ground still holding any. A p05
                    // rather than the minimum, because one freak texel is not a line.
                    let mut line: Vec<f32> = snowed.iter().map(|i| heights[*i]).collect();
                    line.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    readings.push((
                        rotation,
                        snowed.len() as f32 / land.len() as f32,
                        wet as f32 / land.len() as f32,
                        line.get(line.len() / 20).copied(),
                    ));
                }
            }
        }

        println!(
            "\n{} texels, {} of them land; one step is {:.1} ms",
            cells.len(),
            land.len(),
            cost.as_secs_f64() * 1000.0 / (steps as f64 * 2.0),
        );
        println!("\n  hour   snowed   wet    snow line");
        for (rotation, snowed, wet, line) in readings.iter().step_by(steps as usize / 24) {
            match line {
                Some(line) => println!(
                    "  {:>4.1}   {:>5.2}%   {:>5.2}%   {line:.3}",
                    rotation * 24.0,
                    snowed * 100.0,
                    wet * 100.0,
                ),
                None => println!(
                    "  {:>4.1}   {:>5.2}%   {:>5.2}%   none lying",
                    rotation * 24.0,
                    snowed * 100.0,
                    wet * 100.0,
                ),
            }
        }

        let peak = readings
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        let trough = readings
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        println!(
            "\npeak snow {:.2}% of land at hour {:.1}; least {:.2}% at hour {:.1}",
            peak.1 * 100.0,
            peak.0 * 24.0,
            trough.1 * 100.0,
            trough.0 * 24.0,
        );
        println!(
            "mean wet {:.2}% of land",
            readings.iter().map(|r| r.2).sum::<f32>() / readings.len() as f32 * 100.0,
        );

        // How *deep* the cover gets, not only how far it spreads. The pass scales its
        // darkening and its lightening by these numbers directly, so a world where
        // every wet texel sits at 0.15 draws a 3% darkening however the knob is set —
        // which is the difference between an effect and a rumour of one.
        let percentiles = |mut values: Vec<f32>| {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let at = |q: f32| values[((values.len() as f32 - 1.0) * q) as usize];
            (at(0.5), at(0.9), at(0.99), at(1.0))
        };
        let (w50, w90, w99, wmax) = percentiles(land.iter().map(|i| cells[*i].wetness).collect());
        let (s50, s90, s99, smax) = percentiles(land.iter().map(|i| cells[*i].snow).collect());
        println!("wetness over land   p50 {w50:.3}  p90 {w90:.3}  p99 {w99:.3}  max {wmax:.3}");
        println!("snow    over land   p50 {s50:.3}  p90 {s90:.3}  p99 {s99:.3}  max {smax:.3}\n");
    }
}
