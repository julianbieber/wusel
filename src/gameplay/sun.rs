//! The planet's rotation, where that puts the sun, and what the sun delivers.
//!
//! **There is no day/night timer here.** The one piece of state is how far the
//! planet has turned; everything else — the altitude, the bearing a shadow falls
//! along, the colour and the strength of the light — is geometry read off it. That
//! is deliberate, and it is what pays for the module:
//!
//! * **Day length is not a knob.** It falls out of latitude and declination, so a
//!   high-latitude summer day is long because the geometry says so.
//! * **Night is not a state.** It is the altitude below the horizon. No branch in
//!   here names night, and dawn and dusk are not events but a crossing.
//! * **Seasons are one moving knob away.** [`PlanetConfig::orbit_phase`] is fixed
//!   for now; an orbit would animate it and nothing else here would change.
//! * **There is no shadow-strength knob.** A shadowed tile is lit by
//!   [`Insolation::sky`] alone and a lit one by sky plus [`Insolation::direct`], so
//!   "how dark is a shadow" is answered by how much of the light is beam — which is
//!   itself geometry.
//! * **The colour is not keyframed.** A low sun is warm because its beam crosses
//!   more air, and [`PlanetConfig::zenith_extinction`] is per channel. No setting of
//!   these knobs can make noon redder than dusk.
//!
//! Like the weather this is cosmetic *so far*: nothing here reads or writes
//! `WorldMap`. [`Sun`] is the seam anything else reads it through — an answer, not
//! the machinery — and its absence is full daylight, the way the absence of
//! `SkySampler` is a clear sky.
//!
//! The lighting itself is drawn by [`crate::gameplay::tint`], because the shadow
//! test reads the heightmap and that is the only pass which binds it. This module
//! decides; that one draws.
//!
//! **None of the drawing half is visible to a unit test**, so it was measured off
//! real frames instead — captures of one fixed scene at a series of rotations, the
//! same trick gh-13 used for the ramp:
//!
//! ```text
//!   hour   mean sRGB luma   red/blue
//!   14.8        134.7         1.35
//!   16.7        107.7         1.38
//!   17.8         79.2         1.37
//!   18.1         68.9         1.34   just past sunset
//!   18.9         65.3         1.15   night
//!    0.4         65.3         1.15
//! ```
//!
//! Monotone down, warm through the afternoon and cool at night, and flat once the
//! sun is properly down — which is the model working, not a stuck frame: with no
//! beam there is nothing left to change. Night lands at 48% of the afternoon on
//! screen where the model says 0.19, because the pass multiplies in linear space and
//! the frame is sRGB.
//!
//! The shadow was isolated the same way, against a build with `relief_tiles` set so
//! low nothing can occlude: **13.9% of pixels darker, 0.0% lighter**, the deepest at
//! 0.72 of unshadowed. Zero lighter is the part worth keeping — a shadow may only
//! ever take the beam away.

use std::f32::consts::TAU;

use bevy::prelude::*;

use crate::{
    gameplay::{tint::TerrainTintOverlay, weather::WeatherOverlay},
    screens::Screen,
};

/// How a colour is weighed into one number. Used for the level the clouds are lit
/// by, and to normalise the sky's hue so that "how blue" and "how bright" are two
/// separate knobs rather than one tangled one.
const LUMINANCE: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);

/// The planet, its star and its air.
///
/// A knob rather than world state, so like `TerrainConfig` and `WeatherConfig` it is
/// built once and outlives every session — rerolling a world does not move it to a
/// different latitude.
///
/// Every field is a property of a place, an orbit or an atmosphere. There is
/// deliberately no `day_seconds`, no `night_level` and no dusk colour: each of those
/// is a consequence of the ones that are here.
#[derive(Resource, Clone, Debug)]
pub struct PlanetConfig {
    /// Real seconds for one full turn of the planet — the only place real time
    /// enters. Everything else is measured in rotations.
    ///
    /// 300 s puts a whole cycle inside an ordinary session: a shadow visibly creeps
    /// across a ridge without strobing, and one in-world hour is 12.5 s.
    pub rotation_period_seconds: f32,
    /// The rotation a session opens at, on 0..1 from local midnight. 0.30 is about
    /// an hour and a quarter after sunrise at the default latitude — a low sun, so
    /// the first thing on screen is long shadows rather than flat noon.
    pub start_rotation: f32,
    /// Where the world sits on the planet.
    ///
    /// **35 rather than 0 is what makes the bearing worth being a vector.** At the
    /// equator the sun would pass straight overhead, shadows would vanish at noon
    /// and the bearing would flip sign in a single frame. At 35 the sun rises due
    /// east, swings toward the equator through midday and sets due west, so a
    /// shadow rotates about a quarter turn over a day and noon at 55 degrees still
    /// casts one.
    pub latitude_degrees: f32,
    /// The planet's tilt, which is the whole of what a season is. Only read through
    /// [`declination`].
    pub axial_tilt_degrees: f32,
    /// Where the orbit has got to, on 0..1 from the northward equinox. Fixed at an
    /// equinox for now, which puts the declination at 0; seasons are this knob
    /// moving and nothing else in this module changing.
    pub orbit_phase: f32,
    /// Optical depth of the air straight overhead, per channel. Blue is scattered
    /// hardest, which is why a low sun goes warm on its own.
    ///
    /// Exaggerated over the real Rayleigh figures (~0.05/0.10/0.23) because a game
    /// is watched for minutes rather than hours and the sunset has to read.
    pub zenith_extinction: Vec3,
    /// The total light at the reference altitude — equinox noon for this latitude.
    /// 1.0 means the world is drawn exactly as the art was painted, which is what
    /// makes every other hour a departure from it rather than from nothing.
    pub daylight: f32,
    /// How much of that reference light is skylight rather than beam. It is
    /// therefore *how bright a shadow is*: at 0.38 a tile the beam misses keeps a
    /// little over a third of the light, which is about what a clear day does once
    /// the ground's own bounce is counted.
    pub sky_fraction: f32,
    /// What is left of the daytime sky when the sun is on the horizon.
    ///
    /// The sky is lit by the sun, so it has to dim with it — a sky held at
    /// `sky_fraction` all the way down swamps the beam, and dusk comes out neutral
    /// grey instead of warm. It does not dim *to nothing*, though, because at sunset
    /// the air overhead is still in full daylight; that is what this is. Between the
    /// two the level follows the square root of the sun's height, which is roughly
    /// how diffuse light behaves and is much flatter than the beam's own falloff.
    pub horizon_glow: f32,
    /// How far the sky's colour is allowed toward the hue the extinction implies.
    /// The derived hue is very blue — it is what a Rayleigh atmosphere scatters —
    /// and pulling it partway to white is the difference between a blue shadow and
    /// a blue world. Normalising to luminance first is what keeps this from also
    /// changing how bright the shadow is.
    pub sky_saturation: f32,
    /// How much of the cosine law a landscape actually obeys.
    ///
    /// A beam striking *flat ground* at an angle spreads over more of it, which is
    /// the full `sin(altitude)` — and a world of nothing but flat ground goes almost
    /// dark before the sun is anywhere near setting. What is drawn here is a
    /// landscape of slopes and faces, which catch a low sun far better than a plane
    /// does, so the law is softened by this exponent. 1.0 is the plane; at 0.55 a
    /// sun 12 degrees up delivers 47% of the reference beam rather than 26%.
    ///
    /// It is 1 at the reference altitude whatever this is set to, so the white
    /// balance does not move with it.
    pub cosine_response: f32,
    /// What is left when the sun is well down: moonlight and starlight, and the one
    /// place a colour is stated rather than derived, because neither is the sun.
    /// Dim and cool, but never zero — nothing can light the world once the beam is
    /// gone, so this is what keeps night readable.
    ///
    /// It cannot exceed the dusk it fades in under, or the world would visibly
    /// *brighten* after sunset. That caps it at `sky_fraction * horizon_glow`, which
    /// at the defaults puts night at 0.19 of the light noon gets against dusk's 0.21
    /// — turning night up means turning [`PlanetConfig::horizon_glow`] up with it,
    /// and `the_world_never_brightens_as_the_sun_sinks` is the guard.
    pub night_sky: Vec3,
    /// How far below the horizon the sky stays lit. This is the entire dusk: the
    /// sky term fades out across it and [`PlanetConfig::night_sky`] fades in, so
    /// twilight is a crossing rather than an event.
    pub twilight_degrees: f32,
    /// How many tiles tall the full 0..1 height range stands.
    ///
    /// The heightmap is stored with no physical meaning, and "is that ridge high
    /// enough to hide the sun" cannot be answered without one. **This is a lighting
    /// number and nothing else may read it** — the terrain has managed without a
    /// vertical scale and should keep managing, or a tile's kind would start
    /// depending on it.
    ///
    /// A taller world shades more of itself, and
    /// `the_default_relief_measures_what_the_mountains_shade` is where 128 came from
    /// — the share of lowland tiles the mountains put in shadow:
    ///
    /// ```text
    ///   relief    6 deg   12 deg   24 deg   35 deg   45 deg   55 deg (noon)
    ///       64    57.7%    22.7%     5.1%     1.1%     0.1%     0.0%
    ///      128    65.7%    57.8%    25.5%    17.2%     6.9%     1.2%
    ///      192    68.5%    63.6%    34.5%    24.1%    19.5%     7.5%
    ///      256    71.8%    65.9%    58.2%    33.7%    25.4%    18.5%
    /// ```
    ///
    /// At 64 the shadows are gone by mid-morning and the feature is invisible for
    /// ten hours of a twelve-hour day. At 192 and up a ridge shades the country
    /// beside it at *noon*, which stops reading as a shadow and starts reading as
    /// dirt. 128 keeps it a low-sun feature — a quarter of the ground at 24 degrees,
    /// almost none at noon — which is what was asked for.
    pub relief_tiles: f32,
    /// How far above the sun's ray a tile has to stand, in stored height units, to
    /// hide it completely. Softened rather than cut because the sun sweeps a ridge
    /// *past* a sample distance, and a hard test flickers when it does.
    pub shadow_softness: f32,
    /// The three distances along the bearing the shadow test looks at. Three samples
    /// are the entire ray march — 3 texture loads a fragment against a real march's
    /// dozens — and the price is that a lone spire between two of them throws
    /// nothing.
    pub shadow_near_tiles: f32,
    pub shadow_mid_tiles: f32,
    pub shadow_far_tiles: f32,
}

impl Default for PlanetConfig {
    fn default() -> Self {
        Self {
            rotation_period_seconds: 300.0,
            start_rotation: 0.30,
            latitude_degrees: 35.0,
            axial_tilt_degrees: 23.4,
            orbit_phase: 0.0,
            zenith_extinction: Vec3::new(0.04, 0.15, 0.38),
            daylight: 1.0,
            sky_fraction: 0.38,
            horizon_glow: 0.55,
            sky_saturation: 0.30,
            cosine_response: 0.55,
            night_sky: Vec3::new(0.15, 0.19, 0.29),
            twilight_degrees: 8.0,
            relief_tiles: 128.0,
            shadow_softness: 0.012,
            shadow_near_tiles: 1.0,
            shadow_mid_tiles: 5.0,
            shadow_far_tiles: 10.0,
        }
    }
}

/// Where the sun is, in the world's own frame: `+x` east, `+y` north.
#[derive(Clone, Copy, Debug, Default)]
pub struct SunPosition {
    /// Above the horizon, in radians. Negative is night, and that is the only thing
    /// night is.
    pub altitude: f32,
    /// The direction to the sun projected onto the ground, as a unit vector — which
    /// is the direction a shadow is cast *from*, and so the direction the occlusion
    /// test walks.
    ///
    /// At latitude 0 with no declination this is exactly `+-x`. The brief's "rays
    /// run left to right" is therefore what the model reports at the equator, rather
    /// than a shortcut built into it; away from the equator it sweeps.
    pub bearing: Vec2,
    /// How far the ray climbs, in tiles, per tile travelled along the bearing — the
    /// tangent of the altitude. Zero at and below the horizon, and enormous near the
    /// zenith, which is what makes a shadow long at dawn and absent overhead with no
    /// knob saying so.
    pub ray_slope: f32,
}

/// What that position delivers. Two terms, because a shadow is the loss of exactly
/// one of them: `direct` is the beam, `sky` is everything scattered, and a tile the
/// beam cannot reach keeps only the second.
#[derive(Clone, Copy, Debug)]
pub struct Insolation {
    pub direct: Vec3,
    pub sky: Vec3,
}

impl Default for Insolation {
    /// Full daylight and no shadow, which is what anything reading this without the
    /// plugin gets — the same fallback the weather's absent maps give a clear sky.
    fn default() -> Self {
        Self {
            direct: Vec3::ONE,
            sky: Vec3::ZERO,
        }
    }
}

/// The sky as anything outside this module reads it, and the session's only clock.
///
/// [`Sun::rotation`] is the one piece of state: how far the planet has turned since
/// local midnight, wrapped to 0..1. It *is* local solar time — there is no second
/// clock — and everything else on here is re-derived from it every frame.
///
/// World state, so it goes in on entering gameplay and out on leaving: no session
/// inherits the last one's afternoon.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct Sun {
    pub rotation: f32,
    /// How far the star stands off the equator today. Constant while
    /// [`PlanetConfig::orbit_phase`] is, and derived rather than stored so that an
    /// orbit is the only thing a season would have to add.
    #[allow(
        dead_code,
        reason = "part of the reader gh-26 exists to provide; seasons are what will read it"
    )]
    pub declination: f32,
    pub position: SunPosition,
    pub light: Insolation,
}

impl Sun {
    /// Local solar time, on 0..24. The rotation *is* the hour; this only scales it.
    #[allow(
        dead_code,
        reason = "the reader gh-26 exists to provide; nothing simulation-side consumes it yet"
    )]
    pub fn hour(&self) -> f32 {
        self.rotation * 24.0
    }

    /// Whether there is a beam at all. The only definition of day in the crate.
    #[allow(
        dead_code,
        reason = "the reader gh-26 exists to provide; nothing simulation-side consumes it yet"
    )]
    pub fn is_up(&self) -> bool {
        self.position.altitude > 0.0
    }

    /// How much light the world is getting in total, on the same scale
    /// [`PlanetConfig::daylight`] is set in. What the clouds are lit by.
    pub fn light_level(&self) -> f32 {
        (self.light.sky + self.light.direct).dot(LUMINANCE)
    }
}

pub struct SunPlugin;

impl Plugin for SunPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PlanetConfig>();
        app.add_systems(OnEnter(Screen::Gameplay), start_planet_clock);
        app.add_systems(OnExit(Screen::Gameplay), end_planet_clock);
        app.add_systems(
            Update,
            // The planet is the only writer of the rotation and the sync is the only
            // reader, so this pair is the whole ordering this module needs.
            (turn_the_planet, sync_sun_light)
                .chain()
                .run_if(in_state(Screen::Gameplay)),
        );
    }
}

/// Opens the session's rotation at the configured angle, not at wherever the last
/// session left the planet.
fn start_planet_clock(mut commands: Commands, config: Res<PlanetConfig>) {
    commands.insert_resource(sun_at(&config, config.start_rotation.rem_euclid(1.0)));
}

fn end_planet_clock(mut commands: Commands) {
    commands.remove_resource::<Sun>();
}

/// Advances the rotation and re-derives everything that hangs off it.
///
/// The clock does not pause: while gameplay is up the planet turns and nothing but
/// leaving the screen stops it, so the rotation is the session's elapsed time.
fn turn_the_planet(time: Res<Time>, config: Res<PlanetConfig>, mut sun: ResMut<Sun>) {
    // Wrapped rather than accumulated, the same reason `WeatherClock` wraps: an
    // angle that grew all session would eventually quantize, and a full turn is
    // exactly where a wrap is invisible.
    let rotation = (sun.rotation + time.delta_secs() / config.rotation_period_seconds).fract();
    *sun = sun_at(&config, rotation);
}

/// Puts this frame's sun into the tint overlay, which draws it, and its level into
/// the weather's, which is composited over an already-lit world and would otherwise
/// hang white clouds in a midnight sky.
fn sync_sun_light(
    sun: Res<Sun>,
    config: Res<PlanetConfig>,
    overlays: Single<(&mut TerrainTintOverlay, &mut WeatherOverlay)>,
) {
    let (mut tint, mut weather) = overlays.into_inner();
    tint.set_sun(&sun, &config);
    weather.set_light_level(sun.light_level());
}

// -- The model ---------------------------------------------------------------

/// Everything a rotation implies, in one evaluation — so the sun the world is lit by
/// and the sun anything else reads can never be two different suns.
fn sun_at(config: &PlanetConfig, rotation: f32) -> Sun {
    let declination = declination(config);
    let position = position(config, rotation, declination);
    Sun {
        rotation,
        declination,
        position,
        light: insolation(config, position.altitude),
    }
}

/// How far the star stands off the equator: the axial tilt projected onto where the
/// orbit has got to. Zero at an equinox, the full tilt at a solstice.
fn declination(config: &PlanetConfig) -> f32 {
    (config.axial_tilt_degrees.to_radians().sin() * (TAU * config.orbit_phase).sin()).asin()
}

/// Latitude, declination and hour angle into an altitude and a ground bearing.
fn position(config: &PlanetConfig, rotation: f32, declination: f32) -> SunPosition {
    let (sin_lat, cos_lat) = config.latitude_degrees.to_radians().sin_cos();
    let (sin_dec, cos_dec) = declination.sin_cos();
    // The hour angle runs westward from local noon, and the rotation is measured
    // from local midnight — so half a turn apart, and that is the whole conversion.
    let (sin_hour, cos_hour) = (TAU * (rotation - 0.5)).sin_cos();

    let sin_altitude = (sin_lat * sin_dec + cos_lat * cos_dec * cos_hour).clamp(-1.0, 1.0);

    // The sun's direction projected onto the ground, in the world's frame. At
    // latitude 0 with no declination the north term vanishes identically and this is
    // exactly +-x.
    let east = -cos_dec * sin_hour;
    let north = sin_dec * cos_lat - cos_dec * sin_lat * cos_hour;

    let altitude = sin_altitude.asin();
    SunPosition {
        altitude,
        // Straight overhead there is no bearing to have, and no shadow can fall
        // either, so any direction will do.
        bearing: Vec2::new(east, north).normalize_or(Vec2::X),
        ray_slope: if altitude > 0.0 {
            // Bounded because the tangent runs away at the zenith and an infinity
            // would reach the shader. Anything this large already outruns the
            // heightmap, so the clamp changes no pixel.
            altitude.tan().min(1.0e4)
        } else {
            0.0
        },
    }
}

/// What an altitude delivers, split into the beam and everything scattered.
///
/// The whole thing is measured against the **reference altitude** — equinox noon for
/// this latitude — where the two terms add to [`PlanetConfig::daylight`] and the
/// light is exactly white. That is a white balance, and it is doing the same job an
/// eye does: without it a physically blue sky would sit the entire world under a
/// cast the pixel art was never drawn for.
fn insolation(config: &PlanetConfig, altitude: f32) -> Insolation {
    let reference = reference_altitude(config);
    // How high the sun stands as a share of the reference — 1 at equinox noon, 0 at
    // the horizon. Both terms are measured against it, which is what makes the
    // balance hold whatever else is tuned.
    let height = altitude.sin().max(0.0) / reference.sin().max(f32::EPSILON);

    // The sky is lit by the sun, so it dims with it — held flat all the way down it
    // would swamp the beam and dusk would come out neutral grey instead of warm. It
    // does not dim to nothing, because at sunset the air overhead is still in full
    // daylight, and the square root between the two is roughly how diffuse light
    // behaves: much flatter than the beam's own falloff.
    let sky_share = config.daylight * config.sky_fraction;
    let glow = config.horizon_glow + (1.0 - config.horizon_glow) * height.sqrt();
    let daytime_sky = sky_share * glow * sky_hue(config, air_mass(altitude));

    // Dusk: the sky's own light fading out around the horizon while the moon fades
    // in. Nothing here fires at a threshold — this crossing *is* twilight.
    let lit_sky = smoothstep(-config.twilight_degrees.to_radians(), 0.0, altitude);
    let sky = daytime_sky * lit_sky + config.night_sky * (1.0 - lit_sky);

    // The beam is whatever is left of the reference light once the sky has taken its
    // share there — so the two add to `daylight` at the reference and the world is
    // drawn as painted at equinox noon. `glow` is 1 there by construction, so this
    // holds however the sky is tuned.
    let reference_beam = (Vec3::splat(config.daylight)
        - sky_share * sky_hue(config, air_mass(reference)))
    .max(Vec3::ZERO);

    // Two things dim and redden the beam away from that reference: the extra air it
    // crosses, per channel, and the angle it arrives at.
    let extra_air = air_mass(altitude) - air_mass(reference);
    let transmission = exp(-config.zenith_extinction * extra_air);
    let direct = reference_beam * transmission * height.powf(config.cosine_response);

    Insolation {
        // Capped so that a solstice noon — which stands higher than the reference —
        // brightens nothing past the art. At the default equinox this never binds.
        direct: direct.min((Vec3::splat(config.daylight) - sky).max(Vec3::ZERO)),
        sky,
    }
}

/// The highest the sun gets here at an equinox, which is what the light is balanced
/// against. Derived rather than configured, so every latitude is measured against
/// its own noon instead of against a number someone had to keep in step with it.
fn reference_altitude(config: &PlanetConfig) -> f32 {
    (90.0 - config.latitude_degrees.abs()).to_radians()
}

/// How much atmosphere the beam crosses to arrive: 1 straight overhead, ~38 at the
/// horizon.
///
/// Kasten-Young rather than the schoolbook `1/sin`, which diverges at the horizon and
/// would need an arbitrary clamp exactly where the interesting hours are. It is only
/// defined above the horizon; below it the horizon's own value is the right limit,
/// since by then there is no beam left and only the sky's hue still reads this.
fn air_mass(altitude: f32) -> f32 {
    let degrees = altitude.to_degrees().max(0.0);
    1.0 / (degrees.to_radians().sin() + 0.50572 * (degrees + 6.07995).powf(-1.6364))
}

/// The colour of the sky, from what the air took out of the beam.
///
/// Normalised to unit luminance, so [`PlanetConfig::sky_fraction`] means "how bright
/// a shadow is" and [`PlanetConfig::sky_saturation`] means "how blue it is" and
/// neither knob moves the other. The saturation mixes toward white, which also has
/// unit luminance, so the mix cannot change the brightness either.
fn sky_hue(config: &PlanetConfig, air_mass: f32) -> Vec3 {
    let scattered = Vec3::ONE - exp(-config.zenith_extinction * air_mass);
    let luminance = scattered.dot(LUMINANCE);
    let hue = if luminance > f32::EPSILON {
        scattered / luminance
    } else {
        Vec3::ONE
    };
    hue.lerp(Vec3::ONE, 1.0 - config.sky_saturation)
}

/// Per channel, since glam has no component-wise exponential.
fn exp(v: Vec3) -> Vec3 {
    Vec3::new(v.x.exp(), v.y.exp(), v.z.exp())
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rotation a sunrise or a sunset falls at, found by bisection rather than
    /// by inverting the altitude — the inverse has no closed form at a latitude
    /// where the sun may not rise at all.
    fn horizon_crossing(config: &PlanetConfig, mut dark: f32, mut light: f32) -> f32 {
        let up = |rotation: f32| sun_at(config, rotation).is_up();
        assert!(!up(dark) && up(light), "the crossing has to be bracketed");
        for _ in 0..40 {
            let middle = 0.5 * (dark + light);
            if up(middle) {
                light = middle;
            } else {
                dark = middle;
            }
        }
        0.5 * (dark + light)
    }

    /// How much of one turn the sun is above the horizon for.
    fn day_fraction(config: &PlanetConfig) -> f32 {
        (0..2000)
            .filter(|step| sun_at(config, *step as f32 / 2000.0).is_up())
            .count() as f32
            / 2000.0
    }

    /// The one thing the brief allowed assuming, held here as an output instead: at
    /// the equator at an equinox the bearing is exactly `+-x`, so rays really do run
    /// left to right — because that is what the model says there, not because
    /// anything special-cases it.
    #[test]
    fn at_the_equator_at_an_equinox_the_sun_runs_exactly_left_to_right() {
        let config = PlanetConfig {
            latitude_degrees: 0.0,
            orbit_phase: 0.0,
            ..default()
        };

        for step in 0..64 {
            let rotation = step as f32 / 64.0;
            let sun = sun_at(&config, rotation);
            if !sun.is_up() {
                continue;
            }
            assert!(
                sun.position.bearing.y.abs() < 1.0e-5,
                "at rotation {rotation} the bearing left the x axis: {:?}",
                sun.position.bearing,
            );
        }
    }

    /// And away from the equator it does not, which is the entire reason the bearing
    /// is a vector rather than a sign.
    #[test]
    fn away_from_the_equator_the_bearing_sweeps_through_the_day() {
        let config = PlanetConfig::default();

        let morning = sun_at(&config, 0.30).position.bearing;
        let noon = sun_at(&config, 0.50).position.bearing;
        let evening = sun_at(&config, 0.70).position.bearing;

        // Due east, due south, due west: a quarter turn each way over the day.
        assert!(morning.x > 0.0 && noon.x.abs() < 1.0e-5 && evening.x < 0.0);
        assert!(
            noon.y < -0.99,
            "at 35 degrees north the noon sun stands to the south, not overhead: {noon:?}",
        );
        assert!(
            morning.angle_to(evening).abs() > 1.0,
            "the bearing barely moved: {morning:?} to {evening:?}",
        );
    }

    /// Sunrise in the east, sunset in the west, and noon the highest point between
    /// them. None of this is arranged; it is what the hour angle does.
    #[test]
    fn the_sun_rises_in_the_east_and_sets_in_the_west() {
        let config = PlanetConfig::default();

        let sunrise = sun_at(&config, horizon_crossing(&config, 0.20, 0.30));
        let sunset = sun_at(&config, horizon_crossing(&config, 0.80, 0.70));

        assert!(
            sunrise.position.bearing.x > 0.99,
            "the sun did not rise due east: {:?}",
            sunrise.position.bearing,
        );
        assert!(
            sunset.position.bearing.x < -0.99,
            "the sun did not set due west: {:?}",
            sunset.position.bearing,
        );
        assert!(sunrise.rotation < 0.5 && sunset.rotation > 0.5);

        let noon = sun_at(&config, 0.5).position.altitude;
        assert!(
            noon > sun_at(&config, 0.35).position.altitude
                && noon > sun_at(&config, 0.65).position.altitude,
            "noon was not the top of the arc",
        );
    }

    /// Day length is not configured anywhere, so this is the geometry or nothing: at
    /// an equinox the sun is up for exactly half a turn at *every* latitude.
    #[test]
    fn an_equinox_day_is_half_a_turn_at_any_latitude() {
        for latitude_degrees in [0.0, 35.0, 55.0, -20.0] {
            let config = PlanetConfig {
                latitude_degrees,
                orbit_phase: 0.0,
                ..default()
            };
            let day = day_fraction(&config);
            assert!(
                (day - 0.5).abs() < 0.01,
                "at latitude {latitude_degrees} the equinox day was {day} of a turn",
            );
        }
    }

    /// And away from an equinox it is not — a summer day is longer than the winter
    /// one at the same place, with no knob anywhere saying how long either is. This
    /// is what "seasons are one moving knob away" means.
    #[test]
    fn a_summer_day_is_longer_than_a_winter_day_at_the_same_latitude() {
        let summer = day_fraction(&PlanetConfig {
            orbit_phase: 0.25,
            ..default()
        });
        let winter = day_fraction(&PlanetConfig {
            orbit_phase: 0.75,
            ..default()
        });

        assert!(
            summer > winter + 0.05,
            "the seasons did not part: summer {summer}, winter {winter}",
        );
        assert!(
            (summer + winter - 1.0).abs() < 0.02,
            "a solstice pair should still split the year evenly: {summer} and {winter}",
        );
    }

    /// The colour is never keyframed, so this has to fall out of the air mass: the
    /// lower the sun, the warmer its beam, monotonically and at every step.
    #[test]
    fn the_beam_reddens_as_the_sun_sinks() {
        let config = PlanetConfig::default();

        let warmth = |altitude_degrees: f32| {
            let light = insolation(&config, altitude_degrees.to_radians());
            light.direct.x / light.direct.z.max(f32::EPSILON)
        };

        let mut previous = warmth(55.0);
        for altitude in [45.0, 35.0, 25.0, 15.0, 8.0, 3.0] {
            let warmth = warmth(altitude);
            assert!(
                warmth > previous,
                "the beam at {altitude} degrees was not warmer than the step above it",
            );
            previous = warmth;
        }
        assert!(
            warmth(3.0) > 3.0 * warmth(55.0),
            "a sun three degrees up should be markedly redder than one at noon",
        );
    }

    /// The beam reddening is not enough on its own: what a player sees on sunlit
    /// ground is beam *plus* sky, and the first cut of this model had the sky held
    /// flat all the way down to the horizon, where it swamped the beam and made dusk
    /// come out neutral grey. The whole point of an evening is that the light on the
    /// ground goes warm, so that is what this asserts.
    #[test]
    fn dusk_falls_warm_on_the_ground_and_cool_in_the_shade() {
        let config = PlanetConfig::default();
        let dusk = insolation(&config, 12.0_f32.to_radians());

        let lit = dusk.sky + dusk.direct;
        assert!(
            lit.x > lit.z * 1.1,
            "sunlit ground at dusk should read warm, not grey: {lit:?}",
        );
        assert!(
            dusk.sky.z > dusk.sky.x * 1.2,
            "the shade at dusk is lit by the sky, which is the cool half: {:?}",
            dusk.sky,
        );

        // And the day it is a departure from: noon is neutral by construction, so the
        // cast at dusk is the whole of the difference.
        let noon = insolation(&config, reference_altitude(&config));
        let noon_lit = noon.sky + noon.direct;
        assert!(
            (noon_lit.x / noon_lit.z - 1.0).abs() < 0.02,
            "noon is the reference and has to be neutral: {noon_lit:?}",
        );
    }

    /// The sky is lit by the sun, so it has to dim with it — and the moon has to fade
    /// in *under* the dusk it replaces. Get either wrong and the world brightens
    /// after sunset, which no amount of colour tuning hides.
    #[test]
    fn the_world_never_brightens_as_the_sun_sinks() {
        let config = PlanetConfig::default();

        let level = |altitude_degrees: f32| {
            let light = insolation(&config, altitude_degrees.to_radians());
            (light.sky + light.direct).dot(LUMINANCE)
        };

        // Down to the bottom of the twilight band the light strictly falls; below it
        // the moon is all there is and it is flat, so the property is that the world
        // never gets *brighter*, not that it keeps dimming forever.
        let mut previous = level(55.0);
        for altitude in [
            45.0, 35.0, 25.0, 15.0, 8.0, 3.0, 0.0, -4.0, -8.0, -20.0, -50.0,
        ] {
            let level = level(altitude);
            assert!(
                level <= previous,
                "the world got brighter going from above {altitude} degrees to it: \
                 {previous} then {level}",
            );
            previous = level;
        }

        // The part that has to actually move: night is dimmer than the dusk it fades
        // in under, which is what stops the world lighting up after sunset.
        assert!(
            level(-50.0) < level(0.0),
            "night ({}) has to be darker than dusk ({})",
            level(-50.0),
            level(0.0),
        );
    }

    /// A shadow is the absence of the beam and nothing else, so the sky term alone
    /// has to be a usable amount of light — and never zero, or night would be black
    /// with nothing able to light it.
    #[test]
    fn a_shadow_and_a_night_are_lit_by_the_sky_alone() {
        let config = PlanetConfig::default();

        let noon = insolation(&config, reference_altitude(&config));
        let shadow = noon.sky.dot(LUMINANCE);
        assert!(
            (0.2..0.6).contains(&shadow),
            "a noon shadow at {shadow} is either black or barely a shadow",
        );

        let night = insolation(&config, -30.0_f32.to_radians());
        assert_eq!(night.direct, Vec3::ZERO, "there is no beam at night");
        assert!(
            night.sky.dot(LUMINANCE) > 0.15,
            "night has to stay readable — nothing else can light it: {:?}",
            night.sky,
        );
        assert!(
            night.sky.z > night.sky.x,
            "night is lit by the moon and the sky, so it is cool: {:?}",
            night.sky,
        );
    }

    /// The light multiplies, so anything over 1 would draw the world brighter than
    /// the art. The cap has to hold at every hour of every season, including the
    /// solstice noon that stands higher than the reference the balance is taken at.
    #[test]
    fn no_hour_of_any_season_brightens_the_world_past_the_art() {
        for orbit_phase in [0.0, 0.25, 0.5, 0.75] {
            for latitude_degrees in [0.0, 35.0, 55.0] {
                let config = PlanetConfig {
                    orbit_phase,
                    latitude_degrees,
                    ..default()
                };
                for step in 0..500 {
                    let sun = sun_at(&config, step as f32 / 500.0);
                    let total = sun.light.sky + sun.light.direct;
                    assert!(
                        total.max_element() <= config.daylight + 1.0e-5,
                        "at latitude {latitude_degrees}, orbit {orbit_phase}, rotation {} \
                         the light was {total:?}",
                        sun.rotation,
                    );
                    assert!(total.min_element() > 0.0, "no hour is unlit");
                }
            }
        }
    }

    /// Equinox noon is the reference the whole model is measured against, so the
    /// world there is drawn exactly as it was painted — white, and at full strength.
    #[test]
    fn the_world_is_drawn_as_painted_at_equinox_noon() {
        let config = PlanetConfig::default();
        let light = insolation(&config, reference_altitude(&config));
        let total = light.sky + light.direct;

        for channel in [total.x, total.y, total.z] {
            assert!(
                (channel - config.daylight).abs() < 1.0e-4,
                "equinox noon should be neutral and full: {total:?}",
            );
        }
    }

    /// Night is not a state and dusk is not an event: the light has to walk from one
    /// to the other without a step in it.
    #[test]
    fn the_light_never_jumps_between_one_moment_and_the_next() {
        let config = PlanetConfig::default();

        let steps = 20_000;
        let mut previous = sun_at(&config, 0.0).light;
        for step in 1..=steps {
            let light = sun_at(&config, step as f32 / steps as f32).light;
            let jump = (light.direct - previous.direct)
                .abs()
                .max((light.sky - previous.sky).abs())
                .max_element();
            assert!(
                jump < 0.01,
                "the light stepped by {jump} at rotation {}",
                step as f32 / steps as f32,
            );
            previous = light;
        }
    }

    /// Shadows exist only while there is a beam to block, and lengthen as it sinks.
    #[test]
    fn the_ray_lies_flat_at_dawn_and_stands_up_at_noon() {
        let config = PlanetConfig::default();

        let dawn = sun_at(&config, horizon_crossing(&config, 0.20, 0.30));
        assert!(
            dawn.position.ray_slope < 0.02,
            "a ray at the horizon is flat, so its shadow reaches everything",
        );

        let noon = sun_at(&config, 0.5).position.ray_slope;
        assert!(
            noon > 1.0,
            "the noon ray should clear ordinary relief: {noon}"
        );

        let night = sun_at(&config, 0.0);
        assert!(!night.is_up() && night.position.ray_slope == 0.0);
    }

    /// The rotation is the only state, and it wraps rather than growing — a session
    /// hours long has to be as precise as one a minute old.
    #[test]
    fn the_rotation_wraps_rather_than_accumulating() {
        let config = PlanetConfig::default();
        let mut sun = sun_at(&config, config.start_rotation);

        // A hundred turns, at a frame's worth of a turn each.
        let step = 1.0 / 900.0;
        for _ in 0..90_000 {
            sun = sun_at(&config, (sun.rotation + step).fract());
            assert!((0.0..1.0).contains(&sun.rotation));
        }

        let expected = (config.start_rotation + 90_000.0 * step).fract();
        assert!(
            (sun.rotation - expected).abs() < 1.0e-3,
            "the clock drifted to {} rather than {expected}",
            sun.rotation,
        );
    }

    /// `hour` is the rotation and nothing else: there is no second clock to fall out
    /// of step with it.
    #[test]
    fn the_hour_of_the_day_is_the_rotation() {
        let config = PlanetConfig::default();
        assert!((sun_at(&config, 0.0).hour() - 0.0).abs() < 1.0e-5);
        assert!((sun_at(&config, 0.5).hour() - 12.0).abs() < 1.0e-5);
        assert!((sun_at(&config, 0.75).hour() - 18.0).abs() < 1.0e-5);
    }

    /// Absent the plugin, [`Insolation`]'s default is full daylight and no shadow —
    /// which is what lets anything that later reads the sun be tested with no sun at
    /// all.
    #[test]
    fn the_absence_of_a_sun_is_full_daylight() {
        let light = Insolation::default();
        assert_eq!(light.direct + light.sky, Vec3::ONE);
    }

    /// The shader's occlusion test, transcribed — the same arithmetic
    /// `assets/shaders/tint.wgsl` does per fragment, so it can be measured against
    /// real terrain without a GPU. The two are edited together.
    fn shadow_at(
        config: &PlanetConfig,
        heights: &dyn Fn(Vec2) -> f32,
        tile: Vec2,
        sun: &SunPosition,
    ) -> f32 {
        if sun.ray_slope <= 0.0 {
            return 0.0;
        }

        let height = heights(tile);
        [
            config.shadow_near_tiles,
            config.shadow_mid_tiles,
            config.shadow_far_tiles,
        ]
        .into_iter()
        .map(|distance| {
            let at = (tile + sun.bearing * distance).floor();
            let ray = height + distance * sun.ray_slope / config.relief_tiles;
            smoothstep(0.0, config.shadow_softness, heights(at) - ray)
        })
        .fold(0.0, f32::max)
    }

    /// A wall on one side of a tile, and nothing anywhere else — the simplest world
    /// that can cast a shadow at all.
    fn wall_at(east_of: f32, height: f32) -> impl Fn(Vec2) -> f32 {
        move |at: Vec2| if at.x >= east_of { height } else { 0.0 }
    }

    /// A sun at `altitude_degrees` due east, which is where the shadow test's whole
    /// input comes from.
    fn sun_due_east(altitude_degrees: f32) -> SunPosition {
        let altitude = altitude_degrees.to_radians();
        SunPosition {
            altitude,
            bearing: Vec2::X,
            ray_slope: altitude.tan().max(0.0),
        }
    }

    /// The occlusion test, on a world made of one wall: the sun is hidden while the
    /// wall stands above the ray, and the ray is whatever the altitude makes it. No
    /// threshold on the sun's height appears anywhere — the geometry decides, and
    /// the crossing is a slope rather than a switch.
    #[test]
    fn a_wall_shades_the_ground_beside_it_only_while_the_sun_is_low() {
        let config = PlanetConfig::default();
        // A wall a fiftieth of the height range tall — 2.56 tiles at the default
        // relief — standing one tile east of the tile under test. It hides a sun
        // that climbs slower than 2.56 tiles per tile, which is 68.7 degrees.
        let world = wall_at(6.0, 0.02);
        let shadow =
            |degrees| shadow_at(&config, &world, Vec2::new(5.0, 0.0), &sun_due_east(degrees));

        assert!(shadow(10.0) > 0.99, "a low sun is behind the wall");
        assert!(
            shadow(75.0) == 0.0,
            "a sun over the wall's own angle is not"
        );

        // And in between, a partial shadow: the softening is why a ridge sweeping
        // past a sample distance fades rather than flickering.
        let crossing = shadow(60.0);
        assert!(
            crossing > 0.0 && crossing < 1.0,
            "the shadow edge has to be soft, not a switch: {crossing}",
        );
    }

    /// The three distances are not decoration: a ridge ten tiles off stops shading
    /// long before one at your elbow does, and the far sample is the only thing that
    /// sees it at all.
    #[test]
    fn a_distant_ridge_shades_only_while_the_sun_is_very_low() {
        let config = PlanetConfig::default();
        // Far enough east that only the ten-tile sample reaches it.
        let world = wall_at(14.0, 0.02);
        let shadow =
            |degrees| shadow_at(&config, &world, Vec2::new(5.0, 0.0), &sun_due_east(degrees));

        assert!(
            shadow(5.0) > 0.99,
            "at five degrees the ridge still reaches"
        );
        assert!(shadow(20.0) == 0.0, "by twenty the ray clears it");

        // The same wall at the near sample is still shading at twenty degrees, which
        // is the whole point of sampling at more than one distance.
        let near = shadow_at(
            &config,
            &wall_at(6.0, 0.02),
            Vec2::new(5.0, 0.0),
            &sun_due_east(20.0),
        );
        assert!(near > 0.99, "distance is what decides, not height alone");
    }

    /// The far side of a wall is lit: the test looks *toward* the sun, so which side
    /// of an obstacle a tile is on is the whole answer.
    #[test]
    fn the_sunward_side_of_a_wall_is_the_lit_one() {
        let config = PlanetConfig::default();
        let world = wall_at(6.0, 0.10);
        let sun = SunPosition {
            altitude: 10.0_f32.to_radians(),
            bearing: Vec2::X,
            ray_slope: 10.0_f32.to_radians().tan(),
        };

        assert!(
            shadow_at(&config, &world, Vec2::new(5.0, 0.0), &sun) > 0.5,
            "the tile west of the wall, with the sun east of it, is in its shadow",
        );
        assert!(
            shadow_at(&config, &world, Vec2::new(20.0, 0.0), &sun) == 0.0,
            "the tile east of the wall has the sun on it",
        );
    }

    /// No sun, no shadow — and no samples read either, which is what makes night
    /// cost nothing.
    #[test]
    fn nothing_is_shadowed_when_the_sun_is_down() {
        let config = PlanetConfig::default();
        let sun = sun_at(&config, 0.0).position;
        assert!(!sun_at(&config, 0.0).is_up());
        assert_eq!(
            shadow_at(&config, &wall_at(6.0, 0.5), Vec2::new(5.0, 0.0), &sun),
            0.0,
        );
    }

    /// How much of the world the mountains actually shade, which is the only thing
    /// that says whether `relief_tiles` is the right number.
    ///
    /// The knob is a vertical scale for a heightmap that has none, so it cannot be
    /// reasoned about — it has to be read off the terrain. Too tall and every ridge
    /// shades the country beside it at noon; too short and dawn casts nothing.
    ///
    /// `cargo test --release -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, not a check"]
    fn the_default_relief_measures_what_the_mountains_shade() {
        use crate::gameplay::terrain::{TerrainConfig, height_byte};

        let terrain = TerrainConfig::default();
        let sampler = terrain.sampler();
        let config = PlanetConfig::default();

        // A patch of real world, quantized exactly as the heightmap stores it, so the
        // measurement sees the same bytes the shader loads.
        const SIDE: u32 = 256;
        for (label, corner) in [
            ("lowland ", UVec2::new(1024, 3072)),
            ("highland", UVec2::new(3072, 1024)),
        ] {
            let patch: Vec<f32> = (0..SIDE * SIDE)
                .map(|index| {
                    let tile = corner + UVec2::new(index % SIDE, index / SIDE);
                    height_byte(sampler.elevation(tile.x as f32, tile.y as f32)) as f32 / 255.0
                })
                .collect();
            let heights = |at: Vec2| {
                let local = at - corner.as_vec2();
                if local.x < 0.0
                    || local.y < 0.0
                    || local.x >= SIDE as f32
                    || local.y >= SIDE as f32
                {
                    return 0.0;
                }
                patch[(local.y as u32 * SIDE + local.x as u32) as usize]
            };

            println!("\n  {label}   relief   hour   altitude   shaded   deep");
            for relief_tiles in [64.0, 128.0, 192.0, 256.0] {
                let config = PlanetConfig {
                    relief_tiles,
                    ..config.clone()
                };
                for hour in [6.5, 7.0, 8.0, 9.0, 10.0, 12.0] {
                    let sun = sun_at(&config, hour / 24.0).position;
                    let (mut shaded, mut deep) = (0u32, 0u32);
                    // The border is skipped: a sample off the patch reads 0 and would
                    // count as unshaded regardless of what stands there.
                    for y in 16..SIDE - 16 {
                        for x in 16..SIDE - 16 {
                            let tile = corner.as_vec2() + Vec2::new(x as f32, y as f32);
                            let shadow = shadow_at(&config, &heights, tile, &sun);
                            if shadow > 0.05 {
                                shaded += 1;
                            }
                            if shadow > 0.75 {
                                deep += 1;
                            }
                        }
                    }

                    let tiles = ((SIDE - 32) * (SIDE - 32)) as f32;
                    println!(
                        "             {relief_tiles:>5.0}   {hour:>4.1}   {:>7.1}   {:>5.1}%   {:>5.1}%",
                        sun.altitude.to_degrees(),
                        shaded as f32 / tiles * 100.0,
                        deep as f32 / tiles * 100.0,
                    );
                }
            }
        }
        println!();
    }

    /// What a day at the defaults actually looks like, hour by hour.
    ///
    /// `cargo test --release -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, not a check"]
    fn the_default_planet_measures_its_day() {
        let config = PlanetConfig::default();

        println!("\n  hour   altitude   bearing            lit                shadow      slope");
        for step in 0..24 {
            let sun = sun_at(&config, step as f32 / 24.0);
            let lit = sun.light.sky + sun.light.direct;
            println!(
                "  {:>4.1}   {:>7.1}   ({:>5.2},{:>5.2})   ({:.2},{:.2},{:.2})   {:.2}   {:>8.2}",
                sun.hour(),
                sun.position.altitude.to_degrees(),
                sun.position.bearing.x,
                sun.position.bearing.y,
                lit.x,
                lit.y,
                lit.z,
                sun.light.sky.dot(LUMINANCE),
                sun.position.ray_slope,
            );
        }

        let day = day_fraction(&config);
        println!(
            "\n  equinox day {:.1} h, summer {:.1} h, winter {:.1} h\n",
            day * 24.0,
            day_fraction(&PlanetConfig {
                orbit_phase: 0.25,
                ..config.clone()
            }) * 24.0,
            day_fraction(&PlanetConfig {
                orbit_phase: 0.75,
                ..config.clone()
            }) * 24.0,
        );
    }
}
