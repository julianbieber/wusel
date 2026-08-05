//! An inspection overlay: the fields the world is built out of, drawn as false
//! colour over the map they made.
//!
//! Everything the world does comes from a handful of scalar fields — how high, how
//! warm, how wet — and until now none of them could be *seen*. You could read the
//! tiles they produced and guess. This draws the field itself, so "why is there no
//! snow here" is a question you look at rather than reason about.
//!
//! **It needs no new maps.** Every field is already bound to the one post-process
//! pass, because something else already needed it there: the heightmap for the ramp
//! and the sun's shadows, the climate map for what falls as snow, the cover map for
//! what lies, and the cloud probability map — which *is*
//! `TerrainSampler::humidity()`, baked once per session. So this module owns a mode,
//! two colour ramps and a legend, and the pass does the rest.
//!
//! **The overlay replaces the world rather than tinting it**, and short-circuits
//! before the composite. An inspector dimmed by nightfall or hidden under a cloud is
//! not an inspector; `InspectConfig::opacity` is there to blend it back if you want
//! the terrain for orientation.
//!
//! **Each field is shown at the resolution the game actually reads it at**, and the
//! difference is visible: height is one texel per tile and comes out crisp, where
//! temperature and moisture are drawn from maps on a 16-tile grid and come out in
//! soft blocks. That is not a defect to smooth over — the coarse grid is a real
//! property of how the ground steps, and being able to *see* it is worth more than a
//! prettier map would be. It is bilinear rather than nearest for the same reason: the
//! bilinear value is the one the composite decides rain against snow with, so it is
//! the number the game acts on.
//!
//! # The colours are not a free choice
//!
//! Two ramps, and both are chosen against the rules a false-colour map has to obey
//! rather than by taste:
//!
//! - **Never a rainbow.** The instinctive debug ramp — blue through green and yellow
//!   to red — is not monotone in lightness, so the eye reads its bright band as an
//!   *edge* and invents a boundary the data does not have. Both ramps here run one
//!   way in lightness.
//! - **Magnitude gets one hue, light to dark.** [`SEQUENTIAL`] is a single blue ramp,
//!   so "more" is always "darker" and nothing about the hue has to be learned.
//! - **Polarity gets two hues and a neutral middle.** Temperature is the one field
//!   with a meaningful zero — the freezing point — so it gets [`DIVERGING`], cool to
//!   neutral to warm. The payoff is that the **snow line is the neutral band**: you
//!   can see where it is without reading a number.
//! - **Nothing is encoded by colour alone.** The legend names the field and prints
//!   the ends of its range, which is also the relief a low-contrast ramp requires.
//!
//! The diverging poles were validated rather than eyeballed: `#0d366b` against
//! `#701312` measures ΔE 15.2 under protanopia and 20.9 to normal vision, clear of
//! the 8 and 15 floors, with the two arms within 0.02 of each other in lightness.

use bevy::{input::ButtonInput, prelude::*, ui::widget::Text};

use crate::{
    camera::{WorldCamera, visible_half_extent},
    gameplay::{
        deposit::Resource,
        ground::{ClimateMaps, GroundConfig, GroundCover, TemperatureOffset, temperature_offset},
        plan::WorldPlanConfig,
        prospect::ProspectMaps,
        screen::ScreenOverlay,
        sun::{PlanetConfig, Sun},
        weather::SkySampler,
        world::{TILE_DISPLAY_SIZE, WorldMap, tile_position_at},
    },
    screens::Screen,
};

/// The blue sequential ramp, light to dark, as three sRGB stops.
///
/// Magnitude only: the lightest end means "near nothing" and the darkest "all of
/// it". Three stops rather than two because a straight line between the ends drifts
/// off the hue in the middle; the middle stop is the ramp's own 400 step.
const SEQUENTIAL: [Vec3; 3] = [
    Vec3::new(0.804, 0.886, 0.984), // #cde2fb
    Vec3::new(0.224, 0.529, 0.898), // #3987e5
    Vec3::new(0.051, 0.212, 0.420), // #0d366b
];

/// The diverging ramp: cool, neutral, warm.
///
/// The middle stop is a light **neutral**, not a hue — a coloured midpoint would read
/// as a third category and put a false boundary either side of it. Both poles are
/// dark, so lightness peaks in the middle and falls away symmetrically, which is what
/// makes "how far from zero" readable at a glance.
const DIVERGING: [Vec3; 3] = [
    Vec3::new(0.051, 0.212, 0.420), // #0d366b
    Vec3::new(0.941, 0.937, 0.925), // #f0efec
    Vec3::new(0.439, 0.075, 0.071), // #701312
];

/// Which field the overlay is drawing, or none.
///
/// The discriminants *are* what the shader switches on, so the order here is the
/// order there — the same coupling `TerrainKind` has with the tileset, and the same
/// rule: nothing existing moves.
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum OverlayField {
    #[default]
    Off = 0,
    /// The heightmap, exactly as `classify` cut the tile from and the ramp shades by.
    Height = 1,
    /// What it is here *now*: the climate normal plus the day's swing and the season,
    /// so this one moves while you watch it.
    Temperature = 2,
    /// `TerrainSampler::humidity()` — where it rains, where rivers rise, and what
    /// damps the temperature swing. One field with three readers.
    Moisture = 3,
    /// How wet the ground is, from the ground's own state grid.
    Wetness = 4,
    /// And how much snow is lying on it.
    Snow = 5,
    /// How much cloud is overhead this instant.
    Cloud = 6,
    /// **Prospectivity, not seams**: how good this ground is for iron, before the
    /// threshold. A seam exists only where a cell's own jittered candidate landed on
    /// ground that cleared it, so this says "could there be iron here" — where there
    /// *is* one is the mark drawn over it, and `observe deposits`.
    Iron = 7,
    Copper = 8,
    Salt = 9,
}

impl OverlayField {
    /// Every field, **in discriminant order** — which is also key order, and what the
    /// key lookup walks. Nothing may be inserted in the middle.
    pub const ALL: [OverlayField; 10] = [
        OverlayField::Off,
        OverlayField::Height,
        OverlayField::Temperature,
        OverlayField::Moisture,
        OverlayField::Wetness,
        OverlayField::Snow,
        OverlayField::Cloud,
        OverlayField::Iron,
        OverlayField::Copper,
        OverlayField::Salt,
    ];

    pub fn label(self) -> &'static str {
        match self {
            OverlayField::Off => "off",
            OverlayField::Height => "height",
            OverlayField::Temperature => "temperature",
            OverlayField::Moisture => "moisture",
            OverlayField::Wetness => "wetness",
            OverlayField::Snow => "snow",
            OverlayField::Cloud => "cloud",
            OverlayField::Iron => "iron",
            OverlayField::Copper => "copper",
            OverlayField::Salt => "salt",
        }
    }

    /// The resource this field is the prospectivity of, if it is one.
    ///
    /// The bridge between the two enums, and the only one — the overlay names a
    /// resource, so a seventh resource with a recipe is a variant here and a row in
    /// `deposit.rs`, with nothing in between to keep in step.
    pub fn resource(self) -> Option<Resource> {
        match self {
            OverlayField::Iron => Some(Resource::Iron),
            OverlayField::Copper => Some(Resource::Copper),
            OverlayField::Salt => Some(Resource::Salt),
            _ => None,
        }
    }

    /// The digit that selects this field, on the number row and on the numpad.
    ///
    /// **The key is the discriminant**, which is also what the shader switches on —
    /// so pressing `2` and the uniform carrying 2 are the same 2, and there is no
    /// third table anywhere mapping one to the other. `0` is off, because off is the
    /// zero of that enum rather than a seventh state beside it.
    ///
    /// Both spellings, matching the zoom's `Equal`/`NumpadAdd` pair — a keyboard with
    /// a numpad has two keys with a `2` on them and it is not the player's job to
    /// know which one the game reads.
    fn keys(self) -> (KeyCode, KeyCode) {
        match self {
            OverlayField::Off => (KeyCode::Digit0, KeyCode::Numpad0),
            OverlayField::Height => (KeyCode::Digit1, KeyCode::Numpad1),
            OverlayField::Temperature => (KeyCode::Digit2, KeyCode::Numpad2),
            OverlayField::Moisture => (KeyCode::Digit3, KeyCode::Numpad3),
            OverlayField::Wetness => (KeyCode::Digit4, KeyCode::Numpad4),
            OverlayField::Snow => (KeyCode::Digit5, KeyCode::Numpad5),
            OverlayField::Cloud => (KeyCode::Digit6, KeyCode::Numpad6),
            OverlayField::Iron => (KeyCode::Digit7, KeyCode::Numpad7),
            OverlayField::Copper => (KeyCode::Digit8, KeyCode::Numpad8),
            // The last three the number row has, which is also why nothing else may be
            // added without a second way of selecting a field.
            OverlayField::Salt => (KeyCode::Digit9, KeyCode::Numpad9),
        }
    }

    /// That digit as text, for the legend — so the key that got you here is on
    /// screen and the rest of them are one guess away.
    fn key_label(self) -> &'static str {
        match self {
            OverlayField::Off => "0",
            OverlayField::Height => "1",
            OverlayField::Temperature => "2",
            OverlayField::Moisture => "3",
            OverlayField::Wetness => "4",
            OverlayField::Snow => "5",
            OverlayField::Cloud => "6",
            OverlayField::Iron => "7",
            OverlayField::Copper => "8",
            OverlayField::Salt => "9",
        }
    }

    /// The unit this field is measured in, for the legend.
    fn unit(self) -> &'static str {
        match self {
            OverlayField::Temperature => " C",
            _ => "",
        }
    }

    /// Whether the ramp is spent on the whole of the field or on the part of it that
    /// is actually on screen.
    ///
    /// **Fit against a screenful, not against the world** — the same lesson the tint's
    /// `strength` carries, and it is not a refinement but the difference between a
    /// usable overlay and a flat sheet of one colour. The first cut ranged height over
    /// 0..1 and temperature over the world's -20..30, and both came out featureless:
    /// a screen at gameplay zoom holds about 0.3 of the height range and four degrees
    /// of temperature.
    ///
    /// Which fields need it splits cleanly, and the split is a property of the fields
    /// rather than a preference:
    ///
    /// - **Smooth and slow** — height, temperature, moisture — vary over hundreds of
    ///   tiles, so a screen is a slice and the ramp has to follow it.
    /// - **Saturating** — wetness, snow, cloud — are 0 or 1 over most of the map by
    ///   construction. They already use their whole range, and fitting one would make
    ///   a dry screen's numerical noise look like weather.
    ///
    /// A recipe score is **not** fitted, on exactly the saturating fields' argument:
    /// it is zero over most of the world by construction — wrong kind, wrong biome —
    /// so it already uses its whole range, and fitting one would make an empty
    /// screen's numerical noise look like ore.
    fn fits_the_screen(self) -> bool {
        matches!(
            self,
            OverlayField::Height | OverlayField::Temperature | OverlayField::Moisture
        )
    }

    /// Temperature is the one field with a meaningful zero, so it is the one drawn on
    /// the diverging ramp.
    /// Which fields have a meaningful zero, and so get the diverging ramp.
    ///
    /// Temperature's is the freezing point. A recipe score's is `deposit_threshold` —
    /// where ground stops being ordinary and starts being worth digging — which is
    /// exactly the same shape of zero, and gives the same payoff: the line between
    /// prospective and not is the neutral band, visible without reading a number.
    fn diverging(self) -> bool {
        matches!(self, OverlayField::Temperature) || self.resource().is_some()
    }

    /// What this field's values run between, and how they are coloured.
    ///
    /// `None` for [`OverlayField::Off`], which is the whole of what "off" means here
    /// — there is no separate enabled flag to fall out of step with the field.
    ///
    /// `seen` is the spread actually on screen, and is ignored by the fields that do
    /// not fit to it. Absent — no world generated yet, no climate baked — every field
    /// falls back to its full extent, which is wrong-looking rather than broken.
    pub fn range(
        self,
        config: &InspectConfig,
        ground: &GroundConfig,
        plan: &WorldPlanConfig,
        seen: Option<(f32, f32)>,
    ) -> Option<OverlayRange> {
        if self == OverlayField::Off {
            return None;
        }

        let (mut low, mut high) = match (self.fits_the_screen(), seen) {
            (true, Some(seen)) => seen,
            // The full extent of each field: 0..1 for everything the world keeps on
            // the unit interval, and the configured window for the one in degrees.
            _ if self == OverlayField::Temperature => (
                config.temperature_low_celsius,
                config.temperature_high_celsius,
            ),
            _ => (0.0, 1.0),
        };

        // A ramp with no width divides by nothing and paints the screen one colour,
        // which is exactly the failure this is all here to avoid. A flat field is a
        // real thing to see — the floor is what lets you see that it *is* flat.
        let span = (high - low).max(config.min_span);
        let centre = 0.5 * (low + high);
        (low, high) = (centre - 0.5 * span, centre + 0.5 * span);

        // **The freezing point, read from the module that freezes things** rather than
        // restated as zero — that is what puts the neutral band exactly on the snow
        // line rather than near it.
        // **Read from the module that decides it** rather than restated — that is what
        // puts a neutral band exactly on the line rather than near it, for the snow
        // line and for the line between ground worth digging and ground that is not.
        let mid = match self {
            OverlayField::Temperature => ground.freezing_celsius,
            _ if self.resource().is_some() => plan.deposit_threshold,
            _ => 0.5 * (low + high),
        };

        if self.diverging() {
            // Equal arms about the midpoint, which is what a diverging ramp is for:
            // the two halves have to mean the same distance or "how far from
            // freezing" is not readable. It also keeps the midpoint inside the range
            // when the whole screen is on one side of it — a view entirely below
            // freezing shows as all-cool, which is the true answer.
            let reach = (mid - low).max(high - mid).max(0.5 * config.min_span);
            (low, high) = (mid - reach, mid + reach);
        }

        Some(OverlayRange {
            low,
            mid,
            high,
            diverging: self.diverging(),
            unit: self.unit(),
        })
    }
}

/// What a field's values run between, and which ramp reads them.
#[derive(Clone, Copy, Debug)]
pub struct OverlayRange {
    pub low: f32,
    /// Where the ramp's middle stop lands. For a sequential field this is simply
    /// halfway, so the normalization below is a plain linear map and the two cases
    /// stay one line of arithmetic rather than two branches.
    pub mid: f32,
    pub high: f32,
    pub diverging: bool,
    pub unit: &'static str,
}

impl OverlayRange {
    /// A raw value onto the ramp's 0..1, with `mid` pinned to the middle.
    ///
    /// Piecewise, because the two halves need not be the same width: the default
    /// temperature range runs -20 to 30 about a freezing point at 0, so the cool arm
    /// covers 20 degrees and the warm one 30, and both still fill their half of the
    /// ramp.
    ///
    /// **The shader transcribes this**, on the same terms `weather::cloud_density`
    /// and the dither's `snow_lying` are transcribed — the two are edited together.
    /// The Rust side is what the ctl reports a `position` from, and that reader does
    /// not exist on wasm; the shader's copy is what draws the map either way.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn normalize(&self, value: f32) -> f32 {
        let t = if value < self.mid {
            0.5 * (value - self.low) / (self.mid - self.low).max(f32::EPSILON)
        } else {
            0.5 + 0.5 * (value - self.mid) / (self.high - self.mid).max(f32::EPSILON)
        };
        t.clamp(0.0, 1.0)
    }

    /// One end of the range, or its middle.
    fn at(&self, end: LegendEnd) -> f32 {
        match end {
            LegendEnd::Low => self.low,
            LegendEnd::Mid => self.mid,
            LegendEnd::High => self.high,
        }
    }

    /// The colour this ramp gives a normalized position. The legend is built from
    /// this, so a swatch and the map under it cannot disagree.
    pub fn colour(&self, t: f32) -> Vec3 {
        ramp(
            if self.diverging {
                &DIVERGING
            } else {
                &SEQUENTIAL
            },
            t,
        )
    }
}

/// Three stops, interpolated. Clamped rather than wrapped, so a value off the end of
/// the range reads as the end rather than as the other pole.
fn ramp(stops: &[Vec3; 3], t: f32) -> Vec3 {
    let t = t.clamp(0.0, 1.0);
    if t < 0.5 {
        stops[0].lerp(stops[1], t * 2.0)
    } else {
        stops[1].lerp(stops[2], (t - 0.5) * 2.0)
    }
}

/// The knobs. Only the temperature range is really tunable — everything else the
/// overlay draws is on 0..1 by construction.
#[derive(Resource, Clone)]
pub struct InspectConfig {
    /// The ends of the temperature ramp, in degrees Celsius.
    ///
    /// Not the climate map's own quantization window (-40 to 60), which would spend
    /// most of the ramp on temperatures the world never reaches. At the default
    /// terrain the land runs about -9 to +16 before the day's swing and about -20 to
    /// +25 with it, so this is that with a little room either side.
    pub temperature_low_celsius: f32,
    pub temperature_high_celsius: f32,
    /// How much of the false colour to draw, against the world underneath.
    ///
    /// 1.0 by default and that is the right default: a half-transparent field map is
    /// a field map you cannot read the values off. Turn it down when what you want is
    /// *where* rather than *how much*.
    pub opacity: f32,
    /// How many swatches the legend's strip is cut into. Enough that it reads as a
    /// gradient, few enough that the nodes are not a layout cost.
    pub legend_steps: u32,
    /// The narrowest a fitted range is allowed to get, in the field's own units.
    ///
    /// Without a floor, a screen of genuinely flat ground would have its last two
    /// quantization steps stretched across the whole ramp and read as landscape. With
    /// one, flat looks flat. It is in field units and so means 0.06 of the height
    /// range *or* 6 degrees — which is only defensible because no field mixes units
    /// with another.
    pub min_span: f32,
    /// Samples along each side of the grid the fitted range is taken from.
    ///
    /// 24 is 576 samples of a screen ~480 tiles across, so one per 20 tiles. They are
    /// all array reads into maps that are already in memory, which is why this can
    /// happen every frame; a field that had to be *evaluated* per sample could not.
    pub fit_samples_per_side: u32,
    /// The share trimmed off each end of those samples before the range is taken.
    ///
    /// Min and max would let one deep-water tile in a highland view stretch the ramp
    /// over water nobody is looking at. Trimming a fortieth off each end costs a sort
    /// of 576 floats and makes the range about the ground actually in front of you.
    pub fit_trim: f32,
}

impl Default for InspectConfig {
    fn default() -> Self {
        Self {
            temperature_low_celsius: -20.0,
            temperature_high_celsius: 30.0,
            opacity: 1.0,
            legend_steps: 32,
            min_span: 0.06,
            fit_samples_per_side: 24,
            fit_trim: 0.025,
        }
    }
}

/// The field being drawn and the range it is being drawn over, this frame.
///
/// One resource, written once by [`sync_inspect_overlay`], and read by everything
/// that has to agree with the screen: the pass, the legend, and the ctl. Deriving the
/// range a second time anywhere would be a second answer to "what is on screen", and
/// with a fitted range that answer moves every time the camera does.
#[derive(Resource, Clone, Copy, Default)]
pub struct ActiveOverlay {
    pub field: OverlayField,
    pub range: Option<OverlayRange>,
}

/// Everything a field's value can be read out of, gathered once.
///
/// The ctl builds one of these too, so the number it reports is the number the fit
/// was taken from — and both are the CPU's own copy of the field rather than the
/// texture the shader samples, which is what makes an observation a cross-check
/// instead of a second look at the same texel.
pub struct FieldSources<'a> {
    pub world: Option<&'a WorldMap>,
    pub climate: Option<&'a ClimateMaps>,
    pub cover: Option<&'a GroundCover>,
    pub sky: Option<&'a SkySampler>,
    /// The prospectivity map, absent until its bake lands — which is what makes the
    /// three resource fields draw nothing rather than a wrong colour before then.
    pub prospect: Option<&'a ProspectMaps>,
    pub offset: TemperatureOffset,
}

impl FieldSources<'_> {
    /// The selected field at a tile.
    ///
    /// Every arm can come back `None`, and each absence means something real: no
    /// world generated there yet, no climate baked, no sky. A zero would read as
    /// "it is freezing here", which is why none is returned instead.
    pub fn value(&self, field: OverlayField, tile: Vec2) -> Option<f32> {
        match field {
            OverlayField::Off => None,
            OverlayField::Height => self.world?.height(tile.floor().as_ivec2()),
            OverlayField::Temperature => Some(self.climate?.at(tile).temperature(self.offset)),
            OverlayField::Moisture => Some(self.climate?.at(tile).humidity),
            OverlayField::Wetness => Some(self.cover?.at(tile).wetness),
            OverlayField::Snow => Some(self.cover?.at(tile).snow),
            OverlayField::Cloud => {
                let humidity = self.climate?.at(tile).humidity;
                Some(self.sky?.cloud_at(tile, humidity))
            }
            // The recipe score, read from the CPU's own copy of the baked map — a
            // cross-check against what the shader sampled, exactly as every other
            // field's observation is.
            OverlayField::Iron | OverlayField::Copper | OverlayField::Salt => self
                .prospect?
                .score_at(tile, field.resource().expect("a resource field")),
        }
    }

    /// The spread of the field over a rectangle of the world, trimmed at both ends.
    ///
    /// `None` when the field has no value here at all — an unbaked climate, say —
    /// which is what makes the range fall back to the field's full extent rather than
    /// to a range taken from nothing.
    pub fn spread(
        &self,
        field: OverlayField,
        centre: Vec2,
        half_extent: Vec2,
        config: &InspectConfig,
    ) -> Option<(f32, f32)> {
        let side = config.fit_samples_per_side.max(2);
        let mut samples: Vec<f32> = Vec::with_capacity((side * side) as usize);
        for y in 0..side {
            for x in 0..side {
                let step = Vec2::new(x as f32, y as f32) / (side - 1) as f32 * 2.0 - Vec2::ONE;
                if let Some(value) = self.value(field, centre + step * half_extent) {
                    samples.push(value);
                }
            }
        }
        if samples.is_empty() {
            return None;
        }

        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let trim = (samples.len() as f32 * config.fit_trim.clamp(0.0, 0.4)) as usize;
        Some((samples[trim], samples[samples.len() - 1 - trim]))
    }
}

/// Marks the legend, and remembers what it was built for. Comparing that against the
/// current field is the whole of "does this need rebuilding" — no change detection,
/// and it survives the legend being despawned with the session.
#[derive(Component)]
struct OverlayLegend(OverlayField);

/// Which end of the range a label prints.
///
/// The labels are marked rather than rebuilt because a *fitted* range moves with the
/// camera, so the numbers change on any frame you pan — where the field, and so the
/// swatches, change twice a session. Rebuilding thirty-odd nodes for two strings is
/// what `city_panel.rs` documents not doing.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
enum LegendEnd {
    Low,
    Mid,
    High,
}

pub struct InspectPlugin;

impl Plugin for InspectPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<InspectConfig>();
        // The mode is neither a knob nor world state but a *view* setting, so it
        // outlives a session like a knob: coming back to a world you were inspecting
        // and finding the overlay off would be the wrong surprise. It cannot leak
        // onto the menus, because the pass that draws it is gated on an overlay
        // component that only exists during gameplay.
        app.init_resource::<OverlayField>();
        app.init_resource::<ActiveOverlay>();
        app.add_systems(
            Update,
            // The fit reads the camera, so it has to be one chain: measure what is on
            // screen, tell the pass, then tell the legend the same thing.
            (
                select_overlay_field,
                sync_inspect_overlay,
                sync_overlay_legend,
            )
                .chain()
                .run_if(in_state(Screen::Gameplay)),
        );
    }
}

/// A digit selects its field. Nothing else on the keyboard is a number — WASD pans,
/// `+`/`-` zoom, Escape leaves — so the whole row was free.
///
/// Direct selection rather than the cycle this replaced: with seven fields a cycle
/// puts the one you want up to six presses away and offers no way back except round.
fn select_overlay_field(keys: Res<ButtonInput<KeyCode>>, mut field: ResMut<OverlayField>) {
    for candidate in OverlayField::ALL {
        let (row, numpad) = candidate.keys();
        if keys.just_pressed(row) || keys.just_pressed(numpad) {
            *field = candidate;
            return;
        }
    }
}

/// Measures the field over what is on screen, and hands the pass the range to draw
/// it over.
///
/// The camera's own `Transform` and `visible_half_extent`, which is the same pair the
/// extract uses to tell the shader where it is looking — so the rectangle sampled
/// here is the rectangle drawn there.
fn sync_inspect_overlay(
    field: Res<OverlayField>,
    config: Res<InspectConfig>,
    ground: Res<GroundConfig>,
    planet: Res<PlanetConfig>,
    sun: Res<Sun>,
    world: Option<Res<WorldMap>>,
    climate: Option<Res<ClimateMaps>>,
    cover: Option<Res<GroundCover>>,
    sky: Option<Res<SkySampler>>,
    prospect: Option<Res<ProspectMaps>>,
    plan: Res<WorldPlanConfig>,
    camera: Single<(&Camera, &Projection, &Transform), With<WorldCamera>>,
    mut active: ResMut<ActiveOverlay>,
    mut overlay: Single<&mut ScreenOverlay>,
) {
    let range = if *field == OverlayField::Off {
        None
    } else {
        let (camera, projection, transform) = *camera;
        let sources = FieldSources {
            world: world.as_deref(),
            climate: climate.as_deref(),
            cover: cover.as_deref(),
            sky: sky.as_deref(),
            prospect: prospect.as_deref(),
            offset: temperature_offset(&ground, &planet, &sun),
        };
        let seen = sources.spread(
            *field,
            tile_position_at(transform.translation.truncate()),
            visible_half_extent(camera, projection) / TILE_DISPLAY_SIZE.as_vec2(),
            &config,
        );
        field.range(&config, &ground, &plan, seen)
    };

    *active = ActiveOverlay {
        field: *field,
        range,
    };
    overlay.set_overlay(*field, range, config.opacity);
}

/// Builds the legend when the field changes, and keeps its numbers up to date while
/// it does not.
///
/// Two jobs because they happen at two rates: the structure — a title, a strip of
/// swatches, three labels — changes when you switch fields, and the numbers change
/// whenever the camera moves, because the range is fitted to what is on screen.
fn sync_overlay_legend(
    mut commands: Commands,
    active: Res<ActiveOverlay>,
    config: Res<InspectConfig>,
    legend: Query<(Entity, &OverlayLegend)>,
    mut labels: Query<(&LegendEnd, &mut Text)>,
) {
    // The legend goes with the session, so on re-entering gameplay there is none and
    // this rebuilds — which is why the comparison is against the *entity* rather than
    // against a remembered value.
    let built = legend.single().map(|(_, built)| built.0).ok();
    if built != Some(active.field) {
        for (entity, _) in &legend {
            commands.entity(entity).despawn();
        }
        if let Some(range) = active.range {
            spawn_legend(&mut commands, active.field, range, &config);
        }
        // The labels are spawned with their true strings, so there is nothing left to
        // write this frame — and the query above cannot see them yet anyway.
        return;
    }

    let Some(range) = active.range else {
        return;
    };
    for (end, mut text) in &mut labels {
        // Written only when it differs: touching a `Text` costs two full layout
        // passes, and a fitted range holds still whenever the camera does.
        let wanted = reading(range.at(*end), range.unit);
        if text.0 != wanted {
            text.0 = wanted;
        }
    }
}

/// The panel: what is being drawn, the ramp it is drawn with, and the ends of it.
///
/// Plain text on a flat dark panel, and deliberately *not* the outlined text
/// `city_panel.rs` needs — that trick exists because wood is a mid-tone with grain
/// running through it, so one flat ink is legible over part of a plank and lost over
/// the rest. A dark surface has no such problem, and this is a tool rather than the
/// game's furniture.
fn spawn_legend(
    commands: &mut Commands,
    field: OverlayField,
    range: OverlayRange,
    config: &InspectConfig,
) {
    let label = |text: String, size: f32, colour: Color| {
        (
            Text::new(text),
            TextFont::from_font_size(size),
            TextColor(colour),
        )
    };
    let ink = Color::srgb(1.0, 1.0, 1.0);
    let muted = Color::srgb(0.537, 0.529, 0.506); // #898781

    commands
        .spawn((
            OverlayLegend(field),
            DespawnOnExit(Screen::Gameplay),
            Node {
                position_type: PositionType::Absolute,
                left: px(16.0),
                bottom: px(16.0),
                width: px(220.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(px(10.0)),
                row_gap: px(6.0),
                border_radius: BorderRadius::all(px(6.0)),
                ..default()
            },
            // The dark chart surface, a little transparent so the map still shows
            // through the corner it covers.
            BackgroundColor(Color::srgba(0.102, 0.102, 0.098, 0.86)),
        ))
        .with_children(|panel| {
            // The key it is on, beside the name — so the one that got you here is on
            // screen and the other six are one guess away.
            panel.spawn(label(
                format!("[{}] {}", field.key_label(), field.label()),
                15.0,
                ink,
            ));

            // The strip, cut from the same `OverlayRange::colour` the shader
            // transcribes — so a swatch and the map under it cannot disagree about
            // what a value looks like.
            //
            // The hairline ring is not decoration: the sequential ramp's dark end is
            // #0d366b against a #1a1a19 panel, which is under 2:1, so without it the
            // strip has no visible right-hand end and reads as half a ramp.
            panel
                .spawn((
                    Node {
                        width: percent(100),
                        height: px(12.0),
                        flex_direction: FlexDirection::Row,
                        border: UiRect::all(px(1.0)),
                        ..default()
                    },
                    BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.25)),
                ))
                .with_children(|strip| {
                    let steps = config.legend_steps.max(2);
                    for step in 0..steps {
                        let t = step as f32 / (steps - 1) as f32;
                        let colour = range.colour(t);
                        strip.spawn((
                            Node {
                                flex_grow: 1.0,
                                height: percent(100),
                                ..default()
                            },
                            BackgroundColor(Color::srgb(colour.x, colour.y, colour.z)),
                        ));
                    }
                });

            // The ends, and the middle where it means something. Without these the
            // strip says "more is darker" and nothing else — which is exactly the
            // colour-alone failure the legend is here to prevent.
            panel
                .spawn(Node {
                    width: percent(100),
                    flex_direction: FlexDirection::Row,
                    justify_content: JustifyContent::SpaceBetween,
                    ..default()
                })
                .with_children(|row| {
                    let mut end = |which: LegendEnd, colour: Color| {
                        row.spawn((
                            which,
                            label(reading(range.at(which), range.unit), 12.0, colour),
                        ));
                    };
                    end(LegendEnd::Low, muted);
                    // The middle is only printed where it means something. On a
                    // sequential ramp it is just the average of the two ends, and a
                    // number that says nothing is worse than no number.
                    if range.diverging {
                        end(LegendEnd::Mid, ink);
                    }
                    end(LegendEnd::High, muted);
                });
        });
}

/// A range end, printed at the precision the field deserves: whole degrees for a
/// temperature, two places for something on 0..1.
fn reading(value: f32, unit: &str) -> String {
    if unit.is_empty() {
        format!("{value:.2}")
    } else {
        format!("{value:.0}{unit}")
    }
}

#[cfg(test)]
mod tests {
    /// The plan's knobs, for the fields whose neutral band is `deposit_threshold`.
    fn plan() -> WorldPlanConfig {
        WorldPlanConfig::default()
    }

    use super::*;

    /// Relative luminance, which is what "monotone in lightness" is measured on and
    /// the only reason a rainbow ramp is wrong.
    fn lightness(colour: Vec3) -> f32 {
        colour.dot(Vec3::new(0.2126, 0.7152, 0.0722))
    }

    /// The property that makes a sequential ramp readable: more is always darker,
    /// with no turn anywhere in it. A rainbow fails this at its yellow band, which is
    /// why the eye reads a boundary there.
    #[test]
    fn the_sequential_ramp_only_ever_darkens() {
        let mut previous = f32::MAX;
        for step in 0..=200 {
            let level = lightness(ramp(&SEQUENTIAL, step as f32 / 200.0));
            assert!(
                level <= previous + 1.0e-6,
                "the ramp brightened at {}",
                step as f32 / 200.0
            );
            previous = level;
        }
        // And it actually spans a usable range, rather than being monotone by being
        // nearly constant.
        assert!(lightness(SEQUENTIAL[0]) - lightness(SEQUENTIAL[2]) > 0.4);
    }

    /// A diverging ramp has to do the opposite: lightest in the middle and falling
    /// away either side, so distance from the midpoint is what the eye picks up.
    #[test]
    fn the_diverging_ramp_peaks_at_its_neutral_middle() {
        let middle = lightness(ramp(&DIVERGING, 0.5));
        assert!(middle > lightness(ramp(&DIVERGING, 0.0)) + 0.3);
        assert!(middle > lightness(ramp(&DIVERGING, 1.0)) + 0.3);

        // Both arms are monotone away from it, so nothing between the poles reads as
        // an edge.
        for step in 1..=100 {
            let t = step as f32 / 200.0;
            assert!(lightness(ramp(&DIVERGING, t)) >= lightness(ramp(&DIVERGING, t - 0.005)));
            assert!(
                lightness(ramp(&DIVERGING, 1.0 - t))
                    >= lightness(ramp(&DIVERGING, 1.0 - t + 0.005))
            );
        }
    }

    /// And its middle is a *neutral*, not a hue — a coloured midpoint would read as a
    /// third category with a false boundary either side of it.
    #[test]
    fn the_diverging_middle_is_grey_and_the_poles_are_not() {
        let middle = ramp(&DIVERGING, 0.5);
        let spread = |c: Vec3| c.max_element() - c.min_element();
        assert!(
            spread(middle) < 0.05,
            "the diverging midpoint is a colour: {middle:?}"
        );
        assert!(spread(ramp(&DIVERGING, 0.0)) > 0.2, "the cool pole is grey");
        assert!(spread(ramp(&DIVERGING, 1.0)) > 0.2, "the warm pole is grey");
    }

    /// The arithmetic must not assume the two arms are the same width, even though
    /// every range this module builds makes them so. Pinning the midpoint to the
    /// middle of the ramp is what puts the neutral band exactly on the freezing point
    /// rather than near it, and it has to hold however the ends were chosen.
    #[test]
    fn the_midpoint_lands_in_the_middle_of_the_ramp_on_an_uneven_range() {
        let range = OverlayRange {
            low: -20.0,
            mid: 0.0,
            high: 30.0,
            diverging: true,
            unit: " C",
        };

        assert!((range.normalize(0.0) - 0.5).abs() < 1.0e-6);
        assert_eq!(range.normalize(-20.0), 0.0);
        assert_eq!(range.normalize(30.0), 1.0);
        // Both arms fill their half, though one covers 20 degrees and the other 30.
        assert!((range.normalize(-10.0) - 0.25).abs() < 1.0e-6);
        assert!((range.normalize(15.0) - 0.75).abs() < 1.0e-6);

        // And off the ends it saturates rather than wrapping to the other pole, which
        // would draw the coldest ground the colour of the warmest.
        assert_eq!(range.normalize(-100.0), 0.0);
        assert_eq!(range.normalize(100.0), 1.0);
    }

    /// A field on 0..1 gets a plain linear map, because its midpoint is halfway by
    /// construction — the piecewise form has to degenerate rather than kink.
    #[test]
    fn a_unit_field_is_mapped_straight_through() {
        let range = OverlayField::Height
            .range(
                &InspectConfig::default(),
                &GroundConfig::default(),
                &plan(),
                Some((0.0, 1.0)),
            )
            .expect("height has a range");
        for step in 0..=20 {
            let value = step as f32 / 20.0;
            assert!((range.normalize(value) - value).abs() < 1.0e-6);
        }
        assert!(!range.diverging);
    }

    /// The fix for the first cut, as a property. A screenful of world is a *slice* of
    /// each field, so a ramp spent on the field's full extent is spent on nothing: the
    /// range has to follow what is on screen or the map comes out one flat colour.
    #[test]
    fn a_fitted_field_spends_its_ramp_on_what_is_on_screen() {
        let config = InspectConfig::default();
        let ground = GroundConfig::default();

        // A highland screenful: a third of the height range, nowhere near either end.
        let seen = (0.62, 0.79);
        let range = OverlayField::Height
            .range(&config, &ground, &plan(), Some(seen))
            .expect("height has a range");

        assert!(range.low <= seen.0 && seen.1 <= range.high);
        assert!(
            range.high - range.low < 0.25,
            "the ramp still spans {:.2}, which is most of the world rather than this \
             screen",
            range.high - range.low
        );
        // And the two ends of what is on screen land near the two ends of the ramp,
        // which is the whole point — before this they both landed in the same blue.
        assert!(range.normalize(seen.0) < 0.15);
        assert!(range.normalize(seen.1) > 0.85);
    }

    /// A saturating field keeps its full range whatever is on screen. Fitting one
    /// would stretch a dry screen's last two quantization steps across the ramp and
    /// draw numerical noise as weather.
    #[test]
    fn a_saturating_field_is_not_fitted() {
        let config = InspectConfig::default();
        let ground = GroundConfig::default();
        for field in [
            OverlayField::Wetness,
            OverlayField::Snow,
            OverlayField::Cloud,
        ] {
            let range = field
                .range(&config, &ground, &plan(), Some((0.0, 0.004)))
                .expect("a range");
            assert_eq!((range.low, range.high), (0.0, 1.0), "{field:?} was fitted");
        }
    }

    /// A flat screen must stay flat. Without a floor on the span the ramp would be
    /// spent on the last quantization step and a level plain would read as landscape.
    #[test]
    fn a_flat_screenful_does_not_have_its_noise_magnified() {
        let config = InspectConfig::default();
        let range = OverlayField::Height
            .range(
                &config,
                &GroundConfig::default(),
                &plan(),
                Some((0.5, 0.5001)),
            )
            .expect("height has a range");

        // To within an epsilon: the ends are rebuilt from a centre, so subtracting
        // them back loses the last bit or two of a number near 0.5.
        assert!(range.high - range.low >= config.min_span - 1.0e-6);
        // Everything on that screen lands in the middle of the ramp, so it is drawn
        // as the one colour it actually is.
        assert!((range.normalize(0.5) - 0.5).abs() < 0.02);
        assert!((range.normalize(0.5001) - 0.5).abs() < 0.02);
    }

    /// A view wholly below freezing has to draw as wholly cool — so the midpoint stays
    /// in the range even when nothing on screen reaches it, and the arms stay equal.
    #[test]
    fn a_frozen_screenful_still_straddles_the_freezing_point() {
        let range = OverlayField::Temperature
            .range(
                &InspectConfig::default(),
                &GroundConfig::default(),
                &plan(),
                Some((-14.0, -3.0)),
            )
            .expect("temperature has a range");

        assert_eq!(range.mid, 0.0);
        assert!((range.mid - range.low - (range.high - range.mid)).abs() < 1.0e-4);
        // Everything on screen is on the cool arm, and none of it reaches neutral.
        assert!(range.normalize(-3.0) < 0.5);
        assert!(range.normalize(-14.0) >= 0.0);
    }

    /// The temperature midpoint is the simulation's freezing point rather than a
    /// zero written here, so retuning one moves the other.
    #[test]
    fn the_temperature_ramp_turns_over_where_water_freezes() {
        let ground = GroundConfig {
            freezing_celsius: 4.0,
            ..default()
        };
        let range = OverlayField::Temperature
            .range(
                &InspectConfig::default(),
                &ground,
                &plan(),
                Some((-5.0, 12.0)),
            )
            .expect("temperature has a range");
        assert_eq!(range.mid, 4.0);
        assert!((range.normalize(4.0) - 0.5).abs() < 1.0e-6);
    }

    /// Every field is one key press away and no two share a key, or a mode is
    /// unreachable from the keyboard and only the ctl can select it.
    #[test]
    fn every_field_has_a_key_of_its_own() {
        let mut taken = Vec::new();
        for field in OverlayField::ALL {
            let (row, numpad) = field.keys();
            assert!(!taken.contains(&row), "{field:?} reuses a key");
            assert!(!taken.contains(&numpad), "{field:?} reuses a numpad key");
            taken.push(row);
            taken.push(numpad);
        }
    }

    /// **The key, the label and the discriminant are one number.** The shader
    /// switches on the discriminant and the player presses the digit, so if those
    /// ever parted company the overlay would draw a different field from the one the
    /// legend names — and nothing else in the crate would notice.
    #[test]
    fn the_key_a_field_is_on_is_its_own_discriminant() {
        // The whole number row, which is also the ceiling: a field past 9 would need a
        // second way of selecting one, and the enum has none.
        const DIGITS: [KeyCode; 10] = [
            KeyCode::Digit0,
            KeyCode::Digit1,
            KeyCode::Digit2,
            KeyCode::Digit3,
            KeyCode::Digit4,
            KeyCode::Digit5,
            KeyCode::Digit6,
            KeyCode::Digit7,
            KeyCode::Digit8,
            KeyCode::Digit9,
        ];
        assert!(
            OverlayField::ALL.len() <= DIGITS.len(),
            "there are more fields than there are digits to select them with"
        );
        for (index, field) in OverlayField::ALL.into_iter().enumerate() {
            assert_eq!(
                field as u32 as usize, index,
                "{field:?} is not in discriminant order in ALL",
            );
            assert_eq!(
                field.key_label(),
                index.to_string(),
                "{field:?} is labelled with a digit it is not on",
            );
            assert_eq!(
                field.keys().0,
                DIGITS[index],
                "{field:?} is on the wrong key"
            );
        }
    }

    /// Off is the absence of a range and nothing else. There is deliberately no
    /// separate enabled flag, so the two cannot disagree about whether anything is
    /// being drawn.
    #[test]
    fn off_is_the_absence_of_a_range() {
        let config = InspectConfig::default();
        let ground = GroundConfig::default();
        assert!(
            OverlayField::Off
                .range(&config, &ground, &plan(), None)
                .is_none()
        );
        for field in OverlayField::ALL
            .iter()
            .filter(|f| **f != OverlayField::Off)
        {
            // And every other field has one with or without a screenful to fit to,
            // so an unbaked climate is a wrong-looking map rather than a missing one.
            assert!(
                field.range(&config, &ground, &plan(), None).is_some(),
                "{field:?}"
            );
            assert!(
                field
                    .range(&config, &ground, &plan(), Some((0.2, 0.4)))
                    .is_some(),
                "{field:?}"
            );
        }
    }

    /// The default temperature range has to actually contain the world, or the map
    /// saturates at one end and the ramp is spent on nothing.
    #[test]
    fn the_default_temperature_range_spans_the_world_it_draws() {
        use crate::gameplay::terrain::TerrainConfig;

        let config = InspectConfig::default();
        let terrain = TerrainConfig::default();
        let sampler = terrain.sampler();

        let (mut low, mut high) = (f32::MAX, f32::MIN);
        for i in 0..4000u32 {
            let x = (i.wrapping_mul(2654435761) % 4000) as f32;
            let y = (i.wrapping_mul(40503) % 4000) as f32;
            let temperature = sampler.temperature(x, y);
            low = low.min(temperature);
            high = high.max(temperature);
        }

        // The normals alone, without the day's swing — which is why the configured
        // ends stand off them rather than matching.
        //
        // These are the *fallback* ends, used before a climate has been baked and
        // there is nothing on screen to fit to. They still have to contain the world:
        // a fallback that clipped would draw a saturated map on the frames before the
        // bake lands, which is exactly when someone is looking to see whether it did.
        assert!(
            config.temperature_low_celsius < low && high < config.temperature_high_celsius,
            "the world runs {low:.1}..{high:.1} C against a ramp of {:.0}..{:.0}",
            config.temperature_low_celsius,
            config.temperature_high_celsius,
        );
    }
}
