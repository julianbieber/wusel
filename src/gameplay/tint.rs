//! Terrain tint: the rendered world's brightness scaled by how high the tile under
//! each fragment is.
//!
//! A tile's appearance was decided entirely by its `TerrainKind`, so every Grass
//! tile in the world was the same eight pixels and the height the sampler builds up
//! — continent, relief, ridged — was invisible except where it happened to cross a
//! `classify` band edge. This makes it visible *inside* a band.
//!
//! **The height is kept, not re-baked.** `classify` already computes every tile's
//! elevation and used to drop it on the floor, so [`crate::gameplay::terrain`]
//! returns it and [`crate::gameplay::world`] stores it. Sampling the world a second
//! time would cost ~9 core-seconds against the ~34 s the chunks themselves cost, and
//! would put a second answer to "how high is it here" in a crate whose determinism
//! tests rest on there being one. The world pays for it in memory instead: 16 MB of
//! heights beside the 16 MB of kinds, and 16 MB again on the GPU.
//!
//! Like the weather this is cosmetic: nothing here reads or writes `WorldMap`, so no
//! tile can depend on the shading.
//!
//! **This module is the ramp and nothing else.** The heightmap texture, the pipeline
//! and the fragment function live in [`crate::gameplay::screen`], the one
//! post-process pass — this decides how a height becomes a brightness and hands the
//! numbers over through [`ScreenOverlay::set_ramp`]. The sun that multiplies over the
//! top of the ramp is [`crate::gameplay::sun`]'s, on the same terms.
//!
//! The ramp is fake relief — high ground is brighter whether or not anything is
//! shining on it — and it is deliberately unchanged by the sun's arrival, because it
//! is the only cue at noon and through the whole night, when a real sun casts
//! nothing. The two can disagree: the ramp brightens a peak the sun may be behind.

use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{Extent3d, TextureDimension, TextureFormat},
    },
};

use crate::{
    gameplay::{
        noise::TilingNoiseField,
        screen::{ScreenOverlay, attach_screen_overlay},
        terrain::TerrainConfig,
    },
    screens::Screen,
};

/// Salt for the dither field, so the per-tile variation is not a third view of the
/// landscape it is drawn over.
const DITHER_SALT: u32 = 0xd17e_7a11;

/// Noise cells across one period of the dither map. With the default 128-tile period
/// this puts the coarsest patch at 16 tiles and, over three octaves, the finest at 4.
/// A power of two, because that is what lets every octave's lattice wrap exactly.
const DITHER_LATTICE_PERIOD: u32 = 8;

/// Three. A fourth would land inside two texels and only alias — the map is one texel
/// per tile and there is nothing finer than a tile to say.
const DITHER_OCTAVES: u32 = 3;

/// How the height is turned into a brightness.
///
/// A knob rather than world state, so like [`TerrainConfig`] it is built once and
/// outlives every session.
///
/// There is deliberately no resolution knob. The map is one texel per tile because
/// anything coarser stops the steps landing on tile boundaries, which is the whole
/// point — and because one texel per tile is what the generator already produces.
#[derive(Resource, Clone)]
pub struct TerrainTintConfig {
    /// Where the ramp bottoms out, and where it tops out. One ramp over the whole
    /// height range rather than one per band: a per-band ramp reverses at every band
    /// edge, which would draw a contour line along every coastline and treeline.
    ///
    /// `tint_low` is the water line, because that is where land starts; `tint_high`
    /// is the snow line, because everything above it is Snow and a ramp that ran on
    /// past it would spend half its range on one kind.
    pub tint_low: f32,
    pub tint_high: f32,
    /// Half the brightness range: a tile is drawn at `1 ± strength`. Bounded well
    /// under 1 so that no setting of these knobs can clip a tile to black or white,
    /// and small because the tileset is pixel art — this is meant to read as relief,
    /// not as a heatmap over the top of the art.
    ///
    /// **What matters is the spread across a screen, not across the world.** The ramp
    /// spans the whole height range, but a screenful holds only a slice of it, so the
    /// visible effect is a fraction of `2 * strength`.
    /// `the_default_ramp_measures_what_a_screenful_of_world_does` is how that is
    /// taken, and at these defaults it runs:
    ///
    /// ```text
    ///   screen at       height on it     brightness      spread
    ///   world centre    0.541..0.690    0.943..1.021       7.8%
    ///   lowland         0.424..0.655    0.882..1.003      12.1%
    ///   highland        0.424..0.529    0.882..0.937       5.5%
    /// ```
    ///
    /// So a slope reads as a gradient across the view rather than as per-tile relief:
    /// adjacent tiles differ by a few tenths of a percent, opposite sides of the
    /// screen by ~8%. Turning this up is the knob if that reads as too flat — it is
    /// linear in the spread, and 0.85 was blatant enough to measure a 0.79..0.86
    /// darkening against an untinted capture, which is what confirmed the pass draws
    /// what it should.
    pub strength: f32,
    /// How much world one repeat of the dither map covers, in tiles.
    ///
    /// 128 is a screenful at scale 1, so the repeat is never visible twice over at
    /// once. If a whole region sitting at partial coverage ever reads as a pattern,
    /// this goes **up** rather than the map gaining a second octave — the finest
    /// octave is already a tile wide and there is nothing below a tile to say.
    pub dither_period_tiles: u32,
    /// Texels along each side of the map. One per tile, and there is no reason for it
    /// to be anything else: the map exists to give each *tile* its own number.
    pub dither_texels_per_side: u32,
}

impl Default for TerrainTintConfig {
    fn default() -> Self {
        Self {
            tint_low: 0.42,
            tint_high: 0.88,
            strength: 0.82,
            dither_period_tiles: 128,
            dither_texels_per_side: 128,
        }
    }
}

/// One tiling period of a smooth per-tile field, repeated over the world.
///
/// **Smooth rather than white noise, and that is the whole of it.** Thresholding a
/// correlated field gives coherent patches that shrink from their *edges* as the
/// coverage falls, which is what melting snow does; white noise would give
/// salt-and-pepper that dissolves uniformly everywhere at once.
///
/// This is the noise half of gh-13 — a per-tile number so identical tiles do not
/// repeat exactly — arriving early and for a different reason. It lives here because
/// this is the module that decides how a smooth quantity becomes a *tile*, and it is
/// a knob rather than world state: baked once at startup and outliving every session,
/// like the configs.
#[derive(Resource, Clone)]
pub struct GroundDither(
    /// Read by `screen.rs`, which binds it so the pass can threshold the snow cover
    /// tile by tile. A handle and nothing else, like the ground's own two: the render
    /// world has no use for anything a map is made of.
    pub(super) Handle<Image>,
);

impl ExtractResource for GroundDither {
    type Source = Self;

    fn extract_resource(source: &Self) -> Self {
        source.clone()
    }
}

pub struct TerrainTintPlugin;

impl Plugin for TerrainTintPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TerrainTintConfig>();
        app.add_plugins(ExtractResourcePlugin::<GroundDither>::default());
        // At `Startup` rather than on entering gameplay, because the dither is not
        // world state: it is 16 KB built once, and a session that rebuilt it would
        // only be making the same map again. `TerrainConfig` is a knob too, so its
        // seed is already there.
        app.add_systems(Startup, bake_ground_dither);
        // The ramp is config: written once when the overlay appears and never again,
        // unlike the sun and the sky, which move. Ordered after the attach because
        // there is nothing to write into before it — a system is usable as an
        // ordering label wherever its parameter types are visible, and this one's
        // are.
        app.add_systems(
            OnEnter(Screen::Gameplay),
            sync_tint_ramp.after(attach_screen_overlay),
        );
    }
}

fn sync_tint_ramp(
    terrain: Res<TerrainConfig>,
    config: Res<TerrainTintConfig>,
    mut overlay: Single<&mut ScreenOverlay>,
) {
    overlay.set_ramp(&terrain, &config);
}

/// One period of the dither field, one texel per tile.
///
/// Cheap enough to build on the main thread at startup — 16 K samples of a
/// three-octave field, against the weather's 260 K — so it needs none of the task
/// machinery the climate and the sky bakes have.
fn bake_ground_dither(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    terrain: Res<TerrainConfig>,
    config: Res<TerrainTintConfig>,
) {
    let side = config.dither_texels_per_side.max(1);
    let field = TilingNoiseField::new(
        terrain.seed,
        DITHER_SALT,
        DITHER_LATTICE_PERIOD,
        DITHER_OCTAVES,
    );
    let cells_per_texel = DITHER_LATTICE_PERIOD as f32 / side as f32;

    let mut texels = Vec::with_capacity((side * side) as usize);
    for y in 0..side {
        for x in 0..side {
            let cell = Vec2::new(x as f32, y as f32) * cells_per_texel;
            texels.push((field.sample(cell.x, cell.y).clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }

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
    // **Nearest**, unlike every other map in the crate, and repeated. One texel is
    // one tile and the point of the whole map is that a tile gets *its own* number —
    // filtering would blend it with its neighbours' and put a snow edge inside a tile.
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        min_filter: ImageFilterMode::Nearest,
        mag_filter: ImageFilterMode::Nearest,
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        ..default()
    });

    commands.insert_resource(GroundDither(images.add(image)));
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::gameplay::{ground::GroundConfig, terrain::height_byte};

    /// How much of a tile's snow is actually drawn, given its own dither value.
    ///
    /// **The shader's arithmetic, transcribed** — `assets/shaders/screen.wgsl` does
    /// this per fragment, and the two are edited together, on the same terms
    /// `sun.rs`'s `shadow_at` transcribes the occlusion test. It is here so the
    /// property that matters can be checked without a GPU.
    ///
    /// The threshold is squeezed into `softness..1 - softness` rather than being the
    /// dither value itself, and that is what makes the endpoints exact: full coverage
    /// snows every tile and no coverage snows none, whatever number the tile drew.
    fn snow_lying(coverage: f32, dither: f32, softness: f32) -> f32 {
        let softness = softness.clamp(0.0, 0.5);
        let threshold = softness + dither.clamp(0.0, 1.0) * (1.0 - 2.0 * softness);
        smoothstep(threshold - softness, threshold + softness, coverage)
    }

    fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
        if edge1 <= edge0 {
            return if x < edge0 { 0.0 } else { 1.0 };
        }
        let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// A screenful in tiles, taken from a real extract: a half extent of
    /// 59.75 x 72.625 tiles at scale 1.
    const SCREEN_TILES: UVec2 = UVec2::new(120, 145);

    /// What the ramp does to a screenful of world, which is the only thing a player
    /// ever sees at once.
    ///
    /// This is the measurement the defaults were tuned against, and the number that
    /// matters is the *spread within a screen*, not across the world: a ramp can span
    /// the whole height range and still be invisible, because a screen holds only a
    /// slice of it.
    ///
    /// `cargo test --release -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, not a check"]
    fn the_default_ramp_measures_what_a_screenful_of_world_does() {
        let terrain = TerrainConfig::default();
        let config = TerrainTintConfig::default();
        let sampler = terrain.sampler();

        let brightness = |height: f32| {
            let ramp =
                ((height - config.tint_low) / (config.tint_high - config.tint_low)).clamp(0.0, 1.0);
            1.0 + config.strength * (ramp * 2.0 - 1.0)
        };

        println!("\nscreen at        height range      brightness range   spread");
        for (label, centre) in [
            ("world centre", UVec2::splat(2048)),
            ("lowland     ", UVec2::new(1024, 3072)),
            ("highland    ", UVec2::new(3072, 1024)),
        ] {
            let (mut low, mut high) = (f32::MAX, f32::MIN);
            let mut land = 0u32;
            for y in 0..SCREEN_TILES.y {
                for x in 0..SCREEN_TILES.x {
                    let tile = centre + UVec2::new(x, y) - SCREEN_TILES / 2;
                    let height =
                        height_byte(sampler.elevation(tile.x as f32, tile.y as f32)) as f32 / 255.0;
                    // Water is passed through, so it is not part of what the ramp has
                    // to work with.
                    if height <= terrain.shallow_water_max {
                        continue;
                    }
                    land += 1;
                    low = low.min(height);
                    high = high.max(height);
                }
            }

            if land == 0 {
                println!("{label}     all water");
                continue;
            }
            println!(
                "{label}     {low:.3}..{high:.3}     {:.3}..{:.3}      {:.1}%",
                brightness(low),
                brightness(high),
                (brightness(high) - brightness(low)) * 100.0,
            );
        }
        println!();
    }

    /// Brightness stays within `1 ± strength`, so no setting of the knobs can clip
    /// a tile to black or to white.
    #[test]
    fn the_default_ramp_cannot_clip_a_tile_to_black_or_white() {
        let config = TerrainTintConfig::default();
        assert!(config.strength > 0.0, "a zero ramp would shade nothing");
        assert!(
            config.strength < 1.0,
            "a strength of {} would drive a tile to black",
            config.strength
        );
    }

    /// The guard the biome's cover dither has, applied to this one: the threshold is
    /// *per tile*, so nothing about it may be allowed to leak into a region that is
    /// wholly covered or wholly bare. Full coverage snows every tile and no coverage
    /// snows none, whatever number the tile drew.
    ///
    /// This is the property that makes the squeeze into `softness..1 - softness`
    /// worth doing rather than thresholding against the raw dither value — with the
    /// raw value, any tile whose dither fell under the softness would be faintly
    /// snowed on a bare summer afternoon.
    #[test]
    fn the_dither_leaves_full_and_empty_coverage_alone() {
        let softness = GroundConfig::default().snow_dither_softness;
        for i in 0..=255u32 {
            let dither = i as f32 / 255.0;
            assert_eq!(
                snow_lying(0.0, dither, softness),
                0.0,
                "a tile with dither {dither} was snowed at zero coverage"
            );
            assert_eq!(
                snow_lying(1.0, dither, softness),
                1.0,
                "a tile with dither {dither} was bare at full coverage"
            );
        }
    }

    /// And the other half: in between it must *actually* dither, or the map is dead
    /// weight and a snowfield arrives everywhere at once. What makes a melting field
    /// shrink from its edges is that the tiles do not all cross together.
    #[test]
    fn the_dither_splits_a_partly_covered_world_tile_by_tile() {
        let softness = GroundConfig::default().snow_dither_softness;
        let drawn = (0..=255u32)
            .filter(|i| snow_lying(0.5, *i as f32 / 255.0, softness) > 0.5)
            .count();
        assert!(
            (32..224).contains(&drawn),
            "at half coverage {drawn}/256 dither values drew snow, which is not a split"
        );

        // And it is monotone in the coverage, so a field only ever grows as it snows
        // and only ever shrinks as it melts.
        let dither = 0.5;
        let mut previous = 0.0;
        for i in 0..=20u32 {
            let lying = snow_lying(i as f32 / 20.0, dither, softness);
            assert!(lying >= previous, "the field shrank as the coverage rose");
            previous = lying;
        }
    }

    /// The ramp has to span land the world actually produces, and to run the right
    /// way up — the shader divides by `tint_high - tint_low`.
    #[test]
    fn the_default_ramp_spans_land_the_world_actually_has() {
        let terrain = TerrainConfig::default();
        let config = TerrainTintConfig::default();

        assert!(
            terrain.deep_water_max <= config.tint_low,
            "the ramp starts under the sea, where nothing is shaded anyway"
        );
        assert!(
            config.tint_low < config.tint_high,
            "the ramp has to rise with height"
        );
        assert!(
            config.tint_high <= 1.0,
            "elevation is clamped to 1.0, so a higher top is range the world never reaches"
        );
    }
}
