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

use bevy::prelude::*;

use crate::{
    gameplay::{
        screen::{ScreenOverlay, attach_screen_overlay},
        terrain::TerrainConfig,
    },
    screens::Screen,
};

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
}

impl Default for TerrainTintConfig {
    fn default() -> Self {
        Self {
            tint_low: 0.42,
            tint_high: 0.88,
            strength: 0.82,
        }
    }
}

pub struct TerrainTintPlugin;

impl Plugin for TerrainTintPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TerrainTintConfig>();
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::gameplay::terrain::height_byte;

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
