//! Turns a position in the world into a terrain kind.
//!
//! Everything here is a pure function of `(TerrainConfig, global tile position)`.
//! That is what lets the world be cut into chunks at all: a tile must come out
//! the same no matter which chunk happened to generate it, and no matter whether
//! its neighbours have been generated yet.

use bevy::prelude::*;

use crate::gameplay::noise::NoiseField;

/// The six tiles of `assets/textures/terrain.png`, in atlas column order — the
/// discriminant *is* the tileset index, so the two can never drift apart.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum TerrainKind {
    Forest = 0,
    ShallowWater = 1,
    Grass = 2,
    Town = 3,
    Mountain = 4,
    DeepWater = 5,
}

/// Number of layers the terrain atlas is split into.
pub const TERRAIN_KIND_COUNT: u32 = 6;

impl TerrainKind {
    pub fn tileset_index(self) -> u16 {
        self as u16
    }

    /// Only these two kinds can be replaced by a Town.
    fn is_habitable(self) -> bool {
        matches!(self, TerrainKind::Forest | TerrainKind::Grass)
    }
}

/// Thresholds and noise scales that decide what a tile becomes.
#[derive(Resource, Clone)]
pub struct TerrainConfig {
    pub seed: u32,
    elevation_scale: f32,
    vegetation_scale: f32,
    settlement_scale: f32,
    /// Elevation bands, in ascending order; anything above `lowland_max` is mountain.
    deep_water_max: f32,
    shallow_water_max: f32,
    lowland_max: f32,
    /// Vegetation at or above this turns a lowland tile from grass into forest.
    forest_threshold: f32,
    town_threshold: f32,
    town_coast_bonus: f32,
    town_min_spacing: u32,
    coast_radius: u32,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            seed: 0x5eed,
            elevation_scale: 0.04,
            vegetation_scale: 0.09,
            settlement_scale: 0.12,
            deep_water_max: 0.32,
            shallow_water_max: 0.42,
            lowland_max: 0.72,
            forest_threshold: 0.5,
            town_threshold: 0.62,
            town_coast_bonus: 0.06,
            town_min_spacing: 5,
            coast_radius: 2,
        }
    }
}

/// Salts that give each field its own patch of the noise lattice.
const ELEVATION_SALT: u32 = 0x0000_0001;
const VEGETATION_SALT: u32 = 0x9e37_79b9;
const SETTLEMENT_SALT: u32 = 0x85eb_ca6b;

/// Maps an elevation sample plus a vegetation sample to a kind. Elevation alone
/// decides water vs land vs mountain; vegetation only picks between kinds that
/// share the lowland band.
fn classify(config: &TerrainConfig, elevation: f32, vegetation: f32) -> TerrainKind {
    if elevation < config.deep_water_max {
        TerrainKind::DeepWater
    } else if elevation < config.shallow_water_max {
        TerrainKind::ShallowWater
    } else if elevation < config.lowland_max {
        if vegetation >= config.forest_threshold {
            TerrainKind::Forest
        } else {
            TerrainKind::Grass
        }
    } else {
        TerrainKind::Mountain
    }
}

/// The sampled fields for a chunk, padded on every side so that tiles at the
/// chunk border see the same neighbourhood they would in an unbounded world —
/// without that margin a tile's kind would depend on where the chunk was cut.
struct TerrainSamples {
    /// Padding in tiles around the chunk: `town_min_spacing + coast_radius`.
    margin: i32,
    stride: usize,
    base_kind: Vec<TerrainKind>,
    /// Settlement score, only valid within `town_min_spacing` of the chunk.
    score: Vec<f32>,
}

impl TerrainSamples {
    /// `origin` is the global tile coordinate of the chunk's lower-left tile;
    /// every sample below is taken in that global space, never chunk-locally.
    fn generate(config: &TerrainConfig, origin: IVec2, chunk_size: UVec2) -> Self {
        let coast_radius = config.coast_radius as i32;
        let margin = config.town_min_spacing as i32 + coast_radius;
        let stride = chunk_size.x as usize + 2 * margin as usize;
        let rows = chunk_size.y as usize + 2 * margin as usize;

        let elevation_field = NoiseField::new(config.seed, ELEVATION_SALT, config.elevation_scale);
        let vegetation_field =
            NoiseField::new(config.seed, VEGETATION_SALT, config.vegetation_scale);
        let settlement_field =
            NoiseField::new(config.seed, SETTLEMENT_SALT, config.settlement_scale);

        let mut elevation = Vec::with_capacity(stride * rows);
        let mut base_kind = Vec::with_capacity(stride * rows);
        for row in 0..rows {
            for column in 0..stride {
                let x = (origin.x + column as i32 - margin) as f32;
                let y = (origin.y + row as i32 - margin) as f32;
                let e = elevation_field.sample(x, y);
                let v = vegetation_field.sample(x, y);
                elevation.push(e);
                base_kind.push(classify(config, e, v));
            }
        }

        // The coast bonus needs an elevation window of `coast_radius`, so the
        // score is only defined on the region inset that far from the padding.
        let mut score = vec![f32::MIN; stride * rows];
        for row in coast_radius as usize..rows - coast_radius as usize {
            for column in coast_radius as usize..stride - coast_radius as usize {
                let x = (origin.x + column as i32 - margin) as f32;
                let y = (origin.y + row as i32 - margin) as f32;
                let mut s = settlement_field.sample(x, y);

                let coastal = (row as i32 - coast_radius..=row as i32 + coast_radius).any(|ny| {
                    (column as i32 - coast_radius..=column as i32 + coast_radius).any(|nx| {
                        elevation[ny as usize * stride + nx as usize] < config.shallow_water_max
                            && elevation[ny as usize * stride + nx as usize]
                                >= config.deep_water_max
                    })
                });
                if coastal {
                    s += config.town_coast_bonus;
                }

                score[row * stride + column] = s;
            }
        }

        Self {
            margin,
            stride,
            base_kind,
            score,
        }
    }

    /// Index by chunk-local tile coordinates, which may be negative inside the margin.
    fn index(&self, x: i32, y: i32) -> usize {
        (y + self.margin) as usize * self.stride + (x + self.margin) as usize
    }

    fn base_kind(&self, x: i32, y: i32) -> TerrainKind {
        self.base_kind[self.index(x, y)]
    }

    fn score(&self, x: i32, y: i32) -> f32 {
        self.score[self.index(x, y)]
    }
}

/// Decides whether a habitable tile becomes a Town: its settlement score must
/// clear the threshold and be the strict maximum among the habitable tiles
/// around it, which spreads towns out as isolated points instead of blobs.
fn is_town(samples: &TerrainSamples, config: &TerrainConfig, x: i32, y: i32) -> bool {
    let score = samples.score(x, y);
    if score < config.town_threshold {
        return false;
    }

    let spacing = config.town_min_spacing as i32;
    for ny in y - spacing..=y + spacing {
        for nx in x - spacing..=x + spacing {
            if (nx, ny) == (x, y) || !samples.base_kind(nx, ny).is_habitable() {
                continue;
            }
            if samples.score(nx, ny) >= score {
                return false;
            }
        }
    }
    true
}

/// Generates the kind of every tile in the chunk whose lower-left tile sits at
/// the global tile coordinate `origin`, in row-major order from that corner.
///
/// This is the only expensive call in the crate — roughly 4 ms for a 64x64
/// chunk — which is why [`crate::gameplay::world`] keeps it off the main thread
/// wherever it can.
pub fn generate_chunk(
    config: &TerrainConfig,
    origin: IVec2,
    chunk_size: UVec2,
) -> Box<[TerrainKind]> {
    let samples = TerrainSamples::generate(config, origin, chunk_size);

    (0..chunk_size.element_product())
        .map(|i| {
            let x = (i % chunk_size.x) as i32;
            let y = (i / chunk_size.x) as i32;

            let base = samples.base_kind(x, y);
            if base.is_habitable() && is_town(&samples, config, x, y) {
                TerrainKind::Town
            } else {
                base
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHUNK: UVec2 = UVec2::splat(64);
    /// Somewhere in the middle of the world rather than at the origin, so the
    /// tests exercise the same coordinate magnitudes the game actually uses.
    const ORIGIN: IVec2 = IVec2::new(2048, 2048);

    fn kinds(config: &TerrainConfig) -> Box<[TerrainKind]> {
        generate_chunk(config, ORIGIN, CHUNK)
    }

    fn position(i: usize) -> (i32, i32) {
        ((i as u32 % CHUNK.x) as i32, (i as u32 / CHUNK.x) as i32)
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

    #[test]
    fn towns_only_replace_forest_or_grass() {
        let config = TerrainConfig::default();
        let samples = TerrainSamples::generate(&config, ORIGIN, CHUNK);

        for (i, &kind) in kinds(&config).iter().enumerate() {
            if kind == TerrainKind::Town {
                let (x, y) = position(i);
                assert!(samples.base_kind(x, y).is_habitable());
            }
        }
    }

    #[test]
    fn towns_are_never_closer_together_than_the_minimum_spacing() {
        let config = TerrainConfig::default();
        let spacing = config.town_min_spacing as i32;
        let towns: Vec<(i32, i32)> = kinds(&config)
            .iter()
            .enumerate()
            .filter(|&(_, &kind)| kind == TerrainKind::Town)
            .map(|(i, _)| position(i))
            .collect();

        for (i, &(ax, ay)) in towns.iter().enumerate() {
            for &(bx, by) in &towns[i + 1..] {
                assert!(
                    (ax - bx).abs() > spacing || (ay - by).abs() > spacing,
                    "towns at ({ax},{ay}) and ({bx},{by}) are within {spacing}"
                );
            }
        }
    }

    /// The thresholds are only useful if the default config actually produces a
    /// mixed map — a single-biome world would pass every other test here.
    #[test]
    fn the_default_config_produces_every_kind() {
        let tiles = kinds(&TerrainConfig::default());
        for kind in [
            TerrainKind::Forest,
            TerrainKind::ShallowWater,
            TerrainKind::Grass,
            TerrainKind::Town,
            TerrainKind::Mountain,
            TerrainKind::DeepWater,
        ] {
            assert!(
                tiles.contains(&kind),
                "no {kind:?} tiles in a {}x{} chunk",
                CHUNK.x,
                CHUNK.y
            );
        }
    }
}
