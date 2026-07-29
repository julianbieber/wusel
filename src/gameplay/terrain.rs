//! Turns a position in the world into a terrain kind.
//!
//! Everything here is a pure function of `(TerrainConfig, global tile position)`,
//! and — since cities and roads moved out to [`crate::gameplay::plan`] — a pure
//! function of *that tile alone*. No rule here looks at a neighbour any more, so
//! a chunk needs no padding and a tile cannot depend on where the chunk boundary
//! fell.

use bevy::prelude::*;

use crate::gameplay::noise::NoiseField;

/// The seven tiles of `assets/textures/terrain.png`, in atlas column order — the
/// discriminant *is* the tileset index, so the two can never drift apart.
///
/// `Town` and `Road` are never produced here: they are stamped over the base
/// terrain once the whole world exists, by [`crate::gameplay::plan`].
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
}

/// Number of layers the terrain atlas is split into.
pub const TERRAIN_KIND_COUNT: u32 = 7;

impl TerrainKind {
    pub fn tileset_index(self) -> u16 {
        self as u16
    }

    /// Only these two kinds can be built on — a city is clipped by coast and
    /// mountain rather than paving them.
    pub fn is_habitable(self) -> bool {
        matches!(self, TerrainKind::Forest | TerrainKind::Grass)
    }

    /// What a road may never cross.
    pub fn is_water(self) -> bool {
        matches!(self, TerrainKind::ShallowWater | TerrainKind::DeepWater)
    }
}

/// Thresholds and noise scales that decide what a tile becomes.
///
/// The settlement figures are not read here at all — nothing in this module
/// samples that field. They live here because they describe the same landscape
/// as the rest, and [`crate::gameplay::city`] is their only reader.
#[derive(Resource, Clone)]
pub struct TerrainConfig {
    pub seed: u32,
    pub elevation_scale: f32,
    pub vegetation_scale: f32,
    pub settlement_scale: f32,
    /// Elevation bands, in ascending order; anything above `lowland_max` is mountain.
    pub deep_water_max: f32,
    pub shallow_water_max: f32,
    pub lowland_max: f32,
    /// Vegetation at or above this turns a lowland tile from grass into forest.
    pub forest_threshold: f32,
    /// A candidate city site must clear this settlement score to be founded.
    pub town_threshold: f32,
    pub town_coast_bonus: f32,
    pub coast_radius: u32,
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
            coast_radius: 2,
        }
    }
}

/// Salts that give each field its own patch of the noise lattice.
const ELEVATION_SALT: u32 = 0x0000_0001;
const VEGETATION_SALT: u32 = 0x9e37_79b9;
const SETTLEMENT_SALT: u32 = 0x85eb_ca6b;

impl TerrainConfig {
    /// The elevation field, which the road router costs its steps against —
    /// `WorldMap` only records which band a tile fell in, not how high it is.
    pub fn elevation_field(&self) -> NoiseField {
        NoiseField::new(self.seed, ELEVATION_SALT, self.elevation_scale)
    }

    pub fn vegetation_field(&self) -> NoiseField {
        NoiseField::new(self.seed, VEGETATION_SALT, self.vegetation_scale)
    }

    /// What makes one habitable tile a likelier city site than another.
    pub fn settlement_field(&self) -> NoiseField {
        NoiseField::new(self.seed, SETTLEMENT_SALT, self.settlement_scale)
    }
}

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

/// Generates the kind of every tile in the chunk whose lower-left tile sits at
/// the global tile coordinate `origin`, in row-major order from that corner.
///
/// Every sample is taken in global tile space, never chunk-locally — that, and
/// the fact that no rule here reads a neighbouring tile, is what lets the world
/// be cut into chunks at all.
///
/// This is the only expensive call in the crate — roughly 4 ms for a 64x64
/// chunk — which is why [`crate::gameplay::world`] keeps it off the main thread
/// wherever it can.
pub fn generate_chunk(
    config: &TerrainConfig,
    origin: IVec2,
    chunk_size: UVec2,
) -> Box<[TerrainKind]> {
    let elevation_field = config.elevation_field();
    let vegetation_field = config.vegetation_field();

    (0..chunk_size.element_product())
        .map(|i| {
            let x = (origin.x + (i % chunk_size.x) as i32) as f32;
            let y = (origin.y + (i / chunk_size.x) as i32) as f32;
            classify(
                config,
                elevation_field.sample(x, y),
                vegetation_field.sample(x, y),
            )
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
    #[test]
    fn the_default_config_produces_every_base_kind() {
        let tiles = kinds(&TerrainConfig::default());
        for kind in [
            TerrainKind::Forest,
            TerrainKind::ShallowWater,
            TerrainKind::Grass,
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

    /// The two stamped kinds belong to the plan, not to the terrain — if one
    /// ever came out of here, a chunk's contents would depend on its neighbours
    /// again.
    #[test]
    fn the_terrain_never_produces_a_town_or_a_road() {
        let tiles = kinds(&TerrainConfig::default());
        assert!(!tiles.contains(&TerrainKind::Town));
        assert!(!tiles.contains(&TerrainKind::Road));
    }
}
