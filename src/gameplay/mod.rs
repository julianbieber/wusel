use bevy::{
    image::{ImageArrayLayout, ImageLoaderSettings},
    prelude::*,
    sprite_render::{AlphaMode2d, TileData, TilemapChunk, TilemapChunkTileData},
};

use crate::screens::Screen;

pub struct GameplayPlugin;

impl Plugin for GameplayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TerrainConfig>();
        app.add_systems(OnEnter(Screen::Gameplay), spawn_terrain_chunk);
    }
}

/// The six tiles of `assets/textures/terrain.png`, in atlas column order — the
/// discriminant *is* the tileset index, so the two can never drift apart.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u16)]
enum TerrainKind {
    Forest = 0,
    ShallowWater = 1,
    Grass = 2,
    Town = 3,
    Mountain = 4,
    DeepWater = 5,
}

/// Number of layers the terrain atlas is split into.
const TERRAIN_KIND_COUNT: u32 = 6;

impl TerrainKind {
    fn tileset_index(self) -> u16 {
        self as u16
    }

    /// Only these two kinds can be replaced by a Town.
    fn is_habitable(self) -> bool {
        matches!(self, TerrainKind::Forest | TerrainKind::Grass)
    }
}

/// Thresholds and noise scales that decide what a tile becomes.
#[derive(Resource, Clone)]
struct TerrainConfig {
    seed: u32,
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

const NOISE_OCTAVES: u32 = 5;
const NOISE_PERSISTENCE: f32 = 0.5;
const NOISE_LACUNARITY: f32 = 2.0;
/// Normalized fbm only spans about [0.35, 0.65] in practice — the octaves rarely
/// align and gradient noise peaks well below 1. Stretching it around the midpoint
/// makes the thresholds below mean what they say on a [0, 1] scale.
const NOISE_GAIN: f32 = 2.6;

/// Salts that give each field its own patch of the noise lattice.
const ELEVATION_SALT: u32 = 0x0000_0001;
const VEGETATION_SALT: u32 = 0x9e37_79b9;
const SETTLEMENT_SALT: u32 = 0x85eb_ca6b;

/// One fbm field with its own domain offset, so two fields sampled at the same
/// position are independent rather than two views of the same landscape.
struct NoiseField {
    offset: Vec2,
    scale: f32,
}

impl NoiseField {
    fn new(seed: u32, salt: u32, scale: f32) -> Self {
        let h = hash2(seed as i32, salt as i32);
        // Kept well under f32's precision cliff: fbm scales the domain up by the
        // lacunarity of the last octave, so a huge offset would quantize it.
        Self {
            offset: Vec2::new((h & 0xffff) as f32 / 64.0, (h >> 16) as f32 / 64.0),
            scale,
        }
    }

    /// Sample the field at a world position, remapped to [0, 1].
    fn sample(&self, x: f32, y: f32) -> f32 {
        let n = fbm(
            x * self.scale + self.offset.x,
            y * self.scale + self.offset.y,
            NOISE_OCTAVES,
            NOISE_PERSISTENCE,
            NOISE_LACUNARITY,
        );
        (0.5 + n * NOISE_GAIN * 0.5).clamp(0.0, 1.0)
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
    fn generate(config: &TerrainConfig, chunk_size: UVec2) -> Self {
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
                let x = column as f32 - margin as f32;
                let y = row as f32 - margin as f32;
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
                let x = column as f32 - margin as f32;
                let y = row as f32 - margin as f32;
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

/// Walks a chunk's tile indices and writes the resulting tileset index per tile.
fn generate_terrain(config: &TerrainConfig, chunk_size: UVec2) -> Vec<Option<TileData>> {
    let samples = TerrainSamples::generate(config, chunk_size);

    (0..chunk_size.element_product())
        .map(|i| {
            let x = (i % chunk_size.x) as i32;
            let y = (i / chunk_size.x) as i32;

            let base = samples.base_kind(x, y);
            let kind = if base.is_habitable() && is_town(&samples, config, x, y) {
                TerrainKind::Town
            } else {
                base
            };

            Some(TileData::from_tileset_index(kind.tileset_index()))
        })
        .collect()
}

fn spawn_terrain_chunk(
    mut commands: Commands,
    assets: Res<AssetServer>,
    config: Res<TerrainConfig>,
) {
    let chunk_size = UVec2::splat(64);
    // Native atlas tile size: drawing 8x8 art 1:1 keeps it crisp, since any
    // upscale would be resampled by the default linear sampler.
    let tile_display_size = UVec2::splat(8);

    commands.spawn((
        TilemapChunk {
            chunk_size,
            tile_display_size,
            // The atlas is a horizontal strip of TERRAIN_KIND_COUNT tiles, so the
            // array layer index is the atlas column — i.e. the TerrainKind.
            tileset: assets
                .load_builder()
                .with_settings(|s: &mut ImageLoaderSettings| {
                    s.array_layout = Some(ImageArrayLayout::GridCount {
                        columns: TERRAIN_KIND_COUNT,
                        rows: 1,
                    })
                })
                .load("textures/terrain.png"),
            alpha_mode: AlphaMode2d::Opaque,
        },
        TilemapChunkTileData(generate_terrain(&config, chunk_size)),
        DespawnOnExit(Screen::Gameplay),
    ));

    commands.spawn((Camera2d, DespawnOnExit(Screen::Gameplay)));
}

/// Hash function to generate pseudo-random gradients from integer coordinates.
/// No external crates — uses a simple bit-mixing hash.
fn hash2(x: i32, y: i32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x27d4eb2d);
    h ^= (y as u32).wrapping_mul(0x165667b1);
    h ^= h >> 15;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    h
}

/// Returns a pseudo-random unit gradient vector for a lattice point.
fn gradient(ix: i32, iy: i32) -> (f32, f32) {
    let h = hash2(ix, iy);
    // Map hash to an angle in [0, 2*pi)
    let angle = (h as f32 / u32::MAX as f32) * std::f32::consts::TAU;
    (angle.cos(), angle.sin())
}

/// Smoothstep-style fade curve (6t^5 - 15t^4 + 10t^3), as used in Perlin noise.
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// 2D gradient (Perlin-style) noise, returns values roughly in [-1, 1].
pub fn gradient_noise_2d(x: f32, y: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let sx = x - x0 as f32;
    let sy = y - y0 as f32;

    // Dot product of gradient and distance vector at each corner.
    let dot_grad = |ix: i32, iy: i32, dx: f32, dy: f32| -> f32 {
        let (gx, gy) = gradient(ix, iy);
        gx * dx + gy * dy
    };

    let n00 = dot_grad(x0, y0, sx, sy);
    let n10 = dot_grad(x1, y0, sx - 1.0, sy);
    let n01 = dot_grad(x0, y1, sx, sy - 1.0);
    let n11 = dot_grad(x1, y1, sx - 1.0, sy - 1.0);

    let u = fade(sx);
    let v = fade(sy);

    let nx0 = lerp(n00, n10, u);
    let nx1 = lerp(n01, n11, u);

    lerp(nx0, nx1, v)
}

/// Fractal Brownian Motion: sums multiple octaves of gradient noise
/// with increasing frequency and decreasing amplitude, then normalizes.
pub fn fbm(
    x: f32,
    y: f32,
    octaves: u32,
    persistence: f32, // amplitude multiplier per octave, e.g. 0.5
    lacunarity: f32,  // frequency multiplier per octave, e.g. 2.0
) -> f32 {
    let mut total = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amplitude = 0.0;

    for _ in 0..octaves {
        total += gradient_noise_2d(x * frequency, y * frequency) * amplitude;
        max_amplitude += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    // Normalize so output stays roughly in [-1, 1] regardless of octave count.
    total / max_amplitude
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHUNK: UVec2 = UVec2::splat(64);

    fn kinds(config: &TerrainConfig) -> Vec<u16> {
        generate_terrain(config, CHUNK)
            .into_iter()
            .map(|t| t.unwrap().tileset_index)
            .collect()
    }

    #[test]
    fn every_tile_gets_an_index_within_the_atlas() {
        let tiles = kinds(&TerrainConfig::default());
        assert_eq!(tiles.len() as u32, CHUNK.element_product());
        assert!(tiles.iter().all(|&i| (i as u32) < TERRAIN_KIND_COUNT));
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

    #[test]
    fn towns_only_replace_forest_or_grass() {
        let config = TerrainConfig::default();
        let samples = TerrainSamples::generate(&config, CHUNK);

        for (i, &index) in kinds(&config).iter().enumerate() {
            if index == TerrainKind::Town.tileset_index() {
                let (x, y) = ((i as u32 % CHUNK.x) as i32, (i as u32 / CHUNK.x) as i32);
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
            .filter(|&(_, &index)| index == TerrainKind::Town.tileset_index())
            .map(|(i, _)| ((i as u32 % CHUNK.x) as i32, (i as u32 / CHUNK.x) as i32))
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
                tiles.contains(&kind.tileset_index()),
                "no {kind:?} tiles in a {}x{} chunk",
                CHUNK.x,
                CHUNK.y
            );
        }
    }
}
