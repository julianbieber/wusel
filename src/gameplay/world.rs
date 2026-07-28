//! The world: a fixed 64x64 grid of chunks, and the rules for which parts of it
//! exist as entities at any moment.
//!
//! Two separate things are "loaded" here and it matters that they are separate:
//!
//! * **Tile data** ([`WorldMap`]) is generated for the whole world and kept
//!   forever. One byte per tile, so all 4096 chunks cost ~16 MB. A background
//!   pass fills it in off the main thread.
//! * **Chunk entities** exist only near the camera. Each one costs a mesh, a
//!   material and a per-chunk index image, which is why the whole world cannot
//!   be resident — 4096 of them would be hundreds of megabytes.
//!
//! So the world is finite and you can walk to its edge, but you are never
//! holding more of it than you can see.

use bevy::{
    image::{ImageArrayLayout, ImageLoaderSettings},
    prelude::*,
    sprite_render::{AlphaMode2d, TileData, TilemapChunk, TilemapChunkTileData},
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};

use crate::{
    camera::WorldCamera,
    gameplay::terrain::{TERRAIN_KIND_COUNT, TerrainConfig, TerrainKind, generate_chunk},
    screens::Screen,
};

/// Tiles along each axis of one chunk.
pub const CHUNK_SIZE: UVec2 = UVec2::splat(64);
/// Pixels each tile covers on screen. This is the atlas' native tile size:
/// drawing 8x8 art 1:1 keeps it crisp, since any upscale would be resampled.
pub const TILE_DISPLAY_SIZE: UVec2 = UVec2::splat(8);
/// Chunks along each axis of the world — 64x64 chunks is 4096x4096 tiles.
pub const WORLD_CHUNKS: UVec2 = UVec2::splat(64);

/// Chunks this far from the camera's own chunk have entities; beyond that the
/// world is data only. Seven chunks across is 3584 pixels, so this covers any
/// window we are likely to be run in with a chunk to spare on each side.
const RESIDENT_RADIUS: i32 = 3;

/// A chunk takes roughly 4 ms to generate, so this is the worst-case stall the
/// streamer may add to a frame when the camera outruns the background pass.
const MAX_BLOCKING_GENERATIONS_PER_FRAME: usize = 2;

/// Marks a chunk entity with the chunk it is showing. Also the record of which
/// chunks are resident — deriving that from the entities rather than from a
/// separate resource means the two cannot disagree.
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
struct ChunkCoord(UVec2);

/// The tileset handle, loaded once. Every chunk entity shares it.
#[derive(Resource)]
struct TerrainTileset(Handle<Image>);

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TerrainConfig>();
        app.init_resource::<WorldMap>();
        app.init_resource::<BackgroundGeneration>();
        app.add_systems(Startup, load_tileset);
        app.add_systems(OnEnter(Screen::Gameplay), spawn_initial_chunks);
        app.add_systems(
            Update,
            (
                // Runs in the menus too, so that by the time Play is pressed the
                // world is already mostly generated.
                drive_background_generation,
                stream_chunks_around_camera.run_if(in_state(Screen::Gameplay)),
            ),
        );
    }
}

// -- Coordinates ------------------------------------------------------------
//
// Three spaces are in play: chunk coordinates (0..WORLD_CHUNKS), global tile
// coordinates (0..WORLD_CHUNKS * CHUNK_SIZE) which is what the noise is sampled
// in, and world space in pixels. The world is centred on the origin so that
// world-space coordinates stay small enough for f32 to be comfortable.

/// Half the world's extent in world space. The world spans `-this ..= this`.
pub fn world_half_extent() -> Vec2 {
    (WORLD_CHUNKS * CHUNK_SIZE * TILE_DISPLAY_SIZE).as_vec2() / 2.0
}

fn chunk_count() -> usize {
    WORLD_CHUNKS.element_product() as usize
}

fn chunk_index(coord: UVec2) -> usize {
    (coord.y * WORLD_CHUNKS.x + coord.x) as usize
}

fn chunk_coord(index: usize) -> UVec2 {
    let index = index as u32;
    UVec2::new(index % WORLD_CHUNKS.x, index / WORLD_CHUNKS.x)
}

/// The global tile coordinate of a chunk's lower-left tile.
fn chunk_origin_tiles(coord: UVec2) -> IVec2 {
    (coord * CHUNK_SIZE).as_ivec2()
}

/// Where a chunk sits in world space. `TilemapChunk` meshes are centred on their
/// transform, so this is the chunk's centre rather than its corner.
fn chunk_translation(coord: UVec2) -> Vec2 {
    let centre_tile = (coord * CHUNK_SIZE).as_vec2() + CHUNK_SIZE.as_vec2() / 2.0;
    centre_tile * TILE_DISPLAY_SIZE.as_vec2() - world_half_extent()
}

/// Which chunk a world-space position falls in, clamped to the world so that a
/// camera sitting exactly on the edge still resolves to a real chunk.
fn chunk_at(position: Vec2) -> UVec2 {
    let tile = (position + world_half_extent()) / TILE_DISPLAY_SIZE.as_vec2();
    (tile / CHUNK_SIZE.as_vec2())
        .floor()
        .as_ivec2()
        .clamp(IVec2::ZERO, (WORLD_CHUNKS - UVec2::ONE).as_ivec2())
        .as_uvec2()
}

/// Chebyshev distance, which is the shape the resident region has.
fn chunk_distance(a: UVec2, b: UVec2) -> i32 {
    let d = a.as_ivec2() - b.as_ivec2();
    d.x.abs().max(d.y.abs())
}

/// The chunks within `radius` of `centre`, skipping anything off the world.
fn chunks_within(centre: UVec2, radius: i32) -> impl Iterator<Item = UVec2> {
    let centre = centre.as_ivec2();
    let last = (WORLD_CHUNKS - UVec2::ONE).as_ivec2();
    ((centre.y - radius).max(0)..=(centre.y + radius).min(last.y)).flat_map(move |y| {
        ((centre.x - radius).max(0)..=(centre.x + radius).min(last.x))
            .map(move |x| UVec2::new(x as u32, y as u32))
    })
}

// -- Tile data --------------------------------------------------------------

/// Every tile in the world, or `None` for chunks the background pass has not
/// reached yet. One `TerrainKind` per tile, so this is ~16 MB in full.
#[derive(Resource)]
pub struct WorldMap {
    chunks: Vec<Option<Box<[TerrainKind]>>>,
}

impl Default for WorldMap {
    fn default() -> Self {
        Self {
            chunks: vec![None; chunk_count()],
        }
    }
}

impl WorldMap {
    fn get(&self, coord: UVec2) -> Option<&[TerrainKind]> {
        self.chunks[chunk_index(coord)].as_deref()
    }

    fn insert(&mut self, coord: UVec2, tiles: Box<[TerrainKind]>) {
        self.chunks[chunk_index(coord)] = Some(tiles);
    }

    /// Generates a chunk on the calling thread. Only for chunks that are needed
    /// this frame — everything else should come from the background pass.
    fn generate_blocking(&mut self, config: &TerrainConfig, coord: UVec2) {
        let tiles = generate_chunk(config, chunk_origin_tiles(coord), CHUNK_SIZE);
        self.insert(coord, tiles);
    }
}

/// The background pass that fills in the rest of [`WorldMap`].
#[derive(Resource)]
struct BackgroundGeneration {
    /// Chunks still to generate, ordered so that `pop` yields the ones nearest
    /// the middle of the world — which is where the camera starts — first.
    pending: Vec<UVec2>,
    in_flight: Vec<Task<(UVec2, Box<[TerrainKind]>)>>,
}

impl Default for BackgroundGeneration {
    fn default() -> Self {
        let centre = WORLD_CHUNKS / 2;
        let mut pending: Vec<UVec2> = (0..chunk_count()).map(chunk_coord).collect();
        pending.sort_by_key(|&coord| std::cmp::Reverse(chunk_distance(coord, centre)));
        Self {
            pending,
            in_flight: Vec::new(),
        }
    }
}

fn drive_background_generation(
    mut generation: ResMut<BackgroundGeneration>,
    mut map: ResMut<WorldMap>,
    config: Res<TerrainConfig>,
) {
    generation
        .in_flight
        .retain_mut(|task| match block_on(poll_once(task)) {
            Some((coord, tiles)) => {
                map.insert(coord, tiles);
                false
            }
            None => true,
        });

    let pool = AsyncComputeTaskPool::get();
    // One task per worker thread: the queue is thousands of chunks long, so
    // there is no point holding more than the pool can actually run at once.
    let in_flight_limit = pool.thread_num().max(1);
    while generation.in_flight.len() < in_flight_limit {
        let Some(coord) = generation.pending.pop() else {
            break;
        };
        // The streamer may have generated this one already while the camera was
        // sitting on it.
        if map.get(coord).is_some() {
            continue;
        }
        let config = config.clone();
        let task = pool.spawn(async move {
            (
                coord,
                generate_chunk(&config, chunk_origin_tiles(coord), CHUNK_SIZE),
            )
        });
        generation.in_flight.push(task);
    }
}

// -- Chunk entities ---------------------------------------------------------

fn load_tileset(mut commands: Commands, assets: Res<AssetServer>) {
    commands.insert_resource(TerrainTileset(
        assets
            .load_builder()
            .with_settings(|settings: &mut ImageLoaderSettings| {
                // The atlas is a horizontal strip of TERRAIN_KIND_COUNT tiles, so
                // the array layer index is the atlas column — i.e. the TerrainKind.
                settings.array_layout = Some(ImageArrayLayout::GridCount {
                    columns: TERRAIN_KIND_COUNT,
                    rows: 1,
                })
            })
            .load("textures/terrain.png"),
    ));
}

/// Fills the screen before the first frame of gameplay is shown, generating
/// whatever the background pass has not produced yet however long it takes.
fn spawn_initial_chunks(
    mut commands: Commands,
    mut map: ResMut<WorldMap>,
    config: Res<TerrainConfig>,
    tileset: Res<TerrainTileset>,
    resident: Query<(Entity, &ChunkCoord)>,
    camera: Single<&Transform, With<WorldCamera>>,
) {
    refresh_resident_chunks(
        &mut commands,
        &mut map,
        &config,
        &tileset,
        &resident,
        camera.translation.truncate(),
        usize::MAX,
    );
}

/// Keeps the resident region centred on the camera as it moves.
fn stream_chunks_around_camera(
    mut commands: Commands,
    mut map: ResMut<WorldMap>,
    config: Res<TerrainConfig>,
    tileset: Res<TerrainTileset>,
    resident: Query<(Entity, &ChunkCoord)>,
    camera: Single<&Transform, With<WorldCamera>>,
) {
    refresh_resident_chunks(
        &mut commands,
        &mut map,
        &config,
        &tileset,
        &resident,
        camera.translation.truncate(),
        MAX_BLOCKING_GENERATIONS_PER_FRAME,
    );
}

/// Despawns chunk entities the camera has left behind and spawns the ones it has
/// arrived at. `budget` caps how many missing chunks may be generated on this
/// thread; chunks over that budget are simply left for a later frame.
fn refresh_resident_chunks(
    commands: &mut Commands,
    map: &mut WorldMap,
    config: &TerrainConfig,
    tileset: &TerrainTileset,
    resident: &Query<(Entity, &ChunkCoord)>,
    camera: Vec2,
    mut budget: usize,
) {
    let centre = chunk_at(camera);

    let mut already_resident = Vec::new();
    for (entity, coord) in resident.iter() {
        if chunk_distance(coord.0, centre) > RESIDENT_RADIUS {
            commands.entity(entity).despawn();
        } else {
            already_resident.push(coord.0);
        }
    }

    for coord in chunks_within(centre, RESIDENT_RADIUS) {
        if already_resident.contains(&coord) {
            continue;
        }
        if map.get(coord).is_none() {
            if budget == 0 {
                continue;
            }
            budget -= 1;
            map.generate_blocking(config, coord);
        }
        let tiles = map.get(coord).expect("the chunk was just generated");
        spawn_chunk(commands, coord, tiles, tileset);
    }
}

fn spawn_chunk(
    commands: &mut Commands,
    coord: UVec2,
    tiles: &[TerrainKind],
    tileset: &TerrainTileset,
) {
    commands.spawn((
        ChunkCoord(coord),
        TilemapChunk {
            chunk_size: CHUNK_SIZE,
            tile_display_size: TILE_DISPLAY_SIZE,
            tileset: tileset.0.clone(),
            alpha_mode: AlphaMode2d::Opaque,
        },
        TilemapChunkTileData(
            tiles
                .iter()
                .map(|kind| Some(TileData::from_tileset_index(kind.tileset_index())))
                .collect(),
        ),
        Transform::from_translation(chunk_translation(coord).extend(0.0)),
        DespawnOnExit(Screen::Gameplay),
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chunks_own_position_resolves_back_to_it() {
        for coord in [
            UVec2::ZERO,
            UVec2::splat(1),
            WORLD_CHUNKS / 2,
            WORLD_CHUNKS - UVec2::ONE,
        ] {
            assert_eq!(chunk_at(chunk_translation(coord)), coord);
        }
    }

    /// The camera starts at the origin, so the origin has to land in the middle
    /// of the world rather than in a corner of it.
    #[test]
    fn the_origin_is_the_middle_of_the_world() {
        assert_eq!(chunk_at(Vec2::ZERO), WORLD_CHUNKS / 2);
    }

    #[test]
    fn positions_beyond_the_world_clamp_to_its_edge_chunks() {
        let far = world_half_extent() * 4.0;
        assert_eq!(chunk_at(far), WORLD_CHUNKS - UVec2::ONE);
        assert_eq!(chunk_at(-far), UVec2::ZERO);
    }

    #[test]
    fn chunk_indices_round_trip_through_coordinates() {
        for index in [0, 1, WORLD_CHUNKS.x as usize, chunk_count() - 1] {
            assert_eq!(chunk_index(chunk_coord(index)), index);
        }
    }

    /// Away from the world edge the resident region is a full square; at a corner
    /// it is only the quarter of it that exists.
    #[test]
    fn the_resident_region_stops_at_the_world_edge() {
        let side = (2 * RESIDENT_RADIUS + 1) as usize;
        assert_eq!(
            chunks_within(WORLD_CHUNKS / 2, RESIDENT_RADIUS).count(),
            side * side
        );
        assert_eq!(
            chunks_within(UVec2::ZERO, RESIDENT_RADIUS).count(),
            (RESIDENT_RADIUS as usize + 1).pow(2)
        );
    }

    /// Every chunk gets generated exactly once, starting from the middle.
    #[test]
    fn the_background_pass_covers_the_whole_world_centre_first() {
        let generation = BackgroundGeneration::default();
        assert_eq!(generation.pending.len(), chunk_count());

        let mut seen = vec![false; chunk_count()];
        for &coord in &generation.pending {
            let slot = &mut seen[chunk_index(coord)];
            assert!(!*slot, "chunk {coord} is queued twice");
            *slot = true;
        }

        let first = generation.pending.last().expect("the world is not empty");
        assert_eq!(chunk_distance(*first, WORLD_CHUNKS / 2), 0);
    }
}
