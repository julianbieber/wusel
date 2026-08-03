//! The world: a fixed 64x64 grid of chunks, and the rules for which parts of it
//! exist as entities at any moment.
//!
//! Two separate things are "loaded" here and it matters that they are separate:
//!
//! * **Tile data** ([`WorldMap`]) is generated for the whole world and kept for
//!   the session. One byte per tile, so all 4096 chunks cost ~16 MB. A
//!   background pass fills it in off the main thread.
//! * **Chunk entities** exist only near the camera. Each one costs a mesh, a
//!   material and a per-chunk index image, which is why the whole world cannot
//!   be resident — 4096 of them would be hundreds of megabytes.
//!
//! So the world is finite and you can walk to its edge, but you are never
//! holding more of it than you can see.
//!
//! Nothing here outlives [`Screen::Gameplay`]. Entering it builds every world
//! resource from nothing and leaving it removes them, so a session can never
//! inherit a half-generated map — or a task still in flight — from the last one.

use std::sync::Arc;

use bevy::{
    image::{ImageArrayLayout, ImageLoaderSettings},
    platform::collections::HashSet,
    prelude::*,
    sprite_render::{AlphaMode2d, TileData, TilemapChunk, TilemapChunkTileData},
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};

use crate::{
    camera::{WorldCamera, visible_half_extent},
    gameplay::{
        growth::CityGrowthPlugin,
        plan::WorldPlanPlugin,
        terrain::{ChunkTerrain, TERRAIN_KIND_COUNT, TerrainConfig, TerrainKind, generate_chunk},
    },
    screens::Screen,
};

/// Tiles along each axis of one chunk.
pub const CHUNK_SIZE: UVec2 = UVec2::splat(64);
/// Pixels each tile covers on screen. This is the atlas' native tile size:
/// drawing 8x8 art 1:1 keeps it crisp, since any upscale would be resampled.
pub const TILE_DISPLAY_SIZE: UVec2 = UVec2::splat(8);
/// Chunks along each axis of the world — 64x64 chunks is 4096x4096 tiles.
pub const WORLD_CHUNKS: UVec2 = UVec2::splat(64);
/// Tiles along each axis of the world. This is the extent of global tile space.
pub const WORLD_TILES: UVec2 =
    UVec2::new(WORLD_CHUNKS.x * CHUNK_SIZE.x, WORLD_CHUNKS.y * CHUNK_SIZE.y);

/// Chunks kept resident beyond the ones actually on screen, so a chunk exists
/// before it is needed rather than popping in at the edge of the view.
const RESIDENT_MARGIN_CHUNKS: i32 = 1;

/// The worst-case stall the streamer may add to a frame when the camera outruns the
/// background pass. This was two when a chunk cost ~4 ms; the biome rework made
/// generation 1.9x dearer and it came down to one. gh-14's substrate layers made it
/// 1.8x dearer again — 6.6 ms per 64x64 chunk, measured by
/// `terrain::the_default_config_produces_recognisably_different_regions` — so one
/// chunk is now most of a 60 fps frame on its own and there is no room to go back up.
///
/// It cannot go to zero: the streamer is the only thing that fills a chunk the
/// camera has already reached, so a budget of none would leave a hole on screen
/// until the background pass happened to arrive. One late frame beats one empty
/// chunk, which is why this is a stall budget and not a switch.
const MAX_BLOCKING_GENERATIONS_PER_FRAME: usize = 1;

/// Chunk entities built per frame. Panning only ever brings a row of them into
/// view, but a zoom step can bring hundreds at once, and building that many
/// meshes and index images in one frame is a hitch you can see. Over budget they
/// arrive over the next few frames instead.
const MAX_CHUNK_SPAWNS_PER_FRAME: usize = 32;

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
        // The config is a knob rather than world state, so it is the one thing
        // here that outlives a session.
        app.init_resource::<TerrainConfig>();
        // Both of these edit tiles, which is why they are the world's plugins
        // rather than siblings of it the way the tint and the weather are.
        app.add_plugins((WorldPlanPlugin, CityGrowthPlugin));
        app.add_systems(Startup, load_tileset);
        app.add_systems(
            OnEnter(Screen::Gameplay),
            (start_world, spawn_initial_chunks).chain(),
        );
        app.add_systems(OnExit(Screen::Gameplay), tear_down_world);
        // One `configure_sets` for the whole spine, and the state gate hangs off
        // the tuple rather than off each set: a set configured separately would
        // carry no `run_if`, and its systems would run on the menu with no
        // `WorldMap` to read.
        app.configure_sets(
            Update,
            (
                WorldSystems::Streaming,
                WorldSystems::Planning,
                WorldSystems::Growth,
                WorldSystems::Refresh,
            )
                .chain()
                .run_if(in_state(Screen::Gameplay)),
        );
        app.add_systems(
            Update,
            (drive_background_generation, stream_chunks_around_camera)
                .chain()
                .in_set(WorldSystems::Streaming),
        );
        app.add_systems(Update, refresh_edited_chunks.in_set(WorldSystems::Refresh));
    }
}

/// The order the world advances in each frame. [`crate::gameplay::plan`] hangs
/// its systems on `Planning`, which is what puts every tile edit between the
/// streamer that spawned the chunk entities and the refresh that rebuilds them —
/// so an edit is visible in the same frame it lands.
#[derive(SystemSet, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum WorldSystems {
    /// Generating tiles and keeping the resident chunk entities in step.
    Streaming,
    /// Planning cities and roads, and editing the tiles under them.
    Planning,
    /// Growing and shrinking the cities the plan laid out, and editing the tiles
    /// under *them*. After `Planning` because a road must never be routed against
    /// a world whose cities are moving, and before `Refresh` for the same reason
    /// the plan's own edits are: so an edit is visible in the frame it lands.
    Growth,
    /// Rebuilding the resident entities whose tiles the plan changed.
    Refresh,
}

/// Builds every world resource from nothing. There is nothing to resume and
/// nothing to reconcile: the previous session left none of it behind.
fn start_world(mut commands: Commands) {
    commands.insert_resource(WorldMap::default());
    commands.insert_resource(BackgroundGeneration::default());
    commands.insert_resource(DirtyChunks::default());
    commands.insert_resource(HeightUploadQueue::default());
}

/// Throws the world away. Removing the resources drops whatever tasks they were
/// holding, so a chunk that finishes generating after this has nowhere to
/// deliver to and its result is discarded rather than landing in the next
/// session. The chunk entities need no help — they carry `DespawnOnExit`.
fn tear_down_world(mut commands: Commands) {
    commands.remove_resource::<WorldMap>();
    commands.remove_resource::<BackgroundGeneration>();
    commands.remove_resource::<DirtyChunks>();
    // The queue's absence is also the signal that retires the render world's
    // heightmap, so a session can never be shown under the next one's terrain.
    commands.remove_resource::<HeightUploadQueue>();
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

/// Which chunk a global tile falls in. Callers outside this module work in tile
/// space, so this is the only conversion they need.
pub fn chunk_of_tile(tile: IVec2) -> UVec2 {
    (tile.as_uvec2() / CHUNK_SIZE).min(WORLD_CHUNKS - UVec2::ONE)
}

/// The index of the chunk a global tile falls in — the key the plan indexes by.
pub fn chunk_index_of_tile(tile: IVec2) -> usize {
    chunk_index(chunk_of_tile(tile))
}

/// Whether a global tile is inside the world at all.
pub fn tile_in_world(tile: IVec2) -> bool {
    tile.x >= 0 && tile.y >= 0 && tile.x < WORLD_TILES.x as i32 && tile.y < WORLD_TILES.y as i32
}

/// Where a global tile sits in world space, at its centre so that an entity
/// placed there lines up with the tile rather than its corner.
pub fn tile_translation(tile: IVec2) -> Vec2 {
    (tile.as_vec2() + Vec2::splat(0.5)) * TILE_DISPLAY_SIZE.as_vec2() - world_half_extent()
}

/// Where a world-space position sits in *continuous* global tile space: tile `t`
/// covers `t..t + 1`, so a tile's centre comes back as `t + 0.5`.
///
/// The inverse of [`tile_translation`], and the only one of these conversions that
/// keeps its fraction — the weather overlay needs to know where between two tiles a
/// screen pixel falls, which [`chunk_at`] floors away. Not clamped: a caller that
/// samples outside the world has to decide for itself what that means.
pub fn tile_position_at(position: Vec2) -> Vec2 {
    (position + world_half_extent()) / TILE_DISPLAY_SIZE.as_vec2()
}

/// How far the resident region has to reach to cover the screen, in chunks.
///
/// Derived rather than fixed, because zooming out multiplies how much world is
/// visible: the chunk entities have to keep up or the view fills with holes at
/// its edges. The cost of that is why the zoom is bounded — this is a radius, so
/// the entity count it implies grows with its square.
fn resident_radius(visible_half_extent: Vec2) -> i32 {
    let chunk = (CHUNK_SIZE * TILE_DISPLAY_SIZE).as_vec2();
    let reach = (visible_half_extent / chunk).ceil().as_ivec2();
    reach.x.max(reach.y) + RESIDENT_MARGIN_CHUNKS
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

/// One tile of the world set to a kind it was not generated as. This is how the
/// plan gets cities and roads into the map.
#[derive(Clone, Copy, Debug)]
pub struct TileEdit {
    pub tile: IVec2,
    pub kind: TerrainKind,
}

/// Every tile in the world, or `None` for chunks the background pass has not
/// reached yet. One `TerrainKind` per tile, so this is ~16 MB in full.
///
/// And, beside it, the height each of those tiles was classified from — another
/// ~16 MB. That is the whole cost of the heightmap: `classify` computed the value
/// already and used to drop it, so keeping it buys the tint pass, the noise step
/// and any later reader their heights for memory instead of generation time.
///
/// Chunks are held behind an `Arc` so that [`WorldMap::snapshot`] can hand the
/// whole finished world to a background task without copying 16 MB.
#[derive(Resource)]
pub struct WorldMap {
    chunks: Vec<Option<Arc<[TerrainKind]>>>,
    heights: Vec<Option<Arc<[u8]>>>,
}

impl Default for WorldMap {
    fn default() -> Self {
        Self {
            chunks: vec![None; chunk_count()],
            heights: vec![None; chunk_count()],
        }
    }
}

impl WorldMap {
    fn get(&self, coord: UVec2) -> Option<&Arc<[TerrainKind]>> {
        self.chunks[chunk_index(coord)].as_ref()
    }

    /// The heights of a generated chunk, in the same order as its kinds.
    ///
    /// The tint reads the upload queue rather than this, so today only the tests
    /// call it. It is the heightmap kept for the *crate*: the noise step and any
    /// later reader get their heights here without sampling the terrain a second
    /// time, and it is how the map can be checked without a GPU.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn heights(&self, coord: UVec2) -> Option<&Arc<[u8]>> {
        self.heights[chunk_index(coord)].as_ref()
    }

    /// Stores a freshly generated chunk and queues its heights for the GPU.
    ///
    /// The queue is an argument rather than something a later system derives,
    /// because that is what makes "a chunk reaches the texture exactly once" a
    /// property of the type: there is no way to store a chunk without queueing it.
    fn insert(&mut self, coord: UVec2, chunk: ChunkTerrain, uploads: &mut HeightUploadQueue) {
        let index = chunk_index(coord);
        let heights: Arc<[u8]> = chunk.heights.into();
        self.chunks[index] = Some(chunk.kinds.into());
        self.heights[index] = Some(Arc::clone(&heights));
        uploads.push(index, heights);
    }

    /// Generates a chunk on the calling thread. Only for chunks that are needed
    /// this frame — everything else should come from the background pass.
    fn generate_blocking(
        &mut self,
        config: &TerrainConfig,
        coord: UVec2,
        uploads: &mut HeightUploadQueue,
    ) {
        let chunk = generate_chunk(config, chunk_origin_tiles(coord), CHUNK_SIZE);
        self.insert(coord, chunk, uploads);
    }

    /// The kind of one global tile, or `None` outside the world or in a chunk the
    /// background pass has not reached.
    ///
    /// The simulation's read. It cannot go through [`WorldMap::snapshot`]: a
    /// snapshot taken once a step would not show one city the fields another city
    /// claimed earlier in that same step, and claim exclusivity is exactly that
    /// read. Taking one per city instead would be 4096 refcount bumps apiece.
    pub fn tile(&self, tile: IVec2) -> Option<TerrainKind> {
        if !tile_in_world(tile) {
            return None;
        }
        let coord = chunk_of_tile(tile);
        let local = tile - chunk_origin_tiles(coord);
        Some(self.get(coord)?[(local.y * CHUNK_SIZE.x as i32 + local.x) as usize])
    }

    /// A whole world built from a rule rather than from the noise, so a test can
    /// state exactly the terrain it wants a city to grow into. The [`WorldMap`]
    /// counterpart of [`WorldSnapshot::from_fn`], for the readers that take the live
    /// map because they need to see edits as they land.
    #[cfg(test)]
    pub fn from_fn(kind: impl Fn(IVec2) -> TerrainKind) -> Self {
        let chunks: Vec<Option<Arc<[TerrainKind]>>> = (0..chunk_count())
            .map(|index| {
                let origin = chunk_origin_tiles(chunk_coord(index));
                let tiles: Arc<[TerrainKind]> = (0..CHUNK_SIZE.element_product())
                    .map(|i| {
                        kind(
                            origin
                                + IVec2::new((i % CHUNK_SIZE.x) as i32, (i / CHUNK_SIZE.x) as i32),
                        )
                    })
                    .collect();
                Some(tiles)
            })
            .collect();

        Self {
            chunks,
            heights: vec![None; chunk_count()],
        }
    }

    /// A shared read-only view of the whole world, or `None` while any chunk is
    /// still missing. The planner takes one of these instead of a copy.
    pub fn snapshot(&self) -> Option<WorldSnapshot> {
        let chunks: Option<Vec<Arc<[TerrainKind]>>> = self.chunks.iter().cloned().collect();
        Some(WorldSnapshot {
            chunks: chunks?.into(),
        })
    }

    /// Applies the plan's edits and reports which chunks they touched, so the
    /// resident entities showing those chunks can be rebuilt.
    ///
    /// A chunk is rewritten rather than mutated in place *while anyone else is
    /// holding it*: the snapshot the planner is still reading holds the old `Arc`,
    /// and it must keep seeing the world it planned against. `Arc::get_mut` is what
    /// asks that question — it hands out the chunk only when this map is the sole
    /// owner, so the copy-on-write happens exactly when it is needed rather than
    /// on every call. That matters once the simulation is running: it edits a
    /// handful of tiles per chunk every step, forever, where the plan paid for its
    /// copies once and stopped.
    ///
    /// Edits are grouped by sorting rather than by scanning a list of touched
    /// chunks, which was O(edits x chunks) — fine for the plan's per-chunk batches
    /// and not for a step that touches ~200 chunks across the whole world.
    ///
    /// **Kinds only.** What the heightmap records is the height the terrain was
    /// *generated* at, never what was stamped over it, so a road, a town or a river
    /// is shaded by the ground it sits on — and no edit ever re-queues an upload.
    pub fn apply_edits(&mut self, edits: &[TileEdit], dirty: &mut DirtyChunks) {
        let mut ordered: Vec<(usize, TileEdit)> = edits
            .iter()
            .filter(|edit| tile_in_world(edit.tile))
            .map(|&edit| (chunk_index_of_tile(edit.tile), edit))
            .collect();
        ordered.sort_unstable_by_key(|(index, _)| *index);

        let mut start = 0;
        while start < ordered.len() {
            let index = ordered[start].0;
            let end = ordered[start..]
                .iter()
                .position(|(other, _)| *other != index)
                .map_or(ordered.len(), |offset| start + offset);

            if let Some(existing) = self.chunks[index].as_mut() {
                let coord = chunk_coord(index);
                let origin = chunk_origin_tiles(coord);
                let batch = &ordered[start..end];

                match Arc::get_mut(existing) {
                    Some(tiles) => write_edits(tiles, origin, batch),
                    None => {
                        let mut tiles = existing.to_vec();
                        write_edits(&mut tiles, origin, batch);
                        *existing = tiles.into();
                    }
                }
                dirty.mark(coord);
            }

            start = end;
        }
    }
}

fn write_edits(tiles: &mut [TerrainKind], origin: IVec2, batch: &[(usize, TileEdit)]) {
    for (_, edit) in batch {
        let local = edit.tile - origin;
        tiles[(local.y * CHUNK_SIZE.x as i32 + local.x) as usize] = edit.kind;
    }
}

/// The finished world, shared with the planning tasks. Cloning one is a handful
/// of refcount bumps rather than a 16 MB copy.
#[derive(Clone)]
pub struct WorldSnapshot {
    chunks: Arc<[Arc<[TerrainKind]>]>,
}

impl WorldSnapshot {
    /// The kind of a global tile, or `None` outside the world.
    pub fn tile(&self, tile: IVec2) -> Option<TerrainKind> {
        if !tile_in_world(tile) {
            return None;
        }
        let coord = chunk_of_tile(tile);
        let local = tile - chunk_origin_tiles(coord);
        Some(self.chunks[chunk_index(coord)][(local.y * CHUNK_SIZE.x as i32 + local.x) as usize])
    }

    /// The real world, generated up front. Only for the ignored end-to-end test
    /// — this is every chunk the game would ever generate, so it is split across
    /// threads to keep it to a few seconds rather than a minute.
    #[cfg(test)]
    pub fn generated(config: &TerrainConfig) -> Self {
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let per_thread = chunk_count().div_ceil(threads);

        let chunks: Vec<Arc<[TerrainKind]>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|slice| {
                    scope.spawn(move || {
                        let start = slice * per_thread;
                        let end = (start + per_thread).min(chunk_count());
                        (start..end)
                            .map(|index| {
                                let chunk = generate_chunk(
                                    config,
                                    chunk_origin_tiles(chunk_coord(index)),
                                    CHUNK_SIZE,
                                );
                                Arc::<[TerrainKind]>::from(chunk.kinds)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|handle| handle.join().expect("chunk generation panicked"))
                .collect()
        });

        Self {
            chunks: chunks.into(),
        }
    }

    /// The same world with some tiles changed. This is what a test uses to walk
    /// the plan forward one road at a time, the way the driver does.
    #[cfg(test)]
    pub fn with_edits(&self, edits: &[TileEdit]) -> Self {
        let mut chunks: Vec<Arc<[TerrainKind]>> = self.chunks.to_vec();
        let inside: Vec<&TileEdit> = edits.iter().filter(|e| tile_in_world(e.tile)).collect();

        let mut touched: Vec<usize> = inside.iter().map(|e| chunk_index_of_tile(e.tile)).collect();
        touched.sort_unstable();
        touched.dedup();

        for index in touched {
            let origin = chunk_origin_tiles(chunk_coord(index));
            let mut tiles = chunks[index].to_vec();
            for edit in inside
                .iter()
                .filter(|e| chunk_index_of_tile(e.tile) == index)
            {
                let local = edit.tile - origin;
                tiles[(local.y * CHUNK_SIZE.x as i32 + local.x) as usize] = edit.kind;
            }
            chunks[index] = tiles.into();
        }

        Self {
            chunks: chunks.into(),
        }
    }

    /// A world built from a rule rather than from the noise, so a test can state
    /// exactly the terrain it wants to route across.
    #[cfg(test)]
    pub fn from_fn(kind: impl Fn(IVec2) -> TerrainKind) -> Self {
        let chunks: Vec<Arc<[TerrainKind]>> = (0..chunk_count())
            .map(|index| {
                let origin = chunk_origin_tiles(chunk_coord(index));
                (0..CHUNK_SIZE.element_product())
                    .map(|i| {
                        kind(
                            origin
                                + IVec2::new((i % CHUNK_SIZE.x) as i32, (i / CHUNK_SIZE.x) as i32),
                        )
                    })
                    .collect()
            })
            .collect();
        Self {
            chunks: chunks.into(),
        }
    }
}

/// Chunks whose tiles the plan has edited, and whose entity — if one is resident
/// — is showing the tiles from before the edit.
///
/// A set rather than a list because both of its operations are membership tests,
/// and because the simulation never lets it go empty: `refresh_edited_chunks` used
/// to early-return on almost every frame once the plan was done, and now runs its
/// scan every frame instead.
#[derive(Resource, Default)]
pub struct DirtyChunks {
    chunks: HashSet<UVec2>,
}

impl DirtyChunks {
    fn mark(&mut self, coord: UVec2) {
        self.chunks.insert(coord);
    }
}

/// One chunk's heights on their way to the GPU: which chunk, by the world's own
/// chunk index, and the texels themselves.
///
/// The texels ride behind the same `Arc` [`WorldMap`] stores, so queueing a chunk
/// copies no tile data.
pub struct ChunkHeights {
    pub chunk: usize,
    pub texels: Arc<[u8]>,
}

/// Chunks that have been generated but whose heights are not on the GPU yet.
///
/// The same idiom as [`DirtyChunks`] — something the streamer fills and another
/// pass drains — and it lives here rather than with the tint so that
/// [`start_world`] and [`tear_down_world`] own its lifetime. It therefore cannot
/// be absent while a [`WorldMap`] exists, nor outlive one; and its absence is what
/// tells the render world that the session is over and the texture can go.
#[derive(Resource, Default)]
pub struct HeightUploadQueue {
    pending: Vec<ChunkHeights>,
}

impl HeightUploadQueue {
    fn push(&mut self, chunk: usize, texels: Arc<[u8]>) {
        self.pending.push(ChunkHeights { chunk, texels });
    }

    /// Hands the queued chunks over and leaves the queue empty, so a chunk is
    /// queued once, uploaded once, and cannot be missed or repeated.
    pub fn take(&mut self) -> Vec<ChunkHeights> {
        std::mem::take(&mut self.pending)
    }
}

/// The background pass that fills in the rest of [`WorldMap`].
#[derive(Resource)]
pub struct BackgroundGeneration {
    /// Chunks still to generate, ordered so that `pop` yields the ones nearest
    /// the middle of the world — which is where the camera starts — first.
    pending: Vec<UVec2>,
    in_flight: Vec<Task<(UVec2, ChunkTerrain)>>,
}

impl BackgroundGeneration {
    /// Whether every chunk has been generated. The plan waits for this, so that
    /// it never sees a hole in the world it is planning against.
    pub fn is_complete(&self) -> bool {
        self.pending.is_empty() && self.in_flight.is_empty()
    }
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
    mut uploads: ResMut<HeightUploadQueue>,
    config: Res<TerrainConfig>,
) {
    generation
        .in_flight
        .retain_mut(|task| match block_on(poll_once(task)) {
            Some((coord, chunk)) => {
                map.insert(coord, chunk, &mut uploads);
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
    mut uploads: ResMut<HeightUploadQueue>,
    config: Res<TerrainConfig>,
    tileset: Res<TerrainTileset>,
    resident: Query<(Entity, &ChunkCoord)>,
    camera: Single<(&Transform, &Camera, &Projection), With<WorldCamera>>,
) {
    let (transform, camera, projection) = camera.into_inner();
    refresh_resident_chunks(
        &mut commands,
        &mut map,
        &mut uploads,
        &config,
        &tileset,
        &resident,
        transform.translation.truncate(),
        resident_radius(visible_half_extent(camera, projection)),
        usize::MAX,
        usize::MAX,
    );
}

/// Keeps the resident region centred on the camera as it moves, and sized to
/// however much world the camera can currently see.
fn stream_chunks_around_camera(
    mut commands: Commands,
    mut map: ResMut<WorldMap>,
    mut uploads: ResMut<HeightUploadQueue>,
    config: Res<TerrainConfig>,
    tileset: Res<TerrainTileset>,
    resident: Query<(Entity, &ChunkCoord)>,
    camera: Single<(&Transform, &Camera, &Projection), With<WorldCamera>>,
) {
    let (transform, camera, projection) = camera.into_inner();
    refresh_resident_chunks(
        &mut commands,
        &mut map,
        &mut uploads,
        &config,
        &tileset,
        &resident,
        transform.translation.truncate(),
        resident_radius(visible_half_extent(camera, projection)),
        MAX_BLOCKING_GENERATIONS_PER_FRAME,
        MAX_CHUNK_SPAWNS_PER_FRAME,
    );
}

/// Despawns chunk entities that have left the resident region and spawns the
/// ones that have entered it.
///
/// `generation_budget` caps how many missing chunks may be generated on this
/// thread and `spawn_budget` how many entities may be built; anything over
/// either is simply left for a later frame, which is what keeps a zoom step from
/// trying to build a few hundred chunks at once.
fn refresh_resident_chunks(
    commands: &mut Commands,
    map: &mut WorldMap,
    uploads: &mut HeightUploadQueue,
    config: &TerrainConfig,
    tileset: &TerrainTileset,
    resident: &Query<(Entity, &ChunkCoord)>,
    camera: Vec2,
    radius: i32,
    mut generation_budget: usize,
    mut spawn_budget: usize,
) {
    let centre = chunk_at(camera);

    let mut already_resident = Vec::new();
    for (entity, coord) in resident.iter() {
        if chunk_distance(coord.0, centre) > radius {
            commands.entity(entity).despawn();
        } else {
            already_resident.push(coord.0);
        }
    }

    for coord in chunks_within(centre, radius) {
        if already_resident.contains(&coord) || spawn_budget == 0 {
            continue;
        }
        if map.get(coord).is_none() {
            if generation_budget == 0 {
                continue;
            }
            generation_budget -= 1;
            map.generate_blocking(config, coord, uploads);
        }
        spawn_budget -= 1;
        let tiles = map.get(coord).expect("the chunk was just generated");
        spawn_chunk(commands, coord, tiles, tileset);
    }
}

/// Rebuilds the tile data of the resident chunk entities the plan has edited.
///
/// A chunk that is not resident needs nothing: the edit went into [`WorldMap`],
/// so it will be there when the chunk is next spawned.
fn refresh_edited_chunks(
    mut dirty: ResMut<DirtyChunks>,
    map: Res<WorldMap>,
    mut resident: Query<(&ChunkCoord, &mut TilemapChunkTileData)>,
) {
    if dirty.chunks.is_empty() {
        return;
    }

    for (coord, mut data) in resident.iter_mut() {
        if !dirty.chunks.contains(&coord.0) {
            continue;
        }
        let Some(tiles) = map.get(coord.0) else {
            continue;
        };
        *data = tile_data(tiles);
    }

    dirty.chunks.clear();
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
        tile_data(tiles),
        Transform::from_translation(chunk_translation(coord).extend(0.0)),
        DespawnOnExit(Screen::Gameplay),
    ));
}

fn tile_data(tiles: &[TerrainKind]) -> TilemapChunkTileData {
    TilemapChunkTileData(
        tiles
            .iter()
            .map(|kind| Some(TileData::from_tileset_index(kind.tileset_index())))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The atlas is *divided* by `TERRAIN_KIND_COUNT`, so a constant that disagrees
    /// with the PNG does not fail to load — it slices the strip at the wrong offset
    /// and draws every tile in the game wrong. That is exactly what happened when the
    /// Farmland column landed before the enum did, and nothing caught it.
    ///
    /// Reads the PNG's IHDR width directly: 8 bytes of signature, then a 4-byte
    /// length and the `IHDR` tag, then the width as big-endian u32.
    #[test]
    fn the_atlas_has_a_column_for_every_terrain_kind() {
        let png = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/textures/terrain.png"
        ))
        .expect("the tileset is checked in beside the code");
        assert_eq!(&png[1..4], b"PNG", "not a PNG");

        let width = u32::from_be_bytes(png[16..20].try_into().expect("IHDR width"));
        let height = u32::from_be_bytes(png[20..24].try_into().expect("IHDR height"));

        assert_eq!(
            width,
            TERRAIN_KIND_COUNT * TILE_DISPLAY_SIZE.x,
            "the atlas is {width}px wide, which is {} columns of {}px, but \
             TERRAIN_KIND_COUNT is {TERRAIN_KIND_COUNT}",
            width as f32 / TILE_DISPLAY_SIZE.x as f32,
            TILE_DISPLAY_SIZE.x
        );
        assert_eq!(height, TILE_DISPLAY_SIZE.y, "the atlas is one row of tiles");
    }

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

    /// The weather overlay maps screen pixels back to tile space to sample its
    /// maps there, so an error in this conversion would slide the whole sky off
    /// the terrain — and, because the maps are sampled clamped, would show up as a
    /// flat sky rather than as anything that looks like a bug.
    #[test]
    fn a_tile_centre_converts_back_to_its_own_tile() {
        for tile in [
            IVec2::ZERO,
            IVec2::splat(1),
            WORLD_TILES.as_ivec2() / 2,
            WORLD_TILES.as_ivec2() - IVec2::ONE,
        ] {
            let position = tile_position_at(tile_translation(tile));
            assert_eq!(position.floor().as_ivec2(), tile);
            assert!((position - tile.as_vec2() - Vec2::splat(0.5)).length() < 1e-3);
        }
    }

    /// The camera sits at the origin on entering gameplay, and the maps are indexed
    /// from the world's corner, so this is the offset that has to be there.
    #[test]
    fn the_origin_is_the_middle_of_tile_space() {
        assert_eq!(tile_position_at(Vec2::ZERO), WORLD_TILES.as_vec2() / 2.0);
    }

    #[test]
    fn chunk_indices_round_trip_through_coordinates() {
        for index in [0, 1, WORLD_CHUNKS.x as usize, chunk_count() - 1] {
            assert_eq!(chunk_index(chunk_coord(index)), index);
        }
    }

    /// Half a 1920x1080 window, which is the size the resident radius used to be
    /// a hand-tuned constant for.
    const HALF_WINDOW: Vec2 = Vec2::new(960.0, 540.0);

    /// Away from the world edge the resident region is a full square; at a corner
    /// it is only the quarter of it that exists.
    #[test]
    fn the_resident_region_stops_at_the_world_edge() {
        let radius = resident_radius(HALF_WINDOW);
        let side = (2 * radius + 1) as usize;
        assert_eq!(chunks_within(WORLD_CHUNKS / 2, radius).count(), side * side);
        assert_eq!(
            chunks_within(UVec2::ZERO, radius).count(),
            (radius as usize + 1).pow(2)
        );
    }

    /// The radius has to cover the corner of the view, or the view fills with
    /// holes at its edges when you zoom out.
    #[test]
    fn the_resident_region_covers_everything_on_screen_at_every_zoom() {
        let chunk = (CHUNK_SIZE * TILE_DISPLAY_SIZE).as_vec2();

        for scale in [0.25, 0.5, 1.0, 2.0, 4.0] {
            let visible = HALF_WINDOW * scale;
            let reach = resident_radius(visible) as f32 * chunk;
            assert!(
                reach.x >= visible.x && reach.y >= visible.y,
                "at scale {scale} the resident region reaches {reach} but {visible} is visible"
            );
        }
    }

    /// The radius was 3 before it was derived, tuned by hand for an unzoomed
    /// window — unzoomed, it still is, so nothing about panning changed.
    #[test]
    fn an_unzoomed_window_resides_exactly_what_it_used_to() {
        assert_eq!(resident_radius(HALF_WINDOW), 3);
    }

    /// This is what bounds the zoom. Every resident chunk costs a mesh, a
    /// material and an index image — about 100 KB, going by the 4096-chunk
    /// measurement — and the count grows with the square of the radius, so the
    /// widest zoom is as far out as the memory allows rather than as far as
    /// looks nice.
    #[test]
    fn the_widest_zoom_keeps_the_resident_region_affordable() {
        let widest = resident_radius(HALF_WINDOW * crate::camera::MAX_ZOOM_SCALE);
        let chunks = chunks_within(WORLD_CHUNKS / 2, widest).count();

        assert!(
            chunks < 400,
            "{chunks} chunks resident at the widest zoom, roughly {} MB",
            chunks / 10
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

    /// A chunk reaches the heightmap exactly once. Storing it and queueing it are
    /// the same call, so there is no path that does one without the other, and the
    /// queue hands its contents over rather than lending them.
    #[test]
    fn a_generated_chunk_queues_its_heights_once() {
        let config = TerrainConfig::default();
        let mut map = WorldMap::default();
        let mut uploads = HeightUploadQueue::default();
        let coord = WORLD_CHUNKS / 2;

        map.generate_blocking(&config, coord, &mut uploads);

        let heights = map
            .heights(coord)
            .expect("the chunk was just generated")
            .clone();
        assert_eq!(heights.len() as u32, CHUNK_SIZE.element_product());

        let queued = uploads.take();
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].chunk, chunk_index(coord));
        assert_eq!(queued[0].texels.as_ref(), heights.as_ref());

        assert!(uploads.take().is_empty(), "a chunk is queued once");
        assert!(map.heights(coord).is_some(), "and the map keeps its copy");
    }

    /// What the heightmap records is the height the terrain was *generated* at,
    /// never what was stamped over it — so a road is shaded by the ground it sits
    /// on, and no plan edit costs an upload.
    #[test]
    fn an_edit_leaves_the_height_it_was_stamped_over() {
        let config = TerrainConfig::default();
        let mut map = WorldMap::default();
        let mut uploads = HeightUploadQueue::default();
        let coord = WORLD_CHUNKS / 2;

        map.generate_blocking(&config, coord, &mut uploads);
        uploads.take();
        let before = map.heights(coord).expect("generated").clone();

        let local = IVec2::splat(7);
        let mut dirty = DirtyChunks::default();
        map.apply_edits(
            &[TileEdit {
                tile: chunk_origin_tiles(coord) + local,
                kind: TerrainKind::Road,
            }],
            &mut dirty,
        );

        let index = (local.y * CHUNK_SIZE.x as i32 + local.x) as usize;
        assert_eq!(
            map.get(coord).expect("still there")[index],
            TerrainKind::Road
        );
        assert_eq!(
            map.heights(coord).expect("still there").as_ref(),
            before.as_ref()
        );
        assert!(uploads.take().is_empty(), "an edit re-queues nothing");
    }
}
