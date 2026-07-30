//! Cutting rivers into a world that has already been generated.
//!
//! Where water runs at a tile is decided hundreds of tiles uphill, so a river is
//! no more a tile rule than a city or a road is: it belongs here, with the rest
//! of what needs the whole world, rather than in [`crate::gameplay::terrain`].
//!
//! Rivers are cut by particles rather than by a flow field. A flow field would
//! want an elevation grid for every one of the world's 16 M tiles plus a
//! depression fill over it; a hundred particles walking downhill cost a couple
//! of thousand noise samples each and need no such grid.
//!
//! The particles descend a lattice anchored on the **world origin**, which is
//! the same trick [`crate::gameplay::road`] uses and for the same reason: two
//! particles that pass through the same place step between the same nodes, so
//! their paths coincide exactly instead of running parallel a tile apart. That
//! is what makes a tributary join a trunk rather than braid alongside it, and it
//! is what lets flow accumulate at all.
//!
//! A particle that runs out of downhill does not stop — it floods the basin
//! until it finds the rim's low point and carries on from there. That is what
//! turns the pits fbm is full of into either nothing at all or a lake, depending
//! on how much water the basin actually holds, and it is why there is no climb
//! tolerance here: flooding is the mechanism that gets a river past a dip.

use std::{cmp::Ordering, collections::BinaryHeap, collections::HashMap};

use bevy::prelude::*;

use crate::gameplay::{
    noise::hash2,
    plan::WorldPlanConfig,
    terrain::{TerrainConfig, TerrainKind, TerrainSampler},
    world::{
        TileEdit, WORLD_CHUNKS, WORLD_TILES, WorldSnapshot, chunk_index_of_tile, tile_in_world,
    },
};

/// Widest a river may get, however much flow it carries. Past four tiles a
/// channel stops reading as a river and starts reading as a lake with a current.
pub const MAX_RIVER_WIDTH: u32 = 4;

/// Gives the river sources their own patch of the hash space, so a spring's
/// jitter is unrelated to a city's.
const RIVER_SOURCE_SALT: i32 = 0x63b2_59d7u32 as i32;

/// A node with no successor / a node in no lake.
const NONE: u32 = u32::MAX;

/// The tile edits every river and lake in the world implies, in batches that
/// each fall inside one chunk.
///
/// Batched rather than flat because a world of rivers is on the order of 10^5
/// edits, and [`crate::gameplay::world::WorldMap::apply_edits`] scans its
/// touched-chunk list linearly — free for one road, quadratic for this. The
/// batches are also what lets the stamping be spread over frames.
pub struct RiverPlan {
    pub by_chunk: Vec<Vec<TileEdit>>,
}

/// Cuts every river in the world.
///
/// Pure in `(TerrainConfig, WorldPlanConfig, world)`: the springs come out in
/// scan order and each is walked to its end before the next starts, so the same
/// seed gives the same rivers on every run and every platform.
pub fn plan_rivers(
    terrain: &TerrainConfig,
    config: &WorldPlanConfig,
    world: &WorldSnapshot,
) -> RiverPlan {
    let mut lattice = Lattice::new(terrain, config);

    for (index, spring) in springs(terrain, config, world).into_iter().enumerate() {
        // Particle ids start at 1, so that 0 can mean "no particle has been
        // here" in the visited marks.
        lattice.descend(spring, index as u32 + 1, config, world);
    }

    lattice.stamp(config, world)
}

/// Where the rivers rise: one candidate per square cell, kept only if it is a
/// mountain and the rain there clears the threshold.
///
/// The cell is what bounds the count and spreads the springs out; humidity is
/// what decides which mountains are the wet ones.
fn springs(terrain: &TerrainConfig, config: &WorldPlanConfig, world: &WorldSnapshot) -> Vec<IVec2> {
    let cell = config.river_source_cell_tiles.max(1) as i32;
    let cells = WORLD_TILES.as_ivec2() / cell;
    let sampler = terrain.sampler();

    let mut springs = Vec::new();
    for cy in 0..cells.y {
        for cx in 0..cells.x {
            let h = hash2(cx ^ RIVER_SOURCE_SALT, cy);
            let jitter = IVec2::new(
                (h & 0xffff) as i32 % cell,
                ((h >> 16) & 0xffff) as i32 % cell,
            );
            let tile = IVec2::new(cx, cy) * cell + jitter;

            // A river rises in the mountains or not at all.
            if world.tile(tile) != Some(TerrainKind::Mountain) {
                continue;
            }
            if sampler.humidity(tile.x as f32, tile.y as f32) < terrain.river_source_threshold {
                continue;
            }
            springs.push(tile);
        }
    }
    springs
}

/// How wide a channel carrying this much flow is, in tiles.
///
/// Flow is unbounded — every particle upstream ends up crossing the trunk — so
/// the cap is what stops the mouth of a big river from spreading into an inland
/// sea.
pub fn channel_width(flow: u32, config: &WorldPlanConfig) -> u32 {
    (1 + flow / config.river_flow_per_width.max(1)).min(MAX_RIVER_WIDTH)
}

/// A basin the water filled, and where it spilled out of — `None` if it filled
/// to its cap without finding a way out, which is what a closed lake is.
struct Lake {
    nodes: Vec<u32>,
    outlet: Option<u32>,
    /// Whether it held enough water to be worth drawing. Every basin a particle
    /// fills is recorded, because the filling is what lets the next particle
    /// past it; only the ones over `river_lake_min_tiles` become tiles.
    drawn: bool,
}

/// The grid the particles walk, and everything they leave behind on it.
///
/// Anchored on the world origin: node `n` sits at tile `n * stride`, whichever
/// particle is asking. The per-node arrays are dense rather than hashed — at the
/// default stride that is 1024x1024 nodes and about 20 MB of scratch for the
/// duration of the planning task, against a hash lookup on every one of the tens
/// of thousands of steps a world of rivers takes.
struct Lattice {
    stride: i32,
    /// Nodes along each axis.
    size: i32,
    /// The same sampler the visible terrain was generated from, so a particle
    /// descends the hills that are actually drawn.
    sampler: TerrainSampler,
    /// Height of the *water surface* at a node, which is the terrain until a
    /// basin fills and then the level it filled to. Sampled on demand — the
    /// particles only ever visit a thin slice of the world, so sampling all of
    /// it up front would be most of a second wasted. `NaN` means "not yet".
    elevation: Vec<f32>,
    /// How many particles have run from this node to its successor.
    flow: Vec<u32>,
    /// Where the water at this node goes next. Fixed the first time a particle
    /// leaves the node.
    next: Vec<u32>,
    lake_of: Vec<u32>,
    /// Which particle was last here, so one cannot walk a circle it spilled into.
    visited: Vec<u32>,
    lakes: Vec<Lake>,
}

impl Lattice {
    fn new(terrain: &TerrainConfig, config: &WorldPlanConfig) -> Self {
        let stride = config.river_step_tiles.max(1) as i32;
        let size = WORLD_TILES.x as i32 / stride;
        let count = (size * size) as usize;

        Self {
            stride,
            size,
            sampler: terrain.sampler(),
            elevation: vec![f32::NAN; count],
            flow: vec![0; count],
            next: vec![NONE; count],
            lake_of: vec![NONE; count],
            visited: vec![0; count],
            lakes: Vec::new(),
        }
    }

    fn index_of(&self, node: IVec2) -> u32 {
        (node.y * self.size + node.x) as u32
    }

    fn node_at(&self, index: u32) -> IVec2 {
        IVec2::new(index as i32 % self.size, index as i32 / self.size)
    }

    fn tile_at(&self, index: u32) -> IVec2 {
        self.node_at(index) * self.stride
    }

    /// The node nearest a tile. Rounding rather than flooring means a spring
    /// starts at the closest node rather than the one below and left of it.
    fn node_of(&self, tile: IVec2) -> u32 {
        let node = (tile.as_vec2() / self.stride as f32)
            .round()
            .as_ivec2()
            .clamp(IVec2::ZERO, IVec2::splat(self.size - 1));
        self.index_of(node)
    }

    /// Whether a node is on the outermost ring. A particle that gets this far
    /// has run off the map, and there is no more world to descend into.
    fn on_the_edge(&self, index: u32) -> bool {
        let node = self.node_at(index);
        node.x == 0 || node.y == 0 || node.x == self.size - 1 || node.y == self.size - 1
    }

    /// The elevation the *terrain* has here, taken from the noise field rather
    /// than from the world — `WorldMap` records only which band a tile fell in,
    /// and a river needs to know which of two lowland tiles is lower.
    fn elevation_at(&mut self, index: u32) -> f32 {
        let cached = self.elevation[index as usize];
        if !cached.is_nan() {
            return cached;
        }
        let tile = self.tile_at(index);
        let sampled = self.sampler.elevation(tile.x as f32, tile.y as f32);
        self.elevation[index as usize] = sampled;
        sampled
    }

    fn neighbours(&self, index: u32) -> [Option<u32>; 8] {
        let here = self.node_at(index);
        let mut out = [None; 8];
        let mut slot = 0;
        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                if (dx, dy) == (0, 0) {
                    continue;
                }
                let node = here + IVec2::new(dx, dy);
                out[slot] = (node.cmpge(IVec2::ZERO).all()
                    && node.cmplt(IVec2::splat(self.size)).all())
                .then(|| self.index_of(node));
                slot += 1;
            }
        }
        out
    }

    fn lowest_neighbour(&mut self, index: u32) -> Option<u32> {
        let mut best: Option<(f32, u32)> = None;
        for neighbour in self.neighbours(index).into_iter().flatten() {
            let elevation = self.elevation_at(neighbour);
            // The `<` and not `<=` is the tie-break: with equal elevations the
            // lower node index wins, so the outcome cannot depend on the scan.
            if best.is_none_or(|(lowest, _)| elevation < lowest) {
                best = Some((elevation, neighbour));
            }
        }
        best.map(|(_, node)| node)
    }

    /// Walks one particle from its spring to the sea, the world edge, or a lake
    /// it cannot get out of, recording the flow it leaves on the way.
    fn descend(
        &mut self,
        spring: IVec2,
        particle: u32,
        config: &WorldPlanConfig,
        world: &WorldSnapshot,
    ) {
        let mut node = self.node_of(spring);

        for _ in 0..config.river_max_steps {
            if self.visited[node as usize] == particle {
                // A spill can only put a particle somewhere it has not been, so
                // reaching this means the terrain has talked it into a circle.
                return;
            }
            self.visited[node as usize] = particle;

            if self.on_the_edge(node) {
                return;
            }

            // A lake an earlier particle left: this water joins it and carries on
            // from the same outlet, so a chain of lakes on one river takes its
            // flow all the way down instead of losing it at the first shore.
            if self.lake_of[node as usize] != NONE {
                match self.lakes[self.lake_of[node as usize] as usize].outlet {
                    Some(outlet) => {
                        node = outlet;
                        continue;
                    }
                    None => return,
                }
            }

            let Some(lowest) = self.lowest_neighbour(node) else {
                return;
            };
            let downhill = self.elevation_at(lowest) < self.elevation_at(node);
            if !downhill {
                // Nowhere lower to go: fill the basin and leave by its rim.
                match self.flood(node, config) {
                    Some(outlet) => {
                        node = outlet;
                        continue;
                    }
                    None => return,
                }
            }

            self.flow[node as usize] += 1;
            self.next[node as usize] = lowest;
            node = lowest;

            // The sea, which is where a river is supposed to end.
            if world
                .tile(self.tile_at(node))
                .is_none_or(TerrainKind::is_water)
            {
                return;
            }
        }
    }

    /// Floods a basin from a node with nowhere lower to go, and reports where it
    /// spills.
    ///
    /// Always taking the lowest node on the frontier next means the water level
    /// only ever rises, so the first node that comes up *below* that level is a
    /// way out of the basin — which is the cheapest thing that finds a rim's low
    /// point without an elevation grid of the whole world.
    ///
    /// The filled basin is then **raised to the level it filled to**, because
    /// that is what a full basin is: a flat sheet of water whose surface sits at
    /// the rim. Without that the particle would spill out to the rim and, on its
    /// very next step, walk straight back down into the hollow it had just
    /// filled — which is what a lake stops happening.
    ///
    /// A basin that spills before it holds `river_lake_min_tiles` is filled and
    /// raised like any other but never drawn. That threshold is load-bearing:
    /// fbm at this stride is full of dips a tile or two deep, and a pond at each
    /// would turn every river into a string of beads.
    fn flood(&mut self, start: u32, config: &WorldPlanConfig) -> Option<u32> {
        let per_node = (self.stride * self.stride) as u32;
        let max_nodes = (config.river_lake_max_tiles / per_node).max(1) as usize;

        let mut heap = BinaryHeap::new();
        let mut seen = vec![start];
        let mut flooded: Vec<u32> = Vec::new();
        let mut level = f32::NEG_INFINITY;
        let mut outlet = None;

        heap.push(Rise {
            elevation: self.elevation_at(start),
            node: start,
        });

        while let Some(Rise { elevation, node }) = heap.pop() {
            if elevation < level {
                outlet = Some(node);
                break;
            }
            level = elevation;
            flooded.push(node);
            if flooded.len() >= max_nodes {
                break;
            }

            for neighbour in self.neighbours(node).into_iter().flatten() {
                if seen.contains(&neighbour) {
                    continue;
                }
                seen.push(neighbour);
                heap.push(Rise {
                    elevation: self.elevation_at(neighbour),
                    node: neighbour,
                });
            }
        }

        // The surface of the water, flat at the height it spilled from. This is
        // what stops the next step walking back down into the basin, and what
        // lets a later particle cross it rather than filling it again.
        let id = self.lakes.len() as u32;
        for &node in &flooded {
            self.elevation[node as usize] = level;
            self.lake_of[node as usize] = id;
        }
        self.lakes.push(Lake {
            drawn: flooded.len() as u32 * per_node >= config.river_lake_min_tiles,
            nodes: flooded,
            outlet,
        });

        outlet
    }

    /// Turns the flow counts and the lakes into the tile edits they imply.
    fn stamp(&self, config: &WorldPlanConfig, world: &WorldSnapshot) -> RiverPlan {
        let mut tiles: HashMap<IVec2, TerrainKind> = HashMap::new();

        for index in 0..self.flow.len() as u32 {
            let flow = self.flow[index as usize];
            let next = self.next[index as usize];
            if flow == 0 || next == NONE {
                continue;
            }
            self.paint_channel(&mut tiles, index, next, channel_width(flow, config), world);
        }

        // After the channels, so a river running into a lake ends at its shore
        // rather than crossing it.
        for lake in self.lakes.iter().filter(|lake| lake.drawn) {
            for &node in &lake.nodes {
                self.paint_lake(&mut tiles, node, world);
            }
        }

        self.group_by_chunk(tiles)
    }

    /// Paints one segment of channel, `width` tiles across.
    ///
    /// The widening runs across the segment's *dominant* axis rather than along
    /// its true perpendicular: the line is walked one tile at a time along that
    /// axis, so offsetting across it cannot leave the gaps a diagonal
    /// perpendicular would.
    fn paint_channel(
        &self,
        tiles: &mut HashMap<IVec2, TerrainKind>,
        from: u32,
        to: u32,
        width: u32,
        world: &WorldSnapshot,
    ) {
        let (from, to) = (self.tile_at(from), self.tile_at(to));
        let delta = to - from;
        let steps = delta.x.abs().max(delta.y.abs()).max(1);
        let across = if delta.x.abs() >= delta.y.abs() {
            IVec2::Y
        } else {
            IVec2::X
        };
        // Exactly `width` tiles across, leaning one side for an even width since
        // there is no such thing as a centred four-tile band.
        let first = -((width as i32 - 1) / 2);
        let last = width as i32 / 2;

        for step in 0..=steps {
            let centre = from
                + (delta.as_vec2() * (step as f32 / steps as f32))
                    .round()
                    .as_ivec2();
            for offset in first..=last {
                paint(tiles, centre + across * offset, TerrainKind::River, world);
            }
        }
    }

    /// A lake node covers the tiles it is nearest, which tile the plane exactly.
    fn paint_lake(
        &self,
        tiles: &mut HashMap<IVec2, TerrainKind>,
        node: u32,
        world: &WorldSnapshot,
    ) {
        let centre = self.tile_at(node);
        let half = self.stride / 2;
        for dy in -half..half {
            for dx in -half..half {
                paint(
                    tiles,
                    centre + IVec2::new(dx, dy),
                    TerrainKind::ShallowWater,
                    world,
                );
            }
        }
    }

    /// Buckets the edits by chunk and sorts each bucket. The sort is not
    /// cosmetic: the edits come out of a `HashMap`, whose order is not the same
    /// twice, and the plan has to be.
    fn group_by_chunk(&self, tiles: HashMap<IVec2, TerrainKind>) -> RiverPlan {
        let mut buckets: Vec<Vec<TileEdit>> =
            vec![Vec::new(); WORLD_CHUNKS.element_product() as usize];
        for (tile, kind) in tiles {
            buckets[chunk_index_of_tile(tile)].push(TileEdit { tile, kind });
        }

        let by_chunk = buckets
            .into_iter()
            .filter(|edits| !edits.is_empty())
            .map(|mut edits| {
                edits.sort_unstable_by_key(|edit| (edit.tile.y, edit.tile.x));
                edits
            })
            .collect();

        RiverPlan { by_chunk }
    }
}

/// Water does not re-cut the sea it flows into, and it stops at the world edge.
fn paint(
    tiles: &mut HashMap<IVec2, TerrainKind>,
    tile: IVec2,
    kind: TerrainKind,
    world: &WorldSnapshot,
) {
    if !tile_in_world(tile) {
        return;
    }
    if world.tile(tile).is_none_or(TerrainKind::is_water) {
        return;
    }
    tiles.insert(tile, kind);
}

/// A node waiting to be flooded. `BinaryHeap` is a max-heap, so the ordering is
/// inverted to pop the *lowest* node first — which is what makes the water level
/// rise monotonically and the first node below it an outlet.
struct Rise {
    elevation: f32,
    node: u32,
}

impl Ord for Rise {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .elevation
            .partial_cmp(&self.elevation)
            .unwrap_or(Ordering::Equal)
            // Equal heights are broken by node index, so two basins of the same
            // depth cannot flood differently from one run to the next.
            .then_with(|| other.node.cmp(&self.node))
    }
}

impl PartialOrd for Rise {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Rise {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Rise {}

#[cfg(test)]
mod tests {
    use super::*;

    /// A world sloping steadily down to the west, with a band of sea at the far
    /// end and mountains in the east — so a particle started anywhere in the
    /// mountains has somewhere to go and something to reach.
    fn sloping_world() -> WorldSnapshot {
        WorldSnapshot::from_fn(|tile| {
            if tile.x < 256 {
                TerrainKind::DeepWater
            } else if tile.x < 512 {
                TerrainKind::ShallowWater
            } else if tile.x < 3000 {
                TerrainKind::Grass
            } else {
                TerrainKind::Mountain
            }
        })
    }

    fn edits(plan: &RiverPlan) -> Vec<TileEdit> {
        plan.by_chunk
            .iter()
            .flat_map(|batch| batch.iter().copied())
            .collect()
    }

    #[test]
    fn the_plan_is_the_same_on_every_run() {
        let terrain = TerrainConfig::default();
        let config = WorldPlanConfig::default();
        let world = sloping_world();

        let once = plan_rivers(&terrain, &config, &world);
        let twice = plan_rivers(&terrain, &config, &world);

        let (once, twice) = (edits(&once), edits(&twice));
        assert_eq!(once.len(), twice.len());
        for (a, b) in once.iter().zip(twice.iter()) {
            assert_eq!((a.tile, a.kind), (b.tile, b.kind));
        }
    }

    #[test]
    fn every_spring_is_a_wet_mountain() {
        let terrain = TerrainConfig::default();
        let config = WorldPlanConfig::default();
        let world = sloping_world();
        let sampler = terrain.sampler();

        let springs = springs(&terrain, &config, &world);
        assert!(!springs.is_empty(), "the world has no rivers at all");
        for spring in springs {
            assert_eq!(world.tile(spring), Some(TerrainKind::Mountain));
            assert!(
                sampler.humidity(spring.x as f32, spring.y as f32)
                    >= terrain.river_source_threshold
            );
        }
    }

    /// A river is cut into land. Running one over the sea would put a channel
    /// through the middle of the ocean it is supposed to end at.
    #[test]
    fn no_water_is_ever_re_cut() {
        let terrain = TerrainConfig::default();
        let config = WorldPlanConfig::default();
        let world = sloping_world();

        for edit in edits(&plan_rivers(&terrain, &config, &world)) {
            assert!(
                !world.tile(edit.tile).expect("inside the world").is_water(),
                "{} was already water",
                edit.tile
            );
        }
    }

    #[test]
    fn width_climbs_with_flow_and_then_stops() {
        let config = WorldPlanConfig::default();
        let per = config.river_flow_per_width;

        assert_eq!(channel_width(0, &config), 1, "a trickle is still a river");
        assert_eq!(channel_width(per - 1, &config), 1);
        assert_eq!(channel_width(per, &config), 2);
        assert_eq!(channel_width(2 * per, &config), 3);
        assert_eq!(channel_width(u32::MAX, &config), MAX_RIVER_WIDTH);
    }

    /// The cap lands on a *segment*, which is why this measures one rather than
    /// counting river tiles on the finished map: a bend or a confluence puts
    /// more river side by side than any single channel is wide, so the map
    /// cannot tell the two apart.
    #[test]
    fn a_segment_is_painted_exactly_as_wide_as_its_flow() {
        let terrain = TerrainConfig::default();
        let config = WorldPlanConfig::default();
        let lattice = Lattice::new(&terrain, &config);
        let world = WorldSnapshot::from_fn(|_| TerrainKind::Grass);
        let stride = config.river_step_tiles as i32;
        let origin = IVec2::splat(2048);

        for width in 1..=MAX_RIVER_WIDTH {
            // East and then north: the widening runs across the segment's
            // dominant axis, so the two directions take different branches.
            for (step, across) in [(IVec2::X, IVec2::Y), (IVec2::Y, IVec2::X)] {
                let mut tiles = HashMap::new();
                lattice.paint_channel(
                    &mut tiles,
                    lattice.node_of(origin),
                    lattice.node_of(origin + step * stride),
                    width,
                    &world,
                );

                // One slice across the middle of the segment, well clear of
                // either end.
                let slice = origin + step * (stride / 2);
                let band = (-8..=8)
                    .filter(|offset| tiles.contains_key(&(slice + across * offset)))
                    .count() as u32;
                assert_eq!(
                    band, width,
                    "a width-{width} segment going {step} came out {band} tiles across"
                );
            }
        }
    }

    /// A blob would mean the painting had run away. There is no measuring width
    /// off the finished map, but there is a ceiling on how much river the
    /// channels can possibly account for.
    #[test]
    fn the_map_holds_no_more_river_than_the_channels_account_for() {
        let terrain = TerrainConfig::default();
        let config = WorldPlanConfig::default();
        let world = sloping_world();
        let mut lattice = Lattice::new(&terrain, &config);

        for (index, spring) in springs(&terrain, &config, &world).into_iter().enumerate() {
            lattice.descend(spring, index as u32 + 1, &config, &world);
        }

        let mut ceiling = 0usize;
        for index in 0..lattice.flow.len() as u32 {
            let flow = lattice.flow[index as usize];
            if flow == 0 || lattice.next[index as usize] == NONE {
                continue;
            }
            // A segment spans one lattice step, so it covers at most `stride`
            // tiles of line, each widened to at most its own width.
            ceiling += (lattice.stride as usize + 1) * channel_width(flow, &config) as usize;
        }

        let river = edits(&lattice.stamp(&config, &world))
            .iter()
            .filter(|edit| edit.kind == TerrainKind::River)
            .count();
        assert!(river > 0, "not one tile of river was cut");
        assert!(
            river <= ceiling,
            "{river} tiles of river from channels that can only account for {ceiling}"
        );
    }

    /// Every basin a particle fills is recorded — that is what lets the next one
    /// cross it — but only the ones holding real water are drawn. Both halves of
    /// that matter: draw them all and every river becomes a string of beads;
    /// record only the drawn ones and a particle walks back down into the dip it
    /// just filled.
    #[test]
    fn only_basins_holding_real_water_become_lakes() {
        let terrain = TerrainConfig::default();
        let config = WorldPlanConfig::default();
        let mut lattice = Lattice::new(&terrain, &config);
        let world = sloping_world();

        for (index, spring) in springs(&terrain, &config, &world).into_iter().enumerate() {
            lattice.descend(spring, index as u32 + 1, &config, &world);
        }
        assert!(!lattice.lakes.is_empty(), "no basin was ever filled");

        let per_node = (lattice.stride * lattice.stride) as u32;
        for lake in &lattice.lakes {
            let tiles = lake.nodes.len() as u32 * per_node;
            assert_eq!(
                lake.drawn,
                tiles >= config.river_lake_min_tiles,
                "a basin of {tiles} tiles was drawn as {}",
                lake.drawn
            );
            // One node over, since the flood only stops once it has crossed the
            // cap rather than before.
            assert!(
                tiles <= config.river_lake_max_tiles + per_node,
                "a basin of {tiles} tiles grew past the maximum"
            );
            // A filled basin is flat, at the level it spilled from — the whole
            // point of raising it.
            let surface = lattice.elevation[lake.nodes[0] as usize];
            for &node in &lake.nodes {
                assert_eq!(lattice.elevation[node as usize], surface);
            }
        }
    }

    /// A batch that straddled two chunks would make `apply_edits` do the very
    /// linear scan the batching exists to avoid, and would spread one chunk's
    /// tiles over two frames.
    #[test]
    fn every_batch_belongs_to_exactly_one_chunk() {
        let terrain = TerrainConfig::default();
        let config = WorldPlanConfig::default();
        let plan = plan_rivers(&terrain, &config, &sloping_world());

        assert!(!plan.by_chunk.is_empty(), "the plan is empty");
        let mut seen = Vec::new();
        for batch in &plan.by_chunk {
            let chunk = chunk_index_of_tile(batch[0].tile);
            assert!(
                batch
                    .iter()
                    .all(|edit| chunk_index_of_tile(edit.tile) == chunk)
            );
            assert!(!seen.contains(&chunk), "chunk {chunk} is stamped twice");
            seen.push(chunk);
        }
    }

    /// The guard against a particle the terrain talks into a circle: a descent
    /// that never terminated would hang the planning task, not just look wrong.
    #[test]
    fn a_descent_always_terminates() {
        let terrain = TerrainConfig::default();
        let config = WorldPlanConfig::default();
        // A perfectly flat world is the worst case: nothing is ever downhill, so
        // the particle floods, spills, and must still run out of somewhere to go.
        let world = WorldSnapshot::from_fn(|_| TerrainKind::Mountain);
        let mut lattice = Lattice::new(&terrain, &config);

        lattice.descend(IVec2::splat(2048), 1, &config, &world);

        let visited = lattice.visited.iter().filter(|&&mark| mark == 1).count();
        assert!(
            visited <= config.river_max_steps as usize,
            "the particle took more steps than it was allowed"
        );
    }
}
