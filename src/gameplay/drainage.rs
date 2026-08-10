//! Dry valleys: the branching network a landscape drains through, drawn as a
//! change of ground cover rather than as water.
//!
//! This is the half of gh-14 that noise cannot do. The substrate layers in
//! [`crate::gameplay::terrain`] are *fields* — they vary at a scale, but they do
//! not branch, and nothing in them knows which way is downhill. A drainage pattern
//! is the one piece of mid-scale structure that is strictly ordered by height, so
//! it agrees with the relief and with the tint pass for free, and it is what lets
//! you tell which way the ground falls by looking at the trees.
//!
//! It is not a river, and the difference is the whole design:
//!
//! - **A particle never floods.** The river particle that lands in a pit
//!   priority-floods the basin, raises it, and walks on; that is correct for water
//!   and it is why the default world carries 238k tiles of lake against 26.6k of
//!   river. A wadi that reaches a closed depression simply ends there, which is
//!   what a real one does. Dropping the flood removes the expensive half of the
//!   river stage *and* removes any way for this pass to add a tile of standing
//!   water — which is what lets it land before the gh-9 lake retune rather than
//!   after it.
//! - **A spring rises anywhere on land.** A river rises on `Mountain` above a
//!   humidity bar, which is exactly why valleys only exist in the mountains today.
//!   Dry country has *more* drainage expression than wet, not less, because there
//!   is nothing growing over it.
//! - **It paints cover, never water and never height.** A damp channel promotes
//!   the tile it crosses one rung up a moisture ladder. Because it writes only
//!   kinds, no drainage edit costs a heightmap upload, which is the property
//!   [`crate::gameplay::tint`] rests on.
//!
//! The lattice is anchored on the **world origin**, the same trick
//! [`crate::gameplay::river`] and [`crate::gameplay::road`] use and for the same
//! reason: two particles that pass through a place step between the same nodes, so
//! their paths coincide instead of running parallel a tile apart. That coincidence
//! is the only reason flow accumulates and a tributary joins a trunk.

use std::collections::HashMap;

use bevy::prelude::*;

use crate::gameplay::{
    noise::hash2,
    plan::WorldPlanConfig,
    terrain::{TerrainConfig, TerrainKind, TerrainSampler},
    world::{
        TileEdit, WORLD_CHUNKS, WORLD_TILES, WorldSnapshot, chunk_index_of_tile, tile_in_world,
    },
};

/// Gives the valley heads their own patch of the hash space, so a drainage
/// candidate's jitter is unrelated to a spring's or a city's.
const DRAIN_SOURCE_SALT: i32 = 0x1f83_d9abu32 as i32;

/// A node with no successor.
const NONE: u32 = u32::MAX;

/// The tile edits every dry valley in the world implies, in batches that each fall
/// inside one chunk — the same shape [`crate::gameplay::river::RiverPlan`] has, and
/// for the same reason: `apply_edits` scans its touched-chunk list linearly, and
/// the batches are what lets the stamping be spread over frames.
pub struct DrainagePlan {
    pub by_chunk: Vec<Vec<TileEdit>>,
}

/// What a damp channel turns a tile into: one rung up the moisture ladder.
///
/// Everything not named here is returned unchanged, and that single fact is the
/// whole of the "may not touch" rule — the water, the mountain bands, and the three
/// kinds the plan itself stamps are all refused by omission rather than by a list
/// that could fall out of step with one.
///
/// One rung, never two, so the promotion is bounded however much flow crosses a
/// tile: a desert wadi becomes scrub, not woodland.
pub fn dampened(kind: TerrainKind) -> TerrainKind {
    match kind {
        // The desert's two dry rungs both go to scrub — this is the wadi.
        TerrainKind::Gravel | TerrainKind::Sand => TerrainKind::Scrub,
        TerrainKind::Scrub => TerrainKind::Grass,
        // A gallery wood down a valley floor, which is what a lowland drainage
        // line looks like from the air.
        TerrainKind::Grass => TerrainKind::Forest,
        TerrainKind::Marsh => TerrainKind::Reed,
        other => other,
    }
}

/// Cuts every dry valley in the world.
///
/// Pure in `(TerrainConfig, WorldPlanConfig, world)`: the heads come out in scan
/// order and each is walked to its end before the next starts, so the same seed
/// gives the same valleys on every run and every platform.
pub fn plan_drainage(
    sampler: &TerrainSampler,
    terrain: &TerrainConfig,
    config: &WorldPlanConfig,
    world: &WorldSnapshot,
) -> DrainagePlan {
    let mut lattice = Lattice::new(sampler, terrain, config);

    for head in valley_heads(config, world) {
        lattice.descend(head, config, world);
    }

    lattice.stamp(config, world)
}

/// Where the valleys start: one candidate per square cell, kept if it is land the
/// plan has not already claimed.
///
/// No humidity bar and no elevation bar, unlike a spring — a valley head is just a
/// piece of ground that water runs off, and every piece of ground is.
fn valley_heads(config: &WorldPlanConfig, world: &WorldSnapshot) -> Vec<IVec2> {
    let cell = config.drain_source_cell_tiles.max(1) as i32;
    let cells = WORLD_TILES.as_ivec2() / cell;

    let mut heads = Vec::new();
    for cy in 0..cells.y {
        for cx in 0..cells.x {
            let h = hash2(cx ^ DRAIN_SOURCE_SALT, cy);
            let jitter = IVec2::new(
                (h & 0xffff) as i32 % cell,
                ((h >> 16) & 0xffff) as i32 % cell,
            );
            let tile = IVec2::new(cx, cy) * cell + jitter;

            let Some(kind) = world.tile(tile) else {
                continue;
            };
            // Land the plan has not already spoken for. A head on a river or in a
            // town would only ever draw over what is already there.
            if kind.is_water()
                || matches!(
                    kind,
                    TerrainKind::River | TerrainKind::Town | TerrainKind::Road
                )
            {
                continue;
            }
            heads.push(tile);
        }
    }
    heads
}

/// The grid the particles walk, and the flow they leave on it.
///
/// Deliberately thinner than the river stage's: no lake table, no visited marks and
/// no water-surface fixups, because a particle that only ever steps strictly
/// downhill cannot revisit a node and never needs a basin filled. That is three of
/// the six per-node arrays gone.
struct Lattice {
    stride: i32,
    size: i32,
    /// The same sampler the visible terrain was generated from, so a particle
    /// descends the hills that are actually drawn — including the ones the
    /// lithology and dune layers put there.
    sampler: TerrainSampler,
    /// Sampled on demand: the particles visit a thin slice of the world, so
    /// sampling all of it up front would be most of a second wasted. `NaN` means
    /// "not yet".
    elevation: Vec<f32>,
    /// How many particles have run from this node to its successor.
    flow: Vec<u32>,
    next: Vec<u32>,
}

impl Lattice {
    fn new(sampler: &TerrainSampler, _terrain: &TerrainConfig, config: &WorldPlanConfig) -> Self {
        // Its own stride, and a much coarser one than the river's — see
        // `drain_step_tiles`. Still anchored on the world origin, which is the part
        // that has to be shared: it is what makes two particles crossing the same
        // ground step between the same nodes, and so what lets flow accumulate.
        let stride = config.drain_step_tiles.max(1) as i32;
        let size = WORLD_TILES.x as i32 / stride;
        let count = (size * size) as usize;

        Self {
            stride,
            size,
            sampler: sampler.clone(),
            elevation: vec![f32::NAN; count],
            flow: vec![0; count],
            next: vec![NONE; count],
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

    fn node_of(&self, tile: IVec2) -> u32 {
        let node = (tile.as_vec2() / self.stride as f32)
            .round()
            .as_ivec2()
            .clamp(IVec2::ZERO, IVec2::splat(self.size - 1));
        self.index_of(node)
    }

    fn on_the_edge(&self, index: u32) -> bool {
        let node = self.node_at(index);
        node.x == 0 || node.y == 0 || node.x == self.size - 1 || node.y == self.size - 1
    }

    /// The elevation the *terrain* has here, taken from the noise rather than from
    /// the world — `WorldMap` records only which band a tile fell in, and a valley
    /// needs to know which of two lowland tiles is lower.
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
            // `<` and not `<=`: with equal elevations the lower node index wins, so
            // the outcome cannot depend on the scan order.
            if best.is_none_or(|(lowest, _)| elevation < lowest) {
                best = Some((elevation, neighbour));
            }
        }
        best.map(|(_, node)| node)
    }

    /// Walks one particle from its valley head down to the sea, a river, a pit or
    /// the world edge, recording the flow it leaves on the way.
    ///
    /// Every step is *strictly* downhill, which is what makes the visited marks the
    /// river stage needs unnecessary here: a strictly decreasing walk cannot return
    /// to a node it has left, so no cycle is reachable and the step cap is a
    /// backstop rather than the termination argument.
    fn descend(&mut self, head: IVec2, config: &WorldPlanConfig, world: &WorldSnapshot) {
        let mut node = self.node_of(head);

        for _ in 0..config.drain_max_steps {
            if self.on_the_edge(node) {
                return;
            }

            let Some(lowest) = self.lowest_neighbour(node) else {
                return;
            };
            if self.elevation_at(lowest) >= self.elevation_at(node) {
                // Nowhere lower to go. A river would flood the basin from here; a
                // dry valley ends, and that is the single rule that keeps this pass
                // from adding a drop of water to the world.
                return;
            }

            self.flow[node as usize] += 1;
            self.next[node as usize] = lowest;
            node = lowest;

            // Arrived: the sea, a lake, or a channel that already carries water.
            match world.tile(self.tile_at(node)) {
                None => return,
                Some(kind) if kind.is_water() || kind == TerrainKind::River => return,
                Some(_) => {}
            }
        }
    }

    /// How wide a damp channel carrying this much flow is, in tiles.
    ///
    /// Flow below `drain_min_flow` is not drawn at all: every valley head walks a
    /// path, but a path only one particle has ever used is a rill nobody would see,
    /// and drawing them all would return the world to speckle.
    fn channel_width(flow: u32, config: &WorldPlanConfig) -> u32 {
        (flow / config.drain_min_flow.max(1)).clamp(1, config.drain_max_width.max(1))
    }

    /// Turns the flow counts into the cover edits they imply.
    fn stamp(&self, config: &WorldPlanConfig, world: &WorldSnapshot) -> DrainagePlan {
        let mut tiles: HashMap<IVec2, TerrainKind> = HashMap::new();

        for index in 0..self.flow.len() as u32 {
            let flow = self.flow[index as usize];
            let next = self.next[index as usize];
            if flow < config.drain_min_flow.max(1) || next == NONE {
                continue;
            }
            self.paint_channel(
                &mut tiles,
                index,
                next,
                Self::channel_width(flow, config),
                world,
            );
        }

        self.group_by_chunk(tiles)
    }

    /// Paints one segment of channel, `width` tiles across.
    ///
    /// The widening runs across the segment's *dominant* axis rather than its true
    /// perpendicular, the same way the river stage's does and for the same reason:
    /// the line is walked one tile at a time along that axis, so offsetting across
    /// it cannot leave the gaps a diagonal perpendicular would.
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
        let first = -((width as i32 - 1) / 2);
        let last = width as i32 / 2;

        for step in 0..=steps {
            let centre = from
                + (delta.as_vec2() * (step as f32 / steps as f32))
                    .round()
                    .as_ivec2();
            for offset in first..=last {
                paint(tiles, centre + across * offset, world);
            }
        }
    }

    /// Buckets the edits by chunk and sorts each bucket. The sort is not cosmetic:
    /// the edits come out of a `HashMap`, whose order is not the same twice, and
    /// the plan has to be.
    fn group_by_chunk(&self, tiles: HashMap<IVec2, TerrainKind>) -> DrainagePlan {
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

        DrainagePlan { by_chunk }
    }
}

/// Promotes one tile, if the ladder has anywhere to promote it to.
///
/// Reads the kind from the **snapshot** rather than from what has already been
/// painted, which is what bounds the promotion at one rung: two channels crossing
/// the same tile both compute the same answer from the same input, so the result
/// does not depend on how many of them there were or which came first.
fn paint(tiles: &mut HashMap<IVec2, TerrainKind>, tile: IVec2, world: &WorldSnapshot) {
    if !tile_in_world(tile) {
        return;
    }
    let Some(under) = world.tile(tile) else {
        return;
    };
    let damp = dampened(under);
    if damp != under {
        tiles.insert(tile, damp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::gameplay::terrain::shared_test_sampler;
    use crate::gameplay::{terrain::TERRAIN_KIND_COUNT, world::WorldSnapshot};

    /// Every kind, so the ladder can be checked as a total function rather than on
    /// the handful of entries someone remembered to list.
    const KINDS: [TerrainKind; TERRAIN_KIND_COUNT as usize] = [
        TerrainKind::Forest,
        TerrainKind::ShallowWater,
        TerrainKind::Grass,
        TerrainKind::Town,
        TerrainKind::Mountain,
        TerrainKind::DeepWater,
        TerrainKind::Road,
        TerrainKind::River,
        TerrainKind::Sand,
        TerrainKind::Snow,
        TerrainKind::Rock,
        TerrainKind::Marsh,
        TerrainKind::Scrub,
        TerrainKind::Gravel,
        TerrainKind::Reed,
        TerrainKind::Farmland,
    ];

    fn plan() -> (TerrainConfig, WorldPlanConfig, WorldSnapshot, DrainagePlan) {
        let terrain = TerrainConfig::default();
        let config = WorldPlanConfig::default();
        let world = WorldSnapshot::generated(&terrain, shared_test_sampler());
        let plan = plan_drainage(shared_test_sampler(), &terrain, &config, &world);
        (terrain, config, world, plan)
    }

    fn edits(plan: &DrainagePlan) -> Vec<TileEdit> {
        plan.by_chunk.iter().flatten().copied().collect()
    }

    /// The rule the whole stage is built around. gh-9 is open because the river
    /// stage's basin filling puts 238k tiles of lake in the world; this pass walks
    /// the same terrain with the same particles and must not add one.
    #[test]
    #[ignore = "generates the whole 4096x4096 world"]
    fn the_drainage_stage_adds_no_water_at_all() {
        let (_, _, _, plan) = plan();
        for edit in edits(&plan) {
            assert!(
                !edit.kind.is_water() && edit.kind != TerrainKind::River,
                "the drainage stage laid {:?} at {}",
                edit.kind,
                edit.tile
            );
        }
    }

    /// It may not overwrite what the terrain or an earlier stage decided: the
    /// mountain bands, the water, and the three kinds the plan stamps itself.
    #[test]
    #[ignore = "generates the whole 4096x4096 world"]
    fn a_channel_never_overwrites_what_it_may_not_touch() {
        let (_, _, world, plan) = plan();
        for edit in edits(&plan) {
            let under = world.tile(edit.tile).expect("inside the world");
            assert!(
                !under.is_water()
                    && !matches!(
                        under,
                        TerrainKind::River
                            | TerrainKind::Town
                            | TerrainKind::Road
                            | TerrainKind::Mountain
                            | TerrainKind::Rock
                            | TerrainKind::Snow
                    ),
                "a channel overwrote {under:?} at {}",
                edit.tile
            );
        }
    }

    /// The kinds a damp channel is *for*: every one of them has somewhere to go, or
    /// the stage would run over a desert and change nothing.
    #[test]
    fn every_dry_kind_has_a_damper_form() {
        for kind in [
            TerrainKind::Gravel,
            TerrainKind::Sand,
            TerrainKind::Scrub,
            TerrainKind::Grass,
            TerrainKind::Marsh,
        ] {
            assert_ne!(dampened(kind), kind, "{kind:?} is not promoted at all");
        }
    }

    /// The ladder is well founded: following it from any kind reaches a fixed point
    /// rather than a cycle.
    ///
    /// The stage computes every promotion from the snapshot, so it cannot iterate
    /// this in the first place — but "one rung per run" is a claim about the
    /// *caller*, and this is the claim about the mapping that makes it safe. A
    /// two-cycle here would mean a tile's kind depended on how many channels
    /// happened to cross it.
    #[test]
    fn the_moisture_ladder_terminates_rather_than_cycling() {
        for kind in KINDS {
            let mut seen = vec![kind];
            let mut here = kind;
            for _ in 0..TERRAIN_KIND_COUNT {
                let next = dampened(here);
                if next == here {
                    break;
                }
                assert!(
                    !seen.contains(&next),
                    "the ladder cycles: {kind:?} returns to {next:?}"
                );
                seen.push(next);
                here = next;
            }
            assert_eq!(
                dampened(here),
                here,
                "following the ladder from {kind:?} never settles"
            );
        }
    }

    /// Everything the terrain draws above the lowland band, plus the plan's own
    /// three kinds, must be fixed points of the ladder — that is how the "may not
    /// touch" rule is enforced, so it is worth asserting directly.
    #[test]
    fn every_kind_the_stage_may_not_touch_is_a_fixed_point() {
        for kind in [
            TerrainKind::DeepWater,
            TerrainKind::ShallowWater,
            TerrainKind::River,
            TerrainKind::Town,
            TerrainKind::Road,
            TerrainKind::Mountain,
            TerrainKind::Rock,
            TerrainKind::Snow,
            TerrainKind::Forest,
            TerrainKind::Reed,
        ] {
            assert_eq!(dampened(kind), kind, "{kind:?} is not a fixed point");
        }
    }

    /// A channel is only drawn where enough particles agreed on it, and a wider one
    /// needs proportionally more. Without the floor every valley head would draw
    /// its own rill and the world would go back to speckle.
    #[test]
    fn width_needs_flow_and_then_stops() {
        let config = WorldPlanConfig::default();
        let min = config.drain_min_flow;
        assert_eq!(Lattice::channel_width(min, &config), 1);
        assert_eq!(Lattice::channel_width(2 * min, &config), 2);
        assert_eq!(
            Lattice::channel_width(u32::MAX, &config),
            config.drain_max_width
        );
    }

    /// The same seed has to give the same valleys, or a session would not rebuild
    /// the world it left.
    #[test]
    #[ignore = "generates the whole 4096x4096 world"]
    fn the_plan_is_the_same_on_every_run() {
        let terrain = TerrainConfig::default();
        let config = WorldPlanConfig::default();
        let world = WorldSnapshot::generated(&terrain, shared_test_sampler());

        let first = edits(&plan_drainage(
            shared_test_sampler(),
            &terrain,
            &config,
            &world,
        ));
        let second = edits(&plan_drainage(
            shared_test_sampler(),
            &terrain,
            &config,
            &world,
        ));
        assert_eq!(first.len(), second.len());
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!((a.tile, a.kind), (b.tile, b.kind));
        }
    }

    /// Every batch has to fall inside one chunk, or the frame-spread stamping would
    /// touch a chunk it had already finished with.
    #[test]
    #[ignore = "generates the whole 4096x4096 world"]
    fn every_batch_belongs_to_exactly_one_chunk() {
        let (_, _, _, plan) = plan();
        for batch in &plan.by_chunk {
            let chunk = chunk_index_of_tile(batch[0].tile);
            assert!(batch.iter().all(|e| chunk_index_of_tile(e.tile) == chunk));
        }
    }

    /// Where the three drainage figures come from. Density against the two knobs
    /// that set it, as a share of the world — with roads at 0.24% as the mark for
    /// "reads as a feature of the map rather than as speckle", the same yardstick
    /// `river_source_cell_tiles` was chosen against.
    ///
    /// The two columns are not interchangeable. Tightening the cell adds *heads*,
    /// which lengthens the tributary network; lowering the flow floor draws paths
    /// fewer descents agreed on, which adds isolated rills rather than valleys. So
    /// the pair at the top of the useful band — 0.227% at a floor of 2 against
    /// 0.199% at a floor of 3 — are different pictures at the same density, and the
    /// second is the one that branches.
    ///
    /// `cargo test --release -- --ignored --nocapture the_drainage_density`
    #[test]
    #[ignore]
    fn the_drainage_density_against_its_two_knobs() {
        let terrain = TerrainConfig::default();
        let base = WorldPlanConfig::default();
        let world = WorldSnapshot::generated(&terrain, shared_test_sampler());
        println!("\n  cell  step  minflow   tiles   % of world");
        for cell in [48u32, 32, 24, 16] {
            for step in [16u32, 24] {
                for min_flow in [2u32, 3] {
                    let config = WorldPlanConfig {
                        drain_source_cell_tiles: cell,
                        drain_step_tiles: step,
                        drain_min_flow: min_flow,
                        ..base.clone()
                    };
                    let plan = plan_drainage(shared_test_sampler(), &terrain, &config, &world);
                    let n: usize = plan.by_chunk.iter().map(|b| b.len()).sum();
                    println!(
                        "  {cell:>4}  {step:>4}  {min_flow:>7}  {n:>6}  {:>9.3}%",
                        n as f64 / (WORLD_TILES.x as f64 * WORLD_TILES.y as f64) * 100.0
                    );
                }
            }
        }
        println!();
    }

    /// What the stage actually produces on the world the game generates. Not a
    /// check — it is where the config doc comments' figures come from.
    ///
    /// `cargo test --release -- --ignored --nocapture the_default_config_measures_the_drainage`
    #[test]
    #[ignore]
    fn the_default_config_measures_the_drainage_network() {
        let (_, config, world, plan) = plan();
        let edits = edits(&plan);

        let mut by_kind = [0usize; 16];
        for edit in &edits {
            by_kind[edit.kind.tileset_index() as usize] += 1;
        }

        println!(
            "\n{} tiles of dry valley across {} chunks — {} frames of stamping",
            edits.len(),
            plan.by_chunk.len(),
            plan.by_chunk
                .len()
                .div_ceil(config.drain_chunks_stamped_per_frame.max(1) as usize),
        );
        println!(
            "  {:.3}% of the world, against roads at ~0.24%",
            edits.len() as f64 / (WORLD_TILES.x as f64 * WORLD_TILES.y as f64) * 100.0
        );

        for kind in [
            TerrainKind::Scrub,
            TerrainKind::Grass,
            TerrainKind::Forest,
            TerrainKind::Reed,
        ] {
            println!(
                "  {kind:>8?}  {:>7}",
                by_kind[kind.tileset_index() as usize]
            );
        }

        // How much habitable land the wadis added, which is the visible consequence
        // of Scrub being habitable.
        let gained = edits
            .iter()
            .filter(|e| {
                e.kind.is_habitable()
                    && !world.tile(e.tile).expect("inside the world").is_habitable()
            })
            .count();
        println!("  {gained} tiles became habitable that were not\n");
    }
}
