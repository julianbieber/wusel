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
//!
//! **A course is not the steepest way down.** At most nodes several neighbours
//! are below, not one, so which of them the water takes is free shape: it costs
//! nothing against the rule that a river never climbs. Taking the steepest spends
//! that freedom on the only choice that reads badly — on a slope it locks onto
//! one of the eight lattice directions and runs dead straight, and where the fall
//! direction falls between two of them it flips back and forth and comes out as a
//! staircase with 4-tile teeth. So a step is scored instead, on three terms:
//!
//! - **descent**, per tile travelled and measured against a fixed reference drop
//!   rather than against the best step available. That is what makes the terrain
//!   decide the shape rather than the config: steep ground swamps the other two
//!   terms and the river runs near the fall line, gentle ground lets them lead.
//! - **persistence**, how far the step turns off the heading, which is what stops
//!   the staircase.
//! - **meander**, how far the step leans to one side, signed by a low-frequency
//!   field sampled at the node. Its sign holds for tens of tiles and then
//!   reverses — left, then right, then left — and that alternation is the bend.
//!
//! The meander field is keyed on **position and seed only**, never on the
//! particle. That is what keeps the stage deterministic, and it is also what lets
//! two particles reaching the same ground lean the same way rather than fraying
//! apart.
//!
//! Which leaves one thing the heading breaks and has to pay back. Two particles
//! that meet now carry different headings, so they would score the same node
//! differently and run on a tile apart — braiding, where anchoring the lattice on
//! the world origin exists to make them fuse. Hence a node's successor is fixed
//! by the **first** particle to leave it and followed by every later one: a
//! tributary joining a trunk becomes the trunk, which is what a confluence is.

use std::{cmp::Ordering, collections::BinaryHeap, collections::HashMap};

use bevy::prelude::*;

use crate::gameplay::{
    noise::{SignedNoiseField, hash2},
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

/// And the meander field its own, which matters more than the usual reason. A
/// bias correlated with the elevation field would put every bend in the same
/// place as the hill that already decides the course, and the two would cancel
/// instead of compounding.
const RIVER_MEANDER_SALT: u32 = 0x1f4a_c0d3;

/// Octaves in the meander field. Two, because what is wanted from it is a sign
/// that holds over tens of tiles; a finer octave only adds a wobble smaller than
/// the lattice step can express.
const MEANDER_OCTAVES: u32 = 2;

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
    sampler: &TerrainSampler,
    terrain: &TerrainConfig,
    config: &WorldPlanConfig,
    world: &WorldSnapshot,
) -> RiverPlan {
    let mut lattice = Lattice::new(sampler, terrain, config);

    for (index, spring) in springs(sampler, terrain, config, world)
        .into_iter()
        .enumerate()
    {
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
fn springs(
    sampler: &TerrainSampler,
    terrain: &TerrainConfig,
    config: &WorldPlanConfig,
    world: &WorldSnapshot,
) -> Vec<IVec2> {
    let cell = config.river_source_cell_tiles.max(1) as i32;
    let cells = WORLD_TILES.as_ivec2() / cell;

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
    /// Which way the water leans, per position rather than per particle — the
    /// one piece of state that makes a course bend. It is sampled from the ground
    /// and not stored per node, because it is as cheap to sample as to look up
    /// and the particles only ever visit a thin slice of the world.
    meander: SignedNoiseField,
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
    fn new(sampler: &TerrainSampler, terrain: &TerrainConfig, config: &WorldPlanConfig) -> Self {
        let stride = config.river_step_tiles.max(1) as i32;
        let size = WORLD_TILES.x as i32 / stride;
        let count = (size * size) as usize;

        Self {
            stride,
            size,
            sampler: sampler.clone(),
            meander: SignedNoiseField::new(
                terrain.seed,
                RIVER_MEANDER_SALT,
                config.river_meander_scale,
                MEANDER_OCTAVES,
            ),
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

    /// The unit direction from one node to another, in node space.
    fn heading_between(&self, from: u32, to: u32) -> Vec2 {
        (self.node_at(to) - self.node_at(from))
            .as_vec2()
            .normalize_or_zero()
    }

    /// Picks the step the water takes out of a node, or `None` when every
    /// neighbour is above it and there is nothing to do but flood.
    ///
    /// The candidates are the neighbours that are *not above* this node — every
    /// step that does not climb, rather than the single lowest one. Scoring them
    /// is where a course gets its shape; see the module docs for why each term is
    /// there.
    ///
    /// `heading` is `None` on the first step out of a spring, which leaves only
    /// the descent term and so starts every river down its fall line.
    ///
    /// Ground this particle has already crossed is not a candidate. On a slope
    /// that never comes up — the water is leaving as fast as it can — but on
    /// flat ground the bias is the only thing steering, and it will happily curl
    /// the course into a closed ring and hand it back to the visited guard,
    /// which stops the particle but leaves the ring drawn. Refusing the step
    /// instead means a particle that boxes itself in runs out of candidates and
    /// floods, which is what water with nowhere to go does.
    fn choose_step(
        &mut self,
        index: u32,
        particle: u32,
        heading: Option<Vec2>,
        config: &WorldPlanConfig,
    ) -> Option<u32> {
        let here = self.elevation_at(index);
        let node = self.node_at(index);
        let tile = self.tile_at(index);
        // Sampled once per node and not once per candidate: it is a property of
        // where the water *is*, not of where it is thinking of going.
        let lean = self.meander.sample(tile.x as f32, tile.y as f32);
        // Rotating the heading a quarter turn gives the side the field's sign
        // points to; a step's lean is how far it goes that way.
        let left = heading.map(|h| Vec2::new(-h.y, h.x));
        let reference = config.river_reference_drop.max(f32::EPSILON);

        let mut best: Option<(f32, u32)> = None;
        for neighbour in self.neighbours(index).into_iter().flatten() {
            if self.visited[neighbour as usize] == particle {
                continue;
            }
            let elevation = self.elevation_at(neighbour);
            if elevation > here {
                continue;
            }

            let step = (self.node_at(neighbour) - node).as_vec2();
            let direction = step.normalize_or_zero();
            // Per tile travelled, or a diagonal wins on the length of the step
            // alone — which is a lattice bias of exactly the kind the scoring is
            // here to remove.
            let travelled = step.length() * self.stride as f32;
            let descent = (here - elevation) / travelled / reference;

            let persistence = heading.map_or(0.0, |h| direction.dot(h));
            let meander = left.map_or(0.0, |l| lean * direction.dot(l));

            let score = descent
                + persistence * config.river_heading_weight
                + meander * config.river_meander_weight;

            // `>` and not `>=` is the tie-break, and `neighbours` yields in node
            // order, so an exact tie goes to the lower index and the outcome
            // cannot depend on the scan.
            if best.is_none_or(|(highest, _)| score > highest) {
                best = Some((score, neighbour));
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
        // The direction of the last step taken, which is what the persistence and
        // meander terms are measured against. It rides on the particle rather
        // than on the lattice: two particles crossing the same ground may arrive
        // going different ways, and it is the *ground* that has to agree with
        // itself, not them.
        let mut heading: Option<Vec2> = None;
        // Consecutive steps that did not descend. Reset by any real drop, so this
        // counts a run across a flat and not flats met along the way.
        let mut flat_run = 0u32;

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
                        // Across the lake is not a step, so there is no direction
                        // to carry over and nothing about the far shore that the
                        // near one should decide.
                        heading = None;
                        flat_run = 0;
                        continue;
                    }
                    None => return,
                }
            }

            let step = if self.next[node as usize] != NONE {
                // A course another particle already cut. Following it rather than
                // scoring afresh is what makes a confluence a confluence: with a
                // heading in play, two particles would otherwise leave the same
                // node different ways and braid.
                self.next[node as usize]
            } else {
                let here = self.elevation_at(node);
                let Some(candidate) = self.choose_step(node, particle, heading, config) else {
                    // Everything around is above: fill the basin and leave by its
                    // rim.
                    match self.spill(node, config) {
                        Some(outlet) => {
                            heading = Some(self.heading_between(node, outlet));
                            node = outlet;
                            flat_run = 0;
                            continue;
                        }
                        None => return,
                    }
                };

                if self.elevation_at(candidate) < here {
                    flat_run = 0;
                } else {
                    // A level step is not an uphill step, and letting the water
                    // take it is what lets a river wander across a flood plain
                    // instead of pooling on it. Only so far, though: past the cap
                    // the water is standing rather than moving, and the basin is
                    // flooded from where the particle got to.
                    flat_run += 1;
                    if flat_run > config.river_flat_run_nodes {
                        match self.spill(node, config) {
                            Some(outlet) => {
                                heading = Some(self.heading_between(node, outlet));
                                node = outlet;
                                flat_run = 0;
                                continue;
                            }
                            None => return,
                        }
                    }
                }

                self.next[node as usize] = candidate;
                candidate
            };

            self.flow[node as usize] += 1;
            heading = Some(self.heading_between(node, step));
            node = step;

            // The sea, which is where a river is supposed to end.
            if world
                .tile(self.tile_at(node))
                .is_none_or(TerrainKind::is_water)
            {
                return;
            }
        }
    }

    /// Floods the basin at a node and hands back the node the water leaves from,
    /// having drawn the crossing if there is nothing else to show for it.
    ///
    /// A basin under `river_lake_min_tiles` is filled and spilled through but
    /// never drawn, and that used to leave a **hole in the channel**: the water
    /// went in one side and came out the other with nothing on the map in
    /// between. At the default stride a basin has to reach four nodes to be
    /// drawn, and fbm at this scale is full of dips one and two nodes across, so
    /// those holes were most of why no course on the map ran further than a
    /// handful of nodes. Linking the entry node straight to the outlet draws the
    /// water across the puddle it is in fact flowing across.
    ///
    /// A basin large enough to draw is left unlinked, because it is a lake: the
    /// river arrives at its shore and a new one leaves the far side, which is
    /// what a lake on a river looks like.
    fn spill(&mut self, node: u32, config: &WorldPlanConfig) -> Option<u32> {
        let outlet = self.flood(node, config)?;
        if !self.lakes.last().is_some_and(|lake| lake.drawn) {
            self.flow[node as usize] += 1;
            self.next[node as usize] = outlet;
        }
        Some(outlet)
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
            self.paint_channel(
                &mut tiles,
                index,
                next,
                // The node one further downstream, which is what rounds the
                // corner at `next`. Taken from downstream and never from
                // upstream: a node has many predecessors and only ever one
                // successor, so every branch arriving at a junction is curved
                // against the same chain and they fuse instead of splaying.
                self.next[next as usize],
                channel_width(flow, config),
                config,
                world,
            );
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

    /// Paints one segment of channel, `width` tiles across, as a **curve**.
    ///
    /// The three adjustments above change which nodes the water visits; none of
    /// them can change that consecutive 4-tile segments meet at an angle, and at
    /// this stride that angle is most of what reads as jagged. So a segment is
    /// not a straight line between two node centres: it is a quadratic from
    /// `from`, pulled toward `to`, ending halfway between `to` and `after`.
    ///
    /// That end point is the trick. The next segment starts at `to` heading for
    /// `after`, and this one arrives at their midpoint going the same way, so the
    /// corner at `to` is rounded and the two meet without a kink. The straight
    /// middles get painted twice over, which costs a `HashMap` insert of the same
    /// kind and buys not having to know a node's predecessor.
    ///
    /// With no `after` — the mouth of a river — the curve degenerates to the
    /// straight line it used to be, which is right: there is no next corner.
    ///
    /// The widening still runs across the *dominant* axis rather than along a
    /// true perpendicular, now taken from the curve's local tangent rather than
    /// from the segment as a whole: offsetting across the axis the line is walked
    /// on cannot leave the gaps a diagonal perpendicular would.
    fn paint_channel(
        &self,
        tiles: &mut HashMap<IVec2, TerrainKind>,
        from: u32,
        to: u32,
        after: u32,
        width: u32,
        config: &WorldPlanConfig,
        world: &WorldSnapshot,
    ) {
        let start = self.tile_at(from).as_vec2();
        let control = self.tile_at(to).as_vec2();
        let end = if after == NONE {
            control
        } else {
            control.midpoint(self.tile_at(after).as_vec2())
        };

        // Twice the tiles the curve can possibly span, so successive samples are
        // at most half a tile apart and the line comes out connected however it
        // bends. `river_curve_samples` is only a floor under a very short one.
        let span = (control - start).abs().max_element() + (end - control).abs().max_element();
        let steps = ((span.ceil() as i32 * 2).max(config.river_curve_samples as i32)).max(1);

        // Exactly `width` tiles across, leaning one side for an even width since
        // there is no such thing as a centred four-tile band.
        let first = -((width as i32 - 1) / 2);
        let last = width as i32 / 2;

        for step in 0..=steps {
            let t = step as f32 / steps as f32;
            let centre = start.lerp(control, t).lerp(control.lerp(end, t), t);
            let tangent = (control - start).lerp(end - control, t);
            let across = if tangent.x.abs() >= tangent.y.abs() {
                IVec2::Y
            } else {
                IVec2::X
            };
            for offset in first..=last {
                paint(
                    tiles,
                    centre.round().as_ivec2() + across * offset,
                    TerrainKind::River,
                    world,
                );
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

    use crate::gameplay::terrain::shared_test_sampler;

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

    /// Runs every spring, so a test can read the courses off the lattice rather
    /// than off the finished map — width and confluences make a course
    /// unrecoverable once it is tiles.
    fn descended(
        terrain: &TerrainConfig,
        config: &WorldPlanConfig,
        world: &WorldSnapshot,
    ) -> Lattice {
        let mut lattice = Lattice::new(shared_test_sampler(), terrain, config);
        for (index, spring) in springs(shared_test_sampler(), terrain, config, world)
            .into_iter()
            .enumerate()
        {
            lattice.descend(spring, index as u32 + 1, config, world);
        }
        lattice
    }

    /// The tiles one course runs through, following each node's successor from a
    /// spring to wherever the water stopped.
    fn course(lattice: &Lattice, spring: IVec2) -> Vec<IVec2> {
        let mut node = lattice.node_of(spring);
        let mut seen = std::collections::HashSet::from([node]);
        let mut path = vec![lattice.tile_at(node)];
        while lattice.next[node as usize] != NONE {
            node = lattice.next[node as usize];
            if !seen.insert(node) {
                break;
            }
            path.push(lattice.tile_at(node));
        }
        path
    }

    /// How far the course swings off the line from its spring to its mouth, as a
    /// fraction of that line.
    ///
    /// This and not sinuosity is what tells a bend from a staircase, and the
    /// difference is the whole reason the measurement exists. A steepest-descent
    /// walk that alternates between two lattice directions travels 1.2 times the
    /// distance it covers — it *scores* as sinuous — while never leaving the
    /// straight line by more than a node. Excursion sees through that: the
    /// zigzag is worth a couple of tiles of it, and a real bend is worth a
    /// tenth of the course's length.
    fn excursion(path: &[IVec2]) -> Option<f32> {
        let (start, end) = (path[0], *path.last()?);
        let line = (end - start).as_vec2();
        let length = line.length();
        if length < 1.0 {
            return None;
        }
        let normal = Vec2::new(-line.y, line.x) / length;
        Some(
            path.iter()
                .map(|tile| ((*tile - start).as_vec2().dot(normal)).abs())
                .fold(0.0, f32::max)
                / length,
        )
    }

    /// How far the water actually travelled, over how far it got. 1.0 is a
    /// straight line.
    fn sinuosity(path: &[IVec2]) -> Option<f32> {
        let straight = (*path.last()? - path[0]).as_vec2().length();
        if straight < 1.0 {
            return None;
        }
        let walked: f32 = path
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).as_vec2().length())
            .sum();
        Some(walked / straight)
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

        let once = plan_rivers(shared_test_sampler(), &terrain, &config, &world);
        let twice = plan_rivers(shared_test_sampler(), &terrain, &config, &world);

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

        let springs = springs(shared_test_sampler(), &terrain, &config, &world);
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

        for edit in edits(&plan_rivers(
            shared_test_sampler(),
            &terrain,
            &config,
            &world,
        )) {
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
    ///
    /// Measured on a segment with no `after`, which is the one case that is still
    /// a straight line. A curved segment is exactly as wide *across its own
    /// tangent*, and there is no measuring that with a slice through the map.
    #[test]
    fn a_segment_is_painted_exactly_as_wide_as_its_flow() {
        let terrain = TerrainConfig::default();
        let config = WorldPlanConfig::default();
        let lattice = Lattice::new(shared_test_sampler(), &terrain, &config);
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
                    NONE,
                    width,
                    &config,
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
        let mut lattice = Lattice::new(shared_test_sampler(), &terrain, &config);

        for (index, spring) in springs(shared_test_sampler(), &terrain, &config, &world)
            .into_iter()
            .enumerate()
        {
            lattice.descend(spring, index as u32 + 1, &config, &world);
        }

        let mut ceiling = 0usize;
        for index in 0..lattice.flow.len() as u32 {
            let flow = lattice.flow[index as usize];
            if flow == 0 || lattice.next[index as usize] == NONE {
                continue;
            }
            // A segment runs from its node, past its successor, to halfway to the
            // one after — a step and a half of line, not a step — and each tile
            // of it is widened to at most its own width.
            ceiling += (2 * lattice.stride as usize + 1) * channel_width(flow, &config) as usize;
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
        let mut lattice = Lattice::new(shared_test_sampler(), &terrain, &config);
        let world = sloping_world();

        for (index, spring) in springs(shared_test_sampler(), &terrain, &config, &world)
            .into_iter()
            .enumerate()
        {
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
        let plan = plan_rivers(shared_test_sampler(), &terrain, &config, &sloping_world());

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

    /// The mean over every course long enough to have a shape at all — a spring
    /// that meets the sea in three nodes is a trickle, and its shape is a
    /// property of where the coast happened to be.
    fn shape_of_the_courses(
        terrain: &TerrainConfig,
        config: &WorldPlanConfig,
        world: &WorldSnapshot,
        of: fn(&[IVec2]) -> Option<f32>,
    ) -> f32 {
        let lattice = descended(terrain, config, world);
        let measured: Vec<f32> = springs(shared_test_sampler(), terrain, config, world)
            .into_iter()
            .map(|spring| course(&lattice, spring))
            .filter(|path| path.len() >= 8)
            .filter_map(|path| of(&path))
            .collect();
        assert!(
            measured.len() > 50,
            "only {} courses were long enough to measure",
            measured.len()
        );
        measured.iter().sum::<f32>() / measured.len() as f32
    }

    /// The point of the whole exercise, and it has to be measured as **excursion**
    /// rather than as sinuosity or it measures nothing.
    ///
    /// Sinuosity looks like the obvious check — the issue asks for a course
    /// longer than the line from spring to mouth — and it is a trap. Steepest
    /// descent on an eight-neighbour lattice already scores 1.18 on this world,
    /// because a walk alternating between two lattice directions travels 1.2
    /// times the distance it covers while never leaving the straight line by more
    /// than a node. That is the staircase, not a bend. A threshold on sinuosity
    /// passes on the code this task set out to change.
    ///
    /// So the claim is made against the rule it replaces rather than against a
    /// constant: turning the two shape terms off degenerates the scoring back to
    /// steepest descent per tile travelled, and the bends have to survive the
    /// comparison.
    #[test]
    fn the_bends_come_from_the_scoring_and_not_from_the_lattice() {
        let terrain = TerrainConfig::default();
        let world = sloping_world();
        let config = WorldPlanConfig::default();
        let steepest = WorldPlanConfig {
            river_heading_weight: 0.0,
            river_meander_weight: 0.0,
            ..config.clone()
        };

        let bends = shape_of_the_courses(&terrain, &config, &world, excursion);
        let lattice_only = shape_of_the_courses(&terrain, &steepest, &world, excursion);

        assert!(
            bends > lattice_only * 1.1,
            "courses swing {bends:.3} off their own straight line against \
             {lattice_only:.3} for plain steepest descent, so the heading and \
             meander terms are not reaching the water"
        );
    }

    /// The issue's own words: a course longer than the straight line from source
    /// to mouth. Weak on its own — see the test above for why — but it is the
    /// stated goal, and it would catch a change that bought excursion by making
    /// every river shorter.
    #[test]
    fn a_course_wanders_further_than_the_straight_line_to_its_mouth() {
        let terrain = TerrainConfig::default();
        let config = WorldPlanConfig::default();

        let wandered = shape_of_the_courses(&terrain, &config, &sloping_world(), sinuosity);

        assert!(
            wandered > 1.25,
            "courses average a sinuosity of {wandered:.3}, against 1.18 for the \
             steepest-descent walk this replaced"
        );
    }

    /// The rule the whole module rests on, and the one the scoring could most
    /// easily have broken: a step is chosen from the neighbours that do not
    /// climb, not from the single lowest.
    ///
    /// Checked against the elevations as they stand *after* every particle has
    /// run, which is why both ends have to be clear of a lake. A basin is raised
    /// to its spill level once it fills, so a step recorded before that now runs
    /// into a risen surface — water flowing into a lake, which is correct. The
    /// claim is "downhill when chosen", not "downhill forever".
    #[test]
    fn a_step_is_never_uphill_when_it_is_chosen() {
        let terrain = TerrainConfig::default();
        let config = WorldPlanConfig::default();
        let world = sloping_world();
        let lattice = descended(&terrain, &config, &world);

        let mut checked = 0;
        for node in 0..lattice.next.len() as u32 {
            let next = lattice.next[node as usize];
            if next == NONE
                || lattice.lake_of[node as usize] != NONE
                || lattice.lake_of[next as usize] != NONE
            {
                continue;
            }
            checked += 1;
            let (here, there) = (
                lattice.elevation[node as usize],
                lattice.elevation[next as usize],
            );
            assert!(
                there <= here,
                "the water climbed from {} at {here} to {} at {there}",
                lattice.tile_at(node),
                lattice.tile_at(next)
            );
        }
        assert!(checked > 1000, "only {checked} steps were checked");
    }

    /// What the heading costs and has to pay back. Two particles arriving at one
    /// node carry different headings, so scoring afresh would send them different
    /// ways and lay two channels a tile apart — braiding, where anchoring the
    /// lattice on the world origin exists to make them fuse.
    #[test]
    fn two_courses_that_meet_leave_the_junction_together() {
        let terrain = TerrainConfig::default();
        let config = WorldPlanConfig::default();
        let world = sloping_world();
        let mut lattice = Lattice::new(shared_test_sampler(), &terrain, &config);

        let springs = springs(shared_test_sampler(), &terrain, &config, &world);
        let mut fixed: Vec<(u32, u32)> = Vec::new();
        let mut junctions = 0;
        for (index, spring) in springs.into_iter().enumerate() {
            lattice.descend(spring, index as u32 + 1, &config, &world);
            for &(node, successor) in &fixed {
                assert_eq!(
                    lattice.next[node as usize],
                    successor,
                    "the course out of {} was recut by a later particle",
                    lattice.tile_at(node)
                );
            }
            let after: Vec<(u32, u32)> = (0..lattice.next.len() as u32)
                .filter(|n| lattice.next[*n as usize] != NONE)
                .map(|n| (n, lattice.next[n as usize]))
                .collect();
            junctions += after.len() - fixed.len();
            fixed = after;
        }
        assert!(junctions > 0, "no course was ever cut at all");
    }

    /// A level step is what lets a river wander a flood plain instead of pooling
    /// on it, and the cap is what stops a flood plain swallowing the course
    /// whole. Without it the water wanders a perfectly flat world until it runs
    /// out of steps and stops in the middle of nowhere.
    ///
    /// Flat here means *actually* flat: the elevation cache is filled in rather
    /// than left to the sampler, whose fbm has a dip in it somewhere at every
    /// scale.
    #[test]
    fn a_level_plain_does_not_swallow_a_course() {
        let terrain = TerrainConfig::default();
        let config = WorldPlanConfig::default();
        let world = WorldSnapshot::from_fn(|_| TerrainKind::Grass);
        let mut lattice = Lattice::new(shared_test_sampler(), &terrain, &config);
        lattice.elevation.fill(0.5);

        lattice.descend(IVec2::splat(2048), 1, &config, &world);

        let walked = lattice.visited.iter().filter(|&&mark| mark == 1).count();
        assert!(
            walked <= config.river_flat_run_nodes as usize + 2,
            "the water wandered {walked} nodes of flat before it was declared to \
             be standing, against a cap of {}",
            config.river_flat_run_nodes
        );
        assert!(
            walked > 1,
            "the water did not take a level step at all, so the flood plain is \
             still a lake"
        );
        assert!(!lattice.lakes.is_empty(), "the run never ended in a basin");
    }

    /// Why the meander field gets its own salt. A bias that tracked the height
    /// would put every bend where the hill already decides the course, and the
    /// two would cancel instead of compounding — the rivers would come out
    /// straight and the field would look like it was working.
    #[test]
    fn the_meander_field_is_uncorrelated_with_the_height_it_bends() {
        let terrain = TerrainConfig::default();
        let config = WorldPlanConfig::default();
        let lattice = Lattice::new(shared_test_sampler(), &terrain, &config);
        let sampler = terrain.sampler();

        let (mut lean, mut height) = (Vec::new(), Vec::new());
        for i in 0..20000u32 {
            let x = (i.wrapping_mul(2654435761) % 4000) as f32;
            let y = (i.wrapping_mul(40503) % 4000) as f32;
            lean.push(lattice.meander.sample(x, y) as f64);
            height.push(sampler.elevation(x, y) as f64);
        }

        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (lean_mean, height_mean) = (mean(&lean), mean(&height));
        let covariance: f64 = lean
            .iter()
            .zip(&height)
            .map(|(l, h)| (l - lean_mean) * (h - height_mean))
            .sum();
        let spread = |v: &[f64], m: f64| v.iter().map(|x| (x - m).powi(2)).sum::<f64>().sqrt();
        let correlation = covariance / (spread(&lean, lean_mean) * spread(&height, height_mean));

        assert!(
            correlation.abs() < 0.05,
            "the meander field correlates {correlation:.3} with the height, so a \
             bend sits on the hill that already chose the course"
        );
    }

    /// What the shape knobs actually do to the world, rather than to the
    /// synthetic coastline the checks above use. Run it either side of a change
    /// to `river_reference_drop`, `river_heading_weight` or
    /// `river_meander_weight` — none of them has a right value that can be
    /// derived, only one that can be looked at.
    ///
    /// `cargo test --release -- --ignored --nocapture the_shape_of_the_worlds_rivers`
    #[test]
    #[ignore = "measurement, not a check"]
    fn the_shape_of_the_worlds_rivers() {
        let terrain = TerrainConfig::default();
        let world = WorldSnapshot::generated(&terrain, shared_test_sampler());

        // The two shape weights against the default, so what they buy can be read
        // off rather than argued about. At (0, 0) the scoring degenerates to
        // steepest descent per tile travelled, which is the old rule.
        // NOT the lattice stride, which is where the obvious suspicion points:
        // `drainage.rs` found that at a 4-tile stride the relief layer's fine
        // octaves put a local minimum every few nodes, and fixed its own stage by
        // stepping 16. Swept here it does nothing for rivers — courses stay 3 to
        // 5 nodes at every stride from 4 to 16, because a river *floods* a pit
        // rather than stopping at one. Coarsening only buys fewer nodes.

        println!("heading  meander | excursion  sinuosity  median  longest  basins  river nodes");
        for (heading, meander) in [(0.0, 0.0), (0.6, 1.0), (0.6, 2.0), (1.2, 3.0), (2.0, 8.0)] {
            let config = WorldPlanConfig {
                river_heading_weight: heading,
                river_meander_weight: meander,
                ..WorldPlanConfig::default()
            };
            let lattice = descended(&terrain, &config, &world);
            let (mut bendy, mut wandered, mut lens) = (Vec::new(), Vec::new(), Vec::new());
            for spring in springs(shared_test_sampler(), &terrain, &config, &world) {
                let path = course(&lattice, spring);
                lens.push(path.len());
                if path.len() >= 8 {
                    wandered.extend(sinuosity(&path));
                    bendy.extend(excursion(&path));
                }
            }
            lens.sort_unstable();
            let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
            println!(
                "{heading:7.1}  {meander:7.1} | {:9.3}  {:9.3}  {:6}  {:7}  {:6}  {:11}",
                mean(&bendy),
                mean(&wandered),
                lens[lens.len() / 2],
                lens[lens.len() - 1],
                lattice.lakes.len(),
                lattice.next.iter().filter(|n| **n != NONE).count(),
            );
        }

        // What a course's *length* answers to. Shape and length are different
        // questions: the scoring decides how a fragment bends, and these decide
        // how long a fragment gets to be before a lake ends it.
        let shape = |lattice: &Lattice, config: &WorldPlanConfig, world: &WorldSnapshot| {
            let (mut bendy, mut nodes, mut tiles) = (Vec::new(), Vec::new(), Vec::new());
            for spring in springs(
                shared_test_sampler(),
                &TerrainConfig::default(),
                config,
                world,
            ) {
                let path = course(lattice, spring);
                nodes.push(path.len());
                tiles.push(
                    path.windows(2)
                        .map(|p| (p[1] - p[0]).as_vec2().length())
                        .sum::<f32>(),
                );
                if path.len() >= 8 {
                    bendy.extend(excursion(&path));
                }
            }
            nodes.sort_unstable();
            tiles.sort_by(f32::total_cmp);
            (
                bendy.len(),
                nodes[nodes.len() * 9 / 10],
                tiles[tiles.len() * 9 / 10],
                bendy.iter().sum::<f32>() / bendy.len() as f32,
            )
        };
        let water = |lattice: &Lattice, config: &WorldPlanConfig, world: &WorldSnapshot| {
            let plan = lattice.stamp(config, world);
            let (mut river, mut lake) = (0, 0);
            for edit in plan.by_chunk.iter().flatten() {
                match edit.kind {
                    TerrainKind::River => river += 1,
                    _ => lake += 1,
                }
            }
            (river, lake)
        };

        println!("lake_min | courses>=8  p90 nodes  p90 tiles  excursion  river tiles  lake tiles");
        for min in [64u32, 128, 256, 512, 1024, 2048] {
            let config = WorldPlanConfig {
                river_lake_min_tiles: min,
                ..WorldPlanConfig::default()
            };
            let lattice = descended(&terrain, &config, &world);
            let (courses, nodes, tiles, bendy) = shape(&lattice, &config, &world);
            let (river, lake) = water(&lattice, &config, &world);
            println!(
                "{min:8} | {courses:10}  {nodes:9}  {tiles:9.0}  {bendy:9.3}  {river:11}  {lake:10}"
            );
        }

        println!("lake_max | courses>=8  p90 nodes  p90 tiles  excursion  river tiles  lake tiles");
        for max in [512u32, 1024, 2048, 8192, 32768] {
            let config = WorldPlanConfig {
                river_lake_max_tiles: max,
                ..WorldPlanConfig::default()
            };
            let lattice = descended(&terrain, &config, &world);
            let (courses, nodes, tiles, bendy) = shape(&lattice, &config, &world);
            let (river, lake) = water(&lattice, &config, &world);
            println!(
                "{max:8} | {courses:10}  {nodes:9}  {tiles:9.0}  {bendy:9.3}  {river:11}  {lake:10}"
            );
        }

        println!("flat_run | courses>=8  p90 nodes  p90 tiles  excursion  river tiles  lake tiles");
        for flat in [0u32, 8, 24, 64, 256] {
            let config = WorldPlanConfig {
                river_flat_run_nodes: flat,
                ..WorldPlanConfig::default()
            };
            let lattice = descended(&terrain, &config, &world);
            let (courses, nodes, tiles, bendy) = shape(&lattice, &config, &world);
            let (river, lake) = water(&lattice, &config, &world);
            println!(
                "{flat:8} | {courses:10}  {nodes:9}  {tiles:9.0}  {bendy:9.3}  {river:11}  {lake:10}"
            );
        }

        println!("reference drop | excursion  sinuosity  median  longest  river nodes");
        for reference in [0.002f32, 0.004, 0.008, 0.012, 0.020, 0.040] {
            let config = WorldPlanConfig {
                river_reference_drop: reference,
                ..WorldPlanConfig::default()
            };
            let lattice = descended(&terrain, &config, &world);
            let (mut bendy, mut wandered, mut lens) = (Vec::new(), Vec::new(), Vec::new());
            for spring in springs(shared_test_sampler(), &terrain, &config, &world) {
                let path = course(&lattice, spring);
                lens.push(path.len());
                if path.len() >= 8 {
                    wandered.extend(sinuosity(&path));
                    bendy.extend(excursion(&path));
                }
            }
            lens.sort_unstable();
            let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
            println!(
                "{reference:14.4} | {:9.3}  {:9.3}  {:6}  {:7}  {:11}",
                mean(&bendy),
                mean(&wandered),
                lens[lens.len() / 2],
                lens[lens.len() - 1],
                lattice.next.iter().filter(|n| **n != NONE).count(),
            );
        }

        let config = WorldPlanConfig::default();
        let lattice = descended(&terrain, &config, &world);

        let mut bendy: Vec<f32> = Vec::new();
        let mut wandered: Vec<f32> = Vec::new();
        let mut lengths: Vec<usize> = Vec::new();
        for spring in springs(shared_test_sampler(), &terrain, &config, &world) {
            let path = course(&lattice, spring);
            lengths.push(path.len());
            if path.len() >= 8 {
                wandered.extend(sinuosity(&path));
                bendy.extend(excursion(&path));
            }
        }
        bendy.sort_by(f32::total_cmp);
        wandered.sort_by(f32::total_cmp);
        lengths.sort_unstable();

        let at = |v: &[f32], q: f32| v[((v.len() - 1) as f32 * q) as usize];
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        println!(
            "{} springs, {} courses of 8 nodes or more",
            lengths.len(),
            bendy.len()
        );
        println!(
            "excursion  median {:.3}  p90 {:.3}  mean {:.3}   <- bends",
            at(&bendy, 0.5),
            at(&bendy, 0.9),
            mean(&bendy),
        );
        println!(
            "sinuosity  median {:.3}  p90 {:.3}  mean {:.3}   <- bends and staircase together",
            at(&wandered, 0.5),
            at(&wandered, 0.9),
            mean(&wandered),
        );
        println!(
            "course length in nodes  median {}  p90 {}  longest {}",
            lengths[lengths.len() / 2],
            lengths[lengths.len() * 9 / 10],
            lengths[lengths.len() - 1],
        );
        println!(
            "busiest segment carries {} particles; {} basins filled",
            lattice.flow.iter().copied().max().unwrap_or(0),
            lattice.lakes.len(),
        );

        // What the descent term is actually worth, which is the only way to set
        // `river_reference_drop`: it is the drop that makes the terrain and the
        // shape rules weigh the same, so it has to be read off the terrain.
        let mut drops: Vec<f32> = Vec::new();
        for node in 0..lattice.next.len() as u32 {
            let next = lattice.next[node as usize];
            if next == NONE || lattice.lake_of[node as usize] != NONE {
                continue;
            }
            let travelled = (lattice.node_at(next) - lattice.node_at(node))
                .as_vec2()
                .length()
                * lattice.stride as f32;
            drops.push(
                (lattice.elevation[node as usize] - lattice.elevation[next as usize]) / travelled,
            );
        }
        drops.sort_by(f32::total_cmp);
        println!(
            "drop per tile along a course  p10 {:.5}  median {:.5}  p90 {:.5}",
            at(&drops, 0.1),
            at(&drops, 0.5),
            at(&drops, 0.9),
        );
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
        let mut lattice = Lattice::new(shared_test_sampler(), &terrain, &config);

        lattice.descend(IVec2::splat(2048), 1, &config, &world);

        let visited = lattice.visited.iter().filter(|&&mark| mark == 1).count();
        assert!(
            visited <= config.river_max_steps as usize,
            "the particle took more steps than it was allowed"
        );
    }
}
