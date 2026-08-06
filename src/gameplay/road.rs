//! Which cities are connected, and where the road between two of them runs.
//!
//! A route is searched on a coarse lattice rather than tile by tile: a road can
//! span hundreds of tiles, and at that length the lattice costs a few thousand
//! nodes instead of a hundred thousand. The tiles *between* two lattice nodes
//! are still checked one by one for water, though — checking only the nodes
//! would let a road hop a river.

use std::{cmp::Ordering, collections::BinaryHeap};

use bevy::prelude::*;

use crate::gameplay::{
    city::City,
    plan::WorldPlanConfig,
    terrain::{TerrainConfig, TerrainKind},
    world::{TileEdit, WorldSnapshot},
};

/// Two cities with a road between them, and every tile the road runs over.
///
/// The path used to be dropped, on the argument that it was already readable off
/// the `Road` tiles. That argument is wrong and the reason is the merge: a `Road`
/// tile does not say which link laid it, routes share tiles deliberately, and a
/// route's own `edits` *exclude* every tile it reused — so the ground is not a
/// second copy of the path, it is a union of all of them with the seams gone.
/// gh-7's caravans need to know which road they are on, so the tile list the
/// router already builds is kept rather than thrown away. 87 links of a few
/// hundred tiles is ~300 KB, against `WorldMap`'s 32 MB.
///
/// The path runs from `from` to `to` and includes both cities' hops onto the
/// lattice, so its ends are the two centres and not two lattice nodes.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RoadLink {
    pub from: u32,
    pub to: u32,
    pub path: Vec<IVec2>,
}

impl RoadLink {
    /// How far a caravan walks along this road, in tiles. Path length rather than
    /// the distance between the two cities: a road that goes round a lake is
    /// longer than the crow's flight and a trader pays for every tile of it.
    pub fn length_tiles(&self) -> f32 {
        self.path.len().saturating_sub(1) as f32
    }
}

/// Which cities ended up connected.
#[derive(Resource, Default)]
pub struct RoadNetwork {
    pub links: Vec<RoadLink>,
}

/// A finished route: the link it realises and the tiles it paves.
///
/// The two are different lists and neither is derivable from the other. `link`
/// carries every tile the route walks; `edits` carries only the ones that were
/// not already road or town, because re-stamping a reused tile would dirty a
/// chunk for no visible change.
pub struct RoutedRoad {
    pub link: RoadLink,
    pub edits: Vec<TileEdit>,
}

/// Picks the city pairs worth a road, in two passes that remove two different
/// kinds of redundant road.
pub fn choose_pairs(cities: &[City], config: &WorldPlanConfig) -> Vec<(usize, usize)> {
    let neighbours = gabriel_pairs(cities, config.road_max_distance_tiles);
    prune_parallel(
        cities,
        neighbours,
        config.road_min_separation_degrees.to_radians(),
    )
}

/// A pair survives only if no third city lies inside the circle that has the
/// pair as its diameter — the Gabriel graph. That is what stops every city
/// linking to every other one within range: when A and B are already linked
/// through C between them, the long A–B hop is dropped.
///
/// This is cubic in the city count, which is fine because the region grid keeps
/// that count in the low hundreds; the distance cut-off prunes most of it before
/// the third loop is ever reached.
fn gabriel_pairs(cities: &[City], max_distance: u32) -> Vec<(usize, usize)> {
    let max_squared = (max_distance as i64).pow(2);
    let mut pairs = Vec::new();

    for a in 0..cities.len() {
        for b in a + 1..cities.len() {
            let (start, end) = (cities[a].centre, cities[b].centre);
            if squared_distance(start, end) > max_squared {
                continue;
            }

            let blocked = cities.iter().enumerate().any(|(other, city)| {
                other != a && other != b && inside_circle_on(start, end, city.centre)
            });
            if !blocked {
                pairs.push((a, b));
            }
        }
    }

    pairs
}

/// Drops a road when the city at either end already has one leaving in nearly
/// the same direction.
///
/// The Gabriel test cannot see this case: two cities off in the same direction
/// but well to the side of each other both pass it, and the result is a pair of
/// roads running alongside each other out of the same gate. Angle is the only
/// thing that distinguishes them, so angle is what this measures.
fn prune_parallel(
    cities: &[City],
    mut pairs: Vec<(usize, usize)>,
    min_separation: f32,
) -> Vec<(usize, usize)> {
    // Shortest first, so of two roads leaving side by side the one kept is the
    // tighter; by index on a tie, so the outcome cannot depend on the order the
    // pairs happened to be generated in.
    pairs.sort_by_key(|&(a, b)| (squared_distance(cities[a].centre, cities[b].centre), a, b));

    let mut leaving: Vec<Vec<f32>> = vec![Vec::new(); cities.len()];
    let mut kept = Vec::with_capacity(pairs.len());

    for &(a, b) in &pairs {
        let (out, back) = (
            bearing(cities[a].centre, cities[b].centre),
            bearing(cities[b].centre, cities[a].centre),
        );
        let crowded = leaving[a]
            .iter()
            .any(|&taken| angle_between(taken, out) < min_separation)
            || leaving[b]
                .iter()
                .any(|&taken| angle_between(taken, back) < min_separation);
        if crowded {
            continue;
        }

        leaving[a].push(out);
        leaving[b].push(back);
        kept.push((a, b));
    }

    // A city every one of whose roads was pruned would be cut off from the map
    // entirely, which reads as a bug in a way that one road too many does not.
    // Its shortest candidate goes back in.
    for &(a, b) in &pairs {
        if leaving[a].is_empty() || leaving[b].is_empty() {
            leaving[a].push(bearing(cities[a].centre, cities[b].centre));
            leaving[b].push(bearing(cities[b].centre, cities[a].centre));
            kept.push((a, b));
        }
    }

    // Back into length order, rescued pairs included: the caller routes these
    // from the back, and which road is laid first decides which one the rest
    // merge onto.
    kept.sort_by_key(|&(a, b)| (squared_distance(cities[a].centre, cities[b].centre), a, b));
    kept
}

/// The direction from one city to another, in radians.
fn bearing(from: IVec2, to: IVec2) -> f32 {
    (to - from).as_vec2().to_angle()
}

/// The smaller angle between two bearings, so that 350° and 10° are 20° apart
/// rather than 340°.
fn angle_between(a: f32, b: f32) -> f32 {
    let difference = (a - b).abs() % std::f32::consts::TAU;
    difference.min(std::f32::consts::TAU - difference)
}

fn squared_distance(a: IVec2, b: IVec2) -> i64 {
    let d = (a - b).as_i64vec2();
    d.x * d.x + d.y * d.y
}

/// Whether `point` lies inside the circle that has `a`–`b` as its diameter.
/// True exactly when the point sees the diameter at an obtuse angle, which is
/// the dot product below being negative — no midpoint and no square root.
fn inside_circle_on(a: IVec2, b: IVec2, point: IVec2) -> bool {
    let to_a = (point - a).as_i64vec2();
    let to_b = (point - b).as_i64vec2();
    to_a.x * to_b.x + to_a.y * to_b.y < 0
}

/// Routes one road, or `None` when the two cities cannot be joined over land.
///
/// The search is bounded to the two centres' bounding box padded by
/// `route_padding_tiles`: it keeps the work per road bounded, and it stops a
/// route from wandering halfway across the map to avoid a hill.
pub fn route_road(
    terrain: &TerrainConfig,
    config: &WorldPlanConfig,
    world: &WorldSnapshot,
    from: &City,
    to: &City,
) -> Option<RoutedRoad> {
    let lattice = Lattice::new(config, from.centre, to.centre);
    let elevation = lattice.sample_elevation(terrain);

    // Both ends are off-lattice — a city centre is wherever the planner put it —
    // so each needs a walkable hop onto the grid. The nearest node is not always
    // that: a coastal city can have water between it and the node it rounds to,
    // and taking the next one over is the difference between a road and none.
    let (start_node, departure) = lattice_entry(&lattice, from.centre, world)?;
    let (goal_node, approach) = lattice_entry(&lattice, to.centre, world)?;
    let start = lattice.index_of(start_node)?;
    let goal = lattice.index_of(goal_node)?;

    let approach: Vec<IVec2> = approach.into_iter().rev().collect();

    let path = search(&lattice, &elevation, start, goal, world, config)?;

    let mut tiles = departure;
    for pair in path.windows(2) {
        let step = walk_line(
            lattice.position(lattice.node_at(pair[0])),
            lattice.position(lattice.node_at(pair[1])),
            world,
        )?;
        tiles.extend(step);
    }
    tiles.extend(approach);

    let edits = tiles
        .iter()
        .copied()
        .filter(|&tile| match world.tile(tile) {
            // A road meets a city rather than cutting through it.
            Some(TerrainKind::Town) => false,
            // Already paved by an earlier road this one merged onto. Emitting it
            // again would only dirty the chunk for no visible change.
            Some(TerrainKind::Road) => false,
            _ => true,
        })
        .map(|tile| TileEdit {
            tile,
            kind: TerrainKind::Road,
        })
        .collect();

    Some(RoutedRoad {
        link: RoadLink {
            from: from.id,
            to: to.id,
            path: tiles,
        },
        edits,
    })
}

/// The nearest lattice node a city can actually reach, and the tiles of the hop
/// to it. Searches outward from the node the centre rounds to, so the hop is as
/// short as the terrain allows.
fn lattice_entry(
    lattice: &Lattice,
    tile: IVec2,
    world: &WorldSnapshot,
) -> Option<(IVec2, Vec<IVec2>)> {
    let nearest = lattice.node_of(tile);
    let mut best: Option<(f32, IVec2, Vec<IVec2>)> = None;

    for dy in -1..=1 {
        for dx in -1..=1 {
            let node = nearest + IVec2::new(dx, dy);
            if lattice.index_of(node).is_none() {
                continue;
            }
            let position = lattice.position(node);
            let Some(hop) = walk_line(tile, position, world) else {
                continue;
            };
            let distance = tile.as_vec2().distance(position.as_vec2());
            if best
                .as_ref()
                .is_none_or(|(closest, ..)| distance < *closest)
            {
                best = Some((distance, node, hop));
            }
        }
    }

    best.map(|(_, node, hop)| (node, hop))
}

/// The coarse grid a route is searched on.
///
/// It is anchored on the world origin, not on the starting city, and that is
/// what makes roads able to reuse each other at all: every route steps between
/// the *same* lattice nodes, so two roads that take the same step lay down
/// exactly the same tiles. Anchored per-city instead, two roads could run a tile
/// apart for their whole length and neither would ever see the other's tiles.
struct Lattice {
    stride: i32,
    min: IVec2,
    size: IVec2,
}

impl Lattice {
    fn new(config: &WorldPlanConfig, from: IVec2, to: IVec2) -> Self {
        let stride = config.road_node_stride.max(1) as i32;
        let padding = config.route_padding_tiles as i32;
        let low = from.min(to) - IVec2::splat(padding);
        let high = from.max(to) + IVec2::splat(padding);

        let min = (low.as_vec2() / stride as f32).ceil().as_ivec2();
        let max = (high.as_vec2() / stride as f32).floor().as_ivec2();

        Self {
            stride,
            min,
            size: (max - min) + IVec2::ONE,
        }
    }

    /// The node nearest a tile. Rounding rather than flooring means the node a
    /// city snaps to is the closest one, so the hop onto the lattice is short.
    fn node_of(&self, tile: IVec2) -> IVec2 {
        (tile.as_vec2() / self.stride as f32).round().as_ivec2()
    }

    fn index_of(&self, node: IVec2) -> Option<usize> {
        let local = node - self.min;
        (local.cmpge(IVec2::ZERO).all() && local.cmplt(self.size).all())
            .then(|| (local.y * self.size.x + local.x) as usize)
    }

    fn node_at(&self, index: usize) -> IVec2 {
        self.min + IVec2::new(index as i32 % self.size.x, index as i32 / self.size.x)
    }

    fn position(&self, node: IVec2) -> IVec2 {
        node * self.stride
    }

    fn node_count(&self) -> usize {
        (self.size.x * self.size.y) as usize
    }

    /// Elevation at every node, up front. `WorldMap` records only which band a
    /// tile fell in, so the height a road is trying to avoid climbing has to
    /// come back from the noise field.
    fn sample_elevation(&self, terrain: &TerrainConfig) -> Vec<f32> {
        let sampler = terrain.sampler();
        (0..self.node_count())
            .map(|index| {
                let tile = self.position(self.node_at(index));
                sampler.elevation(tile.x as f32, tile.y as f32)
            })
            .collect()
    }
}

/// A* over the lattice. Returns the node indices from start to goal.
fn search(
    lattice: &Lattice,
    elevation: &[f32],
    start: usize,
    goal: usize,
    world: &WorldSnapshot,
    config: &WorldPlanConfig,
) -> Option<Vec<usize>> {
    let goal_position = lattice.position(lattice.node_at(goal));

    let mut cost = vec![f32::INFINITY; lattice.node_count()];
    let mut came_from = vec![usize::MAX; lattice.node_count()];
    let mut open = BinaryHeap::new();

    cost[start] = 0.0;
    open.push(Step {
        estimate: 0.0,
        node: start,
    });

    while let Some(Step { node, .. }) = open.pop() {
        if node == goal {
            return Some(reconstruct(&came_from, start, goal));
        }

        let here = lattice.node_at(node);
        let here_position = lattice.position(here);

        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                if (dx, dy) == (0, 0) {
                    continue;
                }
                let Some(next) = lattice.index_of(here + IVec2::new(dx, dy)) else {
                    continue;
                };
                let next_position = lattice.position(lattice.node_at(next));
                let Some(span) = walk_line(here_position, next_position, world) else {
                    continue;
                };

                let climb = (elevation[node] - elevation[next]).abs();
                // The crossing is charged on top of the discounted step rather
                // than inside it: a bridge costs what it costs, and the fact
                // that the approach runs on an existing road does not make the
                // river any narrower.
                let step = (here_position.as_vec2().distance(next_position.as_vec2())
                    + climb * config.road_elevation_penalty)
                    * reuse_discount(&span, world, config.road_reuse_discount)
                    + crossings(&span, world) as f32 * config.road_river_crossing_penalty;
                let candidate = cost[node] + step;
                if candidate >= cost[next] {
                    continue;
                }

                cost[next] = candidate;
                came_from[next] = node;
                open.push(Step {
                    // Scaled by the reuse discount, because a step that runs on
                    // existing road costs that fraction of its length. Left
                    // unscaled the estimate overstates what is still to pay,
                    // which drives the search straight at the goal — and a
                    // detour to pick up an existing road is precisely what that
                    // would refuse to consider.
                    estimate: candidate
                        + next_position.as_vec2().distance(goal_position.as_vec2())
                            * config.road_reuse_discount,
                    node: next,
                });
            }
        }
    }

    None
}

fn reconstruct(came_from: &[usize], start: usize, goal: usize) -> Vec<usize> {
    let mut path = vec![goal];
    let mut node = goal;
    while node != start {
        node = came_from[node];
        path.push(node);
    }
    path.reverse();
    path
}

/// What to multiply a step's cost by, given how much of it runs on road that is
/// already there.
///
/// This is what makes roads merge instead of running alongside each other: a
/// route will accept a detour to reach an existing road if the road then carries
/// it far enough, and the two arrive at the city as one. Discounting the whole
/// step — the climb as well as the distance — is deliberate: an existing road
/// has already paid for whatever hill it crosses, and re-charging a later route
/// for that climb is exactly what would keep them apart.
fn reuse_discount(span: &[IVec2], world: &WorldSnapshot, discount: f32) -> f32 {
    let paved = span
        .iter()
        .filter(|&&tile| world.tile(tile) == Some(TerrainKind::Road))
        .count();
    let reused = paved as f32 / span.len() as f32;
    1.0 + reused * (discount - 1.0)
}

/// How many tiles of river a step fords.
///
/// A river is not `is_water`, so a route may cross one — a river runs from the
/// mountains down to the sea, and refusing it the way the sea is refused would
/// cut the continent into pieces the network could not span. It is charged for
/// instead, which makes a road go a long way round to find a narrows and cross
/// where an earlier road already has: a crossing lays a `Road` tile, so the next
/// route this way finds road rather than river and pays the reuse discount
/// instead of the crossing.
fn crossings(span: &[IVec2], world: &WorldSnapshot) -> usize {
    span.iter()
        .filter(|&&tile| world.tile(tile) == Some(TerrainKind::River))
        .count()
}

/// The tiles a straight run from `from` to `to` covers, or `None` if any of them
/// is water. Checking the whole line and not just its ends is what keeps a road
/// out of the sea and out of a lake at tile resolution.
fn walk_line(from: IVec2, to: IVec2, world: &WorldSnapshot) -> Option<Vec<IVec2>> {
    let delta = to - from;
    let steps = delta.x.abs().max(delta.y.abs());
    let mut tiles = Vec::with_capacity(steps as usize + 1);

    for step in 0..=steps {
        let tile = if steps == 0 {
            from
        } else {
            from + (delta.as_vec2() * (step as f32 / steps as f32))
                .round()
                .as_ivec2()
        };
        if world.tile(tile)?.is_water() {
            return None;
        }
        tiles.push(tile);
    }

    Some(tiles)
}

/// A node waiting to be expanded. `BinaryHeap` is a max-heap, so the ordering is
/// inverted to pop the cheapest estimate first.
struct Step {
    estimate: f32,
    node: usize,
}

impl Ord for Step {
    fn cmp(&self, other: &Self) -> Ordering {
        // Estimates are always finite here, so the comparison never falls back.
        other
            .estimate
            .partial_cmp(&self.estimate)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.node.cmp(&self.node))
    }
}

impl PartialOrd for Step {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Step {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Step {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gameplay::city::CitySize;

    fn city(id: u32, x: i32, y: i32) -> City {
        City {
            id,
            centre: IVec2::new(x, y),
            size: CitySize::Village,
            radius: 5,
        }
    }

    #[test]
    fn a_city_between_two_others_replaces_the_long_hop() {
        let cities = [city(0, 0, 0), city(1, 100, 0), city(2, 200, 0)];
        let pairs = choose_pairs(&cities, &WorldPlanConfig::default());

        assert!(pairs.contains(&(0, 1)));
        assert!(pairs.contains(&(1, 2)));
        assert!(
            !pairs.contains(&(0, 2)),
            "0 and 2 are already linked through 1"
        );
    }

    #[test]
    fn cities_beyond_the_range_are_never_paired() {
        let cities = [city(0, 0, 0), city(1, 500, 0)];
        assert!(choose_pairs(&cities, &WorldPlanConfig::default()).is_empty());
    }

    /// The case the Gabriel test cannot catch: 1 and 2 are both off to the east
    /// of 0 and neither sits between the others, so both hops pass it — but from
    /// 0 they leave 11 degrees apart and would run side by side the whole way.
    #[test]
    fn a_city_does_not_get_two_roads_leaving_in_the_same_direction() {
        let cities = [city(0, 0, 0), city(1, 100, 0), city(2, 100, 20)];
        let config = WorldPlanConfig::default();

        assert!(
            gabriel_pairs(&cities, config.road_max_distance_tiles).contains(&(0, 2)),
            "this test is pointless unless the Gabriel pass lets 0-2 through"
        );

        let pairs = choose_pairs(&cities, &config);
        assert!(pairs.contains(&(0, 1)), "the shorter of the two is kept");
        assert!(!pairs.contains(&(0, 2)), "the parallel road is dropped");
        assert!(pairs.contains(&(1, 2)), "the link between them survives");
    }

    /// Pruning must not cut a city off the map altogether.
    #[test]
    fn a_city_keeps_a_road_even_when_every_one_of_them_is_parallel() {
        let cities = [city(0, 0, 0), city(1, 100, 0), city(2, 100, 20)];
        // Wide enough that every road out of 0 counts as parallel to every other.
        let config = WorldPlanConfig {
            road_min_separation_degrees: 179.0,
            ..WorldPlanConfig::default()
        };

        let pairs = choose_pairs(&cities, &config);
        for city in 0..cities.len() {
            assert!(
                pairs.iter().any(|&(a, b)| a == city || b == city),
                "city {city} was left with no roads at all"
            );
        }
    }

    #[test]
    fn a_point_off_the_diameter_is_outside_the_circle() {
        let (a, b) = (IVec2::new(0, 0), IVec2::new(100, 0));
        assert!(inside_circle_on(a, b, IVec2::new(50, 10)));
        assert!(!inside_circle_on(a, b, IVec2::new(50, 80)));
        assert!(!inside_circle_on(a, b, IVec2::new(150, 0)));
    }

    /// The two cities the routing tests join, in the middle of the world so the
    /// search box is nowhere near an edge.
    const WEST: IVec2 = IVec2::new(2000, 2048);
    const EAST: IVec2 = IVec2::new(2200, 2048);
    /// The river both tests put between them.
    const RIVER_X: i32 = 2100;

    fn route_across(world: &WorldSnapshot) -> Option<RoutedRoad> {
        route_road(
            &TerrainConfig::default(),
            &WorldPlanConfig::default(),
            world,
            &city(0, WEST.x, WEST.y),
            &city(1, EAST.x, EAST.y),
        )
    }

    #[test]
    fn a_road_goes_around_water_rather_than_through_it() {
        // A river that does not span the search box, so there is a way round it.
        let world = WorldSnapshot::from_fn(|tile| {
            if tile.x == RIVER_X && (2030..=2066).contains(&tile.y) {
                TerrainKind::ShallowWater
            } else {
                TerrainKind::Grass
            }
        });

        let road = route_across(&world).expect("the river can be walked around");
        assert!(!road.edits.is_empty());
        for edit in &road.edits {
            assert_eq!(edit.kind, TerrainKind::Road);
            assert!(
                !world.tile(edit.tile).expect("inside the world").is_water(),
                "the road runs through water at {}",
                edit.tile
            );
        }
    }

    #[test]
    fn two_cities_separated_by_water_stay_unconnected() {
        // The same river, now spanning everything the search is allowed to see.
        let world = WorldSnapshot::from_fn(|tile| {
            if tile.x == RIVER_X {
                TerrainKind::ShallowWater
            } else {
                TerrainKind::Grass
            }
        });

        assert!(route_across(&world).is_none());
    }

    /// A road has to meet a city, not cut across it, or the city it arrives at
    /// would be sliced in half by its own approach.
    #[test]
    fn a_road_never_paves_over_a_town_tile() {
        let world = WorldSnapshot::from_fn(|tile| {
            if tile.distance_squared(EAST) <= 25 {
                TerrainKind::Town
            } else {
                TerrainKind::Grass
            }
        });

        let road = route_across(&world).expect("open ground the whole way");
        for edit in &road.edits {
            assert_ne!(world.tile(edit.tile), Some(TerrainKind::Town));
        }
    }
}
