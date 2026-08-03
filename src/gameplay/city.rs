//! Where cities are, how big they are, and which tiles they cover.
//!
//! Cities are laid out once, from the finished world, rather than decided tile
//! by tile the way towns used to be: a city has a radius, and no chunk margin
//! can hold a rule that reaches that far. Planning is a pure function of
//! `(TerrainConfig, WorldPlanConfig, WorldSnapshot)`, so the same seed lays out
//! the same cities on every run and every platform.

use bevy::{platform::collections::HashMap, prelude::*};

use crate::gameplay::{
    noise::hash2,
    plan::WorldPlanConfig,
    terrain::{TerrainConfig, TerrainKind},
    world::{TileEdit, WORLD_TILES, WorldSnapshot, chunk_index_of_tile},
};

/// How much of the map a city covers. The tiers are what turn one settlement
/// score into "hamlet" or "capital" — the score has to clear the threshold by
/// more and more for each step up, so the biggest cities are the rarest.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CitySize {
    Hamlet,
    Village,
    Borough,
    Metropolis,
}

impl CitySize {
    /// Radius in tiles. A hamlet is a few tiles across; a metropolis is a fifth
    /// of a chunk.
    pub fn radius(self) -> u32 {
        match self {
            CitySize::Hamlet => 3,
            CitySize::Village => 5,
            CitySize::Borough => 8,
            CitySize::Metropolis => 12,
        }
    }

    /// `excess` is how far the site's score cleared the threshold, on 0..1.
    fn from_excess(excess: f32) -> Self {
        match excess {
            e if e < 0.30 => CitySize::Hamlet,
            e if e < 0.55 => CitySize::Village,
            e if e < 0.78 => CitySize::Borough,
            _ => CitySize::Metropolis,
        }
    }

    /// The tier a radius falls in — the inverse of [`CitySize::radius`], and what
    /// keeps the tier meaningful once [`crate::gameplay::growth`] moves the radius
    /// about. A city that grows past a threshold *becomes* the larger tier rather
    /// than keeping the one it was founded at.
    pub fn from_radius(radius: u32) -> Self {
        match radius {
            r if r < CitySize::Village.radius() => CitySize::Hamlet,
            r if r < CitySize::Borough.radius() => CitySize::Village,
            r if r < CitySize::Metropolis.radius() => CitySize::Borough,
            _ => CitySize::Metropolis,
        }
    }
}

/// The largest radius a city may have, whether founded at it or grown to it.
///
/// It is doing three jobs at once, which is why growth is capped here rather than
/// given a bound of its own: it is the margin a site is kept from the world edge so
/// no disc is clipped, it is half of what `resolve_spacing` relies on when it
/// assumes only the eight neighbouring regions can conflict, and it is the top tier's
/// radius. Letting a city grow past it would quietly invalidate the first two.
pub const MAX_CITY_RADIUS: u32 = 12;

/// One city. The entity carrying this *is* the record of the city existing; it
/// is spawned when the plan lands and torn down with the screen.
///
/// **Live state, not a founding record.** `radius` and `size` are what the city is
/// this frame: [`crate::gameplay::growth`] re-derives both from the town's tile
/// count every step, so a hamlet that thrives becomes a borough. Nothing keeps the
/// tier it was founded at — if a reader ever wants "founded a hamlet, now a
/// metropolis", that is a second field and not a second reading of these.
///
/// It stays `Copy`, and the road stage snapshots it by value into `RoadQueue`. That
/// snapshot is stale by construction the moment the simulation starts, and is safe
/// only because the roads are all routed before `WorldPlan::Done` — which is the
/// same gate that keeps a route from being planned against a moving city.
#[derive(Component, Clone, Copy, Debug)]
pub struct City {
    /// Stable across a session, and the seed for this city's outline.
    pub id: u32,
    /// Global tile coordinate of the city's centre. The one thing about a city that
    /// never moves.
    pub centre: IVec2,
    pub size: CitySize,
    pub radius: u32,
}

/// Which cities touch which chunk, so a lookup never scans the world. This is
/// only an index into the city entities — it holds no city data of its own.
#[derive(Resource, Default)]
pub struct CityMap {
    by_chunk: HashMap<usize, Vec<Entity>>,
}

impl CityMap {
    /// Idempotent, because a city's footprint moves: [`crate::gameplay::growth`]
    /// reports the chunks it reaches as it grows into them, and without this a city
    /// that has spent a session touching one chunk would appear in its row a
    /// thousand times.
    pub fn insert(&mut self, chunk: usize, city: Entity) {
        let row = self.by_chunk.entry(chunk).or_default();
        if !row.contains(&city) {
            row.push(city);
        }
    }

    /// The cities whose disc overlaps the given chunk. Nothing reads this yet —
    /// it is the half of the index that exists for gameplay to ask "what is the
    /// player standing in".
    #[allow(dead_code)]
    pub fn in_chunk(&self, chunk: usize) -> &[Entity] {
        self.by_chunk.get(&chunk).map_or(&[], Vec::as_slice)
    }
}

/// A city and the tiles it claims, straight out of the planner.
pub struct PlannedCity {
    pub city: City,
    pub edits: Vec<TileEdit>,
}

impl PlannedCity {
    /// Every chunk this city's tiles fall in, deduplicated — what the entity
    /// gets indexed under in [`CityMap`].
    pub fn chunks(&self) -> Vec<usize> {
        let mut chunks: Vec<usize> = self
            .edits
            .iter()
            .map(|e| chunk_index_of_tile(e.tile))
            .collect();
        chunks.sort_unstable();
        chunks.dedup();
        chunks
    }
}

/// Salt for the per-region jitter, so a city site is not the corner of its region.
const CITY_SITE_SALT: i32 = 0x1f37_2ab1u32 as i32;

/// A site before the spacing pass has had its say.
#[derive(Clone, Copy)]
struct Candidate {
    centre: IVec2,
    score: f32,
    size: CitySize,
    radius: u32,
}

/// Lays out every city in the world.
///
/// The world is cut into square regions and each one proposes at most one site,
/// which is what bounds the city count and spreads them out; the spacing pass
/// then drops a site that would overlap a better one next door.
pub fn plan_cities(
    terrain: &TerrainConfig,
    config: &WorldPlanConfig,
    world: &WorldSnapshot,
) -> Vec<PlannedCity> {
    let region = config.region_size_tiles.max(1) as i32;
    let regions = IVec2::new(WORLD_TILES.x as i32 / region, WORLD_TILES.y as i32 / region);

    let mut candidates: Vec<Option<Candidate>> =
        Vec::with_capacity((regions.x * regions.y) as usize);
    for ry in 0..regions.y {
        for rx in 0..regions.x {
            candidates.push(candidate_for_region(terrain, world, region, rx, ry));
        }
    }

    resolve_spacing(&mut candidates, regions, config.city_min_gap_tiles as i32);

    candidates
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(index, candidate)| {
            let city = City {
                id: index as u32,
                centre: candidate.centre,
                size: candidate.size,
                radius: candidate.radius,
            };
            PlannedCity {
                edits: city_edits(&city, config.city_wobble, world),
                city,
            }
        })
        .collect()
}

/// The one site a region proposes, or `None` if the region has nowhere worth
/// building — its jittered centre is water or mountain, or the settlement score
/// there is too low.
fn candidate_for_region(
    terrain: &TerrainConfig,
    world: &WorldSnapshot,
    region: i32,
    rx: i32,
    ry: i32,
) -> Option<Candidate> {
    let h = hash2(rx ^ CITY_SITE_SALT, ry);
    let jitter = IVec2::new(
        (h & 0xffff) as i32 % region,
        ((h >> 16) & 0xffff) as i32 % region,
    );
    // Kept a full radius clear of the world edge so no disc is clipped by it.
    let margin = MAX_CITY_RADIUS as i32;
    let centre = (IVec2::new(rx, ry) * region + jitter).clamp(
        IVec2::splat(margin),
        WORLD_TILES.as_ivec2() - IVec2::splat(margin + 1),
    );

    if !world.tile(centre)?.is_habitable() {
        return None;
    }

    let mut score = terrain
        .settlement_field()
        .sample(centre.x as f32, centre.y as f32);
    if is_coastal(world, centre, terrain.coast_radius as i32) {
        score += terrain.town_coast_bonus;
    }
    if score < terrain.town_threshold {
        return None;
    }

    let excess =
        ((score - terrain.town_threshold) / (1.0 - terrain.town_threshold)).clamp(0.0, 1.0);
    let size = CitySize::from_excess(excess);
    Some(Candidate {
        centre,
        score,
        size,
        radius: size.radius(),
    })
}

fn is_coastal(world: &WorldSnapshot, centre: IVec2, radius: i32) -> bool {
    (-radius..=radius).any(|dy| {
        (-radius..=radius)
            .any(|dx| world.tile(centre + IVec2::new(dx, dy)) == Some(TerrainKind::ShallowWater))
    })
}

/// Drops sites that would sit on top of each other. Only the eight neighbouring
/// regions can conflict, because a region holds one site and a disc is far
/// smaller than a region.
///
/// The scan order and the explicit tie-break are what make this deterministic:
/// the outcome must not depend on which candidate happened to be visited first.
fn resolve_spacing(candidates: &mut [Option<Candidate>], regions: IVec2, gap: i32) {
    for index in 0..candidates.len() {
        let Some(here) = candidates[index] else {
            continue;
        };
        let region = IVec2::new(index as i32 % regions.x, index as i32 / regions.x);

        for neighbour_index in neighbouring_regions(region, regions) {
            let Some(neighbour) = candidates[neighbour_index] else {
                continue;
            };
            if !too_close(&here, &neighbour, gap) {
                continue;
            }

            if here.score > neighbour.score
                || (here.score == neighbour.score && index < neighbour_index)
            {
                candidates[neighbour_index] = None;
            } else {
                candidates[index] = None;
                break;
            }
        }
    }
}

/// The indices of the up-to-eight regions around this one.
fn neighbouring_regions(region: IVec2, regions: IVec2) -> impl Iterator<Item = usize> {
    (-1..=1)
        .flat_map(move |dy| (-1..=1).map(move |dx| IVec2::new(dx, dy)))
        .filter_map(move |delta| {
            let neighbour = region + delta;
            let inside = neighbour.cmpge(IVec2::ZERO).all() && neighbour.cmplt(regions).all();
            (delta != IVec2::ZERO && inside)
                .then(|| (neighbour.y * regions.x + neighbour.x) as usize)
        })
}

fn too_close(a: &Candidate, b: &Candidate, gap: i32) -> bool {
    let required = a.radius as i64 + b.radius as i64 + gap as i64;
    let delta = (a.centre - b.centre).as_i64vec2();
    delta.x * delta.x + delta.y * delta.y < required * required
}

/// The tiles a city claims: a disc whose radius wobbles with the angle, so the
/// outline reads as round without being a drawn circle. Only habitable tiles are
/// taken, which is what clips a coastal city against its own bay.
fn city_edits(city: &City, wobble: f32, world: &WorldSnapshot) -> Vec<TileEdit> {
    let reach = (city.radius as f32 * (1.0 + wobble)).ceil() as i32;
    let mut edits = Vec::new();

    for dy in -reach..=reach {
        for dx in -reach..=reach {
            let tile = city.centre + IVec2::new(dx, dy);
            let Some(kind) = world.tile(tile) else {
                continue;
            };
            if !kind.is_habitable() {
                continue;
            }

            let distance = ((dx * dx + dy * dy) as f32).sqrt();
            let angle = (dy as f32).atan2(dx as f32);
            if distance > wobbled_radius(city.id, city.radius as f32, wobble, angle) {
                continue;
            }

            edits.push(TileEdit {
                tile,
                kind: TerrainKind::Town,
            });
        }
    }

    edits
}

/// Three harmonics with hashed phases. One harmonic would make every city an
/// egg pointing the same way; three is enough to look irregular while the
/// distance term keeps the shape unmistakably round.
fn wobbled_radius(id: u32, radius: f32, wobble: f32, angle: f32) -> f32 {
    const HARMONICS: u32 = 3;
    // Sum of 1/k over the harmonics, which is what the sum below is bounded by.
    let normalizer: f32 = (1..=HARMONICS).map(|k| 1.0 / k as f32).sum();

    let sum: f32 = (1..=HARMONICS)
        .map(|k| {
            let phase =
                (hash2(id as i32, k as i32) as f32 / u32::MAX as f32) * std::f32::consts::TAU;
            (angle * k as f32 + phase).sin() / k as f32
        })
        .sum();

    radius * (1.0 + wobble * sum / normalizer)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> WorldPlanConfig {
        WorldPlanConfig::default()
    }

    #[test]
    fn a_wobbled_outline_stays_within_the_wobble_of_the_radius() {
        let wobble = 0.25;
        for id in 0..64u32 {
            for step in 0..64 {
                let angle = step as f32 / 64.0 * std::f32::consts::TAU;
                let r = wobbled_radius(id, 8.0, wobble, angle);
                assert!(
                    (8.0 * (1.0 - wobble)..=8.0 * (1.0 + wobble)).contains(&r),
                    "city {id} at angle {angle} has radius {r}"
                );
            }
        }
    }

    /// The tie-break exists so that two equally good neighbours cannot both
    /// survive, whichever order they were visited in.
    #[test]
    fn two_overlapping_sites_never_both_survive() {
        let regions = IVec2::new(2, 1);
        let mut candidates = vec![
            Some(Candidate {
                centre: IVec2::new(10, 10),
                score: 0.7,
                size: CitySize::Village,
                radius: 5,
            }),
            Some(Candidate {
                centre: IVec2::new(13, 10),
                score: 0.7,
                size: CitySize::Village,
                radius: 5,
            }),
        ];

        resolve_spacing(&mut candidates, regions, config().city_min_gap_tiles as i32);

        assert_eq!(candidates.iter().flatten().count(), 1);
        assert!(candidates[0].is_some(), "the tie should go to scan order");
    }

    #[test]
    fn a_site_far_from_its_neighbour_survives() {
        let regions = IVec2::new(2, 1);
        let mut candidates = vec![
            Some(Candidate {
                centre: IVec2::new(10, 10),
                score: 0.7,
                size: CitySize::Village,
                radius: 5,
            }),
            Some(Candidate {
                centre: IVec2::new(200, 10),
                score: 0.9,
                size: CitySize::Village,
                radius: 5,
            }),
        ];

        resolve_spacing(&mut candidates, regions, config().city_min_gap_tiles as i32);

        assert_eq!(candidates.iter().flatten().count(), 2);
    }

    #[test]
    fn a_bigger_score_means_a_bigger_city() {
        assert_eq!(CitySize::from_excess(0.0), CitySize::Hamlet);
        assert_eq!(CitySize::from_excess(1.0), CitySize::Metropolis);
        assert!(CitySize::Hamlet.radius() < CitySize::Metropolis.radius());
        assert_eq!(CitySize::Metropolis.radius(), MAX_CITY_RADIUS);
    }
}
