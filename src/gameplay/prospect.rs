//! How good the ground is for iron, copper and salt — the map behind the seams.
//!
//! gh-28's inspection overlay draws the scalar fields the world is built out of with
//! **no new maps**: every field it shows was already bound to the screen pass because
//! something else needed it there. This one breaks that, and it is worth being honest
//! about why. What a resource overlay wants to draw is [`DepositRecipe`]'s score,
//! which is built from the tile's **kind**, its **biome** and its **elevation**. The
//! elevation is bound; the other two are not and cannot be — there is no world-wide
//! kind texture, only a per-chunk index image. So this is the first overlay field with
//! a bake behind it, on the same terms the weather's probability map is baked: on the
//! task pool, once a session, absent until it lands.
//!
//! **What it draws is prospectivity, not seams.** The field is "how good is this
//! ground for iron" — the recipe score at every point, *before* the threshold. A
//! deposit exists only where a cell's own jittered candidate happened to land on
//! ground that clears it, so the map answers "could there be iron here", and "is
//! there" is answered by the mark channel below and by `observe deposits`. Conflating
//! the two would make the overlay lie about a field it names.
//!
//! **The threshold is the neutral band.** Temperature is on the diverging ramp because
//! it has a meaningful zero — the freezing point — and the payoff is that the snow
//! line is visible without reading a number. A recipe score has exactly the same shape
//! of zero: `deposit_threshold` is where ground stops being ordinary and starts being
//! worth digging. So these three fields are diverging too, with the mid read **from
//! the config that decides it** rather than restated here, the way the temperature
//! overlay reads `GroundConfig::freezing_celsius`.
//!
//! Wood and stone get no overlay. They are `Forest` and `Rock`, `Mountain` and
//! `Gravel` — you can already see them, and a false-colour map of "there is a wood
//! here" over a drawn wood is a worse picture of the same fact.

use crate::gameplay::terrain::TerrainSampler;
use bevy::{
    asset::RenderAssetUsages,
    image::ImageFilterMode,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::TextureFormat,
    },
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};

use crate::{
    gameplay::{
        deposit::{Deposit, RECIPES, Resource},
        ground::map_image,
        world::{WORLD_TILES, WorldSnapshot},
    },
    screens::Screen,
};

/// How many texels across the world is scored.
///
/// One texel per 8 tiles, which is the same resolution the weather's probability map
/// is baked at and for the same reason: prospectivity is a broad field — the recipes
/// read a biome and an elevation band, neither of which has structure at the tile.
/// The bake is one `TerrainSampler::sample` per texel, so this is quadratic in the
/// wall clock it takes.
pub const PROSPECT_TEXELS_PER_SIDE: u32 = 512;

/// How many texels around a seam are marked.
///
/// Small: the mark says "there is one here", not how big it is, and a blob wide enough
/// to read as an area would misreport a point as a field.
const SEAM_TEXEL_RADIUS: i32 = 2;

/// The three resources with a recipe, in the order their scores are packed into the
/// map's first three channels. The fourth channel is the seam category.
///
/// Not `Resource::ALL`: `Food`, `Wood` and `Stone` are area resources with no recipe
/// and so no score to draw. A seventh resource with a recipe would want a fourth
/// channel, which is where this stops being free.
const SCORED: [Resource; 3] = [Resource::Iron, Resource::Copper, Resource::Salt];

/// What one resource's seam encodes in the fourth channel.
///
/// A **category, not a magnitude**: 0 is no seam and each resource is its own step, so
/// one channel serves three overlays without a seam of iron showing up on the salt
/// map. The steps are far enough apart that byte quantization cannot move one into
/// another's tolerance.
fn seam_mark(resource: Resource) -> u8 {
    match SCORED.iter().position(|r| *r == resource) {
        Some(index) => ((index + 1) * 64) as u8,
        None => 0,
    }
}

/// Which channel a field's score is in, or `None` for a resource with no recipe.
pub fn scored_channel(resource: Resource) -> Option<usize> {
    SCORED.iter().position(|r| *r == resource)
}

/// The seam mark the shader should be looking for, on 0..1 — what the uniform carries
/// so the comparison is written once, here, rather than as a constant in the wgsl.
pub fn seam_mark_unit(resource: Resource) -> f32 {
    seam_mark(resource) as f32 / 255.0
}

/// The baked map, CPU side — what `observe overlay` reads a value from.
///
/// One texel per `PROSPECT_TEXELS_PER_SIDE`, holding the iron, copper and salt scores
/// in three channels and the seam category in the fourth.
#[derive(Resource)]
pub struct ProspectMaps {
    side: u32,
    texels: Vec<u8>,
}

impl ProspectMaps {
    /// One resource's score at a tile, on 0..1. `None` for a resource with no recipe,
    /// or a tile outside the world.
    pub fn score_at(&self, tile: Vec2, resource: Resource) -> Option<f32> {
        let channel = scored_channel(resource)?;
        let texel = (tile / WORLD_TILES.as_vec2() * self.side as f32)
            .floor()
            .as_ivec2();
        if texel.cmplt(IVec2::ZERO).any() || texel.cmpge(IVec2::splat(self.side as i32)).any() {
            return None;
        }
        let index = (texel.y as usize * self.side as usize + texel.x as usize) * 4 + channel;
        Some(self.texels[index] as f32 / 255.0)
    }
}

/// The map's handle, extracted to the render world the way every other map's is.
#[derive(Resource, Clone)]
pub struct ProspectTexture(pub(super) Handle<Image>);

impl ExtractResource for ProspectTexture {
    type Source = Self;

    fn extract_resource(source: &Self) -> Self {
        source.clone()
    }
}

/// The bake in flight. Its absence *after* the plan is what `wait prospect` waits on —
/// while it is here, the three fields draw nothing.
#[derive(Resource)]
pub struct ProspectBake(Task<Vec<u8>>);

pub struct ProspectPlugin;

impl Plugin for ProspectPlugin {
    fn build(&self, app: &mut App) {
        // Without this the handle never reaches the render world and the pass binds
        // its blank for the whole session — which looks exactly like a bake that has
        // not landed, and so says nothing.
        app.add_plugins(ExtractResourcePlugin::<ProspectTexture>::default());
        app.add_systems(Update, finish_prospect_bake);
        app.add_systems(OnExit(Screen::Gameplay), tear_down_prospect);
    }
}

/// Starts the bake from the **same snapshot the seams were laid from**.
///
/// Called by the Deposits stage rather than by a system of its own, because that stage
/// is the one that holds the snapshot — so the map and the seams on it can never
/// disagree about the world they read. Taking a second snapshot later would be a
/// second answer to the same question, which is the mistake the crate's one-sampler
/// rule exists to prevent.
///
/// It does **not** gate the plan: the city stage opens as soon as the layout is down,
/// and the map lands whenever it lands.
pub fn start_prospect_bake(
    commands: &mut Commands,
    sampler: &TerrainSampler,
    world: WorldSnapshot,
    sites: Vec<Deposit>,
) {
    let sampler = sampler.clone();
    commands
        .insert_resource(ProspectBake(AsyncComputeTaskPool::get().spawn(
            async move { bake(&sampler, &world, &sites, PROSPECT_TEXELS_PER_SIDE) },
        )));
}

fn finish_prospect_bake(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    bake: Option<ResMut<ProspectBake>>,
) {
    let Some(mut bake) = bake else {
        return;
    };
    let Some(texels) = block_on(poll_once(&mut bake.0)) else {
        return;
    };

    let side = PROSPECT_TEXELS_PER_SIDE;
    commands.insert_resource(ProspectTexture(images.add(map_image(
        side,
        texels.clone(),
        TextureFormat::Rgba8Unorm,
        // Kept on the CPU too, because `observe overlay` reads a value from it — the
        // same cross-check every other field's observation is.
        RenderAssetUsages::RENDER_WORLD,
        // Linear, like every other field: three of the four channels are a continuous
        // score and the blocking is a true property of the resolution it was taken at.
        // The **fourth** is a category and must not be filtered — the shader reads that
        // one with `textureLoad`, which is where that rule is enforced.
        ImageFilterMode::Linear,
    ))));
    commands.insert_resource(ProspectMaps { side, texels });
    commands.remove_resource::<ProspectBake>();
}

/// The maps are world state and go with the session; a bake still in flight is
/// cancelled by dropping it, exactly as the plan's tasks are.
fn tear_down_prospect(mut commands: Commands) {
    commands.remove_resource::<ProspectBake>();
    commands.remove_resource::<ProspectMaps>();
    commands.remove_resource::<ProspectTexture>();
}

/// Scores the whole world against the three recipes, then stamps the seams on top.
fn bake(sampler: &TerrainSampler, world: &WorldSnapshot, sites: &[Deposit], side: u32) -> Vec<u8> {
    let mut texels = vec![0u8; (side as usize).pow(2) * 4];

    for ty in 0..side {
        for tx in 0..side {
            // The texel's centre in tile space, so a texel reports the ground in the
            // middle of what it covers rather than at its corner.
            let tile = ((Vec2::new(tx as f32, ty as f32) + Vec2::splat(0.5)) / side as f32)
                * WORLD_TILES.as_vec2();
            let Some(kind) = world.tile(tile.as_ivec2()) else {
                continue;
            };
            let sample = sampler.sample(tile.x, tile.y);

            let base = (ty as usize * side as usize + tx as usize) * 4;
            for (channel, resource) in SCORED.iter().enumerate() {
                let score = RECIPES
                    .iter()
                    .find(|recipe| recipe.resource == *resource)
                    .map_or(0.0, |recipe| {
                        recipe.score(kind, sample.dominant, sample.elevation)
                    });
                texels[base + channel] = (score.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }

    // The seams are read out of the world once, into the task's own list, *before* it
    // is spawned — a task cannot hold a query, and a seam that were despawned mid-bake
    // would be one the map still drew.
    for site in sites {
        let centre = (site.tile.as_vec2() / WORLD_TILES.as_vec2() * side as f32).as_ivec2();
        let mark = seam_mark(site.resource);
        for dy in -SEAM_TEXEL_RADIUS..=SEAM_TEXEL_RADIUS {
            for dx in -SEAM_TEXEL_RADIUS..=SEAM_TEXEL_RADIUS {
                let texel = centre + IVec2::new(dx, dy);
                if texel.cmplt(IVec2::ZERO).any() || texel.cmpge(IVec2::splat(side as i32)).any() {
                    continue;
                }
                let index = (texel.y as usize * side as usize + texel.x as usize) * 4 + 3;
                texels[index] = mark;
            }
        }
    }

    texels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gameplay::terrain::TerrainConfig;

    use crate::gameplay::terrain::shared_test_sampler;
    use crate::gameplay::{
        deposit::plan_deposits, plan::WorldPlanConfig, terrain::TerrainKind, world::WorldMap,
    };

    /// A category, not a magnitude — and the steps have to stay far enough apart that
    /// byte quantization and the shader's tolerance cannot confuse two of them.
    #[test]
    fn every_seam_mark_is_its_own_step_with_room_around_it() {
        let mut marks: Vec<u8> = SCORED.iter().map(|r| seam_mark(*r)).collect();
        marks.push(0);
        marks.sort_unstable();
        for pair in marks.windows(2) {
            assert!(
                pair[1] - pair[0] >= 32,
                "marks {} and {} are too close to tell apart",
                pair[0],
                pair[1]
            );
        }
        // And a resource with no recipe has no mark at all, or it would draw a seam of
        // something the map does not score.
        for resource in [Resource::Food, Resource::Wood, Resource::Stone] {
            assert_eq!(seam_mark(resource), 0, "{resource:?} got a seam mark");
            assert_eq!(scored_channel(resource), None);
        }
    }

    /// Every scored resource has a channel of its own, and the channels are the first
    /// three — the fourth is the seam category.
    #[test]
    fn each_scored_resource_owns_one_channel() {
        for (index, resource) in SCORED.iter().enumerate() {
            assert_eq!(scored_channel(*resource), Some(index));
            assert!(index < 3, "{resource:?} has no channel left to live in");
        }
    }

    /// **The map draws prospectivity, and the seams are the fourth channel.** A texel
    /// can score well without a seam being there — a seam exists only where a cell's
    /// own jittered candidate happened to land — so the two must not be read off each
    /// other.
    #[test]
    fn a_high_score_is_not_a_seam_and_a_seam_is_not_a_score() {
        let _terrain = TerrainConfig::default();
        let config = WorldPlanConfig::default();
        let world = WorldMap::from_fn(|tile| {
            if tile.x % 3 == 0 {
                TerrainKind::Rock
            } else {
                TerrainKind::Gravel
            }
        })
        .snapshot()
        .expect("from_fn fills every chunk");

        let sites = plan_deposits(shared_test_sampler(), &config, &world);
        let side = 128u32;
        let texels = bake(shared_test_sampler(), &world, &sites, side);

        // Counting is no good here — a seam's mark is a disc of texels, so how it
        // compares to the scored area is a fact about the resolution rather than about
        // the map. What is true at every resolution is that the two are *different
        // questions*.
        let mut scored_unmarked = 0;
        let mut marked = 0;
        for index in 0..(side as usize).pow(2) {
            let texel = &texels[index * 4..index * 4 + 4];
            let scores = &texel[..3];
            if texel[3] == 0 && scores.iter().any(|s| *s > 0) {
                scored_unmarked += 1;
            }
            if texel[3] == 0 {
                continue;
            }
            marked += 1;
            // Every mark is one of the categories and never a blend of two — which is
            // what the fourth channel being categorical means, and what the shader's
            // `textureLoad` is there to preserve.
            assert!(
                SCORED.iter().any(|r| seam_mark(*r) == texel[3]),
                "a mark of {} is not one of the categories",
                texel[3]
            );
            // Note what is deliberately *not* asserted: that a marked texel scores well
            // for its own resource. A seam is laid at a *tile* and the map is sampled at
            // a texel's centre, so the two are up to half a texel apart and the score
            // there can honestly be zero. That gap is the resolution talking, and it is
            // the reason the seam gets a channel of its own rather than being inferred
            // from a high score.
        }

        assert!(marked > 0, "no seam reached the map");
        assert!(
            scored_unmarked > 0,
            "every scoring texel has a seam on it — the map is drawing seams, not \
             prospectivity"
        );
    }

    /// The map is a pure function of the world it was baked from, which is what lets it
    /// and the layout share one snapshot without either being re-derived.
    #[test]
    fn the_same_world_bakes_the_same_map() {
        let _terrain = TerrainConfig::default();
        let config = WorldPlanConfig::default();
        let world = WorldMap::from_fn(|tile| {
            if tile.y % 5 < 2 {
                TerrainKind::Sand
            } else {
                TerrainKind::Marsh
            }
        })
        .snapshot()
        .expect("from_fn fills every chunk");
        let sites = plan_deposits(shared_test_sampler(), &config, &world);

        assert_eq!(
            bake(shared_test_sampler(), &world, &sites, 48),
            bake(shared_test_sampler(), &world, &sites, 48)
        );
    }

    /// A tile's score comes back where the map says it is, and a resource with no
    /// recipe has none anywhere.
    #[test]
    fn a_score_reads_back_at_the_tile_it_was_baked_for() {
        let side = 32u32;
        let mut texels = vec![0u8; (side as usize).pow(2) * 4];
        // One texel in the middle, iron at full score.
        let middle = (side as usize / 2) * side as usize + side as usize / 2;
        texels[middle * 4] = 255;
        let maps = ProspectMaps { side, texels };

        let tile = (Vec2::splat(0.5) + Vec2::splat(side as f32 / 2.0) / side as f32 * 0.0)
            * WORLD_TILES.as_vec2();
        assert_eq!(maps.score_at(tile, Resource::Iron), Some(1.0));
        assert_eq!(maps.score_at(tile, Resource::Copper), Some(0.0));
        assert_eq!(maps.score_at(tile, Resource::Wood), None);
        assert_eq!(
            maps.score_at(Vec2::splat(-1.0), Resource::Iron),
            None,
            "a tile outside the world reported a score"
        );
    }
}
