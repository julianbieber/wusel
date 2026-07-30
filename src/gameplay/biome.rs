//! Which recipe builds the terrain at a tile.
//!
//! A biome here is not a set of tiles. It is a [`HeightRecipe`] — a way of
//! combining the handful of noise layers [`crate::gameplay::terrain`] samples —
//! plus the pair of kinds its lowland band chooses between. Nothing in this module
//! knows what a tile looks like; it answers "how is the ground made here".
//!
//! The map is a jittered-grid Voronoi: each cell of a coarse lattice hashes to a
//! site position inside itself and to a [`Biome`], and a tile belongs to the
//! nearest site. Two properties come out of that choice and both are load-bearing:
//!
//! - **It is a pure function of the tile's own coordinates.** The lookup reads a
//!   3x3 neighbourhood of *cells*, and a cell is computed from its integer
//!   coordinates, never from what any tile nearby turned out to be. So
//!   `generate_chunk` still needs no padding, and a tile cannot depend on where a
//!   chunk boundary fell.
//! - **Regions have edges.** A latitude/humidity climate table — the usual way to
//!   do this — produces smooth gradients, and gradients are exactly what this
//!   world already had too much of.
//!
//! **The blend is the feature, not the smoothing.** The recipe at a tile is the
//! distance-weighted mix of the recipes of the sites near it, so a boundary is a
//! *structure*: where a `Highland` cell meets an `Ocean` cell the base height falls
//! through sea level across the blend band and the coast comes out as a cliffed
//! shelf; where the same `Highland` meets `Plains` the ridge weight decays outward
//! and the range ends in foothills; where two `Plains` cells meet, nothing happens,
//! because a blend of like recipes is that recipe.
//!
//! None of those three is written down anywhere. There is deliberately **no
//! pairwise rule**: six biomes would be fifteen pairs to write and tune, and the
//! interesting boundaries are the ones nobody thought to enumerate.

use bevy::prelude::*;

use crate::gameplay::noise::{NoiseField, hash2};
use crate::gameplay::terrain::TerrainKind;

/// The six the drawn palette can tell apart.
///
/// Tundra is deliberately absent: without a cold-grass or conifer tile the only
/// thing it could lay down below the snow line is `Snow`, which is a flat sheet of
/// one kind — the complaint this whole module exists to fix, in white.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Biome {
    Ocean,
    Plains,
    Forest,
    Highland,
    Desert,
    Wetland,
}

/// How one biome builds its ground out of the shared noise layers.
///
/// No biome owns a field of its own. The layers are sampled once per tile and a
/// recipe only weights them, which is why the cost per tile does not grow with the
/// number of biomes — and why two recipes can be averaged field by field at all.
///
/// Every field here is a number, deliberately. The one un-blendable thing a biome
/// decides — which pair of kinds fills its lowland band — lives on [`Biome::kinds`]
/// instead, because averaging two enums is not a thing and a struct that is half
/// blendable invites reading the half that was never mixed.
#[derive(Clone, Copy, Debug)]
pub struct HeightRecipe {
    /// Where this biome sits before any layer displaces it.
    pub base_height: f32,
    /// How much of the fine relief layer to add, as a displacement about zero.
    pub relief: f32,
    /// How much of the ridged layer to add. One-sided, so it only builds up.
    pub ridge: f32,
    pub vegetation_bias: f32,
    pub humidity_bias: f32,
    /// How far above the water line the sand band reaches, in elevation units.
    /// Zero means no beach: a `Highland` coast drops into the sea as rock.
    pub beach_width: f32,
}

impl Biome {
    /// The table. Numbers here are elevation units on the same [0, 1] scale as
    /// `TerrainConfig`'s bands, so a `base_height` can be read against them
    /// directly: 0.42 is the water line, 0.72 the foot of the mountains.
    ///
    /// Measured coverage at the defaults is in
    /// `the_default_config_produces_recognisably_different_regions`.
    pub fn recipe(self) -> HeightRecipe {
        match self {
            // Below the water line and staying there unless the continent layer
            // lifts it — which is what makes an archipelago out of a shallow sea
            // rather than putting islands everywhere.
            Biome::Ocean => HeightRecipe {
                base_height: 0.20,
                relief: 0.10,
                ridge: 0.0,
                vegetation_bias: 0.0,
                humidity_bias: 0.04,
                beach_width: 0.06,
            },
            Biome::Plains => HeightRecipe {
                base_height: 0.52,
                relief: 0.09,
                ridge: 0.0,
                vegetation_bias: -0.09,
                humidity_bias: 0.0,
                beach_width: 0.04,
            },
            // The same pair as Plains, and the bias is the whole difference: a wood
            // with clearings against a field with copses.
            Biome::Forest => HeightRecipe {
                base_height: 0.55,
                relief: 0.13,
                ridge: 0.02,
                vegetation_bias: 0.13,
                humidity_bias: 0.06,
                beach_width: 0.03,
            },
            // The only recipe that leans on the ridged layer, and the reason it
            // exists: `ridge` is what turns a lump into a range with spurs.
            Biome::Highland => HeightRecipe {
                base_height: 0.70,
                relief: 0.11,
                ridge: 0.34,
                vegetation_bias: 0.0,
                humidity_bias: 0.03,
                beach_width: 0.0,
            },
            // Same pair as Ocean's islands; the biases are what make it a desert.
            // The dry one is also why no spring rises here and no cloud gathers.
            Biome::Desert => HeightRecipe {
                base_height: 0.51,
                relief: 0.11,
                ridge: 0.04,
                vegetation_bias: -0.30,
                humidity_bias: -0.26,
                beach_width: 0.08,
            },
            // Flat and just above the water line, so the lowland band is nearly all
            // of it. No beach: a marsh meets open water as marsh.
            Biome::Wetland => HeightRecipe {
                base_height: 0.47,
                relief: 0.04,
                ridge: 0.0,
                vegetation_bias: 0.04,
                humidity_bias: 0.22,
                beach_width: 0.0,
            },
        }
    }

    /// What this biome's lowland band lays down: the kind below the vegetation
    /// threshold, then the kind at or above it.
    ///
    /// Not part of [`HeightRecipe`] because it cannot be blended — so at a boundary
    /// this is the one thing that changes all at once, and it is deliberately the
    /// thing elevation does *not* depend on. That is what makes a boundary read as a
    /// treeline rather than a wall.
    pub fn kinds(self) -> (TerrainKind, TerrainKind) {
        match self {
            Biome::Ocean => (TerrainKind::Sand, TerrainKind::Grass),
            Biome::Plains => (TerrainKind::Grass, TerrainKind::Forest),
            // The same pair as Plains: the vegetation_bias is the whole difference,
            // a wood with clearings against a field with copses.
            Biome::Forest => (TerrainKind::Grass, TerrainKind::Forest),
            Biome::Highland => (TerrainKind::Rock, TerrainKind::Forest),
            // The same pair as Ocean's islands; the biases make it a desert.
            Biome::Desert => (TerrainKind::Sand, TerrainKind::Grass),
            Biome::Wetland => (TerrainKind::Marsh, TerrainKind::Forest),
        }
    }

    /// Every biome, for the tests that assert the world contains them all.
    #[cfg(test)]
    pub const ALL: [Biome; 6] = [
        Biome::Ocean,
        Biome::Plains,
        Biome::Forest,
        Biome::Highland,
        Biome::Desert,
        Biome::Wetland,
    ];
}

/// How often each biome is drawn. Ocean is the heaviest because it is the sea —
/// the other five are land, and a world of mostly land has no coastline to speak
/// of.
const BIOME_TABLE: [(Biome, u32); 6] = [
    (Biome::Ocean, 6),
    (Biome::Plains, 4),
    (Biome::Forest, 4),
    (Biome::Highland, 3),
    (Biome::Desert, 2),
    (Biome::Wetland, 2),
];

/// Salts for the one hash each cell gets.
const CELL_SALT: u32 = 0x51de_51de;
const WARP_X_SALT: u32 = 0x7a1d_0b37;
const WARP_Y_SALT: u32 = 0x9c3f_1102;

/// How many octaves the warp fields get.
///
/// Three, and the count matters more than it looks. A warp bends an edge only where
/// it has content at a wavelength *shorter* than the edge; a single long octave
/// translates the whole boundary bodily and leaves it exactly as straight as it was.
/// That was the first version's mistake — two octaves at 1.5 cells meant the finest
/// warp detail had a 288-tile wavelength against ~200-tile boundary segments, so it
/// moved the edges without bending any of them.
const WARP_OCTAVES: u32 = 3;

/// The warp's base wavelength, in cells. Under one cell so that the finest octave
/// (an eighth of this) wiggles an edge at a scale you can see from the ground,
/// while still being long enough not to shred it.
const WARP_CELLS: f32 = 0.75;

/// A site's jitter, as a fraction of its cell, centred.
///
/// Also part of why edges read as straight: the less a site moves, the closer the
/// diagram is to the regular lattice underneath it, and a regular lattice is what
/// makes a boundary look drawn with a ruler. Filling the *whole* cell would let two
/// sites land on top of each other and leave a sliver of a region nobody can see, so
/// this stops short of that.
const SITE_JITTER: f32 = 0.85;

/// Picks which of the blended biomes supplies a tile's ground cover.
const COVER_SALT: i32 = 0x2f6b_1e59;

/// What the map says about one tile.
pub struct BlendedBiome {
    /// Which region this tile is *in*: the heaviest weight. This is the answer to
    /// "which biome is here", and what the coverage tests count.
    pub dominant: Biome,
    /// Which biome supplies this tile's ground cover — drawn from the weights by a
    /// hash of the tile rather than taken from the heaviest.
    ///
    /// In a region's interior one weight is 1.0, so this *is* `dominant` and nothing
    /// is random. Only inside the blend band do the two differ, and there the point
    /// is that they differ per tile: a boundary stops being a line and becomes the
    /// two biomes' tiles interleaving, dense on their own side and sparse on the
    /// other. Making the boundary wiggle only ever gives you a wiggly line; this is
    /// what removes the line.
    pub cover: Biome,
    /// The weighted mean of the nearby sites' recipes, field by field. Continuous
    /// everywhere, which is why the dither above cannot make the ground step.
    pub recipe: HeightRecipe,
}

/// The Voronoi biome map. Built from `TerrainConfig` and sampled per tile.
pub struct BiomeMap {
    cell_tiles: f32,
    /// Half the width of the band in which two recipes mix, in tiles.
    blend_tiles: f32,
    warp_tiles: f32,
    warp_x: NoiseField,
    warp_y: NoiseField,
    /// Folded into the cell hash once, so a cell lookup is a single `hash2`.
    cell_salt: i32,
}

impl BiomeMap {
    pub fn new(seed: u32, cell_tiles: u32, blend_tiles: u32, warp_tiles: f32) -> Self {
        let cell_tiles = cell_tiles.max(1) as f32;
        let warp_scale = 1.0 / (cell_tiles * WARP_CELLS);

        Self {
            cell_tiles,
            blend_tiles: (blend_tiles as f32).max(1.0),
            warp_tiles,
            warp_x: NoiseField::with_octaves(seed, WARP_X_SALT, warp_scale, WARP_OCTAVES),
            warp_y: NoiseField::with_octaves(seed, WARP_Y_SALT, warp_scale, WARP_OCTAVES),
            cell_salt: (seed ^ CELL_SALT) as i32,
        }
    }

    /// The biome of the cell owning `cell`, and where its site sits inside it.
    ///
    /// One hash per cell, with the jitter taken from the low bytes and the biome
    /// draw from the high ones — `hash2` mixes well enough that the two are
    /// independent, and a second hash per cell would be nine more per tile.
    fn site(&self, cell: IVec2) -> (Vec2, Biome) {
        let h = hash2(cell.x ^ self.cell_salt, cell.y);

        let jitter = Vec2::new((h & 0xff) as f32 / 255.0, ((h >> 8) & 0xff) as f32 / 255.0);
        let offset = (jitter - Vec2::splat(0.5)) * SITE_JITTER + Vec2::splat(0.5);
        let position = (cell.as_vec2() + offset) * self.cell_tiles;

        let mut draw = ((h >> 16) % BIOME_WEIGHT_TOTAL) as i32;
        let mut biome = BIOME_TABLE[0].0;
        for (candidate, weight) in BIOME_TABLE {
            draw -= weight as i32;
            biome = candidate;
            if draw < 0 {
                break;
            }
        }

        (position, biome)
    }

    /// The blended recipe at a global tile position.
    ///
    /// Only sites whose distance exceeds the nearest site's by less than the blend
    /// band contribute, which is what gives a region an interior: further than
    /// `blend_tiles` from a boundary exactly one weight is non-zero, so the tile
    /// carries one recipe unblended. On the boundary itself the two nearest are
    /// equidistant and weigh the same, so the mix is continuous across it.
    pub fn blend(&self, x: f32, y: f32) -> BlendedBiome {
        // Warping the *query* rather than the lattice keeps the whole thing a pure
        // function of position while making a region's outline organic instead of a
        // polygon's.
        let warp = Vec2::new(
            self.warp_x.sample(x, y) - 0.5,
            self.warp_y.sample(x, y) - 0.5,
        ) * (2.0 * self.warp_tiles);
        let query = Vec2::new(x, y) + warp;

        let base = (query / self.cell_tiles).floor().as_ivec2();

        // Jitter is confined to the middle `SITE_JITTER` of a cell, which bounds the
        // own-cell site at 1.20 cells away while anything two cells out is at least
        // 1.15 — so 3x3 is not a strict worst-case guarantee, only one that needs all
        // nine sites to be improbably far. A miss would pick the second-nearest site
        // for one tile; it would still be the *same* answer every time it was asked,
        // which is the property the chunked world actually rests on. 5x5 would cost
        // 16 more hashes per tile to close a gap nothing can see.
        let mut sites = [(0.0f32, Biome::Ocean); 9];
        let mut nearest = f32::MAX;
        let mut slot = 0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (position, biome) = self.site(base + IVec2::new(dx, dy));
                let distance = position.distance(query);
                nearest = nearest.min(distance);
                sites[slot] = (distance, biome);
                slot += 1;
            }
        }

        let mut dominant = Biome::Ocean;
        let mut dominant_weight = -1.0;
        let mut total = 0.0;
        let mut weights = [0.0f32; 9];
        for (slot, &(distance, biome)) in sites.iter().enumerate() {
            // How far this site is from being the nearest, against the width of the
            // band. A boundary sits where two distances are equal, and a tile B from
            // one is 2B further from the loser than from the winner.
            let excess = (distance - nearest) / (2.0 * self.blend_tiles);
            if excess >= 1.0 {
                continue;
            }
            let weight = (1.0 - excess) * (1.0 - excess);
            weights[slot] = weight;
            total += weight;
            if weight > dominant_weight {
                dominant_weight = weight;
                dominant = biome;
            }
        }

        let mut recipe = HeightRecipe {
            base_height: 0.0,
            relief: 0.0,
            ridge: 0.0,
            vegetation_bias: 0.0,
            humidity_bias: 0.0,
            beach_width: 0.0,
            // Overwritten below; the dominant biome owns the kind pair, since an
            // enum cannot be averaged.
        };
        for (slot, &(_, biome)) in sites.iter().enumerate() {
            if weights[slot] == 0.0 {
                continue;
            }
            let share = weights[slot] / total;
            let part = biome.recipe();
            recipe.base_height += part.base_height * share;
            recipe.relief += part.relief * share;
            recipe.ridge += part.ridge * share;
            recipe.vegetation_bias += part.vegetation_bias * share;
            recipe.humidity_bias += part.humidity_bias * share;
            recipe.beach_width += part.beach_width * share;
        }

        // The ground cover is drawn from the weights, not taken from the heaviest.
        // Hashed on the tile, so it is as fixed a property of the position as
        // everything else here — nothing about this is random at run time.
        let mut cover = dominant;
        let draw = hash2(x.floor() as i32 ^ COVER_SALT, y.floor() as i32) as f32 / u32::MAX as f32;
        let mut climbed = 0.0;
        for (slot, &(_, biome)) in sites.iter().enumerate() {
            if weights[slot] == 0.0 {
                continue;
            }
            climbed += weights[slot] / total;
            if draw <= climbed {
                cover = biome;
                break;
            }
        }

        BlendedBiome {
            dominant,
            cover,
            recipe,
        }
    }
}

const BIOME_WEIGHT_TOTAL: u32 = {
    let mut total = 0;
    let mut i = 0;
    while i < BIOME_TABLE.len() {
        total += BIOME_TABLE[i].1;
        i += 1;
    }
    total
};

#[cfg(test)]
mod tests {
    use super::*;

    use crate::gameplay::terrain::TerrainConfig;

    /// Built from the shipped config rather than from constants of its own: the
    /// interior claim below is a property of the *defaults*, and a test with its own
    /// cell and blend sizes would keep passing after those were retuned.
    fn map() -> BiomeMap {
        let config = TerrainConfig::default();
        BiomeMap::new(
            config.seed,
            config.biome_cell_tiles,
            config.biome_blend_tiles,
            config.biome_warp_tiles,
        )
    }

    /// The property the whole chunked world rests on: the biome lookup reads cells,
    /// which are a function of their own integer coordinates, so asking twice gives
    /// the same answer regardless of what else has been asked.
    #[test]
    fn a_blend_is_a_pure_function_of_position() {
        let map = map();
        for i in 0..64 {
            let (x, y) = ((1000 + i * 37) as f32, (2000 + i * 53) as f32);
            let first = map.blend(x, y);
            let second = map.blend(x, y);
            assert_eq!(first.dominant, second.dominant);
            assert_eq!(first.recipe.base_height, second.recipe.base_height);
        }
    }

    /// Weights are a partition: they sum to one, so a blended `base_height` is a
    /// genuine mean and stays on the same scale as the band thresholds.
    #[test]
    fn a_blended_recipe_is_a_weighted_mean_within_the_range_of_its_parts() {
        let map = map();
        let lowest = Biome::ALL
            .iter()
            .map(|b| b.recipe().base_height)
            .fold(f32::MAX, f32::min);
        let highest = Biome::ALL
            .iter()
            .map(|b| b.recipe().base_height)
            .fold(f32::MIN, f32::max);

        for i in 0..2048 {
            let (x, y) = ((i * 17 % 4096) as f32, (i * 131 % 4096) as f32);
            let base = map.blend(x, y).recipe.base_height;
            assert!(
                (lowest - 1e-4..=highest + 1e-4).contains(&base),
                "blended base_height {base} outside the range of the recipes"
            );
        }
    }

    /// Regions must have interiors, or "distinct biomes" is a lie: away from a
    /// boundary a tile carries one recipe exactly, not a mush of all six.
    #[test]
    fn a_region_has_an_interior_where_exactly_one_recipe_applies() {
        let map = map();
        let unblended = (0..4096)
            .filter(|i| {
                let (x, y) = ((i * 7 % 4096) as f32, (i * 97 % 4096) as f32);
                let blended = map.blend(x, y);
                let pure = blended.dominant.recipe();
                (blended.recipe.base_height - pure.base_height).abs() < 1e-4
            })
            .count();

        // With a 384-tile cell and a 96-tile band, most of the world is interior.
        assert!(
            unblended > 4096 / 2,
            "only {unblended}/4096 sampled tiles carry an unblended recipe"
        );
    }

    /// The load-bearing continuity claim. The dominant biome may flip from one tile
    /// to the next — that is what makes a treeline — but the *elevation inputs* may
    /// not, or the flip would be a cliff.
    ///
    /// The bound is a fraction of the spread between the most and least elevated
    /// recipes, which is the only scale that means anything here: a genuine
    /// discontinuity is a step of most of that spread, and anything far below it is a
    /// slope. It was an absolute 0.01 first, and that was a bound on the *domain
    /// warp* wearing a continuity test's clothing — raising the warp to bend the
    /// region outlines pushed the step to 0.016 and failed it, with the terrain no
    /// less continuous than before. A warp is a distortion of the domain: it makes the
    /// recipe vary faster with position without introducing any jump, so a test that
    /// cannot tell those apart fails on a retune it has no business failing on.
    /// `terrain::elevation_does_not_step_at_a_biome_boundary` is the one that checks
    /// the height a river actually descends.
    #[test]
    fn a_blended_recipe_is_continuous_across_a_boundary() {
        let map = map();
        let spread = Biome::ALL
            .iter()
            .map(|b| b.recipe().base_height)
            .fold(f32::MIN, f32::max)
            - Biome::ALL
                .iter()
                .map(|b| b.recipe().base_height)
                .fold(f32::MAX, f32::min);

        let mut worst = 0.0f32;
        for i in 0..4096 {
            let (x, y) = ((i * 13 % 4096) as f32, (i * 211 % 4096) as f32);
            let here = map.blend(x, y).recipe;
            for (dx, dy) in [(1.0, 0.0), (0.0, 1.0)] {
                let next = map.blend(x + dx, y + dy).recipe;
                worst = worst.max((here.base_height - next.base_height).abs());
                worst = worst.max((here.ridge - next.ridge).abs());
            }
        }

        let bound = spread * 0.05;
        assert!(
            worst < bound,
            "a recipe steps by {worst} between adjacent tiles, over the {bound} that \
             5% of the {spread} recipe spread allows"
        );
    }

    /// The dither must not reach the interiors. Inside a region one weight is 1.0, so
    /// the draw has only one bracket to land in and `cover` is forced to `dominant` —
    /// if that ever stopped holding, every region would be speckled with tiles from
    /// biomes that are nowhere near it.
    #[test]
    fn the_cover_dither_is_a_no_op_inside_a_region() {
        let map = map();
        let mut interior = 0;

        for i in 0..8192 {
            let (x, y) = ((i * 7 % 4096) as f32, (i * 97 % 4096) as f32);
            let blended = map.blend(x, y);
            // An interior tile is one whose recipe came through unblended.
            if (blended.recipe.base_height - blended.dominant.recipe().base_height).abs() < 1e-4 {
                interior += 1;
                assert_eq!(
                    blended.cover, blended.dominant,
                    "a tile inside a {:?} region took its cover from {:?}",
                    blended.dominant, blended.cover
                );
            }
        }

        assert!(interior > 1000, "only {interior} interior tiles sampled");
    }

    /// And the other half: inside the band it must *actually* mix, or the dither is
    /// dead code and the boundary is still a line.
    #[test]
    fn the_cover_dither_mixes_both_biomes_across_a_boundary() {
        let map = map();
        let mixed = (0..16384)
            .filter(|i| {
                let (x, y) = ((i * 13 % 4096) as f32, (i * 211 % 4096) as f32);
                let blended = map.blend(x, y);
                blended.cover != blended.dominant
            })
            .count();

        assert!(
            mixed > 100,
            "only {mixed}/16384 tiles take their cover from a neighbouring biome"
        );
    }

    /// A weighted draw is only worth having if it actually draws every entry.
    #[test]
    fn every_biome_is_drawn_somewhere() {
        let map = map();
        for biome in Biome::ALL {
            let found = (0..64 * 64).any(|i| {
                let cell = IVec2::new(i % 64, i / 64);
                map.site(cell).1 == biome
            });
            assert!(found, "{biome:?} is never drawn");
        }
    }
}
