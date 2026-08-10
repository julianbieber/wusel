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
    // Where this biome sits before any layer displaces it.
    pub base_height: f32,
    /// How much of the fine relief layer to add, as a displacement about zero.
    pub relief: f32,
    /// How much of the ridged layer to add. One-sided, so it only builds up.
    pub ridge: f32,
    /// How much of the dune layer to add. Non-zero only for `Desert`, and the
    /// epsilon skip on it is what keeps the aeolian sample off the other five
    /// biomes' bill — the same trick `ridge` already earns.
    pub dune: f32,
    /// Shifts how much loose material this ground holds before slope strips any:
    /// positive for a `Wetland`'s silt, negative for a `Highland`'s bare rock.
    pub soil_bias: f32,
    pub vegetation_bias: f32,
    pub humidity_bias: f32,
    /// Degrees Celsius on top of what the lapse rate already takes off for height.
    /// It is therefore the part of a region's climate that is *not* derivable from
    /// how high it is: a desert is hot because the sky over it is clear, not because
    /// it is low.
    ///
    /// Nothing in generation reads it — `classify` may not, on the terms
    /// [`crate::gameplay::terrain::TerrainSampler::temperature`] states — so the
    /// coverage figures are unmoved by this column existing.
    pub temperature_bias: f32,
    /// How far above the water line the sand band reaches, in elevation units.
    /// Zero means no beach: a `Highland` coast drops into the sea as rock.
    pub beach_width: f32,
}

/// The three kinds a biome's lowland band chooses between, in ascending
/// vegetation order.
///
/// A triple rather than the pair it replaces, and that is half of what gh-14 is:
/// a binary driven by one 11-tile field is a dither, not a landscape. With three
/// steps a `vegetation_bias` shifts a region along the ladder instead of
/// saturating it against a single cut — the measured symptom was `Desert`, whose
/// bias put 92.5% of its tiles on one side of `forest_threshold` and so made it
/// 80% Sand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoverTriple {
    pub bare: TerrainKind,
    pub mid: TerrainKind,
    pub lush: TerrainKind,
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
                dune: 0.0,
                soil_bias: 0.0,
                vegetation_bias: -0.12,
                humidity_bias: 0.04,
                // Maritime: the sea is a heat store, so a coast is mild. Its *swing*
                // is damped too, and that falls out of the humidity bias rather
                // than being said twice.
                temperature_bias: 1.0,
                beach_width: 0.06,
            },
            Biome::Plains => HeightRecipe {
                base_height: 0.52,
                relief: 0.09,
                ridge: 0.0,
                dune: 0.0,
                soil_bias: 0.05,
                vegetation_bias: -0.18,
                humidity_bias: 0.0,
                // The reference the other five are read against.
                temperature_bias: 0.0,
                beach_width: 0.04,
            },
            // The same triple as Plains, and the bias is the whole difference: a
            // wood with clearings against a field with copses.
            Biome::Forest => HeightRecipe {
                base_height: 0.55,
                relief: 0.13,
                ridge: 0.02,
                dune: 0.0,
                // Leaf litter and a closed canopy: forest soil is the deepest in
                // the table, which is what keeps a wood from going bald on every
                // slope the way a highland does.
                soil_bias: 0.04,
                vegetation_bias: -0.04,
                humidity_bias: 0.06,
                // Canopy shade and what the leaves transpire.
                temperature_bias: -1.0,
                beach_width: 0.03,
            },
            // The only recipe that leans on the ridged layer, and the reason it
            // exists: `ridge` is what turns a lump into a range with spurs.
            Biome::Highland => HeightRecipe {
                base_height: 0.70,
                relief: 0.11,
                ridge: 0.34,
                dune: 0.0,
                // Negative, so a highland is stripped to bedrock on a gentler slope
                // than anywhere else. This is what puts Rock on the shoulders of a
                // range rather than only above `scree_min`.
                soil_bias: -0.15,
                vegetation_bias: -0.09,
                humidity_bias: 0.03,
                // On top of the lapse rate, which has already taken a great deal off
                // for the height: thin exposed air loses what it gains. Small,
                // because the height is doing most of this work already.
                temperature_bias: -2.0,
                beach_width: 0.0,
            },
            // The one recipe that weighs the dune layer. The biases are what make
            // it a desert; the dry one is also why no spring rises here and no
            // cloud gathers.
            Biome::Desert => HeightRecipe {
                base_height: 0.51,
                relief: 0.11,
                ridge: 0.04,
                dune: 1.0,
                // Thin: what is not under a dune is deflated to hardpan, which is
                // what puts Gravel in the interdunes.
                soil_bias: 0.0,
                vegetation_bias: -0.24,
                humidity_bias: -0.26,
                // The biggest entry in the column, and it buys two things at once: a
                // desert is hot by day, and — because the same dry air is what the
                // diurnal amplitude is damped by — it is also the place that
                // freezes hardest at night. Neither is written down anywhere as a
                // rule about deserts.
                temperature_bias: 6.0,
                beach_width: 0.08,
            },
            // Flat and just above the water line, so the lowland band is nearly all
            // of it. No beach: a marsh meets open water as marsh.
            Biome::Wetland => HeightRecipe {
                base_height: 0.47,
                relief: 0.04,
                ridge: 0.0,
                dune: 0.0,
                // Silt, and the deepest soil there is — a wetland is where the rest
                // of the world's stripped material ends up.
                soil_bias: 0.0,
                vegetation_bias: -0.20,
                humidity_bias: 0.22,
                // Standing water evaporating cools it as much as being low warms it,
                // so the bias is nothing and the wet air alone makes it mild — a
                // marsh barely swings between noon and midnight.
                temperature_bias: 0.0,
                beach_width: 0.0,
            },
        }
    }

    /// What this biome's lowland band lays down, in ascending vegetation order.
    ///
    /// Not part of [`HeightRecipe`] because it cannot be blended — so at a boundary
    /// this is the one thing that changes all at once, and it is deliberately the
    /// thing elevation does *not* depend on. That is what makes a boundary read as a
    /// treeline rather than a wall.
    ///
    /// `Scrub` is the tile that made three steps possible: it is the missing rung
    /// between Sand and Grass *and* between Grass and Forest, so five of the six
    /// biomes can now spend a `vegetation_bias` on moving along the ladder instead
    /// of pinning themselves to one end of it.
    pub fn kinds(self) -> CoverTriple {
        match self {
            // Bars and islands: bare sand, scrub where it holds, grass where it is
            // wettest. Never Forest — an island in this world is not a wood.
            Biome::Ocean => CoverTriple {
                bare: TerrainKind::Sand,
                mid: TerrainKind::Scrub,
                lush: TerrainKind::Grass,
            },
            Biome::Plains => CoverTriple {
                bare: TerrainKind::Scrub,
                mid: TerrainKind::Grass,
                lush: TerrainKind::Forest,
            },
            // The same triple as Plains: the vegetation_bias is the whole
            // difference, a wood with clearings against a field with copses.
            Biome::Forest => CoverTriple {
                bare: TerrainKind::Scrub,
                mid: TerrainKind::Grass,
                lush: TerrainKind::Forest,
            },
            Biome::Highland => CoverTriple {
                bare: TerrainKind::Rock,
                mid: TerrainKind::Scrub,
                lush: TerrainKind::Forest,
            },
            // The whole ladder shifted one rung dry of everyone else's: a desert's
            // *lushest* lowland is scrub, and its bare end is the hardpan the wind
            // has deflated to.
            Biome::Desert => CoverTriple {
                bare: TerrainKind::Gravel,
                mid: TerrainKind::Sand,
                lush: TerrainKind::Scrub,
            },
            Biome::Wetland => CoverTriple {
                bare: TerrainKind::Marsh,
                mid: TerrainKind::Reed,
                lush: TerrainKind::Forest,
            },
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
pub const BIOME_TABLE: [(Biome, u32); 6] = [
    (Biome::Ocean, 6),
    (Biome::Plains, 4),
    (Biome::Forest, 4),
    (Biome::Highland, 3),
    (Biome::Desert, 2),
    (Biome::Wetland, 2),
];

/// Salts the region warp's two component fields, so `document.rs` can hand them to
/// `watershed` and get the outline this world has always had.
///
/// The lattice, the jitter, the blend band and the cover dither that used to sit here are
/// **gone**: they are `watershed::regions` now, and a copy kept beside it would be a
/// second answer to "which region is this tile in". What is left of this module is the
/// part a library must not carry — the [`Biome`] enum, the recipe table, and the kind
/// triples an enum cannot be averaged into.
pub const WARP_X_SALT: u32 = 0x7a1d_0b37;
pub const WARP_Y_SALT: u32 = 0x9c3f_1102;

/// How many octaves the warp fields get.
///
/// Three, and the count matters more than it looks. A warp bends an edge only where
/// it has content at a wavelength *shorter* than the edge; a single long octave
/// translates the whole boundary bodily and leaves it exactly as straight as it was.
/// That was the first version's mistake — two octaves at 1.5 cells meant the finest
/// warp detail had a 288-tile wavelength against ~200-tile boundary segments, so it
/// moved the edges without bending any of them.
pub const WARP_OCTAVES: u32 = 3;

/// The warp's base wavelength, in cells. Under one cell so that the finest octave
/// (an eighth of this) wiggles an edge at a scale you can see from the ground,
/// while still being long enough not to shred it.
pub const WARP_CELLS: f32 = 0.75;

#[cfg(test)]
mod tests {
    use super::*;

    use crate::gameplay::terrain::shared_test_sampler;

    // The machinery these used to drive — the jittered lattice, the banded weights, the
    // cover draw — is `watershed::regions` now and is guarded there. What is still
    // wusel's, and is what these ask, is whether the shipped defaults put a recognisable
    // landscape on top of it: a region with an interior, a boundary that does not step,
    // and a table every entry of which is actually drawn.
    //
    // They go through the sampler rather than a map of their own, which is the point of
    // the split: there is one answer to "which region is this" and this is how to ask it.

    /// Regions must have interiors, or "distinct biomes" is a lie: away from a
    /// boundary a tile carries one recipe exactly, not a mush of all six.
    #[test]
    fn a_region_has_an_interior_where_exactly_one_recipe_applies() {
        let sampler = shared_test_sampler();
        let unblended = (0..4096)
            .filter(|i| {
                let (x, y) = ((i * 7 % 4096) as f32, (i * 97 % 4096) as f32);
                let sample = sampler.sample(x, y);
                let pure = sample.dominant.recipe();
                (sample.recipe.base_height - pure.base_height).abs() < 1e-3
            })
            .count();

        assert!(
            unblended > 4096 / 2,
            "only {unblended}/4096 sampled tiles carry an unblended recipe"
        );
    }

    /// The load-bearing continuity claim. The dominant biome may flip from one tile
    /// to the next — that is what makes a treeline — but the *elevation inputs* may
    /// not, or the flip would be a cliff.
    #[test]
    fn a_blended_recipe_is_continuous_across_a_boundary() {
        let sampler = shared_test_sampler();
        let mut crossings = 0;
        let mut worst = 0.0f32;

        for i in 0..8192 {
            let (x, y) = ((i * 13 % 4090) as f32, (i * 61 % 4090) as f32);
            let here = sampler.sample(x, y);
            let next = sampler.sample(x + 1.0, y);
            if here.dominant == next.dominant {
                continue;
            }
            crossings += 1;
            worst = worst.max((here.recipe.base_height - next.recipe.base_height).abs());
        }

        assert!(
            crossings > 32,
            "only {crossings} boundary crossings sampled"
        );
        assert!(
            worst < 0.05,
            "base_height jumps by {worst} across a region boundary"
        );
    }

    /// Every row of the table has to be drawn somewhere, or it is a row nobody can see.
    #[test]
    fn every_biome_is_drawn_somewhere() {
        let sampler = shared_test_sampler();
        for (biome, _) in BIOME_TABLE {
            let found = (0..4096).any(|i| {
                let (x, y) = ((i * 31 % 4096) as f32, (i * 71 % 4096) as f32);
                sampler.sample(x, y).dominant == biome
            });
            assert!(found, "{biome:?} is in the table but nowhere in the world");
        }
    }
}
