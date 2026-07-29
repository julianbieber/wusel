//! Hand-rolled gradient noise. There is deliberately no noise crate here: the
//! world is a pure function of tile position, and owning the hash means that
//! stays true across platforms and dependency bumps.

use bevy::prelude::*;

const NOISE_OCTAVES: u32 = 5;
const NOISE_PERSISTENCE: f32 = 0.5;
const NOISE_LACUNARITY: f32 = 2.0;
/// Normalized fbm only spans about [0.35, 0.65] in practice — the octaves rarely
/// align and gradient noise peaks well below 1. Stretching it around the midpoint
/// makes the terrain thresholds mean what they say on a [0, 1] scale.
const NOISE_GAIN: f32 = 2.6;

/// One fbm field with its own domain offset, so two fields sampled at the same
/// position are independent rather than two views of the same landscape.
pub struct NoiseField {
    offset: Vec2,
    scale: f32,
}

impl NoiseField {
    pub fn new(seed: u32, salt: u32, scale: f32) -> Self {
        let h = hash2(seed as i32, salt as i32);
        // Kept well under f32's precision cliff: fbm scales the domain up by the
        // lacunarity of the last octave, so a huge offset would quantize it.
        Self {
            offset: Vec2::new((h & 0xffff) as f32 / 64.0, (h >> 16) as f32 / 64.0),
            scale,
        }
    }

    /// Sample the field at a global tile position, remapped to [0, 1].
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let n = fbm(
            x * self.scale + self.offset.x,
            y * self.scale + self.offset.y,
            NOISE_OCTAVES,
            NOISE_PERSISTENCE,
            NOISE_LACUNARITY,
        );
        (0.5 + n * NOISE_GAIN * 0.5).clamp(0.0, 1.0)
    }
}

/// One fbm field that repeats exactly every `period` noise units in both axes.
///
/// The seam is the point: a field that tiles can be baked into a small texture and
/// scrolled forever, which is how the weather overlay animates without evaluating
/// any noise per fragment. Nothing in the terrain wants this — a world that repeats
/// every few hundred tiles would be visible from the ground.
pub struct TilingNoiseField {
    offset: Vec2,
    period: u32,
    octaves: u32,
}

impl TilingNoiseField {
    /// `period` must be a power of two: every octave wraps at `period * frequency`,
    /// and with a lacunarity of 2 that is only an integer if `period` is one.
    ///
    /// `octaves` is a parameter here rather than the module's constant because the
    /// field is baked into a texture: octaves finer than a couple of texels cannot
    /// survive the sampling, and asking for them only buys aliasing.
    pub fn new(seed: u32, salt: u32, period: u32, octaves: u32) -> Self {
        debug_assert!(
            period.is_power_of_two(),
            "a tiling period must be a power of two"
        );
        let h = hash2(seed as i32, salt as i32);
        Self {
            offset: Vec2::new((h & 0xffff) as f32 / 64.0, (h >> 16) as f32 / 64.0),
            period,
            octaves,
        }
    }

    /// Sample the field, remapped to [0, 1]. `u` and `v` are in noise units, of
    /// which the field holds `period` before it repeats.
    pub fn sample(&self, u: f32, v: f32) -> f32 {
        let n = tiling_fbm(
            u + self.offset.x,
            v + self.offset.y,
            self.period,
            self.octaves,
            NOISE_PERSISTENCE,
            NOISE_LACUNARITY,
        );
        (0.5 + n * NOISE_GAIN * 0.5).clamp(0.0, 1.0)
    }
}

/// Hash function to generate pseudo-random gradients from integer coordinates.
/// No external crates — uses a simple bit-mixing hash.
pub fn hash2(x: i32, y: i32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x27d4eb2d);
    h ^= (y as u32).wrapping_mul(0x165667b1);
    h ^= h >> 15;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    h
}

/// Returns a pseudo-random unit gradient vector for a lattice point.
fn gradient(ix: i32, iy: i32) -> (f32, f32) {
    let h = hash2(ix, iy);
    // Map hash to an angle in [0, 2*pi)
    let angle = (h as f32 / u32::MAX as f32) * std::f32::consts::TAU;
    (angle.cos(), angle.sin())
}

/// Smoothstep-style fade curve (6t^5 - 15t^4 + 10t^3), as used in Perlin noise.
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// 2D gradient (Perlin-style) noise, returns values roughly in [-1, 1].
pub fn gradient_noise_2d(x: f32, y: f32) -> f32 {
    gradient_noise_with(x, y, gradient)
}

/// The same lattice, with the corner lookups wrapped, so the noise repeats every
/// `period` units. Generic over the lookup so the untiled path above monomorphizes
/// to exactly what it was before — this is the only interpolation in the crate.
fn tiling_gradient_noise_2d(x: f32, y: f32, period: i32) -> f32 {
    gradient_noise_with(x, y, |ix, iy| {
        gradient(ix.rem_euclid(period), iy.rem_euclid(period))
    })
}

fn gradient_noise_with(x: f32, y: f32, grad: impl Fn(i32, i32) -> (f32, f32)) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let sx = x - x0 as f32;
    let sy = y - y0 as f32;

    // Dot product of gradient and distance vector at each corner.
    let dot_grad = |ix: i32, iy: i32, dx: f32, dy: f32| -> f32 {
        let (gx, gy) = grad(ix, iy);
        gx * dx + gy * dy
    };

    let n00 = dot_grad(x0, y0, sx, sy);
    let n10 = dot_grad(x1, y0, sx - 1.0, sy);
    let n01 = dot_grad(x0, y1, sx, sy - 1.0);
    let n11 = dot_grad(x1, y1, sx - 1.0, sy - 1.0);

    let u = fade(sx);
    let v = fade(sy);

    let nx0 = lerp(n00, n10, u);
    let nx1 = lerp(n01, n11, u);

    lerp(nx0, nx1, v)
}

/// Fractal Brownian Motion: sums multiple octaves of gradient noise
/// with increasing frequency and decreasing amplitude, then normalizes.
pub fn fbm(
    x: f32,
    y: f32,
    octaves: u32,
    persistence: f32, // amplitude multiplier per octave, e.g. 0.5
    lacunarity: f32,  // frequency multiplier per octave, e.g. 2.0
) -> f32 {
    let mut total = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amplitude = 0.0;

    for _ in 0..octaves {
        total += gradient_noise_2d(x * frequency, y * frequency) * amplitude;
        max_amplitude += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    // Normalize so output stays roughly in [-1, 1] regardless of octave count.
    total / max_amplitude
}

/// [`fbm`], with every octave's lattice wrapped so the sum repeats every `period`
/// units. Octave `n` runs at `period * lacunarity^n` lattice cells, which is why the
/// period has to be a power of two.
pub fn tiling_fbm(
    x: f32,
    y: f32,
    period: u32,
    octaves: u32,
    persistence: f32,
    lacunarity: f32,
) -> f32 {
    let mut total = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amplitude = 0.0;

    for _ in 0..octaves {
        let lattice_period = (period as f32 * frequency) as i32;
        total += tiling_gradient_noise_2d(x * frequency, y * frequency, lattice_period) * amplitude;
        max_amplitude += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    total / max_amplitude
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The seam is the whole reason the field exists: the weather overlay bakes one
    /// period into a texture and scrolls it forever, so a discontinuity at the wrap
    /// would be a line marching across the sky.
    #[test]
    fn a_tiling_field_matches_itself_across_the_seam() {
        let field = TilingNoiseField::new(0x5eed, 0xc10d, 8, 4);
        let period = 8.0;

        for i in 0..64 {
            let t = i as f32 / 64.0 * period;
            for (u, v) in [(t, 1.7), (1.7, t), (t, t)] {
                assert!(
                    (field.sample(u, v) - field.sample(u + period, v + period)).abs() < 1e-5,
                    "the field does not repeat at ({u}, {v})"
                );
            }
        }
    }

    /// Wrapping the lattice must not flatten the field into a constant — a tiling
    /// field that is all one value would also "match across the seam".
    #[test]
    fn a_tiling_field_still_varies_across_its_period() {
        let field = TilingNoiseField::new(0x5eed, 0xc10d, 8, 4);
        let samples: Vec<f32> = (0..64)
            .map(|i| field.sample(i as f32 / 8.0, 3.25))
            .collect();
        let min = samples.iter().copied().fold(f32::MAX, f32::min);
        let max = samples.iter().copied().fold(f32::MIN, f32::max);
        assert!(max - min > 0.2, "a tiling field spanning only {min}..{max}");
    }
}
