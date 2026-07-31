// Shades the rendered world by the height of the tile under each fragment.
//
// The heightmap is a texture the terrain generator filled in as it went, one texel
// per tile, so this shader evaluates no noise: it reads one value and ramps it.
// That is what keeps exactly one noise implementation in the crate, and what makes
// this cheap enough to run over every fragment twice a frame alongside the weather.
//
// `TerrainTintUniform` is this struct written twice — see gameplay/tint.rs. Field
// order is the binding layout: vectors first, scalars after, so the std140 padding
// is the same on both sides and WebGL2 agrees with the desktop.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

struct TerrainTintUniform {
    view_centre_tiles: vec2<f32>,
    view_half_extent_tiles: vec2<f32>,
    world_tiles: vec2<f32>,
    water_line: f32,
    tint_low: f32,
    tint_high: f32,
    strength: f32,
}

@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@group(0) @binding(2) var height_texture: texture_2d<f32>;
@group(0) @binding(3) var<uniform> tint: TerrainTintUniform;

/// Where a screen uv falls in global tile space. The y flip is load-bearing: uv
/// runs down the screen and the world runs up.
fn tile_position(uv: vec2<f32>) -> vec2<f32> {
    let offset = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0) * tint.view_half_extent_tiles;
    return tint.view_centre_tiles + offset;
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let scene = textureSample(scene_texture, scene_sampler, in.uv);
    let tile = floor(tile_position(in.uv));

    // Off the edge of the world there is no height to read, and nothing drawn to
    // shade either.
    if tile.x < 0.0 || tile.y < 0.0 || tile.x >= tint.world_tiles.x || tile.y >= tint.world_tiles.y {
        return scene;
    }

    // `textureLoad` rather than `textureSample`: it takes the tile's own integer
    // coordinate, so a brightness step falls on a tile boundary by construction
    // rather than by remembering to ask for nearest filtering.
    let height = textureLoad(height_texture, vec2<i32>(tile), 0).r;

    // Water is a separate question, and so is a chunk that has not been generated —
    // a texel nobody has written reads zero, which is below any water line, so the
    // absence of a heightmap is a clear pass-through rather than a wrong answer.
    if height <= tint.water_line {
        return scene;
    }

    // One ramp over the whole height range rather than one per band: a per-band
    // ramp reverses at every band edge, which would draw a contour line along every
    // coastline, treeline and snow line.
    let ramp = clamp((height - tint.tint_low) / (tint.tint_high - tint.tint_low), 0.0, 1.0);
    return vec4(scene.rgb * (1.0 + tint.strength * (ramp * 2.0 - 1.0)), scene.a);
}
