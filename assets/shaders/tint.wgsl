// Shades the rendered world by the height of the tile under each fragment, and then
// lights it with the sun.
//
// The heightmap is a texture the terrain generator filled in as it went, one texel
// per tile, so this shader evaluates no noise: it reads one value and ramps it.
// That is what keeps exactly one noise implementation in the crate, and what makes
// this cheap enough to run over every fragment twice a frame alongside the weather.
//
// The sun is here rather than in a pass of its own because the shadow test reads the
// same heightmap, and this is the only pass that binds it. Where the sun *is* was
// decided on the CPU — see gameplay/sun.rs — so there is no astronomy in here
// either, only three more texture loads.
//
// `TerrainTintUniform` is this struct written twice — see gameplay/tint.rs. Field
// order is the binding layout: vectors first, scalars after, so the std140 padding
// is the same on both sides and WebGL2 agrees with the desktop.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

struct TerrainTintUniform {
    sun_direct: vec4<f32>,
    sun_sky: vec4<f32>,
    view_centre_tiles: vec2<f32>,
    view_half_extent_tiles: vec2<f32>,
    world_tiles: vec2<f32>,
    sun_bearing: vec2<f32>,
    water_line: f32,
    tint_low: f32,
    tint_high: f32,
    strength: f32,
    sun_ray_slope: f32,
    shadow_softness: f32,
    relief_tiles: f32,
    shadow_near_tiles: f32,
    shadow_mid_tiles: f32,
    shadow_far_tiles: f32,
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

fn inside_world(tile: vec2<f32>) -> bool {
    return tile.x >= 0.0 && tile.y >= 0.0
        && tile.x < tint.world_tiles.x && tile.y < tint.world_tiles.y;
}

/// How much of the sun the tile `distance` away along the bearing hides.
///
/// The ray leaves this tile's own surface and climbs `distance * sun_ray_slope`
/// tiles on its way, which is that over `relief_tiles` in the units the heightmap
/// stores. Softened rather than cut because the sun sweeps a ridge *past* a sample
/// distance, and a hard test flickers when it does.
fn occlusion(tile: vec2<f32>, height: f32, distance: f32) -> f32 {
    let at = floor(tile + tint.sun_bearing * distance);
    if !inside_world(at) {
        // Off the edge of the world there is no height to read, so nothing there
        // casts anything.
        return 0.0;
    }

    let ray = height + distance * tint.sun_ray_slope / tint.relief_tiles;
    let standing = textureLoad(height_texture, vec2<i32>(at), 0).r - ray;
    return smoothstep(0.0, tint.shadow_softness, standing);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let scene = textureSample(scene_texture, scene_sampler, in.uv);
    let tile = floor(tile_position(in.uv));

    // Off the edge of the world there is no height to read, and nothing drawn to
    // shade or to light either.
    if !inside_world(tile) {
        return scene;
    }

    // `textureLoad` rather than `textureSample`: it takes the tile's own integer
    // coordinate, so a brightness step falls on a tile boundary by construction
    // rather than by remembering to ask for nearest filtering.
    let height = textureLoad(height_texture, vec2<i32>(tile), 0).r;

    // Water is a separate question, and so is a chunk that has not been generated —
    // a texel nobody has written reads zero, which is below any water line, so the
    // absence of a heightmap leaves the ramp flat rather than wrong. The sun still
    // applies: the ramp is about the ground, the light is about the sky.
    var relief = 1.0;
    if height > tint.water_line {
        // One ramp over the whole height range rather than one per band: a per-band
        // ramp reverses at every band edge, which would draw a contour line along
        // every coastline, treeline and snow line.
        let ramp = clamp((height - tint.tint_low) / (tint.tint_high - tint.tint_low), 0.0, 1.0);
        relief = 1.0 + tint.strength * (ramp * 2.0 - 1.0);
    }

    // How much of the beam reaches this tile. Three samples are the whole ray march,
    // and the strongest occluder wins — a nearer ridge does not add to a farther one,
    // it is simply the same sun already hidden.
    var lit = 1.0;
    if tint.sun_ray_slope > 0.0 {
        var hidden = occlusion(tile, height, tint.shadow_near_tiles);
        hidden = max(hidden, occlusion(tile, height, tint.shadow_mid_tiles));
        hidden = max(hidden, occlusion(tile, height, tint.shadow_far_tiles));
        lit = 1.0 - hidden;
    }
    // else: the sun is down, there is no beam to block, and the test would cost the
    // same as it does at noon to return nothing.

    // A shadow is the loss of the beam, not a darkening — so a shadowed tile keeps
    // the sky and a lit one gets both. There is no shadow-strength knob to fall out
    // of step with the light.
    let light = tint.sun_sky.rgb + tint.sun_direct.rgb * lit;

    return vec4(scene.rgb * relief * light, scene.a);
}
