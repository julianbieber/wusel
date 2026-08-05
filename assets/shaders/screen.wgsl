// The one post-process pass over the world: the height ramp, the sun, and the
// weather, composited in the order of the lines below and in no other.
//
// This was two passes ping-ponging the same view target, which meant the composite
// order was an ordering between systems and the weather had to be *told* how much
// light there was, because the tint had already applied it. Here the light is a
// local, and "the clouds are lit by the same sun as the ground" is true by reading
// rather than by plumbing.
//
// There is no noise in here on purpose. The heightmap is a texture the terrain
// generator filled in as it went; both cloud fields arrive as textures baked by
// src/gameplay/weather.rs from src/gameplay/noise.rs. Evaluating an fbm per fragment
// instead would cost the whole frame budget at 4K, and would put a second noise
// implementation in a crate that keeps exactly one.
//
// `ScreenUniform` below mirrors the Rust struct in src/gameplay/screen.rs field for
// field. They are edited together: a mismatch is a shader compile failure the first
// time you enter gameplay, not a build error.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

struct ScreenUniform {
    sun_direct: vec4<f32>,
    sun_sky: vec4<f32>,
    view_centre_tiles: vec2<f32>,
    view_half_extent_tiles: vec2<f32>,
    world_tiles: vec2<f32>,
    sun_bearing: vec2<f32>,
    coarse_offset: vec2<f32>,
    fine_offset: vec2<f32>,
    shadow_offset_tiles: vec2<f32>,
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
    shape_period_tiles: f32,
    cloud_fine_scale: f32,
    cloud_coarse_weight: f32,
    cloud_cut: f32,
    cloud_softness: f32,
    cloud_brightness: f32,
    cloud_opacity: f32,
    shadow_strength: f32,
    rain_cut: f32,
    rain_softness: f32,
    rain_strength: f32,
    streak_phase: f32,
}

@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@group(0) @binding(2) var height_texture: texture_2d<f32>;
@group(0) @binding(3) var probability_texture: texture_2d<f32>;
@group(0) @binding(4) var probability_sampler: sampler;
@group(0) @binding(5) var shape_texture: texture_2d<f32>;
@group(0) @binding(6) var shape_sampler: sampler;
@group(0) @binding(7) var<uniform> screen: ScreenUniform;

/// How a colour is weighed into one number. Matches `LUMINANCE` in gameplay/sun.rs,
/// which is where the same weighting decides what the simulation reads.
const LUMINANCE = vec3(0.2126, 0.7152, 0.0722);

/// Where a screen uv falls in global tile space. The y flip is load-bearing: uv runs
/// down the screen and the world runs up, and without it the sky is mirrored — which
/// is invisible until you pan.
fn tile_position(uv: vec2<f32>) -> vec2<f32> {
    let offset = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0) * screen.view_half_extent_tiles;
    return screen.view_centre_tiles + offset;
}

fn inside_world(tile: vec2<f32>) -> bool {
    return tile.x >= 0.0 && tile.y >= 0.0
        && tile.x < screen.world_tiles.x && tile.y < screen.world_tiles.y;
}

/// How much of the sun the tile `distance` away along the bearing hides.
///
/// The ray leaves this tile's own surface and climbs `distance * sun_ray_slope`
/// tiles on its way, which is that over `relief_tiles` in the units the heightmap
/// stores. Softened rather than cut because the sun sweeps a ridge *past* a sample
/// distance, and a hard test flickers when it does.
fn occlusion(tile: vec2<f32>, height: f32, distance: f32) -> f32 {
    let at = floor(tile + screen.sun_bearing * distance);
    if !inside_world(at) {
        // Off the edge of the world there is no height to read, so nothing there
        // casts anything.
        return 0.0;
    }

    let ray = height + distance * screen.sun_ray_slope / screen.relief_tiles;
    let standing = textureLoad(height_texture, vec2<i32>(at), 0).r - ray;
    return smoothstep(0.0, screen.shadow_softness, standing);
}

/// The raw cloud field over a tile: how likely cloud is there, times how much shape
/// there is. `cloud_density` steps this into cloud; the rain reads it directly,
/// because the density saturates and the rain wants to know how *thick* the cloud is.
fn cloud_field(tile: vec2<f32>) -> f32 {
    // Clamped at the world's edge, so looking off the map reads the map's border
    // rather than wrapping the far side of the world into shot.
    let probability = textureSample(
        probability_texture,
        probability_sampler,
        tile / screen.world_tiles,
    ).r;

    // Two layers of one tiling map: the second is finer and drifts slower, so the
    // pair shears instead of sliding as a sheet, and the repeat stops being a
    // pattern you can see.
    let period = tile / screen.shape_period_tiles;
    let coarse = textureSample(shape_texture, shape_sampler, period + screen.coarse_offset).r;
    let fine = textureSample(
        shape_texture,
        shape_sampler,
        period * screen.cloud_fine_scale + screen.fine_offset,
    ).r;
    let shape = coarse * screen.cloud_coarse_weight
        + fine * (1.0 - screen.cloud_coarse_weight);

    return probability * shape;
}

/// How much cloud that field amounts to. Transcribes `cloud_density` in weather.rs,
/// which is where the same arithmetic is measured without a GPU.
fn cloud_density(field: f32) -> f32 {
    return smoothstep(
        screen.cloud_cut - screen.cloud_softness,
        screen.cloud_cut + screen.cloud_softness,
        field,
    );
}

/// How hard it is raining. Cut on the raw field, so only a thick cloud rains, and
/// multiplied by the density, so clear sky cannot rain whatever the knobs say.
fn rain_amount(field: f32, density: f32) -> f32 {
    return smoothstep(screen.rain_cut, screen.rain_cut + screen.rain_softness, field)
        * density;
}

/// Rain streaks, in screen space and in pixels. World-space streaks would be
/// sixteen times denser at one end of the zoom range than the other, and would
/// alias into moire at the near end.
fn rain_streaks(position: vec2<f32>, phase: f32) -> f32 {
    // One unit is one streak period, so wrapping the phase at 1 is continuous.
    let along = (position.y * 0.8 + position.x * 0.25) / 26.0 + phase;
    let band = fract(along);
    return smoothstep(0.55, 1.0, band) * smoothstep(0.02, 0.35, fract(position.x / 7.0));
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let scene = textureSample(scene_texture, scene_sampler, in.uv);
    let at = tile_position(in.uv);
    let tile = floor(at);

    // -- the ground: its own relief, then the sun on it ----------------------

    var colour = scene.rgb;
    if inside_world(tile) {
        // `textureLoad` rather than `textureSample`: it takes the tile's own integer
        // coordinate, so a brightness step falls on a tile boundary by construction
        // rather than by remembering to ask for nearest filtering.
        let height = textureLoad(height_texture, vec2<i32>(tile), 0).r;

        // Water is a separate question, and so is a chunk that has not been generated
        // — a texel nobody has written reads zero, which is below any water line, so
        // the absence of a heightmap leaves the ramp flat rather than wrong. The sun
        // still applies: the ramp is about the ground, the light is about the sky.
        var relief = 1.0;
        if height > screen.water_line {
            // One ramp over the whole height range rather than one per band: a
            // per-band ramp reverses at every band edge, which would draw a contour
            // line along every coastline, treeline and snow line.
            let ramp = clamp(
                (height - screen.tint_low) / (screen.tint_high - screen.tint_low),
                0.0,
                1.0,
            );
            relief = 1.0 + screen.strength * (ramp * 2.0 - 1.0);
        }

        // How much of the beam reaches this tile. Three samples are the whole ray
        // march, and the strongest occluder wins — a nearer ridge does not add to a
        // farther one, it is simply the same sun already hidden.
        var lit = 1.0;
        if screen.sun_ray_slope > 0.0 {
            var hidden = occlusion(tile, height, screen.shadow_near_tiles);
            hidden = max(hidden, occlusion(tile, height, screen.shadow_mid_tiles));
            hidden = max(hidden, occlusion(tile, height, screen.shadow_far_tiles));
            lit = 1.0 - hidden;
        }
        // else: the sun is down, there is no beam to block, and the test would cost
        // the same as it does at noon to return nothing.

        // A shadow is the loss of the beam, not a darkening — so a shadowed tile
        // keeps the sky and a lit one gets both. There is no shadow-strength knob to
        // fall out of step with the light.
        let light = screen.sun_sky.rgb + screen.sun_direct.rgb * lit;
        colour = colour * relief * light;
    }
    // else: off the edge of the world there is no height to read, and nothing drawn
    // to shade or to light either — but the weather below still falls on it, exactly
    // as it did when these were two passes.

    // -- the sky: shadow, rain, then the cloud itself -------------------------

    let cloud = cloud_density(cloud_field(at));
    // The shadow is the same field one constant offset away. Sampling the cloud at
    // the tile the offset points to is what ties each shadow to exactly one cloud —
    // within an offset of the border, that cloud is off screen.
    let shadow_field = cloud_field(at + screen.shadow_offset_tiles);
    let shadow = cloud_density(shadow_field);

    colour = colour * (1.0 - screen.shadow_strength * shadow);

    // Rain falls where the shadow is, not where the cloud is: under the cloud, which
    // is also where it is not immediately painted over by it.
    let rain = rain_amount(shadow_field, shadow);
    if rain > 0.0 {
        let grey = vec3(dot(colour, vec3(0.299, 0.587, 0.114)));
        let wet = mix(colour, grey * 0.75, 0.6);
        colour = mix(colour, wet, rain * screen.rain_strength);
        colour += vec3(rain_streaks(in.position.xy, screen.streak_phase))
            * rain * screen.rain_strength * 0.09;
    }

    // The cloud itself, last, so it sits over the world and over its own rain — but
    // never opaquely: the point is terrain seen through weather.
    //
    // Lit by the same sun as the ground under it, read straight off the uniform
    // rather than handed over from another pass — which is what a midnight cloud
    // being a dark shape rather than a white one now rests on.
    let light_level = dot(screen.sun_sky.rgb + screen.sun_direct.rgb, LUMINANCE);
    let cloud_colour = screen.cloud_brightness * light_level;
    colour = mix(colour, vec3(cloud_colour), cloud * screen.cloud_opacity);

    return vec4(colour, scene.a);
}
