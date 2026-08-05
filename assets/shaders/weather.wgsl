// The weather overlay: cloud, shadow and rain composited over the rendered world.
//
// There is no noise in here on purpose. Both fields arrive as textures baked by
// src/gameplay/weather.rs from src/gameplay/noise.rs — the probability map covering
// the whole world once, the shape map holding one tiling period that this scrolls.
// Evaluating an fbm per fragment instead would cost the whole frame budget at 4K,
// and would put a second noise implementation in a crate that keeps exactly one.
//
// `WeatherUniform` below mirrors the Rust struct field for field. They are edited
// together: a mismatch is a shader compile failure at runtime, not a build error.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

struct WeatherUniform {
    view_centre_tiles: vec2<f32>,
    view_half_extent_tiles: vec2<f32>,
    coarse_offset: vec2<f32>,
    fine_offset: vec2<f32>,
    shadow_offset_tiles: vec2<f32>,
    world_tiles: vec2<f32>,
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
    light_level: f32,
}

@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@group(0) @binding(2) var probability_texture: texture_2d<f32>;
@group(0) @binding(3) var probability_sampler: sampler;
@group(0) @binding(4) var shape_texture: texture_2d<f32>;
@group(0) @binding(5) var shape_sampler: sampler;
@group(0) @binding(6) var<uniform> weather: WeatherUniform;

/// Screen to global tile space. The uv is y-down and the world is y-up, hence the
/// flip: without it the sky is mirrored, which is invisible until you pan.
fn tile_position(uv: vec2<f32>) -> vec2<f32> {
    let offset = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0) * weather.view_half_extent_tiles;
    return weather.view_centre_tiles + offset;
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
        tile / weather.world_tiles,
    ).r;

    // Two layers of one tiling map: the second is finer and drifts slower, so the
    // pair shears instead of sliding as a sheet, and the repeat stops being a
    // pattern you can see.
    let period = tile / weather.shape_period_tiles;
    let coarse = textureSample(shape_texture, shape_sampler, period + weather.coarse_offset).r;
    let fine = textureSample(
        shape_texture,
        shape_sampler,
        period * weather.cloud_fine_scale + weather.fine_offset,
    ).r;
    let shape = coarse * weather.cloud_coarse_weight
        + fine * (1.0 - weather.cloud_coarse_weight);

    return probability * shape;
}

/// How much cloud that field amounts to. Transcribes `cloud_density` in weather.rs,
/// which is where the same arithmetic is measured without a GPU.
fn cloud_density(field: f32) -> f32 {
    return smoothstep(
        weather.cloud_cut - weather.cloud_softness,
        weather.cloud_cut + weather.cloud_softness,
        field,
    );
}

/// How hard it is raining. Cut on the raw field, so only a thick cloud rains, and
/// multiplied by the density, so clear sky cannot rain whatever the knobs say.
fn rain_amount(field: f32, density: f32) -> f32 {
    return smoothstep(weather.rain_cut, weather.rain_cut + weather.rain_softness, field)
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
    let tile = tile_position(in.uv);

    let cloud = cloud_density(cloud_field(tile));
    // The shadow is the same field one constant offset away. Sampling the cloud at
    // the tile the offset points to is what ties each shadow to exactly one cloud —
    // within an offset of the border, that cloud is off screen.
    let shadow_field = cloud_field(tile + weather.shadow_offset_tiles);
    let shadow = cloud_density(shadow_field);

    var colour = scene.rgb * (1.0 - weather.shadow_strength * shadow);

    // Rain falls where the shadow is, not where the cloud is: under the cloud, which
    // is also where it is not immediately painted over by it.
    let rain = rain_amount(shadow_field, shadow);
    if rain > 0.0 {
        let grey = vec3(dot(colour, vec3(0.299, 0.587, 0.114)));
        let wet = mix(colour, grey * 0.75, 0.6);
        colour = mix(colour, wet, rain * weather.rain_strength);
        colour += vec3(rain_streaks(in.position.xy, weather.streak_phase))
            * rain * weather.rain_strength * 0.09;
    }

    // The cloud itself, last, so it sits over the world and over its own rain — but
    // never opaquely: the point is terrain seen through weather.
    //
    // Lit by the same sun as the ground it is over — the tint pass ran first, so
    // without this a cloud at midnight would be a white shape over a dark world.
    let cloud_colour = weather.cloud_brightness * weather.light_level;
    colour = mix(colour, vec3(cloud_colour), cloud * weather.cloud_opacity);

    return vec4(colour, scene.a);
}
