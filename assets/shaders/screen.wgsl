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
    dither_period_tiles: f32,
    wet_darkening: f32,
    wet_desaturation: f32,
    snow_lightening: f32,
    snow_dither_softness: f32,
    temperature_swing: f32,
    temperature_seasonal_celsius: f32,
    freezing_celsius: f32,
    freezing_softness_celsius: f32,
    climate_min_celsius: f32,
    climate_span_celsius: f32,
    climate_amplitude_span_celsius: f32,
    overlay_field: f32,
    overlay_low: f32,
    overlay_mid: f32,
    overlay_high: f32,
    overlay_diverging: f32,
    overlay_opacity: f32,
    overlay_seam: f32,
}

@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@group(0) @binding(2) var height_texture: texture_2d<f32>;
@group(0) @binding(3) var probability_texture: texture_2d<f32>;
@group(0) @binding(4) var probability_sampler: sampler;
@group(0) @binding(5) var shape_texture: texture_2d<f32>;
@group(0) @binding(6) var shape_sampler: sampler;
@group(0) @binding(7) var cover_texture: texture_2d<f32>;
@group(0) @binding(8) var cover_sampler: sampler;
@group(0) @binding(9) var dither_texture: texture_2d<f32>;
@group(0) @binding(10) var dither_sampler: sampler;
@group(0) @binding(11) var climate_texture: texture_2d<f32>;
@group(0) @binding(12) var climate_sampler: sampler;
@group(0) @binding(13) var prospect_texture: texture_2d<f32>;
@group(0) @binding(14) var prospect_sampler: sampler;
@group(0) @binding(15) var<uniform> screen: ScreenUniform;

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

/// One lattice of falling dots, in screen space. `thin` is how much of the lattice
/// carries a flake at all — the rest is empty, which is what stops it reading as a
/// grid marching down the window.
///
/// The scattering is a golden-ratio walk over the integer cell rather than a hash: a
/// hash is a second noise implementation to keep, and the pattern only has to be
/// irregular over one screen. `0.618` and `0.381` are the two low-discrepancy
/// constants, so the sequence never lands back where it started.
fn flake_layer(p: vec2<f32>, phase: f32, thin: f32) -> f32 {
    let column = floor(p.x);
    // A column-dependent head start, so neighbouring columns are not falling in rank.
    let fall = p.y + phase + fract(column * 0.618) * 3.0;
    let across = fract(p.x) - 0.5;
    let along = fract(fall) - 0.5;
    let dot_shape = 1.0 - smoothstep(0.10, 0.26, length(vec2(across, along)));
    let carries = step(thin, fract(column * 0.618 + floor(fall) * 0.381));
    return dot_shape * carries;
}

/// Snow falling: two lattices at different scales and speeds, so the fall reads as
/// depth rather than as one sheet sliding past. In screen space, for the same reason
/// the rain streaks are.
fn snow_flakes(position: vec2<f32>, phase: f32) -> f32 {
    let near = flake_layer(position * 0.085, phase * 0.55, 0.62);
    let far = flake_layer(position * 0.045 + vec2(11.0, 4.0), phase * 0.30, 0.74);
    return clamp(near + far * 0.7, 0.0, 1.0);
}

/// The temperature over a tile: the baked climate normal there, plus this step's
/// swing scaled by that place's own amplitude, plus the season.
///
/// The same arithmetic `ClimateCell::temperature` does in gameplay/ground.rs, which
/// is why what falls out of the sky always agrees with what is lying on the ground —
/// they are one model read at two resolutions, not two models.
fn temperature_at(tile: vec2<f32>) -> f32 {
    let climate = textureSample(climate_texture, climate_sampler, tile / screen.world_tiles).rg;
    let normal = screen.climate_min_celsius + climate.r * screen.climate_span_celsius;
    let amplitude = climate.g * screen.climate_amplitude_span_celsius;
    return normal + amplitude * screen.temperature_swing + screen.temperature_seasonal_celsius;
}

/// How much of what is falling is frozen, on 0..1. Ramped rather than switched, so
/// sleet exists and no frame flips a whole region from rain to snow.
fn frozen(temperature: f32) -> f32 {
    return 1.0 - smoothstep(
        screen.freezing_celsius - screen.freezing_softness_celsius,
        screen.freezing_celsius + screen.freezing_softness_celsius,
        temperature,
    );
}

/// How much of a tile's snow is drawn, given its own value from the dither map.
///
/// Transcribed in `gameplay/tint.rs`'s tests, which is where the property that
/// matters is checked without a GPU — the same arrangement `sun.rs` has for the
/// occlusion test. The threshold is squeezed into `softness..1 - softness`
/// rather than being the dither value itself, and that is what makes the endpoints
/// exact: full coverage snows every tile, no coverage snows none.
fn snow_lying(coverage: f32, dither: f32, softness: f32) -> f32 {
    let threshold = softness + dither * (1.0 - 2.0 * softness);
    return smoothstep(threshold - softness, threshold + softness, coverage);
}

// -- the inspection overlay ---------------------------------------------------
//
// The fields the world is built out of, drawn as false colour. Every one of them is
// already bound because something else needed it: the heightmap for the ramp and the
// shadows, the climate for what falls as snow, the cover for what lies, and the
// cloud probability map — which *is* the humidity field. So this adds no binding.
//
// The colours are gameplay/inspect.rs's, and the two are edited together: that
// module builds the legend out of the same ramp, so a swatch and the map under it
// cannot disagree about what a value looks like.

const RAMP_SEQUENTIAL = array(
    vec3(0.804, 0.886, 0.984),
    vec3(0.224, 0.529, 0.898),
    vec3(0.051, 0.212, 0.420),
);
const RAMP_DIVERGING = array(
    vec3(0.051, 0.212, 0.420),
    vec3(0.941, 0.937, 0.925),
    vec3(0.439, 0.075, 0.071),
);

/// The raw value of whichever field is selected.
///
/// `textureSampleLevel` rather than `textureSample`: this runs inside a branch on a
/// uniform, and asking for an explicit level means no implicit derivatives and so no
/// uniformity question to get wrong. None of these maps has mips, so it reads the
/// same texel either way.
fn overlay_value(at: vec2<f32>, tile: vec2<f32>) -> f32 {
    let field = screen.overlay_field;
    if field < 1.5 {
        return textureLoad(height_texture, vec2<i32>(tile), 0).r;
    }
    if field < 2.5 {
        return temperature_at(at);
    }
    if field < 3.5 {
        // The cloud probability map, which is the humidity field unchanged — the same
        // answer the river springs read and the clouds gather on.
        return textureSampleLevel(
            probability_texture,
            probability_sampler,
            at / screen.world_tiles,
            0.0,
        ).r;
    }
    if field < 5.5 {
        // Read at the tile's centre, exactly as the composite reads it, so the
        // overlay shows the number the ground is actually drawn from.
        let cover = textureSampleLevel(
            cover_texture,
            cover_sampler,
            (tile + vec2(0.5)) / screen.world_tiles,
            0.0,
        ).rg;
        if field < 4.5 {
            return cover.r;
        }
        return cover.g;
    }
    if field < 6.5 {
        return cloud_density(cloud_field(at));
    }
    // The three prospectivity fields, whose scores are the map's first three channels
    // in the order gameplay/prospect.rs packs them. **Sampled**, like every other
    // field, because the score is a continuous quantity and the blocking is a true
    // property of the resolution it was taken at.
    let score = textureSampleLevel(
        prospect_texture,
        prospect_sampler,
        at / screen.world_tiles,
        0.0,
    );
    if field < 7.5 {
        return score.r;
    }
    if field < 8.5 {
        return score.g;
    }
    return score.b;
}

/// Whether a seam of the active overlay's own resource is drawn at this texel.
///
/// **`textureLoad`, at nearest, and never sampled.** The fourth channel is a
/// *category*: bilinear filtering across it interpolates between categories, so a
/// texel halfway between an iron seam and nothing reads as copper. This is the one
/// channel that must not be filtered — and the byte quantization is why the
/// comparison needs a tolerance at all rather than an equality.
fn seam_here(at: vec2<f32>) -> bool {
    if screen.overlay_seam <= 0.0 {
        return false;
    }
    let side = f32(textureDimensions(prospect_texture).x);
    let texel = vec2<i32>(clamp(at / screen.world_tiles * side, vec2(0.0), vec2(side - 1.0)));
    let mark = textureLoad(prospect_texture, texel, 0).a;
    return abs(mark - screen.overlay_seam) < 0.06;
}

/// A value onto the ramp's 0..1, with the range's midpoint pinned to the middle.
///
/// Transcribes `OverlayRange::normalize`. Piecewise because the two arms need not be
/// the same width — the temperature range runs -20 to 30 about a freezing point at 0
/// — and both still have to fill their half.
fn overlay_position(value: f32) -> f32 {
    var t: f32;
    if value < screen.overlay_mid {
        t = 0.5 * (value - screen.overlay_low)
            / max(screen.overlay_mid - screen.overlay_low, 1e-6);
    } else {
        t = 0.5 + 0.5 * (value - screen.overlay_mid)
            / max(screen.overlay_high - screen.overlay_mid, 1e-6);
    }
    return clamp(t, 0.0, 1.0);
}

/// Three stops, interpolated — transcribes `inspect::ramp`.
fn overlay_colour(t: f32) -> vec3<f32> {
    var stops = RAMP_SEQUENTIAL;
    if screen.overlay_diverging > 0.5 {
        stops = RAMP_DIVERGING;
    }
    if t < 0.5 {
        return mix(stops[0], stops[1], t * 2.0);
    }
    return mix(stops[1], stops[2], (t - 0.5) * 2.0);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let scene = textureSample(scene_texture, scene_sampler, in.uv);
    let at = tile_position(in.uv);
    let tile = floor(at);

    // The overlay short-circuits everything below, and that is the point of it: an
    // inspector dimmed by nightfall or hidden under a cloud is not an inspector. Off
    // the edge of the world there is no field to read, so the world's own border
    // stays visible.
    if screen.overlay_field > 0.5 && inside_world(tile) {
        // A seam draws at the ramp's warm pole, so a site reads as the strongest mark
        // on a map whose warm end already means "worth digging".
        var t = overlay_position(overlay_value(at, tile));
        if seam_here(at) {
            t = 1.0;
        }
        let field = overlay_colour(t);
        return vec4(mix(scene.rgb, field, screen.overlay_opacity), scene.a);
    }

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
        if height > screen.water_line {
            // One ramp over the whole height range rather than one per band: a
            // per-band ramp reverses at every band edge, which would draw a contour
            // line along every coastline, treeline and snow line.
            let ramp = clamp(
                (height - screen.tint_low) / (screen.tint_high - screen.tint_low),
                0.0,
                1.0,
            );
            let relief = 1.0 + screen.strength * (ramp * 2.0 - 1.0);
            colour = colour * relief;

            // What the weather has left here. Read at the tile's own *centre*, so a
            // whole tile shares one value however far the view is zoomed out — the
            // grid is coarse and smooth, and everything per-tile about the look comes
            // from the dither below rather than from the state.
            //
            // This whole block sits inside the water-line branch, which is what keeps
            // the state grid from ever having to learn where the lakes are: snow can
            // accumulate over one and simply is not drawn.
            let cover = textureSample(
                cover_texture,
                cover_sampler,
                (tile + vec2(0.5)) / screen.world_tiles,
            ).rg;

            // Wet ground goes darker and a little duller. A full desaturation would
            // read as fog rather than as rain.
            let grey = vec3(dot(colour, LUMINANCE));
            let soaked = mix(colour, grey, screen.wet_desaturation) * (1.0 - screen.wet_darkening);
            colour = mix(colour, soaked, cover.r);

            // And snow over the top of it, because snow lies *on* wet ground and
            // hides it. Thresholded per tile against a small tiling map, so it
            // arrives tile by tile and a melting field shrinks from its edges.
            let dither = textureSample(
                dither_texture,
                dither_sampler,
                tile / screen.dither_period_tiles,
            ).r;
            let lying = snow_lying(cover.g, dither, screen.snow_dither_softness);
            // Keeping the relief factor is what makes a snowed slope still read as a
            // slope — a flat white would erase the landscape it settled on.
            colour = mix(colour, vec3(relief * screen.snow_lightening), lying);
        }
        // else: water passes through the whole ground half untouched — the ramp does
        // not shade it, and there is no ground under it to wet or to cover. That one
        // branch is why the state grid never has to know where the lakes are.

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
        colour = colour * light;
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

    // Precipitation falls where the shadow is, not where the cloud is: under the
    // cloud, which is also where it is not immediately painted over by it.
    let falling = rain_amount(shadow_field, shadow);
    if falling > 0.0 {
        // How much of it is frozen, from the same climate map the ground's cover is
        // decided by — so what is coming down always agrees with what is lying.
        //
        // At `at`, not at the cloud: the offset above says which cloud is raining on
        // this fragment, and the rain lands *here*. Reading the temperature at the
        // cloud instead would decide rain against snow ten tiles away, which is
        // invisible in the middle of a shower and wrong exactly at a snow line —
        // where it would put falling snow over thawed ground and rain over a
        // snowfield.
        let snowing = frozen(temperature_at(at));

        // Only the liquid share greys the world under it. What *snow* does to the
        // ground is the cover above, which is a state that outlasts the cloud rather
        // than a look that goes with it.
        let rain = falling * (1.0 - snowing);
        let grey = vec3(dot(colour, vec3(0.299, 0.587, 0.114)));
        let wet = mix(colour, grey * 0.75, 0.6);
        colour = mix(colour, wet, rain * screen.rain_strength);

        // And the veil itself, ramped from streaks to flakes across the freezing
        // point. Flakes are drawn harder than streaks because a flake is an object
        // catching the light where a streak is a smear of one.
        let veil = mix(
            rain_streaks(in.position.xy, screen.streak_phase),
            snow_flakes(in.position.xy, screen.streak_phase),
            snowing,
        );
        colour += vec3(veil) * falling * screen.rain_strength * mix(0.09, 0.34, snowing);
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
