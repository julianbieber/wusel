// The city stats panel's background, and its capacity bar, drawn as planked wood.
//
// No noise field is evaluated here, and that is deliberate: the crate holds exactly one
// noise implementation and it lives on the CPU, in gameplay/noise.rs, because the
// terrain's determinism tests rest on there being one answer to "what does this field
// say here". A wood grain owes nothing to that field, so it is built out of trigonometry
// and the golden ratio instead — which also means there is no map to bake and nothing to
// load before the panel can be drawn.
//
// `WoodPanelMaterial` in gameplay/city_panel.rs is this struct written twice; the field
// order is the binding layout. See the note there before touching either.

#import bevy_ui::ui_vertex_output::UiVertexOutput

struct WoodPanelMaterial {
    grain_dark: vec4<f32>,
    grain_light: vec4<f32>,
    frame: vec4<f32>,
    fill: vec4<f32>,
    plank_px: vec2<f32>,
    ring_px: vec2<f32>,
    ring_warp_px: f32,
    fibre_strength: f32,
    bevel_px: f32,
    fill_fraction: f32,
    is_bar: u32,
}

@group(1) @binding(0)
var<uniform> material: WoodPanelMaterial;

// 1/phi. Successive multiples of it are spread about as evenly over 0..1 as a sequence
// can be, which is what gives each plank an offset and a shade unlike its neighbours'
// without a hash function — and so without a second definition of "random" in the crate.
const GOLDEN_FRACTION: f32 = 0.618033988;

@fragment
fn fragment(in: UiVertexOutput) -> @location(0) vec4<f32> {
    // `in.size` and `in.border_radius` are physical pixels, so every length in the
    // uniform is too. `in.border_widths` is the odd one out and is in UV — it is not
    // used here, and that is the trap this comment exists for.
    let px = in.uv * in.size;

    // --- planks -------------------------------------------------------------------
    let plank_index = floor(px.y / material.plank_px.y);
    let along = px.x + fract(plank_index * GOLDEN_FRACTION) * material.plank_px.x;
    let across = px.y - plank_index * material.plank_px.y;
    // A shade per plank, on about +-6%, so the courses read as separate boards rather
    // than as one sheet with lines ruled across it.
    let plank_shade = 1.0 + (fract(plank_index * GOLDEN_FRACTION * 2.0) - 0.5) * 0.12;

    // --- grain --------------------------------------------------------------------
    // Two wavelengths, and the shorter one is the point. A single sine displaces every
    // ring in a plank by the same amount and leaves the grain as ruled as it started;
    // it is the second, shorter term that bends a ring against its neighbour. The biome
    // warp learned the same lesson about edges.
    let warp = sin(along / material.ring_px.y) * material.ring_warp_px
        + sin(along / (material.ring_px.y * 0.33) + 1.7) * material.ring_warp_px * 0.4;
    let ring = fract((across + warp) / material.ring_px.x);
    // Shaped so the dark line is narrow and the pale wood between it is broad — an
    // unshaped sine gives equal bands, which reads as corduroy rather than as timber.
    let grain = pow(1.0 - abs(ring * 2.0 - 1.0), 4.0);

    // Fine fibre along the plank. Without it the wood between two rings is flat paint.
    let fibre = sin(across * 5.3 + along * 0.13) * sin(across * 11.7) * material.fibre_strength;

    let wood = clamp(grain + fibre, 0.0, 1.0);
    var colour = mix(material.grain_light.rgb, material.grain_dark.rgb, wood) * plank_shade;

    // --- the bar ------------------------------------------------------------------
    if material.is_bar != 0u {
        // The same wood, sunk: darkening the whole node is what reads as a groove cut
        // into the panel behind it, and the filled part is that groove lit rather than
        // a different material laid into it.
        colour = mix(colour, material.frame.rgb, 0.55);
        // A hard step of about a pixel. A soft edge would blur exactly the thing the
        // bar exists to make readable — how far along it is.
        let filled = 1.0 - smoothstep(-0.5, 0.5, px.x - material.fill_fraction * in.size.x);
        colour = mix(colour, material.fill.rgb * (0.85 + 0.3 * wood), filled);
    }

    // --- frame and corners --------------------------------------------------------
    let external_distance = sd_rounded_box((in.uv - 0.5) * in.size, in.size, in.border_radius);
    // Inside the bevel the wood darkens to the frame colour, which is what gives the
    // panel an edge without a border node — a `BorderColor` would not draw at all here,
    // since the material replaces the node's background rather than sitting under it.
    let bevel = 1.0 - smoothstep(-material.bevel_px, 0.0, external_distance);
    colour = mix(material.frame.rgb, colour, clamp(bevel, 0.0, 1.0));

    let alpha = smoothstep(0.5, -0.5, external_distance) * material.grain_light.a;
    return vec4<f32>(colour, alpha);
}

// Copied, not imported, and the reason is worth keeping: the module that defines this
// (`bevy_ui::ui_node`) also declares the view uniform at group 0 and a texture and
// sampler at group 1, which collide with this material's own group-1 binding. Importing
// it is a compile error rather than a convenience, which is why bevy_feathers'
// alpha_pattern.wgsl carries its own copy too.
//
// From: https://github.com/bevyengine/bevy/pull/8973
// The shortest distance from `point` to the boundary of the rounded box: negative
// inside, positive outside, zero on the boundary. `corner_radii` is ordered
// counter-clockwise from the top left.
fn sd_rounded_box(point: vec2<f32>, size: vec2<f32>, corner_radii: vec4<f32>) -> f32 {
    let rs = select(corner_radii.xy, corner_radii.wz, 0.0 < point.y);
    let radius = select(rs.x, rs.y, 0.0 < point.x);
    let corner_to_point = abs(point) - 0.5 * size;
    let q = corner_to_point + radius;
    let l = length(max(q, vec2(0.0)));
    let m = min(max(q.x, q.y), 0.0);
    return l + m - radius;
}
