//! The one post-process pass over the world, and the only thing that draws it.
//!
//! There used to be two — the terrain's tint and the weather — ping-ponging the same
//! `ViewTarget`, so the second read what the first wrote and the composite order was
//! an ordering between systems. That worked, and it leaked: the weather had to be
//! told the light level because the tint had already applied it, and every new effect
//! wanting a bit of both halves widened the seam again.
//!
//! Merged, the composite is one fragment function and the order is the order of its
//! lines:
//!
//! ```text
//!   scene -> height ramp -> ground cover -> sun light -> cloud shadow
//!         -> rain or snowfall -> cloud
//! ```
//!
//! Two things fall out of that ordering for nothing. The cover composites *before*
//! the light, so snow is lit by the same sun as the ground it lies on and shadowed by
//! the same ridge; and the clouds come after, so a shadow crossing a snowfield is a
//! later line rather than a coupling.
//!
//! **Ownership did not move, only the drawing did.** [`crate::gameplay::tint`] still
//! owns the ramp and the dither, [`crate::gameplay::weather`] the sky and its bakes,
//! [`crate::gameplay::sun`] the light, [`crate::gameplay::ground`] the wetness and
//! the snow; each writes its own slice of [`ScreenOverlay`] through a setter, so the
//! uniform's field order stays private in here beside the shader that reads it.
//!
//! What this module owns outright is everything the GPU needs: the heightmap texture,
//! the pipeline, the specializer, the two ping-pong bind groups and the pass.
//!
//! **The heightmap is not an `Image` asset.** Bevy re-uploads a whole `Image` on any
//! change, so a 16 MB map would cross to the GPU in full every time a chunk landed.
//! This owns a raw texture and writes one chunk's rect — 4 KB — as that chunk arrives.
//!
//! **Every map has a blank fallback**, so the pass always draws. A 1x1 zero texture
//! stands in for a heightmap whose session has not started and for a sky whose bake
//! has not landed: zero height is below any water line and zero cloud probability is
//! a clear sky, so an absent map is the *fallback* rather than a missing pass. That is
//! what lets one bind group layout carry maps that arrive at different times.

use bevy::{
    core_pipeline::{Core2d, Core2dSystems, FullscreenShader, tonemapping::tonemapping},
    ecs::query::QueryItem,
    prelude::*,
    render::{
        MainWorld, Render, RenderApp, RenderStartup, RenderSystems,
        camera::ExtractedCamera,
        extract_component::{
            ComponentUniforms, DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin,
            UniformComponentPlugin,
        },
        render_asset::RenderAssets,
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            CachedRenderPipelineId, Canonical, ColorTargetState, ColorWrites, Extent3d,
            FragmentState, Operations, Origin3d, PipelineCache, RenderPassColorAttachment,
            RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, Sampler,
            SamplerBindingType, SamplerDescriptor, ShaderStages, ShaderType, Specializer,
            SpecializerKey, TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect,
            TextureDataOrder, TextureDescriptor, TextureDimension, TextureFormat,
            TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor, TextureViewId,
            Variants,
            binding_types::{sampler, texture_2d, uniform_buffer},
        },
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        sync_component::SyncComponent,
        texture::GpuImage,
        view::{ExtractedView, ViewTarget},
    },
};

use crate::{
    camera::{WorldCamera, visible_half_extent},
    gameplay::{
        ground::{
            CLIMATE_FALLBACK_CELSIUS, CLIMATE_MAX_AMPLITUDE_CELSIUS, CLIMATE_MAX_CELSIUS,
            CLIMATE_MIN_CELSIUS, ClimateTexture, GroundConfig, GroundCoverTexture,
            TemperatureOffset,
        },
        sun::{PlanetConfig, Sun},
        terrain::TerrainConfig,
        tint::{GroundDither, TerrainTintConfig},
        weather::{WeatherConfig, WeatherMaps},
        world::{
            CHUNK_SIZE, ChunkHeights, HeightUploadQueue, TILE_DISPLAY_SIZE, WORLD_CHUNKS,
            WORLD_TILES, tile_position_at,
        },
    },
    screens::Screen,
};

const SCREEN_SHADER_PATH: &str = "shaders/screen.wgsl";

/// Everything the one pass reads, flattened into the layout the shader declares.
///
/// Field order **is** the wgsl binding layout — `assets/shaders/screen.wgsl` declares
/// the same struct, and a mismatch is a shader-compile failure when you enter
/// gameplay rather than a build error. Vectors before scalars, so the std140 padding
/// agrees on both sides and WebGL2 agrees with the desktop.
///
/// There is no `light_level` here, and its absence is the point of the merge: the
/// clouds are lit by the same `sun_sky + sun_direct` the ground is, computed once as
/// a local in the fragment function rather than passed between two passes.
#[derive(Component, Clone, Copy, Default, ShaderType)]
pub(super) struct ScreenUniform {
    /// The beam and the sky, from [`crate::gameplay::sun`]. `vec4` rather than
    /// `vec3` because std140 pads a `vec3` to sixteen bytes, and a padding
    /// disagreement between these two structs is a runtime shader failure rather
    /// than a build error. The fourth component is not read.
    sun_direct: Vec4,
    sun_sky: Vec4,
    view_centre_tiles: Vec2,
    view_half_extent_tiles: Vec2,
    world_tiles: Vec2,
    /// Which way the sun lies, on the ground. The occlusion test walks along it.
    sun_bearing: Vec2,
    coarse_offset: Vec2,
    fine_offset: Vec2,
    shadow_offset_tiles: Vec2,
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
    /// This step's temperature offset from the climate normal. Two scalars rather
    /// than a map, because when it is has nothing to do with where you are — all the
    /// place-dependence is already baked into the climate map.
    temperature_swing: f32,
    temperature_seasonal_celsius: f32,
    freezing_celsius: f32,
    freezing_softness_celsius: f32,
    /// How to read a temperature back out of the climate map's two bytes. Taken from
    /// the constants [`crate::gameplay::ground`] quantized it with, so the encoder
    /// and the decoder cannot drift.
    climate_min_celsius: f32,
    climate_span_celsius: f32,
    climate_amplitude_span_celsius: f32,
}

/// Lives on the one world camera while [`Screen::Gameplay`] is up, and is the only
/// overlay there is.
///
/// Four modules write into it and none of them knows the layout: the ramp is config
/// and is written once on entering, the sun and the sky move and are written every
/// frame. The two view fields are filled in at *extract* time from the camera itself,
/// which is why no ordering against the pan is needed anywhere — a frame of slip
/// there would shear the sky against the terrain for as long as a key was held.
#[derive(Component, Clone, Copy, Default)]
pub(super) struct ScreenOverlay(ScreenUniform);

impl ScreenOverlay {
    /// The height ramp. Config, so this is written once on entering gameplay; the
    /// water line is read from [`TerrainConfig`] rather than restated, so "where the
    /// sea stops" stays one number in the crate.
    pub(super) fn set_ramp(&mut self, terrain: &TerrainConfig, config: &TerrainTintConfig) {
        self.0.water_line = terrain.shallow_water_max;
        self.0.tint_low = config.tint_low;
        self.0.tint_high = config.tint_high;
        self.0.strength = config.strength;
        // The dither's period comes over with the ramp rather than with the ground,
        // because it is a property of the *map* — how much world one repeat covers —
        // and the map is the tint's.
        self.0.dither_period_tiles = config.dither_period_tiles.max(1) as f32;
    }

    /// What the weather has left on the ground, and what the day is doing to it.
    ///
    /// The knobs and the temperature offset arrive together because they are one
    /// question — how to draw the cover this step — and the three quantization
    /// constants ride along so that the shader can decode the climate map without
    /// restating numbers `ground.rs` owns.
    pub(super) fn set_ground(&mut self, config: &GroundConfig, offset: TemperatureOffset) {
        self.0.wet_darkening = config.wet_darkening;
        self.0.wet_desaturation = config.wet_desaturation;
        self.0.snow_lightening = config.snow_lightening;
        self.0.snow_dither_softness = config.snow_dither_softness;
        self.0.freezing_celsius = config.freezing_celsius;
        self.0.freezing_softness_celsius = config.freezing_softness_celsius;
        self.0.temperature_swing = offset.swing;
        self.0.temperature_seasonal_celsius = offset.seasonal_celsius;
        self.0.climate_min_celsius = CLIMATE_MIN_CELSIUS;
        self.0.climate_span_celsius = CLIMATE_MAX_CELSIUS - CLIMATE_MIN_CELSIUS;
        self.0.climate_amplitude_span_celsius = CLIMATE_MAX_AMPLITUDE_CELSIUS;
    }

    /// Where the sun reaches the screen. Everything about the light and the shadow
    /// arrives through this one call, so no other module has to know the field order.
    pub(super) fn set_sun(&mut self, sun: &Sun, config: &PlanetConfig) {
        self.0.sun_direct = sun.light.direct.extend(0.0);
        self.0.sun_sky = sun.light.sky.extend(0.0);
        self.0.sun_bearing = sun.position.bearing;
        self.0.sun_ray_slope = sun.position.ray_slope;
        self.0.shadow_softness = config.shadow_softness;
        self.0.relief_tiles = config.relief_tiles;
        self.0.shadow_near_tiles = config.shadow_near_tiles;
        self.0.shadow_mid_tiles = config.shadow_mid_tiles;
        self.0.shadow_far_tiles = config.shadow_far_tiles;
    }

    /// The sky's knobs and where its clock has drifted them to. The offsets are
    /// passed rather than the clock, so `WeatherClock` stays private to the module
    /// that advances it.
    pub(super) fn set_sky(
        &mut self,
        config: &WeatherConfig,
        coarse_offset: Vec2,
        fine_offset: Vec2,
        streak_phase: f32,
    ) {
        self.0.coarse_offset = coarse_offset;
        self.0.fine_offset = fine_offset;
        self.0.streak_phase = streak_phase;
        self.0.shape_period_tiles = config.shape_period_tiles;
        self.0.cloud_fine_scale = config.cloud_fine_scale;
        self.0.cloud_coarse_weight = config.cloud_coarse_weight;
        self.0.cloud_cut = config.cloud_cut;
        self.0.cloud_softness = config.cloud_softness;
        self.0.cloud_brightness = config.cloud_brightness;
        self.0.cloud_opacity = config.cloud_opacity;
        self.0.shadow_offset_tiles = config.shadow_offset_tiles;
        self.0.shadow_strength = config.shadow_strength;
        self.0.rain_cut = config.rain_cut;
        self.0.rain_softness = config.rain_softness;
        self.0.rain_strength = config.rain_strength;
    }
}

impl SyncComponent for ScreenOverlay {
    // The removal target, and getting it wrong is a one-way trip: extraction only ever
    // inserts, so if the render world is not told what to drop, the whole overlay
    // outlives gameplay and composites over the menus for the rest of the run.
    type Target = ScreenUniform;
}

impl ExtractComponent for ScreenOverlay {
    type QueryData = (
        &'static Self,
        &'static Camera,
        &'static Projection,
        &'static GlobalTransform,
    );
    type QueryFilter = ();
    type Out = ScreenUniform;

    fn extract_component(
        (overlay, camera, projection, transform): QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self::Out> {
        let half_extent = visible_half_extent(camera, projection);
        // A viewport with no size gives no scale to map screen back to world with,
        // and would collapse the whole world onto one tile.
        if half_extent.x <= 0.0 || half_extent.y <= 0.0 {
            return None;
        }

        let mut uniform = overlay.0;
        uniform.view_centre_tiles = tile_position_at(transform.translation().truncate());
        uniform.view_half_extent_tiles = half_extent / TILE_DISPLAY_SIZE.as_vec2();
        Some(uniform)
    }
}

pub struct ScreenEffectPlugin;

impl Plugin for ScreenEffectPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractComponentPlugin::<ScreenOverlay>::default(),
            UniformComponentPlugin::<ScreenUniform>::default(),
        ));
        app.add_systems(OnEnter(Screen::Gameplay), attach_screen_overlay);
        app.add_systems(OnExit(Screen::Gameplay), detach_screen_overlay);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.add_systems(RenderStartup, init_screen_pipeline);
        render_app.add_systems(ExtractSchedule, extract_height_uploads);
        render_app.add_systems(
            Render,
            (
                prepare_height_texture.in_set(RenderSystems::PrepareResources),
                prepare_screen_pipelines.in_set(RenderSystems::Prepare),
                prepare_screen_bind_groups.in_set(RenderSystems::PrepareBindGroups),
            ),
        );
        // After tonemapping, so everything here works on the same values the screen
        // shows; in PostProcess, because `bevy_ui_render` orders its own pass after
        // that whole set — which is what keeps the world's shading and its weather off
        // the menus and tooltips.
        render_app.add_systems(
            Core2d,
            screen_effect_pass
                .in_set(Core2dSystems::PostProcess)
                .after(tonemapping),
        );
    }
}

/// Puts an overlay on the camera, which survives every screen transition and so
/// cannot use `DespawnOnExit`.
///
/// The values it opens with are the fallbacks each contributor would want if it were
/// not in the app at all: full daylight and a flat ray, so a zeroed uniform never
/// draws the world black. The ramp and the sky are filled in by their own modules,
/// ordered after this.
pub(super) fn attach_screen_overlay(
    mut commands: Commands,
    camera: Single<Entity, With<WorldCamera>>,
) {
    commands
        .entity(*camera)
        .insert(ScreenOverlay(ScreenUniform {
            world_tiles: WORLD_TILES.as_vec2(),
            sun_direct: Vec4::new(1.0, 1.0, 1.0, 0.0),
            // A degenerate ramp would divide by zero before `set_ramp` lands. One that
            // spans the whole range shades nothing visibly and cannot produce a NaN.
            tint_low: 0.0,
            tint_high: 1.0,
            ..default()
        }));
}

fn detach_screen_overlay(mut commands: Commands, camera: Single<Entity, With<WorldCamera>>) {
    commands.entity(*camera).remove::<ScreenOverlay>();
}

// -- The heightmap -----------------------------------------------------------

/// The chunks whose heights this frame's extract took off the main world's queue.
#[derive(Resource, Default)]
struct PendingHeightUploads(Vec<ChunkHeights>);

/// The render world's copy of the world's heights: one texel per tile over the whole
/// world, and the only place the shader can learn how high the ground is.
#[derive(Resource)]
struct TerrainHeightTexture {
    texture: Texture,
    view: TextureView,
}

/// Moves the queued chunks across, rather than copying them, so a chunk is queued
/// once, uploaded once, and cannot be missed or repeated.
///
/// The queue's *absence* is the other half of this: it lives and dies with
/// `WorldMap`, so no queue means no session, and the heightmap goes with it. A
/// session can therefore never be shown under the next session's terrain.
fn extract_height_uploads(mut commands: Commands, mut main_world: ResMut<MainWorld>) {
    let Some(mut queue) = main_world.get_resource_mut::<HeightUploadQueue>() else {
        commands.remove_resource::<TerrainHeightTexture>();
        commands.remove_resource::<PendingHeightUploads>();
        return;
    };

    commands.insert_resource(PendingHeightUploads(queue.take()));
}

fn prepare_height_texture(
    mut commands: Commands,
    texture: Option<Res<TerrainHeightTexture>>,
    pending: Option<ResMut<PendingHeightUploads>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let Some(mut pending) = pending else {
        return;
    };

    // Created and written to in the same call, deliberately. Inserting the resource
    // and leaving the writes to the next frame would lose everything queued on the
    // frame it was created — which is the whole first screenful, since entering
    // gameplay generates that unbudgeted. The texture is only needed as a value to
    // write into; the resource is for the bind group, which can wait a frame.
    let texture = match texture {
        Some(texture) => texture.texture.clone(),
        None => {
            // wgpu zero-initializes, and zero is below any water line, so a world
            // whose chunks have not arrived yet is untinted rather than wrong — the
            // absence is the fallback, the way an unbaked sky is clear.
            let texture = render_device.create_texture(&TextureDescriptor {
                label: Some("terrain_height_texture"),
                size: Extent3d {
                    width: WORLD_TILES.x,
                    height: WORLD_TILES.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&TextureViewDescriptor::default());
            commands.insert_resource(TerrainHeightTexture {
                texture: texture.clone(),
                view,
            });
            texture
        }
    };

    // No budget of its own: the background pass yields at most one chunk per pool
    // thread per frame, so this is a few tens of kilobytes — where a whole-map
    // upload would have been 16 MB every time a chunk landed.
    for chunk in pending.0.drain(..) {
        let coord = UVec2::new(
            chunk.chunk as u32 % WORLD_CHUNKS.x,
            chunk.chunk as u32 / WORLD_CHUNKS.x,
        );
        let origin = coord * CHUNK_SIZE;

        render_queue.write_texture(
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d {
                    x: origin.x,
                    y: origin.y,
                    z: 0,
                },
                aspect: TextureAspect::All,
            },
            &chunk.texels,
            TexelCopyBufferLayout {
                offset: 0,
                // One byte per tile, and the chunk's rows are stored bottom-up in
                // the same order the texture's are — so no flip exists anywhere,
                // and the shader can load by tile coordinate directly.
                //
                // 64 is not a multiple of the 256-byte row alignment, which is fine:
                // that requirement is `copy_buffer_to_texture`'s, and `write_texture`
                // explicitly waives it.
                bytes_per_row: Some(CHUNK_SIZE.x),
                rows_per_image: Some(CHUNK_SIZE.y),
            },
            Extent3d {
                width: CHUNK_SIZE.x,
                height: CHUNK_SIZE.y,
                depth_or_array_layers: 1,
            },
        );
    }
}

// -- The pipeline ------------------------------------------------------------

#[derive(Resource)]
struct ScreenEffectPipeline {
    layout: BindGroupLayoutDescriptor,
    /// For the scene texture and for the blank stand-ins. The weather's maps bring
    /// their own, baked with the filtering and address mode each one needs.
    scene_sampler: Sampler,
    /// A 1x1 texture reading zero, standing in for any map that has not landed. Zero
    /// is a height below the water line, a cloud probability of nothing and dry
    /// ground, so every absence is the fallback the module in question documents.
    blank: TextureView,
    /// And the one map whose zero would be wrong — see [`CLIMATE_FALLBACK_CELSIUS`].
    blank_climate: TextureView,
    variants: Variants<RenderPipeline, ScreenEffectSpecializer>,
}

/// One texel of a two-channel texture, written once and read wherever a real map has
/// not landed.
///
/// `Rg8` for all of them: the binding layout only asks that a texture be filterable
/// and two-dimensional, so the same shape stands in for the R8 heightmap, the R8
/// cloud maps and the Rg8 cover alike.
fn blank_texture(
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    label: &str,
    texel: [u8; 2],
) -> TextureView {
    render_device
        .create_texture_with_data(
            render_queue,
            &TextureDescriptor {
                label: Some(label),
                size: Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rg8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            },
            TextureDataOrder::default(),
            &texel,
        )
        .create_view(&TextureViewDescriptor::default())
}

struct ScreenEffectSpecializer;

#[derive(PartialEq, Eq, Hash, Clone, Copy, SpecializerKey)]
struct ScreenEffectPipelineKey {
    target_format: TextureFormat,
}

impl Specializer<RenderPipeline> for ScreenEffectSpecializer {
    type Key = ScreenEffectPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        descriptor: &mut RenderPipelineDescriptor,
    ) -> Result<Canonical<Self::Key>, BevyError> {
        descriptor.fragment_mut()?.set_target(
            0,
            ColorTargetState {
                format: key.target_format,
                blend: None,
                write_mask: ColorWrites::ALL,
            },
        );
        Ok(key)
    }
}

fn init_screen_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "screen_effect_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                // The heightmap needs no sampler — the shader loads it at the tile's
                // own integer coordinate rather than sampling it.
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                // The ground's cover, the dither it is thresholded against and the
                // climate that says whether what falls is rain or snow.
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                uniform_buffer::<ScreenUniform>(true),
            ),
        ),
    );

    // Two stand-ins for maps that have not landed. Both are one texel: the binding
    // layout only cares that a texture is filterable and two-dimensional, so a single
    // `Rg8` reads correctly wherever the shader takes `.r` or `.rg`.
    //
    // wgpu zero-initializes, which is the whole of what the first one is for — zero
    // height is below any water line, zero cloud probability is a clear sky and zero
    // cover is dry ground, so every absence is the fallback its own module already
    // documents.
    let blank = blank_texture(
        &render_device,
        &render_queue,
        "screen_effect_blank_texture",
        [0, 0],
    );
    // The climate is the exception, because *its* zero is -40 C and would put the
    // whole world under snow before the bake landed. Mild and unvarying instead: an
    // unbaked climate rains, which is the behaviour that was there before this
    // feature existed.
    let blank_climate = blank_texture(
        &render_device,
        &render_queue,
        "screen_effect_blank_climate_texture",
        [
            (((CLIMATE_FALLBACK_CELSIUS - CLIMATE_MIN_CELSIUS)
                / (CLIMATE_MAX_CELSIUS - CLIMATE_MIN_CELSIUS))
                * 255.0)
                .round() as u8,
            0,
        ],
    );

    commands.insert_resource(ScreenEffectPipeline {
        layout: layout.clone(),
        scene_sampler: render_device.create_sampler(&SamplerDescriptor::default()),
        blank,
        blank_climate,
        variants: Variants::new(
            ScreenEffectSpecializer,
            RenderPipelineDescriptor {
                label: Some("screen_effect_pipeline".into()),
                layout: vec![layout],
                vertex: fullscreen_shader.to_vertex_state(),
                fragment: Some(FragmentState {
                    shader: asset_server.load(SCREEN_SHADER_PATH),
                    targets: vec![Some(ColorTargetState {
                        format: TextureFormat::Rgba8UnormSrgb,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })],
                    ..default()
                }),
                ..default()
            },
        ),
    });
}

#[derive(Component)]
struct ScreenEffectPipelineId(CachedRenderPipelineId);

fn prepare_screen_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipeline: ResMut<ScreenEffectPipeline>,
    views: Query<(Entity, &ExtractedView), With<ExtractedCamera>>,
) -> Result<(), BevyError> {
    for (entity, view) in &views {
        let id = pipeline.variants.specialize(
            &pipeline_cache,
            ScreenEffectPipelineKey {
                target_format: view.target_format,
            },
        )?;
        commands.entity(entity).insert(ScreenEffectPipelineId(id));
    }

    Ok(())
}

/// A bind group for each of the two textures the view target ping-pongs between,
/// since which one is the source is only known inside the pass.
///
/// `a_view` is which view `a` samples, and it has to be recorded rather than
/// re-derived: `post_process_write` flips the target *before* handing back its
/// source, so `main_texture_view()` inside the pass is the destination, and there
/// is nothing left in the target that says which texture `a` was built from.
#[derive(Component)]
struct ScreenEffectBindGroups {
    a_view: TextureViewId,
    a: BindGroup,
    b: BindGroup,
}

/// Rebuilt every frame rather than cached: the source texture changes with every
/// post-process write and a map's `GpuImage` is replaced whenever its asset is
/// re-uploaded, so there are several things to invalidate against and creating a bind
/// group costs microseconds.
fn prepare_screen_bind_groups(
    mut commands: Commands,
    views: Query<(Entity, &ViewTarget), With<ScreenUniform>>,
    pipeline: Option<Res<ScreenEffectPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    uniforms: Res<ComponentUniforms<ScreenUniform>>,
    heights: Option<Res<TerrainHeightTexture>>,
    maps: Option<Res<WeatherMaps>>,
    cover: Option<Res<GroundCoverTexture>>,
    dither: Option<Res<GroundDither>>,
    climate: Option<Res<ClimateTexture>>,
    images: Res<RenderAssets<GpuImage>>,
    render_device: Res<RenderDevice>,
) {
    let Some(pipeline) = pipeline else {
        return;
    };
    let Some(uniform_binding) = uniforms.uniforms().binding() else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);

    // Each map falls back independently, so one that has not landed cannot take the
    // others down with it: the ramp works before the sky has baked, the sky works
    // over terrain still arriving, and both work before the climate is up.
    let height_view = match &heights {
        Some(heights) => &heights.view,
        None => &pipeline.blank,
    };
    fn map<'a>(
        image: Option<&'a GpuImage>,
        blank: &'a TextureView,
        blank_sampler: &'a Sampler,
    ) -> (&'a TextureView, &'a Sampler) {
        match image {
            Some(image) => (&image.texture_view, &image.sampler),
            None => (blank, blank_sampler),
        }
    }
    let fallback = &pipeline.scene_sampler;
    let (probability_view, probability_sampler) = map(
        maps.as_ref().and_then(|maps| images.get(&maps.probability)),
        &pipeline.blank,
        fallback,
    );
    let (shape_view, shape_sampler) = map(
        maps.as_ref().and_then(|maps| images.get(&maps.shape)),
        &pipeline.blank,
        fallback,
    );
    let (cover_view, cover_sampler) = map(
        cover.as_ref().and_then(|cover| images.get(&cover.0)),
        &pipeline.blank,
        fallback,
    );
    let (dither_view, dither_sampler) = map(
        dither.as_ref().and_then(|dither| images.get(&dither.0)),
        &pipeline.blank,
        fallback,
    );
    let (climate_view, climate_sampler) = map(
        climate.as_ref().and_then(|climate| images.get(&climate.0)),
        &pipeline.blank_climate,
        fallback,
    );

    for (entity, target) in &views {
        let bind_group = |scene: &_| {
            render_device.create_bind_group(
                "screen_effect_bind_group",
                &layout,
                &BindGroupEntries::sequential((
                    scene,
                    &pipeline.scene_sampler,
                    height_view,
                    probability_view,
                    probability_sampler,
                    shape_view,
                    shape_sampler,
                    cover_view,
                    cover_sampler,
                    dither_view,
                    dither_sampler,
                    climate_view,
                    climate_sampler,
                    uniform_binding.clone(),
                )),
            )
        };

        commands.entity(entity).insert(ScreenEffectBindGroups {
            a_view: target.main_texture_view().id(),
            a: bind_group(target.main_texture_view()),
            b: bind_group(target.main_texture_other_view()),
        });
    }
}

/// Shades the world by its height, lights it, and composites the weather over it —
/// in that order, because that is the order of the lines in the fragment function.
///
/// Every part of the guard is the query: a view with no `ScreenUniform` — a menu, or
/// any frame outside gameplay — matches nothing, the system is skipped, and the scene
/// is never even copied.
fn screen_effect_pass(
    view: ViewQuery<(
        &ViewTarget,
        &DynamicUniformIndex<ScreenUniform>,
        &ScreenEffectBindGroups,
        &ScreenEffectPipelineId,
    )>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let (target, uniform_index, bind_groups, pipeline_id) = view.into_inner();

    let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id.0) else {
        return;
    };

    let post_process = target.post_process_write();
    let bind_group = if post_process.source.id() == bind_groups.a_view {
        &bind_groups.a
    } else {
        &bind_groups.b
    };

    let mut pass = ctx
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("screen_effect_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination,
                depth_slice: None,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[uniform_index.index()]);
    pass.draw(0..3, 0..1);
}
