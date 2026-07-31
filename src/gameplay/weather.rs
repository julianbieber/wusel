//! Weather: cloud patches drifting over the world, the shadow each one throws,
//! and rain in the thick of them.
//!
//! Two textures and one full-screen pass, and the split between them is the whole
//! design:
//!
//! * **Where** it is cloudy is the humidity field — the same field the river
//!   springs read — baked once per session over the whole world. That map never
//!   moves, so a wet range is reliably overcast and a dry one reliably is not.
//! * **What** a cloud looks like is a second map holding one *tiling* period of
//!   the same noise. The shader scrolls it, which is the entire animation.
//!
//! So the shader evaluates no noise: it samples two textures. That is not only
//! ~50x cheaper than an fbm per fragment (which at 4K is the whole frame budget),
//! it also keeps [`crate::gameplay::noise`] the only noise in the crate, which is
//! what the terrain's determinism tests rest on.
//!
//! The field is anchored in *world* space, not screen space: the view centre comes
//! from the camera at extract time, after the whole main-world frame, so panning
//! can never drag the clouds a frame behind the terrain under them.
//!
//! Nothing here outlives [`Screen::Gameplay`] except the knobs. The camera does —
//! it is the app's only one and is never despawned — so the overlay component is
//! taken off it on the way out rather than relying on `DespawnOnExit`.

use bevy::{
    asset::RenderAssetUsages,
    core_pipeline::{Core2d, Core2dSystems, FullscreenShader, tonemapping::tonemapping},
    ecs::query::QueryItem,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        camera::ExtractedCamera,
        extract_component::{
            ComponentUniforms, DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin,
            UniformComponentPlugin,
        },
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            CachedRenderPipelineId, Canonical, ColorTargetState, ColorWrites, Extent3d,
            FragmentState, Operations, PipelineCache, RenderPassColorAttachment,
            RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, Sampler,
            SamplerBindingType, SamplerDescriptor, ShaderStages, ShaderType, Specializer,
            SpecializerKey, TextureDimension, TextureFormat, TextureSampleType, TextureViewId,
            Variants,
            binding_types::{sampler, texture_2d, uniform_buffer},
        },
        renderer::{RenderContext, RenderDevice, ViewQuery},
        sync_component::SyncComponent,
        texture::GpuImage,
        view::{ExtractedView, ViewTarget},
    },
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};

use crate::{
    camera::{WorldCamera, visible_half_extent},
    gameplay::{
        ScreenEffectSystems,
        noise::TilingNoiseField,
        terrain::{TerrainConfig, TerrainSampler},
        world::{TILE_DISPLAY_SIZE, WORLD_TILES, tile_position_at},
    },
    screens::Screen,
};

const WEATHER_SHADER_PATH: &str = "shaders/weather.wgsl";

/// Salt for the cloud-shape field, so the sky is not a second view of a landscape.
const CLOUD_SHAPE_SALT: u32 = 0xc10d_5eed;

/// Noise cells across one period of the shape map. With the default 256-tile period
/// this puts the coarsest cloud lump at 32 tiles and, over four octaves, the finest
/// at 4 — five texels of a 256px map, which is about as fine as a baked field can
/// carry. It is a power of two because that is what lets every octave's lattice wrap
/// exactly.
const SHAPE_LATTICE_PERIOD: u32 = 8;

/// Everything about the weather that is a knob rather than world state.
///
/// Flat, like [`TerrainConfig`] and `WorldPlanConfig`, and for the same reason: the
/// shader's uniform is flat regardless, and the couplings between these values —
/// `rain_cut` sitting above the cut, `shadow_offset_tiles` needing to clear one
/// cloud's width — read as neighbours here rather than across nested structs.
#[derive(Resource, Clone)]
pub struct WeatherConfig {
    /// Texels along each side of the probability map, covering the whole world: 512
    /// over 4096 tiles is one texel per 8 tiles. The humidity field's coarsest
    /// wavelength is ~50 tiles, so this oversamples it 6x; the finer octaves it
    /// misses carry ~10% of the amplitude, and a texel sits within 0.05 of the
    /// average over its own 8x8 tiles.
    pub probability_texels_per_side: u32,
    /// Texels along each side of the tiling shape map. 256 over a 256-tile period is
    /// one tile per texel.
    pub shape_texels_per_side: u32,
    /// How much world one repeat of the shape map covers, which sets how big a cloud
    /// is: the coarsest lump in the field is an eighth of it.
    ///
    /// This is the one real tension in the whole overlay. At 512 a cloud was 64 tiles
    /// across — half the screen at scale 1, so the sky read as fog banks rather than
    /// weather. At 256 a cloud is ~32 tiles and a handful fit on screen, at the cost
    /// of the period repeating ~4 times across the widest zoom-out; the second layer
    /// at a fractional scale is what keeps that from reading as a pattern.
    pub shape_period_tiles: f32,
    /// Octaves baked into the shape map. Four, because the fifth would land inside
    /// two texels and only alias.
    pub shape_octaves: u32,
    /// How fast the sky moves, in tiles per second. A chunk is 64 tiles, so 2
    /// crosses one every 32 seconds.
    pub wind_drift_tiles_per_second: Vec2,
    /// The second shape layer's frequency, relative to the first. Deliberately not
    /// a whole number: the two layers beat against each other instead of lining up
    /// every period.
    pub cloud_fine_scale: f32,
    /// How fast the second layer drifts, relative to the first. Below 1 so the
    /// layers shear rather than sliding as one sheet — this is what makes a cloud
    /// look like it is changing shape rather than merely moving.
    pub cloud_fine_drift: f32,
    /// The first layer's share of the shape. The rest is the second layer.
    pub cloud_coarse_weight: f32,
    /// The cut in the raw cloud field below which the sky is clear.
    ///
    /// A cut on a field, *not* a coverage fraction, and sharper than it looks:
    /// probability times shape has a mean of 0.250, so at 0.20 about 62% of the world
    /// is under cloud, at 0.28 it is 35.9% under cloud with 64.4% carrying some, at
    /// 0.40 about 12%, and anything above ~0.55 is a permanently clear sky.
    pub cloud_cut: f32,
    /// Width of the smooth step around `cloud_cut` — the softness of a cloud edge.
    pub cloud_softness: f32,
    /// How bright the cloud itself is. Under 1 because pure white over 8px pixel
    /// art reads as a hole in the world rather than as weather.
    pub cloud_brightness: f32,
    /// How opaque the thickest cloud is allowed to get. Well under 1: at full
    /// opacity a cloud replaces the world under it, and what you want to see is
    /// terrain *through* weather.
    pub cloud_opacity: f32,
    /// Where a cloud's shadow falls, in tiles: cloud altitude times sun angle. The
    /// shadow of a cloud lands `-this` from it, so the default puts it down and to
    /// the right, with the sun up and to the left.
    ///
    /// Has to be a decent fraction of a cloud's own width or the shadow hides
    /// underneath the cloud casting it, and the coarsest lump is
    /// `shape_period_tiles / 8` across.
    pub shadow_offset_tiles: Vec2,
    /// How much of the light a full cloud takes away.
    pub shadow_strength: f32,
    /// How thick a cloud has to be before it rains, as a cut on the raw field rather
    /// than on the density: the density saturates, so almost every cloud clears any
    /// cut placed on it, and cutting there rained on 28.6% of the world at once. At
    /// 0.46 some rain falls on 4.2% of it and rain worth looking at on 1.6%.
    ///
    /// The rain is still *multiplied* by the density, which is what keeps "no rain
    /// out of a clear sky" true for any setting of these knobs rather than only for
    /// ones where this sits above `cloud_cut`.
    pub rain_cut: f32,
    /// Width of the ramp above `rain_cut`, so a rain patch fades in from the edge of
    /// the cloud's thick part instead of arriving with a rim.
    pub rain_softness: f32,
    /// How hard the rain darkens and greys what is under it.
    pub rain_strength: f32,
    /// Streak periods per second. The streaks are drawn in screen space — in world
    /// space they would be 16x denser at one end of the zoom range than the other.
    pub rain_streak_speed: f32,
}

impl Default for WeatherConfig {
    fn default() -> Self {
        Self {
            probability_texels_per_side: 512,
            shape_texels_per_side: 256,
            shape_period_tiles: 256.0,
            shape_octaves: 4,
            wind_drift_tiles_per_second: Vec2::new(2.0, 0.6),
            cloud_fine_scale: 2.6,
            cloud_fine_drift: 0.55,
            cloud_coarse_weight: 0.65,
            cloud_cut: 0.30,
            cloud_softness: 0.08,
            cloud_brightness: 0.92,
            cloud_opacity: 0.55,
            shadow_offset_tiles: Vec2::new(-8.0, 6.0),
            shadow_strength: 0.38,
            rain_cut: 0.46,
            rain_softness: 0.18,
            rain_strength: 0.45,
            rain_streak_speed: 1.8,
        }
    }
}

/// Where the sky has drifted to. In map periods, wrapped to `0..1`, which is what
/// keeps a long session from quantizing: the offsets never grow.
#[derive(Resource, Default)]
struct WeatherClock {
    coarse_offset: Vec2,
    fine_offset: Vec2,
    streak_phase: f32,
}

/// The bake in flight. Dropping the resource cancels it, so maps baked for one
/// world can never land in the next.
#[derive(Resource)]
struct WeatherBake(Task<BakedMaps>);

struct BakedMaps {
    probability: Image,
    shape: Image,
}

/// The two maps, once they are on the GPU's side of the asset server.
///
/// Absent until the bake lands, and that absence is the feature: with no maps the
/// pass leaves the scene alone, so the first few frames of a session are a clear
/// sky rather than a stall.
#[derive(Resource, Clone)]
struct WeatherMaps {
    probability: Handle<Image>,
    shape: Handle<Image>,
}

impl ExtractResource for WeatherMaps {
    type Source = Self;

    fn extract_resource(source: &Self) -> Self {
        source.clone()
    }
}

/// The uniform the shader reads, one per view.
///
/// Field order *is* the wgsl binding layout — the two structs are edited together
/// or the shader fails to compile at runtime. Vectors first, scalars after, so the
/// std140 padding is the same on both sides and WebGL2 agrees with the desktop.
#[derive(Component, Clone, Copy, Default, ShaderType)]
struct WeatherUniform {
    view_centre_tiles: Vec2,
    view_half_extent_tiles: Vec2,
    coarse_offset: Vec2,
    fine_offset: Vec2,
    shadow_offset_tiles: Vec2,
    world_tiles: Vec2,
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

/// The main-world half of that uniform, carried by the one world camera while
/// gameplay is up. It holds everything the main world decides; the two view fields
/// are filled in at extract time from the camera itself, which is why no ordering
/// against the pan is needed anywhere.
#[derive(Component, Clone, Copy, Default)]
struct WeatherOverlay(WeatherUniform);

impl SyncComponent for WeatherOverlay {
    // The removal target, and getting this wrong is a one-way trip: extraction only
    // ever inserts, so if the render world is not told what to drop, the weather
    // outlives gameplay and composites over the menus for the rest of the run.
    type Target = WeatherUniform;
}

impl ExtractComponent for WeatherOverlay {
    type QueryData = (
        &'static Self,
        &'static Camera,
        &'static Projection,
        &'static GlobalTransform,
    );
    type QueryFilter = ();
    type Out = WeatherUniform;

    fn extract_component(
        (overlay, camera, projection, transform): QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self::Out> {
        let half_extent = visible_half_extent(camera, projection);
        // A viewport with no size gives no scale to map screen back to world with,
        // and would collapse the whole sky onto one tile.
        if half_extent.x <= 0.0 || half_extent.y <= 0.0 {
            return None;
        }

        let mut uniform = overlay.0;
        uniform.view_centre_tiles = tile_position_at(transform.translation().truncate());
        uniform.view_half_extent_tiles = half_extent / TILE_DISPLAY_SIZE.as_vec2();
        Some(uniform)
    }
}

pub struct WeatherPlugin;

impl Plugin for WeatherPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WeatherConfig>();
        app.add_plugins((
            ExtractComponentPlugin::<WeatherOverlay>::default(),
            UniformComponentPlugin::<WeatherUniform>::default(),
            ExtractResourcePlugin::<WeatherMaps>::default(),
        ));
        app.add_systems(
            OnEnter(Screen::Gameplay),
            (start_weather_bake, attach_weather_overlay),
        );
        app.add_systems(OnExit(Screen::Gameplay), detach_weather_overlay);
        app.add_systems(
            Update,
            (
                finish_weather_bake,
                // The clock is the only writer of its own state and the overlay is
                // the only reader, so this pair is the whole ordering the main
                // world needs.
                (advance_weather_clock, sync_weather_overlay).chain(),
            )
                .run_if(in_state(Screen::Gameplay)),
        );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.add_systems(RenderStartup, init_weather_pipeline);
        render_app.add_systems(
            Render,
            (
                prepare_weather_pipelines.in_set(RenderSystems::Prepare),
                prepare_weather_bind_groups.in_set(RenderSystems::PrepareBindGroups),
            ),
        );
        // After tonemapping, so the darkening works on the same values the screen
        // shows; in PostProcess, because `bevy_ui_render` puts its pass after that
        // whole set — which is what keeps weather off the menus and tooltips. And
        // over the terrain's own shading, which `ScreenEffectSystems` orders.
        render_app.add_systems(
            Core2d,
            weather_pass
                .in_set(Core2dSystems::PostProcess)
                .in_set(ScreenEffectSystems::Weather)
                .after(tonemapping),
        );
    }
}

// -- The main world ----------------------------------------------------------

fn start_weather_bake(
    mut commands: Commands,
    terrain: Res<TerrainConfig>,
    config: Res<WeatherConfig>,
) {
    // At zero rather than wherever the last session left off: the sky is world
    // state, and a new world gets a new one.
    commands.insert_resource(WeatherClock::default());

    let terrain = terrain.clone();
    let config = config.clone();
    let task = AsyncComputeTaskPool::get().spawn(async move { bake_maps(&terrain, &config) });
    commands.insert_resource(WeatherBake(task));
}

fn attach_weather_overlay(mut commands: Commands, camera: Single<Entity, With<WorldCamera>>) {
    commands.entity(*camera).insert(WeatherOverlay::default());
}

/// Takes the overlay off the camera, which survives this transition, and drops the
/// session's state. A bake still in flight goes with it.
fn detach_weather_overlay(mut commands: Commands, camera: Single<Entity, With<WorldCamera>>) {
    commands.entity(*camera).remove::<WeatherOverlay>();
    commands.remove_resource::<WeatherMaps>();
    commands.remove_resource::<WeatherClock>();
    commands.remove_resource::<WeatherBake>();
}

fn finish_weather_bake(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    bake: Option<ResMut<WeatherBake>>,
) {
    let Some(mut bake) = bake else {
        return;
    };
    let Some(baked) = block_on(poll_once(&mut bake.0)) else {
        return;
    };

    commands.insert_resource(WeatherMaps {
        probability: images.add(baked.probability),
        shape: images.add(baked.shape),
    });
    commands.remove_resource::<WeatherBake>();
}

fn advance_weather_clock(
    time: Res<Time>,
    config: Res<WeatherConfig>,
    mut clock: ResMut<WeatherClock>,
) {
    let delta = time.delta_secs();
    let drift = config.wind_drift_tiles_per_second / config.shape_period_tiles * delta;

    // Wrapped rather than accumulated: an offset that grew all session would
    // eventually quantize, and a map period is exactly where a wrap is invisible.
    clock.coarse_offset = (clock.coarse_offset + drift).fract();
    clock.fine_offset =
        (clock.fine_offset + drift * config.cloud_fine_drift * config.cloud_fine_scale).fract();
    clock.streak_phase = (clock.streak_phase + config.rain_streak_speed * delta).fract();
}

fn sync_weather_overlay(
    config: Res<WeatherConfig>,
    clock: Res<WeatherClock>,
    mut overlay: Single<&mut WeatherOverlay>,
) {
    let uniform = &mut overlay.0;
    uniform.coarse_offset = clock.coarse_offset;
    uniform.fine_offset = clock.fine_offset;
    uniform.streak_phase = clock.streak_phase;
    uniform.world_tiles = WORLD_TILES.as_vec2();
    uniform.shape_period_tiles = config.shape_period_tiles;
    uniform.cloud_fine_scale = config.cloud_fine_scale;
    uniform.cloud_coarse_weight = config.cloud_coarse_weight;
    uniform.cloud_cut = config.cloud_cut;
    uniform.cloud_softness = config.cloud_softness;
    uniform.cloud_brightness = config.cloud_brightness;
    uniform.cloud_opacity = config.cloud_opacity;
    uniform.shadow_offset_tiles = config.shadow_offset_tiles;
    uniform.shadow_strength = config.shadow_strength;
    uniform.rain_cut = config.rain_cut;
    uniform.rain_softness = config.rain_softness;
    uniform.rain_strength = config.rain_strength;
}

// -- The bake ----------------------------------------------------------------

/// The probability that it is cloudy over a tile: the humidity the terrain sampler
/// reports there, unchanged. Rivers rise in the wet mountains and it rains over the
/// same country — and since the biome's `humidity_bias` is part of that answer, a
/// desert is reliably clear and a wetland reliably overcast without the sky knowing
/// what a biome is.
fn cloud_probability_at(sampler: &TerrainSampler, tile: Vec2) -> f32 {
    sampler.humidity(tile.x, tile.y)
}

/// How much cloud is over a tile: the density, from the raw field there.
///
/// Zero is clear sky and one is solid overcast, and the step between them is sharp
/// enough that most of a cloud is at one or the other — which is why the rain below
/// is cut on the raw field instead.
///
/// The shader is the implementation — the density has to be evaluated per fragment,
/// because it moves. This is the same arithmetic in Rust so that the properties the
/// sky is supposed to have can be measured over the whole world without a GPU; the
/// two are edited together, like the uniform and its wgsl struct.
#[cfg(test)]
fn cloud_density(config: &WeatherConfig, field: f32) -> f32 {
    smoothstep(
        config.cloud_cut - config.cloud_softness,
        config.cloud_cut + config.cloud_softness,
        field,
    )
}

/// How hard it is raining under a cloud, from the raw field and the density it
/// produced. Ramped rather than switched, so a rain patch has no rim, and multiplied
/// by the density, so rain out of a clear sky is not unlikely but impossible.
#[cfg(test)]
fn rain_amount(config: &WeatherConfig, field: f32, density: f32) -> f32 {
    smoothstep(
        config.rain_cut,
        config.rain_cut + config.rain_softness,
        field,
    ) * density
}

#[cfg(test)]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Bakes both maps. Called on the compute pool: this is ~260k fbm samples for the
/// probability map alone, which is ~50 ms — the same order as the chunk generation
/// the first gameplay frame is already doing, and no reason to add to it.
fn bake_maps(terrain: &TerrainConfig, config: &WeatherConfig) -> BakedMaps {
    BakedMaps {
        probability: bake_probability_map(terrain, config),
        shape: bake_shape_map(terrain, config),
    }
}

fn bake_probability_map(terrain: &TerrainConfig, config: &WeatherConfig) -> Image {
    let side = config.probability_texels_per_side.max(1);
    let tiles_per_texel = WORLD_TILES.x as f32 / side as f32;
    // Hoisted out of the loop: one sampler for the whole map, since building it
    // costs six noise fields and a biome map.
    let sampler = terrain.sampler();

    let mut texels = Vec::with_capacity((side * side) as usize);
    for y in 0..side {
        for x in 0..side {
            // The texel's centre, so the map is the humidity field at the points it
            // claims to sample rather than at their corners.
            let tile = (Vec2::new(x as f32, y as f32) + Vec2::splat(0.5)) * tiles_per_texel;
            texels.push(to_byte(cloud_probability_at(&sampler, tile)));
        }
    }

    // Clamped, so a view of the world's edge reads the edge of the map rather than
    // wrapping the far side of the world into shot.
    map_image(side, texels, ImageAddressMode::ClampToEdge)
}

fn bake_shape_map(terrain: &TerrainConfig, config: &WeatherConfig) -> Image {
    let side = config.shape_texels_per_side.max(1);
    let field = TilingNoiseField::new(
        terrain.seed,
        CLOUD_SHAPE_SALT,
        SHAPE_LATTICE_PERIOD,
        config.shape_octaves,
    );
    let cells_per_texel = SHAPE_LATTICE_PERIOD as f32 / side as f32;

    let mut texels = Vec::with_capacity((side * side) as usize);
    for y in 0..side {
        for x in 0..side {
            let cell = Vec2::new(x as f32, y as f32) * cells_per_texel;
            texels.push(to_byte(field.sample(cell.x, cell.y)));
        }
    }

    // Repeated, because scrolling it forever is the animation.
    map_image(side, texels, ImageAddressMode::Repeat)
}

fn to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// One byte per texel, sampled smoothly.
///
/// The sampler is set here rather than inherited: the app-wide default is *nearest*
/// for the 8px pixel art, and a nearest-sampled probability map would draw the sky
/// in visible 64px blocks.
fn map_image(side: u32, texels: Vec<u8>, address_mode: ImageAddressMode) -> Image {
    let mut image = Image::new(
        Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        texels,
        TextureFormat::R8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        min_filter: ImageFilterMode::Linear,
        mag_filter: ImageFilterMode::Linear,
        address_mode_u: address_mode,
        address_mode_v: address_mode,
        ..default()
    });
    image
}

// -- The render world --------------------------------------------------------

#[derive(Resource)]
struct WeatherPipeline {
    layout: BindGroupLayoutDescriptor,
    /// For the scene texture. The maps bring their own, baked with the filtering
    /// and address mode each one needs.
    scene_sampler: Sampler,
    variants: Variants<RenderPipeline, WeatherSpecializer>,
}

struct WeatherSpecializer;

#[derive(PartialEq, Eq, Hash, Clone, Copy, SpecializerKey)]
struct WeatherPipelineKey {
    target_format: TextureFormat,
}

impl Specializer<RenderPipeline> for WeatherSpecializer {
    type Key = WeatherPipelineKey;

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

fn init_weather_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "weather_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                uniform_buffer::<WeatherUniform>(true),
            ),
        ),
    );

    commands.insert_resource(WeatherPipeline {
        layout: layout.clone(),
        scene_sampler: render_device.create_sampler(&SamplerDescriptor::default()),
        variants: Variants::new(
            WeatherSpecializer,
            RenderPipelineDescriptor {
                label: Some("weather_pipeline".into()),
                layout: vec![layout],
                vertex: fullscreen_shader.to_vertex_state(),
                fragment: Some(FragmentState {
                    shader: asset_server.load(WEATHER_SHADER_PATH),
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
struct WeatherPipelineId(CachedRenderPipelineId);

fn prepare_weather_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipeline: ResMut<WeatherPipeline>,
    views: Query<(Entity, &ExtractedView), With<ExtractedCamera>>,
) -> Result<(), BevyError> {
    for (entity, view) in &views {
        let id = pipeline.variants.specialize(
            &pipeline_cache,
            WeatherPipelineKey {
                target_format: view.target_format,
            },
        )?;
        commands.entity(entity).insert(WeatherPipelineId(id));
    }

    Ok(())
}

/// A bind group for each of the two textures the view target ping-pongs between,
/// since which one is the source is only known inside the pass.
#[derive(Component)]
struct WeatherBindGroups {
    /// Which view `a` samples. Recorded rather than re-derived, because
    /// `post_process_write` flips the target *before* handing back its source, so
    /// `main_texture_view()` inside the pass is the destination. With this pass
    /// alone the parity worked out anyway; once the tint runs first, guessing it
    /// picks the texture this pass is writing to.
    a_view: TextureViewId,
    a: BindGroup,
    b: BindGroup,
}

/// Rebuilt every frame rather than cached: the source texture changes with every
/// post-process write, and either map's `GpuImage` is replaced whenever the asset
/// is re-uploaded, so there are three things to invalidate against and creating a
/// bind group costs microseconds.
///
/// The view gets no bind group at all until both maps are on the GPU — which is
/// what makes an unbaked sky clear rather than a sampling error.
fn prepare_weather_bind_groups(
    mut commands: Commands,
    views: Query<(Entity, &ViewTarget), With<WeatherUniform>>,
    pipeline: Option<Res<WeatherPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    uniforms: Res<ComponentUniforms<WeatherUniform>>,
    maps: Option<Res<WeatherMaps>>,
    images: Res<RenderAssets<GpuImage>>,
    render_device: Res<RenderDevice>,
) {
    let Some(pipeline) = pipeline else {
        return;
    };
    let Some(uniform_binding) = uniforms.uniforms().binding() else {
        return;
    };
    let Some(maps) = maps else {
        return;
    };
    let Some(probability) = images.get(&maps.probability) else {
        return;
    };
    let Some(shape) = images.get(&maps.shape) else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);

    for (entity, target) in &views {
        let bind_group = |scene: &_| {
            render_device.create_bind_group(
                "weather_bind_group",
                &layout,
                &BindGroupEntries::sequential((
                    scene,
                    &pipeline.scene_sampler,
                    &probability.texture_view,
                    &probability.sampler,
                    &shape.texture_view,
                    &shape.sampler,
                    uniform_binding.clone(),
                )),
            )
        };

        commands.entity(entity).insert(WeatherBindGroups {
            a_view: target.main_texture_view().id(),
            a: bind_group(target.main_texture_view()),
            b: bind_group(target.main_texture_other_view()),
        });
    }
}

/// Composites the weather over the rendered world.
///
/// Every part of the guard is the query: a view with no `WeatherUniform` — a menu,
/// or a session whose bake has not landed — matches nothing, the system is skipped,
/// and the scene is never even copied.
fn weather_pass(
    view: ViewQuery<(
        &ViewTarget,
        &DynamicUniformIndex<WeatherUniform>,
        &WeatherBindGroups,
        &WeatherPipelineId,
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
            label: Some("weather_pass"),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// One sample of the sky: the humidity there, the raw cloud field, and the
    /// density that field produces.
    struct Sky {
        probability: f32,
        field: f32,
        density: f32,
    }

    /// A grid of them over the default world, coarse enough to run in a test and
    /// fine enough to see cloud edges: one sample per 16 tiles.
    fn sample_densities() -> Vec<Sky> {
        let terrain = TerrainConfig::default();
        let config = WeatherConfig::default();
        let field = TilingNoiseField::new(
            terrain.seed,
            CLOUD_SHAPE_SALT,
            SHAPE_LATTICE_PERIOD,
            config.shape_octaves,
        );
        let sampler = terrain.sampler();

        let step = 16.0;
        let steps = (WORLD_TILES.x as f32 / step) as u32;
        let mut samples = Vec::new();
        for y in 0..steps {
            for x in 0..steps {
                let tile = Vec2::new(x as f32, y as f32) * step;
                let probability = cloud_probability_at(&sampler, tile);
                let cell = tile / config.shape_period_tiles * SHAPE_LATTICE_PERIOD as f32;
                let raw = probability * shape_at(&field, &config, cell);
                samples.push(Sky {
                    probability,
                    field: raw,
                    density: cloud_density(&config, raw),
                });
            }
        }
        samples
    }

    /// The two layers the shader mixes, at one point.
    fn shape_at(field: &TilingNoiseField, config: &WeatherConfig, cell: Vec2) -> f32 {
        let coarse = field.sample(cell.x, cell.y);
        let fine = field.sample(
            cell.x * config.cloud_fine_scale,
            cell.y * config.cloud_fine_scale,
        );
        coarse * config.cloud_coarse_weight + fine * (1.0 - config.cloud_coarse_weight)
    }

    /// The weather equivalent of `the_default_config_produces_every_base_kind`: the
    /// failure this guards against is a default that quietly yields one sky — either
    /// a world under permanent overcast or one where it never clouds over at all.
    #[test]
    fn the_default_weather_config_leaves_both_clear_sky_and_cloud_in_the_world() {
        let samples = sample_densities();
        let cloudy = samples.iter().filter(|sky| sky.density > 0.5).count();
        let clear = samples.iter().filter(|sky| sky.density <= 0.0).count();
        let fraction = cloudy as f32 / samples.len() as f32;

        assert!(
            (0.05..0.7).contains(&fraction),
            "{:.1}% of the world is under cloud",
            fraction * 100.0
        );
        assert!(
            clear > samples.len() / 10,
            "the sky is never clear anywhere"
        );
    }

    /// Where the figures in `WeatherConfig`'s doc comments come from. Ignored
    /// because it is a measurement rather than an assertion:
    /// `cargo test --release -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn the_default_config_measures_the_sky() {
        let config = WeatherConfig::default();
        let samples = sample_densities();
        let total = samples.len() as f32;
        let mean = |values: Vec<f32>| values.iter().sum::<f32>() / total;
        let fraction = |count: usize| count as f32 / total * 100.0;

        let mean_humidity = mean(samples.iter().map(|sky| sky.probability).collect());
        let mean_field = mean(samples.iter().map(|sky| sky.field).collect());
        let any_cloud = fraction(samples.iter().filter(|sky| sky.density > 0.0).count());
        let under_cloud = fraction(samples.iter().filter(|sky| sky.density > 0.5).count());
        let any_rain = fraction(
            samples
                .iter()
                .filter(|sky| rain_amount(&config, sky.field, sky.density) > 0.0)
                .count(),
        );
        let real_rain = fraction(
            samples
                .iter()
                .filter(|sky| rain_amount(&config, sky.field, sky.density) > 0.25)
                .count(),
        );

        println!("{} samples, one per 16 tiles", samples.len());
        println!("mean humidity            {mean_humidity:.3}");
        println!("mean raw field           {mean_field:.3}");
        println!("any cloud at all         {any_cloud:.1}%");
        println!("under cloud (d > 0.5)    {under_cloud:.1}%");
        println!("any rain at all          {any_rain:.1}%");
        println!("rain worth seeing        {real_rain:.1}%");
    }

    /// The point of driving the sky from the humidity field: it has to rain over the
    /// wet country and not over the dry, or the map may as well not be sampled.
    #[test]
    fn a_dry_region_never_clouds_over_and_a_wet_one_mostly_does() {
        let samples = sample_densities();
        let mean_density = |lo: f32, hi: f32| {
            let band: Vec<f32> = samples
                .iter()
                .filter(|sky| sky.probability >= lo && sky.probability < hi)
                .map(|sky| sky.density)
                .collect();
            assert!(!band.is_empty(), "no samples with humidity in {lo}..{hi}");
            band.iter().sum::<f32>() / band.len() as f32
        };

        let dry = mean_density(0.0, 0.25);
        let wet = mean_density(0.75, 1.01);
        assert!(
            dry < 0.05,
            "the driest quarter of the world still clouds over ({dry})"
        );
        assert!(
            wet > 0.5,
            "the wettest quarter of the world barely clouds over ({wet})"
        );
    }

    /// Rain is the raw field's ramp *times* the density, so this holds for any
    /// config rather than only for ones whose cuts are in the right order — which is
    /// the whole reason for the multiply.
    #[test]
    fn rain_falls_only_inside_a_cloud() {
        // Deliberately perverse: a rain cut of zero, below the cloud cut, is how you
        // would ask for rain out of a clear sky.
        let config = WeatherConfig {
            rain_cut: 0.0,
            ..default()
        };

        for i in 0..=100 {
            let field = i as f32 / 100.0;
            let density = cloud_density(&config, field);
            if density <= 0.0 {
                assert_eq!(
                    rain_amount(&config, field, density),
                    0.0,
                    "rain with no cloud above it, at a field value of {field}"
                );
            }
        }
    }

    /// That a shadow *is* the cloud field one offset away is true by construction —
    /// the shader calls one function twice. What is not automatic is that the offset
    /// clears a cloud's own body: too short and every shadow hides under the cloud
    /// casting it, which looks like no shadows at all.
    #[test]
    fn a_shadow_falls_far_enough_from_its_cloud_to_be_seen() {
        let terrain = TerrainConfig::default();
        let config = WeatherConfig::default();
        let field = TilingNoiseField::new(
            terrain.seed,
            CLOUD_SHAPE_SALT,
            SHAPE_LATTICE_PERIOD,
            config.shape_octaves,
        );
        let sampler = terrain.sampler();
        let density_at = |tile: Vec2| {
            let cell = tile / config.shape_period_tiles * SHAPE_LATTICE_PERIOD as f32;
            let raw = cloud_probability_at(&sampler, tile) * shape_at(&field, &config, cell);
            cloud_density(&config, raw)
        };

        let step = 16.0;
        let steps = (WORLD_TILES.x as f32 / step) as u32;
        let mut cloudy = 0;
        let mut in_the_open = 0;
        for y in 0..steps {
            for x in 0..steps {
                let tile = Vec2::new(x as f32, y as f32) * step;
                let cloud = density_at(tile);
                if cloud > 0.5 {
                    cloudy += 1;
                    // The tile this cloud shadows: is that tile itself in the clear?
                    if density_at(tile - config.shadow_offset_tiles) < 0.5 {
                        in_the_open += 1;
                    }
                }
            }
        }

        let visible = in_the_open as f32 / cloudy as f32;
        assert!(
            visible > 0.15,
            "only {:.1}% of shadows fall outside the cloud that casts them",
            visible * 100.0
        );
    }

    /// A long session must not quantize the sky. The offsets are wrapped, so the
    /// bound is on the wrap rather than on how long you play.
    #[test]
    fn a_long_session_does_not_quantize_the_weather_clock() {
        let config = WeatherConfig::default();
        let mut clock = WeatherClock::default();
        let delta = 1.0 / 60.0;
        let drift = config.wind_drift_tiles_per_second / config.shape_period_tiles * delta;

        // Ten hours at 60 fps.
        for _ in 0..(60 * 60 * 60 * 10) {
            clock.coarse_offset = (clock.coarse_offset + drift).fract();
            clock.streak_phase = (clock.streak_phase + config.rain_streak_speed * delta).fract();
        }

        assert!(clock.coarse_offset.x.abs() < 1.0 && clock.coarse_offset.y.abs() < 1.0);
        assert!(clock.streak_phase.abs() < 1.0);
        // The step is still resolvable at the end of it, which is the thing an
        // unwrapped accumulator loses.
        let stepped = (clock.coarse_offset + drift).fract();
        assert_ne!(stepped, clock.coarse_offset);
    }
}
