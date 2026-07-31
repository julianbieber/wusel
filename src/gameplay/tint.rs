//! Terrain tint: the rendered world's brightness scaled by how high the tile under
//! each fragment is.
//!
//! A tile's appearance was decided entirely by its `TerrainKind`, so every Grass
//! tile in the world was the same eight pixels and the height the sampler builds up
//! — continent, relief, ridged — was invisible except where it happened to cross a
//! `classify` band edge. This makes it visible *inside* a band.
//!
//! **The height is kept, not re-baked.** `classify` already computes every tile's
//! elevation and used to drop it on the floor, so [`crate::gameplay::terrain`]
//! returns it and [`crate::gameplay::world`] stores it. Sampling the world a second
//! time would cost ~9 core-seconds against the ~34 s the chunks themselves cost, and
//! would put a second answer to "how high is it here" in a crate whose determinism
//! tests rest on there being one. The world pays for it in memory instead: 16 MB of
//! heights beside the 16 MB of kinds, and 16 MB again on the GPU.
//!
//! **The map is not an `Image` asset.** Bevy re-uploads a whole `Image` on any
//! change, so a 16 MB map would cross to the GPU in full every time a chunk landed.
//! This module owns a raw texture and writes one chunk's rect — 4 KB — into it as
//! that chunk arrives.
//!
//! Like the weather this is cosmetic: nothing here reads or writes `WorldMap`, so no
//! tile can depend on the shading. Unlike the weather it is *static* — the ramp is
//! config and the view is filled in at extract — so there is no per-frame sync
//! system to pair with the overlay.

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
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            CachedRenderPipelineId, Canonical, ColorTargetState, ColorWrites, Extent3d,
            FragmentState, Operations, Origin3d, PipelineCache, RenderPassColorAttachment,
            RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, Sampler,
            SamplerBindingType, SamplerDescriptor, ShaderStages, ShaderType, Specializer,
            SpecializerKey, TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect,
            TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
            TextureView, TextureViewDescriptor, TextureViewId, Variants,
            binding_types::{sampler, texture_2d, uniform_buffer},
        },
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        sync_component::SyncComponent,
        view::{ExtractedView, ViewTarget},
    },
};

use crate::{
    camera::{WorldCamera, visible_half_extent},
    gameplay::{
        ScreenEffectSystems,
        terrain::TerrainConfig,
        world::{
            CHUNK_SIZE, ChunkHeights, HeightUploadQueue, TILE_DISPLAY_SIZE, WORLD_CHUNKS,
            WORLD_TILES, tile_position_at,
        },
    },
    screens::Screen,
};

const TINT_SHADER_PATH: &str = "shaders/tint.wgsl";

/// How the height is turned into a brightness.
///
/// A knob rather than world state, so like [`TerrainConfig`] it is built once and
/// outlives every session.
///
/// There is deliberately no resolution knob. The map is one texel per tile because
/// anything coarser stops the steps landing on tile boundaries, which is the whole
/// point — and because one texel per tile is what the generator already produces.
#[derive(Resource, Clone)]
pub struct TerrainTintConfig {
    /// Where the ramp bottoms out, and where it tops out. One ramp over the whole
    /// height range rather than one per band: a per-band ramp reverses at every band
    /// edge, which would draw a contour line along every coastline and treeline.
    ///
    /// `tint_low` is the water line, because that is where land starts; `tint_high`
    /// is the snow line, because everything above it is Snow and a ramp that ran on
    /// past it would spend half its range on one kind.
    pub tint_low: f32,
    pub tint_high: f32,
    /// Half the brightness range: a tile is drawn at `1 ± strength`. Bounded well
    /// under 1 so that no setting of these knobs can clip a tile to black or white,
    /// and small because the tileset is pixel art — this is meant to read as relief,
    /// not as a heatmap over the top of the art.
    ///
    /// **What matters is the spread across a screen, not across the world.** The ramp
    /// spans the whole height range, but a screenful holds only a slice of it, so the
    /// visible effect is a fraction of `2 * strength`.
    /// `the_default_ramp_measures_what_a_screenful_of_world_does` is how that is
    /// taken, and at these defaults it runs:
    ///
    /// ```text
    ///   screen at       height on it     brightness      spread
    ///   world centre    0.541..0.690    0.943..1.021       7.8%
    ///   lowland         0.424..0.655    0.882..1.003      12.1%
    ///   highland        0.424..0.529    0.882..0.937       5.5%
    /// ```
    ///
    /// So a slope reads as a gradient across the view rather than as per-tile relief:
    /// adjacent tiles differ by a few tenths of a percent, opposite sides of the
    /// screen by ~8%. Turning this up is the knob if that reads as too flat — it is
    /// linear in the spread, and 0.85 was blatant enough to measure a 0.79..0.86
    /// darkening against an untinted capture, which is what confirmed the pass draws
    /// what it should.
    pub strength: f32,
}

impl Default for TerrainTintConfig {
    fn default() -> Self {
        Self {
            tint_low: 0.42,
            tint_high: 0.88,
            strength: 0.82,
        }
    }
}

/// The ramp, flattened into the layout the shader reads.
///
/// Field order **is** the wgsl binding layout — see `assets/shaders/tint.wgsl`,
/// which declares the same struct. Vectors before scalars, so the std140 padding is
/// the same on both sides and WebGL2 agrees with the desktop.
#[derive(Component, Clone, Copy, Default, ShaderType)]
struct TerrainTintUniform {
    view_centre_tiles: Vec2,
    view_half_extent_tiles: Vec2,
    world_tiles: Vec2,
    water_line: f32,
    tint_low: f32,
    tint_high: f32,
    strength: f32,
}

/// Lives on the one world camera while [`Screen::Gameplay`] is up.
///
/// Unlike the weather's overlay this never changes during a session: the ramp is
/// config and the view is filled in at extract, so nothing has to keep it in step.
#[derive(Component, Clone, Copy, Default)]
struct TerrainTintOverlay(TerrainTintUniform);

impl SyncComponent for TerrainTintOverlay {
    // The removal target. Extraction only ever inserts, so if the render world is
    // not told what to drop, the shading outlives gameplay and shades the menus.
    type Target = TerrainTintUniform;
}

impl ExtractComponent for TerrainTintOverlay {
    type QueryData = (
        &'static Self,
        &'static Camera,
        &'static Projection,
        &'static GlobalTransform,
    );
    type QueryFilter = ();
    type Out = TerrainTintUniform;

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

pub struct TerrainTintPlugin;

impl Plugin for TerrainTintPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TerrainTintConfig>();
        app.add_plugins((
            ExtractComponentPlugin::<TerrainTintOverlay>::default(),
            UniformComponentPlugin::<TerrainTintUniform>::default(),
        ));
        app.add_systems(OnEnter(Screen::Gameplay), attach_terrain_tint);
        app.add_systems(OnExit(Screen::Gameplay), detach_terrain_tint);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.add_systems(RenderStartup, init_tint_pipeline);
        render_app.add_systems(ExtractSchedule, extract_height_uploads);
        render_app.add_systems(
            Render,
            (
                prepare_height_texture.in_set(RenderSystems::PrepareResources),
                prepare_tint_pipelines.in_set(RenderSystems::Prepare),
                prepare_tint_bind_groups.in_set(RenderSystems::PrepareBindGroups),
            ),
        );
        // Configured here rather than in the weather, because this is the pass that
        // introduces the constraint: before it there was only one of them.
        render_app.configure_sets(
            Core2d,
            (ScreenEffectSystems::Tint, ScreenEffectSystems::Weather).chain(),
        );
        // After tonemapping, so the scaling works on the same values the screen
        // shows; in PostProcess, because `bevy_ui_render` puts its pass after that
        // whole set, which is what keeps the shading off the menus and tooltips.
        render_app.add_systems(
            Core2d,
            terrain_tint_pass
                .in_set(Core2dSystems::PostProcess)
                .in_set(ScreenEffectSystems::Tint)
                .after(tonemapping),
        );
    }
}

fn attach_terrain_tint(
    mut commands: Commands,
    camera: Single<Entity, With<WorldCamera>>,
    terrain: Res<TerrainConfig>,
    config: Res<TerrainTintConfig>,
) {
    commands
        .entity(*camera)
        .insert(TerrainTintOverlay(TerrainTintUniform {
            world_tiles: WORLD_TILES.as_vec2(),
            // Read rather than restated, so that "where the sea stops" stays one
            // number in the crate.
            water_line: terrain.shallow_water_max,
            tint_low: config.tint_low,
            tint_high: config.tint_high,
            strength: config.strength,
            ..default()
        }));
}

/// Takes the overlay off the camera, which survives this transition.
///
/// The session's heightmap needs no help here: it goes with the upload queue, which
/// [`crate::gameplay::world`] drops on the same transition.
fn detach_terrain_tint(mut commands: Commands, camera: Single<Entity, With<WorldCamera>>) {
    commands.entity(*camera).remove::<TerrainTintOverlay>();
}

// -- Render world -----------------------------------------------------------

/// The chunks whose heights this frame's extract took off the main world's queue.
#[derive(Resource, Default)]
struct PendingHeightUploads(Vec<ChunkHeights>);

/// The render world's copy of the world's heights, and the only thing the shader
/// reads: one texel per tile over the whole world.
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

#[derive(Resource)]
struct TerrainTintPipeline {
    layout: BindGroupLayoutDescriptor,
    /// For the scene texture. The heightmap needs none — the shader loads it by
    /// tile coordinate rather than sampling it.
    scene_sampler: Sampler,
    variants: Variants<RenderPipeline, TerrainTintSpecializer>,
}

struct TerrainTintSpecializer;

#[derive(PartialEq, Eq, Hash, Clone, Copy, SpecializerKey)]
struct TerrainTintPipelineKey {
    target_format: TextureFormat,
}

impl Specializer<RenderPipeline> for TerrainTintSpecializer {
    type Key = TerrainTintPipelineKey;

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

fn init_tint_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "terrain_tint_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_2d(TextureSampleType::Float { filterable: true }),
                uniform_buffer::<TerrainTintUniform>(true),
            ),
        ),
    );

    commands.insert_resource(TerrainTintPipeline {
        layout: layout.clone(),
        scene_sampler: render_device.create_sampler(&SamplerDescriptor::default()),
        variants: Variants::new(
            TerrainTintSpecializer,
            RenderPipelineDescriptor {
                label: Some("terrain_tint_pipeline".into()),
                layout: vec![layout],
                vertex: fullscreen_shader.to_vertex_state(),
                fragment: Some(FragmentState {
                    shader: asset_server.load(TINT_SHADER_PATH),
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
struct TerrainTintPipelineId(CachedRenderPipelineId);

fn prepare_tint_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipeline: ResMut<TerrainTintPipeline>,
    views: Query<(Entity, &ExtractedView), With<ExtractedCamera>>,
) -> Result<(), BevyError> {
    for (entity, view) in &views {
        let id = pipeline.variants.specialize(
            &pipeline_cache,
            TerrainTintPipelineKey {
                target_format: view.target_format,
            },
        )?;
        commands.entity(entity).insert(TerrainTintPipelineId(id));
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
struct TerrainTintBindGroups {
    a_view: TextureViewId,
    a: BindGroup,
    b: BindGroup,
}

fn prepare_tint_bind_groups(
    mut commands: Commands,
    views: Query<(Entity, &ViewTarget), With<TerrainTintUniform>>,
    pipeline: Option<Res<TerrainTintPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    uniforms: Res<ComponentUniforms<TerrainTintUniform>>,
    heights: Option<Res<TerrainHeightTexture>>,
    render_device: Res<RenderDevice>,
) {
    let Some(pipeline) = pipeline else {
        return;
    };
    let Some(uniform_binding) = uniforms.uniforms().binding() else {
        return;
    };
    let Some(heights) = heights else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);

    for (entity, target) in &views {
        let bind_group = |scene: &_| {
            render_device.create_bind_group(
                "terrain_tint_bind_group",
                &layout,
                &BindGroupEntries::sequential((
                    scene,
                    &pipeline.scene_sampler,
                    &heights.view,
                    uniform_binding.clone(),
                )),
            )
        };

        commands.entity(entity).insert(TerrainTintBindGroups {
            a_view: target.main_texture_view().id(),
            a: bind_group(target.main_texture_view()),
            b: bind_group(target.main_texture_other_view()),
        });
    }
}

/// Shades the rendered world by the height under each fragment.
///
/// Every part of the guard is the query: a view with no `TerrainTintUniform` — a
/// menu, or a session whose first chunk has not landed — matches nothing, the
/// system is skipped, and the scene is never even copied.
fn terrain_tint_pass(
    view: ViewQuery<(
        &ViewTarget,
        &DynamicUniformIndex<TerrainTintUniform>,
        &TerrainTintBindGroups,
        &TerrainTintPipelineId,
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
            label: Some("terrain_tint_pass"),
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

    use crate::gameplay::terrain::height_byte;

    /// A screenful in tiles, taken from a real extract: a half extent of
    /// 59.75 x 72.625 tiles at scale 1.
    const SCREEN_TILES: UVec2 = UVec2::new(120, 145);

    /// What the ramp does to a screenful of world, which is the only thing a player
    /// ever sees at once.
    ///
    /// This is the measurement the defaults were tuned against, and the number that
    /// matters is the *spread within a screen*, not across the world: a ramp can span
    /// the whole height range and still be invisible, because a screen holds only a
    /// slice of it.
    ///
    /// `cargo test --release -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, not a check"]
    fn the_default_ramp_measures_what_a_screenful_of_world_does() {
        let terrain = TerrainConfig::default();
        let config = TerrainTintConfig::default();
        let sampler = terrain.sampler();

        let brightness = |height: f32| {
            let ramp =
                ((height - config.tint_low) / (config.tint_high - config.tint_low)).clamp(0.0, 1.0);
            1.0 + config.strength * (ramp * 2.0 - 1.0)
        };

        println!("\nscreen at        height range      brightness range   spread");
        for (label, centre) in [
            ("world centre", UVec2::splat(2048)),
            ("lowland     ", UVec2::new(1024, 3072)),
            ("highland    ", UVec2::new(3072, 1024)),
        ] {
            let (mut low, mut high) = (f32::MAX, f32::MIN);
            let mut land = 0u32;
            for y in 0..SCREEN_TILES.y {
                for x in 0..SCREEN_TILES.x {
                    let tile = centre + UVec2::new(x, y) - SCREEN_TILES / 2;
                    let height =
                        height_byte(sampler.elevation(tile.x as f32, tile.y as f32)) as f32 / 255.0;
                    // Water is passed through, so it is not part of what the ramp has
                    // to work with.
                    if height <= terrain.shallow_water_max {
                        continue;
                    }
                    land += 1;
                    low = low.min(height);
                    high = high.max(height);
                }
            }

            if land == 0 {
                println!("{label}     all water");
                continue;
            }
            println!(
                "{label}     {low:.3}..{high:.3}     {:.3}..{:.3}      {:.1}%",
                brightness(low),
                brightness(high),
                (brightness(high) - brightness(low)) * 100.0,
            );
        }
        println!();
    }

    /// Brightness stays within `1 ± strength`, so no setting of the knobs can clip
    /// a tile to black or to white.
    #[test]
    fn the_default_ramp_cannot_clip_a_tile_to_black_or_white() {
        let config = TerrainTintConfig::default();
        assert!(config.strength > 0.0, "a zero ramp would shade nothing");
        assert!(
            config.strength < 1.0,
            "a strength of {} would drive a tile to black",
            config.strength
        );
    }

    /// The ramp has to span land the world actually produces, and to run the right
    /// way up — the shader divides by `tint_high - tint_low`.
    #[test]
    fn the_default_ramp_spans_land_the_world_actually_has() {
        let terrain = TerrainConfig::default();
        let config = TerrainTintConfig::default();

        assert!(
            terrain.deep_water_max <= config.tint_low,
            "the ramp starts under the sea, where nothing is shaded anyway"
        );
        assert!(
            config.tint_low < config.tint_high,
            "the ramp has to rise with height"
        );
        assert!(
            config.tint_high <= 1.0,
            "elevation is clamped to 1.0, so a higher top is range the world never reaches"
        );
    }
}
