# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`wusel` — a Bevy 0.19 2D game: a menu leading into a procedurally generated 4096×4096-tile world you
pan around with WASD. Single binary crate, no workspace members.

## Commands

`just --list` is the entry point; every CI job maps to a recipe.

| | |
|---|---|
| `just run` / `just run-web` | run natively / in the browser (needs the `bevy` CLI) |
| `just test` | `cargo test --locked --workspace` |
| `just clippy` | Clippy over all targets/features on the `ci` profile |
| `just bevy-lints` | Bevy-specific lints (needs `bevy_lint`; install via `just bevy-lint-install`) |
| `just fmt` | `cargo fmt --check` |
| `just check-web` | wasm32 compile check with the `getrandom_backend="wasm_js"` cfg |
| `just all` | everything, in CI order |
| `just deps` | apt packages CI needs (alsa, udev, wayland headers) |

Single test: `cargo test --locked --workspace towns_are_never_closer_together_than_the_minimum_spacing`.
Use bare `cargo test` (not the other recipes' env) — `just test` deliberately sets no `RUSTFLAGS`,
while `fmt`/`docs`/`clippy`/`check-web` all export `-Zshare-generics=y -Zthreads=0`. Mixing the two
in one shell invalidates the build cache and triggers a full rebuild.

The toolchain is pinned nightly (`rust-toolchain.toml`) because of those `-Z` flags. `bevy_lint` runs
on a *different* nightly, pinned in `.github/workflows/ci.yaml` — see the comment there before bumping.

For faster local builds, copy `.cargo/config_fast_builds.toml` to `.cargo/config.toml` (gitignored).

## Architecture

### Plugin / state layout

`main.rs` adds `CameraPlugin` and `ScreenPlugin`. `ScreenPlugin` owns the `Screen` state
(`Main` / `Help` / `Gameplay`) and pulls in `MainScreenPlugin` and `GameplayPlugin`, which in turn
adds `gameplay::world::WorldPlugin`. Each module is a plugin; new features should follow that shape
rather than adding systems in `main.rs`.

Screen transitions are the only lifecycle mechanism: content is spawned in `OnEnter(Screen::X)` and
torn down by tagging entities `DespawnOnExit(Screen::X)`. There is no explicit cleanup system, so a
spawned entity missing that tag leaks across screens.

There is exactly **one** camera in the app (`camera.rs`, marked `WorldCamera`), spawned at `Startup`
and never despawned. The menus need it to render their UI and gameplay needs it to look at the world;
don't add a second one per screen. UI is laid out in screen space, so driving the camera around with
WASD moves the world without disturbing anything on top of it.

Because it outlives every screen, anything hung *on* the camera cannot use `DespawnOnExit` and has to
be removed by hand — `gameplay/weather.rs` adds its `WeatherOverlay` on entering gameplay and takes it
off on leaving, and the whole weather pass is gated on that component being there.

Zoom is the orthographic scale, held to powers of two between `MIN_ZOOM_SCALE` and `MAX_ZOOM_SCALE`
(0.25 to 4) by `+`/`-` or the wheel. Powers of two because the 8px tiles are drawn with nearest
filtering and anything else makes them crawl; bounded because chunk entities cover the screen, so
their count grows with the square of the scale. Anything that needs to know what is on screen —
the pan clamp, the chunk streamer — must go through `visible_half_extent`, since the viewport in
logical pixels only equals world units at scale 1.

### UI

Bevy Feathers (`FeathersPlugins`, `FeathersButton`, theme tokens) plus the `bsn!` scene macro for
declarative hierarchies (`main_screen::main_root`). The dark theme is built and extended with custom
`ThemeToken`s in `TooltipPlugin::build` — that plugin, not the main screen, owns `UiTheme`.

`tooltip.rs` renders a string into a row of `Text` spans, splitting on `' '`, `'.'`, `','`; any word
present in the `TooltipMap` resource becomes a clickable button that spawns a nested tooltip at the
cursor. `TooltipStack` is a stack of `(Entity, closable)`; Escape pops one closable tooltip, and only
returns to `Screen::Main` when none remain.

### Terrain generation (`gameplay/noise.rs`, `gameplay/terrain.rs`)

All noise is hand-rolled — `hash2` → `gradient_noise_2d` → `fbm`. No noise crate; keep it that way
unless there's a reason, since determinism across platforms is what the tests assert.

`generate_chunk(config, origin, chunk_size)` is a pure function of `(config, global tile position)` —
and of *that tile alone*. Every sample is taken in **global tile space** (`origin` is the chunk's
lower-left tile), never chunk-locally, and no rule reads a neighbouring tile, so a chunk needs no
padding and a tile cannot depend on where the boundary fell.
`a_tile_does_not_depend_on_where_the_chunk_boundary_falls` is the test that catches a regression.
The pipeline is two steps: sample the elevation and vegetation fields (independent through per-field
salts hashed into a domain offset, `NoiseField::new`, not separate generators), then `classify` maps
elevation to a band (deep water / shallow water / lowland / mountain) with vegetation only breaking
the Grass/Forest tie *within* the lowland band.

**Anything with a neighbourhood radius belongs in `gameplay/plan.rs`, not here.** A city is a disc, a
road spans hundreds of tiles and a river is decided uphill of where it runs; none of them fits in any
margin, which is why `Town`, `Road` and `River` are stamped over finished terrain rather than
generated. `the_terrain_never_produces_a_town_a_road_or_a_river` guards it.

Everything is driven by the `TerrainConfig` resource — thresholds, scales, seed. Changing a default
there will break `the_default_config_produces_every_base_kind`, which is the point: it guards against
a config that quietly yields a single-biome world. The settlement figures live there but are read only
by `gameplay/city.rs`, and the humidity ones by `gameplay/river.rs` and `gameplay/weather.rs` — the
rivers rise where it rains and the clouds are drawn from the same field, which is why they agree.

**The world is fine-grained, and it constrains what can be built on it.** `elevation_scale` is 0.04,
so the *longest* wavelength in the elevation field is ~25 tiles: there are no continents and no
valleys, just a mottled archipelago where 32% of tiles are water and no point on land is far from a
shore. Cities and roads work at 200+ tiles and ride over that happily. Rivers do not — see below.

This is the only expensive call in the crate: **~4 ms per 64×64 chunk** in release. Treat it as
something to keep off the main thread.

### Rivers, cities and roads (`gameplay/plan.rs`, `river.rs`, `city.rs`, `road.rs`)

Once `WorldMap` is complete, `WorldPlan` walks one session through
`WaitingForTerrain → Rivers → Cities → Roads → Done`, editing tiles under the plan while the player is
already walking around. The in-flight tasks live *inside* the enum, so dropping the resource on leaving
gameplay cancels them — a route planned for one world can never land in the next.

The stage order is load-bearing: each stamps into `WorldMap` and the next takes its snapshot
*afterwards*, so a city is clipped by a river the way it is clipped by a coast, and a road sees both.
That is also why there is no `start_city_plan` — `apply_river_plan` opens the city stage itself, since
only it knows when the last river tile is down.

- **Rivers** — one spring per `river_source_cell_tiles` square, kept if the tile is `Mountain` and the
  **humidity** field clears `river_source_threshold`. Each spring is a particle walking downhill on a
  lattice **anchored on the world origin** (same trick as roads, same reason: two particles that pass
  through a place step between the same nodes, so paths coincide and flow accumulates). A tile's
  channel width is its flow, capped at `MAX_RIVER_WIDTH` = 4.
- **Cities** — one candidate per `region_size_tiles` square, jittered by hash, kept if habitable and
  clearing `town_threshold`. Size tier from how far it clears; the outline is a disc whose radius
  wobbles over three hashed harmonics, clipped to habitable tiles. `City` is a component; `CityMap`
  only indexes those entities by chunk.
- **Roads** — pairs from the Gabriel graph, then an angular prune so no city gets two roads leaving
  within `road_min_separation_degrees`. Routed by A* on a lattice **anchored on the world origin, not
  on the city**: that is the only reason two roads lay down the same tiles and can therefore merge.
  A step on existing road costs `road_reuse_discount ×` its price, and the search heuristic is scaled
  by the same factor or it refuses the detour that reuse exists to buy.

Roads are routed **one at a time**, with the snapshot retaken after each — a route can only reuse what
is already on the ground. The order is therefore part of the result and is fixed: longest first, so
long routes become trunks. Both ends of a route enter the lattice at the nearest *reachable* node,
which is not always the nearest one.

A river is **not** `is_water()`. Folding it in would cut the continent into pieces the road network
cannot span, so a route crosses one for `road_river_crossing_penalty` per tile and lays plain `Road`
over it. Because the crossing is now road, the next route finds road rather than river and pays the
reuse discount instead — which is what makes roads converge on the same bridges. Lakes *are*
`ShallowWater`, so roads go round them and lakeside cities get the coast bonus, both without anything
learning what a lake is.

Two things about rivers are worth knowing before tuning them:

- **A particle never steps uphill; it floods.** With nowhere lower to go it fills the basin by
  priority-flood, and the filled nodes are then **raised to the level they filled to** — a full basin
  is a flat sheet of water. Without that raise the particle spills to the rim and, on its very next
  step, walks straight back into the hollow it just filled; that bug capped every river in the world
  at two steps. Every filled basin is recorded so the next particle can cross it; only ones over
  `river_lake_min_tiles` are *drawn*, or each river becomes a string of beads.
- **The terrain cannot feed the width machinery.** With a 25-tile elevation wavelength, a descent
  meets water in a few steps, so descents rarely meet and the busiest segment in the world carries 3
  particles. `river_flow_per_width` is 2 for that reason and channels wider than 2 essentially do not
  occur. Real trunk rivers would need a low-frequency component in the elevation field — a terrain
  change that moves every existing tile, not a river knob.

Config defaults in `WorldPlanConfig` carry their measurements in the doc comments; the
`#[ignore]`d `the_default_config_lays_out_cities_of_every_size_and_roads_between_them` in `plan.rs`
generates the whole world in ~2 s and is how those numbers were taken —
`cargo test --release -- --ignored --nocapture`.

### The world (`gameplay/world.rs`)

A fixed 64×64 grid of chunks — 4096×4096 tiles — centred on the world origin, so it has a hard edge
you can reach and world-space coordinates stay small enough for f32.

Two different things are "loaded", and keeping them separate is the whole design:

- **Tile data** (`WorldMap`) covers the entire world and lasts the session. One `TerrainKind` byte per
  tile, so all 4096 chunks cost ~16 MB. A background pass on `AsyncComputeTaskPool` fills it in,
  nearest-to-world-centre first. Chunks sit behind an `Arc` so `WorldMap::snapshot` can hand the
  finished world to a planning task without copying it.
- **Chunk entities** exist only within `resident_radius` chunks of the camera. Each costs a mesh, a
  material and a per-chunk index image, which is why all 4096 cannot be resident — that was measured
  at ~400 MB and ~17 s of generation. Residency is derived from the entities' own `ChunkCoord` rather
  than a parallel resource, so the two cannot disagree.

**Nothing here outlives `Screen::Gameplay`.** `start_world` builds every world resource from nothing on
entering and `tear_down_world` removes them on leaving, so a session never inherits a half-generated
map or a task in flight. The two configs are the exception — they are knobs, not world state, which
means a new session rebuilds the *same* world unless the seed is rerolled.

If the camera outruns the background pass, `refresh_resident_chunks` generates what it needs on the
main thread, capped at `MAX_BLOCKING_GENERATIONS_PER_FRAME`, and builds at most
`MAX_CHUNK_SPAWNS_PER_FRAME` entities — a zoom step can bring hundreds of chunks into view at once.
Anything over either budget appears a frame or two later. `OnEnter(Screen::Gameplay)` passes unlimited
budgets so the first frame is complete.

`WorldSystems` orders the frame `Streaming → Planning → Refresh`, which is what puts the plan's tile
edits between the streamer that spawns chunk entities and `refresh_edited_chunks` that rebuilds the
stale ones — so an edit is visible in the frame it lands. The river stage leans on that: it stamps
`river_chunks_stamped_per_frame` chunks a frame (~44 frames for the default world) rather than
~38k edits at once, because `apply_edits` scans its touched-chunk list linearly. A chunk that is *not* resident needs no
refresh at all: the edit went into `WorldMap`, so it is there when the chunk is next spawned.

Three coordinate spaces are in play and the helpers at the top of the module are the only sanctioned
way between them: chunk coordinates (`0..WORLD_CHUNKS`), global tile coordinates (what the noise is
sampled in), and world space in pixels. `tile_position_at` is the odd one out and the weather overlay
is why: it is the only conversion that keeps its fraction, since a screen pixel falls *between* tiles.

### Weather (`gameplay/weather.rs`, `assets/shaders/weather.wgsl`)

Cloud patches drifting over the world, the shadow each throws, and rain in the thick of them. It is
**cosmetic**: nothing here reads or writes `WorldMap`, so no tile can depend on the weather.

Two baked textures and one full-screen pass, and the split is the design:

- **Where** it is cloudy is `TerrainConfig::humidity_field()` — the same field the river springs read
  — sampled over the whole world at one texel per 8 tiles. That map never moves, so a wet range is
  reliably overcast and a dry one reliably clear.
- **What** a cloud looks like is a second map holding one *tiling* period of the same noise
  (`noise::TilingNoiseField`, whose seam `a_tiling_field_matches_itself_across_the_seam` guards). The
  shader scrolls it at two scales and two speeds. That scrolling is the entire animation.

So **the shader evaluates no noise**, and that is load-bearing twice over: an fbm per fragment, needed
twice for cloud plus shadow, measures ~4 ms at 1080p and the whole frame at 4K on an iGPU; and it would
put a second noise implementation in a crate whose determinism tests rest on there being one. Both maps
are baked on `AsyncComputeTaskPool` (~50 ms) rather than in `OnEnter`, which already generates the
first screenful of chunks unbudgeted. Until they land there is no `WeatherMaps` and the sky is clear —
absence is the fallback, and it is deliberately clear rather than overcast.

The field is anchored in **world** space. The view centre and half extent are filled in during
`ExtractComponent`, from the camera itself, after the whole main-world frame — not by a system that
races `camera.rs`'s pan. A frame of slip there would shear the sky 34 px against the terrain at
`MIN_ZOOM_SCALE` for as long as you held a key, so it is worth knowing that no ordering protects this:
the extract point does.

Two things about the composite:

- **A shadow is the cloud field one constant offset away**, so every cloud has exactly one — no second
  field, nothing to keep in step. Within `shadow_offset_tiles` of the viewport border the cloud casting
  a visible shadow is off screen, which is a property of the trick rather than a bug.
- **Rain is cut on the raw field and multiplied by the density.** Cutting on the *density* instead
  looks natural and is wrong: the density saturates, so nearly every cloud clears any cut placed on it
  and it rained on 28.6% of the world at once. The multiply is what makes "no rain from a clear sky"
  true for any setting of the knobs rather than only for ones whose cuts are ordered.

`WeatherConfig` is a knob, so like `TerrainConfig` it outlives a session; `WeatherMaps`, `WeatherClock`
and `WeatherBake` are world state and go on `OnExit`. The clock's offsets are wrapped to a map period
rather than accumulated — unwrapped they quantize the streak phase after a few hours.

Defaults carry their measurements, taken with the `#[ignore]`d `the_default_config_measures_the_sky`
(`cargo test --release -- --ignored --nocapture`).

### Weather shader coupling

`WeatherUniform` in `weather.rs` and `WeatherUniform` in `weather.wgsl` are the same struct written
twice: **field order is the binding layout**. Vectors are declared before scalars so std140 padding
agrees on both sides, including under WebGL2. The shader is loaded by path at runtime, so a mismatch is
a shader-compile failure when you enter gameplay, not a build error — and `just check-web` only
type-checks Rust, so it will not catch a wgsl construct the web backend rejects. Adding a knob means
touching the Rust struct, the wgsl struct, `sync_weather_overlay` and the config doc comment together.

The pass is registered `.in_set(Core2dSystems::PostProcess).after(tonemapping)`. Bevy 0.19 has **no
node-based render graph** for Core2d — no `Node2d`, no `bevy_render::render_graph` — so a post-process
effect is an ordinary system in the `Core2d` schedule taking `ViewQuery` and `RenderContext`;
`bevy_core_pipeline::fullscreen_material` is the in-tree template it was written from (it cannot be
used directly, since its bind group layout is fixed at three entries and cannot carry the two maps).
That placement is what keeps weather off the UI: `bevy_ui_render` orders `ui_pass` *after* the whole
`PostProcess` set.

### Tileset coupling

`TerrainKind`'s discriminant *is* the tileset index *is* the atlas column, so the enum and
`assets/textures/terrain.png` cannot drift. The atlas is a horizontal strip of 8×8 tiles loaded once
(`world::load_tileset`) as an array texture via
`ImageArrayLayout::GridCount { columns: TERRAIN_KIND_COUNT, rows: 1 }`; every chunk entity shares
that one handle.

Adding a terrain kind means: append a column to the PNG, add the enum variant with the matching
discriminant, bump `TERRAIN_KIND_COUNT`, and extend `the_default_config_produces_every_base_kind` —
unless, like `Town` and `Road`, the kind is stamped by the plan rather than generated, in which case
that test must keep *not* seeing it.

Tiles are drawn at their native 8px size with `ImagePlugin::default_nearest`; upscaling
`tile_display_size` would resample the pixel art. `terrain.atlas.json` is sidecar metadata from the
art tool (palette/ramps) and is not read by the game.

## Conventions

Comments here explain *why* a value or structure was chosen (the fbm gain constant, the chunk margin,
the bevy_lint pin), not what the code does. Match that. Tests are named as behavioural sentences.

`Cargo.toml` allows `clippy::too_many_arguments` and `clippy::type_complexity` globally — normal for
Bevy systems, so don't work around them.
