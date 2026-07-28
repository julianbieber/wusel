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

`generate_chunk(config, origin, chunk_size)` is a pure function of `(config, global tile position)`.
Every sample is taken in **global tile space** — `origin` is the chunk's lower-left tile — never
chunk-locally. That is what lets the world be cut into chunks at all. The pipeline, in order:

1. `TerrainSamples::generate` samples three independent fbm fields (elevation, vegetation,
   settlement). Independence comes from per-field salts hashed into a domain offset (`NoiseField::new`),
   not from separate generators.
2. The grid is padded by `margin = town_min_spacing + coast_radius` on every side. This is the load-
   bearing detail: without it, a tile's kind would depend on where the chunk boundary fell. Any new
   rule with a neighbourhood radius must be included in that margin, or chunks will disagree along
   their seams. `a_tile_does_not_depend_on_where_the_chunk_boundary_falls` is the test that catches it.
3. `classify` maps elevation to a band (deep water / shallow water / lowland / mountain); vegetation
   only breaks the tie between Grass and Forest *within* the lowland band.
4. `is_town` promotes a habitable tile to Town when its settlement score clears the threshold and is
   the strict local maximum over `town_min_spacing`, which scatters towns as isolated points.

Everything is driven by the `TerrainConfig` resource — thresholds, scales, seed. Changing a default
there will break `the_default_config_produces_every_kind`, which is the point: it guards against a
config that quietly yields a single-biome world.

This is the only expensive call in the crate: **~4 ms per 64×64 chunk** in release. Treat it as
something to keep off the main thread.

### The world (`gameplay/world.rs`)

A fixed 64×64 grid of chunks — 4096×4096 tiles — centred on the world origin, so it has a hard edge
you can reach and world-space coordinates stay small enough for f32.

Two different things are "loaded", and keeping them separate is the whole design:

- **Tile data** (`WorldMap`) covers the entire world and is kept forever. One `TerrainKind` byte per
  tile, so all 4096 chunks cost ~16 MB. A background pass on `AsyncComputeTaskPool` fills it in,
  nearest-to-world-centre first. It runs during the menus too, so the world is largely generated by
  the time Play is pressed.
- **Chunk entities** exist only within `RESIDENT_RADIUS` chunks of the camera. Each costs a mesh, a
  material and a per-chunk index image, which is why all 4096 cannot be resident — that was measured
  at ~400 MB and ~17 s of generation. Residency is derived from the entities' own `ChunkCoord` rather
  than a parallel resource, so the two cannot disagree.

If the camera outruns the background pass, `refresh_resident_chunks` generates what it needs on the
main thread, capped at `MAX_BLOCKING_GENERATIONS_PER_FRAME`; anything over budget appears a frame or
two later. `OnEnter(Screen::Gameplay)` passes an unlimited budget so the first frame is complete.

Three coordinate spaces are in play and the helpers at the top of the module are the only sanctioned
way between them: chunk coordinates (`0..WORLD_CHUNKS`), global tile coordinates (what the noise is
sampled in), and world space in pixels.

### Tileset coupling

`TerrainKind`'s discriminant *is* the tileset index *is* the atlas column, so the enum and
`assets/textures/terrain.png` cannot drift. The atlas is a horizontal strip of 8×8 tiles loaded once
(`world::load_tileset`) as an array texture via
`ImageArrayLayout::GridCount { columns: TERRAIN_KIND_COUNT, rows: 1 }`; every chunk entity shares
that one handle.

Adding a terrain kind means: append a column to the PNG, add the enum variant with the matching
discriminant, bump `TERRAIN_KIND_COUNT`, and extend `the_default_config_produces_every_kind`.

Tiles are drawn at their native 8px size with `ImagePlugin::default_nearest`; upscaling
`tile_display_size` would resample the pixel art. `terrain.atlas.json` is sidecar metadata from the
art tool (palette/ramps) and is not read by the game.

## Conventions

Comments here explain *why* a value or structure was chosen (the fbm gain constant, the chunk margin,
the bevy_lint pin), not what the code does. Match that. Tests are named as behavioural sentences.

`Cargo.toml` allows `clippy::too_many_arguments` and `clippy::type_complexity` globally — normal for
Bevy systems, so don't work around them.
