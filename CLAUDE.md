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

### Terrain generation (`gameplay/noise.rs`, `gameplay/terrain.rs`, `gameplay/biome.rs`)

All noise is hand-rolled — `hash2` → `gradient_noise_2d` → `fbm` / `ridged_fbm`. No noise crate; keep
it that way unless there's a reason, since determinism across platforms is what the tests assert.

`generate_chunk(config, origin, chunk_size)` is a pure function of `(config, global tile position)` —
and of *that tile alone*. Every sample is taken in **global tile space** (`origin` is the chunk's
lower-left tile), never chunk-locally, and no rule reads a neighbouring **tile**, so a chunk needs no
padding and a tile cannot depend on where the boundary fell. The biome lookup reads neighbouring
*cells*, which are a function of their own integer coordinates, so that property survives it.
`a_tile_does_not_depend_on_where_the_chunk_boundary_falls` is the test that catches a regression, and
it checks both halves of the `ChunkTerrain` the call returns — the kinds and, since the tint, the
height each kind was cut from.

**The height is built in layers, and which layers apply is a function of where you are.** One field
was the original defect: `relief_scale` is 0.04, so its longest wavelength is ~25 tiles, and no
threshold can cut regions out of a field with no structure at the scale a player moves at. Three
layers now, combined per tile:

- **continent** (`continent_scale` 0.0015, ~670-tile wavelength) — what makes land masses. Displaces
  a recipe's base height by ±`continent_relief`.
- **relief** (the old `elevation_scale`, unchanged at 0.04) — demoted to the bumps on top. As
  *relief* a 25-tile wavelength was always right; it was only wrong as the whole landscape.
- **ridged** (`ridge_scale`, `noise::RidgedNoiseField`) — one-sided and crease-shaped, which is what
  makes a mountain *range* instead of a field of lumps. Only `Biome::Highland` weights it, so it is
  skipped where the blended weight is ~0 — most of the world, and that skip is what pays for the
  extra layers.

`TerrainSampler` is the **only** implementation of "how high, how green, how wet is it here", and
that is load-bearing outside this module: `river.rs` walks its particles downhill against it and
`road.rs` costs its steps by it. `TerrainConfig` used to hand out a bare `NoiseField` via
`elevation_field()`/`humidity_field()`; it hands out a sampler instead, because two implementations
would mean rivers running up the visible hills. `settlement_field()` is the one field still raw —
`city.rs` compares scores between sites and nothing biome-dependent enters into it.

Then `classify` cuts the result into bands: deep water / shallow water / **sand** / lowland /
mountain / **rock** / **snow**. What the biome changes is what *fills* a band — the lowland triple comes
from `Biome::kinds()` rather than being Grass and Forest everywhere, and the sand band above the water
line is as wide as the recipe says, which is zero for a `Highland` coast and so gives a cliff instead
of a beach.

**Three more layers make the lowland band read the height instead of discarding it** (gh-14). The band
spans 0.42–0.72, 30% of the height scale, and used to collapse all of it into one vegetation test at an
11-tile wavelength — which is why the mountains, where the height *does* pick the kind, were the only
part of the world that read as landscape. Now:

- **soil** — how much loose material sits on the bedrock, from the *slope* of the height the tile was
  already going to be cut with. Below `bedrock_max` the biome gets no say and the rock shows through.
- **hardness** (`lithology_*`) — a strongly anisotropic field on one world strike, giving hard/soft
  bands 40–160 tiles wide. Its value is that it is **uncorrelated with the biome Voronoi**: a second
  partition cutting across the first, which is what stops a region's interior being self-similar.
  `hardness_is_uncorrelated_with_the_biome_map` is the guard.
- **dune** (`dune_*`) — transverse aeolian crests, weighted per recipe so only `Desert` pays for the
  sample, on the same epsilon skip `ridge` earns. **Dunes reach the tileset through soil and vegetation,
  never through a rule**: a crest is deep loose material nothing grows on, so it climbs a rung to Sand,
  and the deflated trough beside it falls to Gravel. No rule anywhere names `Desert`.

All three also displace the height a little, which is nearly free and deliberate — the tint pass shades
the world by the stored per-tile height, so a hogback ridge and a dune field get their shading with
nothing new drawn.

**The gradient must not cost a second biome lookup.** The expensive part of a height sample is `blend()`,
not the height; the recipe is constant outside a 48-tile band, so `soil` re-uses *this tile's own* recipe
and samples only relief and ridge. Getting that wrong triples the per-tile cost instead of doubling it.

`soil_vegetation_gain` is the knob that made this work at all. Letting soil decide only whether bedrock
breaks through leaves every tile that *has* soil — most of the world, and nearly all of `Plains` —
picking its kind from vegetation alone exactly as before.

### Biomes (`gameplay/biome.rs`)

A **jittered-grid Voronoi**: each cell of a 384-tile lattice hashes (once — jitter from the low bytes,
biome draw from the high ones) to a site inside itself and to a `Biome`, and a tile belongs to the
nearest site of the 3x3 cells around it. Jitter is confined to less than a cell, which is what makes
3x3 sufficient. The query position is warped by `biome_warp_tiles` first, so a region has an organic
outline rather than a polygon's. Voronoi rather than a latitude/humidity climate table on purpose: a
climate table gives smooth gradients, and gradients are what this world already had too much of.

**The blend is the feature, not the smoothing.** A biome carries a `HeightRecipe` — numbers only — and
the recipe at a tile is the distance-weighted mix of the nearby sites' recipes. That is where boundary
structure comes from: `Highland`·`Ocean` falls through sea level across the band and gives a cliffed
shelf, `Highland`·`Plains` decays the ridge weight and gives foothills, `Plains`·`Plains` does nothing
because a blend of like recipes is that recipe. **No rule anywhere names a pair of biomes** — six
biomes would be fifteen pairs to tune, and the interesting boundaries are the ones nobody enumerated.

**Two things fight the fact that a Voronoi edge is a straight line**, because on its own the diagram
looks ruled and that is the first thing you notice from the ground:

- **The domain warp has to be shorter than the edge it bends.** A warp only bends a boundary where it
  has content at a wavelength *shorter* than that boundary; a single long octave translates the whole
  edge bodily and leaves it exactly as straight. `WARP_CELLS` is therefore 0.75 of a cell with 3
  octaves — the first attempt was 1.5 cells with 2, whose finest detail had a 288-tile wavelength
  against ~200-tile edges, and it did nothing at all. Amplitude then buys crookedness against region
  size: `biome_warp_tiles` 160 gives a 1.46× longer outline with regions still ~257 tiles across, and
  `the_warp_trades_region_size_for_a_crooked_outline` has the curve out to 440. Note the *interior
  fraction is useless as a guard here* — it sits at ~56% across that whole range, because warping the
  query is locally structure-preserving; mean region run length is the metric that moves.
- **The kind pair is dithered, not switched.** Even a perfectly wiggly boundary still flips Sand to
  Grass along a line. So `BlendedBiome::cover` picks the biome supplying the tile's kinds by hashing
  the tile and drawing against the blend weights, rather than taking the heaviest. In a region
  interior one weight is 1.0 and this is exactly `dominant`, so nothing is speckled; inside the band
  the two biomes' tiles interleave, dense on their own side and sparse on the other, and the line
  stops existing. ~10.5% of the world takes its cover from a neighbouring region.
  `the_cover_dither_is_a_no_op_inside_a_region` is the guard that matters — without it every region
  would be flecked with tiles from biomes nowhere near it.

Two more details hold the shape together:

- **Weights are banded, not inverse-distance.** A site contributes only while its distance exceeds
  the nearest site's by less than `2 * biome_blend_tiles`, so beyond `biome_blend_tiles` from a
  boundary exactly one weight is non-zero and a region has an *interior*. Without that the world
  would be an average of all six everywhere. `biome_blend_tiles` must stay well under the cell's
  inradius (~192): at 96 only a quarter of the world was interior, which is why it is 48.
- **`HeightRecipe` is numbers only; the kind triple lives on `Biome::kinds()`.** An enum cannot be
  averaged, so the triple comes from a single biome — `cover`, per the dither above. It is therefore the
  one thing about a tile that changes all at once, and deliberately the one thing elevation does *not*
  depend on, which is what lets a boundary read as a treeline rather than a wall. `dominant` (the
  argmax) is the separate question "which region is this", and only the measurements ask it.
- **A triple, not a pair** (gh-14). Two rungs meant a `vegetation_bias` did not move a region along a
  ladder, it pinned the region to one end: `Desert`'s -0.30 put 92.5% of its tiles below the single cut
  and made it 80% Sand. `Scrub` is the tile that made three rungs possible, and five of the six biomes
  use it. `no_region_is_built_out_of_one_or_two_kinds` is the guard, and it is the one test that would
  have caught the world gh-14 was opened about — everything else asks whether a kind exists *somewhere*,
  and nothing asked whether a region was made of only one.

Six biomes, because the drawn palette supports six that look different: `Ocean`, `Plains`, `Forest`,
`Highland`, `Desert`, `Wetland`. Tundra is absent — with no cold-grass or conifer tile the only thing
it could lay down below the snow line is `Snow`, a flat sheet of one kind, which is the defect this
replaced in white. Adding it is one tile and one table row.

`Desert`'s `humidity_bias` is why it has neither springs nor clouds, and `Wetland`'s is why it has
both: biome coherence across rivers and weather falls out of the shared sampler, with neither of those
modules learning what a biome is.

**Anything with a neighbourhood radius belongs in `gameplay/plan.rs`, not here.** A city is a disc, a
road spans hundreds of tiles and a river is decided uphill of where it runs; none of them fits in any
margin, which is why `Town`, `Road` and `River` are stamped over finished terrain rather than
generated. `the_terrain_never_produces_a_town_a_road_or_a_river` guards it.

Everything is driven by the `TerrainConfig` resource — thresholds, scales, seed. Changing a default
there will break `the_default_config_produces_every_base_kind`, which is the point: it guards against
a config that quietly yields a single-biome world. The settlement figures live there but are read only
by `gameplay/city.rs`, and the humidity ones by `gameplay/river.rs` and `gameplay/weather.rs` — the
rivers rise where it rains and the clouds are drawn from the same field, which is why they agree.

Everything is driven by `TerrainConfig`; the biome recipe table lives in `biome.rs` and carries its
measured coverage in `the_default_config_produces_recognisably_different_regions` (`cargo test
--release -- --ignored --nocapture`, and run it *alone* — three measurement tests in parallel contend
for CPU and inflate the per-chunk figure). At the defaults the world is 33.3% water, and every added
tile earns its column: Scrub 12.8%, Sand 9.5%, Rock 7.4%, Gravel 5.2%, Snow 2.8%, Marsh 1.9%, Reed 1.8%.
Biome coverage is unmoved by gh-14 — Ocean 30.6%, Forest 21.4%, Plains 15.9%, Highland 13.9%,
Desert 11.9%, Wetland 6.4% — because the substrate layers change what *fills* a region, not where the
regions are.

What a region is made of, which is what gh-14 was about: Plains runs Scrub 34 / Grass 23 / Forest 20 /
Sand 13, Desert Sand 39 / Gravel 30 / Scrub 17, Wetland Marsh 26 / Reed 25 / Forest 23. Before the
change Desert was **80% Sand** and Plains 53% Grass. Mean run length went *up* (5.9–7.9 tiles against
4–6.5), so the patches are the same size or bigger — they are simply made of four kinds instead of two.

**Measure structure as excess agreement over chance, never raw agreement.** Two tiles of a one-kind
region agree 100% of the time while carrying no structure at all, so raw agreement rewards exactly the
monotony it is supposed to detect — the first pass at gh-14 misdiagnosed a "spectral gap" from an
unnormalised table, and the plateau turned out to be the chance floor of an 80%-Sand desert.
`the_default_config_measures_the_structure_gap` prints the normalised curve for the shipped world
against one with the layers switched off.

This is the only expensive call in the crate, and it keeps getting dearer: 1.75 ms per 64×64 chunk
originally, 3.70 ms after the biome rework, **6.58 ms** after gh-14's substrate layers — so the
whole-world background pass is ~60 s. `MAX_BLOCKING_GENERATIONS_PER_FRAME` is 1 and there is no room to
raise it; one chunk is already most of a 60 fps frame. Treat it as something to keep off the main thread.

### Rivers, cities and roads (`gameplay/plan.rs`, `river.rs`, `city.rs`, `road.rs`)

Once `WorldMap` is complete, `WorldPlan` walks one session through
`WaitingForTerrain → Rivers → Drainage → Cities → Roads → Done`, editing tiles under the plan while the
player is already walking around. The in-flight tasks live *inside* the enum, so dropping the resource on leaving
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
- **Dry valleys** (`drainage.rs`) — the branching network the land drains through, drawn as a change of
  ground cover rather than as water: a wadi through desert sand, a gallery treeline down a lowland
  valley, reed along a marsh channel. **A drainage particle never floods** — that one rule is the whole
  design. It removes the expensive half of the river stage and makes it impossible for this pass to add
  a tile of standing water, which is what lets it land *before* the gh-9 lake retune. It paints cover
  only, so no drainage edit costs a heightmap upload. `dampened` is the moisture ladder, and everything
  not named in it is a fixed point — that is how "may not touch the water, the mountain bands, or the
  plan's own kinds" is enforced, by omission rather than by a list that could fall out of step.
  0.199% of the world, against roads at 0.24%.

  Note the lattice is **16 tiles, four times the river's**, and that is what makes the stage work at
  all: with no flood, a particle stops at the first node with nothing lower beside it, and at a 4-tile
  stride the relief layer's fine octaves put a local minimum every few nodes. The first cut laid 404
  tiles in the entire world. The pits are a property of the sampling scale, not of the landscape.
- **Cities** — one candidate per `region_size_tiles` square, jittered by hash, kept if habitable and
  clearing `town_threshold`. Size tier from how far it clears; the outline is a disc whose radius
  wobbles over three hashed harmonics, clipped to habitable tiles. `City` is a component; `CityMap`
  only indexes those entities by chunk.

  `is_habitable` is Forest, Grass and **Scrub**. Scrub being habitable is a decision, not an oversight:
  it is the bare rung of the `Plains` and `Forest` ladders, so excluding it would have cut city sites
  out of ordinary grassland. The visible consequence is at the other end — a wadi promotes desert Sand
  to Scrub, so towns appear strung along desert drainage lines, which is where real ones are. Everything
  else the reworks added (Sand, Marsh, Rock, Snow, Gravel, Reed) stays uninhabitable, which is still
  most of what makes a desert or a marsh feel different to walk into. Breaking up the uniform grassland
  cost 5 cities of 96 and 5% of the town tiles — measured, and small enough to be worth it.
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

**That bridge mechanism is barely exercised.** Since the biome rework the default world routes 87
roads and bridges a river just **2** times, so the reuse-on-a-crossing path is very nearly live code
with no coverage from the measurement run. The cause is in the terrain, not the router: see the lake note
below.

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
- **The low-frequency component arrived, and it fed the lakes instead.** The biome rework added the
  continent layer the note above asked for, and the result was not trunk rivers: broad low-frequency
  minima are broad *basins*, so the priority-flood has far more to fill. The default world went from
  ~38k river-stage edits to 264k — **26.6k tiles of river against 238k of lake**, roughly 1.4% of the
  world under inland water. Because a lake is `ShallowWater` and a hard barrier, routes mostly go round
  the lakes rather than over the rivers, which is why bridging fell to 2 crossings in the whole world
  (it was 0 before the warp retune moved the regions around). Raising
  `river_lake_min_tiles` will not help: these basins are large, not marginal. It is a river-stage
  retune against the new terrain, and it belongs with gh-9 rather than in a heightmap change. Note the
  count is insensitive to `Wetland`'s flatness — that was tried, and the lake total did not move by a
  single tile.

Config defaults in `WorldPlanConfig` carry their measurements in the doc comments; the
`#[ignore]`d `the_default_config_lays_out_cities_of_every_size_and_roads_between_them` in `plan.rs`
generates the whole world in ~2 s and is how those numbers were taken —
`cargo test --release -- --ignored --nocapture`.

### The world (`gameplay/world.rs`)

A fixed 64×64 grid of chunks — 4096×4096 tiles — centred on the world origin, so it has a hard edge
you can reach and world-space coordinates stay small enough for f32.

Two different things are "loaded", and keeping them separate is the whole design:

- **Tile data** (`WorldMap`) covers the entire world and lasts the session. One `TerrainKind` byte per
  tile, so all 4096 chunks cost ~16 MB — and, since the tint, one height byte beside it for another
  ~16 MB. A background pass on `AsyncComputeTaskPool` fills it in, nearest-to-world-centre first.
  Chunks sit behind an `Arc` so `WorldMap::snapshot` can hand the finished world to a planning task
  without copying it, and so queueing a chunk's heights for the GPU copies no tile data either.
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

- **Where** it is cloudy is `TerrainSampler::humidity()` — the same answer the river springs read
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

There are **two** passes in `PostProcess` now, and both ping-pong the same `ViewTarget`, so the second
reads what the first wrote and the order is part of the result. It is stated in the `ScreenEffectSystems` set
in `gameplay/mod.rs` — `Tint` then `Weather` — rather than with `.before(weather_pass)`, because a
system is only usable as an ordering label where its parameter types are visible.

### Terrain tint (`gameplay/tint.rs`, `assets/shaders/tint.wgsl`)

Scales the rendered world's brightness by the height of the tile under each fragment, so a slope reads
as a slope *inside* a kind's band rather than only where it crosses a `classify` edge. Cosmetic, like
the weather: nothing here touches `WorldMap`.

**The height is kept, not re-baked.** `classify` already computes every tile's elevation and used to
drop it, so `generate_chunk` returns a `ChunkTerrain { kinds, heights }` and `WorldMap` stores both.
Sampling the world a second time measured ~9 core-seconds against the ~34 s the chunks themselves cost,
and would have put a second answer to "how high is it here" in a crate whose determinism tests rest on
there being one. The cost is memory: 16 MB of heights beside the 16 MB of kinds, and 16 MB again on the
GPU. `every_tile_keeps_the_height_it_was_classified_from` is the guard.

**The map is not an `Image` asset**, because Bevy re-uploads a whole `Image` on any change and 16 MB
per landed chunk is not a thing to do 4096 times. `tint.rs` owns a raw `WORLD_TILES` R8 texture and
writes one chunk's 4 KB rect into it. `HeightUploadQueue` lives in `world.rs` beside `DirtyChunks`, and
`WorldMap::insert` takes it as an argument — there is no way to store a chunk without queueing it, so
"a chunk reaches the texture exactly once" is a property of the signature. The extract *moves* the
queue across (`ResMut<MainWorld>`, not `Extract<Res<_>>`, which can only borrow), and the queue's
absence is what retires the texture — it lives and dies with `WorldMap`, so a session can never be
shown under the next session's terrain.

`apply_edits` rewrites kinds only. What the map records is the height the terrain was *generated* at,
never what was stamped over it, so a road or a town is shaded by the ground it sits on and no plan edit
costs an upload.

The shader reads with `textureLoad` at the tile's integer coordinate rather than sampling, so there is
no sampler, no address mode and no filtering to get wrong — a brightness step lands on a tile boundary
by construction. Chunk rows go up, matching the tiles' row-major-from-lower-left order, so no y flip
exists anywhere. A texel nobody has written reads zero, which is below the water line, so an
ungenerated world is untinted rather than wrong — the absence is the fallback, the way an unbaked sky
is clear. `write_texture`'s 64-byte rows are legal: the 256-byte row alignment is
`copy_buffer_to_texture`'s requirement, and `write_texture` waives it.

Two things about the ramp:

- **One ramp over the whole height range, not one per band.** A per-band ramp reverses at every band
  edge and would draw a contour line along every coastline, treeline and snow line.
- **Water passes through untouched**, at or below `TerrainConfig::shallow_water_max` — the water line
  is read from there rather than restated. That leaves a `strength`-sized brightness step at the coast,
  which is invisible because the coast is exactly where the tileset changes anyway. The cutout is exact
  to within one quantization step: `107/255` and `108/255` straddle 0.42, so no byte lands on the line
  and `water_is_exactly_what_falls_below_the_tint_water_line` leaves that byte unconstrained.

`TerrainTintUniform` in `tint.rs` and in `tint.wgsl` are the same struct written twice, on the same
terms as the weather's — vectors before scalars, and a mismatch is a runtime shader-compile failure.

**Tune against a screenful, not against the world.** The ramp spans the whole height range, but a
screen holds only a slice of it, so the visible spread is a fraction of `2 * strength` — 7.8% at the
world centre, 12.1% over lowland, 5.5% over highland at the defaults. `strength` is linear in that and
is the knob if it reads too flat. The `#[ignore]`d `the_default_ramp_measures_what_a_screenful_of_world_does`
takes those figures. Note none of this is visible to a unit test: what proved the pass actually draws
is a pair of captures at `strength` 0 and 0.85 with the weather plugin removed, diffed — the ratio came
out 0.79..0.86 across the view, varying with the terrain under it.

Only the height half of gh-13 is here. The noise half — a per-tile dither so identical tiles do not
repeat exactly — is a second step with its own spec, and would be a small *tiling* dither map beside
this one, the same trick the weather's shape map uses.

### Tileset coupling

`TerrainKind`'s discriminant *is* the tileset index *is* the atlas column, so the enum and
`assets/textures/terrain.png` cannot drift. The atlas is a horizontal strip of 8×8 tiles loaded once
(`world::load_tileset`) as an array texture via
`ImageArrayLayout::GridCount { columns: TERRAIN_KIND_COUNT, rows: 1 }`; every chunk entity shares
that one handle. Fifteen columns now: the biome rework appended Sand (8), Snow (9), Rock (10) and
Marsh (11), and gh-14 appended Scrub (12), Gravel (13) and Reed (14), so nothing existing moved.

`Gravel` is deliberately not `Rock`: `Rock` is cold alpine scree and reads wrong at sea level, where a
desert hardpan and a stripped lowland outcrop both live. `terrain.atlas.json`'s `width_in_tiles` has to
be bumped alongside the PNG for the art tool, though the game never reads it.

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
