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
be removed by hand — `gameplay/screen.rs` adds its `ScreenOverlay` on entering gameplay and takes it
off on leaving, and the whole post-process pass is gated on that component being there.

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

That claim is about **generation**, and since gh-6 it stops there. The terrain, the rivers, the cities
as founded and the roads are still a pure function of the seed on every platform; what happens to them
afterwards is not, because `gameplay/growth.rs` reads a sky that drifts on the frame clock. Nothing
below this line in the pipeline may be tested by reproducing a world.

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
  channel width is its flow, capped at `MAX_RIVER_WIDTH` = 4. A step is **scored** rather than taken
  steepest, and a segment is drawn as a curve through the node centres — see the notes below.
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
  only indexes those entities by chunk. **`City` is live state, not a founding record** — `radius` and
  `size` are what the city is this frame, re-derived from its town's tile count every step by
  `growth.rs`, which is what put `size` to work after it spent a release as `#[allow(dead_code)]`.
  `RoadQueue` snapshots cities by value and is stale by construction the moment the simulation starts;
  it is safe only because every road is routed before `Done`.

  `is_habitable` is Forest, Grass and **Scrub**. Scrub being habitable is a decision, not an oversight:
  it is the bare rung of the `Plains` and `Forest` ladders, so excluding it would have cut city sites
  out of ordinary grassland. The visible consequence is at the other end — a wadi promotes desert Sand
  to Scrub, so towns appear strung along desert drainage lines, which is where real ones are. Everything
  else the reworks added (Sand, Marsh, Rock, Snow, Gravel, Reed) stays uninhabitable, which is still
  most of what makes a desert or a marsh feel different to walk into. Breaking up the uniform grassland
  cost 5 cities of 96 and 5% of the town tiles — measured, and small enough to be worth it. Scrub is
  therefore also farmable, and `growth.rs` gives it its own poorer yield rather than lumping it with
  forest.
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

Some things about rivers are worth knowing before tuning them:

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

  **That last claim was true and is now wrong**, and the reason is worth keeping. It was measured
  before `Lattice::spill` existed, when raising the threshold only stopped a basin being *drawn*. Now
  it also links the channel across it, so the knob trades lake for river instead of deleting water:
  64 → 256 cut inland water by 19% while nearly doubling how far a river runs. The basins really are
  large — that part stands — but "large" turned out to mean "worth crossing", not "impossible to
  shift". Do not trust a measurement of a knob taken before the mechanism it feeds was written.
- **A course is scored, not steepest** (gh-9). At most nodes several neighbours are below, so which
  one the water takes is free shape — it costs nothing against "never climbs". Spending it on
  steepness was what made rivers straight on a slope and a 4-tile staircase where the fall line fell
  between two lattice directions. A step is now scored on **descent** (per tile travelled, against
  `river_reference_drop`) + **persistence** (off the heading, `river_heading_weight`) + **meander**
  (leaning to the side, signed by a low-frequency `SignedNoiseField`, `river_meander_weight`).
  Measuring descent against a *fixed* reference and not against the best candidate is what lets the
  terrain decide: steep ground swamps the other terms and the river runs the fall line, gentle ground
  lets them lead. `river_reference_drop` is the p90 drop along a real course, read off the world by
  the `#[ignore]`d `the_shape_of_the_worlds_rivers` — do not guess it.

  Two consequences that are not optional. The heading gives a particle memory, so two particles
  meeting would score a node differently and **braid**; hence a node's successor is fixed by the first
  particle through and followed by every later one. And on flat ground the bias is the only thing
  steering, so a course will curl into a closed ring — hence a particle may not step onto ground it
  has already crossed, which turns the ring into a flood.
- **Measure bends as excursion, never as sinuosity.** The obvious metric is a trap: steepest descent
  already scored 1.18 sinuosity, because a staircase travels 1.2x the distance it covers without ever
  leaving the straight line. Excursion — the furthest the course swings off its own straight line —
  is what tells a bend from a staircase, and `the_bends_come_from_the_scoring_and_not_from_the_lattice`
  asserts it against the same world with the shape terms switched off rather than against a constant.
- **The thing that actually lengthened the rivers was none of the above.** A basin under
  `river_lake_min_tiles` is filled and spilled through but not drawn, and no successor was recorded
  across it — so the channel had an **invisible hole** every few nodes, and no course ran far enough to
  hold a bend. Linking the entry node to the outlet (`Lattice::spill`) nearly doubled the courses long
  enough to have a shape. A basin *over* the threshold is still left unlinked, because that is a lake:
  the river ends at its shore and a new one leaves the far side. The lattice stride was the other
  suspect and it is innocent — swept 4 to 16 it changes nothing, because a river floods a pit where a
  drainage particle stops at one.

  This also turns `river_lake_min_tiles` into **the** length knob, since every basin under it is now
  crossed rather than ending a course — see its doc comment for the sweep. Raising it 64 → 256 is
  where most of the length came from.

  After all of it: 46.8k tiles of river (was 26.3k, and 0.28% of the world against roads' 0.24%),
  190.7k of lake (was 234.7k), 680 courses of 8+ nodes (was 232), p90 course 108 tiles, mean
  excursion 0.295, 6 road bridges (was 1 — the reuse-on-a-crossing path finally has coverage).

  **Inland water is still the ceiling.** 190.7k tiles of lake against 46.8k of river: a course meets a
  drawn lake every ~110 tiles now rather than every ~20, which is why bends became visible at all, but
  a 50-tile meander wavelength still only fits twice between lakes. Getting further is a question
  about how many basins the continent layer makes, not about river shape, and it wants its own task.

  One knob here is inert and worth knowing about before reaching for it: `river_flat_run_nodes` does
  not change a single tile of the default world at any value from 0 to 256. Level steps are what let a
  course wander a flood plain, and on a continuous noise field an exactly-level step essentially never
  comes up — the cap earns its place against a *synthetic* flat world, where without it the water
  wanders to the step cap and stops in the middle of nowhere.

Config defaults in `WorldPlanConfig` carry their measurements in the doc comments; the
`#[ignore]`d `the_default_config_lays_out_cities_of_every_size_and_roads_between_them` in `plan.rs`
generates the whole world in ~2 s and is how those numbers were taken —
`cargo test --release -- --ignored --nocapture`.

### City growth (`gameplay/growth.rs`)

The first **simulation** in the crate: everything above it is a pure function of the seed evaluated
once, this has state that advances. Once `WorldPlan` reaches `Done`, every city claims land, farms it,
and grows or shrinks against what it feeds. `WorldSystems::Growth` sits between `Planning` and
`Refresh`, so a tile it edits is visible in the frame it lands — and after `Planning`, so no road is
ever routed against a world whose cities are moving.

**Reproducibility ends here, deliberately** (see the note under terrain generation). The consequence is
the test strategy: `step_city` takes the sky as an argument rather than reading it, so with a fixed
`Sky` every property is an ordinary unit test — more land ends up bigger, a footprint that grew can
shrink back, a released tile is habitable, rain never costs a city a field.

Four things carry the design:

- **A tile has one owner, and no code arranges it.** `Farmland` is not `is_habitable`, and a claim
  requires habitable, so a field one city holds is refused to its neighbour by the same predicate that
  clips a city against its coast. There is no nearest-city partition anywhere. Two cities' *founding*
  discs are the one case this cannot separate — both are `Town` — so seeding carries a set for that
  single pass.
- **The ledger remembers claims, never the ground beneath them.** A released tile takes the commonest
  **habitable** kind among its neighbours, so a ring dissolves back into the country it was cut from.
  Voting only among habitable kinds is what makes it safe rather than merely plausible: the tile was
  habitable when claimed, so no release can put water, rock or road where a field was. The fallback —
  the biome's own wet kind — does real work, because a footprint clipped into lobes can leave a field
  whose every neighbour is sea. Every biome's wet kind is habitable, and a test says so.
- **The town is the ledger's prefix and the fields after it are distance-sorted.** That is what makes
  claiming an append, releasing a pop, and town growth a boundary move. The *whole* ledger is not
  sorted and cannot be: a town grows past land it never took, and a later rescan can turn that up
  nearer than tiles the town has since built on.
- **The fields are not a disc, because poor ground is not worth breaking.** A tile's yield is its
  cleared ground times the humidity *at that tile* — sampled per tile, not once per city, which
  matters because the humidity field's wavelength (~50 tiles) is shorter than a city's reach (56), so
  one side of a city is measurably wetter than the other. Anything under `min_field_fertility` of the
  best ground is left alone, so the farmland follows a wet valley and stops at a dry ridge. The
  measure of it: a city holds a median **39%** of the tiles within its reach, where an earlier cut with
  no floor took 97% and every city simply filled its circle. Scrub is habitable since gh-14 and so is
  claimable, but poor enough that most of it sits under the floor — which is why the median fell again
  from 57% when the cover ladder landed.
- **The footprint is sized on a rain-free quantity.** Rain moves the population and the population
  sizes the fields, so a wet spell does grow a city — but the claim/release *comparison* is
  `static_yield` against demand, and `static_yield` is the harvest with the weather taken out. So no
  cloud can make a city give up a ring the next dry step wants back. That round trip is lossy twice
  over (a neighbour can take the freed tile; re-claiming re-reads the yield from whatever it was
  restored to), which is how a world ratchets its forests into grass.

Two loops that look unstable and are not. Population chases `capacity = fields / food_per_person`,
while the town is sized *from* the population and is built **on** the fields — so growing costs food.
That is negative feedback, and it converges as long as `town_people_per_tile` exceeds what one field
feeds; `a_town_tile_houses_more_than_the_field_it_replaces_feeds` is the guard, and it is a stability
condition rather than taste. The logistic is in closed form, not the Euler `p + r·p·(1 - p/K)`, because
that oscillates at large rates and divides by a K that is **zero for every city on its first step**.

Costs are per-step and permanent, where every stage above pays once — and small: **0.53 ms for the
whole world of 92 cities**, once every `step_seconds`. A step is O(1) per city, because the field sum
is maintained incrementally and the rain is sampled once per city rather than once per field. Nearly
all of that 0.36 ms is the one city whose turn it is re-walking its cursor and re-sampling humidity;
a step where nobody rescans is free. Validation is a round-robin sweep of one city per step — walking
every ledger every step would be ~70k tile reads and would falsify the whole reason the sum is kept
incrementally. The founding fields are laid **unbudgeted** (25147 tiles in ~36 ms), like the
streamer's first screenful: making a city claim them at `claims_per_step` would have every city in the
world starving for the hundreds of steps its fields took to fill.

Defaults carry their measurements, from the `#[ignore]`d
`the_default_config_grows_the_world_into_a_steady_state`.

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

`WorldSystems` orders the frame `Streaming → Planning → Growth → Refresh`, which is what puts the plan's tile
edits between the streamer that spawns chunk entities and `refresh_edited_chunks` that rebuilds the
stale ones — so an edit is visible in the frame it lands. The river stage leans on that: it stamps
`river_chunks_stamped_per_frame` chunks a frame (~44 frames for the default world) rather than
~38k edits at once, because `apply_edits` scans its touched-chunk list linearly. A chunk that is *not* resident needs no
refresh at all: the edit went into `WorldMap`, so it is there when the chunk is next spawned.

Three coordinate spaces are in play and the helpers at the top of the module are the only sanctioned
way between them: chunk coordinates (`0..WORLD_CHUNKS`), global tile coordinates (what the noise is
sampled in), and world space in pixels. `tile_position_at` is the odd one out and the weather overlay
is why: it is the only conversion that keeps its fraction, since a screen pixel falls *between* tiles.

### Weather (`gameplay/weather.rs`)

Cloud patches drifting over the world, the shadow each throws, and rain in the thick of them. The
drawing is `gameplay/screen.rs`'s — see "The one post-process pass" below; this module bakes the maps,
advances the clock and hands the knobs over through `ScreenOverlay::set_sky`.

**This is no longer cosmetic, and that is the one thing to know before tuning it.** Nothing here reads
or writes `WorldMap` — the dependency is strictly one way — but `gameplay/growth.rs` reads *this*: a
city's harvest is modulated by the rain over it, so turning `rain_strength` down changes how big the
world's cities get. `SkySampler` is the seam, and it is the only thing this module exposes: the CPU
gets an answer, not the machinery, so `WeatherClock` and the field salts stay private.

The overlay draws from baked, byte-quantized, bilinearly filtered maps while `SkySampler` evaluates the
same fields directly, so the two agree to within a texel rather than exactly. Where they differ the
CPU answer is authoritative — it decides whether a city was rained on; the overlay only has to look
right.

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

### The one post-process pass (`gameplay/screen.rs`, `assets/shaders/screen.wgsl`)

**There is exactly one full-screen pass over the world, and the composite order is the order of the
lines in its fragment function** — not an ordering between systems:

```text
scene → height ramp → ground cover → sun light → cloud shadow → rain or snowfall → cloud
```

There used to be two, ping-ponging the same `ViewTarget`, so the second read what the first wrote.
That worked and it leaked: `WeatherUniform` carried a `light_level` field for no reason other than that
the tint had already lit the world, and gh-26 put the sun in the *tint* pass rather than a third one
because the tint was the only thing binding the heightmap — the same argument as merging, applied once
already. Merged, the light is a local and the field is gone.

**Ownership did not move, only the drawing did.** `tint.rs` still owns the ramp and the dither,
`weather.rs` the sky and its bakes, `sun.rs` the light, `ground.rs` the wetness and the snow; each
writes its own slice of one `ScreenOverlay` through a setter (`set_ramp`, `set_sun`, `set_sky`,
`set_ground`), so the uniform's field order stays private beside the shader that reads it. What
`screen.rs` owns outright is the heightmap texture, the pipeline, the specializer, the two ping-pong
bind groups and the pass.

Two things fall out of that ordering for nothing, and they are why gh-28 wanted the merge first. The
cover composites *before* the light, so snow is lit by the same sun as the ground it lies on and
shadowed by the same ridge, with nothing plumbed; and the clouds come after, so a cloud shadow crossing
a snowfield is a later line rather than a coupling. The water exclusion and the tile quantization the
cover needs were both already written for the height ramp — the cover sits inside the same branch and
reads the same `tile`.

`ScreenUniform` in `screen.rs` and in `screen.wgsl` are the same struct written twice: **field order is
the binding layout**. Vectors are declared before scalars so std140 padding agrees on both sides,
including under WebGL2. The shader is loaded by path at runtime, so a mismatch is a shader-compile
failure when you enter gameplay, not a build error — and `just check-web` only type-checks Rust, so it
will not catch a wgsl construct the web backend rejects. Adding a knob means touching the Rust struct,
the wgsl struct, the owning module's setter and its config doc comment together.

**Every map has a blank fallback and the pass always draws.** A 1×1 zero texture stands in for a
heightmap whose session has not started, a sky whose bake has not landed and a ground with no cover —
zero height is below any water line, zero cloud probability is a clear sky and zero cover is dry
ground, so each absence is the fallback that module already documents. That is what lets one bind group
layout carry maps arriving at different times; with one pass, a missing map that skipped the whole
thing would take the *other* effects down with it.

**The climate map is the one exception, and it is worth knowing why.** Its zero decodes to
`CLIMATE_MIN_CELSIUS`, which would put the whole world under falling snow for the second its bake
takes — so it gets its own blank, written with a byte meaning a mild `CLIMATE_FALLBACK_CELSIUS`. An
unbaked climate rains, which is what the world did before this feature existed. The lesson generalises:
"absence is the fallback" is a claim about what zero *means* in each map, not a property of zero.

Seven textures now — scene, height, cloud probability, cloud shape, ground cover, dither, climate — and
five samplers, against WebGL2's sixteen-texture limit.

The pass is registered `.in_set(Core2dSystems::PostProcess).after(tonemapping)`. Bevy 0.19 has **no
node-based render graph** for Core2d — no `Node2d`, no `bevy_render::render_graph` — so a post-process
effect is an ordinary system in the `Core2d` schedule taking `ViewQuery` and `RenderContext`;
`bevy_core_pipeline::fullscreen_material` is the in-tree template it was written from (it cannot be
used directly, since its bind group layout is fixed at three entries and cannot carry the maps). That
placement is what keeps all of it off the UI: `bevy_ui_render` orders `ui_pass` *after* the whole
`PostProcess` set.

**How the merge was verified, and the trick is reusable.** Captures of the same scene before and after,
diffed channel by channel: the worst difference is **1/255** on 5–10% of channels and nothing larger,
which is exactly the intermediate 8-bit write disappearing. Anything bigger would have been a composite
reordered rather than rounding. Making two runs comparable at all needed `fixed-delta 0` as the *first*
line of the scenario — every clock in the game is driven by `Time::delta`, so a zero delta freezes the
sun at `start_rotation` and the weather at offset zero, while generation, the plan and the bake are
driven by task-pool completion polled per frame and so still land. Without it the frame count before
`wait terrain` returns is wall-clock dependent, and the sun would be at a different hour in each run.

### The sun (`gameplay/sun.rs`)

A day/night cycle modelled as a planet turning, not as a timer with a brightness ramp on it. `Sun` is
the resource game logic reads and the session's only clock; `PlanetConfig` is the knob and outlives a
session like `TerrainConfig`. **Nothing simulation-side reads it yet** — the seam exists so it can, on
the same terms `SkySampler` exists for the rain, and its absence is full daylight the way an absent
`SkySampler` is a clear sky.

**There is one piece of state — how far the planet has turned — and everything else is geometry read
off it.** `Sun::rotation` is wrapped to 0..1 and *is* local solar time; the hour is `rotation * 24`.
That is the whole design and it is what pays for the module:

- **Day length is not a knob.** `sin(altitude) = sin φ sin δ + cos φ cos δ cos H` — latitude,
  declination, hour angle. An equinox day is half a turn at every latitude; the default's solstices are
  14.3 h and 9.7 h with nothing saying so.
- **Night is not a state and dawn is not an event.** Both are the altitude crossing the horizon, and no
  branch in the module names either.
- **Seasons are one moving knob away.** Declination is derived from `axial_tilt_degrees` and
  `orbit_phase`; an orbit would move `orbit_phase` and nothing else here would change.
- **There is no shadow-strength knob.** A shadowed tile is lit by `Insolation::sky` alone and a lit one
  by sky plus `direct`, so "how dark is a shadow" is already answered by how much of the light is beam.
- **No colour is keyframed.** A low sun is warm because its beam crosses more air — Kasten-Young air
  mass against a per-channel `zenith_extinction`. No configuration can make noon redder than dusk.

**The bearing is a vector, not a sign.** The brief allowed assuming rays run left-right; that is what
the model *reports* at latitude 0 with zero declination, and `at_the_equator_at_an_equinox_the_sun_runs_
exactly_left_to_right` pins it. The default latitude is 35, where the sun rises due east, swings south
through midday and sets due west — so a shadow rotates about a quarter turn over a day. At the equator
it would instead pass overhead, shadows would vanish at noon and the bearing would flip in one frame.

Three things about the light are not obvious and are load-bearing:

- **It is white-balanced against a reference altitude** — equinox noon *for this latitude*, derived
  rather than configured. There the two terms sum to `daylight` and the world is drawn exactly as the
  art was painted; every other hour is a departure from the art rather than from nothing. Without it a
  physically blue sky sits the whole world under a cast the pixel art was never drawn for. The beam is
  capped at what the sky leaves, so a solstice noon — higher than the reference — blows nothing out.
- **The sky has to dim with the sun that lights it.** The first cut held it at `sky_fraction` down to
  the horizon, where it swamped the beam and dusk came out neutral grey — the whole point of an evening
  is the ground going warm. `horizon_glow` is what is left of it at the horizon, with a square root
  between; and it cannot be exceeded by `night_sky`, or the world visibly *brightens* after sunset.
  `the_world_never_brightens_as_the_sun_sinks` is the guard.
- **`cosine_response` softens the cosine law.** The full `sin(altitude)` is right for a flat plane and
  makes a landscape go nearly dark long before sunset; what is drawn here is slopes and faces, which
  catch a low sun better. It is 1 at the reference whatever it is set to, so the balance does not move.

**Mountain shadows are three texture loads, not a ray march** — 1, 5 and 10 tiles along the bearing,
strongest occluder wins, each softened over `shadow_softness` because the sun sweeps a ridge *past* a
sample distance and a hard test flickers there. The price is that a lone spire between two samples
throws nothing.

`relief_tiles` is the crate's **first and only vertical scale**: the heightmap is stored on 0..1 with no
physical meaning, and "is that ridge high enough to hide the sun" cannot be answered without one.
**Nothing but the lighting may read it**, or a tile's kind would start depending on it. 128 came off the
world — see the sweep in its doc comment — because at 64 the shadows are gone by mid-morning and at 192
a ridge shades the country beside it at noon, which stops reading as shadow and starts reading as dirt.

The clock never pauses: while gameplay is up the planet turns with `Time` and nothing else gates it.

**There is a third struct written twice, and it is not one of these.** `WoodPanelMaterial` in
`city_panel.rs` and in `wood_panel.wgsl` carry the same discipline — vectors before scalars, and the
layout is 112 bytes — but a *different mechanism*, and reaching for this section's pattern when
writing UI would be the mistake. `screen.rs` is a fullscreen `Core2d` pass with a hand-written
`BindGroupLayoutDescriptor` and a specializer; a `UiMaterial` gets its layout **derived** by
`AsBindGroup` and its handle carried by `MaterialNode`, so what is duplicated across the language
boundary is the field list alone. Following `screen.rs` there would mean writing a render-graph pass
for a rounded rectangle.

Two more differences worth stating, since both mislead by analogy:

- **The failure lands later.** A UI material's pipeline is specialized when the first node carrying it
  is queued, so a mismatch is a shader-compile failure the first time a panel *opens* — not on
  entering gameplay. Look for it there.
- **`sd_rounded_box` is copied, not imported.** `bevy_ui::ui_node` does declare an import path, but the
  module also declares the view uniform at group 0 and a texture and sampler at group 1, which collide
  with a material's own group-1 binding — which is why `bevy_feathers`' `alpha_pattern.wgsl` copies it
  too. Only `bevy_ui::ui_vertex_output::UiVertexOutput` is imported. In that struct `border_radius` is
  in **pixels** while `border_widths` is in **UV**, and `size` is physical pixels, so every length in
  the wood's uniform is physical too.

`just check-web` still only type-checks Rust, so it catches none of this. What does catch it cheaply:
naga parses, validates and lowers the wgsl to GLSL ES 3.00 in isolation if you inline the one import
by hand — that is how `external` was found to be a reserved keyword before the panel was ever opened.

### Terrain tint (`gameplay/tint.rs`)

Scales the rendered world's brightness by the height of the tile under each fragment, so a slope reads
as a slope *inside* a kind's band rather than only where it crosses a `classify` edge. Cosmetic, like
the weather: nothing here touches `WorldMap`.

**The module is the ramp and nothing else** — three knobs, one setter and its tests. The heightmap
texture, the pipeline and the fragment function are `screen.rs`'s, the one post-process pass; this
decides how a height becomes a brightness and hands the numbers over through `ScreenOverlay::set_ramp`.
Being the only thing that bound the heightmap is why gh-26 put the sun in this pass rather than a third
one, and that argument taken one step further is why there is now only one pass at all.

The ramp is written once, on entering gameplay, where the sun and the sky are written every frame — so
`sync_tint_ramp` is ordered `.after(attach_screen_overlay)`, which is legal because a system is usable
as an ordering label wherever its parameter types are visible.

The height ramp is unchanged by the sun's arrival, deliberately: it is fake relief, but it is the only
cue at noon and through the whole night, when a real sun casts nothing. The two can disagree — the ramp
brightens a peak the sun may be behind. Fading it as the sun sinks would fix that and would invalidate
the screenful spread measured below, so it is not done.

**The height is kept, not re-baked.** `classify` already computes every tile's elevation and used to
drop it, so `generate_chunk` returns a `ChunkTerrain { kinds, heights }` and `WorldMap` stores both.
Sampling the world a second time measured ~9 core-seconds against the ~34 s the chunks themselves cost,
and would have put a second answer to "how high is it here" in a crate whose determinism tests rest on
there being one. The cost is memory: 16 MB of heights beside the 16 MB of kinds, and 16 MB again on the
GPU. `every_tile_keeps_the_height_it_was_classified_from` is the guard.

**The map is not an `Image` asset**, because Bevy re-uploads a whole `Image` on any change and 16 MB
per landed chunk is not a thing to do 4096 times. `screen.rs` owns a raw `WORLD_TILES` R8 texture and
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

The ramp's own numbers reach the shader as four fields of `ScreenUniform`, whose coupling rules are in
"The one post-process pass" above.

**Tune against a screenful, not against the world.** The ramp spans the whole height range, but a
screen holds only a slice of it, so the visible spread is a fraction of `2 * strength` — 7.8% at the
world centre, 12.1% over lowland, 5.5% over highland at the defaults. `strength` is linear in that and
is the knob if it reads too flat. The `#[ignore]`d `the_default_ramp_measures_what_a_screenful_of_world_does`
takes those figures. Note none of this is visible to a unit test: what proved the pass actually draws
is a pair of captures at `strength` 0 and 0.85 with the weather plugin removed, diffed — the ratio came
out 0.79..0.86 across the view, varying with the terrain under it.

**The noise half of gh-13 landed early, and for a different reason.** `GroundDither` is the small
*tiling* per-tile map this module always wanted — 128×128 R8, one texel per tile, repeated — and it is
here because this is the module that decides how a smooth quantity becomes a *tile*. gh-28 is what
needed it first: it is what the snow cover is thresholded against. Three things about it:

- **It is a smooth field, not white noise, and that is the whole of it.** Thresholding a correlated
  field gives coherent patches that shrink from their *edges* as the coverage falls, which is what
  melting snow does. White noise gives salt-and-pepper that dissolves uniformly everywhere at once.
- **The threshold is squeezed into `softness..1 - softness`** rather than being the dither value
  itself. That is what makes the endpoints exact — full coverage snows every tile, no coverage snows
  none — and without it any tile whose dither fell under the softness would be faintly snowed on a bare
  summer afternoon. `the_dither_leaves_full_and_empty_coverage_alone` is the guard, and `snow_lying` in
  `tint.rs` is the shader's arithmetic transcribed so it can be checked without a GPU.
- **Nearest and repeated**, unlike every other map in the crate, which are linear. One texel is one
  tile and the point is that a tile gets its *own* number; filtering would put a snow edge inside an
  8px tile.

Baked at `Startup` rather than on entering gameplay, because it is a knob and not world state.

### The ground (`gameplay/ground.rs`)

What the weather leaves behind: rain wets the ground and it dries, below freezing what falls lies as
snow and the snow melts back into wetness. **Nothing here writes a tile.** `WorldMap` is untouched, no
chunk is ever marked dirty, and the alpine `Snow` *kind* keeps meaning the height band it always did —
with a transient snow line moving around underneath it.

**This is the first state the weather has ever had.** Everything in `weather.rs` is a pure function of
`(place, clock)`; wetness and snow are integrals of what has happened, so they have to be remembered.
Three ways were considered:

- **Per tile.** 16.7 M tiles × 2 bytes is 33 MB beside the heightmap's 16, and a step touching every
  one. The resolution buys nothing — a rain patch is a cloud interior, tens of tiles across.
- **Stateless, as an upwind convolution.** Worth knowing about because it almost works: the cloud field
  translates with the wind, so the rain history at a point is a line *upwind* of it and a dozen taps in
  the shader would give wet trails with no state at all. It fails because the two shape layers drift at
  different speeds — so the field is only approximately a translation — and because snow's time
  constant is a whole day, which is hundreds of taps. Still the right trick if wetness ever has to
  exist without a grid.
- **A coarse world grid** ← chosen. 256², one texel per 16 tiles, 128 KB for the whole world. Bilinear
  on the way out so the *envelope* is smooth; the per-tile look comes from the dither, not the grid.

**The grid does not know where the water is, and must not learn.** Snow accumulating over a lake is
harmless because the pass that draws it already excludes everything at or below the water line — it had
to, for the height ramp. That is what keeps this module free of `WorldMap` by construction rather than
by discipline.

`advance` is the whole model and fits on a screen: a `snowing` ramp across the freezing point (so sleet
exists and no frame flips a region), a degree-day melt, meltwater into wetness, and an exponential dry
that is faster the warmer it is. **It is stable for any `dt`** — the gains clamp, the melt is bounded by
the snow there is, and the decay is an exponential of something that cannot be positive. That matters
because the step runs on `AsyncComputeTaskPool` at whatever interval the frame rate leaves it, so `dt`
is not a number this module chooses. `cover_stays_between_none_and_full_under_any_step` fuzzes it.

**A step costs 22.0 ms** — 65k texels × two tiling fbms through `SkySampler` — against a `step_seconds`
of 0.25, so about 9% of one core continuously and thirteen times a 60 fps frame budget. Hence the pool,
with one step in flight; a slow step swallows its backlog rather than queueing, because the model is an
integral and integrating a longer interval is exactly right. The step goes through the public
`SkySampler` rather than the baked texels so the crate still holds **two** transcriptions of the sky's
arithmetic and not three; if that ever has to be cheaper the ladder is 128² first (a quarter of the
bill), then the baked-texel shortcut (~5× at the price of that third transcription), then a compute
shader — which is a product decision as much as a performance one, since WebGL2 has none and the CPU
could no longer read the field without an async readback.

**Temperature is a field the world did not have.** `TerrainSampler::temperature` returns the climate
normal in **degrees Celsius** — a physical unit rather than the crate's usual 0..1, because a freezing
point has to mean something — from `sea_level_celsius − lapse_celsius × elevation + the biome's
temperature_bias + its own noise`. It is deliberately **not** part of `TileSample` and `classify` may
never read it, on exactly the terms `relief_tiles` may only be read by the lighting: a tile's *kind*
must not start depending on the weather. `HeightRecipe` gains one blendable column for it, and the
coverage figures are unmoved because nothing in generation reads it.

Two things fall out of the numbers rather than being arranged:

- **The transient snow band is `2 × amplitude / lapse_celsius` of the height range** — at 26/34 with a
  ~7-degree swing that is elevation 0.55 to 0.97, nearly all the land above the middle of the lowland
  band. That is why the loop is visible in one 300-second day and not only in a configured winter.
- **A desert freezes at night**, because the diurnal amplitude is a *field* damped by humidity: dry air
  swings hard and wet air barely moves. The humidity is already in the climate map, so it costs nothing
  and nothing here learns what a desert is.

The day's swing comes from `sun::warmth_at`, which is `sin(altitude)` scaled to fit −1..1 — geometry
again, so a summer night is milder and a midnight sun never reaches −1, with nothing saying so. The lag
is the caller's (`thermal_lag_rotations`), and it is a *phase shift*: it puts the warmest moment in the
mid-afternoon, which is right, and the temperature trough in the small hours rather than before dawn,
which is not quite. A dawn minimum needs an asymmetric response, which is state.

The seasonal term is `seasonal_amplitude_celsius × sin(declination) / sin(axial_tilt)` and is
**identically zero at the default equinox**, so an orbit landing later changes nothing here.

Defaults carry their measurements from the `#[ignore]`d `the_default_config_measures_a_day_of_weather`.
At the shipped equinox the snow line walks from elevation **0.59 before dawn to 0.77 in the late
afternoon and back**, with the snowed share of land running 8.3% → 3.4% → 8.3% over one day and mean
wetness on 17.3% of it. Two shapes in that worth keeping: the snow peak lands at **04:30**, later than
the temperature trough, because snow is an *integral* — the lag alone could not have put it there; and
the wet share barely moves, because wetness is a steady state with weather passing through it rather
than a cycle. Wetness saturates where it rains (p99 is 1.0 over land) even though the median is 0.

**What proved the pass actually draws it**, since none of the drawing half is visible to a unit test:
the scenario run twice, once with the cover composite in `screen.wgsl` multiplied out, and the frames
diffed. **16.1% of pixels lighter by up to +96 luma** where snow lies, and 0% darker — the snow. In
daylight the wet ground shows as ~1% darker by up to −8. That comparison needs `fixed-delta 0` while
the waits run or the two runs are not comparable; see the scenario's own comment.

`GroundConfig` is a knob and outlives a session; `GroundCover`, `ClimateMaps` and `ClimateBake` are
world state and go on `OnExit`, which cancels a bake or a step still in flight. `GroundCover::at` is
the read seam, built the way `SkySampler` was — an answer, not the machinery — so `growth.rs` starving
a field under snow is a change to that module and not to this one. It is explicitly out of scope here.

### City stats panel (`gameplay/city_panel.rs`, `assets/shaders/wood_panel.wgsl`)

Click a city, get a wooden window showing what `growth.rs` is doing to it. The first thing in the
crate that *asks the world a question* — everything above is a generator or a simulation writing into
`WorldMap` — and the first UI that exists during `Screen::Gameplay`, which is why it sits under
`gameplay/` rather than beside `tooltip.rs`: everything it reads is private to this module tree.

**The pick is geometric, never a tile lookup.** "Which city is under the cursor" is answered from
`CityMap` and the cities' own centres and radii. A `Town` tile knows nothing about which city stamped
it, and since gh-6 a footprint is a claimed set rather than a disc, so the tile would have to be
traced back to an owner nothing indexes. Two consequences worth knowing:

- **The click target has a floor in *screen* pixels**, and that is the whole of the issue's "works at
  every zoom step": at `MAX_ZOOM_SCALE` a 3-tile hamlet is 6 px across. `pick_slack_px` is converted
  through the orthographic scale, so it grows as the world shrinks — hence `camera.rs` now exports
  `orthographic_scale` rather than the pick destructuring `Projection` a fourth time.
- **Nine chunks are enough, and the argument is not the radius arithmetic.** A city is always in the
  row of the chunk holding its *centre* (that tile is habitable by construction, so always stamped)
  and `CityMap` only ever inserts — the index is monotone and over-inclusive, so a candidate that
  fails the distance test is expected rather than a bug. The radius bound is a separate real
  constraint: the scan stays sufficient only while `pick_slack_px < 104`, and a test says so.

**Containment beats proximity, and the obvious rule has it backwards.** Minimising distance *minus*
radius hands a click one tile inside a hamlet (-2) to a metropolis five tiles away (-7) — exactly the
theft the rule was written to prevent. The pick prefers a city that contains the click and among those
the smallest, ties broken on id so nine chunk rows cannot be visited into a different answer.

**The bar's guard is the whole of `bar_fill`.** Population is floored at `min_population` and capacity
is exactly zero on a city's first step — every city in the world — so a bare
`(population / capacity).clamp(0.0, 1.0)` is `inf.clamp(..)`, which is **one**: a full bar at the
moment a city has nothing. Its colour (harvest against demand) and its length (population against
capacity) are independent readings, so a city can show nearly full *and* starving, which is the state
that precedes a collapse.

**Read every frame, write only what changed.** The reading must not be filtered — the sim writes
`CityGrowth` every step. The writing must be: touching a `Text` costs two full text layout passes and
touching the material clones it into the render world and allocates a fresh uniform buffer and bind
group. The numbers move twice a second (`step_seconds` 0.5) against sixty redraws.

**Every string is outlined, because bevy has no text stroke.** `TextShadow` is a single offset and
there is nothing else, so `outlined_text` draws the string in black at all eight neighbouring pixels
with a white copy on top — nine texts each. Wood is a mid-tone with dark grain running through it, so
one flat colour is legible over part of a plank and lost over the rest; a dim second ink for the
labels was the first attempt and was the unreadable one. The trick that keeps this cheap to *maintain*
is that the `CityStatValue` marker rides on every copy, so the readout's ordinary query rewrites all
nine without knowing an outline exists. Only the white copy is in flow — it alone sizes the container,
and being spawned last is what puts it on top.

The three systems share one gated, chained set ordered `after(WorldSystems::Growth)`. The gate is not
optional and hangs off the *set*, exactly as `WorldSystems` does: `CityMap` and `RoadNetwork` exist
only inside a session, `Screen::Main` is the default state, and `.after()` inherits an ordering edge
but never a run condition. The ordering is about determinism rather than staleness — the sim holds
`&mut City` and `&mut CityGrowth`, so without the edge the executor may run the readout either side of
the step and choose differently each frame.

**This is the crate's first `UiMaterial`, and it is not the tint's mechanism** — see the coupling note
below. A city has no name because the crate has no name generator; the tier is a *row* rather than a
header, because `CitySize` is re-derived from the radius every step and a header written at spawn
would be the one thing on the panel that goes stale.

Two known gaps, neither worth blocking on. A city's visible `Town` tiles can extend past `City.radius`
— the radius is `sqrt(town_claims / π)`, an area rounded into a circle, while the claims reach up to
`farm_max_reach_tiles` — so the outskirts of a clipped coastal city do not pick. And the panel takes
bevy's default font rather than the feathers `fonts::REGULAR` the menus inherit, so it does not match;
that is deferred to its own task.

### Tileset coupling

`TerrainKind`'s discriminant *is* the tileset index *is* the atlas column, so the enum and
`assets/textures/terrain.png` cannot drift. The atlas is a horizontal strip of 8×8 tiles loaded once
(`world::load_tileset`) as an array texture via
`ImageArrayLayout::GridCount { columns: TERRAIN_KIND_COUNT, rows: 1 }`; every chunk entity shares
that one handle. Sixteen columns now: the biome rework appended Sand (8), Snow (9), Rock (10) and
Marsh (11), gh-14 appended Scrub (12), Gravel (13) and Reed (14), and gh-6 appended Farmland (15), so
nothing existing moved.

`Gravel` is deliberately not `Rock`: `Rock` is cold alpine scree and reads wrong at sea level, where a
desert hardpan and a stripped lowland outcrop both live.

**The count is not self-checking and the failure is silent.** The strip is *divided* by
`TERRAIN_KIND_COUNT`, so a constant one behind the PNG does not fail to load — it slices the strip at
the wrong offset and draws every tile in the game wrong. That is exactly what the working tree looked
like between the Farmland art landing and the enum being bumped.
`the_atlas_has_a_column_for_every_terrain_kind` now reads the PNG's own IHDR width and fails instead.
`terrain.atlas.json`'s `width_in_tiles` is a third copy of the same number and has to be bumped
alongside for the art tool, though the game never reads it — which is why nothing noticed the drift.

Adding a terrain kind means: append a column to the PNG, add the enum variant with the matching
discriminant, bump `TERRAIN_KIND_COUNT`, and extend `the_default_config_produces_every_base_kind` —
unless, like `Town`, `Road` and `Farmland`, the kind is stamped rather than generated, in which case
that test must keep *not* seeing it and `the_terrain_never_produces_a_kind_the_plan_stamps` must
learn about it. `KIND_BY_INDEX` is a fixed-length array over the constant, so it stops compiling
until the variant is added — that is the guard, and it is worth keeping.

Tiles are drawn at their native 8px size with `ImagePlugin::default_nearest`; upscaling
`tile_display_size` would resample the pixel art. `terrain.atlas.json` is sidecar metadata from the
art tool (palette/ramps) and is not read by the game.

### Driving the game from outside (`control/`, `src/bin/wusel-ctl.rs`, `scenarios/`)

The game can be driven and observed over a Unix socket, so a change that only shows up on screen can
be verified without a person holding the keys. `just drive-start` launches the game as normal;
`just drive <command>` sends one command and prints a line of JSON.

**Standing rule: every new way a player interacts with the game, and every new setup stage, must be
reachable from `wusel-ctl` in the same change that adds it.** This tool only proves what it can
reach, and it degrades silently: a feature the ctl cannot drive is not partially covered, it is
invisible, and nothing says so. Left to drift, the driver ends up exercising a game that no longer
exists while every scenario still passes. Concretely —

- **A new input** (a key, a mouse target, a drag, a hotkey) needs a verb, or an argument to an
  existing one. `click-tile` exists because "click the city at 2343,1969" is how a scenario wants to
  say it; the pixel underneath is the driver's problem, not the author's.
- **A new screen or state** must be reachable via `enter`, and have a `wait` condition if arriving
  there is not instant.
- **A new asynchronous setup stage** — anything on `AsyncComputeTaskPool`, the way generation, the
  weather bake and the plan stages are — needs a `wait` condition. This one is the least obvious and
  the most damaging to skip: without it a scenario has no choice but to count frames, and counting
  frames against a task that lands on wall-clock is flaky by construction. That is the failure mode
  this whole tool exists to remove, and one missing condition reintroduces it.
- **New world state worth asserting on** needs an `observe` topic, so a scenario can check it without
  a human reading a screenshot.
- **A clock the world runs on needs a verb to set it.** gh-28 added `sun <rotation>` and
  `season <orbit_phase>` for this: without them a scenario has to wait 300 real seconds to see
  midnight, and there is no way at all to see a winter. The rotation *is* the sun's only state, so
  writing it is the whole of moving the clock — everything else is re-derived the same frame.

**Two traps in driving a world with clocks in it**, both found by getting them wrong:

- **The clock never stops.** `sun 0.5` sets where the planet is *now*; every `step` after it turns the
  planet on. The hour a capture is taken at is the hour it was set to plus the stepping since, and a
  scenario that forgets this labels a dusk frame "noon" and still passes.
- **A knob outlives the session.** `PlanetConfig` and `GroundConfig` are knobs by design, so `season
  0.75` set by one scenario is still in force for the next one run against the same process. A
  scenario that depends on the season has to *state* it — `scenarios/snow_falls_and_melts.txt` opens
  with `season 0` for exactly this reason.

**`fixed-delta 0` is the trick that makes two runs comparable at all**, and it is worth knowing about
before reaching for anything cleverer. Every clock in the game is driven by `Time::delta`, so a zero
delta freezes the sun, the weather and the ground — while generation, the plan and the bakes still
land, because those are polled per frame rather than driven by time. Put it *before* the waits, which
take a machine-dependent number of frames, and switch to a real delta afterwards: everything from there
on is a pure function of the seed. Without it the sky has drifted a different distance by the first
capture and a before/after diff is measuring the weather. Both the pass merge and the ground cover were
verified this way, and the ground's state came out bit-identical across runs.

The cost is deliberately lopsided in favour of keeping this up: a verb or a topic is a small addition
to `control/command.rs` or `control/observe.rs`, while a *scenario* is a data file needing no
recompile. If adding the hook feels expensive, that is usually a sign the interaction is reaching
into something it should not — the ctl drives the same messages and state a player does, so anything
awkward to reach from it is awkward for the same reason a test would be.

**The app stays a real windowed game.** A headless render-to-image driver was built first and thrown
away: it verifies a rendering path no human ever takes. `Screenshot::primary_window()` photographs
the actual swapchain, through the real post-process chain, at the real window size — that is the
whole point of the tool, and it is why there is no headless mode to maintain.

`ControlPlugin` is inert unless `WUSEL_CONTROL` names a socket path, and the module is
`#[cfg(not(target_arch = "wasm32"))]`. An env var rather than a cargo feature because CI already
builds `--all-features`, so a feature would need care in every recipe to buy nothing. The cfg gate is
also why `WorldMap::generated` and `BackgroundGeneration::remaining` carry
`#[cfg_attr(target_arch = "wasm32", allow(dead_code))]` — their only reader vanishes on wasm, and
`just check-web` is the only recipe that notices.

**A command's reply is held until the effect has happened.** `wait plan` answers when the plan is
done, `hold D 240` after 240 frames, `capture` once the PNG is on disk. What blocks is the *client*;
the app runs on undisturbed, so a human can watch a scenario and grab the keyboard part-way through.
One system in `PreUpdate` polls the in-flight command once a frame, so waiting and acting are the
same mechanism rather than two. It is `.before(InputSystems)`, because `keyboard_input_system` drains
`MessageReader<KeyboardInput>` into `ButtonInput` there and an injected key written after it lands a
frame late. It is exclusive (`&mut World`) so that adding an observation does not mean threading
another dozen resources through a signature.

**`wait`, not frame counts, is what makes a scenario reproducible.** Generation, the weather bake and
the plan run on `AsyncComputeTaskPool` and land on wall-clock, so counting frames to wait for them is
machine-dependent by construction. Frame counts *are* exact for anything the simulation drives, but
only under `fixed-delta`: `TimeUpdateStrategy::ManualDuration` is applied by `time_system` regardless
of who owns the event loop, so the window still renders as fast as it can while `Time::delta` is
pinned. `hold D 240` at `1/60` moves the camera 2047.998 units on any machine.

Input goes in as *messages* (Bevy 0.19 calls them that, not events), so it takes the real path and
the window needs no focus. `hold` presses once and releases once — `ButtonInput` retains `pressed`,
and re-pressing every frame would re-fire the `just_pressed` edge that the zoom and the city click
read. `zoom` therefore costs two frames a step, since one key yields one edge per frame. Screen
transitions set `NextState<Screen>` rather than clicking the Feathers button, so **the menu buttons
themselves stay unverified** — driving `bevy_picking` is a large lift for a transition no scenario
tests. `click-tile` converts through the camera's own `Transform`, not its `GlobalTransform`, for the
reason `city_panel.rs` gives at the matching conversion.

`capture` knows it is finished when the screenshot *entity* is gone: `clear_screenshots` despawns it
in `First`, strictly after the `ScreenshotCaptured` observer has written the file, so its absence
means the PNG is complete — no polling the filesystem and no racing a half-written file.

**`observe log` is the one observation a capture cannot replace.** A shader that fails to compile
draws nothing rather than drawing wrong, so the frame looks merely odd and every other check passes;
the naga diagnostic is the only evidence, and it is otherwise buried in stderr. `control/log.rs` adds
a `tracing` layer keeping the run's `WARN`/`ERROR` in a bounded buffer. It is wired through
`LogPlugin { custom_layer }` in `main.rs` rather than by `ControlPlugin`, because a log layer has to
exist before the logger does and `LogPlugin` is built first — and it installs nothing unless
`WUSEL_CONTROL` is set, since a buffer with no reader is a slow leak.

It **drains**, so that a scenario can bracket a step — clear, do the thing, see what it said. The
consequence is that the *first* `observe log` is the only one that can see startup, which is exactly
where shader compile failures land. Call it before anything else that reads the log, as
`scenarios/pan_east.txt` does; otherwise the interesting errors have already been thrown away.

Adding an **observation** is a Rust change in `control/observe.rs`; adding a **scenario** is a data
file. That asymmetry is the point — it is what keeps a scenario per feature cheap enough to bother
with. `scenarios/pan_east.txt` is the worked example, and `run` is handled client-side so the
protocol stays one command per connection and no script format enters the engine.

Two things a driven run needs that a `just run` does not: `BEVY_ASSET_ROOT`, because a directly
invoked binary looks for `assets/` next to the executable and will otherwise open a window with no
tileset and only a log line to say so; and the same `RUSTFLAGS` as `just run`, or the two alternate
full rebuilds.

## Conventions

Comments here explain *why* a value or structure was chosen (the fbm gain constant, the chunk margin,
the bevy_lint pin), not what the code does. Match that. Tests are named as behavioural sentences.

A feature that adds a way to interact with the game, or a stage that has to finish before the world
is usable, is not done until `wusel-ctl` can drive it — see the standing rule under "Driving the game
from outside". A unit test proves the maths; the ctl is the only thing that proves the game.

`Cargo.toml` allows `clippy::too_many_arguments` and `clippy::type_complexity` globally — normal for
Bevy systems, so don't work around them.
