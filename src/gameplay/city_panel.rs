//! Click a city, read what the simulation is doing to it off a wooden panel.
//!
//! This is the first thing in the crate that *asks the world a question*. Everything
//! else is a generator or a simulation writing into `WorldMap`; this reads back out, and
//! nothing here writes a tile, a [`CityGrowth`] or a plan. It is also the first UI that
//! exists during [`Screen::Gameplay`] — `main_screen.rs` and `tooltip.rs` both belong to
//! the menus — which is why it lives under `gameplay` rather than beside them: every
//! answer it wants ([`CityMap`], [`City`], [`CityGrowth`], [`RoadNetwork`], the tile and
//! screen conversions) is private to this module tree, and none of it is anything the
//! menus could ask.
//!
//! **The pick is geometric, not a tile lookup.** "Which city is under the cursor" is
//! answered from [`CityMap`] and the cities' own centres and radii, never from the `Town`
//! tile the click landed on: a tile knows nothing about which city stamped it, and since
//! the growth simulation a city's footprint is a claimed set rather than a disc, so the
//! tile would have to be traced back to an owner that nothing indexes.
//!
//! **The click target has a floor in *screen* pixels**, and that is the whole of "works
//! at every zoom step" — at [`MAX_ZOOM_SCALE`](crate::camera::MAX_ZOOM_SCALE) a 3-tile
//! hamlet is 6 px across and no cursor could hit it. The slack is converted through the
//! orthographic scale rather than fixed in tiles, so it grows as the world shrinks.

use bevy::{
    ecs::spawn::SpawnIter,
    prelude::*,
    render::render_resource::AsBindGroup,
    shader::ShaderRef,
    ui::{ComputedNode, UiGlobalTransform},
    window::PrimaryWindow,
};

use crate::{
    camera::{WorldCamera, orthographic_scale},
    gameplay::{
        city::{City, CityMap, CitySize},
        deposit::Resource,
        growth::CityGrowth,
        industry::CityIndustry,
        road::RoadNetwork,
        world::{
            CHUNK_SIZE, TILE_DISPLAY_SIZE, WORLD_CHUNKS, WorldSystems, chunk_index_of_tile,
            chunk_of_tile, tile_in_world, tile_position_at,
        },
    },
    screens::Screen,
};

const WOOD_SHADER_PATH: &str = "shaders/wood_panel.wgsl";

/// The wood's own numbers are constants rather than knobs, unlike every other config in
/// the crate. They are art: there is no measurement to carry and nothing about the world
/// that could move them, so a `CityPanelConfig` field apiece would be seven knobs that
/// only ever hold their defaults. What the panel *does* — the slack, the layout, the ink
/// — is a knob, and that is the line.
///
/// Physical pixels, because `UiVertexOutput.size` is.
const PLANK_PX: Vec2 = Vec2::new(160.0, 34.0);
const RING_PX: Vec2 = Vec2::new(9.0, 70.0);
const RING_WARP_PX: f32 = 6.0;
const FIBRE_STRENGTH: f32 = 0.06;
const BEVEL_PX: f32 = 5.0;

/// Layout, in logical pixels. These decide the panel's height *arithmetically*, which is
/// not a shortcut: a node has no computed size until the UI layout runs in `PostUpdate`,
/// so at the moment the panel is spawned there is nothing to measure and nothing for the
/// window clamp to clamp against. The rows never change after spawning, so the sum is
/// exact rather than an estimate.
const PANEL_PADDING_PX: f32 = 10.0;
const HEADER_HEIGHT_PX: f32 = 18.0;
const ROW_HEIGHT_PX: f32 = 14.0;
const ROW_GAP_PX: f32 = 2.0;
const VALUE_COLUMN_PX: f32 = 76.0;
/// The two columns and the gap between them. `panel_width_px` has to be at least their
/// sum plus the padding, or the resource column wraps under the stats one.
const STATS_COLUMN_PX: f32 = 168.0;
const RESOURCE_COLUMN_PX: f32 = 138.0;
const COLUMN_GAP_PX: f32 = 14.0;
const HEADER_FONT_PX: f32 = 14.0;
const ROW_FONT_PX: f32 = 11.0;

/// Where the outline copies of a string sit relative to it, in logical pixels.
///
/// All eight neighbours rather than the four cardinals: at one pixel the diagonals are
/// what close the corners of a glyph, and without them a light letter on dark grain still
/// bleeds at the joints. Nine texts per string is the price, and it is paid in layout of
/// short strings rather than in anything per-frame — see [`outlined_text`].
const OUTLINE_OFFSETS: [Vec2; 8] = [
    Vec2::new(-1.0, -1.0),
    Vec2::new(0.0, -1.0),
    Vec2::new(1.0, -1.0),
    Vec2::new(-1.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(-1.0, 1.0),
    Vec2::new(0.0, 1.0),
    Vec2::new(1.0, 1.0),
];

/// Which number a value [`Text`] in the panel carries, so the readout can find the one
/// text it has to rewrite without keeping a positional list of entities.
///
/// `Tier` is a row rather than part of the header, and that is deliberate: [`CitySize`]
/// is re-derived from the radius every step, so a hamlet that thrives becomes a borough.
/// A tier written into the header at spawn time would be the one thing on the panel that
/// goes stale. The id is the only genuinely fixed thing about a city.
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
pub enum CityStat {
    Tier,
    Population,
    Harvest,
    Demand,
    Capacity,
    TownTiles,
    FieldTiles,
    /// Properties of the *city* rather than of a resource, which is why they join this
    /// column instead of the resource one beside it.
    Happiness,
    Idle,
    Roads,
    Centre,
}

impl CityStat {
    /// Every stat, in the order they are laid out. The spawn loop and the readout both
    /// walk this, so adding a tenth stat is one variant and one match arm rather than
    /// three lists that have to agree.
    const ALL: [CityStat; 11] = [
        CityStat::Tier,
        CityStat::Population,
        CityStat::Harvest,
        CityStat::Demand,
        CityStat::Capacity,
        CityStat::TownTiles,
        CityStat::FieldTiles,
        CityStat::Happiness,
        CityStat::Idle,
        CityStat::Roads,
        CityStat::Centre,
    ];

    fn label(self) -> &'static str {
        match self {
            CityStat::Tier => "Tier",
            CityStat::Population => "Population",
            CityStat::Harvest => "Harvest",
            CityStat::Demand => "Demand",
            CityStat::Capacity => "Supports",
            CityStat::TownTiles => "Town",
            CityStat::FieldTiles => "Fields",
            CityStat::Happiness => "Content",
            CityStat::Idle => "Idle",
            CityStat::Roads => "Roads",
            CityStat::Centre => "At",
        }
    }

    /// What this stat reads for a city right now.
    ///
    /// Both extras are optional so that every row has one answer to "what is not known
    /// yet", rather than the caller special-casing some of them. `growth` is missing for
    /// the whole road-planning stage — the entities are spawned when the city plan lands
    /// but `seed_cities` waits for `WorldPlan::Done`, and the roads are routed one at a
    /// time — during which the towns are on the map and clickable. `roads` is missing
    /// only while the panel is being spawned, since counting them needs a resource the
    /// scene builder has no reason to hold.
    fn format(
        self,
        city: &City,
        growth: Option<&CityGrowth>,
        industry: Option<&CityIndustry>,
        roads: Option<usize>,
    ) -> String {
        const UNKNOWN: &str = "—";
        let of_growth =
            |f: &dyn Fn(&CityGrowth) -> String| growth.map_or_else(|| UNKNOWN.to_string(), f);
        // Missing on exactly the same terms `growth` is, and for the same reason: both
        // are written by `seed_cities`, which waits for `WorldPlan::Done` while the
        // towns are already on the map and clickable.
        let of_industry =
            |f: &dyn Fn(&CityIndustry) -> String| industry.map_or_else(|| UNKNOWN.to_string(), f);
        match self {
            CityStat::Tier => tier_name(city.size).to_string(),
            CityStat::Centre => format!("{}, {}", city.centre.x, city.centre.y),
            CityStat::Population => of_growth(&|g| format!("{:.0}", g.population)),
            CityStat::Harvest => of_growth(&|g| format!("{:.1}", g.food)),
            CityStat::Demand => of_growth(&|g| format!("{:.1}", g.demand)),
            CityStat::Capacity => of_growth(&|g| format!("{:.0}", g.capacity)),
            CityStat::TownTiles => of_growth(&|g| g.town().to_string()),
            CityStat::FieldTiles => of_growth(&|g| g.fields().to_string()),
            CityStat::Happiness => of_industry(&|i| format!("{:.0}%", i.happiness() * 100.0)),
            CityStat::Idle => of_industry(&|i| format!("{:.0}", i.idle())),
            CityStat::Roads => roads.map_or_else(|| UNKNOWN.to_string(), |n| n.to_string()),
        }
    }
}

/// Shuts every panel there is.
///
/// One implementation, because "at most one panel is open" is an invariant three systems
/// depend on and would otherwise each enforce their own way. Every match rather than the
/// first: a duplicate would linger invisibly, since `Option<Single<..>>` reads "more than
/// one" as "none".
fn close_panels(commands: &mut Commands, panels: impl IntoIterator<Item = Entity>) {
    for panel in panels {
        commands.entity(panel).try_despawn();
    }
}

fn tier_name(size: CitySize) -> &'static str {
    match size {
        CitySize::Hamlet => "Hamlet",
        CitySize::Village => "Village",
        CitySize::Borough => "Borough",
        CitySize::Metropolis => "Metropolis",
    }
}

/// The panel's root node, and the only record that a panel is open. A component rather
/// than a resource: the entity *is* the panel, so the two cannot disagree about whether
/// one exists, and `DespawnOnExit` retires it with the session.
#[derive(Component)]
pub struct CityStatsPanel {
    city: Entity,
}

/// One per value text in the panel — including every outline copy of it, which is what
/// lets the readout rewrite all nine without knowing the outline is there.
#[derive(Component, Clone, Copy)]
pub struct CityStatValue(CityStat);

/// One row of the resource column, on exactly the same terms: it rides on all nine
/// outline copies, so the ordinary query rewrites them without knowing the outline
/// exists.
#[derive(Component, Clone, Copy)]
pub struct CityResourceValue(Resource);

/// What one resource row says: the store, and who is working it.
///
/// **The stock and the hands together**, because the store and who is working it are
/// one reading — splitting them into two columns would be twelve rows to say six
/// things. The Food row *is* the granary, so the one number that says how long a city
/// can eat through a bad spell needs no row of its own; the Harvest row beside it
/// still reports the harvest alone, and Supports still reports the capacity, which now
/// has the granary's release in it. **The two differing is exactly the reading "this
/// city is living off its stores".**
fn resource_row(industry: Option<&CityIndustry>, resource: Resource) -> String {
    match industry {
        None => "—".to_string(),
        Some(industry) => format!(
            "{:.0} ({:.0})",
            industry.stock(resource),
            industry.hands(resource)
        ),
    }
}

/// The resource column's label. Capitalised here rather than on [`Resource`], because
/// that label is also the word the ctl matches on and a verb a player types should not
/// carry capitals.
fn resource_label(resource: Resource) -> String {
    let label = resource.label();
    let mut chars = label.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => label.to_string(),
    }
}

/// The capacity bar's node.
///
/// It carries a material handle of its own rather than the shared one, because its fill
/// is written as the city changes and every panel would otherwise show the fill of
/// whichever city was picked last. The last values written are kept here so the readout
/// can tell whether the material needs touching at all — see [`refresh_city_panel`].
#[derive(Component)]
pub struct CityCapacityBar {
    last_fill: f32,
    last_growing: bool,
}

/// The knobs. Like `TerrainConfig` and `WeatherConfig` this outlives a session, because
/// it is taste rather than world state.
#[derive(Resource)]
pub struct CityPanelConfig {
    /// The floor on how big a city's click target is, in *screen* pixels. In tiles it is
    /// this over the tile size times the orthographic scale, so a hamlet keeps a
    /// cursor-sized target however far out you zoom.
    ///
    /// Bounded above by the three-by-three chunk scan, which stays sufficient while
    /// `MAX_CITY_RADIUS + pick_slack_px / 2 < CHUNK_SIZE` — that is, below 104. At 12 the
    /// headroom is enormous; `the_click_target_stays_inside_the_chunks_the_pick_scans`
    /// is what would catch a bump past it.
    pub pick_slack_px: f32,
    pub panel_width_px: f32,
    /// How far the panel sits from the cursor, so it does not open under it.
    pub cursor_offset_px: Vec2,
    /// The text, and the ring drawn around every glyph of it. Wood is a mid-tone with
    /// dark grain running through it, so one flat colour is legible over some of the
    /// plank and lost over the rest; the ring is what makes a glyph readable wherever it
    /// happens to fall. There is no dim ink — a second, greyer colour was what made the
    /// labels unreadable in the first place.
    pub ink: Color,
    pub outline: Color,
    pub bar_height_px: f32,
    /// What the bar burns to when the harvest covers the demand, and when it does not.
    /// The colour and the length are independent readings — a city can be nearly full
    /// *and* starving, and that is exactly the state that precedes a collapse.
    pub bar_growing: Color,
    pub bar_starving: Color,
}

impl Default for CityPanelConfig {
    fn default() -> Self {
        Self {
            pick_slack_px: 12.0,
            panel_width_px: 340.0,
            cursor_offset_px: Vec2::new(14.0, 14.0),
            ink: Color::WHITE,
            outline: Color::BLACK,
            bar_height_px: 7.0,
            bar_growing: Color::srgb(0.55, 0.78, 0.42),
            bar_starving: Color::srgb(0.83, 0.44, 0.30),
        }
    }
}

/// The shared handle every panel background is drawn with. A resource because an asset
/// needs an owner to stay alive, and built through `FromWorld` rather than in the
/// plugin's `build` so that `UiMaterialPlugin` has certainly registered the store first —
/// `add_plugins` runs a plugin's build immediately, so doing both in one `build` is an
/// ordering trap that happens to work.
#[derive(Resource)]
pub struct WoodPanelHandle(Handle<WoodPanelMaterial>);

impl FromWorld for WoodPanelHandle {
    fn from_world(world: &mut World) -> Self {
        let mut materials = world.resource_mut::<Assets<WoodPanelMaterial>>();
        Self(materials.add(WoodPanelMaterial::panel()))
    }
}

/// The wood, as the shader sees it.
///
/// This and `WoodPanelMaterial` in `assets/shaders/wood_panel.wgsl` are the same struct
/// written twice, and the field order *is* the binding layout — vectors before scalars,
/// so std140 padding agrees on both sides including under WebGL2. It lands on 112 bytes:
/// four `vec4` to 64, two `vec2` to 80, five scalars to 100, rounded to the struct's
/// 16-byte alignment.
///
/// The same discipline the weather and the tint carry, but **not** the same mechanism,
/// and following theirs would be a mistake: those are fullscreen `Core2d` passes with
/// hand-written bind group layouts and specializers. Here `AsBindGroup` derives the
/// layout from the fields and `MaterialNode` carries the handle, so what is duplicated
/// across the language boundary is the field list alone. A mismatch is a shader-compile
/// failure the first time a panel *opens* — not on entering gameplay, since a UI
/// material's pipeline is specialized when the first node carrying it is queued.
///
/// One material type, not two. The bar is the same plank seen through a groove, which is
/// what keeps its wood the *same* wood as the panel behind it rather than a second thing
/// to tune. `is_bar` is a flag rather than a negative `fill_fraction` sentinel: a
/// sentinel leaves two fields conditionally meaningless with the rule living only in a
/// comment, and a clamp written later without reading it would turn every panel into a
/// bar.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct WoodPanelMaterial {
    #[uniform(0)]
    grain_dark: Vec4,
    #[uniform(0)]
    grain_light: Vec4,
    #[uniform(0)]
    frame: Vec4,
    #[uniform(0)]
    fill: Vec4,
    #[uniform(0)]
    plank_px: Vec2,
    #[uniform(0)]
    ring_px: Vec2,
    #[uniform(0)]
    ring_warp_px: f32,
    #[uniform(0)]
    fibre_strength: f32,
    #[uniform(0)]
    bevel_px: f32,
    #[uniform(0)]
    fill_fraction: f32,
    #[uniform(0)]
    is_bar: u32,
}

impl WoodPanelMaterial {
    /// Named constructors, so no call site ever sets `is_bar` by hand.
    fn panel() -> Self {
        Self {
            grain_dark: Vec4::new(0.20, 0.11, 0.055, 1.0),
            grain_light: Vec4::new(0.46, 0.29, 0.155, 1.0),
            frame: Vec4::new(0.11, 0.06, 0.03, 1.0),
            fill: Vec4::ZERO,
            plank_px: PLANK_PX,
            ring_px: RING_PX,
            ring_warp_px: RING_WARP_PX,
            fibre_strength: FIBRE_STRENGTH,
            bevel_px: BEVEL_PX,
            fill_fraction: 0.0,
            is_bar: 0,
        }
    }

    fn bar(fill: Color, fraction: f32) -> Self {
        Self {
            fill: LinearRgba::from(fill).to_vec4(),
            fill_fraction: fraction,
            is_bar: 1,
            // A bar is a shallow groove, so it gets a smaller bevel than the panel it is
            // cut into — at the panel's the frame would swallow a 7px-high node whole.
            bevel_px: 2.0,
            ..Self::panel()
        }
    }
}

impl UiMaterial for WoodPanelMaterial {
    fn fragment_shader() -> ShaderRef {
        WOOD_SHADER_PATH.into()
    }
}

/// The three systems, gated and ordered together.
///
/// The gate is not optional and it hangs off the *set*, exactly as `WorldSystems` does
/// and for the same reason: [`CityMap`] and [`RoadNetwork`] exist only between
/// `OnEnter` and `OnExit(Screen::Gameplay)`, `Screen::Main` is the default state, and
/// `.after()` inherits an ordering edge but never a run condition. Ungated, the readout
/// would fail to find `RoadNetwork` on the app's very first frame.
///
/// Ordering after [`WorldSystems::Growth`] is not about staleness but about determinism:
/// the simulation holds `&mut City` and `&mut CityGrowth`, so without the edge the
/// executor is free to run the readout either side of the step, and to choose
/// differently on different frames.
#[derive(SystemSet, Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct CityPanelSystems;

pub struct CityPanelPlugin;

impl Plugin for CityPanelPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(UiMaterialPlugin::<WoodPanelMaterial>::default());
        app.init_resource::<CityPanelConfig>();
        app.init_resource::<WoodPanelHandle>();
        app.configure_sets(
            Update,
            CityPanelSystems
                .after(WorldSystems::Growth)
                .run_if(in_state(Screen::Gameplay)),
        );
        // Chained, and the edges are load-bearing: the rows are spawned through
        // `Commands` and read by the next system in the same schedule, which works
        // because an ordering edge out of a system with deferred work gets a sync point
        // inserted for it. `after_ignore_deferred` here would silently break that.
        app.add_systems(
            Update,
            (pick_city_on_click, refresh_city_panel, close_city_panel)
                .chain()
                .in_set(CityPanelSystems),
        );
    }
}

/// How many tiles of slack a cursor's width is worth at this zoom.
///
/// Screen pixels are only world units at a scale of 1, which is the same trick the pan
/// clamp and the chunk streamer make — hence [`orthographic_scale`] rather than a second
/// destructuring of the projection here.
fn slack_tiles(pick_slack_px: f32, scale: f32) -> f32 {
    pick_slack_px * scale / TILE_DISPLAY_SIZE.x as f32
}

/// The city a click lands on, out of the candidates the chunk scan turned up.
///
/// `click` is in continuous tile space, so a tile's centre is `t + 0.5`.
///
/// Containment first, and among the cities that contain the click the *smallest*. The
/// obvious rule — minimise distance minus radius — has it exactly backwards: a click one
/// tile inside a hamlet scores -2 against a metropolis five tiles away at -7, so it hands
/// the click to the larger city it was meant to protect the smaller one from. Ties break
/// on the lower id, so the answer cannot depend on which of the nine chunk rows happened
/// to be visited first.
fn pick_city(click: Vec2, slack: f32, candidates: &[(Entity, City)]) -> Option<Entity> {
    candidates
        .iter()
        .filter_map(|&(entity, city)| {
            let centre = city.centre.as_vec2() + Vec2::splat(0.5);
            let distance = click.distance(centre);
            let radius = city.radius as f32;
            if distance > radius + slack {
                return None;
            }
            // Inside the disc a city is ranked by how small it is, outside it by how near
            // — the tier keeps the two scales from ever being compared with each other.
            let (tier, rank) = if distance <= radius {
                (0u8, radius)
            } else {
                (1u8, distance)
            };
            Some((tier, rank, city.id, entity))
        })
        .min_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.cmp(&b.2)))
        .map(|(_, _, _, entity)| entity)
}

/// The chunks a pick has to look in: the click's own and the eight around it.
///
/// A city is always in the row of the chunk holding its *centre* — that tile is habitable
/// by construction and so always stamped — and [`CityMap`] only ever inserts, so the
/// index is monotone and over-inclusive. That, rather than any radius arithmetic, is why
/// this cannot miss; the radius bound in [`CityPanelConfig::pick_slack_px`] is what keeps
/// nine chunks enough.
///
/// The indices come from each neighbour's own origin tile rather than from a chunk-index
/// helper, because `chunk_index` is private to `world` and one caller is not enough to
/// widen it. Clamping the *coordinates* first is what matters: an out-of-range tile would
/// wrap through `as_uvec2` to the far edge of the world.
fn chunks_around(tile: IVec2) -> impl Iterator<Item = usize> {
    let centre = chunk_of_tile(tile).as_ivec2();
    let bounds = WORLD_CHUNKS.as_ivec2();
    (-1..=1)
        .flat_map(move |dy| (-1..=1).map(move |dx| centre + IVec2::new(dx, dy)))
        .filter(move |coord| coord.cmpge(IVec2::ZERO).all() && coord.cmplt(bounds).all())
        .map(|coord| chunk_index_of_tile(coord * CHUNK_SIZE.as_ivec2()))
}

/// A string in `ink`, ringed in `outline`.
///
/// **Bevy 0.19 has no text stroke.** `TextShadow` is a single offset and there is nothing
/// else, so an outline has to be drawn: the same string in black at each of the eight
/// offsets, with the light copy over the top. Wood is a mid-tone with grain running
/// through it, so a plain colour — even white — loses its edges wherever a dark ring
/// crosses it; the outline is what makes the glyphs readable against *any* part of the
/// plank rather than against the average of it.
///
/// The `extra` bundle rides on **every** copy, and that is what keeps the cost to layout
/// rather than to logic: a value's nine texts all carry the same [`CityStatValue`], so
/// the readout's ordinary query updates them without knowing an outline exists. Pass `()`
/// for a label that never changes.
///
/// The light copy is the one *in flow*, so it alone sizes the container; the ring is
/// absolutely positioned and contributes nothing to layout. It is also spawned last,
/// which is what puts it on top — bevy draws siblings in order.
fn outlined_text<B: Bundle + Clone>(
    text: String,
    font_size: f32,
    config: &CityPanelConfig,
    width: Option<f32>,
    justify: Justify,
    extra: B,
) -> impl Bundle {
    let sized = |position: PositionType, offset: Vec2| Node {
        position_type: position,
        left: px(offset.x),
        top: px(offset.y),
        width: width.map_or(Val::Auto, px),
        ..default()
    };

    let copies: Vec<_> = OUTLINE_OFFSETS
        .iter()
        .map(|&offset| (config.outline, sized(PositionType::Absolute, offset)))
        .chain(std::iter::once((
            config.ink,
            sized(PositionType::Relative, Vec2::ZERO),
        )))
        .map(|(colour, node)| {
            (
                Text::new(text.clone()),
                TextFont::from_font_size(font_size),
                TextColor(colour),
                TextLayout::justify(justify),
                node,
                extra.clone(),
            )
        })
        .collect();

    (
        Node {
            width: width.map_or(Val::Auto, px),
            ..default()
        },
        Children::spawn(SpawnIter(copies.into_iter())),
    )
}

/// Where a panel opens: beside the cursor, and never off the screen.
///
/// The direct analogue of `clamp_to_world` in `camera.rs`, and unit-testable for the same
/// reason — a city near the right edge of the window is exactly where a panel would
/// otherwise open where it cannot be read.
fn panel_origin(cursor: Vec2, offset: Vec2, panel: Vec2, window: Vec2) -> Vec2 {
    let limit = (window - panel).max(Vec2::ZERO);
    (cursor + offset).clamp(Vec2::ZERO, limit)
}

/// How full the capacity bar is.
///
/// The guard is the whole function. Population is floored at `min_population` and the
/// capacity is exactly zero on a city's first step — every city in the world — so a bare
/// `(population / capacity).clamp(0.0, 1.0)` is `inf.clamp(..)`, which is **one**: a full
/// bar at the moment a city has nothing, the opposite of what the panel is for.
fn bar_fill(population: f32, capacity: f32) -> f32 {
    if !capacity.is_finite() || capacity <= 0.0 || !population.is_finite() {
        return 0.0;
    }
    (population / capacity).clamp(0.0, 1.0)
}

/// The panel's height, summed from what it is about to contain.
///
/// The taller of the two columns, since they sit side by side — which is the whole
/// point of a second column rather than more rows: the stats column is eleven rows
/// already, and six resources under it would make the window taller than a hamlet is
/// wide on screen.
fn panel_height_px(config: &CityPanelConfig) -> f32 {
    let rows = CityStat::ALL.len().max(Resource::ALL.len()) as f32;
    PANEL_PADDING_PX * 2.0
        + HEADER_HEIGHT_PX
        + config.bar_height_px
        + ROW_HEIGHT_PX * rows
        + ROW_GAP_PX * (rows + 1.0)
}

fn pick_city_on_click(
    mut commands: Commands,
    buttons: Res<ButtonInput<MouseButton>>,
    window: Single<&Window, With<PrimaryWindow>>,
    camera: Single<(&Camera, &Transform, &Projection), With<WorldCamera>>,
    panels: Query<(Entity, &ComputedNode, &UiGlobalTransform), With<CityStatsPanel>>,
    cities: Query<(Entity, &City)>,
    city_map: Res<CityMap>,
    config: Res<CityPanelConfig>,
    wood: Res<WoodPanelHandle>,
    mut materials: ResMut<Assets<WoodPanelMaterial>>,
) {
    if buttons.get_just_pressed().next().is_none() {
        return;
    }
    let Some(cursor) = window.cursor_position() else {
        return;
    };

    // A click on the panel is not a click on the world. The node's geometry is in
    // physical pixels while the cursor is in logical ones, and on an ordinary display
    // the two agree — which is what would make getting this wrong invisible here and
    // wrong by half on a HiDPI screen.
    let physical_cursor = cursor * window.scale_factor();
    if panels
        .iter()
        .any(|(_, node, transform)| node.contains_point(*transform, physical_cursor))
    {
        return;
    }

    // Any click outside the panel dismisses it, whichever button.
    close_panels(&mut commands, panels.iter().map(|(entity, _, _)| entity));

    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }

    let (camera, camera_transform, projection) = *camera;
    // The camera's own `Transform`, not its `GlobalTransform`: the pan writes the first
    // in `Update` and propagation to the second runs in `PostUpdate`, so the global one
    // is a frame behind — about a tile at scale 1 and four at MAX_ZOOM_SCALE while a
    // movement key is held. The camera has no parent, so this conversion is exact.
    let Ok(world) = camera.viewport_to_world_2d(&GlobalTransform::from(*camera_transform), cursor)
    else {
        return;
    };

    let click = tile_position_at(world);
    let tile = click.floor().as_ivec2();
    // `tile_position_at` is deliberately unclamped and `chunk_of_tile` sends a negative
    // tile through `as_uvec2`, which wraps it to the *far* edge of the world. Unguarded,
    // a click just off the left border would scan the wrong nine chunks and miss a city
    // that really is under the cursor.
    if !tile_in_world(tile) {
        return;
    }

    let mut candidates: Vec<(Entity, City)> = Vec::new();
    for chunk in chunks_around(tile) {
        for &entity in city_map.in_chunk(chunk) {
            if candidates.iter().any(|(known, _)| *known == entity) {
                continue;
            }
            if let Ok((entity, city)) = cities.get(entity) {
                candidates.push((entity, *city));
            }
        }
    }

    let slack = slack_tiles(config.pick_slack_px, orthographic_scale(projection));
    let Some(city_entity) = pick_city(click, slack, &candidates) else {
        return;
    };
    let Ok((_, city)) = cities.get(city_entity) else {
        return;
    };

    spawn_panel(
        &mut commands,
        &config,
        &wood,
        &mut materials,
        city_entity,
        city,
        cursor,
        window.size(),
    );
}

fn spawn_panel(
    commands: &mut Commands,
    config: &CityPanelConfig,
    wood: &WoodPanelHandle,
    materials: &mut Assets<WoodPanelMaterial>,
    city_entity: Entity,
    city: &City,
    cursor: Vec2,
    window: Vec2,
) {
    let size = Vec2::new(config.panel_width_px, panel_height_px(config));
    let origin = panel_origin(cursor, config.cursor_offset_px, size, window);
    let bar = materials.add(WoodPanelMaterial::bar(config.bar_growing, 0.0));

    commands
        .spawn((
            CityStatsPanel { city: city_entity },
            DespawnOnExit(Screen::Gameplay),
            Node {
                position_type: PositionType::Absolute,
                left: px(origin.x),
                top: px(origin.y),
                width: px(size.x),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(px(PANEL_PADDING_PX)),
                row_gap: px(ROW_GAP_PX),
                // The corner the layout reserves; the shader draws the same one from
                // `UiVertexOutput.border_radius`, so the two cannot disagree.
                border_radius: BorderRadius::all(px(6.0)),
                ..default()
            },
            MaterialNode(wood.0.clone()),
        ))
        .with_children(|panel| {
            // The id is the one thing about a city that never changes, so it is the one
            // thing that can be written once. The tier is a row, because it moves.
            //
            // Wrapped in a node of its own so the header keeps an explicit height: the
            // outline's container is sized by the text inside it, and `panel_height_px`
            // has to stay arithmetic rather than become a guess about line heights.
            panel
                .spawn(Node {
                    height: px(HEADER_HEIGHT_PX),
                    ..default()
                })
                .with_children(|header| {
                    header.spawn(outlined_text(
                        format!("City #{}", city.id),
                        HEADER_FONT_PX,
                        config,
                        None,
                        Justify::Left,
                        (),
                    ));
                });
            panel.spawn((
                CityCapacityBar {
                    last_fill: 0.0,
                    last_growing: true,
                },
                Node {
                    width: percent(100),
                    height: px(config.bar_height_px),
                    border_radius: BorderRadius::all(px(2.0)),
                    ..default()
                },
                MaterialNode(bar),
            ));
            // Two columns side by side. Both are spawned with their true strings rather
            // than placeholders, so nothing about what is displayed depends on when the
            // sync point for these commands happens to fall. Every value is fixed width
            // and right aligned, so a population crossing from 999 to 1000 does not
            // shove its own label sideways.
            panel
                .spawn(Node {
                    width: percent(100),
                    column_gap: px(COLUMN_GAP_PX),
                    ..default()
                })
                .with_children(|body| {
                    body.spawn(Node {
                        width: px(STATS_COLUMN_PX),
                        flex_direction: FlexDirection::Column,
                        row_gap: px(ROW_GAP_PX),
                        ..default()
                    })
                    .with_children(|column| {
                        for stat in CityStat::ALL {
                            spawn_row(
                                column,
                                config,
                                stat.label().to_string(),
                                stat.format(city, None, None, None),
                                CityStatValue(stat),
                            );
                        }
                    });

                    body.spawn(Node {
                        width: px(RESOURCE_COLUMN_PX),
                        flex_direction: FlexDirection::Column,
                        row_gap: px(ROW_GAP_PX),
                        ..default()
                    })
                    .with_children(|column| {
                        // Walked from `Resource::ALL`, never from a hand-written list of
                        // six — so a seventh resource is a table row in `deposit.rs` and
                        // nothing here.
                        for resource in Resource::ALL {
                            spawn_row(
                                column,
                                config,
                                resource_label(resource),
                                resource_row(None, resource),
                                CityResourceValue(resource),
                            );
                        }
                    });
                });
        });
}

/// One label-and-value row, in whichever column. Shared so the two columns cannot drift
/// in height or alignment, which is what `panel_height_px` assumes of both.
fn spawn_row(
    parent: &mut ChildSpawnerCommands,
    config: &CityPanelConfig,
    label: String,
    value: String,
    marker: impl Bundle + Clone,
) {
    parent
        .spawn(Node {
            width: percent(100),
            height: px(ROW_HEIGHT_PX),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            ..default()
        })
        .with_children(|row| {
            row.spawn(outlined_text(
                label,
                ROW_FONT_PX,
                config,
                None,
                Justify::Left,
                (),
            ));
            row.spawn(outlined_text(
                value,
                ROW_FONT_PX,
                config,
                Some(VALUE_COLUMN_PX),
                Justify::Right,
                marker,
            ));
        });
}

/// Re-reads the city every frame and writes only what changed.
///
/// The *reading* is what must not be filtered: the simulation writes [`CityGrowth`] every
/// step, and a panel that lagged would be visibly stale beside a city that is visibly
/// growing. The *writing* is another matter. Touching a [`Text`] marks it changed and
/// costs two full text layout passes, and touching the material clones it into the render
/// world and allocates a fresh uniform buffer and bind group. The numbers move twice a
/// second and the panel redraws sixty times a second, so writing unconditionally would do
/// that work about thirty times over for nothing.
fn refresh_city_panel(
    mut commands: Commands,
    panel: Option<Single<(Entity, &CityStatsPanel)>>,
    cities: Query<(&City, Option<&CityGrowth>, Option<&CityIndustry>)>,
    roads: Res<RoadNetwork>,
    config: Res<CityPanelConfig>,
    mut values: Query<(&CityStatValue, &mut Text)>,
    mut resources: Query<(&CityResourceValue, &mut Text), Without<CityStatValue>>,
    mut bars: Query<(&mut CityCapacityBar, &MaterialNode<WoodPanelMaterial>)>,
    mut materials: ResMut<Assets<WoodPanelMaterial>>,
) {
    let Some(panel) = panel else {
        return;
    };
    let (panel_entity, panel_state) = *panel;

    let Ok((city, growth, industry)) = cities.get(panel_state.city) else {
        // Nothing despawns a city mid-session — the cities and this panel go together on
        // leaving gameplay — so this is the shape of the guard rather than a live path.
        // It is not a substitute for the state gate on the set.
        close_panels(&mut commands, [panel_entity]);
        return;
    };

    let links = roads
        .links
        .iter()
        .filter(|link| link.from == city.id || link.to == city.id)
        .count();

    for (value, mut text) in &mut values {
        text.set_if_neq(Text(value.0.format(city, growth, industry, Some(links))));
    }
    for (value, mut text) in &mut resources {
        text.set_if_neq(Text(resource_row(industry, value.0)));
    }

    let (fill, growing) = match growth {
        Some(growth) => (
            bar_fill(growth.population, growth.capacity),
            growth.food >= growth.demand,
        ),
        None => (0.0, true),
    };
    for (mut bar, node) in &mut bars {
        if bar.last_fill == fill && bar.last_growing == growing {
            continue;
        }
        let colour = if growing {
            config.bar_growing
        } else {
            config.bar_starving
        };
        if let Some(mut material) = materials.get_mut(&node.0) {
            *material = WoodPanelMaterial::bar(colour, fill);
        }
        bar.last_fill = fill;
        bar.last_growing = growing;
    }
}

/// Escape shuts the panel.
///
/// It consumes the key rather than sharing it, which is safe because there is no way out
/// of gameplay to compete with — `handle_escape_help` is gated on `Screen::Help`. Whoever
/// adds a gameplay exit has to decide the precedence, and the panel should almost
/// certainly win while it is open.
fn close_city_panel(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    panels: Query<Entity, With<CityStatsPanel>>,
) {
    if !keys.just_pressed(KeyCode::Escape) {
        return;
    }
    close_panels(&mut commands, panels);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{MAX_ZOOM_SCALE, MIN_ZOOM_SCALE};
    use crate::gameplay::city::MAX_CITY_RADIUS;

    fn city(id: u32, centre: IVec2, radius: u32) -> City {
        City {
            id,
            centre,
            size: CitySize::from_radius(radius),
            radius,
        }
    }

    fn at(tile: IVec2) -> Vec2 {
        tile.as_vec2() + Vec2::splat(0.5)
    }

    #[test]
    fn a_click_inside_a_citys_disc_finds_that_city() {
        let entity = Entity::from_raw_u32(1).unwrap();
        let candidates = [(entity, city(0, IVec2::new(100, 100), 5))];
        assert_eq!(
            pick_city(at(IVec2::new(102, 101)), 0.0, &candidates),
            Some(entity)
        );
    }

    #[test]
    fn a_click_on_open_ground_finds_no_city() {
        let entity = Entity::from_raw_u32(1).unwrap();
        let candidates = [(entity, city(0, IVec2::new(100, 100), 5))];
        assert_eq!(pick_city(at(IVec2::new(140, 100)), 2.0, &candidates), None);
    }

    /// The slack exists so that zooming out cannot shrink a city past the cursor. In
    /// tiles it therefore has to *grow* as the scale does.
    #[test]
    fn the_click_target_of_a_city_never_shrinks_as_the_world_zooms_out() {
        let slack_px = 12.0;
        let close = slack_tiles(slack_px, MIN_ZOOM_SCALE);
        let far = slack_tiles(slack_px, MAX_ZOOM_SCALE);
        assert!(far > close, "{far} should exceed {close}");

        // A hamlet is 3 tiles across. Zoomed all the way out that is 6 screen pixels, so
        // without the slack there would be nothing to click.
        let hamlet = 3.0;
        assert!(
            hamlet + far >= slack_px / 2.0,
            "a hamlet must stay a cursor wide at every zoom step"
        );
    }

    /// The rule the obvious one gets wrong: minimising distance *minus* radius would hand
    /// this click to the metropolis.
    #[test]
    fn a_click_inside_two_overlapping_cities_takes_the_smaller_one() {
        let hamlet = Entity::from_raw_u32(1).unwrap();
        let metropolis = Entity::from_raw_u32(2).unwrap();
        let candidates = [
            (hamlet, city(0, IVec2::new(100, 100), 3)),
            (metropolis, city(1, IVec2::new(105, 100), 12)),
        ];
        assert_eq!(
            pick_city(at(IVec2::new(101, 100)), 0.0, &candidates),
            Some(hamlet)
        );
    }

    #[test]
    fn two_cities_the_same_distance_from_a_click_are_separated_by_their_id() {
        let first = Entity::from_raw_u32(7).unwrap();
        let second = Entity::from_raw_u32(3).unwrap();
        let candidates = [
            (first, city(4, IVec2::new(100, 100), 5)),
            (second, city(9, IVec2::new(110, 100), 5)),
        ];
        // Exactly between the two, and outside both discs.
        let click = Vec2::new(105.5, 100.5);
        assert_eq!(pick_city(click, 8.0, &candidates), Some(first));
    }

    /// A city outside its own disc is still reachable through the slack, which is what
    /// makes a hamlet clickable when it is six pixels across.
    #[test]
    fn a_click_just_outside_a_small_city_still_finds_it_through_the_slack() {
        let entity = Entity::from_raw_u32(1).unwrap();
        // Five tiles from the centre of a city whose radius is three, so the click is two
        // tiles outside it and only a slack of at least two reaches.
        let candidates = [(entity, city(0, IVec2::new(100, 100), 3))];
        assert_eq!(pick_city(at(IVec2::new(105, 100)), 1.0, &candidates), None);
        assert_eq!(
            pick_city(at(IVec2::new(105, 100)), 4.0, &candidates),
            Some(entity)
        );
    }

    /// The nine chunks are only enough while the reach of a pick stays under a chunk.
    /// This is the assertion that a later bump of `pick_slack_px` would trip.
    #[test]
    fn the_click_target_stays_inside_the_chunks_the_pick_scans() {
        let config = CityPanelConfig::default();
        let widest = MAX_CITY_RADIUS as f32 + slack_tiles(config.pick_slack_px, MAX_ZOOM_SCALE);
        assert!(
            widest < CHUNK_SIZE.x as f32,
            "a pick reaching {widest} tiles would need more than the nine chunks scanned"
        );
    }

    #[test]
    fn every_chunk_a_pick_scans_is_inside_the_world() {
        let corner = chunks_around(IVec2::ZERO).count();
        assert_eq!(corner, 4, "a corner has only four chunks around it");

        let far = IVec2::new(
            WORLD_CHUNKS.x as i32 * CHUNK_SIZE.x as i32 - 1,
            WORLD_CHUNKS.y as i32 * CHUNK_SIZE.y as i32 - 1,
        );
        assert_eq!(chunks_around(far).count(), 4);
        assert_eq!(chunks_around(IVec2::new(2000, 2000)).count(), 9);
    }

    /// The defect the draft shipped: population is floored above zero and capacity is
    /// exactly zero on a city's first step, so the division is infinite and a plain clamp
    /// reads it as *full*.
    #[test]
    fn a_city_with_no_capacity_shows_an_empty_bar_rather_than_a_full_one() {
        assert_eq!(bar_fill(20.0, 0.0), 0.0);
        assert_eq!(bar_fill(20.0, -1.0), 0.0);
        assert_eq!(bar_fill(f32::NAN, 10.0), 0.0);
    }

    #[test]
    fn a_population_over_its_capacity_fills_the_bar_and_no_further() {
        assert_eq!(bar_fill(300.0, 100.0), 1.0);
        assert_eq!(bar_fill(50.0, 100.0), 0.5);
    }

    /// The two columns sit side by side, so the panel has to be wide enough to hold
    /// both — otherwise flexbox wraps the resource column under the stats one and the
    /// height `panel_height_px` computed is a lie.
    #[test]
    fn the_panel_is_wide_enough_for_both_its_columns() {
        let config = CityPanelConfig::default();
        let needed = STATS_COLUMN_PX + COLUMN_GAP_PX + RESOURCE_COLUMN_PX + PANEL_PADDING_PX * 2.0;
        assert!(
            config.panel_width_px >= needed,
            "the panel is {} px wide and its columns need {needed}",
            config.panel_width_px
        );
        // And each column's value text has to fit inside its own column, or the label
        // beside it is squeezed to nothing.
        assert!(VALUE_COLUMN_PX < RESOURCE_COLUMN_PX.min(STATS_COLUMN_PX));
    }

    /// Every resource gets a row, and it is built by walking `Resource::ALL` rather
    /// than by a hand-written list — so a seventh resource is a table row in
    /// `deposit.rs` and nothing in this module.
    #[test]
    fn the_resource_column_names_its_resource_and_shows_the_hands_beside_the_stock() {
        assert_eq!(resource_label(Resource::Food), "Food");
        assert_eq!(resource_label(Resource::Copper), "Copper");
        for resource in Resource::ALL {
            let label = resource_label(resource);
            assert!(
                label.starts_with(|c: char| c.is_uppercase()),
                "{label} is not capitalised for the panel"
            );
            assert_eq!(label.to_lowercase(), resource.label(), "{label} drifted");
        }
    }

    #[test]
    fn a_panel_opens_fully_inside_the_window_wherever_the_cursor_is() {
        let window = Vec2::new(1920.0, 1080.0);
        let panel = Vec2::new(190.0, 240.0);
        let offset = Vec2::splat(14.0);

        let middle = panel_origin(Vec2::new(400.0, 400.0), offset, panel, window);
        assert_eq!(middle, Vec2::new(414.0, 414.0));

        let corner = panel_origin(window, offset, panel, window);
        assert_eq!(corner, window - panel);
        assert!((corner + panel).cmple(window).all());
    }

    /// A window smaller than the panel would otherwise give an inverted clamp range,
    /// which `Vec2::clamp` treats as a contract violation.
    #[test]
    fn a_window_smaller_than_the_panel_pins_it_to_the_corner() {
        let panel = Vec2::new(400.0, 400.0);
        assert_eq!(
            panel_origin(Vec2::splat(50.0), Vec2::ZERO, panel, Vec2::splat(100.0)),
            Vec2::ZERO
        );
    }

    /// The growth rows are dashes until the plan reaches Done, but the city's own facts
    /// are true from the moment it is spawned.
    #[test]
    fn a_city_without_growth_shows_dashes_rather_than_nothing() {
        let city = city(7, IVec2::new(-12, 40), 8);
        assert_eq!(CityStat::Population.format(&city, None, None, None), "—");
        assert_eq!(CityStat::Tier.format(&city, None, None, None), "Borough");
        assert_eq!(CityStat::Centre.format(&city, None, None, None), "-12, 40");
        // Roads are known before the growth is, and each is missing on its own terms.
        assert_eq!(CityStat::Roads.format(&city, None, None, Some(3)), "3");
        // The industry half reads unknown on exactly the same terms, and a city is
        // clickable through the whole road-planning stage during which it is.
        assert_eq!(CityStat::Happiness.format(&city, None, None, None), "—");
        assert_eq!(CityStat::Idle.format(&city, None, None, None), "—");
        for resource in Resource::ALL {
            assert_eq!(resource_row(None, resource), "—");
        }
        assert_eq!(CityStat::Roads.format(&city, None, None, None), "—");
    }
}
