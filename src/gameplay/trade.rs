//! Independent traders, their caravans, and the goods and money they move.
//!
//! The first thing in the crate that is a *game* rather than a world. Everything
//! below it either generates terrain once or advances a city against its own land;
//! no two cities have ever exchanged anything, and the road network has been
//! decoration since it was laid. A caravan is the first agent with a purse, a choice
//! and a position on screen.
//!
//! **The road network is the whole reason this is cheap.** [`road`] already decides
//! which cities are worth joining and where the road between two of them runs, and
//! the Gabriel graph is a near-planar neighbour graph — a city has two or three links,
//! so a one-hop choice is a handful of comparisons. Nothing here pathfinds. What it
//! needed was for `route_road` to *keep* the tile list it already builds, which is
//! gh-7's one change outside these two modules.
//!
//! **What the economy already gave us for free, and it is most of the feature.**
//! Three properties of [`industry`] are exactly the pressures a trade system needs,
//! and none of them had to be invented:
//!
//! - A store is clamped to its cap and **the overflow is discarded**. A city on a rich
//!   seam already destroys iron every step, so "buy where it is worthless" is a
//!   mechanic that had to be *noticed* rather than written.
//! - gh-24 measured **20 cities of 92** working any seam. Four in five hold iron,
//!   copper and salt at zero for the whole session, are permanently short on the
//!   upkeep and comfort baskets, and are unhappy for it — and happiness scales the
//!   growth rate. A caravan of iron is worth *population* to the city that receives
//!   it, through a channel that already exists.
//! - Food is a granary since gh-24, and the capacity is measured against what is on
//!   the table. Grain arriving in a dry spell stops K falling, via
//!   `Labour::granary_release`, with nothing new plumbed.
//!
//! So the payoff is not that goods move. It is that the map's *geography of scarcity*,
//! which gh-24 laid down and left inert, starts paying out.
//!
//! **One hop, priced at both ends.** A caravan standing in a city values every
//! resource here against every road neighbour, takes the best positive margin net of
//! the journey, and goes. The lookahead is exactly one hop deliberately: a multi-hop
//! circuit search would want prices *at arrival*, and this world has no way to
//! forecast one. The prices it reads are the ones now, and they will have moved by the
//! time it gets there — that staleness is the whole of the risk, and it is what makes
//! a caravan occasionally arrive, refuse to sell, and carry on.
//!
//! **A trader will not sell at a loss, and that is why cargo persists.** A [`Lot`]
//! carries what it cost as a weighted average, and a sale is refused below
//! `paid * (1 + trader_margin)`. The issue's "if not sold, the resource stays in the
//! caravan and can be transported to the next city" only means anything if a sale can
//! be refused, and a caravan holding goods nobody here will pay for departs for
//! wherever they are worth most even when it has nothing to buy.
//!
//! **A caravan may not buy back what it just sold, and that is not a detail.** Selling
//! raises the city's stock, which lowers its price — so without the rule a caravan
//! could sell high, buy the same goods back cheap from the market it just moved, and
//! repeat: a pump that mints money out of the price curve and drains the treasury it
//! is standing in. The slice keeps each turn of it small, which would have made it a
//! slow leak rather than an obvious bug. [`Errand::Resting`] carries the flags for the
//! whole stay.
//!
//! **What is drawn, and what draws it.** A caravan is an ordinary [`Sprite`] — flat
//! colour, no art — whose translation is a linear walk along the road's stored tile
//! path. It is the first sprite in the crate; everything drawn so far is a chunk mesh
//! or UI. Two things fall out of that and neither is a defect:
//!
//! - **A caravan is lit like the ground it stands on.** Sprites land in the
//!   `ViewTarget` before the one post-process pass, so the height ramp, the sun, a
//!   cloud shadow and lying snow all apply to it. That is correct — it is a wagon on a
//!   hill at dusk — and it is free.
//! - **The inspection overlay hides caravans**, because the overlay short-circuits to
//!   false colour before the composite. Also correct: the overlay replaces the world,
//!   and a wagon is part of the world.
//!
//! The sprite has a floor in *screen* pixels, exactly as `city_panel.rs`'s pick target
//! does and for the same reason: at `MAX_ZOOM_SCALE` a 16-px dot is 4 px.
//!
//! **A caravan's speed is not on the growth clock.** It is tiles per real second,
//! driven by `Time::delta` directly. gh-7 slowed the economy eightfold precisely so
//! that a journey would span many steps; slowing the journeys by the same factor would
//! have bought nothing at all, and it is one line away from being undone.

use bevy::{platform::collections::HashMap, prelude::*};

use crate::{
    camera::{WorldCamera, orthographic_scale},
    gameplay::{
        city::City,
        deposit::{RESOURCE_COUNT, Resource},
        growth::{CityGrowth, GrowthClock, GrowthConfig},
        industry::{CityIndustry, IndustryConfig, consumption, store_cap},
        market::{CityTreasury, MarketConfig, price, purchase, sale},
        noise::hash2,
        plan::WorldPlan,
        road::RoadNetwork,
        world::{WorldSystems, tile_translation},
    },
    screens::Screen,
};

/// Knobs for the traders. Configuration rather than world state, so like
/// [`MarketConfig`] this outlives a session.
#[derive(Resource, Clone)]
pub struct TradeConfig {
    /// **Sized against the number of *roads*, not the number of cities.** The default
    /// world routes 71 roads, so 80 wagons is about one per road — which is what makes
    /// the network look used from the ground rather than empty with occasional traffic,
    /// and it is what lets a screenful of country reliably contain one. The simulation
    /// cost is a rounding error: a wagon does a handful of price comparisons when it
    /// arrives somewhere and nothing at all while it is moving.
    pub trader_count: u32,
    /// Wagons to a trader. More than one is what makes a [`Trader`] an entity rather
    /// than a colour: they share one purse, so a trader that has overreached has all of
    /// them idle at once.
    pub caravans_per_trader: u32,
    /// What a trader opens with. Enough to fill a wagon with the dearest goods in the
    /// world at the glut price, or the first journey of the session cannot happen.
    pub trader_start_money: f32,
    /// The margin over what it paid below which a trader will not sell.
    ///
    /// Zero would have a caravan dump its load at cost the moment a price ticked over,
    /// and there would be no such thing as carrying goods past a city.
    pub trader_margin: f32,
    /// What one wagon holds, in units, across every resource it carries at once.
    ///
    /// Sized against what a city *consumes*, not against what it stores: at the
    /// shipped defaults a large city's upkeep is ~113 iron a step, so a full wagon is
    /// about eighteen steps of it — one caravan can keep roughly one city supplied,
    /// which is the scale that makes 36 wagons visible across 92 cities.
    pub caravan_capacity: f32,
    /// Tiles per **real** second. See the module note: this is deliberately not on the
    /// growth clock.
    ///
    /// At 12 a wagon crosses a 1280-px screen at zoom 1 in about thirteen seconds —
    /// motion you can see rather than a blur or a creep — and a 300-tile road takes
    /// 25 s, which is six economy steps at the shipped `step_seconds`.
    pub caravan_speed_tiles: f32,
    /// How long a caravan stands in a city before loading and leaving.
    ///
    /// Long enough to see it stopped, and it is also what stops a wagon bouncing
    /// between two neighbours every frame when a margin is marginal.
    pub caravan_stop_seconds: f32,
    /// What a journey costs per tile, in money, against the margin it is expected to
    /// earn. Not deducted from anyone — nobody is paid it — it is the bar a trip has
    /// to clear, and the only thing stopping a caravan shuttling one unit back and
    /// forth over a two-coin difference.
    pub journey_cost_per_tile: f32,
    /// The drawn size in tiles, and the floor it may not shrink below on screen.
    pub caravan_sprite_tiles: f32,
    pub caravan_min_screen_px: f32,
}

impl Default for TradeConfig {
    fn default() -> Self {
        Self {
            trader_count: 20,
            caravans_per_trader: 4,
            trader_start_money: 20_000.0,
            trader_margin: 0.10,
            caravan_capacity: 2000.0,
            caravan_speed_tiles: 12.0,
            caravan_stop_seconds: 3.0,
            journey_cost_per_tile: 2.0,
            caravan_sprite_tiles: 2.0,
            caravan_min_screen_px: 6.0,
        }
    }
}

/// One independent trader.
///
/// The purse is here and not on the caravan, so a trader's wagons compete for one pot
/// and a trader that has overreached has all of them idle at once. That is the only
/// thing in the feature that makes a *trader* an entity rather than a label — without
/// a shared purse, `Trader` would be a colour.
#[derive(Component, Debug)]
pub struct Trader {
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub id: u32,
    money: f32,
}

impl Trader {
    pub fn money(&self) -> f32 {
        self.money
    }

    fn transfer(&mut self, amount: f32) {
        self.money = (self.money + amount).max(0.0);
    }
}

/// What a caravan holds of one resource, and what it cost.
///
/// `paid_per_unit` is a weighted average over everything bought into the lot. There is
/// no per-purchase history and none is wanted: the only question ever asked of it is
/// "would selling here be a loss", and an average answers that.
#[derive(Clone, Copy, Debug)]
pub struct Lot {
    pub resource: Resource,
    pub units: f32,
    pub paid_per_unit: f32,
}

/// Where a caravan is along a road.
///
/// `travelled_tiles` rather than a 0..1 fraction, so a speed is in tiles per second
/// and reads the same on a short road and a long one.
#[derive(Clone, Copy, Debug)]
pub struct Leg {
    /// Index into [`RoadNetwork::links`].
    pub link: usize,
    /// Whether the walk runs backwards along the stored path — a link is stored once
    /// and travelled both ways.
    pub reversed: bool,
    pub travelled_tiles: f32,
    /// Carried rather than looked up each frame. It is the length of the link's path
    /// and could be read back off [`RoadNetwork`], but a leg that knows how long it is
    /// cannot be advanced against the wrong road.
    pub length_tiles: f32,
}

/// What a caravan is doing. Exhaustive rather than a bag of options: it is standing in
/// a city, or it is on a road between two.
#[derive(Clone, Copy, Debug)]
pub enum Errand {
    Resting {
        city: Entity,
        seconds_left: f32,
        /// Which resources have been sold during *this* stay, and so may not be bought
        /// back here. See the module note — without this the price curve is a pump.
        sold: [bool; RESOURCE_COUNT],
    },
    Travelling {
        leg: Leg,
        to: Entity,
    },
}

/// One wagon. An entity with a [`Sprite`] and a [`Transform`], so what is simulated
/// and what is drawn are the same thing and cannot drift apart by a frame.
#[derive(Component, Debug)]
pub struct Caravan {
    pub trader: Entity,
    pub errand: Errand,
    cargo: Vec<Lot>,
}

impl Caravan {
    /// A wagon with no trader behind it and nowhere to be, for the whole-world
    /// measurement — which runs the pure trade functions against a `Vec` of cities
    /// rather than against an app, so it has no entity to name.
    #[cfg(test)]
    pub fn empty() -> Self {
        Self {
            trader: Entity::PLACEHOLDER,
            errand: Errand::Resting {
                city: Entity::PLACEHOLDER,
                seconds_left: 0.0,
                sold: [false; RESOURCE_COUNT],
            },
            cargo: Vec::new(),
        }
    }

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn cargo(&self) -> &[Lot] {
        &self.cargo
    }

    pub fn carried(&self) -> f32 {
        // Folded from a positive zero rather than summed, so an empty hold reports 0
        // and not the -0.0 an empty `f32` sum can serialise to — which is noise in a
        // scenario's diff and nothing else.
        self.cargo.iter().fold(0.0, |total, lot| total + lot.units)
    }

    fn space(&self, capacity: f32) -> f32 {
        (capacity - self.carried()).max(0.0)
    }

    /// Adds bought goods to the lot for that resource, or opens one.
    fn load(&mut self, resource: Resource, units: f32, price_per_unit: f32) {
        match self.cargo.iter_mut().find(|lot| lot.resource == resource) {
            Some(lot) => {
                let total = lot.units + units;
                lot.paid_per_unit = (lot.units * lot.paid_per_unit + units * price_per_unit)
                    / total.max(f32::EPSILON);
                lot.units = total;
            }
            None => self.cargo.push(Lot {
                resource,
                units,
                paid_per_unit: price_per_unit,
            }),
        }
    }

    /// Takes sold goods out, dropping a lot that is empty. `paid_per_unit` is left
    /// alone on a partial sale — selling does not change what the rest cost.
    fn unload(&mut self, resource: Resource, units: f32) {
        if let Some(index) = self.cargo.iter().position(|lot| lot.resource == resource) {
            let lot = &mut self.cargo[index];
            lot.units -= units;
            if lot.units <= f32::EPSILON {
                self.cargo.swap_remove(index);
            }
        }
    }
}

/// One road leaving one city, as a caravan needs it: which link, which way along it,
/// where it comes out and how far it is.
#[derive(Clone, Copy, Debug)]
pub struct RoadEdge {
    pub link: usize,
    pub reversed: bool,
    pub to: Entity,
    pub length_tiles: f32,
}

/// The road network keyed by city *entity* rather than by city id.
///
/// [`RoadNetwork`] stores links as pairs of ids because that is what the planner has,
/// and resolving an id to an entity on every decision would be a scan of every city in
/// the world. Built once when the traders are seeded, which is safe because nothing
/// spawns or despawns a city after the plan is done.
#[derive(Resource, Default)]
pub struct RoadGraph {
    edges: HashMap<Entity, Vec<RoadEdge>>,
}

impl RoadGraph {
    pub fn edges_from(&self, city: Entity) -> &[RoadEdge] {
        self.edges.get(&city).map_or(&[], Vec::as_slice)
    }

    fn build(network: &RoadNetwork, by_id: &HashMap<u32, Entity>) -> Self {
        let mut edges: HashMap<Entity, Vec<RoadEdge>> = HashMap::default();
        for (index, link) in network.links.iter().enumerate() {
            // A link whose path is a single tile is not a road anybody can walk, and
            // interpolating along it would divide by its zero length.
            if link.path.len() < 2 {
                continue;
            }
            let (Some(&from), Some(&to)) = (by_id.get(&link.from), by_id.get(&link.to)) else {
                continue;
            };
            let length = link.length_tiles();
            edges.entry(from).or_default().push(RoadEdge {
                link: index,
                reversed: false,
                to,
                length_tiles: length,
            });
            edges.entry(to).or_default().push(RoadEdge {
                link: index,
                reversed: true,
                to: from,
                length_tiles: length,
            });
        }
        Self { edges }
    }
}

/// When each city was last paid.
///
/// The income is per *step*, and the growth loop runs whole steps out of elapsed time
/// — several in one frame, or none, and it drops the backlog of a stall. So paying per
/// frame would hand a fast machine more money than a slow one, and paying by elapsed
/// time would pay for steps that were dropped. Session state, so it goes on `OnExit`.
#[derive(Resource, Default)]
struct IncomeLedger {
    last_step: u64,
}

pub struct TradePlugin;

impl Plugin for TradePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MarketConfig>();
        app.init_resource::<TradeConfig>();
        app.add_systems(OnEnter(Screen::Gameplay), start_trade);
        app.add_systems(OnExit(Screen::Gameplay), tear_down_trade);
        app.add_systems(
            Update,
            (
                seed_traders,
                collect_city_income,
                drive_caravans,
                place_caravans,
            )
                .chain()
                .in_set(WorldSystems::Trade),
        );
    }
}

fn start_trade(mut commands: Commands) {
    commands.insert_resource(IncomeLedger::default());
}

/// The traders and their wagons carry `DespawnOnExit(Screen::Gameplay)`, so all that
/// is dropped here is the index and the ledger. A purse can never be inherited by the
/// next world.
fn tear_down_trade(mut commands: Commands) {
    commands.remove_resource::<RoadGraph>();
    commands.remove_resource::<IncomeLedger>();
}

/// Hands every city a purse and puts the traders on the map, once there is a road
/// network to trade over.
///
/// Runs once: the presence of [`RoadGraph`] is what says it has, so there is no flag
/// to keep in step with the world.
fn seed_traders(
    mut commands: Commands,
    graph: Option<Res<RoadGraph>>,
    plan: Res<WorldPlan>,
    network: Res<RoadNetwork>,
    config: Res<TradeConfig>,
    market: Res<MarketConfig>,
    cities: Query<(Entity, &City, &CityGrowth)>,
) {
    if graph.is_some() || !matches!(*plan, WorldPlan::Done) {
        return;
    }
    // The growth step seeds the cities on the first frame after the plan is done, and
    // a city with no `CityGrowth` has no town to size a purse from.
    if cities.is_empty() {
        return;
    }

    let mut by_id = HashMap::default();
    for (entity, city, growth) in &cities {
        by_id.insert(city.id, entity);
        commands.entity(entity).insert(CityTreasury::new(
            growth.town() as f32 * market.city_start_money_per_town_tile,
        ));
    }

    let graph = RoadGraph::build(&network, &by_id);

    // Only cities with a road are worth starting in, and in id order so the same seed
    // puts the same wagons in the same places — the simulation stopped being
    // reproducible at `growth.rs`, but a spawn table is free to be and there is no
    // reason to make it otherwise.
    let mut connected: Vec<(u32, Entity)> = cities
        .iter()
        .filter(|(entity, ..)| !graph.edges_from(*entity).is_empty())
        .map(|(entity, city, _)| (city.id, entity))
        .collect();
    connected.sort_unstable_by_key(|(id, _)| *id);
    let connected: Vec<Entity> = connected.into_iter().map(|(_, entity)| entity).collect();
    if connected.is_empty() {
        // An island world with no roads at all. The graph still goes in, so this does
        // not run again every frame looking for one.
        commands.insert_resource(graph);
        return;
    }

    for index in 0..config.trader_count {
        let hue = 360.0 * index as f32 / config.trader_count.max(1) as f32;
        let colour = Color::hsl(hue, 0.85, 0.55);
        let trader = commands
            .spawn((
                Trader {
                    id: index,
                    money: config.trader_start_money,
                },
                DespawnOnExit(Screen::Gameplay),
            ))
            .id();

        for wagon in 0..config.caravans_per_trader {
            let pick = hash2(index as i32, wagon as i32) as usize % connected.len();
            let city = connected[pick];
            commands.spawn((
                Caravan {
                    trader,
                    errand: Errand::Resting {
                        city,
                        seconds_left: config.caravan_stop_seconds,
                        sold: [false; RESOURCE_COUNT],
                    },
                    cargo: Vec::new(),
                },
                Sprite::from_color(
                    colour,
                    Vec2::splat(config.caravan_sprite_tiles * TILE_PIXELS),
                ),
                // Above the chunk meshes, which sit at z 0.
                Transform::from_translation(Vec3::new(0.0, 0.0, CARAVAN_Z)),
                DespawnOnExit(Screen::Gameplay),
            ));
        }
    }

    commands.insert_resource(graph);
}

/// Mints each city its income for however many steps have passed since it last ran.
///
/// A system of its own rather than a line inside the growth loop, because the
/// ownership rule that keeps `CityGrowth` and `CityIndustry` apart applies to the
/// purse too: nothing in `growth.rs` may learn that money exists. The cost is one
/// accessor on [`GrowthClock`].
fn collect_city_income(
    ledger: Option<ResMut<IncomeLedger>>,
    clock: Option<Res<GrowthClock>>,
    market: Res<MarketConfig>,
    mut cities: Query<(&CityGrowth, &mut CityTreasury)>,
) {
    let (Some(mut ledger), Some(clock)) = (ledger, clock) else {
        return;
    };
    let steps = clock.step().saturating_sub(ledger.last_step);
    if steps == 0 {
        return;
    }
    ledger.last_step = clock.step();

    for (growth, mut treasury) in &mut cities {
        treasury.transfer(growth.population * market.city_income_per_person * steps as f32);
    }
}

/// What a caravan needs to know about a city in order to deal with it.
///
/// **The whole of the trade logic is written against this and not against the ECS**, on
/// exactly the terms `step_city` takes the sky as an argument: the decisions are then
/// ordinary functions over numbers, and the one whole-world measurement can run them
/// without an app. The system below builds a stall out of the city's components, lets
/// the pure code mutate it, and writes the difference back — so there is one
/// implementation of a bargain rather than one for the game and one for the test.
#[derive(Clone, Copy, Debug)]
pub struct Stall {
    pub stocks: [f32; RESOURCE_COUNT],
    pub caps: [f32; RESOURCE_COUNT],
    pub demand: [f32; RESOURCE_COUNT],
    pub money: f32,
}

impl Stall {
    /// Reads one out of a city's own components.
    pub fn of(
        industry_config: &IndustryConfig,
        growth_config: &GrowthConfig,
        industry: &CityIndustry,
        growth: &CityGrowth,
        money: f32,
    ) -> Self {
        Self {
            stocks: Resource::ALL.map(|resource| industry.stock(resource)),
            caps: Resource::ALL.map(|resource| store_cap(industry_config, growth, resource)),
            demand: consumption(industry_config, growth_config, growth.population),
            money,
        }
    }

    pub fn price(&self, config: &MarketConfig, resource: Resource) -> f32 {
        price(
            config,
            resource,
            self.stocks[resource.index()],
            self.demand[resource.index()],
        )
    }

    /// What it can still take before what it buys would be discarded.
    fn room(&self, resource: Resource) -> f32 {
        (self.caps[resource.index()] - self.stocks[resource.index()]).max(0.0)
    }
}

/// What one visit's selling came to.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sold {
    /// What the trader was paid, which is exactly what left the city's purse.
    pub earned: f32,
    /// Which resources moved, and so may not be bought back here — see the module note
    /// on the pump this closes.
    pub resources: [bool; RESOURCE_COUNT],
}

/// Sells whatever this city will pay a profit for.
///
/// A lot is offered only above what it cost plus the margin, which is what makes an
/// unsold load stay aboard. The price is re-read per lot, because each sale moves the
/// stock the next one is priced against.
pub fn sell(
    config: &TradeConfig,
    market: &MarketConfig,
    caravan: &mut Caravan,
    stall: &mut Stall,
) -> Sold {
    let mut sold = Sold::default();
    for lot in caravan.cargo.clone() {
        let quoted = stall.price(market, lot.resource);
        if quoted < lot.paid_per_unit * (1.0 + config.trader_margin) {
            continue;
        }
        let Some(bargain) = sale(
            market,
            quoted,
            lot.units,
            stall.money,
            stall.room(lot.resource),
        ) else {
            continue;
        };

        stall.stocks[lot.resource.index()] += bargain.units;
        stall.money -= bargain.total();
        sold.earned += bargain.total();
        caravan.unload(lot.resource, bargain.units);
        sold.resources[lot.resource.index()] = true;
    }
    sold
}

/// Which road to take and what to load for it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Journey {
    /// Index into the slice of neighbours that was offered.
    pub edge: usize,
    /// What to buy here before setting out — `None` when the trip is worth taking for
    /// the load already aboard.
    pub buy: Option<Resource>,
}

/// Picks the road worth taking, or `None` when nothing here is worth carrying anywhere.
///
/// **Three ways a journey can pay, and the third is the one that makes the network
/// work.** The obvious two are carry and deliver: buy what the neighbour wants more, or,
/// holding goods this city refused, go where they are worth most — the second is what
/// the issue's "transported to the next city" needs. The third is to travel *empty to
/// fetch*, and leaving it out is what the first whole-world run diagnosed: iron flows
/// one way, from the fifth of cities that hold a seam to the rest, so a wagon that has
/// delivered its load is standing in a city with nothing worth buying and a return trip
/// that earns nothing. Only 226 journeys were made in two thousand steps, six wagons of
/// thirty-six never moved at all, and the traded world was indistinguishable from one
/// with no traders in it.
///
/// A deadhead is not a fourth kind of decision: it is the same round trip valued from
/// the other end. Goods cheaper *there* than *here* are worth going to get, priced at
/// what the wagon could afford once it arrived — so the leg out is justified by the leg
/// back, and the loop closes without a second hop of lookahead.
///
/// All three are net of the journey's cost, so a wagon does not cross the world for a
/// penny.
///
/// `neighbours` is `(length in tiles, that city's stall)` in the order the caller will
/// read the answer back in.
pub fn choose(
    config: &TradeConfig,
    market: &MarketConfig,
    caravan: &Caravan,
    purse: f32,
    here: &Stall,
    neighbours: &[(f32, Stall)],
    sold: &[bool; RESOURCE_COUNT],
) -> Option<Journey> {
    let space = caravan.space(config.caravan_capacity);
    let mut best: Option<(f32, Journey)> = None;
    let mut keep = |value: f32, journey: Journey| {
        if value > 0.0 && best.as_ref().is_none_or(|(top, _)| value > *top) {
            best = Some((value, journey));
        }
    };

    for (edge, (length, there)) in neighbours.iter().enumerate() {
        let toll = length * config.journey_cost_per_tile;
        // What the load already aboard would fetch over there rather than here. Zero
        // for an empty wagon, and it is what sends a refused cargo on to the next city.
        let carried: f32 = caravan
            .cargo
            .iter()
            .map(|lot| {
                lot.units * (there.price(market, lot.resource) - here.price(market, lot.resource))
            })
            .sum();

        for resource in Resource::ALL {
            if sold[resource.index()] {
                continue;
            }
            let (bid, ask) = (there.price(market, resource), here.price(market, resource));
            if bid <= ask {
                continue;
            }
            let Some(bargain) = purchase(market, ask, here.stocks[resource.index()], purse, space)
            else {
                continue;
            };
            keep(
                bargain.units * (bid - ask) + carried - toll,
                Journey {
                    edge,
                    buy: Some(resource),
                },
            );
        }

        keep(carried - toll, Journey { edge, buy: None });

        // Travelling empty to fetch: goods cheaper there than here, valued at what a
        // wagon could afford once it arrived. `buy: None`, because the buying happens on
        // arrival like any other visit — this only decides that the road is worth taking.
        for resource in Resource::ALL {
            let (ask, worth) = (there.price(market, resource), here.price(market, resource));
            if ask >= worth {
                continue;
            }
            let Some(bargain) = purchase(market, ask, there.stocks[resource.index()], purse, space)
            else {
                continue;
            };
            keep(
                bargain.units * (worth - ask) - toll,
                Journey { edge, buy: None },
            );
        }
    }

    if let Some((_, journey)) = best {
        return Some(journey);
    }

    // **Nothing here pays, so an empty wagon wanders toward the best market it can
    // see.** Standing still earns nothing either, and without this a wagon that has
    // delivered into a region where everything is glutted is stranded for the session —
    // sixteen of thirty-six were, in the run this was written against. It is only
    // offered to an *empty* wagon: one with a load has already been asked whether
    // carrying it anywhere pays, and overruling that answer would sell at a loss by the
    // back door.
    //
    // The toll is deliberately not charged here. It exists to stop a wagon crossing the
    // world for a penny of margin, and there is no margin to weigh it against — what is
    // being chosen is where to be, not what to earn.
    if !caravan.cargo.is_empty() {
        return None;
    }
    let mut wander: Option<(f32, usize)> = None;
    for (edge, (_, there)) in neighbours.iter().enumerate() {
        let cheapest = Resource::ALL
            .iter()
            .map(|resource| here.price(market, *resource) - there.price(market, *resource))
            .fold(f32::NEG_INFINITY, f32::max);
        if wander.as_ref().is_none_or(|(best, _)| cheapest > *best) {
            wander = Some((cheapest, edge));
        }
    }
    wander.map(|(_, edge)| Journey { edge, buy: None })
}

/// Loads for a chosen journey, and reports what it cost — which is exactly what entered
/// the city's purse.
pub fn buy(
    config: &TradeConfig,
    market: &MarketConfig,
    caravan: &mut Caravan,
    stall: &mut Stall,
    purse: f32,
    resource: Resource,
) -> f32 {
    let Some(bargain) = purchase(
        market,
        stall.price(market, resource),
        stall.stocks[resource.index()],
        purse,
        caravan.space(config.caravan_capacity),
    ) else {
        return 0.0;
    };
    stall.stocks[resource.index()] -= bargain.units;
    stall.money += bargain.total();
    caravan.load(resource, bargain.units, bargain.price_per_unit);
    bargain.total()
}

/// Advances every leg, and settles a caravan's business when it arrives and again when
/// it is rested.
///
/// Thin by design: everything it decides is decided above, and what is left here is
/// reading stalls out of components and writing the differences back. It touches the
/// same `CityIndustry` the growth step does, which is why it sits in a set ordered
/// after it rather than beside it.
fn drive_caravans(
    time: Res<Time>,
    config: Res<TradeConfig>,
    market: Res<MarketConfig>,
    industry_config: Res<IndustryConfig>,
    growth_config: Res<GrowthConfig>,
    graph: Option<Res<RoadGraph>>,
    mut caravans: Query<&mut Caravan>,
    mut traders: Query<&mut Trader>,
    mut cities: Query<(&CityGrowth, &mut CityIndustry, &mut CityTreasury)>,
) {
    let Some(graph) = graph else {
        return;
    };
    let delta = time.delta_secs();

    for mut caravan in &mut caravans {
        match caravan.errand {
            Errand::Travelling { mut leg, to } => {
                leg.travelled_tiles += config.caravan_speed_tiles * delta;
                if leg.travelled_tiles < leg.length_tiles {
                    caravan.errand = Errand::Travelling { leg, to };
                    continue;
                }
                // Arrived: unload what this city will pay for, then stand a while.
                let sold = settle_sales(
                    &config,
                    &market,
                    &industry_config,
                    &growth_config,
                    &mut caravan,
                    &mut traders,
                    &mut cities,
                    to,
                );
                caravan.errand = Errand::Resting {
                    city: to,
                    seconds_left: config.caravan_stop_seconds,
                    sold: sold.resources,
                };
            }
            Errand::Resting {
                city,
                seconds_left,
                mut sold,
            } => {
                let left = seconds_left - delta;
                if left > 0.0 {
                    caravan.errand = Errand::Resting {
                        city,
                        seconds_left: left,
                        sold,
                    };
                    continue;
                }

                match set_out(
                    &config,
                    &market,
                    &industry_config,
                    &growth_config,
                    &graph,
                    &mut caravan,
                    &mut traders,
                    &mut cities,
                    city,
                    &sold,
                ) {
                    Some(errand) => caravan.errand = errand,
                    None => {
                        // Nothing worth carrying anywhere. Try the sale again — the
                        // prices have moved since it arrived — and stand another turn.
                        let now = settle_sales(
                            &config,
                            &market,
                            &industry_config,
                            &growth_config,
                            &mut caravan,
                            &mut traders,
                            &mut cities,
                            city,
                        );
                        for (flag, sold_now) in sold.iter_mut().zip(now.resources) {
                            *flag |= sold_now;
                        }
                        caravan.errand = Errand::Resting {
                            city,
                            seconds_left: config.caravan_stop_seconds,
                            sold,
                        };
                    }
                }
            }
        }
    }
}

/// Runs [`sell`] against a city's components and writes the difference back.
#[allow(clippy::too_many_arguments)]
fn settle_sales(
    config: &TradeConfig,
    market: &MarketConfig,
    industry_config: &IndustryConfig,
    growth_config: &GrowthConfig,
    caravan: &mut Caravan,
    traders: &mut Query<&mut Trader>,
    cities: &mut Query<(&CityGrowth, &mut CityIndustry, &mut CityTreasury)>,
    city: Entity,
) -> Sold {
    if caravan.cargo.is_empty() {
        return Sold::default();
    }
    let Ok(mut trader) = traders.get_mut(caravan.trader) else {
        return Sold::default();
    };
    let Ok((growth, mut industry, mut treasury)) = cities.get_mut(city) else {
        return Sold::default();
    };

    let mut stall = Stall::of(
        industry_config,
        growth_config,
        &industry,
        growth,
        treasury.money(),
    );
    let before = stall.stocks;
    let sold = sell(config, market, caravan, &mut stall);

    for resource in Resource::ALL {
        industry.move_stock(
            resource,
            stall.stocks[resource.index()] - before[resource.index()],
        );
    }
    treasury.transfer(-sold.earned);
    trader.transfer(sold.earned);
    sold
}

/// Runs [`choose`] against the city and its neighbours, loads for the answer, and turns
/// it into the errand.
#[allow(clippy::too_many_arguments)]
fn set_out(
    config: &TradeConfig,
    market: &MarketConfig,
    industry_config: &IndustryConfig,
    growth_config: &GrowthConfig,
    graph: &RoadGraph,
    caravan: &mut Caravan,
    traders: &mut Query<&mut Trader>,
    cities: &mut Query<(&CityGrowth, &mut CityIndustry, &mut CityTreasury)>,
    city: Entity,
    sold: &[bool; RESOURCE_COUNT],
) -> Option<Errand> {
    let edges = graph.edges_from(city);
    if edges.is_empty() {
        return None;
    }
    let purse = traders.get(caravan.trader).ok()?.money();

    // Every stall is read out before anything is written: a `Stall` owns its numbers, so
    // no borrow of the city query outlives the reading.
    let stall_of = |cities: &Query<(&CityGrowth, &mut CityIndustry, &mut CityTreasury)>,
                    entity: Entity| {
        cities.get(entity).ok().map(|(growth, industry, treasury)| {
            Stall::of(
                industry_config,
                growth_config,
                industry,
                growth,
                treasury.money(),
            )
        })
    };
    let here = stall_of(cities, city)?;
    let neighbours: Vec<(f32, Stall)> = edges
        .iter()
        .filter_map(|edge| Some((edge.length_tiles, stall_of(cities, edge.to)?)))
        .collect();
    // A filtered list would put the answer's index against the wrong road. A city whose
    // neighbour cannot be read is simply not travelled to this turn.
    if neighbours.len() != edges.len() {
        return None;
    }

    let journey = choose(config, market, caravan, purse, &here, &neighbours, sold)?;
    let edge = edges[journey.edge];

    if let Some(resource) = journey.buy
        && let Ok(mut trader) = traders.get_mut(caravan.trader)
        && let Ok((_, mut industry, mut treasury)) = cities.get_mut(city)
    {
        let mut stall = here;
        let spent = buy(config, market, caravan, &mut stall, purse, resource);
        industry.move_stock(
            resource,
            stall.stocks[resource.index()] - here.stocks[resource.index()],
        );
        treasury.transfer(spent);
        trader.transfer(-spent);
    }

    Some(Errand::Travelling {
        leg: Leg {
            link: edge.link,
            reversed: edge.reversed,
            travelled_tiles: 0.0,
            length_tiles: edge.length_tiles,
        },
        to: edge.to,
    })
}

/// Writes every caravan's position and drawn size.
///
/// A resting wagon sits on its city's centre; a travelling one on the linear
/// interpolation between the two path tiles its distance falls between — so it is on
/// the road by construction rather than by a check.
fn place_caravans(
    config: Res<TradeConfig>,
    network: Option<Res<RoadNetwork>>,
    cities: Query<&City>,
    camera: Option<Single<&Projection, With<WorldCamera>>>,
    mut caravans: Query<(&Caravan, &mut Transform, &mut Sprite)>,
) {
    let Some(network) = network else {
        return;
    };
    // At MAX_ZOOM_SCALE a wagon is a quarter of its drawn size on screen, so the
    // sprite grows with the scale rather than vanishing — the same trick, and the same
    // accessor, `city_panel.rs`'s pick slack uses.
    let scale = camera.map_or(1.0, |projection| {
        orthographic_scale(projection.into_inner())
    });
    let size =
        (config.caravan_sprite_tiles * TILE_PIXELS).max(config.caravan_min_screen_px * scale);

    for (caravan, mut transform, mut sprite) in &mut caravans {
        let position = match caravan.errand {
            Errand::Resting { city, .. } => cities.get(city).ok().map(|city| city.centre.as_vec2()),
            Errand::Travelling { leg, .. } => network
                .links
                .get(leg.link)
                .map(|link| walk(&link.path, leg.reversed, leg.travelled_tiles)),
        };
        let Some(position) = position else {
            continue;
        };
        // `tile_translation` centres on the tile, and the fractional part of the walk
        // is carried through it so a wagon slides rather than stepping tile to tile.
        let whole = position.floor();
        let translation = tile_translation(whole.as_ivec2()) + (position - whole) * TILE_PIXELS;
        transform.translation = translation.extend(CARAVAN_Z);

        if sprite.custom_size != Some(Vec2::splat(size)) {
            sprite.custom_size = Some(Vec2::splat(size));
        }
    }
}

/// Where along a path a given distance falls, in continuous tile space.
///
/// The path's tiles are contiguous, so one tile is one step and the distance is an
/// index with a fraction — no arc length to accumulate.
fn walk(path: &[IVec2], reversed: bool, travelled: f32) -> Vec2 {
    let last = path.len().saturating_sub(1);
    let travelled = travelled.clamp(0.0, last as f32);
    let index = travelled.floor() as usize;
    let fraction = travelled - index as f32;
    let at = |i: usize| {
        let i = if reversed {
            last - i.min(last)
        } else {
            i.min(last)
        };
        path[i].as_vec2()
    };
    at(index).lerp(at(index + 1), fraction)
}

/// The size of one tile on screen. `TILE_DISPLAY_SIZE` is a `UVec2` of equal
/// components; this is that number where a scalar is wanted.
const TILE_PIXELS: f32 = 8.0;

/// Above the chunk meshes, which sit at zero, and below nothing — the UI is a separate
/// pass entirely.
const CARAVAN_Z: f32 = 1.0;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gameplay::market::Bargain;

    fn caravan() -> Caravan {
        Caravan {
            trader: Entity::PLACEHOLDER,
            errand: Errand::Resting {
                city: Entity::PLACEHOLDER,
                seconds_left: 0.0,
                sold: [false; RESOURCE_COUNT],
            },
            cargo: Vec::new(),
        }
    }

    #[test]
    fn loading_a_lot_twice_averages_what_it_cost() {
        let mut wagon = caravan();
        wagon.load(Resource::Iron, 100.0, 2.0);
        wagon.load(Resource::Iron, 100.0, 4.0);
        let lot = wagon.cargo()[0];
        assert_eq!(lot.units, 200.0);
        assert!((lot.paid_per_unit - 3.0).abs() < 1e-4);
    }

    #[test]
    fn selling_part_of_a_lot_leaves_what_the_rest_cost_alone() {
        let mut wagon = caravan();
        wagon.load(Resource::Salt, 100.0, 5.0);
        wagon.unload(Resource::Salt, 40.0);
        let lot = wagon.cargo()[0];
        assert!((lot.units - 60.0).abs() < 1e-4);
        assert!((lot.paid_per_unit - 5.0).abs() < 1e-4);
    }

    #[test]
    fn an_emptied_lot_leaves_the_wagon_rather_than_lingering_at_zero() {
        let mut wagon = caravan();
        wagon.load(Resource::Wood, 10.0, 1.0);
        wagon.unload(Resource::Wood, 10.0);
        assert!(wagon.cargo().is_empty());
        assert_eq!(wagon.carried(), 0.0);
    }

    #[test]
    fn a_wagon_never_reports_more_space_than_it_has() {
        let mut wagon = caravan();
        wagon.load(Resource::Food, 500.0, 1.0);
        wagon.load(Resource::Stone, 400.0, 1.0);
        assert!((wagon.carried() - 900.0).abs() < 1e-3);
        assert!((wagon.space(1000.0) - 100.0).abs() < 1e-3);
        assert_eq!(wagon.space(500.0), 0.0);
    }

    #[test]
    fn a_walk_stays_on_the_path_and_both_ways_along_it() {
        let path: Vec<IVec2> = (0..10).map(|x| IVec2::new(x, 5)).collect();
        assert_eq!(walk(&path, false, 0.0), Vec2::new(0.0, 5.0));
        assert_eq!(walk(&path, false, 4.5), Vec2::new(4.5, 5.0));
        assert_eq!(walk(&path, false, 100.0), Vec2::new(9.0, 5.0));
        // Reversed is the same road walked the other way, so its start is the far end.
        assert_eq!(walk(&path, true, 0.0), Vec2::new(9.0, 5.0));
        assert_eq!(walk(&path, true, 4.5), Vec2::new(4.5, 5.0));
        assert_eq!(walk(&path, true, 100.0), Vec2::new(0.0, 5.0));
    }

    #[test]
    fn a_walk_on_a_one_tile_path_does_not_divide_by_its_own_length() {
        let path = vec![IVec2::new(3, 3)];
        assert_eq!(walk(&path, false, 0.0), Vec2::new(3.0, 3.0));
        assert_eq!(walk(&path, true, 7.0), Vec2::new(3.0, 3.0));
    }

    #[test]
    fn a_traders_purse_cannot_be_pushed_below_broke() {
        let mut trader = Trader { id: 0, money: 5.0 };
        trader.transfer(-100.0);
        assert_eq!(trader.money(), 0.0);
    }

    #[test]
    fn a_round_trip_through_a_bargain_moves_the_same_money_both_ways() {
        // The property every transaction rests on: what leaves one purse enters the
        // other, so the world's money changes only where the income mints it.
        let bargain = Bargain {
            units: 37.5,
            price_per_unit: 2.4,
        };
        let mut trader = Trader {
            id: 0,
            money: 1000.0,
        };
        let mut city = CityTreasury::new(1000.0);
        trader.transfer(-bargain.total());
        city.transfer(bargain.total());
        assert!((trader.money() + city.money() - 2000.0).abs() < 1e-3);
    }
}
