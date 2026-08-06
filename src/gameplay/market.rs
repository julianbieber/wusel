//! What a resource is worth in a city, and how much of it changes hands.
//!
//! Split off [`crate::gameplay::trade`] on the line `deposit.rs` and `industry.rs`
//! already draw: this is numbers, that is agents. Nothing here moves, nothing here
//! is drawn, and no function below takes a `World` — which is what makes every
//! price and every bargain an ordinary unit test with no app in it.
//!
//! **A price is availability against need, and neither alone.** The issue asks for a
//! base value with "a range specified by need and surplus", and read literally that
//! is right: a resource's price in a city is its base value scaled by how well
//! stocked the city is *against its own consumption*, not against a global constant
//! and not against the store cap. So a metropolis pays more for grain than a hamlet
//! holding the same granary, because consumption is per head — which is what makes a
//! big city a destination rather than another node, and it costs one division.
//!
//! The need comes from [`industry::consumption`] and may not come from anywhere else.
//! The baskets, the per-head rates and the build cost live in that module and will
//! move again; a second table here would be the `TERRAIN_KIND_COUNT` failure mode,
//! silently one behind and wrong in a direction nothing tests.
//!
//! **The curve has flat ends on purpose.** A trade moves a stock, which moves the
//! price, which is what the *next* trade sees. `smoothstep` rather than a straight
//! line means a slice-sized bargain near either end barely moves the price, so a
//! caravan that set out on a margin still finds one when it arrives; a linear ramp
//! makes the last unit of a sale worth as much less as the first, and caravans
//! oscillate.
//!
//! **Every bargain is bounded, and the rule for which bounds get sliced is one
//! sentence: the city's side is sliced, the trader's side is hard.** A city must not
//! be sold empty, bankrupted, or filled to its cap in a single visit — those are the
//! three ways one wagon could distort a market it should only nudge, and
//! `trade_slice` is the whole of the price stability. A trader's purse and a wagon's
//! capacity are not markets and need no such protection: a caravan may fill itself in
//! one go, because that is what a caravan is for.
//!
//! **Money is the one quantity in the feature that is not conserved.** A bargain
//! moves it and creates none; [`crate::gameplay::trade`] mints
//! `city_income_per_person` per head per step, out of nothing, standing for trade with
//! a wider world that is not modelled. Without a source the cities' total purse falls
//! monotonically as traders take their margin, until nobody can buy and trade stops
//! for good. The rejected alternative was a toll paid back to the arriving city, which
//! keeps the total exactly constant — it was rejected because a broke city cannot buy,
//! so it collects no toll, so it stays broke: an absorbing state with no way out.
//!
//! Minting has the opposite failure and a gentler one. Prices are `base x scarcity`
//! with no term that could inflate, so over a long session the supply grows until
//! money stops binding and only goods ration trade. That is acceptable for a first cut
//! and it is *measurable* rather than a guess — see
//! `the_default_config_moves_goods_between_cities`, which prints the median treasury
//! over time. A purse growing without bound is the signal that this wants a sink after
//! all, and the toll is the thing to reach for. Do not tune it away by lowering the
//! income: that only moves when it happens.
//!
//! At the shipped values the measurement reads, over 16000 steps of the default world:
//! **16861 journeys by 80 wagons with none stranded**, 1.04M units of food, 27k of iron,
//! 30k of copper and 17k of salt delivered, **84-100% of every scarce good landing in a
//! city that can produce none of it**, and 26 of 92 cities taking delivery of something
//! they cannot make. The median treasury climbs 6.0k -> 128k across the whole run, which
//! is the drift above: about 2 a step against an income of 10. A trader opens with 20k
//! and ends between 33k and 201k, so the trade is profitable without being a windfall.
//!
//! **Food is the biggest trade good, and that is gh-7's flat granary showing through
//! rather than a knob.** A fixed barn caps a city at
//! `granary_max / harvest_interval_steps` people, so the larger half of the world sits
//! exactly on its food ceiling and grain is genuinely scarce there. The issue is
//! called "traders moving food and money between cities"; it took a storage limit to
//! make the first half of that sentence true.

use bevy::prelude::*;

use crate::gameplay::{
    deposit::{RESOURCE_COUNT, Resource},
    growth::{CityGrowth, GrowthConfig},
    industry::{CityIndustry, IndustryConfig, consumption},
};

/// Knobs for the market. Configuration rather than world state, so like
/// [`GrowthConfig`] and [`IndustryConfig`] this outlives a session — which means a
/// scenario that depends on one of these has to state it.
#[derive(Resource, Clone)]
pub struct MarketConfig {
    /// What one unit is worth in a city that is exactly averagely stocked, per
    /// resource in [`Resource::ALL`]'s order.
    ///
    /// Roughly inverse to how much of the world can produce it. Every city grows food
    /// and most can cut wood or quarry stone, while gh-24 measured **20 cities of 92**
    /// working any seam at all — so iron, copper and salt are the goods four cities in
    /// five can only buy.
    ///
    /// It scales both ends of the range together, so it says how *valuable* a resource
    /// is and never how *variable*. A base value cannot create a spread where the
    /// stocks are even, and cannot flatten one where they are not.
    pub base_value: [f32; RESOURCE_COUNT],
    /// The two ends of the range, as multiples of the base: what a city with nothing
    /// pays, and what a fully stocked one offers.
    ///
    /// Their ratio is the most a trader can make on one load before costs, and it has
    /// to comfortably exceed 1 + `trader_margin` or no journey is ever worth taking.
    pub price_scarce_multiple: f32,
    pub price_glut_multiple: f32,
    /// How many steps of its own consumption counts as fully stocked.
    ///
    /// The one knob that says what "need" means, and it is in **steps** rather than
    /// seconds deliberately: gh-7 slowed the world by changing what a step is worth in
    /// seconds, and this has to move with that change rather than against it.
    pub price_horizon_steps: f32,
    /// The floor under a city's consumption, so a resource it happens to want none of
    /// still has a finite hoard and so a finite price. Without it an unwanted resource
    /// divides by zero and every city in the world is infinitely glutted.
    pub min_consumption: f32,
    /// The largest share of a city-side bound one bargain may move. See the module
    /// note: the city's side is sliced and the trader's is not.
    pub trade_slice: f32,
    /// What a city opens with, per tile of town.
    ///
    /// Sized like its stores and for the same reason `seed_industry` opens those full:
    /// a city that has stood for years is not starting from an empty strongroom.
    ///
    /// **This is the knob that decides whether money binds, and the income is not.**
    /// A city needs a *stock* of money large enough to fund a delivery before it earns
    /// any of it back, and a delivery is `trade_slice` of the room under its cap — at
    /// the defaults about 1400 units of iron, or 34k. A large city opens with about
    /// that, so it can afford one delivery and then has to trade its way to the next.
    pub city_start_money_per_town_tile: f32,
    /// Minted per head per step. The only quantity in the crate that is created from
    /// nothing — see the module note for why it exists and what to watch.
    ///
    /// **Sized against the net drain, not against the gross trade**, and the difference
    /// is a factor of a hundred. Money circulates: a city pays a trader for iron and is
    /// paid by the next trader for its grain, so what actually leaves the cities is only
    /// the traders' margin — measured at ~16 a step for a city of five thousand. The
    /// first cut sized this against the *gross* import bill instead, at 0.6, and the
    /// world's median treasury went from 6.7k to **6.0M in 2000 steps**: money stopped
    /// binding within a few hundred steps, which is the failure this module's note warns
    /// about arriving immediately rather than eventually.
    ///
    /// It is also, indirectly, the throttle on how much trade happens at all, because a
    /// bargain is bounded by what the buyer can pay. The sweep, over 16000 steps:
    ///
    /// | income | scarce goods delivered | wagons stranded | median treasury |
    /// |--------|------------------------|-----------------|-----------------|
    /// | 0.002  | 74k                    | 0               | 6.0k -> 128k    |
    /// | 0.01   | 95k                    | 0               | 6.0k -> 664k    |
    /// | 0.3    | 247k                   | -               | 6.4k -> 3.0M    |
    ///
    /// (the 0.01 and 0.3 rows were taken at 36 wagons, before the fleet was sized
    /// against the road count; the shape of the trade-off is what they are kept for)
    ///
    /// 0.002 is chosen for the flattest drift rather than the fattest flow: what the
    /// higher settings buy is bounded anyway, since the world only *produces* about
    /// 100k iron in that time, and what they cost is the constraint that makes "a city
    /// has a limited amount of money" mean anything.
    pub city_income_per_person: f32,
}

impl Default for MarketConfig {
    fn default() -> Self {
        Self {
            // Food 1 is the unit the rest are quoted in. Wood and stone are bulk;
            // stone is under food because its standing demand is the notional building
            // spend and nothing else. The three seam goods are dear because most of the
            // world cannot make them at any price.
            base_value: [1.0, 1.5, 0.8, 8.0, 6.0, 5.0],
            price_scarce_multiple: 3.0,
            price_glut_multiple: 0.35,
            price_horizon_steps: 30.0,
            min_consumption: 0.5,
            trade_slice: 0.25,
            city_start_money_per_town_tile: 200.0,
            city_income_per_person: 0.002,
        }
    }
}

/// The money a city holds.
///
/// A third component beside [`CityGrowth`] and [`CityIndustry`], on exactly the
/// ownership argument those two split on: `growth.rs` writes the population and the
/// fields, `industry.rs` writes the stores and the hands, and **nothing but a bargain
/// and the income writes this**. In particular no line in `growth.rs` learns that
/// money exists.
///
/// A city without one cannot trade at all, which is what makes the seeding order safe
/// rather than something to arrange: a caravan arriving before the treasuries are
/// handed out simply finds nobody to deal with.
#[derive(Component, Debug, Default)]
pub struct CityTreasury {
    money: f32,
}

impl CityTreasury {
    pub fn new(money: f32) -> Self {
        Self { money }
    }

    pub fn money(&self) -> f32 {
        self.money
    }

    /// Moves money into or out of the purse, floored at broke.
    ///
    /// Floored rather than asserted because the floor is unreachable by design — a
    /// bargain is bounded by what the buyer can pay — so this is the shape of the
    /// guard and not a live path. `no_city_ever_spends_money_it_does_not_hold` is what
    /// actually holds the line.
    pub fn transfer(&mut self, amount: f32) {
        self.money = (self.money + amount).max(0.0);
    }
}

/// What one unit of a resource is worth in one city, from what it holds against what
/// it consumes.
///
/// Strictly positive and non-increasing in the stock, both of which are asserted:
/// they are what let a caravan compare two cities and know which way goods should
/// flow.
pub fn price(
    config: &MarketConfig,
    resource: Resource,
    stock: f32,
    consumption_per_step: f32,
) -> f32 {
    let hoard = consumption_per_step.max(config.min_consumption) * config.price_horizon_steps;
    let cover = (stock / hoard.max(f32::EPSILON)).clamp(0.0, 1.0);
    // Smoothstep, for the flat ends — see the module note.
    let eased = cover * cover * (3.0 - 2.0 * cover);
    let multiple = config.price_scarce_multiple
        + (config.price_glut_multiple - config.price_scarce_multiple) * eased;
    config.base_value[resource.index()] * multiple
}

/// Every price in one city, which is what a caravan compares. One call rather than
/// six, because the consumption behind them is one call either way.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub fn prices(
    config: &MarketConfig,
    industry_config: &IndustryConfig,
    growth_config: &GrowthConfig,
    industry: &CityIndustry,
    growth: &CityGrowth,
) -> [f32; RESOURCE_COUNT] {
    let demand = consumption(industry_config, growth_config, growth.population);
    let mut out = [0.0; RESOURCE_COUNT];
    for resource in Resource::ALL {
        out[resource.index()] = price(
            config,
            resource,
            industry.stock(resource),
            demand[resource.index()],
        );
    }
    out
}

/// One transaction: how many units at what price. Returned rather than applied, so
/// the bounds can be checked without a city to apply them to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bargain {
    pub units: f32,
    pub price_per_unit: f32,
}

impl Bargain {
    pub fn total(&self) -> f32 {
        self.units * self.price_per_unit
    }
}

/// What a caravan may buy out of a city in one visit.
///
/// The city's stock is sliced — one wagon may not empty a market — while the trader's
/// purse and the wagon's free space are hard: a caravan may fill itself in one go,
/// which is what a caravan is for.
pub fn purchase(
    config: &MarketConfig,
    price_per_unit: f32,
    city_stock: f32,
    trader_money: f32,
    caravan_space: f32,
) -> Option<Bargain> {
    let affordable = trader_money / price_per_unit.max(f32::EPSILON);
    let units = (config.trade_slice * city_stock)
        .min(affordable)
        .min(caravan_space);
    finish(units, price_per_unit)
}

/// What a city will buy off a caravan in one visit.
///
/// Both city-side bounds are sliced — its purse and the room under its cap — while
/// the lot itself is not: a caravan is free to offer everything it carries.
pub fn sale(
    config: &MarketConfig,
    price_per_unit: f32,
    lot_units: f32,
    city_money: f32,
    city_room: f32,
) -> Option<Bargain> {
    let affordable = city_money / price_per_unit.max(f32::EPSILON);
    let units = lot_units
        .min(config.trade_slice * affordable)
        .min(config.trade_slice * city_room);
    finish(units, price_per_unit)
}

/// A bargain too small to be worth the arithmetic is no bargain, and that is an
/// ordinary outcome rather than a failure — a caravan standing in a city with nothing
/// to sell it is the common case.
fn finish(units: f32, price_per_unit: f32) -> Option<Bargain> {
    (units > MIN_UNITS).then_some(Bargain {
        units,
        price_per_unit,
    })
}

/// Below this a transaction is dropped. Its job is to stop a caravan spending a
/// visit moving a millionth of a unit and calling it trade; the value is arbitrary
/// and nothing is tuned against it.
const MIN_UNITS: f32 = 0.5;

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> MarketConfig {
        MarketConfig::default()
    }

    #[test]
    fn an_empty_city_pays_the_scarce_price_and_a_stocked_one_offers_the_glut_price() {
        let config = config();
        let base = config.base_value[Resource::Iron.index()];
        let hoard = 10.0 * config.price_horizon_steps;

        let empty = price(&config, Resource::Iron, 0.0, 10.0);
        let full = price(&config, Resource::Iron, hoard, 10.0);

        assert!((empty - base * config.price_scarce_multiple).abs() < 1e-3);
        assert!((full - base * config.price_glut_multiple).abs() < 1e-3);
    }

    #[test]
    fn a_price_never_rises_as_the_stock_does_and_never_reaches_zero() {
        let config = config();
        let mut last = f32::INFINITY;
        for step in 0..=200 {
            let stock = step as f32 * 5.0;
            let now = price(&config, Resource::Food, stock, 8.0);
            assert!(now > 0.0, "a price must stay positive, got {now}");
            assert!(now <= last + 1e-4, "price rose with stock at {stock}");
            last = now;
        }
    }

    #[test]
    fn a_bigger_city_pays_more_for_the_same_granary() {
        // Consumption is per head, so the same store covers a hamlet and does not
        // cover a metropolis. This is the whole of why a large city is a destination.
        let config = config();
        let hamlet = price(&config, Resource::Food, 500.0, 20.0);
        let metropolis = price(&config, Resource::Food, 500.0, 400.0);
        assert!(metropolis > hamlet);
    }

    #[test]
    fn a_resource_nothing_wants_still_has_a_finite_price() {
        let config = config();
        let quoted = price(&config, Resource::Salt, 0.0, 0.0);
        assert!(quoted.is_finite() && quoted > 0.0);
    }

    #[test]
    fn a_purchase_never_takes_more_than_a_slice_of_the_market() {
        let config = config();
        let bargain = purchase(&config, 2.0, 1000.0, 1e9, 1e9).expect("a rich trader buys");
        assert!((bargain.units - config.trade_slice * 1000.0).abs() < 1e-3);
    }

    #[test]
    fn a_purchase_is_bounded_hard_by_the_purse_and_the_wagon() {
        let config = config();
        // The purse binds: 100 money at 2.0 buys 50, well under a slice of the market.
        let poor = purchase(&config, 2.0, 100_000.0, 100.0, 1e9).expect("some is affordable");
        assert!((poor.units - 50.0).abs() < 1e-3);
        // The wagon binds, and is *not* sliced — a caravan may fill itself in one go.
        let small = purchase(&config, 2.0, 100_000.0, 1e9, 40.0).expect("the wagon takes some");
        assert!((small.units - 40.0).abs() < 1e-3);
    }

    #[test]
    fn a_city_never_buys_past_its_room_or_its_purse() {
        let config = config();
        let broke = sale(&config, 10.0, 1e6, 100.0, 1e6).expect("a poor city buys a little");
        assert!(broke.total() <= 100.0 + 1e-3);
        let cramped = sale(&config, 10.0, 1e6, 1e9, 80.0).expect("a full city buys a little");
        assert!(cramped.units <= 80.0 + 1e-3);
    }

    #[test]
    fn a_bargain_with_nothing_in_it_is_no_bargain() {
        let config = config();
        assert!(purchase(&config, 5.0, 0.0, 1e9, 1e9).is_none());
        assert!(sale(&config, 5.0, 1e6, 0.0, 1e6).is_none());
        assert!(sale(&config, 5.0, 0.0, 1e9, 1e9).is_none());
    }

    #[test]
    fn a_treasury_cannot_be_pushed_below_broke() {
        let mut purse = CityTreasury::new(10.0);
        purse.transfer(-1000.0);
        assert_eq!(purse.money(), 0.0);
    }

    #[test]
    fn the_price_range_is_wide_enough_to_pay_a_traders_margin() {
        // A stability condition rather than taste: if the glut and scarce ends are
        // closer together than the margin, no journey in the world is ever worth
        // taking and every caravan idles for the session.
        let config = config();
        let spread = config.price_scarce_multiple / config.price_glut_multiple;
        assert!(
            spread > 1.5,
            "the price range must pay for a journey, spread was {spread}"
        );
    }
}
