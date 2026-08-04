//! What a client can ask for, and how far along it is.
//!
//! Every command is polled once a frame until it reports [`Poll::Done`], so "wait for
//! something" and "do something over 240 frames" are the same mechanism rather than two.
//! Progress lives in the variant itself: most of it is derived from how many frames have
//! elapsed since the command arrived, which leaves nothing to keep in step.

use std::{path::PathBuf, time::Duration};

use bevy::{
    input::{
        ButtonState,
        keyboard::{Key, KeyboardInput, NativeKey},
        mouse::MouseButtonInput,
    },
    prelude::*,
    render::view::screenshot::{Screenshot, save_to_disk},
    time::TimeUpdateStrategy,
    window::PrimaryWindow,
};
use serde_json::{Value, json};

use super::observe::{self, Topic};
use crate::{
    camera::WorldCamera,
    gameplay::{
        plan::WorldPlan,
        weather::WeatherMaps,
        world::{BackgroundGeneration, tile_translation},
    },
    screens::Screen,
};

/// Ten minutes at 60 Hz. Long enough for a whole-world plan on a slow machine, short
/// enough that a stuck scenario reports rather than hangs.
const DEFAULT_WAIT_FRAMES: u32 = 36_000;

pub(super) enum Poll {
    Running,
    Done(Value),
    Failed(String),
}

pub(super) enum Command {
    Ping,
    Step(u32),
    Wait {
        condition: Condition,
        timeout: u32,
    },
    Enter(Screen),
    Hold {
        keys: Vec<KeyCode>,
        frames: u32,
    },
    Zoom {
        key: KeyCode,
        steps: u32,
    },
    Cursor(Vec2),
    Click(MouseButton),
    ClickTile(IVec2),
    Capture {
        path: PathBuf,
        /// The screenshot entity, once spawned. Its *absence from the world* is what
        /// says the PNG is on disk — see the poll arm.
        entity: Option<Entity>,
    },
    Observe(Topic),
    FixedDelta(Duration),
    Realtime,
    Quit,
}

pub(super) enum Condition {
    Terrain,
    Plan,
    Sky,
    Screen(Screen),
}

impl Command {
    pub(super) fn verb(&self) -> &'static str {
        match self {
            Self::Ping => "ping",
            Self::Step(_) => "step",
            Self::Wait { .. } => "wait",
            Self::Enter(_) => "enter",
            Self::Hold { .. } => "hold",
            Self::Zoom { .. } => "zoom",
            Self::Cursor(_) => "cursor",
            Self::Click(_) => "click",
            Self::ClickTile(_) => "click-tile",
            Self::Capture { .. } => "capture",
            Self::Observe(_) => "observe",
            Self::FixedDelta(_) => "fixed-delta",
            Self::Realtime => "realtime",
            Self::Quit => "quit",
        }
    }

    pub(super) fn parse(line: &str) -> Result<Self, String> {
        let mut words = line.split_whitespace();
        let verb = words.next().ok_or("empty command")?;
        let rest: Vec<&str> = words.collect();

        match verb {
            "ping" => Ok(Self::Ping),
            "step" => Ok(Self::Step(optional_number(rest.first(), 1)?)),
            "wait" => {
                let what = rest.first().ok_or("wait needs something to wait for")?;
                Ok(Self::Wait {
                    condition: Condition::parse(what)?,
                    timeout: optional_number(rest.get(1), DEFAULT_WAIT_FRAMES)?,
                })
            }
            "enter" => Ok(Self::Enter(screen(
                rest.first().ok_or("enter needs a screen")?,
            )?)),
            "hold" => {
                let keys = rest.first().ok_or("hold needs keys")?;
                let keys = keys
                    .split(',')
                    .map(key_code)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self::Hold {
                    keys,
                    frames: optional_number(rest.get(1), 60)?,
                })
            }
            "zoom" => {
                let steps: i32 = number(rest.first().ok_or("zoom needs a step count")?)?;
                // A positive step zooms in, which `camera.rs` spells `Equal`.
                Ok(Self::Zoom {
                    key: if steps >= 0 {
                        KeyCode::Equal
                    } else {
                        KeyCode::Minus
                    },
                    steps: steps.unsigned_abs(),
                })
            }
            "cursor" => Ok(Self::Cursor(Vec2::new(
                number(rest.first().ok_or("cursor needs x")?)?,
                number(rest.get(1).ok_or("cursor needs y")?)?,
            ))),
            "click" => Ok(Self::Click(match rest.first() {
                None | Some(&"left") => MouseButton::Left,
                Some(&"right") => MouseButton::Right,
                Some(&"middle") => MouseButton::Middle,
                Some(other) => return Err(format!("unknown mouse button: {other}")),
            })),
            "click-tile" => Ok(Self::ClickTile(IVec2::new(
                number(rest.first().ok_or("click-tile needs a tile x")?)?,
                number(rest.get(1).ok_or("click-tile needs a tile y")?)?,
            ))),
            "capture" => Ok(Self::Capture {
                path: PathBuf::from(rest.first().ok_or("capture needs a path")?),
                entity: None,
            }),
            "observe" => Ok(Self::Observe(Topic::parse(
                rest.first().ok_or("observe needs a topic")?,
            )?)),
            "fixed-delta" => Ok(Self::FixedDelta(Duration::from_secs_f32(seconds(
                rest.first().copied().unwrap_or("1/60"),
            )?))),
            "realtime" => Ok(Self::Realtime),
            "quit" => Ok(Self::Quit),
            other => Err(format!("unknown command: {other}")),
        }
    }

    /// `elapsed` is frames since the command was read, which is what most of these are
    /// driven by — holding a key for 240 frames is "press at 0, release at 240" rather
    /// than a counter that has to be decremented exactly once per frame.
    pub(super) fn poll(&mut self, world: &mut World, elapsed: u32) -> Poll {
        match self {
            Self::Ping => Poll::Done(json!({})),

            Self::Step(frames) => {
                if elapsed >= *frames {
                    Poll::Done(json!({}))
                } else {
                    Poll::Running
                }
            }

            Self::Wait { condition, timeout } => {
                if condition.met(world) {
                    Poll::Done(json!({}))
                } else if elapsed >= *timeout {
                    Poll::Failed(format!(
                        "timed out after {elapsed} frames waiting for {}; {}",
                        condition.name(),
                        condition.progress(world)
                    ))
                } else {
                    Poll::Running
                }
            }

            // Setting the state rather than clicking the button: driving `bevy_picking`
            // through a Feathers button is a large lift for a transition no scenario is
            // testing. Waiting for `State` to agree is what makes the reply mean "the
            // screen is up", `OnEnter` included, rather than "the change is queued".
            Self::Enter(target) => {
                if elapsed == 0 {
                    world.resource_mut::<NextState<Screen>>().set(*target);
                    return Poll::Running;
                }
                if world.resource::<State<Screen>>().get() == target {
                    Poll::Done(json!({}))
                } else {
                    Poll::Running
                }
            }

            // Pressed once and released once, because `ButtonInput` keeps `pressed`
            // between frames. Re-sending the press every frame would re-fire
            // `just_pressed`, which is the edge the zoom and the city click read.
            Self::Hold { keys, frames } => {
                let Some(window) = primary_window(world) else {
                    return Poll::Failed("no primary window".into());
                };
                if elapsed == 0 {
                    for key in keys.clone() {
                        write_key(world, window, key, ButtonState::Pressed);
                    }
                    Poll::Running
                } else if elapsed >= *frames {
                    for key in keys.clone() {
                        write_key(world, window, key, ButtonState::Released);
                    }
                    Poll::Done(json!({ "held": frames }))
                } else {
                    Poll::Running
                }
            }

            // Two frames per step: the zoom reads `just_pressed`, so each step needs its
            // own press edge and there is only one edge per key per frame.
            Self::Zoom { key, steps } => {
                let Some(window) = primary_window(world) else {
                    return Poll::Failed("no primary window".into());
                };
                let total = steps.saturating_mul(2);
                if total == 0 {
                    return Poll::Done(json!({ "steps": 0 }));
                }
                let state = if elapsed.is_multiple_of(2) {
                    ButtonState::Pressed
                } else {
                    ButtonState::Released
                };
                write_key(world, window, *key, state);
                if elapsed + 1 >= total {
                    Poll::Done(json!({ "steps": steps }))
                } else {
                    Poll::Running
                }
            }

            Self::Cursor(position) => match set_cursor(world, *position) {
                true => Poll::Done(json!({ "cursor": [position.x, position.y] })),
                false => Poll::Failed("no primary window".into()),
            },

            Self::Click(button) => click(world, *button, elapsed, json!({})),

            Self::ClickTile(tile) => {
                if elapsed == 0 {
                    let Some(position) = tile_on_screen(world, *tile) else {
                        return Poll::Failed(format!(
                            "tile {},{} is not on screen",
                            tile.x, tile.y
                        ));
                    };
                    if !set_cursor(world, position) {
                        return Poll::Failed("no primary window".into());
                    }
                }
                click(world, MouseButton::Left, elapsed, json!({}))
            }

            Self::Capture { path, entity } => match entity {
                None => {
                    if let Some(parent) = path.parent()
                        && let Err(error) = std::fs::create_dir_all(parent)
                    {
                        return Poll::Failed(format!(
                            "cannot create {}: {error}",
                            parent.display()
                        ));
                    }
                    *entity = Some(
                        world
                            .spawn(Screenshot::primary_window())
                            .observe(save_to_disk(path.clone()))
                            .id(),
                    );
                    Poll::Running
                }
                // `clear_screenshots` despawns the entity in `First`, which runs strictly
                // after the `ScreenshotCaptured` observer has written the file. So the
                // entity being gone is the signal that the PNG is on disk — no polling
                // the filesystem, and no racing a half-written file.
                Some(id) => {
                    if world.entities().contains(*id) {
                        Poll::Running
                    } else {
                        Poll::Done(json!({ "path": path.display().to_string() }))
                    }
                }
            },

            Self::Observe(topic) => Poll::Done(observe::run(world, topic)),

            Self::FixedDelta(delta) => {
                world.insert_resource(TimeUpdateStrategy::ManualDuration(*delta));
                Poll::Done(json!({ "delta_seconds": delta.as_secs_f32() }))
            }

            Self::Realtime => {
                world.insert_resource(TimeUpdateStrategy::Automatic);
                Poll::Done(json!({}))
            }

            Self::Quit => {
                world.write_message(AppExit::Success);
                Poll::Done(json!({}))
            }
        }
    }
}

impl Condition {
    fn parse(word: &str) -> Result<Self, String> {
        match word {
            "terrain" => Ok(Self::Terrain),
            "plan" => Ok(Self::Plan),
            "sky" => Ok(Self::Sky),
            "main" | "help" | "gameplay" => Ok(Self::Screen(screen(word)?)),
            other => Err(format!(
                "unknown wait condition: {other} (terrain, plan, sky, main, help, gameplay)"
            )),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Terrain => "terrain",
            Self::Plan => "plan",
            Self::Sky => "sky",
            Self::Screen(_) => "screen",
        }
    }

    fn met(&self, world: &World) -> bool {
        match self {
            Self::Terrain => world
                .get_resource::<BackgroundGeneration>()
                .is_some_and(BackgroundGeneration::is_complete),
            Self::Plan => matches!(world.get_resource::<WorldPlan>(), Some(WorldPlan::Done)),
            Self::Sky => world.get_resource::<WeatherMaps>().is_some(),
            Self::Screen(target) => world
                .get_resource::<State<Screen>>()
                .is_some_and(|screen| screen.get() == target),
        }
    }

    /// What to say when the wait ran out. A timeout with no reading of how far it got is
    /// a bug report nobody can act on.
    fn progress(&self, world: &World) -> String {
        match self {
            Self::Terrain => match world.get_resource::<BackgroundGeneration>() {
                Some(generation) => format!("{} chunks still to generate", generation.remaining()),
                None => "no world is loaded".into(),
            },
            Self::Plan => match world.get_resource::<WorldPlan>() {
                Some(plan) => format!("plan is at {}", observe::plan_stage(plan)),
                None => "no plan is running".into(),
            },
            Self::Sky => "the weather maps have not been baked".into(),
            Self::Screen(_) => match world.get_resource::<State<Screen>>() {
                Some(screen) => format!("screen is {:?}", screen.get()),
                None => "no screen state".into(),
            },
        }
    }
}

/// Press on the first frame, release on the second. Both edges matter: the panel opens
/// on `just_pressed` and a button left down would leak into whatever runs next.
fn click(world: &mut World, button: MouseButton, elapsed: u32, done: Value) -> Poll {
    let Some(window) = primary_window(world) else {
        return Poll::Failed("no primary window".into());
    };
    if elapsed == 0 {
        write_mouse(world, window, button, ButtonState::Pressed);
        Poll::Running
    } else {
        write_mouse(world, window, button, ButtonState::Released);
        Poll::Done(done)
    }
}

/// Where a tile currently sits on screen, or `None` if it is off it.
fn tile_on_screen(world: &mut World, tile: IVec2) -> Option<Vec2> {
    let mut cameras = world.query_filtered::<(&Camera, &Transform), With<WorldCamera>>();
    let (camera, transform) = cameras.iter(world).next()?;
    // The camera's own `Transform`, not its `GlobalTransform`, for the reason
    // `city_panel.rs` gives at the matching conversion: the pan writes the first in
    // `Update` and propagation runs in `PostUpdate`, so the global one is a frame behind.
    camera
        .world_to_viewport(
            &GlobalTransform::from(*transform),
            tile_translation(tile).extend(0.0),
        )
        .ok()
}

fn primary_window(world: &mut World) -> Option<Entity> {
    let mut windows = world.query_filtered::<Entity, With<PrimaryWindow>>();
    windows.iter(world).next()
}

fn set_cursor(world: &mut World, position: Vec2) -> bool {
    let Some(entity) = primary_window(world) else {
        return false;
    };
    // Only the window's own record of the cursor, not the desktop's — `set_cursor_position`
    // does not move the real pointer. So a scenario clicking around does not steal the
    // mouse, and the only thing that overwrites it is genuine mouse motion.
    match world.get_mut::<Window>(entity) {
        Some(mut window) => {
            window.set_cursor_position(Some(position));
            true
        }
        None => false,
    }
}

fn write_key(world: &mut World, window: Entity, key_code: KeyCode, state: ButtonState) {
    world.write_message(KeyboardInput {
        key_code,
        // `keyboard_input_system` only reads `key_code` and `state`; the logical key is
        // layout-dependent and nothing in this crate looks at it.
        logical_key: Key::Unidentified(NativeKey::Unidentified),
        state,
        text: None,
        repeat: false,
        window,
    });
}

fn write_mouse(world: &mut World, window: Entity, button: MouseButton, state: ButtonState) {
    world.write_message(MouseButtonInput {
        button,
        state,
        window,
    });
}

fn screen(word: &str) -> Result<Screen, String> {
    match word {
        "main" => Ok(Screen::Main),
        "help" => Ok(Screen::Help),
        "gameplay" => Ok(Screen::Gameplay),
        other => Err(format!("unknown screen: {other} (main, help, gameplay)")),
    }
}

fn key_code(word: &str) -> Result<KeyCode, String> {
    match word.to_ascii_lowercase().as_str() {
        "w" => Ok(KeyCode::KeyW),
        "a" => Ok(KeyCode::KeyA),
        "s" => Ok(KeyCode::KeyS),
        "d" => Ok(KeyCode::KeyD),
        "escape" | "esc" => Ok(KeyCode::Escape),
        "equal" | "plus" => Ok(KeyCode::Equal),
        "minus" => Ok(KeyCode::Minus),
        "space" => Ok(KeyCode::Space),
        other => Err(format!("unknown key: {other}")),
    }
}

fn number<T: std::str::FromStr>(word: &str) -> Result<T, String> {
    word.parse().map_err(|_| format!("not a number: {word}"))
}

fn optional_number<T: std::str::FromStr>(word: Option<&&str>, fallback: T) -> Result<T, String> {
    match word {
        Some(word) => number(word),
        None => Ok(fallback),
    }
}

/// Accepts `1/60` as well as `0.0166`, because a frame budget is the thing being named
/// and writing it as a fraction is how anyone would say it.
fn seconds(word: &str) -> Result<f32, String> {
    if let Some((numerator, denominator)) = word.split_once('/') {
        let numerator: f32 = number(numerator)?;
        let denominator: f32 = number(denominator)?;
        if denominator == 0.0 {
            return Err("a delta cannot be divided by zero".into());
        }
        return Ok(numerator / denominator);
    }
    number(word)
}
