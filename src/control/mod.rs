//! Drive the running game from outside the process.
//!
//! The app keeps its real window, its real winit loop and its real swapchain — this
//! adds a way to talk to it, not a second way to run it. A headless driver was the
//! first design and was dropped: it would have verified a rendering path no human ever
//! takes, and a capture off the real swapchain is the only evidence worth looking at.
//!
//! **A command is synchronous from the client's side.** The reply is held until the
//! effect has actually happened — `wait plan` answers when the plan is done, `capture`
//! when the PNG is on disk — so a caller never sleeps and hopes. What blocks is the
//! *client*; the app runs on undisturbed, which is what lets a human watch a scenario
//! and grab the keyboard part-way through.
//!
//! Activation is the `WUSEL_CONTROL` environment variable rather than a cargo feature:
//! CI already builds `--all-features`, so a feature would need care in every recipe to
//! buy nothing, and unset this costs one environment read at startup.

use bevy::{log::BoxedLayer, prelude::*};

pub struct ControlPlugin;

/// The extra tracing layer that keeps the run's warnings and errors where `observe log`
/// can reach them.
///
/// Passed to `LogPlugin::custom_layer` in `main.rs` rather than installed by
/// [`ControlPlugin`], because a log layer has to exist before the logger does and
/// `LogPlugin` is built first. It is a bare `fn` pointer by `LogPlugin`'s definition, so
/// it cannot capture and inserts its own resource.
pub fn log_layer(app: &mut App) -> Option<BoxedLayer> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        log::layer(app)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = app;
        None
    }
}

impl Plugin for ControlPlugin {
    fn build(&self, app: &mut App) {
        // Unix sockets do not exist on wasm, and `just check-web` has to keep passing.
        #[cfg(not(target_arch = "wasm32"))]
        server::build(app);
        #[cfg(target_arch = "wasm32")]
        let _ = app;
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod command;
#[cfg(not(target_arch = "wasm32"))]
mod log;
#[cfg(not(target_arch = "wasm32"))]
mod observe;
#[cfg(not(target_arch = "wasm32"))]
mod server;
