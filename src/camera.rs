//! The app's one and only camera.
//!
//! It is spawned at startup and never despawned, because the menus need a camera
//! to render their UI just as much as gameplay needs one to look at the world.
//! UI is laid out in screen space, so driving this around with WASD moves the
//! world without disturbing anything on top of it.

use bevy::prelude::*;

use crate::{gameplay::world_half_extent, screens::Screen};

/// World units per second at full tilt. A chunk is 512 units across, so this
/// crosses one chunk per second.
const CAMERA_SPEED: f32 = 512.0;

#[derive(Component)]
pub struct WorldCamera;

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_camera);
        app.add_systems(Update, move_camera.run_if(in_state(Screen::Gameplay)));
    }
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn((Camera2d, WorldCamera));
}

fn move_camera(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    camera: Single<(&mut Transform, &Camera), With<WorldCamera>>,
) {
    let (mut transform, camera) = camera.into_inner();

    let mut direction = Vec2::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        direction.y += 1.0;
    }
    if keys.pressed(KeyCode::KeyS) {
        direction.y -= 1.0;
    }
    if keys.pressed(KeyCode::KeyA) {
        direction.x -= 1.0;
    }
    if keys.pressed(KeyCode::KeyD) {
        direction.x += 1.0;
    }
    // Normalizing keeps diagonals from being faster than the cardinals, and
    // rejects the no-keys-held case in the same step.
    let Some(direction) = direction.try_normalize() else {
        return;
    };

    let moved = transform.translation.truncate() + direction * CAMERA_SPEED * time.delta_secs();
    let half_viewport = camera.logical_viewport_size().unwrap_or(Vec2::ZERO) / 2.0;
    transform.translation = clamp_to_world(moved, half_viewport).extend(transform.translation.z);
}

/// Stops the camera where the edge of the world reaches the edge of the screen,
/// so you can reach the world's border without panning off into the void.
///
/// `half_viewport` is in world units, which for the default 2D projection is the
/// same as logical pixels.
fn clamp_to_world(position: Vec2, half_viewport: Vec2) -> Vec2 {
    let limit = (world_half_extent() - half_viewport).max(Vec2::ZERO);
    position.clamp(-limit, limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_camera_stops_before_the_world_runs_out() {
        let half_viewport = Vec2::new(960.0, 540.0);
        let far = world_half_extent() * 2.0;
        assert_eq!(
            clamp_to_world(far, half_viewport),
            world_half_extent() - half_viewport
        );
    }

    #[test]
    fn a_position_inside_the_world_is_left_alone() {
        let position = Vec2::new(1234.0, -567.0);
        assert_eq!(clamp_to_world(position, Vec2::new(960.0, 540.0)), position);
    }

    /// A viewport wider than the world would otherwise give an inverted clamp
    /// range, which `Vec2::clamp` treats as a contract violation.
    #[test]
    fn a_viewport_larger_than_the_world_pins_the_camera_to_the_middle() {
        let huge = world_half_extent() * 4.0;
        assert_eq!(clamp_to_world(Vec2::splat(100.0), huge), Vec2::ZERO);
    }
}
