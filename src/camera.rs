//! The app's one and only camera.
//!
//! It is spawned at startup and never despawned, because the menus need a camera
//! to render their UI just as much as gameplay needs one to look at the world.
//! UI is laid out in screen space, so driving this around with WASD moves the
//! world without disturbing anything on top of it.

use bevy::{
    input::mouse::{AccumulatedMouseScroll, MouseScrollUnit},
    prelude::*,
};

use crate::{gameplay::world_half_extent, screens::Screen};

/// World units per second at full tilt. A chunk is 512 units across, so this
/// crosses one chunk per second.
const CAMERA_SPEED: f32 = 512.0;

/// The range the orthographic scale is held to: a quarter of a world unit per
/// pixel up to four. Zooming further out is not a knob that has been turned down
/// for taste — chunk entities cover the screen, so their number grows with the
/// square of the scale, and past 4 that stops being affordable.
pub const MIN_ZOOM_SCALE: f32 = 0.25;
pub const MAX_ZOOM_SCALE: f32 = 4.0;

/// Zoom halves and doubles rather than sliding. The tiles are 8px pixel art
/// drawn with nearest-neighbour filtering, so at any ratio other than a power of
/// two their texels stop landing on whole pixels and the art crawls as you pan.
const ZOOM_STEP: f32 = 2.0;

/// Pixels of trackpad scroll that count as one wheel line.
const SCROLL_PIXELS_PER_TICK: f32 = 50.0;

#[derive(Component)]
pub struct WorldCamera;

/// Wheel scroll banked toward the next whole zoom step. A mouse reports whole
/// lines and a trackpad reports pixels by the dozen, so both are converted to
/// ticks and saved up until one adds to a step.
#[derive(Resource, Default)]
struct BankedScroll(f32);

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BankedScroll>();
        app.add_systems(Startup, spawn_camera);
        app.add_systems(
            Update,
            // Zoom first: the pan clamp depends on how much world is visible, so
            // panning against last frame's zoom could leave the camera outside
            // the world for a frame.
            (zoom_camera, move_camera)
                .chain()
                .run_if(in_state(Screen::Gameplay)),
        );
    }
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn((Camera2d, WorldCamera));
}

/// How much world the camera can see, from its centre out to its edge.
///
/// Everything that has to agree on what is on screen takes it from here — the
/// pan clamp below, and the chunk streamer deciding how far its entities must
/// reach.
pub fn visible_half_extent(camera: &Camera, projection: &Projection) -> Vec2 {
    let scale = match projection {
        Projection::Orthographic(orthographic) => orthographic.scale,
        _ => 1.0,
    };
    camera.logical_viewport_size().unwrap_or(Vec2::ZERO) / 2.0 * scale
}

fn zoom_camera(
    keys: Res<ButtonInput<KeyCode>>,
    scroll: Res<AccumulatedMouseScroll>,
    mut banked: ResMut<BankedScroll>,
    projection: Single<&mut Projection, With<WorldCamera>>,
) {
    let mut steps = banked.take_steps(&scroll);
    if keys.just_pressed(KeyCode::Equal) || keys.just_pressed(KeyCode::NumpadAdd) {
        steps += 1;
    }
    if keys.just_pressed(KeyCode::Minus) || keys.just_pressed(KeyCode::NumpadSubtract) {
        steps -= 1;
    }
    if steps == 0 {
        return;
    }

    let Projection::Orthographic(orthographic) = &mut *projection.into_inner() else {
        return;
    };
    orthographic.scale = stepped_scale(orthographic.scale, steps);
}

impl BankedScroll {
    fn take_steps(&mut self, scroll: &AccumulatedMouseScroll) -> i32 {
        self.0 += match scroll.unit {
            MouseScrollUnit::Line => scroll.delta.y,
            MouseScrollUnit::Pixel => scroll.delta.y / SCROLL_PIXELS_PER_TICK,
        };
        let whole = self.0.trunc();
        self.0 -= whole;
        whole as i32
    }
}

/// Applies `steps` of zoom. A positive step zooms *in*, which is a smaller
/// orthographic scale — fewer world units across the same screen.
///
/// Starting from 1.0 and only ever halving or doubling, with bounds that are
/// themselves powers of two, the scale can only ever land on a power of two.
fn stepped_scale(scale: f32, steps: i32) -> f32 {
    (scale * ZOOM_STEP.powi(-steps)).clamp(MIN_ZOOM_SCALE, MAX_ZOOM_SCALE)
}

fn move_camera(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    camera: Single<(&mut Transform, &Camera, &Projection), With<WorldCamera>>,
) {
    let (mut transform, camera, projection) = camera.into_inner();

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
    let visible = visible_half_extent(camera, projection);
    transform.translation = clamp_to_world(moved, visible).extend(transform.translation.z);
}

/// Stops the camera where the edge of the world reaches the edge of the screen,
/// so you can reach the world's border without panning off into the void.
///
/// `visible` is half of what the camera can see, in world units — which is the
/// viewport in logical pixels only at a scale of 1, so it has to come from
/// [`visible_half_extent`] rather than from the viewport directly.
fn clamp_to_world(position: Vec2, visible: Vec2) -> Vec2 {
    let limit = (world_half_extent() - visible).max(Vec2::ZERO);
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

    /// Zooming out shows more world, so the camera has to stop further from the
    /// edge — panning against the unscaled viewport is what would let you drive
    /// off into the void.
    #[test]
    fn zooming_out_stops_the_camera_further_from_the_edge() {
        let window = Vec2::new(960.0, 540.0);
        let far = world_half_extent() * 2.0;

        let close = clamp_to_world(far, window);
        let wide = clamp_to_world(far, window * MAX_ZOOM_SCALE);

        assert!(wide.x < close.x && wide.y < close.y);
        assert_eq!(wide, world_half_extent() - window * MAX_ZOOM_SCALE);
    }

    #[test]
    fn zoom_stops_at_the_ends_of_its_range() {
        assert_eq!(stepped_scale(MIN_ZOOM_SCALE, 1), MIN_ZOOM_SCALE);
        assert_eq!(stepped_scale(MAX_ZOOM_SCALE, -1), MAX_ZOOM_SCALE);
        assert_eq!(stepped_scale(1.0, 99), MIN_ZOOM_SCALE);
        assert_eq!(stepped_scale(1.0, -99), MAX_ZOOM_SCALE);
    }

    /// Every reachable scale must be a power of two, or the 8px art stops
    /// landing on whole pixels.
    #[test]
    fn every_zoom_level_is_a_power_of_two() {
        let mut scale = 1.0;
        for step in [1, 1, 1, -1, -1, -1, -1, -1, 1] {
            scale = stepped_scale(scale, step);
            assert_eq!(
                scale.log2().fract(),
                0.0,
                "scale {scale} is not a power of two"
            );
            assert!((MIN_ZOOM_SCALE..=MAX_ZOOM_SCALE).contains(&scale));
        }
    }

    #[test]
    fn a_step_in_zooms_and_a_step_out_zooms_back() {
        assert_eq!(stepped_scale(stepped_scale(1.0, 1), -1), 1.0);
        assert!(stepped_scale(1.0, 1) < 1.0, "a positive step zooms in");
    }

    /// A mouse reports whole lines; a trackpad reports pixels by the dozen, and
    /// a flick of it should not run the zoom through its whole range.
    #[test]
    fn scroll_is_banked_until_it_adds_up_to_a_step() {
        let mut banked = BankedScroll::default();
        let nudge = AccumulatedMouseScroll {
            unit: MouseScrollUnit::Pixel,
            delta: Vec2::new(0.0, SCROLL_PIXELS_PER_TICK / 2.0),
        };

        assert_eq!(banked.take_steps(&nudge), 0, "half a tick is not a step");
        assert_eq!(banked.take_steps(&nudge), 1, "the halves add up to one");

        let line = AccumulatedMouseScroll {
            unit: MouseScrollUnit::Line,
            delta: Vec2::new(0.0, -1.0),
        };
        assert_eq!(banked.take_steps(&line), -1, "a wheel line is a whole step");
    }
}
