//! Headless test for the on-screen LIC demo scene (`cargo run`).
//!
//! Verifies `setup_lic_display` spawns a quad + camera and registers a
//! `LicMaterial` whose noise and vector-field textures exist. Full
//! pixel readback needs a display runner and stays out of `cargo test`.

use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::sprite::Material2dPlugin;
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;
use line_integral_convolution_gpu::lic::{LicMaterial, LicParams, setup_lic_display};

#[test]
fn test_lic_display_scene_wires_up() {
    let mut headless = App::new();
    headless
        .add_plugins(
            DefaultPlugins
                .build()
                .disable::<WinitPlugin>()
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                }),
        )
        .add_plugins(Material2dPlugin::<LicMaterial>::default())
        .add_systems(Startup, setup_lic_display);
    headless.finish();
    // NOTE: no `update()` — driving the full schedule needs the render
    // extraction channels that only exist with a display runner. Running the
    // Startup schedule directly is enough: the scene setup touches only the
    // main world (entities + asset registration).
    headless.world_mut().run_schedule(Startup);

    assert!(
        headless.get_sub_app(RenderApp).is_some(),
        "RenderApp sub-app missing"
    );

    // A Camera2d and a LicMaterial-driven quad must exist.
    let mut cameras = headless
        .world_mut()
        .query_filtered::<(), With<Camera2d>>();
    assert!(
        cameras.iter(headless.world()).next().is_some(),
        "setup_lic_display must spawn a Camera2d"
    );

    let material = headless
        .world()
        .get_resource::<Assets<LicMaterial>>()
        .expect("LicMaterial assets must be registered")
        .iter()
        .next()
        .expect("one LicMaterial expected")
        .1
        .clone();
    let images = headless
        .world()
        .get_resource::<Assets<Image>>()
        .expect("Image assets must exist");
    assert!(
        images.get(&material.noise_texture).is_some(),
        "noise texture must exist"
    );
    assert!(
        images.get(&material.vector_field_texture).is_some(),
        "vector field texture must exist"
    );
    assert_eq!(
        material.params.forward_count,
        LicParams::triangular_kernel(12).forward_count,
        "kernel must be the demo triangular kernel (radius 12)"
    );
}
