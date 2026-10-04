//! Headless test for the spec-driven on-screen LIC scene.
//!
//! Verifies `setup_lic_scene` builds a quad + camera from the spec's
//! generators and that the kernel update system re-evaluates them. Full
//! pixel readback needs a display runner and stays out of `cargo test`.

use std::time::Duration;

use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::sprite::Material2dPlugin;
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;
use line_integral_convolution_gpu::lic::{
    LicMaterial, LicScene, LicSceneSpec, setup_lic_scene, update_lic_kernel,
};
use line_integral_convolution_gpu::noise::generate_noise_grayscale;

fn test_spec() -> LicSceneSpec {
    LicSceneSpec::new(
        64,
        36,
        |w, h, _t| generate_noise_grayscale(w, h, 0.06),
        |w, h, _t| vec![0.0; (w * h * 2) as usize],
        // Radius grows with time so the update path is observable: t=0 -> 4.
        |t| line_integral_convolution_gpu::lic::LicParams::triangular_kernel(4 + t as u32),
    )
}

#[test]
fn test_lic_scene_builds_from_spec_and_kernel_updates() {
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
        .insert_resource(test_spec())
        .add_systems(Startup, setup_lic_scene);
    headless.finish();
    // NOTE: no `update()` — driving the full schedule needs the render
    // extraction channels that only exist with a display runner. Running the
    // Startup schedule directly is enough: scene setup touches only the
    // main world (entities + asset registration).
    headless.world_mut().run_schedule(Startup);

    assert!(
        headless.get_sub_app(RenderApp).is_some(),
        "RenderApp sub-app missing"
    );

    // A Camera2d must exist.
    let mut cameras = headless
        .world_mut()
        .query_filtered::<(), With<Camera2d>>();
    assert!(
        cameras.iter(headless.world()).next().is_some(),
        "setup_lic_scene must spawn a Camera2d"
    );

    // Scene resource + material with both textures, kernel evaluated at t=0.
    let scene = headless
        .world()
        .get_resource::<LicScene>()
        .expect("LicScene resource must exist")
        .clone();
    let material = headless
        .world()
        .get_resource::<Assets<LicMaterial>>()
        .expect("LicMaterial assets must be registered")
        .get(&scene.material)
        .expect("scene material must exist")
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
        material.params.forward_count, 5,
        "kernel at t=0 must be triangular radius 4"
    );

    // Advance one second and run the kernel updater directly: radius 4 -> 5.
    headless
        .world_mut()
        .resource_mut::<Time>()
        .advance_by(Duration::from_secs(1));
    headless
        .world_mut()
        .run_system_cached(update_lic_kernel)
        .unwrap();
    let updated = headless
        .world()
        .get_resource::<Assets<LicMaterial>>()
        .expect("LicMaterial assets must be registered")
        .get(&scene.material)
        .expect("scene material must exist")
        .clone();
    assert_eq!(
        updated.params.forward_count, 6,
        "kernel update at t=1 must re-evaluate to radius 5"
    );
}
