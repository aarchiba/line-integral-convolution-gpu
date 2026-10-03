//! Headless render test with pixel readback

use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::sprite::Material2dPlugin;
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;
use line_integral_convolution_gpu::material::{NoiseDisplayMaterial, setup_noise_display};

#[test]
fn test_headless_render_and_readback() {
    // NOTE: build_app() is not used here — even constructing it initializes
    // winit's event loop, which requires the main thread. Full pixel readback
    // needs a display runner and stays out of `cargo test`.

    // Headless renderer check: DefaultPlugins without winit so finish() can
    // run on a test thread and initialize the GPU adapter.
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
        .add_plugins(Material2dPlugin::<NoiseDisplayMaterial>::default())
        .add_systems(Startup, setup_noise_display);
    headless.finish();

    assert!(
        headless.get_sub_app(RenderApp).is_some(),
        "RenderApp sub-app missing"
    );

    println!("Headless render completed");
}
