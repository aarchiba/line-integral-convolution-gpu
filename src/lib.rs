//! Line Integral Convolution GPU - Library crate

use bevy::prelude::*;
use bevy::sprite::Material2dPlugin;

pub mod noise;
pub mod material;
pub mod lic;

use material::{NoiseDisplayMaterial, setup_noise_display};

pub fn build_app() -> App {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "LIC Vector Field".into(),
            resolution: (1280.0_f32, 720.0_f32).into(),
            ..default()
        }),
        ..default()
    }))
    .insert_resource(ClearColor(Color::srgb(0.1, 0.2, 0.4)))
    .add_plugins(Material2dPlugin::<NoiseDisplayMaterial>::default())
    .add_systems(Startup, setup_noise_display);

    app
}