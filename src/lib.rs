//! Line Integral Convolution GPU - Library crate

use bevy::prelude::*;
use bevy::sprite::Material2dPlugin;

pub mod lic;
pub mod material;
pub mod noise;

use lic::{setup_lic_display, LicMaterial};

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
    .add_plugins(Material2dPlugin::<LicMaterial>::default())
    .add_systems(Startup, setup_lic_display);

    app
}
