//! Line Integral Convolution GPU - Library crate

use bevy::prelude::*;
use bevy::sprite::Material2dPlugin;

pub mod lic;
pub mod material;
pub mod noise;

use lic::{setup_lic_scene, LicMaterial, LicSceneSpec};

/// Build the app from a user-provided scene spec. There are no built-in
/// defaults for ink, vector field, or kernel: the spec's generators produce
/// the initial values (at `t = 0`) and any scheduled `update_lic_*` system
/// re-evaluates them every frame.
pub fn build_app(spec: LicSceneSpec) -> App {
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
    .insert_resource(spec)
    .add_systems(Startup, setup_lic_scene);

    app
}
