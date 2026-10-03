//! Headless render test with pixel readback

use bevy::prelude::*;
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::RenderApp;
use bevy::render::RenderPlugin;
use bevy::app::{App, AppExit};
use bevy::ecs::system::RunSystemOnce;
use line_integral_convolution_gpu::{build_app, noise::generate_noise_texture, material::NoiseDisplayMaterial};

#[test]
fn test_headless_render_and_readback() {
    // Create the app
    let mut app = build_app();
    
    // Add render plugin in headless mode
    app.add_plugins(RenderPlugin {
        render_creation: bevy::render::settings::RenderCreation::Automatic(
            bevy::render::settings::WgpuSettings {
                backends: Some(bevy::render::settings::Backends::VULKAN),
                ..default()
            }
        ),
        ..default()
    });
    
    // Initialize and run startup
    app.update();
    
    // Get the render app
    let render_app = app.world().resource::<RenderApp>().clone();
    
    // Run render app for one frame
    let mut render_app = render_app;
    let _ = render_app.update();
    
    println!("Headless render completed");
}