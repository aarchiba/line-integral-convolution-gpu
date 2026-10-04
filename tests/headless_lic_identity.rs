//! Headless tests for Ticket 4: LIC Shader - Identity Pass (issue #5)
//!
//! Identity pass verifies pipeline connectivity: noise -> LicMaterial ->
//! offscreen -> blit. Full GPU pixel-perfect readback needs a display runner
//! (out of `cargo test` scope); these tests cover what is headless-runnable:
//! shader validation, LicParams identity, offscreen double-buffering, and a
//! no-panic RenderApp pipeline assembly.

use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::render_resource::TextureUsages;
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;
use line_integral_convolution_gpu::lic::{
    LicMaterial, LicParams, OffscreenTargets, create_offscreen_targets,
};
use line_integral_convolution_gpu::lic::{CpuVectorFieldNode, LicPlugin};

fn headless_app() -> App {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .build()
            .disable::<WinitPlugin>()
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            }),
    );
    app
}

#[test]
fn test_lic_identity_shader_compiles() {
    let source =
        std::fs::read_to_string("assets/shaders/lic.wgsl").expect("lic.wgsl must exist");
    // Identity contract: samples noise, ignores vector field for output.
    assert!(
        source.contains("textureSample(noise_texture"),
        "identity shader must sample noise_texture"
    );
    assert!(
        source.contains("vector_field_texture"),
        "identity shader must still bind vector field texture (pipeline connectivity)"
    );
    assert!(
        source.contains("LicParams") || source.contains("params"),
        "identity shader must bind LicParams uniform"
    );

    // Validate WGSL via the headless RenderDevice (panics on invalid WGSL).
    // NOTE: Bevy shaders use `#import` preprocessing, which raw wgpu/naga
    // cannot parse. Strip preprocessor lines and stub the imported
    // VertexOutput type so the remaining WGSL body is validated directly.
    // (Same pattern as assets/shaders/noise_display.wgsl.)
    let mut in_import = false;
    let mut kept: Vec<&str> = Vec::new();
    for l in source.lines() {
        let t = l.trim_start();
        if t.starts_with("#import") || t.starts_with("#define") {
            in_import = true;
            if l.contains('}') {
                in_import = false;
            }
            continue;
        }
        if in_import {
            if l.contains('}') {
                in_import = false;
            }
            continue;
        }
        if t.starts_with('#') {
            continue;
        }
        kept.push(l);
    }
    let body = kept.join("\n");
    let stub = "struct VertexOutput { @location(0) uv: vec2<f32>, };\n";
    let validated = format!("{stub}{body}");
    let mut app = headless_app();
    app.finish();
    let render_app = app
        .get_sub_app(RenderApp)
        .expect("RenderApp must exist after finish()");
    let render_device = render_app
        .world()
        .get_resource::<bevy::render::renderer::RenderDevice>()
        .expect("RenderDevice must exist");
    let _module = render_device.create_shader_module(bevy::render::render_resource::ShaderModuleDescriptor {
        label: Some("lic_identity_test"),
        source: bevy::render::render_resource::ShaderSource::Wgsl(validated.into()),
    });
}

#[test]
fn test_lic_params_identity() {
    let params = LicParams::identity();
    assert_eq!(params.forward_count, 1);
    assert_eq!(params.backward_count, 0);
    assert_eq!(params.forward(0), 1.0);
    for i in 1..256 {
        assert_eq!(params.forward(i), 0.0, "identity kernel must be [1, 0, 0, ...]");
    }
    for i in 0..256 {
        assert_eq!(params.backward(i), 0.0, "identity kernel has no backward taps");
    }
}

#[test]
fn test_offscreen_double_buffer() {
    let mut images = Assets::<Image>::default();
    let mut targets: OffscreenTargets = create_offscreen_targets(&mut images, 64, 32);
    assert_eq!(targets.width, 64);
    assert_eq!(targets.height, 32);

    for handle in [&targets.texture_a, &targets.texture_b] {
        let image = images.get(handle).expect("offscreen image must exist");
        let usage: TextureUsages = image.texture_descriptor.usage;
        assert!(
            usage.contains(TextureUsages::RENDER_ATTACHMENT),
            "offscreen must be renderable"
        );
        assert!(
            usage.contains(TextureUsages::COPY_SRC),
            "offscreen must support readback (COPY_SRC)"
        );
        assert_eq!(image.width(), 64);
        assert_eq!(image.height(), 32);
    }

    // Ping-pong: write A / read B, then swap.
    let (w0, r0) = (
        targets.write_texture().clone(),
        targets.read_texture().clone(),
    );
    assert_eq!(w0, targets.texture_a);
    assert_eq!(r0, targets.texture_b);
    targets.swap();
    assert_eq!(*targets.write_texture(), targets.texture_b);
    assert_eq!(*targets.read_texture(), targets.texture_a);
    targets.swap();
    assert_eq!(*targets.write_texture(), w0);
}

#[test]
fn test_headless_lic_pipeline_no_panic() {
    let mut app = headless_app();
    // CpuVectorFieldNode::new touches no GPU state until prepare(), so it is
    // safe to construct on the test thread as the ticket's dummy field node.
    let field_node = CpuVectorFieldNode::new(64, 64);
    app.add_plugins(LicPlugin::new(field_node));
    app.finish();

    assert!(
        app.get_sub_app(RenderApp).is_some(),
        "RenderApp sub-app missing"
    );
    assert!(
        app.world().get_resource::<OffscreenTargets>().is_some(),
        "LicPlugin must insert OffscreenTargets"
    );
    // LicMaterial asset type must be registered by LicPlugin.
    assert!(
        app.world()
            .get_resource::<Assets<LicMaterial>>()
            .is_some(),
        "LicMaterial assets must be registered"
    );
}
