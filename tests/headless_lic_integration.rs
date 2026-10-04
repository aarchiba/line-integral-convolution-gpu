//! Headless tests for Ticket 5: LIC Shader - Basic Streamline Integration (issue #6)
//!
//! Full GPU pixel-perfect readback needs a display runner (out of `cargo test`
//! scope, same as the identity pass). These tests cover what is
//! headless-runnable:
//! - updated `lic.wgsl` validates via the headless RenderDevice
//! - `triangular_kernel` matches the ticket's specified values/counts
//! - a pure-CPU reference mirroring the shader's exact loop semantics
//!   (center sampled once; backward starts one step behind) reproduces 1D
//!   horizontal/vertical convolution of known noise with the kernel.

use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;
use line_integral_convolution_gpu::lic::LicParams;

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

fn strip_bevy_imports(source: &str) -> String {
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
    kept.join("\n")
}

#[test]
fn test_lic_streamline_shader_compiles() {
    let source =
        std::fs::read_to_string("assets/shaders/lic.wgsl").expect("lic.wgsl must exist");
    // Streamline contract markers (not the old identity passthrough).
    assert!(
        source.contains("vector_field_texture"),
        "LIC shader must sample the vector field texture"
    );
    assert!(
        source.contains("step_uv"),
        "LIC shader must derive a UV-space step from the vector field"
    );
    assert!(
        source.contains("forward_count"),
        "LIC shader must loop over forward_count"
    );
    assert!(
        source.contains("backward_count"),
        "LIC shader must loop over backward_count"
    );
    assert!(
        source.contains("forward_weights"),
        "LIC shader must read forward_weights"
    );
    assert!(
        source.contains("backward_weights"),
        "LIC shader must read backward_weights"
    );
    assert!(
        source.contains("wsum") || source.contains("sum"),
        "LIC shader must normalize by the weight sum"
    );
    // Must no longer be the identity passthrough (identity returned the raw
    // noise sample with no accumulation).
    assert!(
        !source.contains("IDENTITY PASS"),
        "lic.wgsl must no longer be the identity pass"
    );

    let body = strip_bevy_imports(&source);
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
    let _module = render_device.create_shader_module(
        bevy::render::render_resource::ShaderModuleDescriptor {
            label: Some("lic_streamline_test"),
            source: bevy::render::render_resource::ShaderSource::Wgsl(validated.into()),
        },
    );
}

#[test]
fn test_triangular_kernel_values() {
    // Ticket spec: forward[0]=1.0, forward[i]=1-i/r, backward[i]=1-(i+1)/r,
    // forward_count=r+1, backward_count=r.
    let params = LicParams::triangular_kernel(4);
    assert_eq!(params.forward_count, 5);
    assert_eq!(params.backward_count, 4);
    let eps = 1e-6;
    assert!((params.forward(0) - 1.0).abs() < eps);
    assert!((params.forward(1) - 0.75).abs() < eps);
    assert!((params.forward(2) - 0.5).abs() < eps);
    assert!((params.forward(3) - 0.25).abs() < eps);
    assert!((params.forward(4) - 0.0).abs() < eps);
    assert!((params.backward(0) - 0.75).abs() < eps);
    assert!((params.backward(1) - 0.5).abs() < eps);
    assert!((params.backward(2) - 0.25).abs() < eps);
    assert!((params.backward(3) - 0.0).abs() < eps);
}

#[test]
fn test_triangular_kernel_radius_zero_is_identity() {
    let params = LicParams::triangular_kernel(0);
    assert_eq!(params.forward_count, 1);
    assert_eq!(params.backward_count, 0);
    assert_eq!(params.forward(0), 1.0);
}

/// Pure-CPU mirror of the shader's per-pixel loop.
///
/// - `noise`: row-major grayscale in [0,1], size w*h
/// - `step_px`: constant-field step in pixels (e.g. (1,0) = one pixel right)
/// - sampling: nearest texel with clamp-to-edge (matches the GPU sampler for
///   on-texel UVs produced by whole-pixel steps from pixel centers)
fn cpu_lic_pixel(
    noise: &[f32],
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    step_px: (i32, i32),
    params: &LicParams,
) -> f32 {
    let sample = |sx: i32, sy: i32| -> f32 {
        let cx = sx.clamp(0, w as i32 - 1) as usize;
        let cy = sy.clamp(0, h as i32 - 1) as usize;
        noise[cy * w + cx]
    };
    let mut acc = 0.0f32;
    let mut wsum = 0.0f32;
    // Forward: (x,y) + i*step for i in 0..forward_count
    for i in 0..params.forward_count as i32 {
        let wgt = params.forward(i as usize);
        acc += sample(x as i32 + i * step_px.0, y as i32 + i * step_px.1) * wgt;
        wsum += wgt;
    }
    // Backward: (x,y) - (i+1)*step for i in 0..backward_count
    // (center sampled exactly once; see lic.wgsl comment).
    for i in 0..params.backward_count as i32 {
        let wgt = params.backward(i as usize);
        let k = i + 1;
        acc += sample(x as i32 - k * step_px.0, y as i32 - k * step_px.1) * wgt;
        wsum += wgt;
    }
    acc / wsum
}

fn horizontal_gradient(w: usize, h: usize) -> Vec<f32> {
    (0..h)
        .flat_map(|_| (0..w).map(move |x| x as f32 / (w - 1) as f32))
        .collect()
}

fn vertical_gradient(w: usize, h: usize) -> Vec<f32> {
    (0..h)
        .flat_map(|y| (0..w).map(move |_| y as f32 / (h - 1) as f32))
        .collect()
}

#[test]
fn test_constant_horizontal_field_matches_1d_convolution() {
    // radius 2: weights forward=[1, 0.5, 0], backward=[0.5, 0]; the zero end
    // taps contribute nothing, so the effective stencil is {-1, 0, +1} with
    // weights {0.5, 1.0, 0.5} / 2.0.
    let (w, h) = (8usize, 4usize);
    let noise = horizontal_gradient(w, h);
    let params = LicParams::triangular_kernel(2);

    // Hand-computed interior pixel: x=3 -> noise values {2/7, 3/7, 4/7}.
    let expected = (0.5 * (2.0 / 7.0) + 1.0 * (3.0 / 7.0) + 0.5 * (4.0 / 7.0)) / 2.0;
    let got = cpu_lic_pixel(&noise, w, h, 3, 1, (1, 0), &params);
    assert!(
        (got - expected).abs() < 1e-5,
        "horizontal LIC must equal 1D convolution: got {got}, want {expected}"
    );

    // No vertical transfer: gradient varies only in x, so every row agrees.
    for y in 0..h {
        let v = cpu_lic_pixel(&noise, w, h, 3, y, (1, 0), &params);
        assert!(
            (v - expected).abs() < 1e-5,
            "horizontal field must not transfer values vertically (row {y}: {v})"
        );
    }

    // Clamp-to-edge at the left border: x=0 samples {-1->0, 0, +1}.
    let edge_expected = (0.5 * (0.0 / 7.0) + 1.0 * (0.0 / 7.0) + 0.5 * (1.0 / 7.0)) / 2.0;
    let edge = cpu_lic_pixel(&noise, w, h, 0, 0, (1, 0), &params);
    assert!(
        (edge - edge_expected).abs() < 1e-5,
        "border must clamp: got {edge}, want {edge_expected}"
    );
}

#[test]
fn test_constant_vertical_field_matches_1d_convolution() {
    let (w, h) = (4usize, 8usize);
    let noise = vertical_gradient(w, h);
    let params = LicParams::triangular_kernel(2);

    // Interior pixel y=3 -> values {2/7, 3/7, 4/7} vertically.
    let expected = (0.5 * (2.0 / 7.0) + 1.0 * (3.0 / 7.0) + 0.5 * (4.0 / 7.0)) / 2.0;
    let got = cpu_lic_pixel(&noise, w, h, 1, 3, (0, 1), &params);
    assert!(
        (got - expected).abs() < 1e-5,
        "vertical LIC must equal 1D convolution: got {got}, want {expected}"
    );

    // No horizontal transfer: gradient varies only in y, so every column agrees.
    for x in 0..w {
        let v = cpu_lic_pixel(&noise, w, h, x, 3, (0, 1), &params);
        assert!(
            (v - expected).abs() < 1e-5,
            "vertical field must not transfer values horizontally (col {x}: {v})"
        );
    }
}
