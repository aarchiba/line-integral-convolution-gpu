//! Animated LIC demo: per-frame updates of ink, vector field, and kernel.
//!
//! The whole program is one [`LicSceneSpec`]: three generator functions that
//! each take the time in seconds. Init and animation call the *same*
//! generators (at `t = 0` and per frame), so nothing is specified twice.
//! `main` only chooses which inputs animate by scheduling update systems.
//!
//! - ink: fresh white noise every frame (the classic LIC "ink").
//! - field: a vortex orbiting the center.
//! - kernel: triangular radius breathing 4..16.
//!
//! Run with: `cargo run --example lic_animated`
//! (add `--release` for full speed; debug builds regenerate ~230k pixels of
//! noise + field per frame on the CPU).

use std::sync::Mutex;

use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use line_integral_convolution_gpu::{
    build_app,
    lic::{
        LicParams, LicSceneSpec, update_lic_field, update_lic_ink, update_lic_kernel,
        vector_field::analytical,
    },
    noise::generate_noise_grayscale,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn main() {
    let rng = Mutex::new(ChaCha8Rng::seed_from_u64(7));
    build_app(LicSceneSpec::new(
        640,
        360,
        move |w, h, _t| {
            // Fresh white noise every frame.
            let mut rng = rng.lock().unwrap();
            (0..w * h).map(|_| rng.gen::<f32>()).collect()
        },
        |w, h, t| {
            let cx = w as f32 / 2.0 + 120.0 * (t * 0.5).cos();
            let cy = h as f32 / 2.0 + 80.0 * (t * 0.5).sin();
            analytical::vortex(w, h, (cx, cy), 60.0)
        },
        |t| LicParams::triangular_kernel((10.0 + 6.0 * (t * 1.5).sin()).round().clamp(1.0, 64.0) as u32),
    ))
    // FPS counter: logs `fps` to the console once per second.
    .add_plugins((FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin::default()))
    .add_systems(Update, (update_lic_ink, update_lic_field, update_lic_kernel))
    .run();
}

/// Fixed-ink variant of the noise generator, kept for reference: the same
/// Perlin the binary shows. Swap it into the spec above and drop
/// `update_lic_ink` to hold the ink fixed.
#[allow(dead_code)]
fn static_perlin_ink(w: u32, h: u32, _t: f32) -> Vec<f32> {
    generate_noise_grayscale(w, h, 0.06)
}
