//! Kernel-animated LIC: ink and vector field held fixed, kernel does the work.
//!
//! The whole program is one [`LicSceneSpec`]: the ink and field generators
//! ignore time (written once, used for init), the kernel generator carries
//! the animation. Only `update_lic_kernel` is scheduled, so no redundant
//! per-frame work happens for the fixed inputs.
//!
//! The kernel is a cosine wave rolling along the streamline under a
//! triangular envelope: the stripes move at `SPEED_PX_PER_SEC`, which reads
//! as flow along the (static) vortex field. Ink is fixed seeded white noise
//! (uncorrelated per pixel, so the streaks read clearly).
//!
//! Run with: `cargo run --example kernel_flow`

use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use line_integral_convolution_gpu::{
    build_app,
    lic::{update_lic_kernel, vector_field::analytical, LicParams, LicSceneSpec},
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Half-length of the kernel in steps (forward and backward).
const RADIUS: i32 = 64;
/// Cosine wavelength along the streamline, in kernel bins.
const SPATIAL_WAVELENGTH: f32 = 32.0;
/// Stripe travel speed along the streamline, in pixels per second.
const SPEED_PX_PER_SEC: f32 = 10.0;

fn main() {
    build_app(LicSceneSpec::new(
        640,
        360,
        white_noise_ink,
        |w, h, _t| {
            let cx = (w - 1) as f32 / 2.0;
            let cy = (h - 1) as f32 / 2.0;
            analytical::vortex(w, h, (cx, cy), 4.0)
        },
        animate_kernel,
    ))
    // FPS counter: logs `fps` to the console once per second.
    .add_plugins((FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin::default()))
    .add_systems(Update, update_lic_kernel)
    .run();
}

/// Fixed white-noise ink: one uniform sample per pixel in `[0, 1)`, seeded so
/// the image is identical on every run. Seeded fresh on each call (same seed),
/// so init and any re-evaluation agree.
fn white_noise_ink(w: u32, h: u32, _t: f32) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    (0..w * h).map(|_| rng.gen::<f32>()).collect()
}

/// Rolling kernel for time `t` (seconds): triangular envelope times a cosine
/// wave travelling in the +forward direction, normalized to sum to 1.0.
///
/// Signed step `k` is 0 at the center, positive forward, negative backward;
/// it maps onto the [`LicParams`] split layout as `forward[k]` /
/// `backward[-k - 1]`.
fn animate_kernel(t: f32) -> LicParams {
    // Cosine phase at signed step k: stripes travel forward over time.
    let coeff = |k: i32| -> f32 {
        let x = k as f32;
        let envelope = 1.0 - (x.abs() / RADIUS as f32);
        let phase = (x - SPEED_PX_PER_SEC * t) / SPATIAL_WAVELENGTH * std::f32::consts::TAU;
        envelope * (1.0 + phase.cos())
    };

    let radius = RADIUS as usize;
    let mut forward = Vec::with_capacity(radius + 1);
    let mut backward = Vec::with_capacity(radius);
    for i in 0..=radius {
        forward.push(coeff(i as i32));
    }
    for i in 0..radius {
        backward.push(coeff(-(i as i32) - 1));
    }

    // Normalize so the coefficients sum to 1.0 (the shader divides by the
    // weight sum anyway; this keeps brightness exactly stable).
    let sum: f32 = forward.iter().chain(backward.iter()).sum();
    if sum > 0.0 {
        for w in forward.iter_mut().chain(backward.iter_mut()) {
            *w /= sum;
        }
    }

    LicParams::from_weights(&forward, &backward)
}
