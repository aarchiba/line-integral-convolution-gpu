//! Property tests for analytical vector fields (issue #7).
//!
//! Rendering properties translated into executable assertions over the CPU
//! LIC reference (`lic::cpu_reference`, which mirrors `lic.wgsl` loop
//! semantics: center sampled once, backward starts one step behind,
//! nearest + clamp-to-edge). Full GPU pixel-perfect readback still needs a
//! display runner, so these run the exact algorithm the shader implements.
//!
//! Properties:
//! - horizontal field → no vertical transfer of values
//! - vertical field → no horizontal transfer of values
//! - vortex field → rotational symmetry (90° CW)
//! - saddle field → mirror symmetry (left-right)
//! - constant fields → exact CPU convolution ground truth

use line_integral_convolution_gpu::lic::cpu_reference::{
    cpu_lic_image_constant, cpu_lic_image_general, mirror_field_lr, mirror_image_lr,
    rotate_field_cw, rotate_image_cw,
};
use line_integral_convolution_gpu::lic::vector_field::analytical;
use line_integral_convolution_gpu::lic::LicParams;
use proptest::prelude::*;
use proptest::test_runner::Config as ProptestConfig;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Horizontal field: output row y depends only on noise row y.
    /// Noise constant across rows (each row identical) => all output rows equal.
    #[test]
    fn horizontal_field_has_no_vertical_transfer(
        h in 2usize..8,
        radius in 1u32..4,
        seed_row in prop::collection::vec(0.0f32..1.0, 4..10),
    ) {
        let w = seed_row.len();
        let noise: Vec<f32> = (0..h).flat_map(|_| seed_row.iter().copied()).collect();
        let field = analytical::constant_horizontal(w as u32, h as u32, 1.0);
        let params = LicParams::triangular_kernel(radius);
        let out = cpu_lic_image_general(&noise, &field, w, h, &params);
        for y in 1..h {
            for x in 0..w {
                prop_assert!(
                    (out[y * w + x] - out[x]).abs() < 1e-5,
                    "row {y} differs from row 0 at x={x} (w={w},h={h},r={radius})"
                );
            }
        }
    }

    /// Vertical field: output column x depends only on noise column x.
    #[test]
    fn vertical_field_has_no_horizontal_transfer(
        w in 2usize..8,
        radius in 1u32..4,
        seed_col in prop::collection::vec(0.0f32..1.0, 4..10),
    ) {
        let h = seed_col.len();
        let noise: Vec<f32> = seed_col.iter().flat_map(|v| vec![*v; w]).collect();
        let field = analytical::constant_vertical(w as u32, h as u32, 1.0);
        let params = LicParams::triangular_kernel(radius);
        let out = cpu_lic_image_general(&noise, &field, w, h, &params);
        for y in 0..h {
            for x in 1..w {
                prop_assert!(
                    (out[y * w + x] - out[y * w]).abs() < 1e-5,
                    "col {x} differs from col 0 at y={y} (w={w},h={h},r={radius})"
                );
            }
        }
    }

    /// Vortex at the image center: rotating noise + field 90° CW rotates the
    /// output the same way.
    #[test]
    fn vortex_field_has_rotational_symmetry(
        n in 5usize..9,
        radius in 1u32..3,
        noise in prop::collection::vec(0.0f32..1.0, 25..81),
        strength in 1.0f32..4.0,
    ) {
        let n = n.max(5);
        let noise: Vec<f32> = noise.into_iter().cycle().take(n * n).collect();
        let c = ((n - 1) as f32) / 2.0;
        let field = analytical::vortex(n as u32, n as u32, (c, c), strength);
        let params = LicParams::triangular_kernel(radius);
        let out = cpu_lic_image_general(&noise, &field, n, n, &params);
        let rot_noise = rotate_image_cw(&noise, n);
        let rot_field = rotate_field_cw(&field, n);
        let rot_out = cpu_lic_image_general(&rot_noise, &rot_field, n, n, &params);
        let expected = rotate_image_cw(&out, n);
        for i in 0..n * n {
            prop_assert!(
                (rot_out[i] - expected[i]).abs() < 1e-4,
                "rotation mismatch at {i} (n={n},r={radius}): {} vs {}",
                rot_out[i],
                expected[i]
            );
        }
    }

    /// Saddle at the image center: mirroring noise + field left-right mirrors
    /// the output the same way.
    #[test]
    fn saddle_field_has_mirror_symmetry(
        w in 5usize..9,
        h in 5usize..9,
        radius in 1u32..3,
        noise in prop::collection::vec(0.0f32..1.0, 25..81),
        strength in 0.2f32..1.0,
    ) {
        let noise: Vec<f32> = noise.into_iter().cycle().take(w * h).collect();
        let cx = ((w - 1) as f32) / 2.0;
        let cy = ((h - 1) as f32) / 2.0;
        let field = analytical::saddle(w as u32, h as u32, (cx, cy), strength);
        let params = LicParams::triangular_kernel(radius);
        let out = cpu_lic_image_general(&noise, &field, w, h, &params);
        let mir_noise = mirror_image_lr(&noise, w, h);
        let mir_field = mirror_field_lr(&field, w, h);
        let mir_out = cpu_lic_image_general(&mir_noise, &mir_field, w, h, &params);
        let expected = mirror_image_lr(&out, w, h);
        for i in 0..w * h {
            prop_assert!(
                (mir_out[i] - expected[i]).abs() < 1e-4,
                "mirror mismatch at {i} (w={w},h={h},r={radius}): {} vs {}",
                mir_out[i],
                expected[i]
            );
        }
    }
}

/// Exact ground truth: constant horizontal field with radius 2 on a linear
/// gradient must equal the hand-computed 1D convolution. (The kernel's zero
/// end taps contribute nothing, so the effective stencil is
/// {-1, 0, +1} with weights {0.5, 1.0, 0.5} / 2.0.)
#[test]
fn constant_horizontal_matches_exact_convolution() {
    let (w, h) = (8usize, 4usize);
    let noise: Vec<f32> = (0..h)
        .flat_map(|_| (0..w).map(move |x| x as f32 / (w - 1) as f32))
        .collect();
    let params = LicParams::triangular_kernel(2);
    let out = cpu_lic_image_constant(&noise, w, h, (1, 0), &params);
    let expected = (0.5 * (2.0 / 7.0) + 1.0 * (3.0 / 7.0) + 0.5 * (4.0 / 7.0)) / 2.0;
    assert!(
        (out[w + 3] - expected).abs() < 1e-5,
        "interior pixel must equal 1D convolution: {} vs {expected}",
        out[w + 3]
    );
    // Border clamps: x=0 samples {-1->0, 0, +1}.
    let edge = (0.5 * (0.0 / 7.0) + 1.0 * (0.0 / 7.0) + 0.5 * (1.0 / 7.0)) / 2.0;
    assert!(
        (out[0] - edge).abs() < 1e-5,
        "border must clamp: {} vs {edge}",
        out[0]
    );
}

/// The general (varying-field) path must reduce exactly to the constant path
/// for uniform fields.
#[test]
fn general_path_matches_constant_path_on_uniform_fields() {
    let (w, h) = (8usize, 6usize);
    let noise: Vec<f32> = (0..w * h).map(|i| (i as f32 + 1.0) / (w * h) as f32).collect();
    let params = LicParams::triangular_kernel(2);
    for (field, step) in [
        (
            analytical::constant_horizontal(w as u32, h as u32, 1.0),
            (1, 0),
        ),
        (
            analytical::constant_vertical(w as u32, h as u32, 1.0),
            (0, 1),
        ),
    ] {
        let general = cpu_lic_image_general(&noise, &field, w, h, &params);
        let exact = cpu_lic_image_constant(&noise, w, h, step, &params);
        for (i, (g, c)) in general.iter().zip(exact.iter()).enumerate() {
            assert!(
                (g - c).abs() < 1e-5,
                "pixel {i}: general {g} != constant {c}"
            );
        }
    }
}
