//! CPU reference for LIC (issue #7).
//!
//! Pure-CPU mirror of `assets/shaders/lic.wgsl` loop semantics, generalized to
//! spatially varying fields:
//! - center sampled exactly once (forward loop covers step 0),
//! - backward loop starts one step behind the center,
//! - nearest-texel sampling with clamp-to-edge (matches the GPU sampler for
//!   whole-pixel steps from pixel centers),
//! - per-step field lookup: the step vector is re-read at each new position
//!   (nearest texel), so constant fields reduce to 1D convolution.
//!
//! All steps are in **pixel units**; callers pass pixel-space fields (the same
//! data stored in `CpuVectorFieldNode::pixel_data`).

use crate::lic::LicParams;

/// Sample helper: nearest texel with clamp-to-edge.
fn sample_clamped(data: &[f32], w: usize, h: usize, sx: i32, sy: i32) -> f32 {
    let cx = sx.clamp(0, w as i32 - 1) as usize;
    let cy = sy.clamp(0, h as i32 - 1) as usize;
    data[cy * w + cx]
}

/// Field lookup with clamp-to-edge. Returns `(dx, dy)` in pixels.
fn field_at(field: &[f32], w: usize, h: usize, sx: i32, sy: i32) -> (f32, f32) {
    let cx = sx.clamp(0, w as i32 - 1) as usize;
    let cy = sy.clamp(0, h as i32 - 1) as usize;
    let idx = (cy * w + cx) * 2;
    (field[idx], field[idx + 1])
}

/// LIC value at one pixel for a general (possibly varying) pixel-space field.
pub fn cpu_lic_pixel_general(
    noise: &[f32],
    field: &[f32],
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    params: &LicParams,
) -> f32 {
    assert_eq!(noise.len(), w * h);
    assert_eq!(field.len(), w * h * 2);
    let mut acc = 0.0f32;
    let mut wsum = 0.0f32;

    // Forward: pos_0 = (x, y); pos_{i} = pos_{i-1} + field(pos_{i-1}).
    // Weights: forward[i] for step i (forward[0] = center).
    let (mut px, mut py) = (x as f32, y as f32);
    for i in 0..params.forward_count as usize {
        let wgt = params.forward(i);
        acc += sample_clamped(noise, w, h, px.round() as i32, py.round() as i32) * wgt;
        wsum += wgt;
        if i + 1 < params.forward_count as usize {
            let (sdx, sdy) = field_at(field, w, h, px.round() as i32, py.round() as i32);
            px += sdx;
            py += sdy;
        }
    }

    // Backward: pos = (x, y) - step(center) first, then keep stepping.
    // Weights: backward[i] for the (i+1)-th step behind center.
    if params.backward_count > 0 {
        let (sdx0, sdy0) = field_at(field, w, h, x as i32, y as i32);
        let (mut bx, mut by) = (x as f32 - sdx0, y as f32 - sdy0);
        for i in 0..params.backward_count as usize {
            let wgt = params.backward(i);
            acc += sample_clamped(noise, w, h, bx.round() as i32, by.round() as i32) * wgt;
            wsum += wgt;
            let (sdx, sdy) = field_at(field, w, h, bx.round() as i32, by.round() as i32);
            bx -= sdx;
            by -= sdy;
        }
    }

    acc / wsum
}

/// Full-image LIC for a general field.
pub fn cpu_lic_image_general(
    noise: &[f32],
    field: &[f32],
    w: usize,
    h: usize,
    params: &LicParams,
) -> Vec<f32> {
    (0..h)
        .flat_map(|y| {
            (0..w).map(move |x| cpu_lic_pixel_general(noise, field, w, h, x, y, params))
        })
        .collect()
}

/// LIC value at one pixel for a **constant** step (fast path / exact ground
/// truth for constant fields). `step_px` is in pixels, e.g. `(1, 0)`.
pub fn cpu_lic_pixel_constant(
    noise: &[f32],
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    step_px: (i32, i32),
    params: &LicParams,
) -> f32 {
    let sample = |sx: i32, sy: i32| -> f32 {
        sample_clamped(noise, w, h, sx, sy)
    };
    let mut acc = 0.0f32;
    let mut wsum = 0.0f32;
    for i in 0..params.forward_count as i32 {
        let wgt = params.forward(i as usize);
        acc += sample(x as i32 + i * step_px.0, y as i32 + i * step_px.1) * wgt;
        wsum += wgt;
    }
    for i in 0..params.backward_count as i32 {
        let wgt = params.backward(i as usize);
        let k = i + 1;
        acc += sample(x as i32 - k * step_px.0, y as i32 - k * step_px.1) * wgt;
        wsum += wgt;
    }
    acc / wsum
}

/// Full-image LIC for a constant step.
pub fn cpu_lic_image_constant(
    noise: &[f32],
    w: usize,
    h: usize,
    step_px: (i32, i32),
    params: &LicParams,
) -> Vec<f32> {
    (0..h)
        .flat_map(|y| {
            (0..w).map(move |x| cpu_lic_pixel_constant(noise, w, h, x, y, step_px, params))
        })
        .collect()
}

/// Rotate a square image 90° clockwise: `out[x, y] = input[n-1-y, x]`.
pub fn rotate_image_cw(data: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(data.len(), n * n);
    let mut out = vec![0.0; n * n];
    for y in 0..n {
        for x in 0..n {
            out[y * n + x] = data[(n - 1 - y) + x * n];
        }
    }
    out
}

/// Rotate a square vector field 90° clockwise, rotating vectors as well:
/// position rotates CW and each `(dx, dy)` becomes `(-dy, dx)`.
/// (Image y points down; this matches the pixel-index rotation above.)
pub fn rotate_field_cw(field: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(field.len(), n * n * 2);
    let mut out = vec![0.0; n * n * 2];
    for y in 0..n {
        for x in 0..n {
            let sx = n - 1 - y;
            let sy = x;
            let sidx = (sy * n + sx) * 2;
            let didx = (y * n + x) * 2;
            let (dx, dy) = (field[sidx], field[sidx + 1]);
            out[didx] = -dy;
            out[didx + 1] = dx;
        }
    }
    out
}

/// Mirror an image left-right: `out[x, y] = input[w-1-x, y]`.
pub fn mirror_image_lr(data: &[f32], w: usize, h: usize) -> Vec<f32> {
    assert_eq!(data.len(), w * h);
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = data[y * w + (w - 1 - x)];
        }
    }
    out
}

/// Mirror a vector field left-right, flipping vector x-components:
/// position mirrors and each `(dx, dy)` becomes `(-dx, dy)`.
pub fn mirror_field_lr(field: &[f32], w: usize, h: usize) -> Vec<f32> {
    assert_eq!(field.len(), w * h * 2);
    let mut out = vec![0.0; w * h * 2];
    for y in 0..h {
        for x in 0..w {
            let sidx = (y * w + (w - 1 - x)) * 2;
            let didx = (y * w + x) * 2;
            out[didx] = -field[sidx];
            out[didx + 1] = field[sidx + 1];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lic::vector_field::analytical;

    #[test]
    fn general_matches_constant_for_uniform_fields() {
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
            for y in 0..h {
                for x in 0..w {
                    let g = cpu_lic_pixel_general(&noise, &field, w, h, x, y, &params);
                    let c = cpu_lic_pixel_constant(&noise, w, h, x, y, step, &params);
                    assert!(
                        (g - c).abs() < 1e-5,
                        "general must equal constant path at ({x},{y}): {g} vs {c}"
                    );
                }
            }
        }
    }

    #[test]
    fn rotate_helpers_are_self_inverse_after_four_turns() {
        let n = 4;
        let data: Vec<f32> = (0..n * n).map(|i| i as f32).collect();
        let mut r = data.clone();
        for _ in 0..4 {
            r = rotate_image_cw(&r, n);
        }
        assert_eq!(r, data);
    }
}
