//! Analytical vector field generators (issue #7).
//!
//! Pure-CPU pixel-space field generators. Each function returns interleaved
//! `(dx_px, dy_px)` per pixel in row-major order, ready to store in
//! [`crate::lic::vector_field::CpuVectorFieldNode::pixel_data`].
//!
//! Coordinates are in pixels with the origin at the image top-left,
//! x right, y down (matching Bevy image row-major layout). `center` is given
//! in pixel coordinates. Offsets are `(x - cx, y - cy)`.

use crate::lic::vector_field::CpuVectorFieldNode;

/// Constant horizontal field: `(strength, 0.0)` everywhere.
pub fn constant_horizontal(width: u32, height: u32, strength: f32) -> Vec<f32> {
    let mut data = vec![0.0; (width * height * 2) as usize];
    for i in 0..(width * height) as usize {
        data[i * 2] = strength;
        data[i * 2 + 1] = 0.0;
    }
    data
}

/// Constant vertical field: `(0.0, strength)` everywhere.
pub fn constant_vertical(width: u32, height: u32, strength: f32) -> Vec<f32> {
    let mut data = vec![0.0; (width * height * 2) as usize];
    for i in 0..(width * height) as usize {
        data[i * 2] = 0.0;
        data[i * 2 + 1] = strength;
    }
    data
}

/// Vortex field: `(-dy, dx) / r^2 * strength`, where `(dx, dy)` is the offset
/// from `center` in pixels and `r^2 = dx^2 + dy^2`.
///
/// Returns the zero vector at the exact center (singularity) to avoid NaN.
pub fn vortex(
    width: u32,
    height: u32,
    center: (f32, f32),
    strength: f32,
) -> Vec<f32> {
    let mut data = vec![0.0; (width * height * 2) as usize];
    let (cx, cy) = center;
    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let r2 = dx * dx + dy * dy;
            let idx = ((y * width + x) * 2) as usize;
            if r2 == 0.0 {
                data[idx] = 0.0;
                data[idx + 1] = 0.0;
            } else {
                data[idx] = -dy / r2 * strength;
                data[idx + 1] = dx / r2 * strength;
            }
        }
    }
    data
}

/// Saddle field: `(dx, -dy) * strength`, where `(dx, dy)` is the offset from
/// `center` in pixels.
pub fn saddle(
    width: u32,
    height: u32,
    center: (f32, f32),
    strength: f32,
) -> Vec<f32> {
    let mut data = vec![0.0; (width * height * 2) as usize];
    let (cx, cy) = center;
    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let idx = ((y * width + x) * 2) as usize;
            data[idx] = dx * strength;
            data[idx + 1] = -dy * strength;
        }
    }
    data
}

impl CpuVectorFieldNode {
    /// Build a node from existing interleaved pixel-space data.
    ///
    /// Panics if `pixel_data.len() != width * height * 2`.
    pub fn from_pixel_data(width: u32, height: u32, pixel_data: Vec<f32>) -> Self {
        assert_eq!(
            pixel_data.len(),
            (width * height * 2) as usize,
            "pixel_data must hold (dx, dy) per pixel"
        );
        let mut node = Self::new(width, height);
        node.pixel_data = pixel_data;
        node
    }

    /// Node holding a constant horizontal field.
    pub fn constant_horizontal(width: u32, height: u32, strength: f32) -> Self {
        Self::from_pixel_data(width, height, constant_horizontal(width, height, strength))
    }

    /// Node holding a constant vertical field.
    pub fn constant_vertical(width: u32, height: u32, strength: f32) -> Self {
        Self::from_pixel_data(width, height, constant_vertical(width, height, strength))
    }

    /// Node holding a vortex field.
    pub fn vortex(width: u32, height: u32, center: (f32, f32), strength: f32) -> Self {
        Self::from_pixel_data(width, height, vortex(width, height, center, strength))
    }

    /// Node holding a saddle field.
    pub fn saddle(width: u32, height: u32, center: (f32, f32), strength: f32) -> Self {
        Self::from_pixel_data(width, height, saddle(width, height, center, strength))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_fields_fill_every_pixel() {
        let h = constant_horizontal(4, 3, 2.0);
        assert_eq!(h.len(), 4 * 3 * 2);
        for i in 0..12 {
            assert_eq!(h[i * 2], 2.0);
            assert_eq!(h[i * 2 + 1], 0.0);
        }
        let v = constant_vertical(4, 3, -1.5);
        for i in 0..12 {
            assert_eq!(v[i * 2], 0.0);
            assert_eq!(v[i * 2 + 1], -1.5);
        }
    }

    #[test]
    fn vortex_center_is_zero_and_scales_with_strength() {
        // 3x3, center pixel is the singularity: row 1, col 1 -> idx 8.
        let v = vortex(3, 3, (1.0, 1.0), 1.0);
        let c = 8;
        assert_eq!(v[c], 0.0);
        assert_eq!(v[c + 1], 0.0);
        // Pixel right of center: row 1, col 2 -> idx 10. Offset (1, 0) -> (-0, 1)/1.
        let r = 10;
        assert!((v[r] - 0.0).abs() < 1e-6);
        assert!((v[r + 1] - 1.0).abs() < 1e-6);
        // Doubling strength doubles vectors.
        let v2 = vortex(3, 3, (1.0, 1.0), 2.0);
        assert!((v2[r + 1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn saddle_sign_pattern() {
        // 3x3, center (1,1): right is (+, -0), down is (+0, -).
        let s = saddle(3, 3, (1.0, 1.0), 1.0);
        let right = 10; // row 1, col 2
        assert!((s[right] - 1.0).abs() < 1e-6);
        assert!((s[right + 1] - 0.0).abs() < 1e-6);
        let down = 14; // row 2, col 1
        assert!((s[down] - 0.0).abs() < 1e-6);
        assert!((s[down + 1] + 1.0).abs() < 1e-6);
    }
}
