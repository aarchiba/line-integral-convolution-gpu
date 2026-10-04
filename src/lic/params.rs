//! LIC parameters
//!
//! Logical contract (per ticket): 256 forward + 256 backward `f32` weights
//! with explicit counts. GPU layout packs them as `vec4` arrays because WGSL
//! `uniform` address space requires array stride 16 — a plain
//! `array<f32, 256>` is rejected by naga/wgpu validation.

// bytemuck_derive emits a per-field `fn check` that newer rustc flags as
// dead_code. Allow it for this module (upstream toolchain noise, not our code).
#![allow(dead_code)]

use bevy::math::Vec4;
use bevy::render::render_resource::ShaderType;
use bytemuck::{Pod, Zeroable};

pub const MAX_STEPS: usize = 256;
const PACKED: usize = MAX_STEPS / 4;

/// LIC kernel parameters (uniform buffer, 16-byte aligned).
#[derive(Debug, Clone, Copy, ShaderType)]
pub struct LicParams {
    pub forward_count: u32,
    pub backward_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub forward_weights: [Vec4; PACKED],
    pub backward_weights: [Vec4; PACKED],
}

// bytemuck for raw buffer uploads (Vec4 is Pod).
unsafe impl Pod for LicParams {}
unsafe impl Zeroable for LicParams {}

impl Default for LicParams {
    fn default() -> Self {
        Self {
            forward_count: 1,
            backward_count: 0,
            _pad0: 0,
            _pad1: 0,
            forward_weights: [Vec4::ZERO; PACKED],
            backward_weights: [Vec4::ZERO; PACKED],
        }
    }
}

impl LicParams {
    fn get(packed: &[Vec4; PACKED], i: usize) -> f32 {
        packed[i / 4][i % 4]
    }

    fn set(packed: &mut [Vec4; PACKED], i: usize, v: f32) {
        packed[i / 4][i % 4] = v;
    }

    /// Logical forward weight `i` (0..256).
    pub fn forward(&self, i: usize) -> f32 {
        Self::get(&self.forward_weights, i)
    }

    /// Logical backward weight `i` (0..256).
    pub fn backward(&self, i: usize) -> f32 {
        Self::get(&self.backward_weights, i)
    }

    pub fn identity() -> Self {
        let mut params = Self::default();
        Self::set(&mut params.forward_weights, 0, 1.0);
        params.forward_count = 1;
        params.backward_count = 0;
        params
    }

    pub fn triangular_kernel(radius: u32) -> Self {
        let mut params = Self::default();
        let radius = (radius as usize).min(MAX_STEPS - 1);
        if radius == 0 {
            Self::set(&mut params.forward_weights, 0, 1.0);
            params.forward_count = 1;
            params.backward_count = 0;
            return params;
        }
        let r = radius as f32;

        Self::set(&mut params.forward_weights, 0, 1.0);
        params.forward_count = (radius + 1) as u32;

        for i in 1..=radius {
            let weight = 1.0 - i as f32 / r;
            Self::set(&mut params.forward_weights, i, weight);
            Self::set(&mut params.backward_weights, i - 1, weight);
        }
        params.backward_count = radius as u32;

        params
    }

    /// Build kernel parameters from explicit weight slices.
    ///
    /// `forward[0]` is the center weight; `forward[i]` covers step `i`
    /// forward and `backward[i]` covers step `i + 1` backward (same
    /// contract as [`LicParams::triangular_kernel`]). Slices longer than
    /// [`MAX_STEPS`] are truncated to fit the uniform buffer.
    pub fn from_weights(forward: &[f32], backward: &[f32]) -> Self {
        let mut params = Self::default();
        let fwd_len = forward.len().min(MAX_STEPS);
        let bwd_len = backward.len().min(MAX_STEPS);
        for (i, &w) in forward[..fwd_len].iter().enumerate() {
            Self::set(&mut params.forward_weights, i, w);
        }
        for (i, &w) in backward[..bwd_len].iter().enumerate() {
            Self::set(&mut params.backward_weights, i, w);
        }
        params.forward_count = fwd_len as u32;
        params.backward_count = bwd_len as u32;
        params
    }
}
