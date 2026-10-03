//! LIC parameters

// bytemuck_derive emits a per-field `fn check` that newer rustc flags as
// dead_code. Allow it for this module (upstream toolchain noise, not our code).
#![allow(dead_code)]

use bevy::render::render_resource::ShaderType;
use bytemuck::{Pod, Zeroable};

/// LIC kernel parameters
#[derive(Debug, Clone, Copy, Pod, Zeroable, ShaderType)]
#[repr(C)]
pub struct LicParams {
    pub forward_count: u32,
    pub backward_count: u32,
    pub forward_weights: [f32; 256],
    pub backward_weights: [f32; 256],
}

impl Default for LicParams {
    fn default() -> Self {
        Self {
            forward_count: 1,
            backward_count: 0,
            forward_weights: [0.0; 256],
            backward_weights: [0.0; 256],
        }
    }
}

impl LicParams {
    pub fn identity() -> Self {
        let mut params = Self::default();
        params.forward_weights[0] = 1.0;
        params.forward_count = 1;
        params
    }
    
    pub fn triangular_kernel(radius: u32) -> Self {
        let mut params = Self::default();
        let radius = radius.min(255) as usize;
        let total_weight = (radius * (radius + 1) / 2) as f32 * 2.0 + 1.0;
        
        params.forward_weights[0] = 1.0 / total_weight;
        params.forward_count = (radius + 1) as u32;
        
        for i in 1..=radius {
            let weight = (radius + 1 - i) as f32 / total_weight;
            params.forward_weights[i] = weight;
            params.backward_weights[i - 1] = weight;
        }
        params.backward_count = radius as u32;
        
        params
    }
}