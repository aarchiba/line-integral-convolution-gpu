//! Pre-cooked vector field node (zero-copy pass-through)

use bevy::prelude::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::render_resource::*;
use crate::lic::vector_field::VectorFieldNode;

/// Vector field node that uses a pre-existing UV-space texture
pub struct PrecookedVectorFieldNode {
    pub texture_view: TextureView,
    pub format: TextureFormat,
    pub size: UVec2,
}

impl PrecookedVectorFieldNode {
    pub fn new(texture_view: TextureView, format: TextureFormat, size: UVec2) -> Self {
        Self {
            texture_view,
            format,
            size,
        }
    }
}

impl VectorFieldNode for PrecookedVectorFieldNode {
    fn prepare(&mut self, _render_device: &RenderDevice, _render_queue: &RenderQueue) {
        // No-op: texture is already ready on GPU
    }
    
    fn output_view(&self) -> &TextureView {
        &self.texture_view
    }
    
    fn format(&self) -> TextureFormat {
        self.format
    }
    
    fn size(&self) -> UVec2 {
        self.size
    }
}