//! Vector field production abstraction

use bevy::prelude::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::render_resource::*;

/// Trait for vector field production nodes
pub trait VectorFieldNode: Send + Sync + 'static {
    /// Prepare GPU resources for this frame
    fn prepare(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue);
    
    /// Get the output texture view (UV-space RG16Float)
    fn output_view(&self) -> &TextureView;
    
    /// Texture format (default RG16Float)
    fn format(&self) -> TextureFormat {
        TextureFormat::Rg16Float
    }
    
    /// Texture size
    fn size(&self) -> UVec2;
}

pub mod analytical;
pub mod cpu;
pub mod precooked;

pub use cpu::CpuVectorFieldNode;
pub use precooked::PrecookedVectorFieldNode;