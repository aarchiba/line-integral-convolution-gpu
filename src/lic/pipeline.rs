//! LIC render graph pipeline

use bevy::prelude::*;
use bevy::render::render_graph::{Node, NodeRunError, RenderGraphContext};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use crate::lic::vector_field::VectorFieldNode;

/// Wrapper node that runs a VectorFieldNode in the render graph
pub struct VectorFieldNodeWrapper {
    node: Box<dyn VectorFieldNode>,
}

impl VectorFieldNodeWrapper {
    pub fn new(node: impl VectorFieldNode) -> Self {
        Self {
            node: Box::new(node),
        }
    }
}

impl Node for VectorFieldNodeWrapper {
    fn update(&mut self, world: &mut World) {
        let render_device = world.resource::<RenderDevice>();
        let render_queue = world.resource::<RenderQueue>();
        self.node.prepare(render_device, render_queue);
    }
    
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        _render_context: &mut RenderContext,
        _world: &World,
    ) -> Result<(), NodeRunError> {
        // The actual work is done in update()
        Ok(())
    }
}

/// Plugin for LIC rendering
pub struct LicPlugin {
    _vector_field_node: Box<dyn VectorFieldNode>,
}

impl LicPlugin {
    pub fn new(vector_field_node: impl VectorFieldNode) -> Self {
        Self {
            _vector_field_node: Box::new(vector_field_node),
        }
    }
}

impl Plugin for LicPlugin {
    fn build(&self, _app: &mut App) {
        // Register render graph nodes would go here
        // For now just the vector field node wrapper
    }
}